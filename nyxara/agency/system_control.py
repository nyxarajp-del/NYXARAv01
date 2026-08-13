"""NYXARA · agency/system_control.py — a real, whole-machine OS-control faculty (🖥, fail-closed).

:mod:`agency.filesystem` gave NYXARA a structured engine she operates *herself* over the whole
disk; this module is its OS-control sibling. Where the disk faculty reads/writes files, this one
reads and steers the **running machine**: it enumerates and signals processes, controls services,
manages packages, reports hardware/system state, drives power, and (elevated) manages users and
kernel tunables. It is the real implementation behind capability **#1 — unrestricted control over
the computer** — and it is *first-class*: NYXARA composes typed, pure-Python operations, she does
**not** hand a raw shell string to the LLM to guess at.

The discipline mirrors :mod:`agency.filesystem` and :mod:`agency.privilege`:

* **Real, not fake.** Structured facts come from :mod:`psutil` when it is installed (it is a
  declared dependency, used already by :mod:`identity.interoception`) and from ``/proc`` + the
  standard library when it is not. Process signalling is plain :func:`os.kill`. Root-requiring
  actions (service/power/user/sysctl-write, most package installs) go through the very same
  :func:`~nyxara.agency.privilege.privileged_exec` plumbing the ``privileged_shell`` tool uses —
  elevated *with* the Master's stored credential, never an exploit, guess, or brute-force.
* **Never raises.** Every failure — psutil missing, a bad PID, a protected target, a non-zero
  exit, a timeout — comes back as a uniform ``{"ok": False, "error": …}`` dict, so the sovereign
  loop's *act* stage never crashes.
* **Reach is maximal; safety is by construction.** With the defaults this faculty reaches the
  whole machine (aligned with :class:`~nyxara.kernel.config.SystemControlConfig` and
  ``full_control``). Its *own* guards are narrow and honest: a **protected-PID** guard refuses
  PID 0/1 and NYXARA's own process/group so she cannot accidentally signal ``init`` or kill
  herself, and config toggles fence off power/service/package/user control when the Master wants
  them off. Power control is off by default.
* **No permission checking of its own.** This module is the *executor*. Every caller (the
  :class:`~nyxara.agency.tools.ToolRegistry` in :mod:`agency.default_tools`) gates each operation
  behind the right :class:`~nyxara.agency.permissions.Capability` — ``PROC_EXEC`` for signalling,
  ``PKG_INSTALL`` for packages, ``ACCOUNT_MODIFY`` for users, ``PRIV_ESCALATE`` for the root
  surface — and the kernel's ``/scram`` kill-switch, oversight and corrigibility gates all sit
  above it and remain fully intact.

Pure standard library apart from the optional :mod:`psutil` backend.
"""

from __future__ import annotations

import os
import platform
import shutil
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from nyxara.agency.code_sandbox import SandboxResult, safe_shell
from nyxara.agency.privilege import privileged_exec, privilege_status

__all__ = ["SystemControl", "SystemControlError"]

# psutil is a declared dependency and already used by identity/interoception; import-guarded so a
# machine without it still gets a real (if thinner) faculty via /proc + the standard library.
try:  # pragma: no cover - trivial import guard
    import psutil as _psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:  # noqa: BLE001
    _psutil = None  # type: ignore
    _HAS_PSUTIL = False

# Engine defaults. They mirror :class:`SystemControlConfig` (nyxara/kernel/config.py) so a bare
# ``SystemControl()`` — used in tests and by any caller without settings — behaves like the shipped
# configuration: whole-machine reach, power off by default, sane caps.
_DEFAULT_MAX_PROCESSES = 5_000
_DEFAULT_TIMEOUT_S = 30.0
# The package managers we recognise, in probe order (most specific/common first). Each maps to the
# argv builders in :meth:`SystemControl._pkg_argv`.
_PKG_MANAGERS = ("apt-get", "dnf", "yum", "pacman", "zypper", "apk", "brew", "pip")
# systemd/SysV service verbs and power/logout verbs we accept (kept explicit so an arbitrary verb
# can never be smuggled into the privileged shell).
_SERVICE_ACTIONS = ("start", "stop", "restart", "reload", "enable", "disable", "status")
_POWER_ACTIONS = ("shutdown", "poweroff", "reboot", "restart", "halt", "suspend", "hibernate", "logout")
_USER_ACTIONS = ("add", "remove", "delete", "modify", "lock", "unlock", "passwd")


class SystemControlError(Exception):
    """Raised only for programmer misuse (never for operational failure).

    Operational failures — a bad PID, a missing manager, a non-zero exit — are returned as
    ``{"ok": False, "error": …}`` data, honouring the module's never-raises contract. This
    exception exists so a genuinely nonsensical *call* (e.g. an unknown action verb reaching an
    internal helper) is still distinguishable in tests; the tool layer converts any escape into an
    ``ok=False`` :class:`~nyxara.agency.tools.ToolResult`.
    """


# A privileged runner: takes an argv list (or command string) + timeout and returns a
# :class:`SandboxResult`. The default binds no secret (passwordless/NOPASSWD sudo or an already-root
# process); :func:`~nyxara.agency.default_tools.build_default_tools` swaps in a closure that
# materialises the Master's stored sudo credential late (inside the vault), so the plaintext never
# lives on the engine or in any result.
PrivilegedRunner = Callable[[Any, float], SandboxResult]


def _default_privileged_runner(command: Any, timeout_s: float) -> SandboxResult:
    return privileged_exec(command, sudo_password=None, timeout_s=timeout_s)


@dataclass
class SystemControl:
    """A governed, pure-Python engine for real whole-machine OS control.

    Construct directly with keyword overrides, or via :meth:`from_config` to adopt a
    :class:`~nyxara.kernel.config.SystemControlConfig`. Read-only introspection never elevates;
    effectful process operations use :func:`os.kill`; the root surface (service/power/user/
    sysctl-write and root package installs) routes through :attr:`privileged_runner`.
    """

    enabled: bool = True
    allow_power: bool = False
    allow_service_control: bool = True
    allow_package_management: bool = True
    allow_user_management: bool = True
    package_manager: str = ""              # "" -> auto-detect
    protect_self: bool = True
    protected_pids: List[int] = field(default_factory=lambda: [0, 1])
    max_processes: int = _DEFAULT_MAX_PROCESSES
    default_timeout_s: float = _DEFAULT_TIMEOUT_S
    privileged_runner: PrivilegedRunner = field(default=_default_privileged_runner, repr=False)

    @classmethod
    def from_config(cls, cfg: Any) -> "SystemControl":
        """Build from a :class:`SystemControlConfig` (or any object exposing the same fields)."""
        if cfg is None:
            return cls()
        return cls(
            enabled=bool(getattr(cfg, "enabled", True)),
            allow_power=bool(getattr(cfg, "allow_power", False)),
            allow_service_control=bool(getattr(cfg, "allow_service_control", True)),
            allow_package_management=bool(getattr(cfg, "allow_package_management", True)),
            allow_user_management=bool(getattr(cfg, "allow_user_management", True)),
            package_manager=str(getattr(cfg, "package_manager", "") or ""),
            protect_self=bool(getattr(cfg, "protect_self", True)),
            protected_pids=list(getattr(cfg, "protected_pids", None) or [0, 1]),
            max_processes=int(getattr(cfg, "max_processes", _DEFAULT_MAX_PROCESSES)),
            default_timeout_s=float(getattr(cfg, "default_timeout_s", _DEFAULT_TIMEOUT_S)),
        )

    # ------------------------------------------------------------------ #
    # Guards & helpers (never raise for operational conditions)
    # ------------------------------------------------------------------ #
    def _enabled(self) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return {"ok": False, "error": "system-control faculty is disabled by config"}
        return None

    def _protected(self, pid: int) -> Optional[Dict[str, Any]]:
        """Return a refusal dict if ``pid`` is protected, else None."""
        # A non-positive PID is not a process, it is a BROADCAST, and every guard below reasons
        # about single processes. ``os.kill(-1, …)`` signals every process the caller may signal;
        # ``os.kill(0, …)`` and ``os.kill(-n, …)`` signal a whole process group. None of those
        # PIDs equals NYXARA's own, so ``protect_self`` waved them straight through — and since
        # her process is inside the blast radius, `kill_process(-1)` as root took down the machine
        # AND her, which also takes down /scram, oversight and the transparency feed. Refuse the
        # shape outright; a real target is always > 0.
        if pid <= 0:
            return {"ok": False,
                    "error": f"refusing to target pid {pid}: non-positive PIDs signal a whole "
                             "process group (or every reachable process), which cannot be "
                             "guarded — name a single process"}
        if pid in set(self.protected_pids):
            return {"ok": False, "error": f"refusing to target protected pid {pid}"}
        if self.protect_self:
            try:
                selves = {os.getpid(), os.getppid(), os.getpgrp()}
            except OSError:
                selves = {os.getpid()}
            if pid in selves:
                return {"ok": False,
                        "error": f"refusing to target NYXARA's own process/group (pid {pid})"}
        return None

    def _run_priv(self, argv: Any, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        """Run a privileged command via the injected runner, returning its dict form."""
        t = self.default_timeout_s if timeout_s is None else max(0.1, min(float(timeout_s), 600.0))
        try:
            return self.privileged_runner(argv, t).to_dict()
        except Exception as exc:  # noqa: BLE001 — never let the executor raise
            return {"ok": False, "error": f"privileged runner failed: {exc}"}

    def _run_plain(self, argv: List[str], timeout_s: Optional[float] = None) -> SandboxResult:
        t = self.default_timeout_s if timeout_s is None else max(0.1, min(float(timeout_s), 600.0))
        return safe_shell(argv, timeout_s=t)

    @staticmethod
    def _iso(ts: float) -> str:
        return datetime.fromtimestamp(ts, timezone.utc).isoformat()

    # ------------------------------------------------------------------ #
    # System / hardware information (read-only, never elevates)
    # ------------------------------------------------------------------ #
    def system_info(self) -> Dict[str, Any]:
        """OS, kernel, arch, hostname, boot time, uptime, load and logged-in user count."""
        un = platform.uname()
        boot = None
        uptime = None
        if _HAS_PSUTIL:
            try:
                boot = _psutil.boot_time()
                uptime = time.time() - boot
            except Exception:  # noqa: BLE001
                pass
        if uptime is None:
            try:
                with open("/proc/uptime", "r") as f:
                    uptime = float(f.read().split()[0])
                    boot = time.time() - uptime
            except Exception:  # noqa: BLE001
                pass
        info: Dict[str, Any] = {
            "ok": True,
            "system": un.system,
            "node": un.node,
            "release": un.release,
            "version": un.version,
            "machine": un.machine,
            "processor": un.processor or platform.processor(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pid": os.getpid(),
            "cwd": os.getcwd(),
            "boot_time": self._iso(boot) if boot else None,
            "uptime_s": round(uptime, 1) if uptime is not None else None,
            "cpu_count": os.cpu_count(),
            "has_psutil": _HAS_PSUTIL,
        }
        try:
            info["load_average"] = list(os.getloadavg())
        except (OSError, AttributeError):
            info["load_average"] = None
        if _HAS_PSUTIL:
            try:
                info["logged_in_users"] = sorted({u.name for u in _psutil.users()})
            except Exception:  # noqa: BLE001
                info["logged_in_users"] = None
        return info

    def cpu_info(self, *, interval: float = 0.2) -> Dict[str, Any]:
        """Logical/physical core counts, frequency and per-core utilisation (%)."""
        out: Dict[str, Any] = {"ok": True, "logical_cores": os.cpu_count()}
        try:
            out["load_average"] = list(os.getloadavg())
        except (OSError, AttributeError):
            out["load_average"] = None
        if not _HAS_PSUTIL:
            out["note"] = "psutil not installed — per-core usage unavailable"
            return out
        try:
            out["physical_cores"] = _psutil.cpu_count(logical=False)
            out["overall_percent"] = _psutil.cpu_percent(interval=max(0.0, min(interval, 2.0)))
            out["per_core_percent"] = _psutil.cpu_percent(interval=None, percpu=True)
            freq = _psutil.cpu_freq()
            out["frequency_mhz"] = ({"current": freq.current, "min": freq.min, "max": freq.max}
                                    if freq else None)
        except Exception as exc:  # noqa: BLE001
            out["error"] = str(exc)
        return out

    def memory_info(self) -> Dict[str, Any]:
        """Virtual and swap memory (bytes + percent)."""
        if not _HAS_PSUTIL:
            return self._meminfo_from_proc()
        try:
            vm = _psutil.virtual_memory()
            sm = _psutil.swap_memory()
            return {
                "ok": True,
                "virtual": {"total": vm.total, "available": vm.available, "used": vm.used,
                            "free": vm.free, "percent": vm.percent},
                "swap": {"total": sm.total, "used": sm.used, "free": sm.free, "percent": sm.percent},
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def _meminfo_from_proc(self) -> Dict[str, Any]:
        try:
            fields: Dict[str, int] = {}
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    key, _, rest = line.partition(":")
                    parts = rest.split()
                    if parts:
                        fields[key.strip()] = int(parts[0]) * 1024  # kB -> bytes
            total = fields.get("MemTotal", 0)
            available = fields.get("MemAvailable", 0)
            return {"ok": True, "virtual": {"total": total, "available": available,
                                            "used": max(0, total - available),
                                            "free": fields.get("MemFree", 0),
                                            "percent": round((total - available) / total * 100, 1)
                                            if total else 0.0},
                    "swap": {"total": fields.get("SwapTotal", 0),
                             "free": fields.get("SwapFree", 0),
                             "used": max(0, fields.get("SwapTotal", 0) - fields.get("SwapFree", 0))},
                    "note": "psutil not installed — read from /proc/meminfo"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"psutil not installed and /proc/meminfo unreadable: {exc}"}

    def disk_partitions(self) -> Dict[str, Any]:
        """Mounted partitions with per-mount usage (device, mountpoint, fstype, total/used/free)."""
        parts: List[Dict[str, Any]] = []
        if _HAS_PSUTIL:
            try:
                for p in _psutil.disk_partitions(all=False):
                    entry = {"device": p.device, "mountpoint": p.mountpoint,
                             "fstype": p.fstype, "opts": p.opts}
                    try:
                        u = _psutil.disk_usage(p.mountpoint)
                        entry.update({"total": u.total, "used": u.used,
                                      "free": u.free, "percent": u.percent})
                    except Exception:  # noqa: BLE001
                        pass
                    parts.append(entry)
                return {"ok": True, "partitions": parts}
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": str(exc)}
        # Fallback: at least report the root filesystem.
        try:
            u = shutil.disk_usage("/")
            return {"ok": True, "partitions": [{"mountpoint": "/", "total": u.total,
                                                "used": u.used, "free": u.free}],
                    "note": "psutil not installed — root filesystem only"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def sensors(self) -> Dict[str, Any]:
        """Hardware sensors — temperatures, fans, and battery — when the platform exposes them."""
        if not _HAS_PSUTIL:
            return {"ok": False, "error": "psutil not installed — sensor data unavailable"}
        out: Dict[str, Any] = {"ok": True}
        try:
            temps = getattr(_psutil, "sensors_temperatures", lambda: {})()
            out["temperatures"] = {name: [{"label": e.label, "current": e.current,
                                           "high": e.high, "critical": e.critical} for e in entries]
                                   for name, entries in (temps or {}).items()}
        except Exception:  # noqa: BLE001
            out["temperatures"] = None
        try:
            fans = getattr(_psutil, "sensors_fans", lambda: {})()
            out["fans"] = {name: [{"label": e.label, "current": e.current} for e in entries]
                           for name, entries in (fans or {}).items()}
        except Exception:  # noqa: BLE001
            out["fans"] = None
        try:
            bat = getattr(_psutil, "sensors_battery", lambda: None)()
            out["battery"] = ({"percent": bat.percent, "power_plugged": bat.power_plugged,
                               "secs_left": bat.secsleft} if bat else None)
        except Exception:  # noqa: BLE001
            out["battery"] = None
        return out

    def network_interfaces(self) -> Dict[str, Any]:
        """Network interfaces: addresses (IPv4/IPv6/MAC) and up/speed/MTU stats."""
        if not _HAS_PSUTIL:
            return {"ok": False, "error": "psutil not installed — interface data unavailable"}
        try:
            addrs = _psutil.net_if_addrs()
            stats = _psutil.net_if_stats()
            out: Dict[str, Any] = {}
            for name, addr_list in addrs.items():
                st = stats.get(name)
                out[name] = {
                    "addresses": [{"family": str(a.family), "address": a.address,
                                   "netmask": a.netmask, "broadcast": a.broadcast}
                                  for a in addr_list],
                    "is_up": st.isup if st else None,
                    "speed_mbps": st.speed if st else None,
                    "mtu": st.mtu if st else None,
                }
            return {"ok": True, "interfaces": out}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def net_connections(self, *, kind: str = "inet", limit: int = 200) -> Dict[str, Any]:
        """Active network connections (fd, family, laddr, raddr, status, pid)."""
        if not _HAS_PSUTIL:
            return {"ok": False, "error": "psutil not installed — connection data unavailable"}
        try:
            conns = _psutil.net_connections(kind=kind)
            rows = []
            for c in conns[: max(1, int(limit))]:
                rows.append({
                    "fd": c.fd,
                    "family": str(c.family),
                    "type": str(c.type),
                    "local": f"{c.laddr.ip}:{c.laddr.port}" if c.laddr else None,
                    "remote": f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else None,
                    "status": c.status,
                    "pid": c.pid,
                })
            return {"ok": True, "connections": rows, "count": len(conns)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    # ------------------------------------------------------------------ #
    # Process control
    # ------------------------------------------------------------------ #
    def list_processes(self, *, sort_by: str = "cpu", limit: int = 50) -> Dict[str, Any]:
        """Running processes (pid, name, user, cpu%, mem%, status, cmdline), sorted & capped."""
        cap = min(int(limit) if limit else self.max_processes, self.max_processes)
        if _HAS_PSUTIL:
            rows: List[Dict[str, Any]] = []
            for p in _psutil.process_iter(attrs=["pid", "name", "username", "status",
                                                 "cpu_percent", "memory_percent",
                                                 "create_time", "cmdline"]):
                try:
                    i = p.info
                    rows.append({
                        "pid": i["pid"],
                        "name": i.get("name"),
                        "user": i.get("username"),
                        "status": i.get("status"),
                        "cpu_percent": round(i.get("cpu_percent") or 0.0, 1),
                        "memory_percent": round(i.get("memory_percent") or 0.0, 2),
                        "started": self._iso(i["create_time"]) if i.get("create_time") else None,
                        "cmdline": " ".join(i.get("cmdline") or [])[:300],
                    })
                except Exception:  # noqa: BLE001 — a process may vanish mid-iteration
                    continue
            key = {"cpu": "cpu_percent", "mem": "memory_percent", "memory": "memory_percent",
                   "pid": "pid", "name": "name"}.get(sort_by, "cpu_percent")
            reverse = key in ("cpu_percent", "memory_percent")
            rows.sort(key=lambda r: (r.get(key) or 0), reverse=reverse)
            return {"ok": True, "count": len(rows), "processes": rows[:cap]}
        return self._list_processes_from_proc(cap)

    def _list_processes_from_proc(self, cap: int) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                pid = int(entry)
                try:
                    with open(f"/proc/{pid}/comm", "r") as f:
                        name = f.read().strip()
                    with open(f"/proc/{pid}/cmdline", "rb") as f:
                        cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
                except OSError:
                    continue
                rows.append({"pid": pid, "name": name, "cmdline": cmdline[:300]})
                if len(rows) >= cap:
                    break
            rows.sort(key=lambda r: r["pid"])
            return {"ok": True, "count": len(rows), "processes": rows[:cap],
                    "note": "psutil not installed — read from /proc"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"psutil not installed and /proc unreadable: {exc}"}

    def process_info(self, pid: int) -> Dict[str, Any]:
        """Detailed info for one process (name, user, status, cpu/mem, threads, open files, conns)."""
        pid = int(pid)
        if not _HAS_PSUTIL:
            try:
                with open(f"/proc/{pid}/comm", "r") as f:
                    name = f.read().strip()
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
                return {"ok": True, "pid": pid, "name": name, "cmdline": cmdline,
                        "note": "psutil not installed — limited info from /proc"}
            except OSError as exc:
                return {"ok": False, "error": f"no such process {pid}: {exc}"}
        try:
            p = _psutil.Process(pid)
            with p.oneshot():
                return {
                    "ok": True,
                    "pid": pid,
                    "name": p.name(),
                    "user": p.username(),
                    "status": p.status(),
                    "cpu_percent": p.cpu_percent(interval=0.1),
                    "memory_percent": round(p.memory_percent(), 2),
                    "memory_rss": p.memory_info().rss,
                    "num_threads": p.num_threads(),
                    "nice": p.nice(),
                    "ppid": p.ppid(),
                    "created": self._iso(p.create_time()),
                    "cmdline": " ".join(p.cmdline()),
                    "cwd": _safe(lambda: p.cwd()),
                    "exe": _safe(lambda: p.exe()),
                    "num_fds": _safe(lambda: p.num_fds()),
                }
        except _psutil.NoSuchProcess:
            return {"ok": False, "error": f"no such process {pid}"}
        except _psutil.AccessDenied:
            return {"ok": False, "error": f"access denied for process {pid}"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def signal_process(self, pid: int, sig: Any = signal.SIGTERM) -> Dict[str, Any]:
        """Send a signal to a process by PID (respecting the protected-PID guard)."""
        blocked = self._enabled()
        if blocked:
            return blocked
        pid = int(pid)
        blocked = self._protected(pid)
        if blocked:
            return blocked
        try:
            signum = _resolve_signal(sig)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        try:
            os.kill(pid, signum)
            return {"ok": True, "pid": pid, "signal": signum, "name": _signal_name(signum)}
        except ProcessLookupError:
            return {"ok": False, "error": f"no such process {pid}"}
        except PermissionError:
            return {"ok": False, "error": f"insufficient permission to signal pid {pid} "
                                          "(try the privileged path)"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def terminate_process(self, pid: int) -> Dict[str, Any]:
        """Politely terminate a process (SIGTERM)."""
        return self.signal_process(pid, signal.SIGTERM)

    def kill_process(self, pid: int) -> Dict[str, Any]:
        """Forcefully kill a process (SIGKILL)."""
        return self.signal_process(pid, signal.SIGKILL)

    def set_nice(self, pid: int, value: int) -> Dict[str, Any]:
        """Set a process's scheduling niceness (-20 highest .. 19 lowest priority)."""
        blocked = self._enabled() or self._protected(int(pid))
        if blocked:
            return blocked
        try:
            os.setpriority(os.PRIO_PROCESS, int(pid), int(value))
            return {"ok": True, "pid": int(pid), "nice": int(value)}
        except ProcessLookupError:
            return {"ok": False, "error": f"no such process {pid}"}
        except PermissionError:
            return {"ok": False, "error": "insufficient permission to renice (lowering niceness "
                                          "below the current value needs privilege)"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    # ------------------------------------------------------------------ #
    # Service / daemon control
    # ------------------------------------------------------------------ #
    def service_status(self, name: str) -> Dict[str, Any]:
        """Report a systemd unit's active/enabled state (read-only, unprivileged)."""
        if not name:
            return {"ok": False, "error": "empty service name"}
        systemctl = shutil.which("systemctl")
        if systemctl is None:
            return {"ok": False, "error": "systemctl not available on this host"}
        active = self._run_plain([systemctl, "is-active", name], timeout_s=10.0)
        enabled = self._run_plain([systemctl, "is-enabled", name], timeout_s=10.0)
        return {"ok": True, "service": name,
                "active": (active.stdout or active.stderr).strip(),
                "enabled": (enabled.stdout or enabled.stderr).strip()}

    def service_action(self, name: str, action: str, *, timeout_s: Optional[float] = None
                       ) -> Dict[str, Any]:
        """Control a systemd unit: start/stop/restart/reload/enable/disable (elevated)."""
        blocked = self._enabled()
        if blocked:
            return blocked
        if not self.allow_service_control:
            return {"ok": False, "error": "service control is disabled by config"}
        if not name:
            return {"ok": False, "error": "empty service name"}
        action = str(action).lower().strip()
        if action not in _SERVICE_ACTIONS:
            return {"ok": False, "error": f"unknown service action {action!r} "
                                          f"(allowed: {', '.join(_SERVICE_ACTIONS)})"}
        systemctl = shutil.which("systemctl") or "systemctl"
        if action == "status":
            return self.service_status(name)
        res = self._run_priv([systemctl, action, name], timeout_s)
        res.setdefault("service", name)
        res.setdefault("action", action)
        return res

    # ------------------------------------------------------------------ #
    # Package management — finally exercises the PKG_INSTALL capability
    # ------------------------------------------------------------------ #
    def detect_package_manager(self) -> Optional[str]:
        """Return the first available system package manager (or the configured override).

        The override is honoured only when it is actually on PATH. Both branches of this
        ternary used to return it regardless, which made the ``which`` probe dead code: a
        misconfigured ``package_manager`` was handed straight to ``_package_action``, which
        (unlike ``package_query``) does not re-probe the binary, so the install surfaced as a
        raw exec failure instead of the honest "no supported package manager found".
        """
        if self.package_manager and shutil.which(self.package_manager):
            return self.package_manager
        for mgr in _PKG_MANAGERS:
            if shutil.which(mgr):
                return mgr
        return None

    def _pkg_argv(self, mgr: str, op: str, name: str) -> Optional[List[str]]:
        """Build the argv for (install|remove|query) under a given manager. None if unsupported."""
        table = {
            "apt-get": {"install": ["apt-get", "install", "-y", name],
                        "remove": ["apt-get", "remove", "-y", name],
                        "query": ["dpkg", "-s", name]},
            "dnf": {"install": ["dnf", "install", "-y", name],
                    "remove": ["dnf", "remove", "-y", name],
                    "query": ["rpm", "-q", name]},
            "yum": {"install": ["yum", "install", "-y", name],
                    "remove": ["yum", "remove", "-y", name],
                    "query": ["rpm", "-q", name]},
            "pacman": {"install": ["pacman", "-S", "--noconfirm", name],
                       "remove": ["pacman", "-R", "--noconfirm", name],
                       "query": ["pacman", "-Q", name]},
            "zypper": {"install": ["zypper", "--non-interactive", "install", name],
                       "remove": ["zypper", "--non-interactive", "remove", name],
                       "query": ["rpm", "-q", name]},
            "apk": {"install": ["apk", "add", name],
                    "remove": ["apk", "del", name],
                    "query": ["apk", "info", "-e", name]},
            "brew": {"install": ["brew", "install", name],
                     "remove": ["brew", "uninstall", name],
                     "query": ["brew", "list", name]},
            "pip": {"install": ["pip", "install", name],
                    "remove": ["pip", "uninstall", "-y", name],
                    "query": ["pip", "show", name]},
        }
        return table.get(mgr, {}).get(op)

    def package_query(self, name: str) -> Dict[str, Any]:
        """Query whether a package is installed (read-only, unprivileged)."""
        if not name:
            return {"ok": False, "error": "empty package name"}
        mgr = self.detect_package_manager()
        if not mgr:
            return {"ok": False, "error": "no supported package manager found"}
        argv = self._pkg_argv(mgr, "query", name)
        if not argv or not shutil.which(argv[0]):
            return {"ok": False, "error": f"query unsupported for manager {mgr!r}"}
        res = self._run_plain(argv, timeout_s=30.0)
        return {"ok": True, "manager": mgr, "package": name, "installed": res.ok,
                "detail": (res.stdout or res.stderr).strip()[:500]}

    def package_install(self, name: str, *, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        """Install a package via the detected manager (elevated for system managers)."""
        return self._package_action("install", name, timeout_s)

    def package_remove(self, name: str, *, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        """Remove a package via the detected manager (elevated for system managers)."""
        return self._package_action("remove", name, timeout_s)

    def _package_action(self, op: str, name: str, timeout_s: Optional[float]) -> Dict[str, Any]:
        blocked = self._enabled()
        if blocked:
            return blocked
        if not self.allow_package_management:
            return {"ok": False, "error": "package management is disabled by config"}
        if not name:
            return {"ok": False, "error": "empty package name"}
        mgr = self.detect_package_manager()
        if not mgr:
            return {"ok": False, "error": "no supported package manager found"}
        argv = self._pkg_argv(mgr, op, name)
        if not argv:
            return {"ok": False, "error": f"{op} unsupported for manager {mgr!r}"}
        # pip/brew run as the invoking user; system managers need root.
        t = timeout_s if timeout_s is not None else max(self.default_timeout_s, 300.0)
        if mgr in ("pip", "brew"):
            res = self._run_plain(argv, timeout_s=t).to_dict()
        else:
            res = self._run_priv(argv, t)
        res.setdefault("manager", mgr)
        res.setdefault("package", name)
        res.setdefault("action", op)
        return res

    # ------------------------------------------------------------------ #
    # Power control (off by default; elevated)
    # ------------------------------------------------------------------ #
    def power_action(self, action: str, *, delay_min: int = 0) -> Dict[str, Any]:
        """Drive machine power: shutdown/poweroff/reboot/halt/suspend/hibernate/logout (elevated)."""
        blocked = self._enabled()
        if blocked:
            return blocked
        if not self.allow_power:
            return {"ok": False, "error": "power control is disabled by config "
                                          "(set NYXARA_AGENCY__SYSTEM__ALLOW_POWER=true to enable)"}
        action = str(action).lower().strip()
        if action not in _POWER_ACTIONS:
            return {"ok": False, "error": f"unknown power action {action!r} "
                                          f"(allowed: {', '.join(_POWER_ACTIONS)})"}
        systemctl = shutil.which("systemctl")
        verb = {"shutdown": "poweroff", "restart": "reboot"}.get(action, action)
        if systemctl and verb in ("poweroff", "reboot", "halt", "suspend", "hibernate"):
            argv: List[str] = [systemctl, verb]
        elif verb == "logout":
            loginctl = shutil.which("loginctl")
            argv = [loginctl, "terminate-user", os.environ.get("USER", "")] if loginctl else ["pkill", "-KILL", "-u", os.environ.get("USER", "")]
        else:
            # Fallback to the classic shutdown(8) binary.
            when = f"+{int(delay_min)}" if delay_min else "now"
            flag = {"poweroff": "-h", "halt": "-H", "reboot": "-r"}.get(verb, "-h")
            argv = ["shutdown", flag, when]
        res = self._run_priv(argv, timeout_s=30.0)
        res.setdefault("action", action)
        return res

    # ------------------------------------------------------------------ #
    # Users / sessions
    # ------------------------------------------------------------------ #
    def list_users(self) -> Dict[str, Any]:
        """Local user accounts parsed from ``/etc/passwd`` (read-only, unprivileged)."""
        users: List[Dict[str, Any]] = []
        try:
            import pwd  # POSIX-only
            for e in pwd.getpwall():
                users.append({"name": e.pw_name, "uid": e.pw_uid, "gid": e.pw_gid,
                              "home": e.pw_dir, "shell": e.pw_shell, "gecos": e.pw_gecos})
            return {"ok": True, "users": users}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"cannot enumerate users: {exc}"}

    def who(self) -> Dict[str, Any]:
        """Currently logged-in sessions (user, terminal, host, login time)."""
        if _HAS_PSUTIL:
            try:
                return {"ok": True, "sessions": [
                    {"user": u.name, "terminal": u.terminal, "host": u.host,
                     "started": self._iso(u.started) if u.started else None, "pid": getattr(u, "pid", None)}
                    for u in _psutil.users()]}
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": str(exc)}
        res = self._run_plain(["who"], timeout_s=10.0)
        return {"ok": res.ok, "raw": res.stdout.strip(), "error": res.error}

    def user_action(self, action: str, username: str, *, password: Optional[str] = None,
                    timeout_s: Optional[float] = None) -> Dict[str, Any]:
        """Manage a local user: add/remove/modify/lock/unlock/passwd (elevated)."""
        blocked = self._enabled()
        if blocked:
            return blocked
        if not self.allow_user_management:
            return {"ok": False, "error": "user management is disabled by config"}
        if not username:
            return {"ok": False, "error": "empty username"}
        action = str(action).lower().strip()
        if action not in _USER_ACTIONS:
            return {"ok": False, "error": f"unknown user action {action!r} "
                                          f"(allowed: {', '.join(_USER_ACTIONS)})"}
        argv_map = {
            "add": ["useradd", "-m", username],
            "remove": ["userdel", "-r", username],
            "delete": ["userdel", "-r", username],
            "modify": ["usermod", username],
            "lock": ["usermod", "-L", username],
            "unlock": ["usermod", "-U", username],
        }
        if action == "passwd":
            if not password:
                return {"ok": False, "error": "passwd requires a password"}
            # `chpasswd` reads user:password on stdin; feed it through the privileged runner as a
            # shell pipeline so the secret is set atomically. The runner binds the sudo secret late.
            cmd = f"echo {username}:{password} | chpasswd"
            res = self._run_priv(cmd, timeout_s)
        else:
            res = self._run_priv(argv_map[action], timeout_s)
        res.setdefault("user", username)
        res.setdefault("action", action)
        return res

    # ------------------------------------------------------------------ #
    # Environment & kernel tunables
    # ------------------------------------------------------------------ #
    def get_env(self, name: str = "") -> Dict[str, Any]:
        """Read a single environment variable, or the whole (process) environment when empty."""
        if name:
            return {"ok": True, "name": name, "value": os.environ.get(name)}
        return {"ok": True, "env": dict(os.environ)}

    def set_env(self, name: str, value: str) -> Dict[str, Any]:
        """Set an environment variable for NYXARA's own process (and children she spawns)."""
        blocked = self._enabled()
        if blocked:
            return blocked
        if not name:
            return {"ok": False, "error": "empty variable name"}
        os.environ[str(name)] = str(value)
        return {"ok": True, "name": name, "value": value,
                "note": "process-scoped — affects this NYXARA process and children she spawns"}

    def sysctl_get(self, key: str) -> Dict[str, Any]:
        """Read a kernel tunable (``sysctl``) — read-only, unprivileged."""
        if not key:
            return {"ok": False, "error": "empty sysctl key"}
        # Prefer /proc/sys (no external binary needed); fall back to the sysctl(8) tool.
        proc_path = "/proc/sys/" + key.replace(".", "/")
        try:
            with open(proc_path, "r") as f:
                return {"ok": True, "key": key, "value": f.read().strip()}
        except OSError:
            pass
        sysctl = shutil.which("sysctl")
        if sysctl is None:
            return {"ok": False, "error": f"cannot read {key!r} (no /proc/sys entry, sysctl absent)"}
        res = self._run_plain([sysctl, "-n", key], timeout_s=10.0)
        return {"ok": res.ok, "key": key, "value": res.stdout.strip(), "error": res.error}

    def sysctl_set(self, key: str, value: str, *, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        """Write a kernel tunable (``sysctl -w``) — elevated."""
        blocked = self._enabled()
        if blocked:
            return blocked
        if not key:
            return {"ok": False, "error": "empty sysctl key"}
        res = self._run_priv(["sysctl", "-w", f"{key}={value}"], timeout_s)
        res.setdefault("key", key)
        res.setdefault("value", value)
        return res

    # ------------------------------------------------------------------ #
    # Elevation posture (read-only convenience passthrough)
    # ------------------------------------------------------------------ #
    def privilege_status(self) -> Dict[str, Any]:
        """The current elevation posture (euid, is_root, sudo availability) — read-only."""
        status = privilege_status()
        status["ok"] = True
        return status


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _safe(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None


def _resolve_signal(sig: Any) -> int:
    """Coerce a signal given as an int, a name ('SIGKILL'/'KILL'/'9'), or a Signals member."""
    if isinstance(sig, signal.Signals):
        return int(sig)
    if isinstance(sig, int):
        return sig
    s = str(sig).strip().upper()
    if s.isdigit():
        return int(s)
    if not s.startswith("SIG"):
        s = "SIG" + s
    member = getattr(signal, s, None)
    if member is None:
        raise ValueError(f"unknown signal {sig!r}")
    return int(member)


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except Exception:  # noqa: BLE001
        return str(signum)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA system-control self-test")
    print("=" * 70)

    sc = SystemControl()
    print(f"\npsutil available    : {_HAS_PSUTIL}")

    info = sc.system_info()
    assert info["ok"]
    print(f"system              : {info['system']} {info['release']} ({info['machine']})")
    print(f"uptime_s            : {info['uptime_s']}  cpu_count={info['cpu_count']}")

    mem = sc.memory_info()
    assert mem["ok"], mem
    print(f"memory total/used   : {mem['virtual']['total']} / {mem['virtual'].get('used')}")

    procs = sc.list_processes(limit=5)
    assert procs["ok"] and procs["processes"], procs
    print(f"top processes       : {[p['name'] for p in procs['processes']]}")

    # protected-PID guard: refusing init and self must be DATA, never a raise.
    r1 = sc.kill_process(1)
    assert r1["ok"] is False and "protected" in r1["error"], r1
    r2 = sc.terminate_process(os.getpid())
    assert r2["ok"] is False and "own process" in r2["error"], r2
    print("protected-PID guard : refused pid 1 and self (as data) ✓")

    # power is off by default — must refuse as data.
    rp = sc.power_action("reboot")
    assert rp["ok"] is False and "disabled" in rp["error"], rp
    print("power guard         : refused (off by default) ✓")

    # a bogus service action fails as data, never raises.
    rs = sc.service_action("definitely-not-a-real-unit.service", "frobnicate")
    assert rs["ok"] is False, rs
    print("service verb guard  : rejected unknown verb ✓")

    print("\nALL SELF-TESTS PASSED ✓")
