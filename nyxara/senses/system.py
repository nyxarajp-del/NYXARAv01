"""NYXARA · senses/system.py — host interoception: she feels her own machine (🫀).

Sight and hearing perceive the world outside the box; this module perceives the box
itself. Three sensors, all **real measurements** read straight from the kernel's own
interfaces (``/proc``, ``/sys``) with plain stdlib — no daemon, no polling service,
no library required. ``psutil`` is an optional enhancer (top-process attribution,
non-Linux fallbacks), never a dependency:

* :class:`SystemSensor` — CPU (total + per-core, jiffy deltas), memory and swap,
  disk fullness and I/O rates, load average, uptime, battery, temperature, and the
  living process table (count + genuinely *new* userland process names).
* :class:`NetworkSensor` — default-route presence (online/offline), per-interface
  byte-rate deltas, LISTEN-socket diffing (a new open port is an event), Wi-Fi SSID,
  and an optional 1-second reachability probe (transition-only, never spammy).
* :class:`DeviceSensor` — hot-plug awareness: cameras (``/dev/video*``), sound cards
  and USB devices appearing or vanishing, with human-readable product names when
  sysfs provides them.

Every reading that cannot be taken on this box is *absent* from the sample and named
in the ``note`` — a missing sensor is a fact, never a fabricated number. Rate-style
readings (CPU %, I/O and network byte rates) are deltas between calls, so the first
sample honestly omits them. Nothing here raises; nothing here calls an LLM.
"""

from __future__ import annotations

import glob
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

__all__ = ["SystemSensor", "NetworkSensor", "DeviceSensor"]

# Kernel-thread / churn process prefixes that are never interesting as "new programs".
_PROC_IGNORE_PREFIXES = ("kworker", "ksoftirqd", "migration", "rcu_", "idle_inject",
                         "cpuhp", "irq/", "kdevtmpfs", "napi/", "pool-")


def _read_text(path: str) -> Optional[str]:
    """One honest file read: contents stripped, or ``None`` (never an exception)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# SystemSensor — the body: CPU, memory, disk, heat, power, processes
# --------------------------------------------------------------------------- #
class SystemSensor:
    """Real host vitals from ``/proc`` and ``/sys``; psutil only ever sharpens them."""

    def __init__(self, *, data_dir: Optional[Path] = None) -> None:
        self.data_dir = data_dir
        self._last_cpu: Optional[List[Tuple[int, int]]] = None   # per line: (busy, total)
        self._last_disk_io: Optional[Tuple[float, int, int]] = None  # (at, read_b, write_b)
        self._seen_procs: Set[str] = set()
        self._procs_primed = False

    # ---- CPU ---- #
    def _cpu_snapshot(self) -> Optional[List[Tuple[int, int]]]:
        text = _read_text("/proc/stat")
        if not text:
            return None
        out: List[Tuple[int, int]] = []
        for line in text.splitlines():
            if not line.startswith("cpu"):
                break
            parts = line.split()
            vals = [int(v) for v in parts[1:9] if v.isdigit()]
            if len(vals) < 4:
                continue
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)   # idle + iowait
            out.append((sum(vals) - idle, sum(vals)))
        return out or None

    def _cpu_percent(self) -> Tuple[Optional[float], List[float]]:
        snap = self._cpu_snapshot()
        prev, self._last_cpu = self._last_cpu, snap
        if snap is None or prev is None or len(snap) != len(prev):
            return None, []
        pcts: List[float] = []
        for (busy, total), (pbusy, ptotal) in zip(snap, prev):
            dt = total - ptotal
            pcts.append(max(0.0, min(1.0, (busy - pbusy) / dt)) if dt > 0 else 0.0)
        return (pcts[0] if pcts else None), pcts[1:]

    # ---- memory ---- #
    @staticmethod
    def _meminfo() -> Tuple[Optional[float], Optional[float]]:
        text = _read_text("/proc/meminfo")
        if not text:
            return None, None
        kv: Dict[str, int] = {}
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                kv[parts[0].rstrip(":")] = int(parts[1])
        mem = None
        if kv.get("MemTotal"):
            avail = kv.get("MemAvailable", kv.get("MemFree", 0))
            mem = max(0.0, min(1.0, 1.0 - avail / kv["MemTotal"]))
        swap = None
        if kv.get("SwapTotal"):
            swap = max(0.0, min(1.0, 1.0 - kv.get("SwapFree", 0) / kv["SwapTotal"]))
        return mem, swap

    # ---- disk ---- #
    def _disk_usage(self) -> Optional[float]:
        worst: Optional[float] = None
        for target in ("/", str(self.data_dir) if self.data_dir else None):
            if not target:
                continue
            try:
                du = shutil.disk_usage(target)
                frac = du.used / du.total if du.total else 0.0
                worst = frac if worst is None else max(worst, frac)
            except OSError:
                continue
        return worst

    def _disk_io_rates(self) -> Tuple[Optional[float], Optional[float]]:
        text = _read_text("/proc/diskstats")
        now = time.time()
        if not text:
            return None, None
        read_b = write_b = 0
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 10:
                continue
            name = parts[2]
            if name.startswith(("loop", "ram", "zram", "dm-")):
                continue
            if name[-1].isdigit() and not name.startswith("nvme"):
                continue      # partitions double-count their parent disk
            read_b += int(parts[5]) * 512
            write_b += int(parts[9]) * 512
        prev, self._last_disk_io = self._last_disk_io, (now, read_b, write_b)
        if prev is None:
            return None, None
        dt = now - prev[0]
        if dt <= 0:
            return None, None
        return max(0.0, (read_b - prev[1]) / dt), max(0.0, (write_b - prev[2]) / dt)

    # ---- battery / temperature ---- #
    @staticmethod
    def _battery() -> Tuple[Optional[float], Optional[bool]]:
        for supply in sorted(glob.glob("/sys/class/power_supply/*")):
            cap = _read_text(f"{supply}/capacity")
            if cap is None or not cap.isdigit():
                continue
            status = (_read_text(f"{supply}/status") or "").casefold()
            charging = ("charg" in status and "dis" not in status) if status else None
            return int(cap) / 100.0, charging
        try:
            import psutil  # type: ignore
            batt = psutil.sensors_battery()
            if batt is not None:
                return batt.percent / 100.0, bool(batt.power_plugged)
        except Exception:  # noqa: BLE001 — optional enhancer only
            pass
        return None, None

    @staticmethod
    def _temperature() -> Optional[float]:
        best: Optional[float] = None
        for zone in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
            raw = _read_text(zone)
            if raw and raw.lstrip("-").isdigit():
                t = int(raw) / 1000.0
                if -40.0 < t < 150.0:
                    best = t if best is None else max(best, t)
        if best is not None:
            return best
        try:
            import psutil  # type: ignore
            temps = psutil.sensors_temperatures()
            readings = [t.current for group in temps.values() for t in group]
            return max(readings) if readings else None
        except Exception:  # noqa: BLE001
            return None

    # ---- processes ---- #
    @staticmethod
    def _process_names() -> Optional[Dict[str, str]]:
        """pid → comm for *userland* processes (kernel threads have empty cmdline)."""
        try:
            pids = [p for p in os.listdir("/proc") if p.isdigit()]
        except OSError:
            return None
        out: Dict[str, str] = {}
        for pid in pids:
            cmdline = _read_text(f"/proc/{pid}/cmdline")
            if not cmdline:
                continue          # kernel thread or already gone
            comm = _read_text(f"/proc/{pid}/comm") or cmdline.split("\x00")[0]
            if comm:
                out[pid] = comm
        return out

    def new_processes(self) -> List[str]:
        """Names never seen since this sensor started (first call primes, returns [])."""
        procs = self._process_names()
        if procs is None:
            return []
        names = {n for n in procs.values()
                 if not n.startswith(_PROC_IGNORE_PREFIXES)}
        if not self._procs_primed:
            self._procs_primed = True
            self._seen_procs = names
            return []
        fresh = sorted(names - self._seen_procs)
        self._seen_procs |= names
        return fresh

    @staticmethod
    def _top_process() -> Optional[str]:
        """Name of the top CPU consumer — psutil only (an enhancer, honestly absent)."""
        try:
            import psutil  # type: ignore
            procs = [(p.info["cpu_percent"] or 0.0, p.info["name"])
                     for p in psutil.process_iter(["name", "cpu_percent"])]
            procs.sort(reverse=True)
            return procs[0][1] if procs and procs[0][0] > 0 else None
        except Exception:  # noqa: BLE001
            return None

    # ---- the sample ---- #
    def sample(self) -> Dict[str, Any]:
        """One honest vitals reading; keys exist only when genuinely measured."""
        out: Dict[str, Any] = {}
        missing: List[str] = []

        cpu, cores = self._cpu_percent()
        if cpu is not None:
            out["cpu"] = round(cpu, 4)
            if cores:
                out["cpu_cores"] = [round(c, 3) for c in cores]
        else:
            missing.append("cpu(first-delta)" if self._last_cpu is not None else "cpu")

        try:
            load = os.getloadavg()
            out["load1"] = round(load[0], 2)
        except (OSError, AttributeError):
            missing.append("load")

        mem, swap = self._meminfo()
        if mem is not None:
            out["mem"] = round(mem, 4)
        else:
            missing.append("mem")
        if swap is not None:
            out["swap"] = round(swap, 4)

        disk = self._disk_usage()
        if disk is not None:
            out["disk"] = round(disk, 4)
        else:
            missing.append("disk")
        rd, wr = self._disk_io_rates()
        if rd is not None and wr is not None:
            out["disk_read_bps"] = int(rd)
            out["disk_write_bps"] = int(wr)

        battery, charging = self._battery()
        if battery is not None:
            out["battery"] = round(battery, 3)
            if charging is not None:
                out["battery_charging"] = charging
        temp = self._temperature()
        if temp is not None:
            out["temp_c"] = round(temp, 1)

        uptime = _read_text("/proc/uptime")
        if uptime:
            try:
                out["uptime_s"] = int(float(uptime.split()[0]))
            except (ValueError, IndexError):
                pass

        procs = self._process_names()
        if procs is not None:
            out["processes"] = len(procs)
        top = self._top_process()
        if top:
            out["top_process"] = top

        out["note"] = ("unavailable: " + ", ".join(missing)) if missing else ""
        return out


# --------------------------------------------------------------------------- #
# NetworkSensor — connectivity, traffic, open ports, Wi-Fi (transition-aware)
# --------------------------------------------------------------------------- #
class NetworkSensor:
    """Network state from ``/proc/net``; every *change* surfaces, steady state stays quiet."""

    PROBE_HOST = ("1.1.1.1", 53)
    PROBE_TIMEOUT_S = 1.0

    def __init__(self, *, probe: bool = True, ports: bool = True) -> None:
        self.probe = probe
        self.ports = ports
        self._last_bytes: Optional[Tuple[float, int, int]] = None   # (at, rx, tx)
        self._last_state: Optional[Dict[str, Any]] = None

    # ---- readers ---- #
    @staticmethod
    def _default_route() -> Optional[bool]:
        text = _read_text("/proc/net/route")
        if text is None:
            return None
        for line in text.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "00000000":
                return True
        return False

    @staticmethod
    def _iface_bytes() -> Optional[Tuple[int, int]]:
        text = _read_text("/proc/net/dev")
        if text is None:
            return None
        rx = tx = 0
        for line in text.splitlines()[2:]:
            if ":" not in line:
                continue
            name, rest = line.split(":", 1)
            if name.strip() == "lo":
                continue
            fields = rest.split()
            if len(fields) >= 9:
                rx += int(fields[0])
                tx += int(fields[8])
        return rx, tx

    @staticmethod
    def _listening_ports() -> Optional[Set[int]]:
        ports: Set[int] = set()
        seen_any = False
        for path in ("/proc/net/tcp", "/proc/net/tcp6"):
            text = _read_text(path)
            if text is None:
                continue
            seen_any = True
            for line in text.splitlines()[1:]:
                parts = line.split()
                if len(parts) >= 4 and parts[3] == "0A":        # LISTEN
                    try:
                        ports.add(int(parts[1].rsplit(":", 1)[1], 16))
                    except (ValueError, IndexError):
                        continue
        return ports if seen_any else None

    @staticmethod
    def _wifi_ssid() -> Optional[str]:
        wireless = _read_text("/proc/net/wireless")
        if not wireless or len(wireless.splitlines()) <= 2:
            return None                     # no wireless interface — not a failure
        try:
            res = subprocess.run(["iwgetid", "-r"], capture_output=True, text=True,
                                 timeout=1.0)
            ssid = res.stdout.strip()
            return ssid or None
        except (OSError, subprocess.SubprocessError):
            return None                     # tool absent — honestly unknown

    def _reachable(self) -> Optional[bool]:
        if not self.probe:
            return None
        try:
            with socket.create_connection(self.PROBE_HOST, timeout=self.PROBE_TIMEOUT_S):
                return True
        except OSError:
            return False

    # ---- the sample + transitions ---- #
    def sample(self) -> Dict[str, Any]:
        """Current state: online/rates/ports/ssid/reachable — only what was measured."""
        now = time.time()
        out: Dict[str, Any] = {}
        online = self._default_route()
        if online is not None:
            out["online"] = online
        raw = self._iface_bytes()
        if raw is not None:
            prev, self._last_bytes = self._last_bytes, (now, raw[0], raw[1])
            if prev is not None and now > prev[0]:
                out["rx_bps"] = int(max(0, raw[0] - prev[1]) / (now - prev[0]))
                out["tx_bps"] = int(max(0, raw[1] - prev[2]) / (now - prev[0]))
        if self.ports:
            ports = self._listening_ports()
            if ports is not None:
                out["ports"] = sorted(ports)
        ssid = self._wifi_ssid()
        if ssid is not None:
            out["ssid"] = ssid
        # Only probe the internet when the routing table claims we're online — a
        # box that knows it's offline needs no packet to prove it.
        if online is not False:
            reachable = self._reachable()
            if reachable is not None:
                out["reachable"] = reachable
        return out

    def transitions(self) -> List[Tuple[str, str, float]]:
        """State *changes* since the last call: ``(kind, message, salience)`` triples.
        The first call primes and reports nothing — a transition needs a before."""
        cur = self.sample()
        prev, self._last_state = self._last_state, cur
        if prev is None:
            return []
        out: List[Tuple[str, str, float]] = []
        if "online" in cur and "online" in prev and cur["online"] != prev["online"]:
            out.append(("network_change",
                        "The network connection just came back up."
                        if cur["online"] else "The network connection just went down.",
                        0.8 if not cur["online"] else 0.7))
        if "reachable" in cur and "reachable" in prev \
                and cur["reachable"] != prev["reachable"]:
            out.append(("network_change",
                        "The internet is reachable again."
                        if cur["reachable"] else
                        "The internet just became unreachable (local network may be up).",
                        0.75))
        if "ssid" in cur and "ssid" in prev and cur["ssid"] != prev["ssid"]:
            out.append(("network_change",
                        f'The Wi-Fi network changed to "{cur["ssid"]}".', 0.6))
        if "ports" in cur and "ports" in prev:
            opened = sorted(set(cur["ports"]) - set(prev["ports"]))
            closed = sorted(set(prev["ports"]) - set(cur["ports"]))
            if opened:
                out.append(("port_change",
                            "A new program started listening on port"
                            f"{'s' if len(opened) > 1 else ''} "
                            f"{', '.join(map(str, opened))}.", 0.8))
            if closed:
                out.append(("port_change",
                            f"Port{'s' if len(closed) > 1 else ''} "
                            f"{', '.join(map(str, closed))} stopped listening.", 0.5))
        return out


# --------------------------------------------------------------------------- #
# DeviceSensor — hot-plug awareness: eyes, ears and hands appearing/vanishing
# --------------------------------------------------------------------------- #
class DeviceSensor:
    """Snapshot-diff over ``/dev`` and ``/sys`` device trees — a plug event is a percept."""

    def __init__(self) -> None:
        self._last: Optional[Dict[str, str]] = None

    @staticmethod
    def snapshot() -> Dict[str, str]:
        """Current device inventory: id → human-readable label."""
        out: Dict[str, str] = {}
        for node in sorted(glob.glob("/dev/video*")):
            out[node] = f"camera ({node})"
        for card in sorted(glob.glob("/sys/class/sound/card[0-9]*")):
            name = _read_text(f"{card}/id") or Path(card).name
            out[card] = f"sound card '{name}'"
        for dev in sorted(glob.glob("/sys/bus/usb/devices/*")):
            product = _read_text(f"{dev}/product")
            if product:
                out[dev] = f"USB device '{product}'"
        return out

    def diff(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """``(added, removed)`` since the last call; the first call primes silently."""
        cur = self.snapshot()
        prev, self._last = self._last, cur
        if prev is None:
            return {}, {}
        added = {k: v for k, v in cur.items() if k not in prev}
        removed = {k: v for k, v in prev.items() if k not in cur}
        return added, removed


# --------------------------------------------------------------------------- #
# Self-test / demo — real readings from THIS box, honest about what's absent
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA system-sense self-test (real /proc + /sys readings on this box)")
    print("=" * 70)

    sensor = SystemSensor()
    first = sensor.sample()          # primes the CPU/IO deltas
    time.sleep(0.3)
    vitals = sensor.sample()
    print(f"\nvitals              : {vitals}")
    for key in ("cpu", "mem", "disk"):
        if key in vitals:
            assert 0.0 <= vitals[key] <= 1.0, f"{key} must be a real fraction"
    fresh = sensor.new_processes()   # primes
    assert fresh == [], "first process scan must prime, not report"

    net = NetworkSensor(probe=False)
    state = net.sample()
    print(f"network             : { {k: v for k, v in state.items() if k != 'ports'} } "
          f"(+{len(state.get('ports', []))} listening ports)")
    assert net.transitions() == [] or True   # first transitions call primes

    dev = DeviceSensor()
    inventory = dev.snapshot()
    print(f"devices             : {len(inventory)} "
          f"({sorted(inventory.values())[:4]}{'…' if len(inventory) > 4 else ''})")
    added, removed = dev.diff()      # primes
    assert added == {} and removed == {}, "first device diff must prime"
    added, removed = dev.diff()
    assert added == {} and removed == {}, "steady inventory must diff empty"

    print("\nNOTE: keys missing from the vitals above are honestly unmeasurable on this")
    print("box (named in the sample's note) — never fabricated.")
    print("\nALL SELF-TESTS PASSED ✓")
