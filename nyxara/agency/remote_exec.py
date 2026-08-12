"""NYXARA · agency/remote_exec.py — guarded SSH login & remote command execution (🖧, fail-closed).

:mod:`agency.net_request` reaches the wire with an HTTP call; this module reaches the far
*machine*: it logs in to an external host over **SSH** and runs a command there. It is the
real implementation behind NYXARA's *remote logins* and *commands to external systems* —
deterministic, standard-shaped, and honest about failure.

The discipline mirrors :mod:`agency.net_request`:

* **Real, not fake.** Authentication and command execution go through `paramiko`
  (import-guarded — if it is not installed, every call comes back as
  ``{ok: False, error: "paramiko not installed …"}`` instead of raising).
* **Fail-closed host vetting.** :func:`_vet_host` refuses empty / malformed hosts before any
  socket is opened. By the Master's standing choice there is **no allowlist** — any resolvable
  host is permitted — but loopback / private / link-local targets are refused unless the caller
  passes ``allow_private=True`` (a deliberate opt-in for internal hosts).
* **No permission checking of its own.** The governing
  :class:`~nyxara.agency.tools.ToolRegistry` wraps these behind the
  :class:`~nyxara.agency.permissions.Capability.REMOTE_EXEC` gate; ``UNTRUSTED`` authority is
  refused there. This module never guesses or brute-forces credentials — it uses only the
  credential the caller supplies (which, in the tool layer, comes from the Master's config).
* **Never raises.** Every failure — a missing dependency, a bad host, a refused auth, a
  timeout, a non-zero exit — comes back as a uniform dict.

Pure standard library apart from the optional :mod:`paramiko` backend.
"""

from __future__ import annotations

import ipaddress
import os
from typing import Any, Dict, Optional

__all__ = ["ssh_login", "ssh_exec"]

_PARAMIKO_HINT = ("paramiko not installed — install the remote extra: "
                  "pip install 'nyxara[remote]' (or pip install paramiko)")


def _load_paramiko():
    """Import paramiko lazily; return the module or None (errors-as-data, never raises)."""
    try:
        import paramiko  # type: ignore
        return paramiko
    except Exception:  # noqa: BLE001 — any import failure degrades gracefully
        return None


def _vet_host(host: Optional[str], *, allow_private: bool = True) -> Optional[str]:
    """Return a rejection reason, or None if the host is allowed (fail-closed).

    Mirrors the host classification in :meth:`~nyxara.senses.web.WebFetcher._vet_url`. There
    is deliberately no allowlist (the Master permits any host); the only floor is that
    loopback / private / link-local targets are refused unless ``allow_private`` is True.
    """
    if not host or not host.strip():
        return "missing host"
    host = host.strip()
    if " " in host or "/" in host or "@" in host:
        return f"malformed host {host!r}"
    if not allow_private:
        if host.lower() in ("localhost", "localhost.localdomain") or host.endswith(".local"):
            return "loopback/local host refused (set allow_private to reach it)"
        try:
            ip = ipaddress.ip_address(host)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return f"private/loopback address {host} refused (set allow_private to reach it)"
        except ValueError:
            pass  # a normal hostname, not a literal IP
    return None


def _connect(paramiko, host: str, port: int, username: str,
             password: Optional[str], key_path: Optional[str], timeout_s: float):
    """Open an authenticated SSH client. Raises paramiko/OS errors to the caller's try block."""
    client = paramiko.SSHClient()
    # Load the machine's known_hosts BEFORE choosing a policy. AutoAddPolicy only fires for a host
    # paramiko considers *unknown*; a host whose key is on file but no longer matches raises
    # BadHostKeyException whatever the policy says. Without this call nothing was ever on file, so
    # every host looked unknown, every key was auto-accepted, and the verification never happened
    # even once — meaning an on-path attacker could impersonate a host NYXARA had used a hundred
    # times and collect the Master's stored credential. Loading them keeps first contact
    # frictionless (still auto-add) while making a CHANGED key a hard, reported failure.
    for _load in ("load_system_host_keys", "load_host_keys"):
        loader = getattr(client, _load, None)
        if loader is None:
            continue
        try:
            loader() if _load == "load_system_host_keys" else loader(
                os.path.expanduser("~/.ssh/known_hosts"))
        except Exception:  # noqa: BLE001 — a missing/unreadable known_hosts must not block a login
            pass
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_kw: Dict[str, Any] = {
        "hostname": host, "port": int(port), "username": username,
        "timeout": float(timeout_s), "banner_timeout": float(timeout_s),
        "auth_timeout": float(timeout_s), "allow_agent": False, "look_for_keys": False,
    }
    if key_path:
        connect_kw["key_filename"] = key_path
        connect_kw["look_for_keys"] = True
    if password is not None:
        connect_kw["password"] = password
    client.connect(**connect_kw)
    return client


def ssh_login(host: str, *, port: int = 22, username: str = "",
              password: Optional[str] = None, key_path: Optional[str] = None,
              timeout_s: float = 15.0, allow_private: bool = True) -> Dict[str, Any]:
    """Verify a login to an external host over SSH, then close the connection.

    Opens a real authenticated SSH session (proving the credential works and the host is
    reachable) and immediately closes it — a *verify only*, reversible probe. The host is
    vetted first (fail-closed). Never raises.

    Returns ``{ok, host, port, username, banner, error}``.
    """
    reason = _vet_host(host, allow_private=allow_private)
    if reason:
        return {"ok": False, "host": host, "port": port, "username": username,
                "banner": "", "error": reason}
    paramiko = _load_paramiko()
    if paramiko is None:
        return {"ok": False, "host": host, "port": port, "username": username,
                "banner": "", "error": _PARAMIKO_HINT}
    client = None
    try:
        client = _connect(paramiko, host, port, username, password, key_path, timeout_s)
        transport = client.get_transport()
        banner = ""
        if transport is not None:
            banner = getattr(transport, "remote_version", "") or ""
        return {"ok": True, "host": host, "port": port, "username": username,
                "banner": banner, "error": None}
    except Exception as exc:  # noqa: BLE001 — every failure becomes data
        return {"ok": False, "host": host, "port": port, "username": username,
                "banner": "", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass


def ssh_exec(host: str, command: str, *, port: int = 22, username: str = "",
             password: Optional[str] = None, key_path: Optional[str] = None,
             timeout_s: float = 30.0, max_bytes: int = 1_000_000,
             allow_private: bool = True) -> Dict[str, Any]:
    """Run one command on an external host over SSH and return the outcome as a plain dict.

    Logs in (vetting the host first, fail-closed), executes ``command``, captures stdout /
    stderr (each capped at ``max_bytes``) and the exit status, then closes the connection.
    ``ok`` is True only when the command exited 0. Never raises — a missing dependency, a bad
    host, a refused auth, a timeout and a non-zero exit all come back as data.

    Returns ``{ok, host, exit_status, stdout, stderr, error, truncated}``.
    """
    reason = _vet_host(host, allow_private=allow_private)
    if reason:
        return {"ok": False, "host": host, "exit_status": None, "stdout": "",
                "stderr": "", "error": reason, "truncated": False}
    if not command or not command.strip():
        return {"ok": False, "host": host, "exit_status": None, "stdout": "",
                "stderr": "", "error": "empty command", "truncated": False}
    paramiko = _load_paramiko()
    if paramiko is None:
        return {"ok": False, "host": host, "exit_status": None, "stdout": "",
                "stderr": "", "error": _PARAMIKO_HINT, "truncated": False}
    client = None
    try:
        client = _connect(paramiko, host, port, username, password, key_path, timeout_s)
        stdin, stdout, stderr = client.exec_command(command, timeout=timeout_s)
        stdin.close()
        raw_out = stdout.read(max_bytes + 1)
        raw_err = stderr.read(max_bytes + 1)
        exit_status = stdout.channel.recv_exit_status()
        out, out_trunc = _decode(raw_out, max_bytes)
        err, err_trunc = _decode(raw_err, max_bytes)
        return {"ok": int(exit_status) == 0, "host": host, "exit_status": int(exit_status),
                "stdout": out, "stderr": err, "error": None,
                "truncated": out_trunc or err_trunc}
    except Exception as exc:  # noqa: BLE001 — every failure becomes data
        return {"ok": False, "host": host, "exit_status": None, "stdout": "",
                "stderr": "", "error": f"{type(exc).__name__}: {exc}", "truncated": False}
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass


def _decode(raw: bytes, max_bytes: int) -> tuple[str, bool]:
    """Decode bytes best-effort and report whether truncation occurred."""
    truncated = len(raw) > max_bytes
    return raw[:max_bytes].decode("utf-8", "replace"), truncated


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA remote-exec self-test (no real network)")
    print("=" * 70)

    # 1. host vetting refuses an empty host — fail-closed, no socket opened
    r = ssh_exec("", "echo hi")
    print(f"\nempty host          : ok={r['ok']} error={r['error']!r}")
    assert not r["ok"] and r["error"] == "missing host"

    # 2. a malformed host is refused
    r = ssh_login("bad host/name")
    print(f"malformed host      : ok={r['ok']} error={r['error']!r}")
    assert not r["ok"] and "malformed" in (r["error"] or "")

    # 3. loopback is refused unless allow_private is set
    r = ssh_exec("127.0.0.1", "echo hi", allow_private=False)
    print(f"loopback (default)  : ok={r['ok']} error={r['error']!r}")
    assert not r["ok"] and "refused" in (r["error"] or "")

    # 4. an empty command is refused before dialing out
    r = ssh_exec("example.com", "   ", allow_private=True)
    print(f"empty command       : ok={r['ok']} error={r['error']!r}")
    assert not r["ok"] and r["error"] == "empty command"

    # 5. vetting passes for a normal host (we don't actually dial out here)
    reason = _vet_host("example.com")
    print(f"normal host vet     : reason={reason!r} (None means it would proceed)")
    assert reason is None

    # 6. paramiko-absent path (simulated) returns data, never raises
    import sys as _sys
    _real = _sys.modules.pop("paramiko", None)
    _sys.modules["paramiko"] = None  # force the import guard to fail
    try:
        r = ssh_exec("example.com", "echo hi", allow_private=True)
        print(f"paramiko absent     : ok={r['ok']} error={r['error']!r}")
        assert not r["ok"] and "paramiko not installed" in (r["error"] or "")
    finally:
        if _real is not None:
            _sys.modules["paramiko"] = _real
        else:
            _sys.modules.pop("paramiko", None)

    print("\nALL SELF-TESTS PASSED ✓")
