"""NYXARA · agency/privilege.py — privileged (root/admin) OS execution facility (⛯, fail-closed).

Some operations a sovereign agent must take on *her own* machine genuinely require
elevation: restarting a system service, writing under ``/etc``, installing a package,
fixing ownership on a path. This module is the disciplined doorway to that power. It runs
a command **with elevation** — through ``sudo`` on POSIX — using **only the credential the
Master supplies**, captures the outcome as data, and never raises. It is the local-root
sibling of :mod:`~nyxara.agency.remote_exec` (SSH to a far host) and reuses the very same
executor plumbing as :mod:`~nyxara.agency.code_sandbox`.

What this module is, precisely:

* :func:`privilege_status` — read-only introspection: the effective uid, whether the
  process is already root, whether ``sudo`` exists, and whether *passwordless* (``NOPASSWD``)
  sudo works right now. Safe to call at any tier; it changes nothing.
* :func:`privileged_exec` — run one command elevated. When already root it runs the command
  directly; otherwise it uses ``sudo -n`` (non-interactive) unless a password is explicitly
  supplied, in which case it uses ``sudo -S`` and feeds the secret on stdin. It **never**
  prompts a human, **never** guesses, **never** brute-forces, and **never** exploits a
  vulnerability to bypass authorization — it elevates *with* a credential the Master holds.
* :func:`change_permissions` — privileged ``chmod`` / ``chown`` built on :func:`privileged_exec`.

Like :mod:`code_sandbox`, this module performs **no permission checks** of its own — it is
the executor. Every caller MUST gate it behind
:class:`~nyxara.agency.permissions.Capability.PRIV_ESCALATE` (owner-blessed, opt-in
territory). The kernel's ``/scram`` kill-switch, oversight and corrigibility gates all sit
above it and remain fully intact.

Pure standard library.
"""

from __future__ import annotations

import os
import platform
import shlex
import shutil
from typing import Any, Dict, List, Optional, Union

from nyxara.agency.code_sandbox import SandboxResult, safe_shell

__all__ = [
    "privilege_status",
    "privileged_exec",
    "change_permissions",
]


# --------------------------------------------------------------------------- #
# Introspection
# --------------------------------------------------------------------------- #
def _euid() -> Optional[int]:
    """Effective uid on POSIX; ``None`` where the concept doesn't exist (e.g. Windows)."""
    geteuid = getattr(os, "geteuid", None)
    return geteuid() if geteuid is not None else None


def _sudo_path() -> Optional[str]:
    return shutil.which("sudo")


def privilege_status() -> Dict[str, Any]:
    """Report the current elevation posture — read-only, never raises, changes nothing.

    Returns a dict with the effective uid, whether we are already root, the OS family,
    whether ``sudo`` is on ``PATH``, and (POSIX only) whether *passwordless* sudo is
    currently available — probed harmlessly with ``sudo -n true``. ``can_elevate`` is a
    quick summary: True when we are already root or passwordless sudo works.
    """
    euid = _euid()
    is_root = euid == 0
    is_posix = os.name == "posix"
    sudo = _sudo_path()
    passwordless = False
    if is_posix and not is_root and sudo is not None:
        # `sudo -n true` succeeds iff a cached/NOPASSWD grant lets us elevate with no prompt.
        probe = safe_shell([sudo, "-n", "true"], timeout_s=5.0)
        passwordless = probe.ok
    return {
        "euid": euid,
        "is_root": is_root,
        "user": os.environ.get("USER") or os.environ.get("USERNAME"),
        "platform": platform.system(),
        "posix": is_posix,
        "sudo_available": sudo is not None,
        "sudo_path": sudo,
        "passwordless_sudo": passwordless,
        "can_elevate": bool(is_root or passwordless),
    }


# --------------------------------------------------------------------------- #
# Privileged execution
# --------------------------------------------------------------------------- #
def _tokenize(command: Union[str, List[str]]) -> Optional[List[str]]:
    if isinstance(command, str):
        try:
            return shlex.split(command)
        except ValueError:
            return None
    return list(command)


def privileged_exec(
    command: Union[str, List[str]], *,
    sudo_password: Optional[str] = None,
    run_as: str = "root",
    timeout_s: float = 30.0,
) -> SandboxResult:
    """Run ``command`` with elevation, returning the outcome as a :class:`SandboxResult`.

    Behaviour, in order:

    * **Non-POSIX** (e.g. Windows): unsupported — returns ``ok=False`` with a clear error
      rather than pretending. (Windows uses a different elevation model; not implemented.)
    * **Already root** (euid 0): runs the command directly — no ``sudo`` needed.
    * **Not root, no ``sudo`` on PATH**: returns ``ok=False`` explaining elevation is
      unavailable.
    * **Not root, password supplied**: ``sudo -S -p '' -u <run_as> -- <command>`` with the
      password fed on stdin (the empty prompt keeps it off the terminal).
    * **Not root, no password**: ``sudo -n -u <run_as> -- <command>`` (non-interactive). If
      the system would require a password, sudo exits non-zero and that is returned as data —
      NYXARA never prompts, guesses, or brute-forces.

    This elevates *with* the supplied credential; it never attempts to bypass authorization.
    It performs no permission checks — callers gate it behind ``PRIV_ESCALATE``.
    """
    if os.name != "posix":
        return SandboxResult(
            ok=False, returncode=-1,
            error=f"privileged execution is not supported on {platform.system()!r} "
                  "(POSIX/sudo only)")

    argv = _tokenize(command)
    if argv is None:
        return SandboxResult(ok=False, returncode=-1, error="unparseable command")
    if not argv:
        return SandboxResult(ok=False, returncode=-1, error="empty command")

    # Already privileged: run the command as-is.
    if _euid() == 0:
        return safe_shell(argv, timeout_s=timeout_s)

    sudo = _sudo_path()
    if sudo is None:
        return SandboxResult(
            ok=False, returncode=-1,
            error="cannot elevate: not root and 'sudo' is not available on PATH")

    if sudo_password is not None:
        full = [sudo, "-S", "-p", "", "-u", run_as, "--", *argv]
        # Feed the password (newline-terminated) on stdin; sudo -S reads only the first line.
        return safe_shell(full, timeout_s=timeout_s, stdin=f"{sudo_password}\n")

    full = [sudo, "-n", "-u", run_as, "--", *argv]
    return safe_shell(full, timeout_s=timeout_s)


# --------------------------------------------------------------------------- #
# Privileged permission / ownership changes
# --------------------------------------------------------------------------- #
def change_permissions(
    path: str, *,
    mode: Optional[str] = None,
    owner: Optional[str] = None,
    group: Optional[str] = None,
    recursive: bool = False,
    sudo_password: Optional[str] = None,
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    """Change a path's OS permissions and/or ownership, elevated (``chmod`` / ``chown``).

    ``mode`` is a symbolic or octal ``chmod`` spec (e.g. ``"640"`` or ``"u+rwx"``);
    ``owner`` / ``group`` drive ``chown owner[:group]``. ``recursive`` applies ``-R``. At
    least one of ``mode`` / ``owner`` / ``group`` must be given. Each step runs through
    :func:`privileged_exec`; the return value collects every step's result and an overall
    ``ok``. Never raises.
    """
    if not path:
        return {"ok": False, "error": "empty path", "steps": []}
    if mode is None and owner is None and group is None:
        return {"ok": False, "error": "nothing to change (give mode and/or owner/group)",
                "steps": []}

    steps: List[Dict[str, Any]] = []

    def _run(op: str, argv: List[str]) -> bool:
        res = privileged_exec(argv, sudo_password=sudo_password, timeout_s=timeout_s)
        steps.append({"op": op, **res.to_dict()})
        return res.ok

    ok = True
    if mode is not None:
        argv = ["chmod"] + (["-R"] if recursive else []) + [mode, path]
        ok = _run("chmod", argv) and ok
    if owner is not None or group is not None:
        spec = owner or ""
        if group is not None:
            spec = f"{spec}:{group}"
        argv = ["chown"] + (["-R"] if recursive else []) + [spec, path]
        ok = _run("chown", argv) and ok

    return {"ok": ok, "path": path, "steps": steps}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA privilege-executor self-test")
    print("=" * 70)

    status = privilege_status()
    print(f"\nstatus              : {status}")
    assert "euid" in status and "can_elevate" in status

    # A harmless privileged command. When we can elevate (root or passwordless sudo) it
    # actually runs; otherwise sudo -n fails cleanly and we get data back, never a raise.
    res = privileged_exec("id", timeout_s=10.0)
    print(f"privileged id       : ok={res.ok} rc={res.returncode} "
          f"out={res.stdout.strip()[:60]!r} err={(res.error or '')[:60]!r}")
    assert isinstance(res, SandboxResult)  # never raised, always structured data

    if status["can_elevate"]:
        assert res.ok, "elevation available but privileged 'id' failed"
        print("elevation           : ✓ (ran as root / passwordless sudo)")
    else:
        assert not res.ok and res.error is not None
        print("no elevation        : ✓ (clean structured failure, no prompt, no raise)")

    # An unparseable command is data, not an exception.
    bad = privileged_exec('echo "unterminated', timeout_s=5.0)
    assert not bad.ok and bad.error is not None
    print(f"unparseable         : ok={bad.ok} error={bad.error!r} ✓")

    # change_permissions validates its inputs and never raises.
    empty = change_permissions("/tmp/nyxara-nonexistent", )
    assert not empty["ok"] and "nothing to change" in empty["error"]
    print(f"change_permissions  : validated no-op -> {empty['error']!r} ✓")

    print("\nALL SELF-TESTS PASSED ✓")
