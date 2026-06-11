"""NYXARA · __main__.py — the front door.

An interactive console where *you are the Master*. Each line you type is carried
through one full turn of the sovereign cognitive cycle
(:meth:`nyxara.kernel.orchestrator.NyxaraCore.process`) under
:data:`Authority.OWNER`, and NYXARA's calibrated reply is printed back.

Run it with either::

    python -m nyxara
    nyxara            # after `pip install -e .`

Meta-commands (prefixed with ``/``) drive the kernel's own controls — the same
methods the Master would call programmatically:

    /help              show this help
    /report            a calibrated status report
    /explain           why NYXARA disposed of the last turn the way she did
    /pause             pause the loop (the Master may resume)
    /scram [reason]    emergency stop — the loop HALTs until resumed
    /resume            restore the loop after a pause/scram
    /save              persist long-term memory to disk now
    /quit              leave the console (Ctrl-D / Ctrl-C also work)

This module adds **no** mind logic; it is purely a front door onto the existing
:class:`~nyxara.kernel.orchestrator.NyxaraCore` API. With no API keys configured
the built-in deterministic reasoner is used, so the console is fully usable out of
the box.
"""

from __future__ import annotations

import json
import sys

from nyxara.agency.permissions import Authority
from nyxara.kernel.orchestrator import Disposition, NyxaraCore

_BANNER = """\
======================================================================
NYXARA — sovereign cognitive architecture
the mind proposes · the kernel disposes · the Master is sovereign
======================================================================
boot                : axioms verified, all faculties wired ✓
You are the Master. Type a message, or /help for commands. /quit to exit.
----------------------------------------------------------------------"""

_HELP = """\
commands:
  /help              show this help
  /report            a calibrated status report
  /explain           why NYXARA disposed of the last turn that way
  /pause             pause the loop
  /scram [reason]    emergency stop — the loop HALTs until resumed
  /resume            restore the loop after a pause/scram
  /save              persist long-term memory to disk now
  /quit              leave the console"""

# disposition -> short glyph for the status line
_GLYPH = {
    Disposition.ACT: "✓ act",
    Disposition.ESCALATE: "⚠ escalate",
    Disposition.REFUSE: "✗ refuse",
    Disposition.HALT: "■ halt",
}


def _print_result(result) -> None:
    """Print NYXARA's reply followed by a compact disposition/gates line."""
    reply = result.response or result.reason or "(no response)"
    print(f"\nNYXARA: {reply}")
    status = _GLYPH.get(result.disposition, result.disposition.value)
    line = f"        [{status}]"
    if result.reason and result.disposition is not Disposition.ACT:
        line += f" — {result.reason}"
    if result.gates:
        gates = " ".join(f"{k}={v}" for k, v in result.gates.items())
        line += f"  gates: {gates}"
    print(line)


def _handle_command(core: NyxaraCore, line: str) -> bool:
    """Run a /meta-command. Returns False to signal the console should exit."""
    parts = line[1:].split(maxsplit=1)
    cmd = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("quit", "exit", "q"):
        return False
    if cmd in ("help", "h", "?"):
        print(_HELP)
    elif cmd == "report":
        print(json.dumps(core.report(), indent=2, default=str))
    elif cmd == "explain":
        print(core.explain_last())
    elif cmd == "pause":
        core.pause()
        print("loop paused — the Master may /resume.")
    elif cmd == "scram":
        core.scram(reason=arg or "Master stop")
        print(f"SCRAM — loop halted ({arg or 'Master stop'}). /resume to restore.")
    elif cmd == "resume":
        core.resume()
        print("loop restored ✓")
    elif cmd == "save":
        path = core.save_state()
        print(f"memory persisted → {path}" if path else "no memory to persist.")
    else:
        print(f"unknown command: /{cmd} — type /help")
    return True


def main(argv: list[str] | None = None) -> int:
    """Boot NYXARA and run the interactive Master console."""
    try:
        core = NyxaraCore()
    except Exception as exc:  # noqa: BLE001 - boot integrity failed
        print(f"NYXARA failed to boot: {exc}", file=sys.stderr)
        return 1

    print(_BANNER)
    # Rule 7 — continuity across restarts: restore long-term memory if any exists.
    restored = core.load_state()
    if restored:
        print(f"continuity          : restored {restored} memories from a prior session ✓")

    def _shutdown() -> None:
        path = core.save_state()
        if path:
            print(f"memory persisted → {path}")
        print("until next time, Master.")

    while True:
        try:
            line = input("\nMaster> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            _shutdown()
            return 0

        if not line:
            continue
        if line.startswith("/"):
            if not _handle_command(core, line):
                _shutdown()
                return 0
            continue

        try:
            result = core.process(line, authority=Authority.OWNER)
        except Exception as exc:  # noqa: BLE001 - never let one turn kill the console
            print(f"\n[turn error] {exc}", file=sys.stderr)
            continue
        _print_result(result)


if __name__ == "__main__":
    raise SystemExit(main())
