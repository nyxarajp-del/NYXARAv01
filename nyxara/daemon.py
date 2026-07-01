"""``nyxara-daemon`` — the always-on entry point (HTTP server + background mind).

This is the process the systemd unit and the Windows scheduled task launch. It is a
thin wrapper over the existing server main (:func:`nyxara.server.__main__.main`) that
simply switches the **background mind on by default**: it forces
``NYXARA_SERVER__AUTONOMIC=true`` (unless the Master has already set it explicitly), so
one long-running process is both a reachable API *and* NYXARA's continuous,
self-directed :class:`~nyxara.kernel.autonomic.AutonomicLoop` — every autonomic turn
still passing the identical sovereign gates.

Run it directly the same way a service manager does::

    nyxara-daemon
    # equivalent to:
    NYXARA_SERVER__AUTONOMIC=true nyxara-serve

Every ``NYXARA_SERVER__*`` setting (host, port, token, CORS, autonomic cadence) is still
honoured; see ``docs/persistence.md`` for installing it as an always-alive service.
"""

from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> int:
    # Default the background mind ON for the daemon; respect an explicit override.
    os.environ.setdefault("NYXARA_SERVER__AUTONOMIC", "true")
    from nyxara.server.__main__ import main as serve_main
    return serve_main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
