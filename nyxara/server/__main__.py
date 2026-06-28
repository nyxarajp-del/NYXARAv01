"""``python -m nyxara.server`` — serve NYXARA over HTTP/WebSocket.

Honours every ``NYXARA_SERVER__*`` setting (host, port, token, CORS). On a keyless machine
NYXARA still answers through her deterministic reasoner, so the server is usable out of the
box; set a provider's API key for LLM-backed, deliberate replies, and a token to lock it.
"""

from __future__ import annotations

import sys

from nyxara.kernel.config import get_settings


def main(argv: list[str] | None = None) -> int:
    # First run, zero-setup: install the heavy LoRA stack (torch + transformers + peft) if
    # missing — same auto-bootstrap as the console front door, so a served NYXARA also gets her
    # own forged brain. Best-effort; on a successful first install it restarts the process so
    # the new stack is loaded from import time.
    try:
        from nyxara.growth.autodeps import ensure_runtime_deps
        ensure_runtime_deps(log=print)
    except Exception as exc:  # noqa: BLE001 — a failed auto-install must never block serving
        print(f"brain stack       : skipped ({exc})")

    # Primary brain — forge her OWN Qwen3-4B LoRA on the very first boot when `self` is the
    # provider (the default). Already-forged → instant; no stack → the n-gram brain; any
    # failure → logged, never fatal.
    try:
        from nyxara.growth.bootstrap import ensure_primary_model
        ensure_primary_model(log=print)
    except Exception as exc:  # noqa: BLE001 — a failed forge must never block serving
        print(f"primary brain      : skipped ({exc})")

    settings = get_settings()
    try:
        from nyxara.server.app import run
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    cfg = settings.server
    locked = "token-locked" if cfg.api_token else "OPEN (no token set)"
    print(f"NYXARA serving on http://{cfg.host}:{cfg.port}  [{settings.profile.value}, {locked}]")
    print("  POST /v1/chat   GET /v1/report   WS /v1/ws   GET /health")
    run(settings=settings)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
