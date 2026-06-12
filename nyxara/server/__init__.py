"""NYXARA · server — the authenticated HTTP/WebSocket transport over the sovereign loop.

This package lets NYXARA be reached from an app, a phone, or the web instead of only the
local console. It is deliberately thin: every request is carried through the *same*
``NyxaraCore.process`` cycle and every gate (corrigibility, honesty, permissions, the
guardian, oversight). The network is just another mouth — never a path around the control
law. The web layer (FastAPI/uvicorn) is an optional dependency; import it via the
``[server]`` extra. ``create_app`` raises a clear, actionable error if it is missing.
"""

from __future__ import annotations

from typing import Any

__all__ = ["create_app", "run"]


def create_app(core: Any = None, *, settings: Any = None) -> Any:
    """Build and return the FastAPI application (lazy import so the dep stays optional)."""
    from nyxara.server.app import create_app as _factory
    return _factory(core=core, settings=settings)


def run(core: Any = None, *, settings: Any = None, **uvicorn_kwargs: Any) -> None:
    """Serve NYXARA over HTTP/WebSocket with uvicorn (blocking)."""
    from nyxara.server.app import run as _run
    _run(core=core, settings=settings, **uvicorn_kwargs)
