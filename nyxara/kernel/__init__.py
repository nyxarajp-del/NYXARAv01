"""Nyxara kernel — config, core loop, global workspace, continuous cognition.

The kernel is sovereign. It owns the loop, selects faculties, and disposes of the
proposals they emit. Nothing in higher layers acts on its own; it all flows through
the orchestrator here.
"""
from __future__ import annotations

from nyxara.kernel.config import Settings, get_settings

__all__ = ["Settings", "get_settings"]
