"""NYXARA — sovereign cognitive architecture. Owner: Jaypal Khoja (JP).

The control law: *the mind proposes; the kernel disposes; the Master is sovereign.*

The whole mind is wired in :class:`nyxara.kernel.orchestrator.NyxaraCore`. The
typical entry point is::

    from nyxara import NyxaraCore, Authority

    core = NyxaraCore()
    result = core.process("Hello NYXARA", authority=Authority.OWNER)
    print(result.response)

Or run the interactive console with ``python -m nyxara`` (see
:mod:`nyxara.__main__`).
"""

from __future__ import annotations

from nyxara.agency.permissions import Authority
from nyxara.kernel.orchestrator import (
    Candidate,
    CycleResult,
    Disposition,
    NyxaraCore,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "NyxaraCore",
    "Authority",
    "Disposition",
    "Candidate",
    "CycleResult",
]
