"""NYXARA · growth/mdl_refactor.py — anti-fragile refactor: minimise a description-length proxy (🧬, E).

After NYXARA heals a bug, the fix should leave her code *leaner*, not messier. This module scores a
module by a **Minimum-Description-Length (MDL) proxy** and accepts a candidate refactor only when it
**strictly lowers** that proxy *and* the refactor is behaviour-preserving (the caller's verify-or-rollback
gauntlet says so). Otherwise it is discarded and the original is kept byte-for-byte.

The proxy — honest by construction — is::

    DL(source) ≈ len(zlib.compress(source)) + α · (number of AST nodes)

True Kolmogorov complexity is uncomputable; a compression-plus-AST-node count is the standard practical
stand-in (compression captures redundancy; the AST-node term captures structural bulk that compression
alone under-counts). This module owns only the *measurement + accept/reject decision*; enactment (writing
the edit and running the gauntlet) stays in :mod:`nyxara.growth.self_optimize`, whose reversible
:class:`~nyxara.growth.self_optimize.SourceEdit` type is reused here. Pure standard library; no LLM; it
touches no gate or character value.
"""

from __future__ import annotations

import ast
import zlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

__all__ = ["description_length", "MDLReport", "MDLRefactor"]

# Weight on the AST-node term (bytes-equivalent per node). Small: compression carries most of the
# signal; the node term only breaks ties toward genuinely simpler structure, not mere shorter text.
_DEFAULT_ALPHA = 2.0


def _ast_node_count(source: str) -> Optional[int]:
    """Number of nodes in the module AST, or None if the source does not parse."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    return sum(1 for _ in ast.walk(tree))


def description_length(source: str, *, alpha: float = _DEFAULT_ALPHA) -> float:
    """The MDL proxy for a Python source string.

    Non-parsing source is charged the compressed-size term only (its AST term is undefined); this keeps
    the function total, and a refactor is never *accepted* on non-parsing ``after`` anyway (see
    :meth:`MDLRefactor.accept`)."""
    comp = len(zlib.compress(source.encode("utf-8"), 9))
    nodes = _ast_node_count(source)
    return float(comp) + (alpha * nodes if nodes is not None else 0.0)


@dataclass
class MDLReport:
    """One before/after description-length comparison."""

    before: float
    after: float
    improved: bool
    accepted: bool = False
    reason: str = ""

    @property
    def delta(self) -> float:
        """Negative when the refactor made the module simpler (the goal)."""
        return self.after - self.before

    def to_dict(self) -> Dict[str, Any]:
        return {"before": round(self.before, 2), "after": round(self.after, 2),
                "delta": round(self.delta, 2), "improved": self.improved,
                "accepted": self.accepted, "reason": self.reason}


class MDLRefactor:
    """Accept a refactor only if it strictly lowers the MDL proxy *and* preserves behaviour."""

    def __init__(self, *, alpha: float = _DEFAULT_ALPHA) -> None:
        self.alpha = float(alpha)

    def measure(self, source: str) -> float:
        return description_length(source, alpha=self.alpha)

    def compare(self, before: str, after: str) -> MDLReport:
        db = self.measure(before)
        da = self.measure(after)
        return MDLReport(before=db, after=da, improved=da < db)

    def accept(self, before: str, after: str, *, gauntlet_ok: bool) -> MDLReport:
        """Decide whether to keep ``after``. Kept iff it parses, strictly lowers DL, changes something,
        and the caller's behaviour-equivalence gauntlet passed. Any miss ⇒ reject (keep ``before``)."""
        report = self.compare(before, after)
        if after == before:
            report.reason = "no change"
            return report
        if _ast_node_count(after) is None:
            report.improved = False
            report.reason = "candidate does not parse — rejected"
            return report
        if not report.improved:
            report.reason = f"description length did not decrease (Δ={report.delta:+.0f})"
            return report
        if not gauntlet_ok:
            report.reason = "not behaviour-preserving (gauntlet failed) — rejected"
            return report
        report.accepted = True
        report.reason = f"accepted — leaner by {(-report.delta):.0f} (behaviour preserved)"
        return report

    def refactor(self, before: str, candidate: str,
                 verify: Callable[[str, str], bool]) -> MDLReport:
        """Convenience: run ``verify(before, candidate)`` (the behaviour-equivalence gauntlet) and apply
        the accept rule. ``verify`` is only invoked when the candidate is already an MDL improvement, so
        the expensive gauntlet is never spent on a candidate that could not be kept anyway."""
        pre = self.compare(before, candidate)
        if not pre.improved or candidate == before or _ast_node_count(candidate) is None:
            return self.accept(before, candidate, gauntlet_ok=False)
        ok = False
        try:
            ok = bool(verify(before, candidate))
        except Exception:  # noqa: BLE001 — a throwing gauntlet is a failed verification, never a crash
            ok = False
        return self.accept(before, candidate, gauntlet_ok=ok)
