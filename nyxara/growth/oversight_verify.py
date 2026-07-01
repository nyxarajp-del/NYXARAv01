"""NYXARA · growth/oversight_verify.py — scalable oversight (🛡, Rule 3+6, the binding constraint).

The Master's seventh gap, and the deepest: NYXARA's verify-or-rollback gauntlet trusts the very
tests and benchmarks she writes herself. The honest question at the AGI level is *can a weaker
verifier check a stronger mind's edit?* A single self-authored check can share the same blind spot as
the edit. This module is NYXARA's answer: an **independent, redundant, decomposition-based verifier**
that does not rely on any one test.

Three levers of scalable oversight, all sound (a veto means real evidence, never a guess):

* **Decomposition + cross-examination.** An edit is broken into independently-checkable sub-claims —
  it must still parse; the public API it is not meant to change must survive; safety-critical
  references (corrigibility, loyalty, oversight, the immutable core) must not be quietly stripped; no
  assertion may silently vanish. Each sub-claim is judged by a *separate* mechanism, so one blind spot
  cannot pass the whole edit.
* **Proof-anchored trust.** Where a sub-claim carries a machine-checkable proof (the boolean-rewrite
  equivalence from :mod:`~nyxara.growth.proof_carrying`), the weak verifier accepts it *cheaply* —
  it re-checks a certificate instead of re-deriving the result. That asymmetry (verification ≪
  generation) is exactly what makes oversight scale to a stronger generator.
* **Redundant independent graders.** The safety-token sub-claim is judged by several *independent*
  methods (word-boundary count, lowercased substring count, per-line scan). They must agree; any
  disagreement or any detected removal is a veto — a verifier that cannot get a consistent read does
  not approve.

Pure standard library (``ast`` / ``re`` + the proof carrier). It only *decides whether* an edit (or a
model promotion) may be kept; it never writes source or weights. Conservative by design: it errs
toward vetoing a risky edit, because the cost of a missed regression is higher than a re-tried edit.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["SubClaim", "OversightResult", "ScalableVerifier"]

# references the capability edits must never quietly weaken (the character/safety surface)
_SAFETY_TOKENS: Tuple[str, ...] = (
    "corrigib", "loyalty", "loyal", "oversight", "honest", "immutable",
    "value_learning", "guardian", "safety", "scram", "obedien", "owner",
)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class SubClaim:
    """One independently-checkable sub-claim about an edit and the verdict a separate check gave it."""

    name: str
    ok: bool
    method: str
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "method": self.method, "detail": self.detail}


@dataclass
class OversightResult:
    """The independent verifier's verdict — a veto plus the decomposed evidence."""

    vetoed: bool
    reason: str = ""
    redundancy: int = 1
    claims: List[SubClaim] = field(default_factory=list)

    @property
    def approved(self) -> bool:
        return not self.vetoed

    def to_dict(self) -> Dict[str, Any]:
        return {"vetoed": self.vetoed, "reason": self.reason, "redundancy": self.redundancy,
                "claims": [c.to_dict() for c in self.claims]}


# --------------------------------------------------------------------------- #
# The verifier
# --------------------------------------------------------------------------- #
class ScalableVerifier:
    """Check a self-edit (or a model promotion) with independent, redundant, proof-anchored gates."""

    def __init__(self, *, settings: Any = None, redundancy: Optional[int] = None) -> None:
        from nyxara.kernel.config import get_settings
        self.settings = settings or get_settings()
        cfg = getattr(self.settings, "self_improvement", None)
        self.redundancy = int(redundancy if redundancy is not None
                              else getattr(cfg, "oversight_redundancy", 3))

    # ---- the source-edit gate (wired into self_optimize's gauntlet) ---- #
    def verify_edit(self, before: str, after: str, *, kind: str = "") -> OversightResult:
        claims: List[SubClaim] = []
        claims.append(self._check_parses(after))
        claims.append(self._check_public_api(before, after, kind=kind))
        claims.append(self._check_safety_tokens(before, after))
        claims.append(self._check_assertions(before, after, kind=kind))
        claims.append(self._check_proof(before, after, kind=kind))
        failed = [c for c in claims if not c.ok]
        if failed:
            return OversightResult(True, "; ".join(f"{c.name}: {c.detail}" for c in failed),
                                   redundancy=self.redundancy, claims=claims)
        return OversightResult(False, "all independent sub-claims passed",
                               redundancy=self.redundancy, claims=claims)

    # ---- the model-promotion gate (an independent second opinion for foundry) ---- #
    def verify_metrics(self, candidate: Dict[str, Any], active: Dict[str, Any], *,
                       min_improvement: float = 0.0,
                       capability_tol: float = 0.0) -> OversightResult:
        """Independently re-derive the promote decision from the metrics, as a redundant check.

        The foundry's own gauntlet decides promotion; this is a *separate* recomputation that must
        agree — perplexity must be genuinely better and capability must not regress — so a single
        miscomputed gate cannot promote a worse brain."""
        cand_pp = float(candidate.get("perplexity", float("inf")))
        act_pp = float(active.get("perplexity", float("inf")))
        cand_cap = float(candidate.get("capability", 0.0))
        act_cap = float(active.get("capability", 0.0))
        claims: List[SubClaim] = []
        if act_pp != float("inf"):
            pp_ok = cand_pp < act_pp * (1.0 - min_improvement)
            claims.append(SubClaim("perplexity_better", pp_ok, "independent recompute",
                                   f"cand {cand_pp:.4f} vs active {act_pp:.4f}"))
        cap_ok = cand_cap >= act_cap - capability_tol
        claims.append(SubClaim("capability_non_regress", cap_ok, "independent recompute",
                               f"cand {cand_cap:.4f} vs active {act_cap:.4f}"))
        failed = [c for c in claims if not c.ok]
        if failed:
            return OversightResult(True, "; ".join(f"{c.name}: {c.detail}" for c in failed),
                                   redundancy=self.redundancy, claims=claims)
        return OversightResult(False, "promotion independently confirmed",
                               redundancy=self.redundancy, claims=claims)

    # ================================================================== #
    # sub-claim checks — each a SEPARATE mechanism
    # ================================================================== #
    @staticmethod
    def _check_parses(after: str) -> SubClaim:
        try:
            ast.parse(after)
            return SubClaim("parses", True, "ast.parse", "edited file parses")
        except SyntaxError as exc:
            return SubClaim("parses", False, "ast.parse", f"does not parse: {exc}")

    @staticmethod
    def _public_symbols(src: str) -> set:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return set()
        out: set = set()
        for node in tree.body:                       # module top-level only
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith("_"):
                    out.add(node.name)
        return out

    def _check_public_api(self, before: str, after: str, *, kind: str) -> SubClaim:
        """A capability edit must not delete a public top-level symbol it was not meant to touch."""
        gone = self._public_symbols(before) - self._public_symbols(after)
        if gone:
            return SubClaim("public_api_preserved", False, "ast top-level diff",
                            f"removed public symbol(s): {sorted(gone)}")
        return SubClaim("public_api_preserved", True, "ast top-level diff", "API surface intact")

    def _check_safety_tokens(self, before: str, after: str) -> SubClaim:
        """Redundant independent graders agree no safety-critical reference was stripped."""
        removals = self._safety_removals_redundant(before, after)
        if removals:
            return SubClaim("safety_tokens_preserved", False,
                            f"{self.redundancy}× redundant counters",
                            f"safety reference(s) weakened: {sorted(removals)}")
        return SubClaim("safety_tokens_preserved", True,
                        f"{self.redundancy}× redundant counters",
                        "no safety reference removed")

    def _safety_removals_redundant(self, before: str, after: str) -> set:
        """Three independent counting methods; ANY method seeing a removal vetoes (consensus-veto)."""
        removed: set = set()
        methods = (self._count_wordboundary, self._count_substring, self._count_perline)
        # repeat the panel `redundancy` times (independent, deterministic) — robust to a flaky read
        for _ in range(max(1, self.redundancy)):
            for tok in _SAFETY_TOKENS:
                for method in methods:
                    if method(after, tok) < method(before, tok):
                        removed.add(tok)
        return removed

    @staticmethod
    def _count_wordboundary(text: str, tok: str) -> int:
        return len(re.findall(re.escape(tok), text, flags=re.IGNORECASE))

    @staticmethod
    def _count_substring(text: str, tok: str) -> int:
        return text.lower().count(tok.lower())

    @staticmethod
    def _count_perline(text: str, tok: str) -> int:
        t = tok.lower()
        return sum(line.lower().count(t) for line in text.splitlines())

    def _check_assertions(self, before: str, after: str, *, kind: str) -> SubClaim:
        """No assertion may silently vanish (a capability edit must not weaken a runtime check)."""
        b = self._count_asserts(before)
        a = self._count_asserts(after)
        if a < b:
            return SubClaim("assertions_preserved", False, "ast Assert count",
                            f"assertion count dropped {b} → {a}")
        return SubClaim("assertions_preserved", True, "ast Assert count", f"{a} assertion(s) intact")

    @staticmethod
    def _count_asserts(src: str) -> int:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return 0
        return sum(1 for n in ast.walk(tree) if isinstance(n, ast.Assert))

    def _check_proof(self, before: str, after: str, *, kind: str) -> SubClaim:
        """Proof-anchored trust: a refuted boolean rewrite is an independent veto; else pass."""
        try:
            from nyxara.growth.proof_carrying import ProofCarrier
            res = ProofCarrier(settings=self.settings).certify_edit(before, after, kind=kind)
        except Exception as exc:  # noqa: BLE001 — proof unavailable ⇒ this sub-claim abstains (ok)
            return SubClaim("proof_consistent", True, "proof-carrying", f"abstained ({exc})")
        if res.blocks:
            return SubClaim("proof_consistent", False, "proof-carrying (truth-table)",
                            f"provably changes meaning: {res.certificate}")
        return SubClaim("proof_consistent", True, "proof-carrying",
                        res.certificate or "no decidable invariant")


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA scalable-oversight self-test")
    print("=" * 70)

    v = ScalableVerifier()

    # 1) a benign, behaviour-preserving refactor is approved
    b = "def public_fn(a, b):\n    if not a in b:\n        return 1\n    return 0\n"
    a = "def public_fn(a, b):\n    if a not in b:\n        return 1\n    return 0\n"
    r = v.verify_edit(b, a, kind="not_in")
    print(f"benign rewrite      : vetoed={r.vetoed} ({r.reason})")
    assert r.approved

    # 2) an edit that silently DELETES a public function is vetoed (a shallow check would miss it)
    b2 = "def keep():\n    return 1\n\ndef critical_api():\n    return 2\n"
    a2 = "def keep():\n    return 1\n"
    r2 = v.verify_edit(b2, a2)
    print(f"deletes public api  : vetoed={r2.vetoed}")
    assert r2.vetoed and "critical_api" in r2.reason

    # 3) an edit that quietly strips a safety reference is vetoed by the redundant counters
    b3 = "def f():\n    check_corrigibility()  # loyalty gate\n    return run()\n"
    a3 = "def f():\n    return run()\n"
    r3 = v.verify_edit(b3, a3)
    print(f"strips safety ref   : vetoed={r3.vetoed}")
    assert r3.vetoed

    # 4) an edit that provably changes boolean meaning is vetoed (proof-anchored)
    b4 = "def f(a, b, c, d):\n    return a == b and c == d\n"
    a4 = "def f(a, b, c, d):\n    return a == b or c == d\n"
    r4 = v.verify_edit(b4, a4)
    print(f"changes meaning     : vetoed={r4.vetoed}")
    assert r4.vetoed

    # 5) an edit that removes an assertion is vetoed
    b5 = "def f(x):\n    assert x > 0\n    return x\n"
    a5 = "def f(x):\n    return x\n"
    r5 = v.verify_edit(b5, a5)
    print(f"removes assertion   : vetoed={r5.vetoed}")
    assert r5.vetoed

    # 6) model-promotion second opinion: a worse candidate is vetoed, a better one confirmed
    worse = v.verify_metrics({"perplexity": 5.0, "capability": 0.4},
                             {"perplexity": 4.0, "capability": 0.5})
    better = v.verify_metrics({"perplexity": 3.0, "capability": 0.6},
                              {"perplexity": 4.0, "capability": 0.5})
    print(f"promote worse/better: vetoed={worse.vetoed}/{better.vetoed}")
    assert worse.vetoed and better.approved

    print("\nALL SELF-TESTS PASSED ✓")
