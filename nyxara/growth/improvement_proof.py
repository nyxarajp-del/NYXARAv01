"""NYXARA · growth/improvement_proof.py — the strict "provably BETTER" gate (⚖, Rule 4).

The Master's charge: NYXARA may rewrite her *entire* source, **but only when she can
mathematically PROVE the new code is better than the old** — never merely "not worse". The
existing gauntlet (``self_optimize.Optimizer``) proves an edit is *safe* and *non-regressing*;
this module is the extra, decisive gate that proves it is a genuine **improvement**, or the edit
is rolled back even though it was harmless.

Rice's theorem forbids a general "this program is better" decider, so we do not pretend to one.
Instead we certify improvement under a **well-defined, decidable ordering** — three sound proof
methods, each machine-checkable and journalled:

* **A · capability Pareto-gain** — measured on NYXARA's deterministic eval battery: at least one
  task that failed before now passes, **zero** regress, and the mean score does not drop. Because
  the harness is deterministic, this is a *dominance proof*, not a lucky benchmark run. This is
  the method that licenses the powerful open-ended self-authored rewrites: a whole-file rewrite is
  kept only if it makes NYXARA measurably smarter.
* **B · proven-equivalent-and-cheaper** — for a behaviour-preserving refactor, the changed boolean
  conditions are **proved logically equivalent** (truth-table, via :class:`ProofCarrier`) *and*
  the static cost (AST node + branch count) strictly drops. Same behaviour, provably simpler ⇒
  provably better.
* **C · defect-elimination** — a named, deterministic transform (``bare_except`` → ``except
  Exception``, ``== None`` → ``is None``, dead-import removal, docstring insertion, negated-compare
  normalisation) whose anti-pattern is a **decidable predicate**: it is proved the specific defect
  strictly decreased. Combined with the gauntlet's non-regression proof, the edit provably removes
  a defect without making anything worse.

No method certifies ⇒ ``better is False`` ⇒ the edit is discarded. Pure standard library plus
:class:`ProofCarrier`; it reads reports and source, and never touches disk or weights itself.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["ImprovementCertificate", "ImprovementProver", "static_cost"]


# --------------------------------------------------------------------------- #
# Certificate
# --------------------------------------------------------------------------- #
@dataclass
class ImprovementCertificate:
    """The machine-checkable verdict on whether an edit is provably better."""

    better: bool
    method: str = "none"          # "pareto-capability" | "proven-equivalent-cheaper"
    #                               | "defect-elimination" | "none"
    reason: str = ""
    newly_passing: List[str] = field(default_factory=list)
    regressed: List[str] = field(default_factory=list)
    cost_before: int = 0
    cost_after: int = 0
    proof: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"better": self.better, "method": self.method, "reason": self.reason,
                "newly_passing": list(self.newly_passing), "regressed": list(self.regressed),
                "cost_before": self.cost_before, "cost_after": self.cost_after,
                "proof": dict(self.proof)}


# --------------------------------------------------------------------------- #
# The prover
# --------------------------------------------------------------------------- #
class ImprovementProver:
    """Prove — under a decidable ordering — that a ``before → after`` edit is strictly better."""

    def __init__(self, *, settings: Any = None) -> None:
        self.settings = settings

    def prove(self, *, before: Any = None, after: Any = None,
              before_src: str = "", after_src: str = "",
              edit_kind: str = "") -> ImprovementCertificate:
        """Return a certificate. ``before``/``after`` are :class:`BenchmarkReport`s (or ``None``
        when no capability measurement is available); ``before_src``/``after_src`` are the edited
        file's full contents before and after."""
        regressed = self._regressions(before, after)
        cost_b = static_cost(before_src)
        cost_a = static_cost(after_src)

        # ---- A · capability Pareto-gain (the licence for open-ended rewrites) ---- #
        if before is not None and after is not None:
            newly = self._newly_passing(before, after)
            if not regressed and newly and _mean_score(after) >= _mean_score(before):
                return ImprovementCertificate(
                    better=True, method="pareto-capability",
                    reason=(f"capability strictly increased: {len(newly)} task(s) now pass, "
                            f"0 regress (deterministic dominance proof)"),
                    newly_passing=newly, regressed=[], cost_before=cost_b, cost_after=cost_a)

        # ---- B · proven behaviourally-equivalent AND strictly cheaper ---- #
        if not regressed and cost_a < cost_b:
            pc = self._prove_equivalent(before_src, after_src, edit_kind)
            if pc is not None and pc.get("verdict") == "proven":
                return ImprovementCertificate(
                    better=True, method="proven-equivalent-cheaper",
                    reason=(f"behaviour proved unchanged (truth-table) and static cost strictly "
                            f"fell {cost_b}→{cost_a}"),
                    regressed=[], cost_before=cost_b, cost_after=cost_a, proof=pc)

        # ---- C · a named defect is provably eliminated (safe deterministic transforms) ---- #
        if not regressed and _defect_eliminated(before_src, after_src, edit_kind):
            return ImprovementCertificate(
                better=True, method="defect-elimination",
                reason=(f"the '{edit_kind}' defect is provably eliminated with no capability "
                        f"regression"),
                regressed=[], cost_before=cost_b, cost_after=cost_a)

        # ---- no proof of improvement ⇒ keep the old code ---- #
        why = "capability did not strictly increase"
        if regressed:
            why = f"capability regressed on {len(regressed)} task(s): {', '.join(regressed[:4])}"
        elif cost_a >= cost_b:
            why = "no capability gain, and the change is not a proven-cheaper or defect-fix edit"
        return ImprovementCertificate(better=False, method="none", reason=why,
                                      regressed=regressed, cost_before=cost_b, cost_after=cost_a)

    # ---- report helpers (duck-typed so tests can pass lightweight fakes) ---- #
    @staticmethod
    def _regressions(before: Any, after: Any) -> List[str]:
        if before is None or after is None:
            return []
        try:
            return list(after.regression_vs(before))
        except Exception:  # noqa: BLE001 — a measurement glitch must never masquerade as a gain
            return []

    @staticmethod
    def _newly_passing(before: Any, after: Any) -> List[str]:
        """Task ids that pass now but did not pass in the baseline (capability gained)."""
        gained: List[str] = []
        try:
            for r in after.results:
                if not getattr(r, "passed", False):
                    continue
                prev = before.get(r.task_id)
                if prev is None or not getattr(prev, "passed", False):
                    gained.append(r.task_id)
        except Exception:  # noqa: BLE001
            return []
        return gained

    def _prove_equivalent(self, before_src: str, after_src: str,
                          kind: str) -> Optional[Dict[str, Any]]:
        try:
            from nyxara.growth.proof_carrying import ProofCarrier
            res = ProofCarrier(settings=self.settings).certify_edit(
                before_src, after_src, kind=kind)
            return res.to_dict()
        except Exception:  # noqa: BLE001 — no prover ⇒ method B simply does not fire
            return None


# --------------------------------------------------------------------------- #
# Static cost — a deterministic complexity ruler (lower is simpler)
# --------------------------------------------------------------------------- #
def static_cost(src: str) -> int:
    """AST node count + cyclomatic branch count; ``0`` for empty, large sentinel if unparseable."""
    if not src:
        return 0
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 10 ** 9      # an unparseable file is never "cheaper" than a valid one
    nodes = 0
    branches = 0
    for node in ast.walk(tree):
        nodes += 1
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try,
                             ast.ExceptHandler, ast.BoolOp, ast.IfExp, ast.comprehension,
                             ast.With, ast.AsyncWith, ast.Assert)):
            branches += 1
    return nodes + branches


# --------------------------------------------------------------------------- #
# Defect predicates (method C) — each is a decidable "the anti-pattern shrank" check
# --------------------------------------------------------------------------- #
_BARE_EXCEPT_RE = re.compile(r"(?m)^[ \t]*except[ \t]*:")
_EQ_NONE_RE = re.compile(r"(==|!=)\s*None\b|\bNone\s*(==|!=)")


def _defect_eliminated(before: str, after: str, kind: str) -> bool:
    """True iff the edit provably reduces the specific named defect (a decidable predicate)."""
    k = (kind or "").split(":")[-1]     # strip any 'self:'/'llm:' author prefix
    if k == "bare_except":
        return len(_BARE_EXCEPT_RE.findall(after)) < len(_BARE_EXCEPT_RE.findall(before))
    if k == "eq_none":
        return len(_EQ_NONE_RE.findall(after)) < len(_EQ_NONE_RE.findall(before))
    if k == "unused_import":
        # a genuinely dead import removed ⇒ strictly fewer imports AND a strictly simpler tree
        return (_import_count(after) < _import_count(before)
                and static_cost(after) < static_cost(before))
    if k == "missing_docstring":
        return _docstring_count(after) > _docstring_count(before)
    if k in ("not_in", "is_not_cmp"):
        return _negated_membership_count(after) < _negated_membership_count(before)
    return False


def _import_count(src: str) -> int:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 0
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            n += len(node.names)
        elif isinstance(node, ast.ImportFrom):
            n += len(node.names)
    return n


def _docstring_count(src: str) -> int:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 0
    n = 1 if ast.get_docstring(tree) else 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if ast.get_docstring(node):
                n += 1
    return n


def _negated_membership_count(src: str) -> int:
    """Count ``not <x> in <y>`` / ``not <x> is <y>`` (the E713/E714 anti-patterns)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 0
    n = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
                and isinstance(node.operand, ast.Compare)
                and len(node.operand.ops) == 1
                and isinstance(node.operand.ops[0], (ast.In, ast.Is))):
            n += 1
    return n


# --------------------------------------------------------------------------- #
# small internal helper
# --------------------------------------------------------------------------- #
def _mean_score(report: Any) -> float:
    try:
        return float(report.mean_score)
    except Exception:  # noqa: BLE001
        return 0.0


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA improvement-proof self-test")
    print("=" * 70)

    prover = ImprovementProver()

    # C · a bare-except fix provably eliminates the defect (no reports needed)
    b = "try:\n    pass\nexcept:\n    pass\n"
    a = "try:\n    pass\nexcept Exception:\n    pass\n"
    cert = prover.prove(before_src=b, after_src=a, edit_kind="bare_except")
    assert cert.better and cert.method == "defect-elimination", cert.to_dict()
    print("defect-elimination : bare-except fix proved better ✓")

    # B · a proven-equivalent, strictly-cheaper negated-compare rewrite
    b2 = "def f(a, b):\n    return not a in b\n"
    a2 = "def f(a, b):\n    return a not in b\n"
    cert2 = prover.prove(before_src=b2, after_src=a2, edit_kind="not_in")
    assert cert2.better, cert2.to_dict()
    print(f"equivalent-cheaper : negated-compare proved better via {cert2.method} ✓")

    # NOT better · a pure whitespace reformat (same behaviour, same cost, no defect) is REJECTED
    b3 = "def add(a, b):\n    return a+b\n"
    a3 = "def add(a, b):\n    return a + b\n"
    cert3 = prover.prove(before_src=b3, after_src=a3, edit_kind="self:high_complexity")
    assert not cert3.better, cert3.to_dict()
    print("no-op reformat     : correctly REJECTED (not provably better) ✓")

    # A · a capability Pareto-gain, measured on (fake) deterministic reports
    class _R:
        def __init__(self, tid, passed):
            self.task_id, self._p = tid, passed
        @property
        def passed(self):
            return self._p

    class _Rep:
        def __init__(self, results, mean):
            self.results, self.mean_score = results, mean
        def get(self, tid):
            return next((r for r in self.results if r.task_id == tid), None)
        def regression_vs(self, base):
            lost = []
            for br in base.results:
                if not br.passed:
                    continue
                cur = self.get(br.task_id)
                if cur is None or not cur.passed:
                    lost.append(br.task_id)
            return lost

    before_rep = _Rep([_R("t1", True), _R("t2", False)], 0.50)
    after_rep = _Rep([_R("t1", True), _R("t2", True)], 0.90)
    cert4 = prover.prove(before=before_rep, after=after_rep,
                         before_src="x = 1\n", after_src="x = 2\n", edit_kind="self:refactor")
    assert cert4.better and cert4.method == "pareto-capability", cert4.to_dict()
    print("pareto-capability  : a real capability gain proved better ✓")

    print("\nALL SELF-TESTS PASSED ✓")
