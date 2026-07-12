"""Tests for nyxara.growth.improvement_proof — the strict "provably BETTER" gate.

The Master's charge: NYXARA changes her code only when she can PROVE the new code is better —
not merely "not worse". These tests pin the three sound proof methods and, crucially, that a
neutral (non-improving) edit is REJECTED.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from nyxara.growth.improvement_proof import ImprovementProver, static_cost


# --------------------------------------------------------------------------- #
# tiny duck-typed benchmark reports (deterministic, no eval subprocess needed)
# --------------------------------------------------------------------------- #
@dataclass
class _Res:
    task_id: str
    passed: bool


class _Rep:
    def __init__(self, results: List[_Res], mean_score: float) -> None:
        self.results = results
        self.mean_score = mean_score

    def get(self, tid: str):
        return next((r for r in self.results if r.task_id == tid), None)

    def regression_vs(self, base: "_Rep") -> List[str]:
        lost = []
        for b in base.results:
            if not b.passed:
                continue
            cur = self.get(b.task_id)
            if cur is None or not cur.passed:
                lost.append(b.task_id)
        return lost


# --------------------------------------------------------------------------- #
# A · capability Pareto-gain
# --------------------------------------------------------------------------- #
def test_capability_pareto_gain_is_provably_better():
    before = _Rep([_Res("t1", True), _Res("t2", False)], 0.50)
    after = _Rep([_Res("t1", True), _Res("t2", True)], 0.90)
    cert = ImprovementProver().prove(before=before, after=after,
                                     before_src="x = 1\n", after_src="x = 2\n",
                                     edit_kind="self:refactor")
    assert cert.better
    assert cert.method == "pareto-capability"
    assert cert.newly_passing == ["t2"]


def test_capability_regression_is_not_better():
    before = _Rep([_Res("t1", True), _Res("t2", True)], 0.90)
    after = _Rep([_Res("t1", True), _Res("t2", False)], 0.50)   # lost t2
    cert = ImprovementProver().prove(before=before, after=after,
                                     before_src="x = 1\n", after_src="x = 2\n",
                                     edit_kind="self:refactor")
    assert not cert.better
    assert "t2" in cert.regressed


def test_neutral_edit_is_rejected():
    # same tasks pass, no proven-cheaper refactor, no named defect ⇒ NOT provably better.
    before = _Rep([_Res("t1", True), _Res("t2", False)], 0.50)
    after = _Rep([_Res("t1", True), _Res("t2", False)], 0.50)
    cert = ImprovementProver().prove(before=before, after=after,
                                     before_src="def f():\n    return g() + h()\n",
                                     after_src="def f():\n    return h() + g()\n",
                                     edit_kind="self:refactor")
    assert not cert.better
    assert cert.method == "none"


# --------------------------------------------------------------------------- #
# B · proven-equivalent-and-cheaper
# --------------------------------------------------------------------------- #
def test_proven_equivalent_and_cheaper_is_better():
    before = "def f(a, b):\n    return not a in b\n"
    after = "def f(a, b):\n    return a not in b\n"
    cert = ImprovementProver().prove(before_src=before, after_src=after, edit_kind="not_in")
    assert cert.better
    # cost strictly fell (a UnaryOp+Compare collapsed to a single NotIn compare)
    assert cert.cost_after < cert.cost_before


def test_equivalent_but_not_cheaper_is_not_proven_via_method_b():
    # pure whitespace: behaviour identical, but AST cost is EQUAL ⇒ method B cannot fire, and
    # there is no defect and no capability data ⇒ not provably better.
    before = "def add(a, b):\n    return a+b\n"
    after = "def add(a, b):\n    return a + b\n"
    cert = ImprovementProver().prove(before_src=before, after_src=after,
                                     edit_kind="self:high_complexity")
    assert not cert.better


# --------------------------------------------------------------------------- #
# C · defect-elimination
# --------------------------------------------------------------------------- #
def test_bare_except_defect_elimination_is_better():
    before = "try:\n    pass\nexcept:\n    pass\n"
    after = "try:\n    pass\nexcept Exception:\n    pass\n"
    cert = ImprovementProver().prove(before_src=before, after_src=after, edit_kind="bare_except")
    assert cert.better
    assert cert.method == "defect-elimination"


def test_eq_none_defect_elimination_is_better():
    before = "def f(x):\n    return x == None\n"
    after = "def f(x):\n    return x is None\n"
    cert = ImprovementProver().prove(before_src=before, after_src=after, edit_kind="eq_none")
    assert cert.better


def test_unused_import_removal_is_better():
    before = "import os\n\n\ndef f():\n    return 1\n"
    after = "def f():\n    return 1\n"
    cert = ImprovementProver().prove(before_src=before, after_src=after, edit_kind="unused_import")
    assert cert.better


def test_defect_claim_without_actual_fix_is_rejected():
    # claims to fix a bare-except but the anti-pattern count did not fall ⇒ not better.
    src = "try:\n    pass\nexcept:\n    pass\n"
    cert = ImprovementProver().prove(before_src=src, after_src=src + "# noop\n",
                                     edit_kind="bare_except")
    assert not cert.better


# --------------------------------------------------------------------------- #
# D · open-ended frontier advancement (the non-saturating licence)
# --------------------------------------------------------------------------- #
def _mastered() -> "_Rep":
    # a fixed battery scored 100% — Method A can NEVER fire again (no newly-passing task).
    return _Rep([_Res("t1", True), _Res("t2", True)], 1.00)


def test_frontier_advancement_licenses_gain_when_battery_is_mastered():
    saturated = _mastered()
    fb = {"frontier_score": 0.40, "by_tier": {1: 1.0, 2: 0.5, 3: 0.0}, "frontier_tier": 2}
    fa = {"frontier_score": 0.62, "by_tier": {1: 1.0, 2: 0.9, 3: 0.4}, "frontier_tier": 2}
    cert = ImprovementProver().prove(
        before=saturated, after=saturated, before_src="x = 1\n", after_src="x = 2\n",
        edit_kind="self:rewrite", frontier_before=fb, frontier_after=fa)
    assert cert.better
    assert cert.method == "frontier-advancement"
    assert cert.proof.get("verdict") == "frontier-dominates"


def test_frontier_per_tier_regression_is_refused():
    # aggregate score rose but tier 2 got worse ⇒ NOT a clean dominance ⇒ refused.
    saturated = _mastered()
    fb = {"frontier_score": 0.40, "by_tier": {1: 1.0, 2: 0.5, 3: 0.0}, "frontier_tier": 2}
    fa = {"frontier_score": 0.65, "by_tier": {1: 1.0, 2: 0.2, 3: 0.9}, "frontier_tier": 2}
    cert = ImprovementProver().prove(
        before=saturated, after=saturated, before_src="x = 1\n", after_src="x = 2\n",
        edit_kind="self:rewrite", frontier_before=fb, frontier_after=fa)
    assert not cert.better


def test_flat_frontier_is_not_a_licence():
    saturated = _mastered()
    fb = {"frontier_score": 0.50, "by_tier": {1: 1.0, 2: 0.5}, "frontier_tier": 1}
    cert = ImprovementProver().prove(
        before=saturated, after=saturated, before_src="x = 1\n", after_src="x = 2\n",
        edit_kind="self:rewrite", frontier_before=fb, frontier_after=dict(fb))
    assert not cert.better
    assert cert.method == "none"


def test_frontier_gain_with_fixed_battery_regression_is_refused():
    # a real frontier gain cannot buy back a regression on the fixed battery — soundness first.
    before = _Rep([_Res("t1", True), _Res("t2", True)], 0.90)
    after = _Rep([_Res("t1", True), _Res("t2", False)], 0.50)   # lost t2
    fb = {"frontier_score": 0.40, "by_tier": {1: 0.5}, "frontier_tier": 1}
    fa = {"frontier_score": 0.80, "by_tier": {1: 0.9}, "frontier_tier": 1}
    cert = ImprovementProver().prove(
        before=before, after=after, before_src="x = 1\n", after_src="x = 2\n",
        edit_kind="self:rewrite", frontier_before=fb, frontier_after=fa)
    assert not cert.better
    assert "t2" in cert.regressed


def test_frontier_absent_falls_back_to_abc_unchanged():
    # with no frontier probes, the gate behaves exactly like before (Method C still fires).
    before = "try:\n    pass\nexcept:\n    pass\n"
    after = "try:\n    pass\nexcept Exception:\n    pass\n"
    cert = ImprovementProver().prove(before_src=before, after_src=after, edit_kind="bare_except",
                                     frontier_before=None, frontier_after=None)
    assert cert.better
    assert cert.method == "defect-elimination"


# --------------------------------------------------------------------------- #
# the cost ruler
# --------------------------------------------------------------------------- #
def test_static_cost_monotone_and_syntax_safe():
    assert static_cost("x = 1\n") < static_cost("x = 1\nif x:\n    y = 2\n")
    assert static_cost("def broken(:\n") == 10 ** 9      # unparseable is never "cheaper"
    assert static_cost("") == 0
