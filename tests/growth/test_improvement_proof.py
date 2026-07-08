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
# the cost ruler
# --------------------------------------------------------------------------- #
def test_static_cost_monotone_and_syntax_safe():
    assert static_cost("x = 1\n") < static_cost("x = 1\nif x:\n    y = 2\n")
    assert static_cost("def broken(:\n") == 10 ** 9      # unparseable is never "cheaper"
    assert static_cost("") == 0
