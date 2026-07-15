"""Tests for the widened deterministic self-refactor library (RSI, LLM-free).

Covers the six new transforms end to end — detector (`self_review`) → weakness (`weakness`) →
transform (`self_optimize`) → provably-better certificate (`improvement_proof`) → kept edit
(`Optimizer.apply`) — plus the continuous, observable RSI driver (`run_continuous`). Every
transform is AST-validated and behaviour-preserving (or, for the mutable-default fix,
behaviour-improving), so each is *kept* under the strict provably-better gate, not rolled back.
"""

from __future__ import annotations

from pathlib import Path

from nyxara.growth.improvement_proof import ImprovementProver
from nyxara.growth.recursive_improvement import (RecursiveSelfImprovement,
                                                 SelfImprovementReport)
from nyxara.growth.self_optimize import (GauntletResult, Optimizer, SourceEdit, _parses,
                                         _dedent_redundant_else, _fix_double_negation,
                                         _fix_empty_collection_call, _fix_mutable_default_arg,
                                         _fix_not_eq_compare, _remove_redundant_pass)
from nyxara.growth.self_review import SelfReviewer
from nyxara.growth.weakness import WeaknessSynthesizer
from nyxara.kernel.config import get_settings


# --------------------------------------------------------------------------- #
# 1) transforms — correct, valid, behaviour-preserving
# --------------------------------------------------------------------------- #
def test_not_eq_compare_transform():
    assert _fix_not_eq_compare("x = not (a == b)\n", 1) == "x = a != b\n"
    assert _fix_not_eq_compare("x = not (a != b)\n", 1) == "x = a == b\n"
    # a comparison to None is left to the eq_none transform, never mis-handled here
    assert _fix_not_eq_compare("x = not (a == None)\n", 1) is None


def test_double_negation_transform():
    assert _fix_double_negation("y = not not flag\n", 1) == "y = bool(flag)\n"
    assert _parses(_fix_double_negation("y = not not (a or b)\n", 1))


def test_empty_collection_call_transform():
    assert _fix_empty_collection_call("z = list()\n", 1) == "z = []\n"
    assert _fix_empty_collection_call("z = dict()\n", 1) == "z = {}\n"
    assert _fix_empty_collection_call("z = tuple()\n", 1) == "z = ()\n"
    # a non-empty call is not an empty-literal defect
    assert _fix_empty_collection_call("z = list(range(3))\n", 1) is None


def test_redundant_pass_transform():
    assert _remove_redundant_pass("if a:\n    pass\n    go()\n", 2) == "if a:\n    go()\n"
    # a lone `pass` (the block's only statement) is NOT redundant and must be kept
    assert _remove_redundant_pass("if a:\n    pass\n", 2) is None
    # a `pass` carrying a comment is left alone (removing it would drop the comment)
    assert _remove_redundant_pass("if a:\n    pass  # note\n    go()\n", 2) is None


def test_redundant_else_return_transform():
    src = "def f(a):\n    if a:\n        return 1\n    else:\n        return 2\n"
    out = _dedent_redundant_else(src, 5)
    assert out == "def f(a):\n    if a:\n        return 1\n    return 2\n"
    # an elif is never de-indented (it is not the redundant-else anti-pattern)
    elif_src = "def f(a):\n    if a:\n        return 1\n    elif b:\n        return 2\n"
    assert _dedent_redundant_else(elif_src, 5) is None


def test_mutable_default_arg_transform():
    out = _fix_mutable_default_arg("def g(x=[]):\n    return x\n", 1)
    assert out == "def g(x=None):\n    if x is None:\n        x = []\n    return x\n"
    # the guard is inserted AFTER a docstring, never before it
    doc = _fix_mutable_default_arg('def g(x=[]):\n    """doc."""\n    return x\n', 1)
    assert doc == ('def g(x=None):\n    """doc."""\n    if x is None:\n'
                   '        x = []\n    return x\n')
    # an immutable default (a tuple) is never rewritten
    assert _fix_mutable_default_arg("def g(x=()):\n    return x\n", 1) is None


# --------------------------------------------------------------------------- #
# 2) detectors — the reviewer flags each new anti-pattern
# --------------------------------------------------------------------------- #
def _kinds(tmp_path: Path, src: str) -> dict:
    f = tmp_path / "mod.py"
    f.write_text(src, encoding="utf-8")
    return SelfReviewer(root=tmp_path).review_file(f)


def test_detectors_fire(tmp_path: Path):
    cases = {
        "not_eq_cmp": "def f(a, b):\n    return not (a == b)\n",
        "double_negation": "def f(x):\n    return not not x\n",
        "empty_collection_call": "def f():\n    return list()\n",
        "redundant_pass": "def f(a):\n    if a:\n        pass\n        a()\n",
        "redundant_else_return": "def f(a):\n    if a:\n        return 1\n    else:\n        return 2\n",
        "mutable_default_arg": "def f(x=[]):\n    return x\n",
    }
    for kind, src in cases.items():
        found = {c.kind for c in _kinds(tmp_path, src)}
        assert kind in found, f"{kind} not detected; got {found}"


# --------------------------------------------------------------------------- #
# 3) weakness mapping — each kind is a deterministic source edit
# --------------------------------------------------------------------------- #
def test_weakness_marks_deterministic(tmp_path: Path):
    from nyxara.growth.self_review import CodeReviewReport
    src = "def f(x=[]):\n    return not (a == b)\n"
    f = tmp_path / "mod.py"
    f.write_text(src, encoding="utf-8")
    findings = SelfReviewer(root=tmp_path).review_file(f)
    report = CodeReviewReport(findings=findings, files_scanned=1)
    weaknesses = WeaknessSynthesizer().synthesize(code=report).ranked()
    by_kind = {}
    for w in weaknesses:
        for kind in ("mutable_default_arg", "not_eq_cmp"):
            if kind in w.id:
                by_kind[kind] = w
    for kind in ("mutable_default_arg", "not_eq_cmp"):
        assert kind in by_kind, f"no weakness for {kind}"
        assert by_kind[kind].is_source_edit and by_kind[kind].edit_strategy == "deterministic"


# --------------------------------------------------------------------------- #
# 4) provably-better — each new transform certifies as defect-elimination
# --------------------------------------------------------------------------- #
def test_prover_certifies_defect_elimination():
    prover = ImprovementProver()
    cases = [
        ("not_eq_cmp", "x = not (a == b)\n", "x = a != b\n"),
        ("double_negation", "y = not not f\n", "y = bool(f)\n"),
        ("empty_collection_call", "z = list()\n", "z = []\n"),
        ("redundant_pass", "if a:\n    pass\n    go()\n", "if a:\n    go()\n"),
        ("redundant_else_return",
         "def f(a):\n    if a:\n        return 1\n    else:\n        return 2\n",
         "def f(a):\n    if a:\n        return 1\n    return 2\n"),
        ("mutable_default_arg", "def g(x=[]):\n    return x\n",
         "def g(x=None):\n    if x is None:\n        x = []\n    return x\n"),
    ]
    for kind, before, after in cases:
        cert = prover.prove(before_src=before, after_src=after, edit_kind=kind)
        assert cert.better and cert.method == "defect-elimination", (kind, cert.to_dict())
    # a broken (unparseable) after is never certified as "the defect shrank"
    bad = prover.prove(before_src="z = list()\n", after_src="z = [\n", edit_kind="empty_collection_call")
    assert not bad.better


# --------------------------------------------------------------------------- #
# 5) end-to-end — Optimizer keeps a new-kind edit (real prover, no LLM, no benchmark subprocess)
# --------------------------------------------------------------------------- #
def _enacting_settings():
    s = get_settings().model_copy(deep=True)
    s.self_improvement.autonomous_enact = True
    return s


def test_optimizer_keeps_new_kind_edits(tmp_path: Path):
    settings = _enacting_settings()
    cases = [
        ("mutable_default_arg", "def g(x=[]):\n    return x\n", _fix_mutable_default_arg, 1),
        ("not_eq_cmp", "r = not (a == b)\n", _fix_not_eq_compare, 1),
        ("redundant_pass", "if a:\n    pass\n    go()\n", _remove_redundant_pass, 2),
    ]
    for kind, before, fix, lineno in cases:
        f = tmp_path / f"{kind}.py"
        f.write_text(before, encoding="utf-8")
        after = fix(before, lineno)
        edit = SourceEdit(str(f), kind, before, after, f"fix {kind}")
        opt = Optimizer(root=tmp_path, settings=settings,
                        gauntlet=lambda files: GauntletResult(True, {"syntax": True}, "ok"))
        opt._baseline_path = tmp_path / "skip.json"     # non-existent ⇒ no benchmark subprocess
        out = opt.apply(edit)
        opt.close()
        assert out.kept and not out.rolled_back, (kind, out.reason)
        assert out.improvement and out.improvement["method"] == "defect-elimination", kind
        assert f.read_text(encoding="utf-8") == after, kind


# --------------------------------------------------------------------------- #
# 6) the continuous, observable RSI driver
# --------------------------------------------------------------------------- #
def test_run_continuous_threads_index_and_tallies():
    rsi = RecursiveSelfImprovement()
    counter = {"n": 0}

    def fake_run(*, enact=None, category=None):
        counter["n"] += 1
        r = SelfImprovementReport()
        r.intelligence_index = 0.40 + 0.05 * counter["n"]
        r.kept, r.rolled_back, r.lessons_stored = 1, 0, 2
        return r

    rsi.run = fake_run
    summary = rsi.run_continuous(3, enact=False)
    assert summary["cycles_ran"] == 3
    assert summary["index_trajectory"] == [0.45, 0.50, 0.55]
    assert abs(summary["index_gain"] - 0.10) < 1e-9
    assert summary["kept"] == 3 and summary["lessons_stored"] == 6
    assert summary["stopped_early"] is None


def test_run_continuous_halts_on_closed_gate():
    rsi = RecursiveSelfImprovement()
    ran = {"n": 0}

    def fake_run(*, enact=None, category=None):
        ran["n"] += 1
        return SelfImprovementReport()

    rsi.run = fake_run

    class ClosedGate:
        def gate(self):
            return False

    summary = rsi.run_continuous(5, enact=True, oversight=ClosedGate())
    assert summary["cycles_ran"] == 0 and ran["n"] == 0
    assert "scram" in (summary["stopped_early"] or "").lower()
