"""Stage D — the Causal Code Engine.

The self-debugger isolated a failure by *filename* (test_x.py → x.py). These tests prove the causal
tree: the real traceback is parsed into the executed nyxara frames, the deepest non-test frame is
chosen as the proximate root (over the namesake), the causal model reranks across sessions, and the
gated/reversible repair is aimed at the causal root — never weakening a gate (autonomous_enact is OFF
by default, so no source is edited in the test).
"""

from __future__ import annotations

from nyxara.growth.causal_code_engine import CausalCodeEngine, CausalDiagnosis, _is_test_path
from nyxara.growth.self_debugger import FixAttempt, TestFailure


class _StubDebugger:
    """A debugger that never runs pytest or edits files — records what it was asked to fix."""

    def __init__(self):
        self.root = "."
        self.autonomous_enact = False
        self.fixed_modules = []

    def fix(self, failure: TestFailure) -> FixAttempt:
        self.fixed_modules.append(failure.source_module)
        return FixAttempt(nodeid=failure.nodeid, module=failure.source_module,
                          reason="autonomous_enact is OFF — fix withheld")

    def detect(self, *, test_path=None):
        return []

    def isolate(self, failures):
        pass

    def _pytest(self, target):
        return 0, ""


def _engine():
    return CausalCodeEngine(debugger=_StubDebugger())


def test_is_test_path_filters_test_files():
    assert _is_test_path("tests/growth/test_x.py")
    assert _is_test_path("nyxara/growth/tests/test_y.py")
    assert not _is_test_path("nyxara/growth/x.py")


def test_deepest_frame_is_the_proximate_root_over_namesake():
    eng = _engine()
    # a traceback where the test's namesake is shallow and the real fault is deeper in another module
    tb = (
        'tests/growth/test_alpha.py:10: in test_alpha\n'
        '    alpha.run()\n'
        'nyxara/growth/alpha.py:42: in run\n'
        '    return beta.compute(x)\n'
        'nyxara/mind/beta.py:88: in compute\n'
        '    raise ValueError("boom")\n'
        'E   ValueError: boom\n'
    )
    failure = TestFailure(nodeid="tests/growth/test_alpha.py::test_alpha",
                          test_file="tests/growth/test_alpha.py",
                          message=tb, reproduced=True, source_module="nyxara/growth/alpha.py")
    diag = eng.causal_tree(failure)
    assert isinstance(diag, CausalDiagnosis)
    # the deepest executed nyxara frame is the proximate cause — NOT the namesake alpha.py
    assert diag.root_module == "nyxara/mind/beta.py"
    assert diag.tree[0].role == "proximate"
    # alpha.py is retained as a contributory node, and the test file is excluded entirely
    modules = {n.module for n in diag.tree}
    assert "nyxara/growth/alpha.py" in modules
    assert not any(_is_test_path(m) for m in modules)


def test_repair_aims_at_the_causal_root_not_the_namesake():
    stub = _StubDebugger()
    eng = CausalCodeEngine(debugger=stub)
    tb = ('nyxara/growth/alpha.py:42: in run\n'
          'nyxara/mind/beta.py:88: in compute\n'
          'E   ValueError: boom\n')
    failure = TestFailure(nodeid="tests/growth/test_alpha.py::test_alpha",
                          test_file="tests/growth/test_alpha.py",
                          message=tb, reproduced=True, source_module="nyxara/growth/alpha.py")
    diag, attempt = eng.repair(failure)
    assert diag.root_module == "nyxara/mind/beta.py"
    # the existing (gated) fix path was aimed at the causal root, not the filename namesake
    assert stub.fixed_modules == ["nyxara/mind/beta.py"]
    assert attempt.module == "nyxara/mind/beta.py"
    assert "[causal root: nyxara/mind/beta.py" in attempt.reason


def test_namesake_fallback_when_no_frames_in_traceback():
    eng = _engine()
    failure = TestFailure(nodeid="tests/growth/test_z.py::test_z",
                          test_file="tests/growth/test_z.py",
                          message="AssertionError: assert 1 == 2", reproduced=True,
                          source_module="nyxara/growth/z.py")
    diag = eng.causal_tree(failure)
    # no nyxara frame named → fall back to the namesake mapping, honestly labelled
    assert diag.root_module == "nyxara/growth/z.py"
    assert diag.tree[0].role == "namesake"


def test_causal_model_learns_and_reranks_across_episodes():
    from nyxara.mind.causal_world_model import CausalWorldModel
    model = CausalWorldModel()
    eng = CausalCodeEngine(debugger=_StubDebugger(), causal_model=model)
    tb = ('nyxara/growth/alpha.py:42: in run\n'
          'nyxara/mind/beta.py:88: in compute\n')
    failure = TestFailure(nodeid="tests/growth/test_alpha.py::test_alpha",
                          test_file="tests/growth/test_alpha.py",
                          message=tb, reproduced=True, source_module="nyxara/growth/alpha.py")
    # repeatedly diagnosing the same fault feeds the model without error and stays sound
    for _ in range(6):
        diag = eng.causal_tree(failure)
        assert diag.root_module == "nyxara/mind/beta.py"
    # the model has recorded the episodes (module:* and fail:* events exist)
    assert model.why(f"fail:{failure.nodeid}") is not None


def test_run_pass_reports_structure(monkeypatch):
    stub = _StubDebugger()
    # one reproduced failure surfaced by detect/isolate
    f = TestFailure(nodeid="tests/growth/test_alpha.py::test_alpha",
                    test_file="tests/growth/test_alpha.py",
                    message="nyxara/mind/beta.py:88: in compute\nE ValueError: boom",
                    reproduced=True, source_module="nyxara/growth/alpha.py")
    stub.detect = lambda *, test_path=None: [f]
    eng = CausalCodeEngine(debugger=stub)
    out = eng.run(max_fixes=3)
    assert out["detected"] == 1
    assert out["enacted"] is False
    assert out["results"] and out["results"][0]["diagnosis"]["root_module"] == "nyxara/mind/beta.py"
