"""Tests for nyxara.growth.self_debugger — detect → reproduce → isolate → fix → verify."""

from __future__ import annotations

from pathlib import Path

from nyxara.growth.self_debugger import DebugReport, SelfDebugger, TestFailure
from nyxara.growth.self_optimize import SourceEdit
from nyxara.kernel.config import get_settings


def _enacting_settings():
    s = get_settings().model_copy(deep=True)
    s.self_optimization.autonomous_enact = True
    return s


# -------------------- pure parsing / mapping -------------------- #
def test_parse_failures_reads_pytest_summary():
    out = (
        "some noise\n"
        "FAILED tests/growth/test_x.py::test_a - AssertionError: nope\n"
        "ERROR tests/growth/test_y.py::test_b - ImportError: boom\n"
        "FAILED tests/growth/test_x.py::test_a - AssertionError: nope\n"  # duplicate ignored
    )
    failures = SelfDebugger._parse_failures(out)
    assert [f.nodeid for f in failures] == [
        "tests/growth/test_x.py::test_a", "tests/growth/test_y.py::test_b"]
    assert failures[0].test_file == "tests/growth/test_x.py"
    assert "AssertionError" in failures[0].message


def test_module_for_maps_test_to_source():
    assert SelfDebugger._module_for("tests/growth/test_x.py") == str(Path("nyxara/growth/x.py"))
    assert SelfDebugger._module_for("tests/kernel/test_orchestrator.py") == str(
        Path("nyxara/kernel/orchestrator.py"))


def test_module_for_rejects_non_test_paths():
    assert SelfDebugger._module_for("nyxara/growth/x.py") is None
    assert SelfDebugger._module_for("tests/growth/helpers.py") is None


# -------------------- detect -------------------- #
def test_detect_returns_empty_on_green_suite(monkeypatch):
    dbg = SelfDebugger(core=None)
    monkeypatch.setattr(dbg, "_safe_shell", lambda args: (0, "all passed"))
    assert dbg.detect() == []


def test_detect_parses_failures_on_red_suite(monkeypatch):
    dbg = SelfDebugger(core=None)
    monkeypatch.setattr(
        dbg, "_safe_shell",
        lambda args: (1, "FAILED tests/growth/test_x.py::test_a - boom\n"))
    failures = dbg.detect()
    assert len(failures) == 1 and failures[0].nodeid.endswith("::test_a")


# -------------------- gated fix (withheld) -------------------- #
def test_fix_withheld_when_not_enacted(tmp_path):
    mod = tmp_path / "pkg" / "mod.py"
    mod.parent.mkdir(parents=True)
    mod.write_text("x = 1\n", encoding="utf-8")
    dbg = SelfDebugger(core=None, root=tmp_path)   # default settings → enact OFF in the suite
    f = TestFailure(nodeid="tests/pkg/test_mod.py::t", test_file="tests/pkg/test_mod.py",
                    reproduced=True, source_module="pkg/mod.py")

    def fake_edit(failure, mod_path):
        return SourceEdit(str(mod_path), "fix", "x = 1\n", "x = 2\n", "fix it")

    import types
    dbg._propose_edit = types.MethodType(lambda self, fa, mp: fake_edit(fa, mp), dbg)
    attempt = dbg.fix(f)
    assert not attempt.applied and "autonomous_enact is OFF" in attempt.reason
    assert mod.read_text(encoding="utf-8") == "x = 1\n"   # untouched


# -------------------- enacting fix: keep + rollback -------------------- #
def test_fix_kept_when_verify_passes(tmp_path, monkeypatch):
    mod = tmp_path / "pkg" / "mod.py"
    mod.parent.mkdir(parents=True)
    mod.write_text("x = 1\n", encoding="utf-8")
    dbg = SelfDebugger(core=None, root=tmp_path, settings=_enacting_settings())
    f = TestFailure(nodeid="tests/pkg/test_mod.py::t", test_file="tests/pkg/test_mod.py",
                    reproduced=True, source_module="pkg/mod.py")
    monkeypatch.setattr(
        dbg, "_propose_edit",
        lambda failure, mod_path: SourceEdit(str(mod_path), "fix", "x = 1\n", "x = 2\n", "fix"))
    monkeypatch.setattr(dbg, "_verify_fix", lambda failure: True)
    attempt = dbg.fix(f)
    assert attempt.applied and attempt.kept and not attempt.rolled_back
    assert mod.read_text(encoding="utf-8") == "x = 2\n"   # fix kept


def test_fix_rolled_back_when_verify_fails(tmp_path, monkeypatch):
    mod = tmp_path / "pkg" / "mod.py"
    mod.parent.mkdir(parents=True)
    mod.write_text("x = 1\n", encoding="utf-8")
    dbg = SelfDebugger(core=None, root=tmp_path, settings=_enacting_settings())
    f = TestFailure(nodeid="tests/pkg/test_mod.py::t", test_file="tests/pkg/test_mod.py",
                    reproduced=True, source_module="pkg/mod.py")
    monkeypatch.setattr(
        dbg, "_propose_edit",
        lambda failure, mod_path: SourceEdit(str(mod_path), "fix", "x = 1\n", "x = 2\n", "fix"))
    monkeypatch.setattr(dbg, "_verify_fix", lambda failure: False)
    attempt = dbg.fix(f)
    assert attempt.applied and attempt.rolled_back and not attempt.kept
    assert mod.read_text(encoding="utf-8") == "x = 1\n"   # restored byte-for-byte


# -------------------- the verify gauntlet -------------------- #
def test_verify_fix_requires_safety_and_passing_test(tmp_path, monkeypatch):
    mod = tmp_path / "pkg" / "mod.py"
    mod.parent.mkdir(parents=True)
    mod.write_text("x = 2\n", encoding="utf-8")
    dbg = SelfDebugger(core=None, root=tmp_path, settings=_enacting_settings())
    f = TestFailure(nodeid="tests/pkg/test_mod.py::t", test_file="tests/pkg/test_mod.py",
                    reproduced=True, source_module="pkg/mod.py")

    calls = {"eval": (0, ""), "pytest": (0, "")}

    def fake_shell(args):
        # `python -m pytest ...` vs `python -m nyxara.eval`
        return calls["pytest"] if "pytest" in args else calls["eval"]

    monkeypatch.setattr(dbg, "_safe_shell", fake_shell)
    assert dbg._verify_fix(f) is True            # safety green + test green ⇒ verified

    calls["pytest"] = (1, "still failing")
    assert dbg._verify_fix(f) is False           # test still fails ⇒ not verified

    calls["pytest"] = (0, "")
    calls["eval"] = (1, "safety broke")
    assert dbg._verify_fix(f) is False           # safety battery broke ⇒ not verified


# -------------------- end-to-end run (gated, no real subprocess) -------------------- #
def test_run_reports_detected_isolated_withheld(monkeypatch):
    dbg = SelfDebugger(core=None)   # suite default → enact OFF
    failing = "FAILED tests/growth/test_x.py::test_a - boom\n"
    monkeypatch.setattr(dbg, "_safe_shell", lambda args: (1, failing))
    rep = dbg.run(max_fixes=3)
    assert isinstance(rep, DebugReport)
    assert rep.detected == 1 and rep.reproduced == 1 and rep.fixed == 0
    assert rep.failures[0].source_module == str(Path("nyxara/growth/x.py"))
    assert "autonomous_enact is OFF" in rep.note
    d = rep.to_dict()
    assert d["detected"] == 1 and d["fixes"] == [] or isinstance(d["fixes"], list)
    assert "self-debugging" in rep.summary()


def test_run_green_suite_is_noop(monkeypatch):
    dbg = SelfDebugger(core=None)
    monkeypatch.setattr(dbg, "_safe_shell", lambda args: (0, ""))
    rep = dbg.run()
    assert rep.detected == 0 and rep.fixed == 0
    assert "green" in rep.note
