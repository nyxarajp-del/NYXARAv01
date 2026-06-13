"""Tests for nyxara.growth.self_optimize — auto-apply with verify-or-rollback."""

from __future__ import annotations

from pathlib import Path

from nyxara.growth.self_optimize import (EditGenerator, GauntletResult, Optimizer,
                                         SourceEdit, _fix_bare_except, _insert_docstring,
                                         _parses)
from nyxara.kernel.config import get_settings


def _enacting_settings():
    s = get_settings().model_copy(deep=True)
    s.self_improvement.autonomous_enact = True
    return s


def _bare_except_edit(path: Path) -> SourceEdit:
    original = "try:\n    pass\nexcept:\n    pass\n"
    path.write_text(original, encoding="utf-8")
    after = _fix_bare_except(original, 3)
    return SourceEdit(str(path), "bare_except", original, after, "fix bare except")


# ---- deterministic transforms ---- #
def test_fix_bare_except_transform():
    out = _fix_bare_except("try:\n    pass\nexcept:\n    pass\n", 3)
    assert "except Exception:" in out and "except:" not in out
    assert _parses(out)


def test_insert_docstring_transform():
    out = _insert_docstring("def f():\n    return 1\n", 1, "does a thing")
    assert out and '"""' in out and _parses(out)


# ---- the safety proof: rollback ---- #
def test_failing_gauntlet_rolls_back_byte_for_byte(tmp_path: Path):
    f = tmp_path / "victim.py"
    edit = _bare_except_edit(f)
    original = edit.before
    opt = Optimizer(settings=_enacting_settings(),
                    gauntlet=lambda files: GauntletResult(False, {}, "stub fail"))
    out = opt.apply(edit)
    assert out.applied and out.rolled_back and not out.kept
    assert f.read_text(encoding="utf-8") == original


def test_passing_gauntlet_keeps_edit(tmp_path: Path):
    f = tmp_path / "v.py"
    edit = _bare_except_edit(f)
    opt = Optimizer(settings=_enacting_settings(),
                    gauntlet=lambda files: GauntletResult(True, {"syntax": True}, "ok"))
    out = opt.apply(edit)
    assert out.kept and not out.rolled_back
    assert "except Exception:" in f.read_text(encoding="utf-8")


def test_gauntlet_exception_rolls_back(tmp_path: Path):
    f = tmp_path / "v.py"
    edit = _bare_except_edit(f)
    original = edit.before

    def boom(files):
        raise RuntimeError("gauntlet exploded")

    opt = Optimizer(settings=_enacting_settings(), gauntlet=boom)
    out = opt.apply(edit)
    assert out.rolled_back and f.read_text(encoding="utf-8") == original


def test_enact_off_withholds_and_does_not_write(tmp_path: Path):
    s = get_settings().model_copy(deep=True)
    s.self_improvement.autonomous_enact = False
    f = tmp_path / "safe.py"
    f.write_text("x = 1\n", encoding="utf-8")
    edit = SourceEdit(str(f), "bare_except", "x = 1\n", "x = 2\n", "noop")
    out = Optimizer(settings=s).apply(edit)
    assert not out.applied and not out.kept
    assert f.read_text(encoding="utf-8") == "x = 1\n"


def test_edit_is_journalled(tmp_path: Path):
    from nyxara.planning.journal import Journal
    f = tmp_path / "v.py"
    edit = _bare_except_edit(f)
    journal = Journal()
    opt = Optimizer(settings=_enacting_settings(), journal=journal,
                    gauntlet=lambda files: GauntletResult(True, {}, "ok"))
    opt.apply(edit)
    assert len(journal.actions()) >= 1


# ---- edit generation ---- #
def test_edit_generator_fixes_bare_except(tmp_path: Path):
    f = tmp_path / "g.py"
    f.write_text("try:\n    pass\nexcept:\n    pass\n", encoding="utf-8")

    class _W:
        id = "code-bare_except-g.py-3"
        locus = f"{f}:3"
        remediation = "use except Exception"
        title = "bare except"

    edit = EditGenerator().generate(_W())
    assert edit is not None and "except Exception:" in edit.after
    assert _parses(edit.after)
