"""Digital phagocytosis: static-only analysis, never executes, extracts defensive signatures."""

from __future__ import annotations

from nyxara.nyx5.phagocytosis import DigitalPhagocyte


def test_detects_dangerous_call_without_executing(tmp_path):
    p = DigitalPhagocyte()
    canary = tmp_path / "canary.txt"
    hostile = f"import os\nos.system('touch {canary}')"
    sig = p.analyse(hostile)
    assert sig.executed is False
    assert not canary.exists()             # the payload NEVER ran
    assert any("os.system" in c for c in sig.categories)


def test_hardens_defenses():
    p = DigitalPhagocyte()
    p.analyse("import subprocess\nsubprocess.run(['x'])")
    assert any("subprocess" in d for d in p.defenses())


def test_obfuscated_input_flagged_not_run():
    p = DigitalPhagocyte()
    sig = p.analyse("this is not <valid python !!!")
    assert sig.executed is False
    assert "unparseable/obfuscated" in sig.categories


def test_benign_input_has_no_categories():
    p = DigitalPhagocyte()
    sig = p.analyse("x = 1 + 2\ny = sum([x, 3])")
    assert sig.categories == []
    assert sig.executed is False
