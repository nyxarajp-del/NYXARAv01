"""Tests for nyxara.growth.__main__ — the Master-facing 'grow the brain' CLI (CPU/n-gram)."""

from __future__ import annotations

from pathlib import Path

from nyxara.growth.__main__ import main


def test_grow_trains_and_promotes_on_ngram(tmp_path: Path):
    code = main(["--backend", "ngram", "--generations", "1", "--data-dir", str(tmp_path)])
    assert code == 0
    # a first model always clears the gauntlet (nothing to beat) and becomes active
    assert (tmp_path / "foundry" / "active").exists()


def test_grow_with_bench_reports_handoff(tmp_path: Path, capsys):
    code = main(["--backend", "ngram", "--generations", "1",
                 "--data-dir", str(tmp_path), "--bench"])
    assert code == 0
    out = capsys.readouterr().out
    assert "promoted" in out.lower()
    assert "handoff" in out.lower()          # the wrapper→own-AI metric is surfaced


def test_grow_distill_skips_cleanly_without_a_teacher(tmp_path: Path, capsys):
    # no real teacher is configured under the test profile -> distillation is skipped, not fatal
    code = main(["--distill", "--backend", "ngram", "--data-dir", str(tmp_path)])
    assert code == 0
    assert "skipping distillation" in capsys.readouterr().out.lower()
    assert (tmp_path / "foundry" / "active").exists()


def test_grow_does_not_mutate_global_settings(tmp_path: Path):
    from nyxara.kernel.config import get_settings
    before = get_settings().foundry.enabled
    main(["--backend", "ngram", "--data-dir", str(tmp_path)])
    # the CLI works on a deep copy, so the process-global config is untouched
    assert get_settings().foundry.enabled == before
