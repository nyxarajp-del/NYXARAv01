"""Tests for nyxara.growth.self_optimize_main — the CLI entry point."""

from __future__ import annotations

import json

from nyxara.growth.self_optimize_main import main


def test_all_runs_and_returns_zero(capsys):
    rc = main(["--all", "--no-debug", "--generations", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "self-optimization" in out
    assert "safety-verification" in out


def test_single_phase_verify(capsys):
    rc = main(["--verify"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "safety-verification" in out


def test_save_writes_json(tmp_path, capsys):
    path = tmp_path / "report.json"
    rc = main(["--all", "--no-debug", "--generations", "0", "--save", str(path)])
    assert rc == 0
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["phases"]) == 11
    assert data["completed"] >= 1


def test_analyze_phase_group(capsys):
    rc = main(["--analyze"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "self-analysis" in out
