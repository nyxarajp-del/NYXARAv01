"""Tests for nyxara.growth.improve_main — the Master-facing RSI CLI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from nyxara.growth.improve_main import main


def test_all_runs_and_exits_zero(capsys):
    code = main(["--all"])
    assert code == 0
    assert "recursive self-improvement" in capsys.readouterr().out.lower()


def test_review_only(capsys):
    code = main(["--review"])
    assert code == 0
    assert "code-review" in capsys.readouterr().out.lower()


def test_optimize_without_enact_changes_nothing(capsys):
    target = Path("nyxara/growth/autolearn.py")
    before = hashlib.sha256(target.read_bytes()).hexdigest()
    code = main(["--optimize"])     # no --enact → dry-run
    assert code == 0
    assert hashlib.sha256(target.read_bytes()).hexdigest() == before


def test_save_writes_json(tmp_path: Path):
    out = tmp_path / "report.json"
    main(["--review", "--save", str(out)])
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "findings" in data


def test_fail_on_weakness_exits_nonzero_at_low_threshold(monkeypatch, capsys):
    # force a very low severity threshold so any weakness trips the gate
    from nyxara.kernel import config as cfg
    s = cfg.get_settings().model_copy(deep=True)
    s.self_improvement.weakness_fail_severity = 0.01
    monkeypatch.setattr(cfg, "get_settings", lambda: s)
    code = main(["--weaknesses", "--fail-on-weakness"])
    assert code == 1
    assert "fail" in capsys.readouterr().out.lower()
