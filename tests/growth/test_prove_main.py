"""Tests for nyxara.growth.prove_main — the honest, offline self-drive proof.

Hermetic and offline: no LLM, no network, no torch. Every section exercises a real code path
(the Optimizer gauntlet, the LawDiscoveryEngine, the MissionExecutive) and must fail honestly
rather than print a fake pass. The proof never mutates NYXARA's own source.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from nyxara.growth import prove_main


def test_prove_runs_offline_and_exits_zero(capsys):
    rc = prove_main.main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ALL THREE PROVEN" in out
    for section in ("Recursive self-improvement", "Scientific invention", "Long-horizon autonomy"):
        assert section in out


def test_prove_self_improvement_rollback_is_byte_identical():
    ev = prove_main.prove_self_improvement()
    assert ev["ok"]
    assert ev["rollback_on_failure"] is True
    assert ev["byte_identical_after_rollback"] is True
    assert ev["kept_when_provably_better"] is True
    assert ev["improvement_method"] == "defect-elimination"


def test_prove_recovers_half_m_v_squared():
    ev = prove_main.prove_invention()
    assert ev["ok"]
    assert ev["verdict"] == "corroborated"
    assert ev["holdout_r2"] > 0.999
    assert abs(ev["recovered_coefficient"] - 0.5) < 1e-3


def test_prove_advances_one_milestone():
    ev = prove_main.prove_autonomy()
    assert ev["ok"]
    assert ev["mission_status"] == "completed"
    assert ev["mission_progress"] > 0.0
    assert ev["plan_steps"] >= 100
    assert ev["plan_acyclic"] is True


def test_prove_never_touches_real_source():
    pkg = Path(prove_main.__file__).resolve().parents[1]   # the nyxara/ package root
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in pkg.rglob("*.py")}
    prove_main.main([])
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in pkg.rglob("*.py")}
    assert before == after, "the proof must never mutate NYXARA's own source"


def test_prove_json_report_is_written(tmp_path):
    out = tmp_path / "proof.json"
    rc = prove_main.main(["--json", str(out)])
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["all_passed"] is True
    assert report["offline"] is True
    assert report["used_llm"] is False
    assert "honest_bounds" in report


def test_prove_report_is_json_serialisable():
    report = prove_main.run_proof()
    # a machine-readable report must round-trip through JSON with no custom encoder
    assert json.loads(json.dumps(report))["all_passed"] is True
