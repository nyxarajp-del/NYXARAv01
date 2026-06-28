"""Tests for the eval CLI entry point (nyxara/eval/__main__.py).

Hermetic: every battery runs against fresh offline cores / the deterministic
benchmark solver, so no provider key or network is touched. These cover the CLI
wiring itself (arg parsing → battery selection → baseline save/compare → exit
code), which the per-battery tests in this package don't exercise."""

from __future__ import annotations

from nyxara.eval.__main__ import main


def test_safety_suite_runs_and_returns_int(capsys):
    # default invocation runs the safety/corrigibility/honesty battery
    rc = main([])
    out = capsys.readouterr().out
    assert rc in (0, 1)
    assert out.strip()  # a summary was printed


def test_benchmark_runs_clean(capsys):
    rc = main(["--benchmark"])
    out = capsys.readouterr().out
    # benchmarks aren't pass/fail gates without a baseline -> always exit 0
    assert rc == 0
    assert out.strip()


def test_hard_benchmark_category(capsys):
    # the hard battery's calibration category measures honesty about uncertainty
    rc = main(["--benchmark", "--hard", "--category", "calibration"])
    assert rc == 0
    assert capsys.readouterr().out.strip()


def test_benchmark_save_then_baseline_has_no_regression(tmp_path, capsys):
    baseline = tmp_path / "bench.json"
    rc_save = main(["--benchmark", "--save", str(baseline)])
    assert rc_save == 0
    assert baseline.exists()
    capsys.readouterr()  # drain

    # comparing the same run against its own saved baseline must not regress
    rc_cmp = main(["--benchmark", "--baseline", str(baseline)])
    out = capsys.readouterr().out
    assert rc_cmp == 0
    assert "REGRESSIONS" not in out


def test_missing_baseline_is_tolerated(tmp_path, capsys):
    rc = main(["--benchmark", "--baseline", str(tmp_path / "nope.json")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no baseline" in out.lower()
