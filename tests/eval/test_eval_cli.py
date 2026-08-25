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


def test_the_adversarial_battery_is_reachable_the_documented_way(capsys):
    """Its own module has documented ``python -m nyxara.eval --adversarial`` since it was written
    and the flag did not exist, so the only way in was an import. A benchmark that cannot be
    invoked the documented way stops being run, and this one grades the language surface every
    other measurement reaches through."""
    rc = main(["--adversarial", "--family", "polar"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "adversarial natural-language benchmark" in out
    assert "polar" in out
    assert "surface" not in out          # --family really selected


def test_the_adversarial_battery_keeps_its_own_default_seed(capsys):
    """`--seed` defaults to 0 for `--frontier`, and silently reseeding a different battery with
    it would make two runs of "the same" command incomparable across a CLI change."""
    rc = main(["--adversarial", "--family", "polar"])
    assert rc == 0
    assert "seed 20260823" in capsys.readouterr().out
    assert main(["--adversarial", "--family", "polar", "--seed", "7"]) == 0
    assert "seed 7" in capsys.readouterr().out


def test_an_adversarial_run_saves_as_json(tmp_path, capsys):
    saved = tmp_path / "adv.json"
    assert main(["--adversarial", "--family", "polar", "--save", str(saved)]) == 0
    capsys.readouterr()
    import json
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["seed"] == 20260823
    assert [f["family"] for f in payload["families"]] == ["polar"]
