"""Tests for nyxara.growth.invention — genuine invention (Pillar F · Edge 3).

The contract these tests pin down is the honest one: the engine *invents* its own candidate designs
(no LLM), but **nothing is kept unless an independent, quantitative model certifies it** — feasible,
genuinely novel against the real known art, and *better than the best known baseline*. For the
algorithm domain that certification is exact and re-runnable: every kept algorithm is re-executed
against a brute-force ``sorted`` oracle and must be both correct and faster than the baseline. We also
pin the feasibility filter, the anti-remix novelty filter, determinism, and serialisation.
"""

from __future__ import annotations

from nyxara.growth.invention import (
    DOMAINS,
    DesignCandidate,
    Invention,
    InventionEngine,
    InventionReport,
    algorithm_metrics,
    synthesize_sort,
    _BATTERY_BASELINES,
    _battery_eval,
)


def _engine(seed: int = 7) -> InventionEngine:
    # No memory / knowledge / flywheel sinks: fully offline, in-process, deterministic.
    return InventionEngine(seed=seed)


# --------------------------------------------------------------------------- #
# Soundness — the whole point: a kept invention is feasible, novel and better than the known art.
# --------------------------------------------------------------------------- #
def test_engine_keeps_genuinely_novel_better_designs():
    rep = _engine().invent(generations=5, population=40)
    assert rep.novel_kept > 0, "engine should invent at least one feasible, novel, better design"
    for iv in rep.inventions:
        assert iv.feasible is True
        assert iv.improvement > 1.0, "a kept design must beat the best known baseline"
        assert iv.novelty >= _engine().novelty_floor, "a kept design must be far from the known art"


def test_every_kept_algorithm_is_correct_and_faster_than_baseline():
    # The Eureka-grade domain: re-execute each kept algorithm against the brute-force oracle.
    rep = _engine().invent(generations=5, population=40)
    kept_alg = [iv for iv in rep.inventions if iv.domain == "algorithm"]
    for iv in kept_alg:
        m = algorithm_metrics(iv.candidate.genome)
        assert m["correct"] is True, f"kept algorithm is not correct: {iv.candidate.genome}"
        assert m["speedup"] > 1.0, f"kept algorithm is not faster than baseline: {iv.candidate.genome}"


def test_synthesized_sorts_actually_sort():
    # Any genome the engine can emit must produce a real, correct sort (or be filtered as infeasible).
    suite = [[5, 1, 4, 2, 8], [3, 3, 3], [], [9], list(range(20, 0, -1))]
    for genome in (
        {"strategy": "insertion"},
        {"strategy": "merge", "base_threshold": 4},
        {"strategy": "quick", "base_threshold": 8, "pivot": "median3", "introsort_fallback": True},
        {"strategy": "heap"},
        {"strategy": "counting"},
    ):
        run = synthesize_sort(genome)
        for arr in suite:
            out, ops = run(arr)
            assert out == sorted(arr), f"{genome} failed on {arr}: got {out}"
            assert ops >= 0


# --------------------------------------------------------------------------- #
# Feasibility — an infeasible design is discarded, exactly like a refuted conjecture.
# --------------------------------------------------------------------------- #
def test_infeasible_chemistry_is_rejected():
    # An aqueous electrolyte (window ≈ 1.5 V) cannot hold a 3.8 V cathode — oxidative breakdown.
    bad = _battery_eval({"anode": "graphite", "cathode": "NMC811", "electrolyte": "aqueous"})
    assert bad["feasible"] is False
    # A dendrite-prone lithium-metal anode in a liquid electrolyte is a known failure mode.
    dendrite = _battery_eval({"anode": "lithium_metal", "cathode": "sulfur",
                              "electrolyte": "carbonate_liquid"})
    assert dendrite["feasible"] is False
    # A real, conventional Li-ion cell is feasible.
    ok = _battery_eval({"anode": "graphite", "cathode": "LiFePO4", "electrolyte": "carbonate_liquid"})
    assert ok["feasible"] is True and ok["fom"] > 0


def test_series_capacity_model_is_below_each_electrode():
    # The paired-electrode gravimetric capacity is the harmonic combination → below either electrode.
    m = _battery_eval({"anode": "silicon", "cathode": "LiFePO4", "electrolyte": "carbonate_liquid"})
    assert m["feasible"] is True
    assert m["capacity_mAh_per_g"] < 160.0  # limited by the smaller (cathode) capacity


# --------------------------------------------------------------------------- #
# Novelty — the anti-remix filter: a re-derived textbook design is not a new invention.
# --------------------------------------------------------------------------- #
def test_known_art_reads_as_non_novel():
    eng = _engine()
    for baseline in _BATTERY_BASELINES:
        classic = DesignCandidate("battery", dict(baseline))
        assert eng._novelty(classic) < eng.novelty_floor, f"known art kept as novel: {baseline}"


def test_repeated_invention_is_not_counted_novel_again():
    eng = _engine()
    rep1 = eng.invent(generations=4, population=36)
    assert rep1.novel_kept > 0
    # The same engine, re-run: its archive already holds the first batch, so far fewer are kept again.
    rep2 = eng.invent(generations=4, population=36)
    assert rep2.novel_kept < rep1.novel_kept


# --------------------------------------------------------------------------- #
# Determinism + offline — same seed, same inventions; no network, no LLM, no sinks.
# --------------------------------------------------------------------------- #
def test_runs_are_deterministic_under_a_fixed_seed():
    a = [iv.summary for iv in _engine(seed=42).invent(4, 30).inventions]
    b = [iv.summary for iv in _engine(seed=42).invent(4, 30).inventions]
    assert a == b and len(a) > 0


def test_domains_can_be_restricted():
    rep = _engine().invent(generations=3, population=18, domains=("algorithm",))
    assert set(rep.by_domain).issubset({"algorithm"})
    assert all(iv.domain == "algorithm" for iv in rep.inventions)


# --------------------------------------------------------------------------- #
# Serialisation — the report and its parts round-trip to plain JSON-able dicts.
# --------------------------------------------------------------------------- #
def test_report_serialises_to_a_json_able_dict():
    import json

    rep = _engine().invent(generations=3, population=24)
    assert isinstance(rep, InventionReport)
    d = rep.to_dict()
    json.dumps(d)  # must not raise
    assert d["novel_kept"] == rep.novel_kept
    assert d["generated"] >= d["feasible"] >= d["novel_kept"]
    assert isinstance(d["inventions"], list)
    if rep.inventions:
        iv = rep.inventions[0]
        assert isinstance(iv, Invention)
        assert iv.to_dict()["improvement"] > 1.0


def test_score_is_novelty_times_bounded_gain():
    rep = _engine().invent(generations=2, population=18)
    for iv in rep.inventions:
        gain = 1.0 - (1.0 / iv.improvement)
        assert abs(iv.score() - round(iv.novelty * max(0.0, gain), 6)) < 1e-9


def test_total_inventions_accumulates_across_runs():
    eng = _engine()
    eng.invent(3, 30)
    first = eng.total_inventions
    assert first > 0
    eng.invent(3, 30)
    assert eng.total_inventions >= first


def test_all_three_named_domains_are_present():
    assert DOMAINS == ("battery", "aircraft", "algorithm")
