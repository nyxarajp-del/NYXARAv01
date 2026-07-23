"""Stage F — Epistemic Auto-Evolution / Synthetic Hypothesis Discovery.

She generates never-seen concurrent-fault scenarios, self-hypothesizes, and tests them in a modelled
sandbox. These tests prove: scenarios combine simultaneous stressors, novelty is tracked (no re-runs),
a compound failure mode invisible to single-stressor tests is discovered, and a probe that raises is
recorded as a discovered failure (not an engine crash).
"""

from __future__ import annotations

from nyxara.growth.synthetic_hypothesis import (
    FaultScenario,
    SyntheticHypothesisEngine,
    SyntheticReport,
)


def test_scenarios_combine_simultaneous_stressors():
    eng = SyntheticHypothesisEngine(seed=1)
    scenarios = eng.generate(10)
    assert len(scenarios) == 10
    for s in scenarios:
        assert isinstance(s, FaultScenario)
        assert 2 <= len(s.stressors) <= eng.max_stressors   # genuinely concurrent, not single-fault
        assert s.signature


def test_novelty_is_tracked_no_reruns():
    eng = SyntheticHypothesisEngine(seed=2)
    r1 = eng.hunt(n=8)
    assert isinstance(r1, SyntheticReport)
    assert r1.novel == r1.tested >= 1
    # re-generating from the same seeded rng yields the same signatures → all already-seen
    eng.rng.seed(2)
    r2 = eng.hunt(n=8)
    assert r2.novel == 0   # nothing new — she does not waste effort re-running known corners


def test_discovers_compound_failure_invisible_to_single_faults():
    eng = SyntheticHypothesisEngine(seed=7)
    # a scenario the built-in modelled system fails: saturated CPU + lossy net + high concurrency
    bad = FaultScenario(stressors={"cpu_spike": 0.97, "packet_drop": 0.5, "concurrency": 12.0})
    res = eng.test(bad, eng._default_probe)
    assert res.survived is False
    assert "fault" in res.detail.lower() or "fail" in res.detail.lower()
    # and a benign single stressor survives (proving the failure is genuinely compound)
    ok = FaultScenario(stressors={"cpu_spike": 0.97, "latency_ms": 100.0})
    assert eng.test(ok, eng._default_probe).survived is True


def test_hunt_finds_at_least_one_failing_edge_case_over_a_batch():
    eng = SyntheticHypothesisEngine(seed=13)
    # over enough Monte-Carlo draws, the compound failure region is hit at least once
    found = 0
    for _ in range(20):
        rep = eng.hunt(n=12)
        found += rep.failed
    assert found >= 1
    assert eng.report()["known_signatures"] > 0


def test_probe_that_raises_is_a_discovered_failure_not_a_crash():
    eng = SyntheticHypothesisEngine(seed=3)

    def exploding_probe(cfg):
        raise RuntimeError("deadlock")

    rep = eng.hunt(probe=exploding_probe, n=4)
    assert rep.failed == rep.tested >= 1
    assert all(not f.survived for f in rep.failures)
    assert any("RuntimeError" in f.detail for f in rep.failures)


def test_hypothesis_is_falsifiable_and_concrete():
    eng = SyntheticHypothesisEngine(seed=5)
    s = FaultScenario(stressors={"cpu_spike": 0.8, "mem_spike": 0.9})
    hyp = eng.hypothesize(s)
    assert hyp.statement and hyp.variable == "survives_fault"
    assert hyp.prediction is True
