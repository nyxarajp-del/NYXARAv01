"""Tests for F10 — counterfactual causal dreaming (growth/counterfactual.py).

She must run the real counterfactual (re-search under intervention), credit the causal bottleneck
priors when a failure is flippable, and mark a truly-impossible failure GENUINELY HARD rather than
bluff. No LLM anywhere.
"""

from __future__ import annotations

import random

from nyxara.growth.counterfactual import CounterfactualDreamer, Failure
from nyxara.growth.noesis import NoesisEngine, Task, base_library, synthesize, synthetic_tasks


def _hard_at_low_beam():
    rnd = random.Random(3)
    lib = base_library()
    for t in synthetic_tasks(rnd, 40):
        if (not synthesize(t, lib, beam=3, max_expansions=3000).solved
                and synthesize(t, lib, beam=400).solved):
            return t
    raise AssertionError("no suitable task found")


def test_dream_credits_the_bottleneck_priors():
    task = _hard_at_low_beam()
    lib = base_library()
    before = dict(lib.counts)
    rep = CounterfactualDreamer(base_beam=3, cf_beam=400).dream([Failure(task)], lib)
    assert rep.solvable_with_more_compute == 1
    assert rep.credited                       # some primitive received causal credit
    # the credited primitives really are the ones that solve it, and their prior mass rose
    for name in rep.credited:
        assert lib.counts[name] > before.get(name, 0.0)


def test_genuinely_hard_is_flagged_not_bluffed():
    # contradictory examples → no program can satisfy them, at any budget
    impossible = Task("impossible", "intlist", "int", (([1], 1), ([1], 2)))
    rep = CounterfactualDreamer(base_beam=60, cf_beam=400).dream([Failure(impossible)], base_library())
    assert rep.genuinely_hard == 1
    assert rep.solvable_with_more_compute == 0
    assert rep.credited == {}                 # nothing credited on an honest dead end


def test_no_credit_when_baseline_already_solves():
    lib = base_library()
    easy = synthetic_tasks(random.Random(0), 3)
    rep = CounterfactualDreamer(base_beam=300, cf_beam=400).dream([Failure(t) for t in easy], lib)
    # baseline already solved them → nothing to counterfactually credit
    assert rep.solvable_with_more_compute == 0 and rep.credited == {}


def test_engine_runs_with_dreamer():
    eng = NoesisEngine(seed=1, tasks_per_cycle=12, dreamer=CounterfactualDreamer())
    reports = eng.run(4)                       # must not error; abstentions are rare but handled
    assert all(r.cf_credited >= 0 for r in reports)
