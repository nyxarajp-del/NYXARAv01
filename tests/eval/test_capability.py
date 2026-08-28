"""An eval that scores a system on what it was just taught is measuring the loader.

Every number the three corpora report is measured on the records that were loaded, which is the
right check for a loader and worth nothing as a capability estimate. This file is about the split
that makes the difference, and about the two ways a split can lie:

* holding out too much — the first version removed whole ``world_knowledge`` records, so the probe
  asked her to produce facts about a subject she had never heard of, and scored the correct refusal
  as failure;
* holding out the wrong claim — the generator appends ``<member> is_a <concept>`` last, so taking
  the final claim withheld the exact edge the generalisation probe needs to inherit through.

Both are pinned here. So is the distinction the surface rests on: ``asked`` counts only probes that
could be formed, and a family with no probes reports ``None`` rather than zero, because "she got it
wrong" and "this run could not ask" must never share a denominator.

`brain.think` is slow — the fabric settles at about forty turns a second — so the tests that need a
real run use a small slice and the full sweep lives behind the CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from nyxara.eval.capability import (
    CapabilityReport, ProbeResult, run, split,
)
from nyxara.njp.unified import load

_RECORDS = (Path(__file__).resolve().parents[2] / "nyxara" / "njp" / "data"
            / "world_unified.jsonl.gz")


@pytest.fixture(scope="module")
def records():
    return load(_RECORDS)


# --------------------------------------------------------------------------- #
# The split
# --------------------------------------------------------------------------- #
def test_the_split_is_deterministic(records):
    first = split(records, holdout=0.25, seed=0)
    second = split(records, holdout=0.25, seed=0)
    assert [r["id"] for r in first[1]] == [r["id"] for r in second[1]]


def test_a_different_seed_holds_out_different_records(records):
    _t0, held0 = split(records, holdout=0.25, seed=0)
    _t1, held1 = split(records, holdout=0.25, seed=1)
    assert {r["id"] for r in held0} != {r["id"] for r in held1}


def test_holdout_is_roughly_the_share_asked_for(records):
    _taught, held = split(records, holdout=0.25, seed=0)
    assert 0.18 < len(held) / len(records) < 0.32


def test_a_knowledge_subject_is_still_taught_when_a_claim_is_withheld(records):
    """Claim-level, not record-level: the subject stays known and one claim goes missing.

    Whole-record holdout scored ``relation`` at 0.002 and that was the *correct* answer to a
    question nobody should ask — a system that invented properties for a subject it had never
    heard of would be worse, not better.
    """
    taught, held = split(records, holdout=0.25, seed=0)
    taught_ids = {r["id"] for r in taught}
    split_ones = [r for r in held if r["id"] in taught_ids and r.get("knowledge")]
    assert split_ones, "no record was split at the claim level"
    for probe in split_ones[:20]:
        partial = next(r for r in taught if r["id"] == probe["id"])
        assert len(probe["knowledge"]) == 1
        assert probe["knowledge"][0] not in partial["knowledge"]
        assert partial["knowledge"], "the subject was left with nothing taught about it"


def test_the_withheld_claim_is_never_the_is_a_edge(records):
    """Otherwise the split removes the very edge the generalisation probe inherits through."""
    taught, held = split(records, holdout=0.25, seed=0)
    taught_ids = {r["id"] for r in taught}
    for probe in held:
        if probe["id"] not in taught_ids or not probe.get("knowledge"):
            continue
        withheld = probe["knowledge"][0]
        partial = next(r for r in taught if r["id"] == probe["id"])
        if any(len(c) > 1 and c[1] != "is_a" for c in partial["knowledge"]) and len(withheld) > 1:
            assert withheld[1] != "is_a", probe["id"]


# --------------------------------------------------------------------------- #
# The reported shape
# --------------------------------------------------------------------------- #
def test_nothing_asked_is_none_and_not_zero():
    """The distinction the whole report rests on."""
    assert ProbeResult(family="x").score is None
    assert ProbeResult(family="x", asked=1).score == 0.0
    assert ProbeResult(family="x", asked=2, correct=1).score == 0.5


def test_abstention_is_not_counted_wrong():
    probe = ProbeResult(family="x", asked=10, answered=4, correct=4)
    assert probe.score == 0.4
    assert probe.precision == 1.0


def test_the_surface_is_unweighted_over_families():
    """Weighted, ``recall`` would stand in for the whole surface because the fact corpus is large,
    and a system that only recalled would score as though it could also transfer."""
    report = CapabilityReport()
    report.probes = [ProbeResult(family="recall", asked=1000, correct=1000),
                     ProbeResult(family="transfer", asked=2, correct=0)]
    assert report.surface == 0.5


def test_a_family_with_no_probes_is_left_out_of_the_surface():
    report = CapabilityReport()
    report.probes = [ProbeResult(family="recall", asked=4, correct=2),
                     ProbeResult(family="transfer")]
    assert report.surface == 0.5


# --------------------------------------------------------------------------- #
# A real, small run
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def small():
    return run(records=_RECORDS, holdout=0.3, seed=0, limit=220)


def test_the_control_says_the_load_worked(small):
    """Without it a low held-out number could mean the load failed rather than that she cannot
    generalise, and those call for opposite work."""
    assert small.taught_recall is not None
    assert small.taught_recall > 0.8, small.taught_recall


def test_the_held_out_surface_is_lower_than_the_taught_recall(small):
    """If they matched, the split leaked. That is the only thing this assertion is checking."""
    assert small.surface is not None
    assert small.surface < small.taught_recall


def test_some_families_are_asked(small):
    families = {name for name, probe in small.by_family().items() if probe.asked}
    assert len(families) >= 2


def test_the_report_serialises(small):
    payload = small.to_dict()
    assert payload["taught"] > 0 and payload["held_out"] > 0
    assert set(payload) >= {"taught", "held_out", "taught_recall", "surface", "families"}


# --------------------------------------------------------------------------- #
# The planning family, and the control that proves it can fail
# --------------------------------------------------------------------------- #
def test_the_planning_probe_can_fail():
    """A probe that cannot report a wrong answer is not measuring anything.

    ``planning`` reads 1.00 on the corpus, and a perfect row is exactly the shape a tautological
    probe takes. So: invert the planner — make it choose the setting that moves the lever the
    *wrong* way — and require the row to collapse. If it does not, the probe is agreeing with the
    planner by construction and the number is worthless.
    """
    from nyxara.eval.capability import _probe_planning
    from nyxara.njp.brain import NJPBrain
    from nyxara.njp.rollout import Target
    from nyxara.njp.universe import InternalUniverse

    brain = NJPBrain()
    universe = brain.universe
    assert universe is not None
    universe.declare("field.fertiliser", "field.yield", sign=1)
    for i in range(8):
        dose = 10.0 * i
        universe.observe({"field.fertiliser": dose, "field.yield": 5.0 + 0.6 * dose},
                         order=["field.fertiliser", "field.yield"])

    record = [{"category": "experience",
               "law": [["field.fertiliser", "causes", "field.yield", "1"]]}]

    honest = _probe_planning(brain, record, "experience")
    assert honest.asked == 1 and honest.answered == 1 and honest.correct == 1

    # Now the control. The planner picks the worst-scoring candidate instead of the best, which
    # for a positive law means pushing the fertiliser down.
    planner = brain.rollout
    real_search = planner.search

    def inverted(target: Target, **kw):
        plan = real_search(target, **kw)
        if plan.candidates:
            plan.chosen = plan.candidates[-1]
        return plan

    planner.search = inverted                      # type: ignore[method-assign]
    try:
        broken = _probe_planning(brain, record, "experience")
    finally:
        planner.search = real_search                # type: ignore[method-assign]

    assert broken.asked == 1 and broken.answered == 1
    assert broken.correct == 0, "the probe accepted a plan that moves the lever the wrong way"


def test_the_planning_probe_skips_an_ambiguous_target():
    """Two laws reaching one variable is two right answers, and must not be asked about."""
    from nyxara.eval.capability import _probe_planning
    from nyxara.njp.brain import NJPBrain

    brain = NJPBrain()
    universe = brain.universe
    universe.declare("field.fertiliser", "shared.yield", sign=1)
    universe.declare("lab.catalyst", "shared.yield", sign=1)
    for i in range(8):
        universe.observe({"field.fertiliser": 10.0 * i, "shared.yield": 5.0 + 0.6 * (10.0 * i)},
                         order=["field.fertiliser", "shared.yield"])
        universe.observe({"lab.catalyst": 1.0 * i, "shared.yield": 5.0 + 6.0 * i},
                         order=["lab.catalyst", "shared.yield"])

    record = [{"category": "experience",
               "law": [["field.fertiliser", "causes", "shared.yield", "1"]]}]
    assert _probe_planning(brain, record, "experience").asked == 0


def test_the_planning_probe_skips_a_target_with_a_lever_behind_its_lever():
    """An upstream lever is a genuine second answer, so the pair is not well posed either."""
    from nyxara.eval.capability import _probe_planning
    from nyxara.njp.brain import NJPBrain

    brain = NJPBrain()
    universe = brain.universe
    universe.declare("sky.altitude", "air.temperature", sign=-1)
    universe.declare("air.temperature", "run.rate", sign=1)
    for i in range(8):
        altitude = 0.5 * i
        temperature = 20.0 - 6.0 * altitude
        universe.observe({"sky.altitude": altitude, "air.temperature": temperature,
                          "run.rate": 1.0 + 0.3 * temperature},
                         order=["sky.altitude", "air.temperature", "run.rate"])

    record = [{"category": "experience",
               "law": [["air.temperature", "causes", "run.rate", "1"]]}]
    assert _probe_planning(brain, record, "experience").asked == 0


def test_planning_is_scored_on_the_taught_set():
    """The arrow has to exist to be planned over; only the plan is held out."""
    from nyxara.eval.capability import _DERIVED_FAMILIES

    assert "planning" in _DERIVED_FAMILIES


# --------------------------------------------------------------------------- #
# The revision family, and the control that proves it can fail
# --------------------------------------------------------------------------- #
def _corrected_brain():
    """A brain taught one correction, through the path that states it as a correction."""
    import os
    os.environ.setdefault("NYXARA_NJP__EVOLVE_ENABLED", "false")
    from nyxara.njp.brain import NJPBrain
    from nyxara.njp.unified import absorb

    record = {"id": "probe", "category": "contradiction", "domain": "astronomy",
              "contradiction": [["deoxygenated blood is blue",
                                 "deoxygenated blood is dark red"]]}
    brain = NJPBrain()
    absorb(brain, [record], check=False)
    return brain, [record]


def test_the_revision_probe_can_fail():
    """Un-retire the claim and the row must collapse, or the probe is not reading anything.

    ``revision`` reads 1.00 on the corpus, and a perfect row is the shape a tautological probe
    takes. The control puts the retired claim back on the live shelf — which is exactly the
    failure ``Grounder._revise`` names, *"she would announce the clash and keep answering with the
    superseded fact"* — and requires the score to go with it.
    """
    from nyxara.eval.capability import _probe_revision

    brain, records = _corrected_brain()
    honest = _probe_revision(brain, records, "contradiction")
    assert honest.asked == 1 and honest.answered == 1 and honest.correct == 1

    for triples in brain.grounder.facts.values():
        for triple in triples:
            triple.superseded = False           # the correction is announced and not applied
    broken = _probe_revision(brain, records, "contradiction")
    assert broken.asked == 0 or broken.correct == 0, \
        "the probe scored a store that kept answering with the retired claim"


def test_the_revision_probe_reads_the_answer_not_the_flag():
    """A probe that checked `superseded` would be checking that its own cause ran."""
    import inspect

    from nyxara.eval import capability

    source = inspect.getsource(capability._probe_revision)
    assert "_ask(" in source, "the probe must go through the read path"


def test_revision_is_scored_on_the_taught_set():
    from nyxara.eval.capability import _DERIVED_FAMILIES

    assert "revision" in _DERIVED_FAMILIES


def test_a_correction_retires_the_claim_it_replaces():
    """The defect underneath the row: nought of eleven corpus corrections used to supersede."""
    brain, _records = _corrected_brain()
    objects = {t.object: t.superseded
               for triples in brain.grounder.facts.values() for t in triples}
    assert objects.get("blue") is True
    assert objects.get("dark red") is False
