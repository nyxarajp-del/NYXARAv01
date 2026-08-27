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
