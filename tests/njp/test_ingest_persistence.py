"""Every save rewrote a corpus that was already sitting on disk.

`Grounder.to_dict` wrote every fact into the single `njp.json` sidecar with no cap. Measured at
188 bytes per serialised fact, a 250,000-row ConceptNet load makes that file ~47 MB and every
autosave a stall — for a copy of a file that is already on disk, hashed and immutable.

`Manifold.to_dict` set the precedent, refusing to persist a matrix that living rebuilds
*approximately*. A corpus is rebuilt *exactly*, which is a stronger guarantee than the one that
argument was first made on. So a bulk fact is left out of the sidecar and replayed on load.

The whole risk of that trade is in one place and most of this file is about it: a bulk fact that
a **conversation later corrected** is no longer what the file says. If the replay put the
withdrawn version back live, she would answer tomorrow with what she was corrected about today —
the same defect `load_dict` already argues against for dropping the `superseded` flag, one layer
up. And a corpus that has gone missing or changed must be *reported*, because a store that
quietly shrank overnight and said nothing is worse than one that never left out a fact.
"""

from __future__ import annotations

import json

import pytest

from nyxara.njp.grounding import GroundedTriple, Grounder
from nyxara.njp.ingest import ingest_triples


def _corpus(tmp_path, rows, name="cn.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


_ROWS = [
    {"subject": "sparrow", "predicate": "is_a", "object": "bird"},
    {"subject": "crow", "predicate": "is_a", "object": "bird"},
    {"subject": "salmon", "predicate": "is_a", "object": "fish"},
]


@pytest.fixture()
def loaded(tmp_path):
    """A grounder taught one small corpus, through the real bulk path."""
    grounder = Grounder()
    path = _corpus(tmp_path, _ROWS)
    ingest_triples(type("_H", (), {"grounder": grounder})(), path, source="conceptnet")
    return grounder, path


def _reloaded(grounder):
    fresh = Grounder()
    fresh.load_dict(json.loads(json.dumps(grounder.to_dict())))
    return fresh


# --------------------------------------------------------------------------- #
# The trade
# --------------------------------------------------------------------------- #

def test_a_regenerable_fact_is_not_written_into_the_json_sidecar(loaded):
    grounder, _ = loaded
    assert grounder.to_dict()["facts"] == []
    assert [row["source"] for row in grounder.to_dict()["ingested"]] == ["conceptnet"]


def test_the_sidecar_does_not_grow_with_the_size_of_the_corpus(tmp_path):
    """Ten times the corpus, the same sidecar. That is the entire point of the manifest."""
    def sidecar_bytes(n: int) -> int:
        grounder = Grounder()
        rows = [{"subject": f"s{i}", "predicate": "is_a", "object": "thing"} for i in range(n)]
        ingest_triples(type("_H", (), {"grounder": grounder})(),
                       _corpus(tmp_path, rows, f"c{n}.jsonl"), source="conceptnet")
        return len(json.dumps(grounder.to_dict()))

    small, large = sidecar_bytes(20), sidecar_bytes(200)
    assert large < small * 1.2, f"sidecar went {small} → {large} bytes for 10x the corpus"


def test_a_reloaded_store_has_every_fact_back(loaded):
    grounder, _ = loaded
    fresh = _reloaded(grounder)
    assert [t.object for t in fresh._lookup("sparrow", "is_a")] == ["bird"]
    assert {t.subject for t in fresh._lookup_inverse("bird", "is_a")} == {"sparrow", "crow"}


def test_the_indexes_are_built_over_replayed_facts_too(loaded):
    """The replay goes through `_assert`, so `_index` runs; `_reindex` then runs over everything."""
    grounder, _ = loaded
    fresh = _reloaded(grounder)
    assert fresh._known >= {"sparrow", "crow", "salmon", "bird", "fish"}
    assert fresh._by_subject["sparrow"] == {"is_a"}


# --------------------------------------------------------------------------- #
# The one that would have been a real defect
# --------------------------------------------------------------------------- #

def test_a_bulk_fact_that_a_conversation_revised_survives_the_round_trip(loaded):
    """The retraction must outlive the reload, or the replay undoes yesterday's correction."""
    grounder, _ = loaded
    stored = grounder._lookup("sparrow", "is_a")[0]
    stored.superseded = True
    grounder._assert(GroundedTriple(subject="sparrow", predicate="is_a", object="raptor",
                                    confidence=0.9, source="master"))

    written = grounder.to_dict()["facts"]
    assert {(f["object"], f["superseded"]) for f in written} == {("bird", True), ("raptor", False)}

    fresh = _reloaded(grounder)
    live = [t.object for t in fresh._lookup("sparrow", "is_a")]
    assert live == ["raptor"], f"the replay resurrected a retracted fact: {live}"


def test_a_contested_bulk_fact_is_also_written_out_in_full(loaded):
    grounder, _ = loaded
    grounder._lookup("crow", "is_a")[0].contested = True
    assert [f["object"] for f in grounder.to_dict()["facts"]] == ["bird"]


def test_a_fact_she_was_told_is_never_treated_as_regenerable(loaded):
    """Only sources named in the manifest are replayable. A conversation is not a file."""
    grounder, _ = loaded
    grounder._assert(GroundedTriple(subject="jay", predicate="has_name", object="jay",
                                    confidence=0.9, source="master"))
    assert [f["subject"] for f in grounder.to_dict()["facts"]] == ["jay"]


# --------------------------------------------------------------------------- #
# Honest degradation
# --------------------------------------------------------------------------- #

def test_a_missing_ingest_source_is_reported_rather_than_silently_forgotten(loaded):
    grounder, path = loaded
    state = json.loads(json.dumps(grounder.to_dict()))
    path.unlink()

    fresh = Grounder()
    fresh.load_dict(state)
    assert fresh._lookup("sparrow", "is_a") == []
    gaps = fresh.stats()["ingest_gaps"]
    assert gaps and "missing" in gaps[0] and "conceptnet" in gaps[0]


def test_a_corpus_that_changed_under_her_is_reported_rather_than_trusted(loaded):
    """A corpus edited since she read it is not the corpus she was taught."""
    grounder, path = loaded
    state = json.loads(json.dumps(grounder.to_dict()))
    path.write_text(json.dumps({"subject": "sparrow", "predicate": "is_a",
                                "object": "reptile"}) + "\n", encoding="utf-8")

    fresh = Grounder()
    fresh.load_dict(state)
    assert fresh._lookup("sparrow", "is_a") == []
    assert any("changed" in gap for gap in fresh.stats()["ingest_gaps"])


def test_stats_names_the_corpora_she_is_standing_on(loaded):
    grounder, _ = loaded
    got = grounder.stats()["ingested"]
    assert [row["source"] for row in got] == ["conceptnet"]
    assert got[0]["count"] == 3
    assert grounder.stats()["ingest_gaps"] == []


# --------------------------------------------------------------------------- #
# Manifest bookkeeping
# --------------------------------------------------------------------------- #

def test_reloading_the_same_corpus_twice_leaves_one_manifest_row(tmp_path):
    grounder = Grounder()
    holder = type("_H", (), {"grounder": grounder})()
    path = _corpus(tmp_path, _ROWS)
    ingest_triples(holder, path, source="conceptnet")
    second = ingest_triples(holder, path, source="conceptnet")
    assert len(grounder.ingested) == 1
    assert second.asserted == 0 and second.duplicate == 3, (
        "re-ingesting a file she already read must be idempotent, not additive")


def test_a_second_corpus_gets_its_own_manifest_row(tmp_path):
    grounder = Grounder()
    holder = type("_H", (), {"grounder": grounder})()
    ingest_triples(holder, _corpus(tmp_path, _ROWS), source="conceptnet")
    ingest_triples(holder, _corpus(tmp_path, [
        {"subject": "oak", "predicate": "is_a", "object": "tree"}], "seed.jsonl"), source="seed")
    assert {row["source"] for row in grounder.ingested} == {"conceptnet", "seed"}


def test_a_load_that_stored_nothing_does_not_claim_a_corpus(tmp_path):
    grounder = Grounder()
    ingest_triples(type("_H", (), {"grounder": grounder})(),
                   _corpus(tmp_path, [], "empty.jsonl"), source="conceptnet")
    assert grounder.ingested == []


def test_the_brain_round_trips_through_its_own_sidecar(tmp_path):
    """The path the kernel actually uses: `NJPBrain.to_dict` → JSON → `load_dict`."""
    from nyxara.njp.brain import NJPBrain

    brain = NJPBrain()
    ingest_triples(brain, _corpus(tmp_path, _ROWS), source="conceptnet")
    state = json.loads(json.dumps(brain.to_dict()))
    assert state["grounding"]["facts"] == []

    fresh = NJPBrain()
    fresh.load_dict(state)
    assert [t.object for t in fresh.grounder._lookup("salmon", "is_a")] == ["fish"]
