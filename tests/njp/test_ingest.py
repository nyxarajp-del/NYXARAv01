"""A file of facts is not a conversation, and pushing it through one would be a lie about her day.

`seed_kinds` already made this argument at 137 rows. At corpus scale it stops being only about
honesty and becomes about whether the load finishes at all: the fabric settles and expands at
~40 turns/s, so a quarter-million triples through `think` is 1.7 hours, where `_assert` is
~54,000/s and flat. And `HoloMemory` links every write to the last eight keys, so a bulk load
routed through `think` would cross-link a million unrelated facts as though one conversation had
mentioned them together.

So the tests that matter here are mostly about what ingestion **does not** touch. The rest are
about the two things that silently make a stored fact useless: a predicate folded the wrong way,
and a cap that drops the tail without saying so.
"""

from __future__ import annotations

import gzip
import json

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.grounding import Grounder
from nyxara.njp.ingest import ingest_triples, stream_rows, stream_triples


def _write(path, rows, *, gz: bool = False):
    """Write rows as JSONL, optionally gzipped. Non-dict rows are written verbatim."""
    text = "\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n"
    if gz:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def _triples(n: int, *, predicate: str = "is_a"):
    return [{"subject": f"thing{i}", "predicate": predicate, "object": "animal"}
            for i in range(n)]


@pytest.fixture()
def brain():
    return NJPBrain()


# --------------------------------------------------------------------------- #
# What it must not touch
# --------------------------------------------------------------------------- #

def test_ingest_writes_no_episodic_memory_and_no_turn_count(brain, tmp_path):
    """The whole design decision, pinned. `HoloMemory` cross-links every write to the last eight
    keys, so a bulk load through `think` would make a million unrelated facts look co-occurrent."""
    before_traces = len(getattr(brain.memory, "traces", {}) or {})
    before_turns = brain.grounder.turns
    before_cells = brain.fabric.n_cells

    report = ingest_triples(brain, _write(tmp_path / "t.jsonl", _triples(50)))

    assert report.asserted == 50
    assert len(getattr(brain.memory, "traces", {}) or {}) == before_traces
    assert brain.grounder.turns == before_turns
    assert brain.fabric.n_cells == before_cells


def test_ingest_does_not_move_the_extraction_rate(brain, tmp_path):
    """`extraction_rate` is a statement about the parser, not about a file she was handed."""
    ingest_triples(brain, _write(tmp_path / "t.jsonl", _triples(30)))
    stats = brain.grounder.stats()
    assert stats["turns"] == 0
    assert stats["unparsed"] == 0


def test_a_failed_ingest_reports_what_it_managed_and_never_raises(brain, tmp_path):
    missing = tmp_path / "does_not_exist.jsonl"
    report = ingest_triples(brain, missing)
    assert report.asserted == 0
    assert report.read == 0


# --------------------------------------------------------------------------- #
# Folding — a stored fact the reasoner cannot see
# --------------------------------------------------------------------------- #

def test_ingested_predicates_are_folded_the_way_a_question_folds_them(brain, tmp_path):
    """`needs` is stored as `requires`, and a question asking "needs" reads `requires`.

    Without the fold the fact is written under a key no question ever forms — present in the
    store, invisible to every reader, and counted as a success by every report.
    """
    ingest_triples(brain, _write(tmp_path / "t.jsonl", [
        {"subject": "plant", "predicate": "needs", "object": "water"}]))
    assert ("plant", "requires") in brain.grounder.facts
    assert ("plant", "needs") not in brain.grounder.facts


def test_a_relation_named_the_way_its_source_names_it_is_still_folded(brain, tmp_path):
    """ConceptNet's own spelling. `_predicate("IsA")` is `"isa"` — a real trap, measured."""
    ingest_triples(brain, _write(tmp_path / "t.jsonl", [
        {"subject": "sparrow", "predicate": "IsA", "object": "bird"}]))
    stored = {predicate for _subject, predicate in brain.grounder.facts}
    assert stored == {"isa"}, (
        "folding does not split CamelCase — the converter must emit snake_case, and this test "
        "exists so that fact is discovered here rather than after a 250,000-row load")


def test_allow_is_checked_after_folding(brain, tmp_path):
    """The caller names relations the way this package does; the file names them as its source did."""
    report = ingest_triples(brain, _write(tmp_path / "t.jsonl", [
        {"subject": "plant", "predicate": "needs", "object": "water"},
        {"subject": "plant", "predicate": "is_a", "object": "organism"}]),
        allow={"requires"})
    assert report.asserted == 1
    assert ("plant", "requires") in brain.grounder.facts


# --------------------------------------------------------------------------- #
# Caps, dedup, and saying so
# --------------------------------------------------------------------------- #

def test_a_file_larger_than_the_cap_stops_at_the_cap_and_says_so(brain, tmp_path):
    """A load that silently dropped its tail would report the same numbers as one that finished."""
    report = ingest_triples(brain, _write(tmp_path / "t.jsonl", _triples(100)), max_facts=40)
    assert report.asserted == 40
    assert report.capped is True


def test_an_uncapped_load_does_not_claim_it_was_capped(brain, tmp_path):
    report = ingest_triples(brain, _write(tmp_path / "t.jsonl", _triples(10)), max_facts=40)
    assert report.capped is False


def test_the_same_triple_twice_is_asserted_once(brain, tmp_path):
    rows = [{"subject": "aag", "predicate": "causes", "object": "garmi"}] * 3
    report = ingest_triples(brain, _write(tmp_path / "t.jsonl", rows))
    assert (report.asserted, report.duplicate) == (1, 2)


def test_a_row_below_the_confidence_floor_is_skipped(brain, tmp_path):
    report = ingest_triples(brain, _write(tmp_path / "t.jsonl", [
        {"subject": "a", "predicate": "is_a", "object": "b", "confidence": 0.2},
        {"subject": "c", "predicate": "is_a", "object": "d", "confidence": 0.9}]),
        min_confidence=0.5)
    assert (report.asserted, report.skipped) == (1, 1)


def test_a_malformed_line_is_skipped_and_the_rest_still_land(brain, tmp_path):
    path = _write(tmp_path / "t.jsonl", [
        {"subject": "a", "predicate": "is_a", "object": "b"},
        "{not json at all",
        {"subject": "c", "predicate": "is_a", "object": "d"},
        {"subject": "", "predicate": "is_a", "object": "e"}])
    report = ingest_triples(brain, path)
    assert report.asserted == 2


# --------------------------------------------------------------------------- #
# Fan-out
# --------------------------------------------------------------------------- #

def test_the_concept_layer_is_fed_in_batches_rather_than_one_list(brain, tmp_path):
    """`ConceptGenesis` has a real capacity with FIFO eviction and per-observation similarity
    work, so one 250,000-item list would evict most of what it had just been told."""
    seen = []
    original = brain.genesis.observe_triples
    brain.genesis.observe_triples = lambda triples: (  # type: ignore[method-assign]
        seen.append(len(list(triples))) or original(triples))
    ingest_triples(brain, _write(tmp_path / "t.jsonl", _triples(25)), batch=10)
    assert seen == [10, 10, 5]


def test_law_predicates_reach_the_world_model_and_ordinary_relations_do_not(brain, tmp_path):
    """A ConceptNet edge is never an event: nothing happened at a time."""
    report = ingest_triples(brain, _write(tmp_path / "t.jsonl", [
        {"subject": "aag", "predicate": "causes", "object": "garmi"},
        {"subject": "sparrow", "predicate": "is_a", "object": "bird"}]), to_world=True)
    assert report.laws == 1
    assert any(link.cause == "aag" and link.effect == "garmi"
               for link in brain.world.links(causal_only=False))


def test_the_law_cap_is_respected(brain, tmp_path):
    rows = [{"subject": f"c{i}", "predicate": "causes", "object": f"e{i}"} for i in range(20)]
    report = ingest_triples(brain, _write(tmp_path / "t.jsonl", rows),
                            to_world=True, max_laws=5, batch=3)
    assert report.laws == 5
    assert report.asserted == 20


def test_a_bulk_load_does_not_touch_the_causal_skeleton_unless_asked(brain, tmp_path):
    """The default, and it was measured rather than chosen.

    `InternalUniverse` holds 512 relations. With this on, 8,000 ingested facts stated 2,312 laws,
    `sync_from_world` imported 512 of them and filled it, and the benchmark's own `water → growth`
    never got in — `causal_prediction` went 1.00 to 0.00 and `to_world=False` restored it.
    A commonsense corpus is testimony about the world in general, not observation of hers.
    """
    report = ingest_triples(brain, _write(tmp_path / "t.jsonl", [
        {"subject": "aag", "predicate": "causes", "object": "garmi"}]))
    assert report.laws == 0
    assert brain.world.links(causal_only=True) == []


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #

def test_a_checkpoint_runs_at_the_interval_it_was_given(brain, tmp_path):
    calls = []
    ingest_triples(brain, _write(tmp_path / "t.jsonl", _triples(25)),
                   checkpoint=calls.append, checkpoint_every=10)
    assert calls == [10, 20]


def test_a_failing_checkpoint_costs_durability_and_not_the_run(brain, tmp_path):
    def explode(_n):
        raise OSError("disk full")

    report = ingest_triples(brain, _write(tmp_path / "t.jsonl", _triples(25)),
                            checkpoint=explode, checkpoint_every=10)
    assert report.asserted == 25


def test_a_gzipped_corpus_reads_the_same_as_a_plain_one(brain, tmp_path):
    report = ingest_triples(brain, _write(tmp_path / "t.jsonl.gz", _triples(12), gz=True))
    assert report.asserted == 12


def test_a_plain_json_list_reads_too(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps(_triples(4)), encoding="utf-8")
    assert len(list(stream_triples(path))) == 4


def test_stream_rows_raises_on_a_corrupt_line_when_asked_to(tmp_path):
    """The one thing that differs between this module's reader and the study corpus's."""
    path = _write(tmp_path / "t.jsonl", ["{broken"])
    assert list(stream_rows(path)) == []
    with pytest.raises(ValueError):
        list(stream_rows(path, on_error="raise"))


def test_the_report_carries_provenance(brain, tmp_path):
    report = ingest_triples(brain, _write(tmp_path / "t.jsonl", _triples(6)),
                            source="conceptnet")
    assert report.source == "conceptnet"
    assert len(report.digest) == 64
    assert report.subjects == 6
    assert report.predicates == {"is_a": 6}
    assert all(t.source == "conceptnet"
               for triples in brain.grounder.facts.values() for t in triples)


def test_a_row_carrying_its_own_confidence_overrides_the_default(brain, tmp_path):
    ingest_triples(brain, _write(tmp_path / "t.jsonl", [
        {"subject": "a", "predicate": "is_a", "object": "b", "confidence": 0.91}]),
        default_confidence=0.4)
    assert brain.grounder.facts[("a", "is_a")][0].confidence == pytest.approx(0.91)


# --------------------------------------------------------------------------- #
# The seed file still works, through the shared path
# --------------------------------------------------------------------------- #

def test_seeding_kinds_still_lands_through_the_one_bulk_path(brain):
    """`seed_kinds` delegates here now, so a fix to folding or dedup reaches both at once."""
    from nyxara.njp.study import seed_kinds

    report = seed_kinds(brain)
    assert report.asserted > 0
    assert report.kinds > 0
    assert all(t.source == "seed"
               for triples in brain.grounder.facts.values() for t in triples)


def test_a_bare_grounder_is_enough_to_ingest_into(tmp_path):
    """No concept layer, no world model, no fabric — the fact store is the only requirement."""
    holder = type("_H", (), {"grounder": Grounder()})()
    report = ingest_triples(holder, _write(tmp_path / "t.jsonl", _triples(5)))
    assert report.asserted == 5
