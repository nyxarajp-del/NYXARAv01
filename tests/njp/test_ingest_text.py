"""Prose is most of what the world writes down, and none of it is a conversation with her.

`ingest_triples` already argues why a corpus must not go through `think`. A prose loader is where
that argument is easiest to break, because there *is* a method that reads a sentence and stores
what it finds — `Grounder.ground` — and using it looks like reuse rather than like a mistake. It
would increment `turns`, file the result under `unparsed`, and run `_contradicts`/`_revise` on
every extraction. The first two make `stats()["extraction_rate"]` a description of a downloaded
file instead of of how well she reads the Master; the third spends the run retracting Wikipedia
against itself in file order.

So these tests are mostly about what the loader must not touch, and about the one thing a text
corpus needs that a triple corpus does not: a manifest row that says which reader can restore it.
"""

from __future__ import annotations

import json

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.ingest import ingest_text, stream_statements


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def brain():
    return NJPBrain()


@pytest.fixture()
def corpus(tmp_path):
    return _write(tmp_path / "prose.jsonl", [
        {"text": "A photon is an elementary particle.", "source": "wiki/Photon"},
        {"text": "Boiling requires heat.", "source": "wiki/Boiling"},
        {"text": "Anarchism is a political philosophy.", "source": "wiki/Anarchism"},
        {"text": "Qwerty zxcvb mnbvc plugh.", "source": "wiki/Nonsense"},
    ])


# --------------------------------------------------------------------------- #
# What it must not touch
# --------------------------------------------------------------------------- #

def test_no_turn_counter_no_episodic_memory_no_fabric(brain, corpus):
    """The whole design decision. A file she was handed is not a day she had."""
    before = (brain.grounder.turns, brain.grounder.unparsed, brain.grounder.grounded_turns,
              len(getattr(brain.memory, "traces", {}) or {}), brain.fabric.n_cells)
    ingest_text(brain, corpus)
    after = (brain.grounder.turns, brain.grounder.unparsed, brain.grounder.grounded_turns,
             len(getattr(brain.memory, "traces", {}) or {}), brain.fabric.n_cells)
    assert before == after


def test_the_files_extraction_rate_is_the_files_own_number(brain, corpus):
    """Reported on the report, never on `Grounder.stats`, because they answer different
    questions about different subjects."""
    report = ingest_text(brain, corpus)
    assert report.read == 4 and report.parsed == 3 and report.unparsed == 1
    assert report.extraction_rate == 0.75
    assert brain.grounder.stats()["extraction_rate"] != 0.75 or brain.grounder.turns == 0


def test_bulk_facts_do_not_revise_each_other(brain, tmp_path):
    """`located_in` is functional, and an encyclopedia states a person's places in several
    articles. Through the conversational path the run would end holding whichever arrived last."""
    path = _write(tmp_path / "places.jsonl", [
        {"text": "Ada lives in London."},
        {"text": "Ada lives in Paris."},
    ])
    ingest_text(brain, path)
    stored = [t for triples in brain.grounder.facts.values() for t in triples]
    assert stored, "nothing was extracted, so this test proves nothing"
    assert not any(t.superseded for t in stored)


# --------------------------------------------------------------------------- #
# What it does
# --------------------------------------------------------------------------- #

def test_extracted_facts_are_answerable(brain, corpus):
    ingest_text(brain, corpus)
    assert "particle" in (brain.grounder.answer("what is a photon?").text or "").lower()


def test_provenance_survives_but_the_load_tag_leads(brain, corpus):
    """`_regenerable` matches the manifest on `source`, so a triple tagged only with its article
    URL would never match the load that produced it and would go to the sidecar in full."""
    ingest_text(brain, corpus, source="wikipedia")
    sources = {t.source for triples in brain.grounder.facts.values() for t in triples}
    assert all(s.startswith("wikipedia") for s in sources)
    assert any("wiki/Photon" in s for s in sources)


def test_reingesting_the_same_file_asserts_nothing_new(brain, corpus):
    first = ingest_text(brain, corpus)
    second = ingest_text(brain, corpus)
    assert first.asserted > 0
    assert second.asserted == 0
    assert second.duplicate >= first.asserted


def test_max_statements_caps_the_read_and_says_so(brain, corpus):
    report = ingest_text(brain, corpus, max_statements=2)
    assert report.read == 2 and report.capped is True


def test_max_facts_caps_the_write_and_says_so(brain, corpus):
    report = ingest_text(brain, corpus, max_facts=1)
    assert report.asserted == 1 and report.capped is True


def test_allow_filters_on_the_folded_predicate(brain, corpus):
    report = ingest_text(brain, corpus, allow={"requires"})
    assert set(report.predicates) == {"requires"}


def test_a_missing_file_reports_rather_than_raises(brain, tmp_path):
    report = ingest_text(brain, tmp_path / "absent.jsonl")
    assert report.read == 0 and report.asserted == 0


def test_stream_statements_skips_rows_with_no_statement(tmp_path):
    path = _write(tmp_path / "mixed.jsonl", [
        {"text": "A photon is an elementary particle."}, {"source": "x"}, {"text": "hi"}])
    assert [s for s, _ in stream_statements(path)] == ["A photon is an elementary particle."]


# --------------------------------------------------------------------------- #
# Persistence — the half that fails silently
# --------------------------------------------------------------------------- #

def test_the_manifest_records_the_form_so_replay_uses_the_right_reader(brain, corpus):
    ingest_text(brain, corpus, source="wikipedia")
    row = [r for r in brain.grounder.ingested if r["source"] == "wikipedia"][0]
    assert row["form"] == "text"


def test_a_saved_text_corpus_comes_back_after_a_reload(brain, corpus):
    """Replaying a text corpus with the triple reader restores nothing, and reports identically
    to a corpus that had nothing to restore. That is the failure this pins."""
    ingest_text(brain, corpus, source="wikipedia")
    fresh = NJPBrain()
    fresh.grounder.load_dict(brain.grounder.to_dict())
    assert "particle" in (fresh.grounder.answer("what is a photon?").text or "").lower()
    assert fresh.grounder.ingest_gaps == []


def test_a_manifest_row_without_a_form_still_replays_as_triples(brain, tmp_path):
    """Every manifest written before prose corpora existed lacks the key."""
    path = _write(tmp_path / "t.jsonl",
                  [{"subject": "dog", "predicate": "is_a", "object": "animal"}])
    from nyxara.njp.ingest import ingest_triples
    ingest_triples(brain, path, source="conceptnet")
    for row in brain.grounder.ingested:
        row.pop("form", None)
    fresh = NJPBrain()
    fresh.grounder.load_dict(brain.grounder.to_dict())
    assert fresh.grounder.ingest_gaps == []
    assert "animal" in (fresh.grounder.answer("what is a dog?").text or "").lower()
