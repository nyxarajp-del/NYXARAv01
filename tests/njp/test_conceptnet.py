"""A relation named the way its source names it is a relation no question can reach.

`Grounder._predicate` normalises spelling and folds synonyms; it does not split CamelCase.
Measured: `_predicate("IsA")` is `"isa"` and `_predicate("PartOf")` is `"partof"` — names that
appear in no affinity table, no general-answer set and no law set, so the fact is stored and
never read. A converter that emitted ConceptNet's own relation names would therefore report a
successful quarter-million-row load and leave her knowing nothing new.

So most of this file is about the map: that every relation it emits is one the rest of the
package can actually use, and that the two relations deliberately left out stay out. The rest is
about the filters, because a corpus filter that is wrong in the permissive direction is invisible
until something downstream breaks.

No test here opens a socket. The fixture is a handful of lines of ConceptNet-shaped TSV.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import prepare_conceptnet as cn  # noqa: E402

from nyxara.njp.core import _TRANSITIVE_PRIOR  # noqa: E402
from nyxara.njp.grounding import (  # noqa: E402
    _FUNCTIONAL, _GENERAL_ANSWER, _PREDICATE_AFFINITY, Grounder,
)

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "prepare_conceptnet.py"


def _row(rel: str, start: str, end: str, weight: float = 3.0) -> str:
    meta = json.dumps({"dataset": "/d/conceptnet/4/en", "weight": weight})
    return "\t".join([f"/a/[{rel}/,{start}/,{end}/]", rel, start, end, meta])


_DUMP = [
    _row("/r/IsA", "/c/en/sparrow", "/c/en/bird"),
    _row("/r/IsA", "/c/en/dog/n", "/c/en/animal"),            # a POS-tagged concept
    _row("/r/PartOf", "/c/en/wheel", "/c/en/car"),
    _row("/r/Causes", "/c/en/fire", "/c/en/heat"),
    _row("/r/HasProperty", "/c/en/sparrow", "/c/en/small"),
    _row("/r/CapableOf", "/c/en/sparrow", "/c/en/fly"),
    _row("/r/AtLocation", "/c/en/cat", "/c/en/house"),        # excluded on purpose
    _row("/r/RelatedTo", "/c/en/sparrow", "/c/en/feather"),   # excluded on purpose
    _row("/r/IsA", "/c/fr/moineau", "/c/fr/oiseau"),          # not English
    _row("/r/IsA", "/c/en/rumour", "/c/en/hearsay", 1.0),     # below the weight floor
    _row("/r/IsA", "/c/en/thing", "/c/en/thing"),             # self-edge
]


@pytest.fixture()
def dump(tmp_path):
    path = tmp_path / "assertions.csv"
    path.write_text("\n".join(_DUMP) + "\n", encoding="utf-8")
    return path


def _converted(dump, **kwargs):
    return list(cn.convert(str(dump), **kwargs))


# --------------------------------------------------------------------------- #
# The map
# --------------------------------------------------------------------------- #

def test_camelcase_relations_are_mapped_before_folding_not_after():
    """The trap this whole script exists to avoid, pinned at both ends."""
    assert Grounder._predicate("IsA") == "isa"
    assert Grounder._predicate("PartOf") == "partof"
    assert "isa" not in _GENERAL_ANSWER and "isa" not in _PREDICATE_AFFINITY
    # And the map never emits one.
    assert all(name.islower() and " " not in name for name in cn._RELATIONS.values())


def test_every_emitted_predicate_survives_the_fold_unchanged():
    """Folding must be a no-op on these names, or the map is describing an edge nobody stores."""
    for name in set(cn._RELATIONS.values()):
        assert Grounder._predicate(name) == name


def test_every_emitted_predicate_is_one_the_package_can_use():
    """Retrieval-reachable, or chainable by the Core. A future row cannot quietly be dead.

    Two kinds of usable, and both are real: a predicate retrieval can score when it could not
    read the question (`_GENERAL_ANSWER` / `_PREDICATE_AFFINITY`), or one `core._compose` can
    chain because it carries a transitivity prior.
    """
    reachable = set(_GENERAL_ANSWER) | set(_PREDICATE_AFFINITY)
    for table in _PREDICATE_AFFINITY.values():
        reachable |= set(table)
    usable = reachable | set(_TRANSITIVE_PRIOR)
    unusable = {name for name in cn._RELATIONS.values() if name not in usable}
    assert unusable <= {"capable_of", "has_property"}, (
        f"{sorted(unusable)} reach neither retrieval nor the Core — a fact stored under one is "
        "stored and never read")


def test_the_two_weakest_relations_are_documented_rather_than_assumed():
    """`capable_of` and `has_property` are kept for schema induction and are not question-reachable.

    Measured, and the reason no affinity row was asserted on a guess: the gap is in the question
    grammar, not in the affinity table. The fact is stored and `_lookup` finds it; what does not
    exist is a pattern that reads an English question into that predicate.
    """
    grounder = Grounder()
    from nyxara.njp.grounding import GroundedTriple
    grounder._assert(GroundedTriple(subject="sparrow", predicate="capable_of", object="fly",
                                    confidence=0.7, source="test"))
    assert [t.object for t in grounder._lookup("sparrow", "capable_of")] == ["fly"]
    assert grounder._read_question("what is sparrow capable of?") == ("sparrow capable of", "is_a")


def test_at_location_is_not_emitted_because_located_in_holds_one_value(dump):
    """ConceptNet lists dozens of locations per concept; `located_in` admits exactly one."""
    assert "located_in" in _FUNCTIONAL
    assert "/r/AtLocation" not in cn._RELATIONS
    assert all(row["predicate"] != "located_in" for row in _converted(dump))


def test_related_to_is_not_emitted_because_it_is_the_could_not_read_this_fallback(dump):
    assert Grounder._predicate("") == "related_to"
    assert "/r/RelatedTo" not in cn._RELATIONS
    assert all(row["predicate"] != "related_to" for row in _converted(dump))


# --------------------------------------------------------------------------- #
# The filters
# --------------------------------------------------------------------------- #

def test_the_english_edges_survive_and_carry_the_mapped_names(dump):
    got = {(r["subject"], r["predicate"], r["object"]) for r in _converted(dump)}
    assert ("sparrow", "is_a", "bird") in got
    assert ("wheel", "part_of", "car") in got
    assert ("fire", "causes", "heat") in got


def test_a_pos_tagged_concept_keeps_its_word_rather_than_being_dropped(dump):
    """`/c/en/dog/n` is `dog`. Dropping every sense-tagged edge would lose most of the dump."""
    assert ("dog", "is_a", "animal") in {
        (r["subject"], r["predicate"], r["object"]) for r in _converted(dump)}


def test_non_english_concepts_are_dropped(dump):
    assert all("moineau" not in r["subject"] for r in _converted(dump))


def test_edges_below_the_weight_floor_are_dropped(dump):
    assert all(r["subject"] != "rumour" for r in _converted(dump, min_weight=2.0))
    assert any(r["subject"] == "rumour" for r in _converted(dump, min_weight=0.5))


def test_a_self_edge_is_dropped(dump):
    assert all(r["subject"] != r["object"] for r in _converted(dump))


def test_confidence_rises_with_corroboration_and_is_capped(dump):
    """Weight counts independent contributors, so it is corroboration and not certainty."""
    rows = {r["subject"]: r["confidence"] for r in _converted(dump, min_weight=0.5)}
    assert rows["rumour"] < rows["sparrow"]
    assert max(rows.values()) <= 0.9


def test_a_long_surface_is_dropped():
    assert cn._surface("/c/en/" + "x" * 80, lang="en") is None
    assert cn._surface("/c/en/blue_whale", lang="en") == "blue whale"
    assert cn._surface("/c/fr/moineau", lang="en") is None


def test_the_counters_account_for_every_row_read(dump):
    counters = {}
    _converted(dump, counters=counters)
    dropped = sum(v for k, v in counters.items() if k.startswith("dropped_"))
    assert counters["read"] == counters["kept"] + dropped


# --------------------------------------------------------------------------- #
# Sampling and the CLI
# --------------------------------------------------------------------------- #

def test_a_capped_run_samples_rather_than_truncating():
    """The dump is sorted by relation, so a head-of-file cap is a different corpus, not a smaller
    one — the same argument `study.Corpus.load` makes."""
    rows = [{"subject": f"s{i}", "predicate": "is_a", "object": "thing"} for i in range(200)]
    got = cn._sample(iter(rows), cap=20, seed=0)
    assert len(got) == 20
    assert {r["subject"] for r in got} != {f"s{i}" for i in range(20)}


def test_the_same_seed_gives_the_same_corpus_twice():
    rows = [{"subject": f"s{i}", "predicate": "is_a", "object": "thing"} for i in range(200)]
    assert cn._sample(iter(rows), cap=20, seed=7) == cn._sample(iter(rows), cap=20, seed=7)


def test_the_converter_runs_as_a_subprocess_and_opens_no_socket(dump, tmp_path):
    """Mirrors `test_study.py`'s CLI test: `-m` executes the module body, imports do not."""
    out = tmp_path / "cn.jsonl.gz"
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--in", str(dump), "--out", str(out), "--cap", "0"],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    with gzip.open(out, "rt", encoding="utf-8") as handle:
        written = [json.loads(line) for line in handle if line.strip()]
    assert written and all({"subject", "predicate", "object", "confidence"} <= set(r)
                           for r in written)
    assert json.loads(proc.stdout)["dropped"]["language"] >= 1


def test_the_output_is_exactly_what_ingest_reads(dump, tmp_path):
    """The two halves of Stage 2 meet here: converter out, ingest in, no reshaping between."""
    from nyxara.njp.brain import NJPBrain
    from nyxara.njp.ingest import ingest_triples

    out = tmp_path / "cn.jsonl"
    cn._write(_converted(dump), str(out))
    brain = NJPBrain()
    report = ingest_triples(brain, out, source="conceptnet")
    assert report.asserted == len(_converted(dump))
    assert ("sparrow", "is_a") in brain.grounder.facts
    # No laws by default, and that default was measured: a commonsense corpus filling the
    # 512-relation universe cost `causal_prediction` its whole score. Asked for explicitly,
    # only `fire causes heat` is law-shaped here.
    assert report.laws == 0
    assert ingest_triples(NJPBrain(), out, source="conceptnet", to_world=True).laws == 1


def test_an_unmapped_relation_asked_for_by_name_is_refused(dump, tmp_path):
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--in", str(dump),
         "--out", str(tmp_path / "x.jsonl"), "--relation", "/r/NotAThing"],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2
    assert "/r/NotAThing" in proc.stderr


def test_a_missing_input_file_fails_rather_than_writing_an_empty_corpus(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--in", str(tmp_path / "nope.csv"),
         "--out", str(tmp_path / "x.jsonl")],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2
    assert not (tmp_path / "x.jsonl").exists()
