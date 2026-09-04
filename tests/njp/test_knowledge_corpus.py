"""A question that does not parse is a fact she cannot be asked for, however well it is stored.

`test_conceptnet.py` makes this argument about relation *names*. A hand-written QA corpus has the
same defect available to it one layer up: the triples can be perfect and every question in the
exam can still read as something else, in which case the score measures the question grammar and
not what she knows. So the load-bearing test here is :func:`test_every_askable_template_reads_back`
— it puts each template through the live `Grounder._read_question` and demands the exact
``(subject, predicate)`` pair back, so no template can be added on a guess.

The rest is about the two invariants the generator promises and the exam depends on: every
question is answerable from a triple in the same build, and the artefacts in ``nyxara/njp/data``
are the ones the KB currently produces.

No test here opens a socket. The end-to-end test builds a bare `NJPBrain` and loads the shipped
triples, which is 3,745 assertions in about 160 ms.
"""

from __future__ import annotations

import gzip
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import prepare_knowledge_corpus as kc  # noqa: E402

from nyxara.njp.brain import NJPBrain  # noqa: E402
from nyxara.njp.core import _TRANSITIVE_PRIOR  # noqa: E402
from nyxara.njp.grounding import (  # noqa: E402
    _FUNCTIONAL, _GENERAL_ANSWER, _PREDICATE_AFFINITY, Grounder,
)
from nyxara.njp.ingest import ingest_triples  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "prepare_knowledge_corpus.py"
_TRIPLES = _ROOT / "nyxara" / "njp" / "data" / "world_knowledge.jsonl.gz"
_QA = _ROOT / "nyxara" / "njp" / "data" / "world_qa.jsonl.gz"

#: A two-word probe on purpose. A one-word subject hides the whole class of failures this file
#: exists to catch: "when does melting occur" parses, "when does solar eclipse occur" did not.
_PROBE = "solar eclipse"


@pytest.fixture(scope="module")
def grounder():
    return Grounder()


@pytest.fixture(scope="module")
def rows():
    return kc.parse()


@pytest.fixture(scope="module")
def shipped_triples():
    with gzip.open(_TRIPLES, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


@pytest.fixture(scope="module")
def shipped_qa():
    with gzip.open(_QA, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


# --------------------------------------------------------------------------- #
# The templates, against the live question grammar
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("predicate", sorted(kc._ASKABLE))
def test_every_askable_template_reads_back(grounder, predicate):
    """`_read_question` must return this predicate and this subject, not something adjacent.

    Both halves matter. A template that returns the right predicate for the wrong subject looks
    fine in a spot check and answers nothing, which is precisely how `known_for` failed.
    """
    template, _end = kc._ASKABLE[predicate]
    question = template.format(s=_PROBE)
    assert grounder._read_question(question) == (_PROBE, predicate), question


def test_known_for_is_unreachable_and_therefore_absent(grounder):
    """The docstring's reason for leaving `known_for` out, kept honest.

    `_QUESTION_PATTERNS` has a line for it. The `purpose` line above it matches the trailing
    " for" first, so the pattern is dead. If that ordering is ever fixed upstream this test fails
    and the predicate can be added.
    """
    assert "known_for" not in kc._ASKABLE
    assert grounder._read_question("what is marie curie known for?") != ("marie curie", "known_for")


@pytest.mark.parametrize("predicate", sorted(kc._GRAPH_ONLY))
def test_graph_only_predicates_carry_no_question(predicate):
    assert predicate not in kc._ASKABLE
    assert predicate not in kc._ANSWER


def test_part_of_really_has_no_question_form(grounder):
    """Why `part_of` is graph-only: every natural phrasing reads as `is_a` on a mangled subject."""
    for question in ("what is a wheel part of?", "what is the part of of wheel?"):
        assert grounder._read_question(question)[1] != "part_of", question


@pytest.mark.parametrize("predicate", sorted(set(kc._ASKABLE) | set(kc._GRAPH_ONLY)))
def test_predicate_survives_folding(grounder, predicate):
    """`Grounder._predicate` must return the name unchanged, or the stored key is not this one."""
    assert grounder._predicate(predicate) == predicate


@pytest.mark.parametrize("predicate", sorted(set(kc._ASKABLE) | set(kc._GRAPH_ONLY)))
def test_predicate_is_reachable_by_something(predicate):
    """Every emitted relation is retrievable, chainable, or asked for by name — no dead edges."""
    affinity_names = set(_PREDICATE_AFFINITY) | {
        name for values in _PREDICATE_AFFINITY.values()
        for name in (values if isinstance(values, dict) else [values])
    }
    # `njp/predator.py` is the fifth reader, added at V.39. It goes after an explanation offering
    # two answers and asks whether the store says they exclude each other — so `excludes` is
    # retrievable by an organ rather than by a question, which is a reader this list did not have
    # a category for. Adding the organ is the honest move; loosening the assertion would not be.
    from nyxara.njp.predator import EXCLUDES

    reachable = (predicate in _GENERAL_ANSWER
                 or predicate in affinity_names
                 or predicate in _TRANSITIVE_PRIOR
                 or predicate in _FUNCTIONAL
                 or predicate in kc._ASKABLE
                 or predicate in EXCLUDES)
    assert reachable, f"{predicate} is stored by nothing that can read it back"


# --------------------------------------------------------------------------- #
# The parser's invariants
# --------------------------------------------------------------------------- #
def _kb(tmp_path, text: str) -> Path:
    directory = tmp_path / "kb"
    directory.mkdir(exist_ok=True)
    (directory / "probe.kb").write_text(text, encoding="utf-8")
    return directory


def test_unknown_predicate_is_refused(tmp_path):
    with pytest.raises(kc.KBError, match="unreachable predicate"):
        kc.parse(_kb(tmp_path, "sparrow | flies_like=a bird\n"))


def test_self_edge_is_refused(tmp_path):
    with pytest.raises(kc.KBError, match="self-edge"):
        kc.parse(_kb(tmp_path, "bird | is_a=Bird\n"))


def test_duplicate_triple_is_refused(tmp_path):
    with pytest.raises(kc.KBError, match="duplicate"):
        kc.parse(_kb(tmp_path, "sparrow | is_a=bird\nsparrow | is_a=bird\n"))


def test_second_value_for_a_functional_relation_is_refused(tmp_path):
    """One capital per country. A second is a contradiction `_revise` fires on, not an addition.

    This is the check that lets this KB carry capitals at all where `prepare_conceptnet.py` had to
    drop `/r/AtLocation` wholesale — and it caught a real mistake while the shipped file was being
    written: `isaac newton | author=the laws of motion | author=Principia Mathematica`, which had
    the relation backwards.
    """
    with pytest.raises(kc.KBError, match="one value"):
        kc.parse(_kb(tmp_path, "France | capital=Paris\nFrance | capital=Lyon\n"))


def test_malformed_line_is_refused(tmp_path):
    with pytest.raises(kc.KBError, match="not 'predicate=object'"):
        kc.parse(_kb(tmp_path, "sparrow | a bird\n"))


def test_confidence_default_and_override(tmp_path):
    parsed = kc.parse(_kb(tmp_path, "bird | capable_of=fly@0.75 | is_a=vertebrate\n"))
    emitted = {row["predicate"]: row["confidence"] for row in kc.triples(parsed, confidence=0.85)}
    assert emitted["capable_of"] == 0.75
    assert emitted["is_a"] == 0.85


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("phrase,article", [
    ("gas giant", "a"),          # head-final: the head is "giant", not the s-ending "gas"
    ("noble gas", "a"),          # "gas" is singular despite the s
    ("unit of force", "a"),      # written with a vowel, said with a consonant
    ("hour", "an"),              # the mirror case
    ("alkali metal", "an"),      # a modified mass noun is countable again
    ("water", ""),               # a bare one is not
    ("amino acids", ""),         # already plural
    ("Albert Einstein", ""),     # already determined
    ("the base of a food chain", ""),
])
def test_article_selection(phrase, article):
    assert kc._article(phrase) == article


def test_answers_are_capitalised_without_flattening_acronyms():
    assert kc._sentence("is_a", "DNA", ["nucleic acid"]) == "DNA is a nucleic acid."
    # The subject stays bare. Putting an article on it too would need the article rule to run on
    # words like "thermodynamics" and "evolution", which are not in `_MASS` and are not countable
    # either — "a thermodynamics is a branch of physics" is a worse sentence to train on than the
    # dictionary register this keeps.
    assert kc._sentence("is_a", "sparrow", ["bird"]) == "Sparrow is a bird."


def test_multiple_objects_become_one_answer():
    """One question per relation, every object in it.

    Asking "what are the properties of copper?" once per object would put three questions in the
    exam with a third of the answer each, and grade a complete answer wrong twice.
    """
    said = kc._sentence("has_property", "copper", ["ductile", "conductive", "reddish brown"])
    assert said == "Copper is ductile, conductive and reddish brown."


# --------------------------------------------------------------------------- #
# The shipped knowledge base
# --------------------------------------------------------------------------- #
def test_shipped_kb_parses_without_warnings():
    warnings: list = []
    parsed = kc.parse(warnings=warnings)
    assert warnings == []
    assert len(parsed) > 3_000
    assert len(kc.domains()) >= 15


def test_every_question_is_answerable_from_a_triple_in_the_same_build(rows):
    """The invariant the exam rests on: no question in the QA file has no fact behind it."""
    held = {(row["subject"], row["predicate"]) for row in rows}
    grounder = Grounder()
    for pair in kc.qa_pairs(rows):
        # Lowercased first, because that is what `Grounder.answer` does before it reads a
        # question. Reading the raw string here would test a path production never takes.
        subject, predicate = grounder._read_question(pair["question"].lower())
        assert (subject, predicate) in held, pair["question"]


def test_shipped_artefacts_are_what_the_kb_currently_produces(rows, shipped_triples, shipped_qa):
    """Editing the KB and forgetting to rebuild is the failure this catches."""
    assert list(kc.triples(rows)) == shipped_triples
    assert list(kc.qa_pairs(rows)) == shipped_qa


def test_triples_carry_exactly_the_four_keys_ingest_reads(shipped_triples):
    for row in shipped_triples[:200]:
        assert set(row) == {"subject", "predicate", "object", "confidence"}
        assert 0.0 < row["confidence"] <= 1.0


def test_qa_carries_exactly_the_two_keys_the_tutor_reads(shipped_qa):
    for row in shipped_qa[:200]:
        assert set(row) == {"question", "answer"}
        assert row["question"].strip() and row["answer"].strip()


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
def test_she_answers_her_own_corpus_and_never_answers_it_wrongly(shipped_qa):
    """Load the triples, ask the questions back, and check both halves of the outcome.

    The second half is the one worth having. Abstention is a normal outcome here — a relation with
    several objects comes back CONFLICTING, because `answer` will not pick one of two equally
    supported readings — and `study.Tutor` counts that separately from being wrong. What must
    never happen is a confident answer that is not in the gold text, and on the shipped corpus
    that count is zero.
    """
    brain = NJPBrain()
    report = ingest_triples(brain, str(_TRIPLES), source="world_knowledge")
    assert report.asserted > 3_000 and not report.capped

    sample = random.Random(20260826).sample(shipped_qa, 400)
    answered, wrong = 0, []
    for pair in sample:
        said = brain.grounder.answer(pair["question"])
        if not said.text:
            continue                       # UNKNOWN or CONFLICTING — an abstention, not an error
        answered += 1
        if said.text.lower() not in pair["answer"].lower():
            wrong.append((pair["question"], said.text, pair["answer"]))
    assert wrong == []
    assert answered >= len(sample) * 0.7, f"only {answered}/{len(sample)} answered"


def test_a_multi_word_subject_answers_the_same_as_a_one_word_one():
    """The regression guard for the `_answer_polar` fix that shipping this corpus turned up.

    "when does melting occur" answered and "when does solar eclipse occur" did not, because
    `compile_meaning` read the second as ``polar(solar, eclipse, occur)`` and `_answer_polar` runs
    before the ordinary path. Both must answer, and a real polar question must still be polar.
    """
    brain = NJPBrain()
    ingest_triples(brain, str(_TRIPLES), source="world_knowledge")
    for question in ("When does melting occur?", "When does solar eclipse occur?",
                     "When does antibiotic resistance occur?"):
        assert brain.grounder.answer(question).text, question
    brain.grounder.ground("a sparrow needs water")
    assert brain.grounder.answer("Do sparrows need water?").polar is True


# --------------------------------------------------------------------------- #
# The CLI
# --------------------------------------------------------------------------- #
def test_cli_check_reports_without_writing(tmp_path):
    result = subprocess.run([sys.executable, str(_SCRIPT), "--check"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["facts"] > 3_000 and report["qa_pairs"] > 2_000
    assert not list(tmp_path.iterdir())


def test_cli_writes_both_artefacts(tmp_path):
    triples, qa = tmp_path / "t.jsonl.gz", tmp_path / "q.jsonl"
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--domain", "physics",
         "--triples", str(triples), "--qa", str(qa)],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["triples_written"] == report["facts"]
    assert report["qa_written"] == report["qa_pairs"]
    with gzip.open(triples, "rt", encoding="utf-8") as handle:
        assert json.loads(handle.readline())["subject"]
    assert json.loads(qa.read_text(encoding="utf-8").splitlines()[0])["question"]


def test_cli_refuses_an_unknown_domain(tmp_path):
    result = subprocess.run([sys.executable, str(_SCRIPT), "--check", "--domain", "phlogiston"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 2
    assert "no such domain" in result.stderr
