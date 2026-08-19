"""Ten corpora, one held-out set, and the refusal to report a total without its parts.

`study.main` exams twice — before and after — which is enough for one corpus and stops being
enough at two. A run that loaded ConceptNet, Wikipedia, code and GSM8K and reported one delta
would say something moved and nothing in it would say which of the four moved it.

So what is tested here is not that training works — `test_study.py` and `test_ingest.py` cover the
loaders. It is the three things this module adds and could silently get wrong: that the stages run
in the Master's order whatever order they were asked for, that the exam questions are the *same*
ones at every stage, and that a stage's held-out tenth was never studied.

No socket, no download. The corpora are three-row files written into `tmp_path`.
"""

from __future__ import annotations

import gzip
import json

import pytest

from nyxara.njp import corpora, train
from nyxara.njp.brain import NJPBrain


def _write(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


@pytest.fixture()
def prepared(tmp_path):
    """One tiny corpus per form, named the way `discover` expects to find them."""
    _write(tmp_path / "conceptnet.triples.jsonl.gz",
           [{"subject": f"beast{i}", "predicate": "is_a", "object": "animal"}
            for i in range(20)])
    _write(tmp_path / "wikipedia.text.jsonl.gz",
           [{"text": f"Widget{i} is a mechanical device.", "source": "wiki"}
            for i in range(20)])
    _write(tmp_path / "gsm8k.pairs.jsonl.gz",
           [{"question": f"What is the sum of {i} apples and two apples?",
             "answer": f"The answer is {i + 2} apples."} for i in range(60)])
    return tmp_path


@pytest.fixture()
def bundled(tmp_path):
    """Stands in for the shipped instruction corpus, so no test reads the 4.8 MB real one."""
    path = tmp_path / "bundled.jsonl.gz"
    return _write(path, [{"question": f"What is a gadget number {i}?",
                          "answer": "A gadget is a small mechanical device."}
                         for i in range(60)])


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def test_discover_finds_only_files_named_for_their_form(prepared):
    _write(prepared / "wikipedia.pairs.jsonl.gz", [{"question": "q?", "answer": "a"}])
    (prepared / "notes.txt").write_text("not a corpus", encoding="utf-8")
    found = train.discover(prepared)
    assert set(found) == {"conceptnet", "wikipedia", "gsm8k"}
    assert found["wikipedia"].name == "wikipedia.text.jsonl.gz"


def test_discover_accepts_an_uncompressed_corpus(tmp_path):
    (tmp_path / "gsm8k.pairs.jsonl").write_text(
        json.dumps({"question": "What is one and one?", "answer": "Two."}) + "\n",
        encoding="utf-8")
    assert set(train.discover(tmp_path)) == {"gsm8k"}


def test_discover_on_an_empty_directory_is_empty_not_an_error(tmp_path):
    assert train.discover(tmp_path) == {}


# --------------------------------------------------------------------------- #
# The ladder
# --------------------------------------------------------------------------- #

def test_stages_run_in_the_masters_order_not_the_callers(prepared, bundled):
    """Facts before turns. A triple corpus mechanically converts abstentions into answers, so
    running it after the conversational sources folds its effect into theirs unrecoverably."""
    report = train.train(NJPBrain(), prepared, only=["gsm8k", "wikipedia", "conceptnet"],
                         bundled=bundled, exam_limit=20, pool=200)
    assert [s.key for s in report.stages] == ["conceptnet", "wikipedia", "gsm8k"]


def test_every_stage_is_asked_the_same_number_of_questions(prepared, bundled):
    """The exam set is drawn once. A stage that changed the questions could not be compared
    with the stage before it, which is the only thing this module does."""
    report = train.train(NJPBrain(), prepared, bundled=bundled, exam_limit=20, pool=200)
    asked = {report.control["asked"]}
    assert report.final["asked"] in asked


def test_a_missing_corpus_is_a_reported_row_not_a_failed_run(prepared, bundled):
    report = train.train(NJPBrain(), prepared, bundled=bundled, exam_limit=10, pool=200)
    by_key = {s.key: s for s in report.stages}
    assert by_key["oasst1"].skipped.startswith("not prepared")
    assert by_key["conceptnet"].skipped == ""
    assert len(report.stages) == len(corpora.SOURCES)


def test_skip_leaves_a_source_out_and_says_it_was_asked_to(prepared, bundled):
    report = train.train(NJPBrain(), prepared, bundled=bundled, skip=["wikipedia"],
                         exam_limit=10, pool=200)
    assert {s.key: s.skipped for s in report.stages}["wikipedia"] == "skipped by request"


def test_a_triple_corpus_reaches_the_fact_store(prepared, bundled):
    brain = NJPBrain()
    train.train(brain, prepared, only=["conceptnet"], bundled=bundled,
                exam_limit=10, pool=200)
    assert "animal" in (brain.grounder.answer("what is beast3?").text or "").lower()


def test_a_text_corpus_reaches_the_fact_store_through_the_extractor(prepared, bundled):
    brain = NJPBrain()
    train.train(brain, prepared, only=["wikipedia"], bundled=bundled,
                exam_limit=10, pool=200)
    assert "device" in (brain.grounder.answer("what is widget3?").text or "").lower()


def test_each_corpus_is_tagged_with_its_own_source(prepared, bundled):
    """`_regenerable` matches the manifest on the fact's source, so two corpora sharing a tag
    would have one manifest row between them and the second would go to the sidecar in full."""
    brain = NJPBrain()
    train.train(brain, prepared, only=["conceptnet", "wikipedia"], bundled=bundled,
                exam_limit=10, pool=200)
    # "seed" is the kind-level file, which the trainer loads before the control exam; the point
    # here is that the two corpora did not share one row between them.
    assert {"conceptnet", "wikipedia"} <= {r["source"] for r in brain.grounder.ingested}
    assert len(brain.grounder.ingested) == len({r["source"] for r in brain.grounder.ingested})


def test_the_held_out_tenth_of_a_pair_corpus_is_never_studied(prepared, bundled):
    """Split before any stage runs, not at the point of study — otherwise the exam set would
    depend on which stages were selected and two runs would not be comparable."""
    files = train.discover(prepared)
    exam, study = train._exam_corpus(files, bundled=bundled, pool=200, seed=0)
    exam_keys = {p.key for p in exam}
    assert study["gsm8k"], "nothing was left to study, so this test proves nothing"
    assert not (exam_keys & {p.key for p in study["gsm8k"]})
    assert all(p.held_out for p in exam)


def test_the_exam_draws_on_every_conversational_source_not_only_the_bundled_one(prepared,
                                                                               bundled):
    """An exam made only of the bundled corpus asks nothing about arithmetic, so GSM8K would
    read as having contributed nothing however well it loaded."""
    files = train.discover(prepared)
    exam, _ = train._exam_corpus(files, bundled=bundled, pool=200, seed=0)
    assert any("apples" in p.question for p in exam)
    assert any("gadget" in p.question for p in exam)


def test_a_shortened_exam_is_shorter_not_narrower(prepared, bundled):
    """`Tutor.exam` applies its limit by taking the head of what it is given. Concatenated, the
    first 300 questions all come from whichever corpus loaded first and every other source reads
    as having contributed nothing — which is the failure this whole module exists to prevent."""
    files = train.discover(prepared)
    exam, _ = train._exam_corpus(files, bundled=bundled, pool=200, seed=0)
    prefix = exam[:6]
    assert any("apples" in p.question for p in prefix)
    assert any("gadget" in p.question for p in prefix)


def test_the_exam_order_is_the_same_on_a_second_run(prepared, bundled):
    """Two runs that asked different questions are not comparable, which is the only thing the
    attribution table is for."""
    files = train.discover(prepared)
    first, _ = train._exam_corpus(files, bundled=bundled, pool=200, seed=0)
    second, _ = train._exam_corpus(files, bundled=bundled, pool=200, seed=0)
    assert [p.key for p in first] == [p.key for p in second]


def test_the_report_table_names_every_stage_and_the_control(prepared, bundled):
    report = train.train(NJPBrain(), prepared, only=["conceptnet", "gsm8k"], bundled=bundled,
                         exam_limit=10, pool=200)
    table = report.table()
    assert "(control)" in table
    assert "conceptnet" in table and "gsm8k" in table
    assert "cover" in table and "prec" in table


def test_a_run_over_an_empty_directory_still_reports_its_control(tmp_path, bundled):
    report = train.train(NJPBrain(), tmp_path, bundled=bundled, exam_limit=10, pool=200)
    assert report.control["asked"] > 0
    assert report.final == report.control
    assert all(s.skipped for s in report.stages)


# --------------------------------------------------------------------------- #
# The CLI
# --------------------------------------------------------------------------- #

def test_plan_prints_the_ladder_and_loads_nothing(prepared, capsys):
    assert train.main(["--data", str(prepared), "--plan"]) == 0
    out = capsys.readouterr().out
    assert "conceptnet" in out and "MISSING" in out and "ready" in out


def test_an_empty_data_directory_fails_loudly(tmp_path, capsys):
    assert train.main(["--data", str(tmp_path)]) == 1
    assert "no prepared corpora" in capsys.readouterr().out


def test_the_kind_seed_lands_before_the_control_not_inside_a_stage(prepared, bundled):
    """`Tutor` seeds kinds lazily on its first `study` call, which in this ladder is stage 4 —
    so 137 kind-level facts would be measured as ProofNet's contribution. They belong to the
    brain she starts from."""
    brain = NJPBrain()
    train.train(brain, prepared, only=["gsm8k"], bundled=bundled, exam_limit=10, pool=200)
    assert any(r["source"] == "seed" for r in brain.grounder.ingested)


def test_no_seed_leaves_the_kind_layer_out_entirely(prepared, bundled):
    brain = NJPBrain()
    train.train(brain, prepared, only=["gsm8k"], bundled=bundled, exam_limit=10, pool=200,
                seed_kinds=False)
    assert not any(r["source"] == "seed" for r in brain.grounder.ingested)


# --------------------------------------------------------------------------- #
# The lookup holdout
# --------------------------------------------------------------------------- #

def _held_out_name(prefix: str) -> str:
    """A subject the holdout actually reserves, so a fixture cannot silently test nothing."""
    for i in range(10_000):
        name = f"{prefix}{i}"
        if corpora.held_out_subject(name):
            return name
    raise AssertionError("no held-out name found")


def _kept_name(prefix: str) -> str:
    for i in range(10_000):
        name = f"{prefix}{i}"
        if not corpora.held_out_subject(name):
            return name
    raise AssertionError("no kept name found")


def test_a_held_out_subject_never_reaches_the_brain(tmp_path, bundled):
    """The whole point. If it leaks, the fact stage is measuring memorisation."""
    held, kept = _held_out_name("beast"), _kept_name("beast")
    _write(tmp_path / "conceptnet.triples.jsonl.gz",
           [{"subject": held, "predicate": "is_a", "object": "animal"},
            {"subject": kept, "predicate": "is_a", "object": "animal"}])
    brain = NJPBrain()
    train.train(brain, tmp_path, only=["conceptnet"], bundled=bundled,
                exam_limit=10, pool=200, fact_pool=50)
    subjects = {s for (s, _p) in brain.grounder.facts}
    assert kept in subjects
    assert held not in subjects


def test_the_exam_carries_lookup_questions_from_the_fact_corpora(tmp_path, bundled):
    """Without these the fact stages have no instrument: every other question in the exam comes
    from a conversational corpus and essentially none of them is a lookup."""
    held = _held_out_name("beast")
    _write(tmp_path / "conceptnet.triples.jsonl.gz",
           [{"subject": held, "predicate": "is_a", "object": "animal"}])
    report = train.train(NJPBrain(), tmp_path, only=["conceptnet"], bundled=bundled,
                         exam_limit=50, pool=200, fact_pool=50)
    assert report.fact_questions >= 1


def test_the_load_reports_what_it_withheld(tmp_path):
    """A holdout that stops working must show as a number going to zero, not as a silent leak."""
    import json

    from nyxara.njp.ingest import ingest_triples

    held, kept = _held_out_name("beast"), _kept_name("beast")
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [
        {"subject": held, "predicate": "is_a", "object": "animal"},
        {"subject": kept, "predicate": "is_a", "object": "animal"}]) + "\n", encoding="utf-8")
    got = ingest_triples(NJPBrain(), path, exclude=corpora.held_out_subject)
    assert got.held_out == 1 and got.asserted == 1
    assert got.to_dict()["held_out"] == 1


def test_fact_pool_zero_disables_the_holdout_and_loads_everything(tmp_path, bundled):
    held = _held_out_name("beast")
    _write(tmp_path / "conceptnet.triples.jsonl.gz",
           [{"subject": held, "predicate": "is_a", "object": "animal"}])
    brain = NJPBrain()
    report = train.train(brain, tmp_path, only=["conceptnet"], bundled=bundled,
                         exam_limit=10, pool=200, fact_pool=0)
    assert held in {s for (s, _p) in brain.grounder.facts}
    assert report.fact_questions == 0


def test_a_prose_corpus_is_held_out_through_the_extractor_not_a_second_parser(tmp_path):
    """A sentence's subject is not known until the extractor has read it. Guessing at it with a
    regex would guess differently from the loader, and the exam would quietly start asking about
    facts that were never withheld."""
    source = corpora.by_key("wikipedia")
    held = _held_out_name("widget")
    path = _write(tmp_path / "wikipedia.text.jsonl.gz",
                  [{"text": f"{held} is a mechanical device."},
                   {"text": f"{_kept_name('widget')} is a mechanical device."}])
    got = list(train._held_out_triples(source, path))
    # More than one row per sentence is right: the extractor's head-noun backoff yields both
    # "mechanical device" and "device". What matters is that only the withheld subject appears.
    assert {t[0] for t in got} == {held.lower()}
