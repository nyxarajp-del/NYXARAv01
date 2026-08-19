"""NYXARA · njp/train.py — the ten corpora in the Master's order, and one exam between each (🎓).

:mod:`nyxara.njp.study` teaches her from *one* corpus and exams her twice, before and after. The
Master named ten, in priority order, and the moment there is more than one source the two-exam
shape stops being enough: a run that loaded ConceptNet, Wikipedia, code and GSM8K and then
reported one before/after delta would say something moved, and nothing in it would say **which of
the four moved it**. A number you cannot attribute is a number you cannot act on — you cannot
drop the source that hurt, because you do not know there was one.

So this module is the same two-exam idea taken seriously: **the same held-out questions, asked
after every stage.** One stage per corpus, one delta each, one table at the end.

Ten priorities, eleven corpora: priority 8 is "GSM8K + MATH", and they are kept as two stages
rather than concatenated because they are two different measurements — grade-school word problems
and competition algebra fail in different ways, and a merged row could not say which.

**The order is load-bearing, and it is the Master's order for a reason that outlives him naming
it.** Facts before turns. A corpus of triples mechanically converts abstentions into answers —
coverage rises whatever it loaded, good or bad — so running the fact sources after the
conversational ones would fold that mechanical rise into their measured deltas and there would be
no way back out. Loaded first, its effect is one row of the table on its own.

**Read ``precision`` down the table, not ``coverage``.** This is the same warning
:func:`nyxara.njp.study.main` gives about a single ingest, and it matters more with ten of them.
Every load raises coverage. Only a load that raised ``correct`` alongside it added knowledge; one
that raised coverage while precision fell added *noise*, and it is now answering questions it
used to have the sense to abstain on. The table prints both, side by side, per stage, so that
failure has to be visible.

**What this does not do.** It does not fetch anything — ``scripts/prepare_datasets.py`` does that,
offline, once. It does not convert anything — :mod:`nyxara.njp.corpora` decides what each source
becomes. It does not have a loader of its own — :func:`~nyxara.njp.ingest.ingest_triples`,
:func:`~nyxara.njp.ingest.ingest_text` and :class:`~nyxara.njp.study.Tutor` are the only three
ways into this brain and this module picks between them. What is here that is nowhere else is the
*order*, the *exam between stages*, and the refusal to report a total without its parts.

Pure standard library. No network, no LLM.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
import random
from pathlib import Path
from typing import (Any, Dict, Iterator, List, Optional, Sequence, Set,
                    Tuple)

from nyxara.njp import corpora

__all__ = ["StageResult", "TrainReport", "discover", "train", "main"]


@dataclass
class StageResult:
    """One corpus loaded, and what the held-out exam read immediately afterwards."""

    key: str = ""
    priority: int = 0
    form: str = ""
    path: str = ""
    rows: int = 0
    report: Dict[str, Any] = field(default_factory=dict)
    coverage: float = 0.0
    precision: float = 0.0
    accuracy: float = 0.0
    on_topic: float = 0.0
    d_coverage: float = 0.0
    d_precision: float = 0.0
    skipped: str = ""
    ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "priority": self.priority, "form": self.form,
                "path": self.path, "rows": self.rows,
                "coverage": round(self.coverage, 4), "precision": round(self.precision, 4),
                "accuracy": round(self.accuracy, 4), "on_topic": round(self.on_topic, 4),
                "d_coverage": round(self.d_coverage, 4),
                "d_precision": round(self.d_precision, 4),
                "skipped": self.skipped, "ms": round(self.ms, 1),
                "report": self.report}


@dataclass
class TrainReport:
    """The whole ladder: the control, every stage, and what the curriculum says at the end."""

    exam_size: int = 0
    #: How many of the held-out questions are lookups drawn from the fact corpora themselves.
    #: Zero means the fact stages have no instrument and their rows measure nothing.
    fact_questions: int = 0
    #: The control: the same lookups, on pairs that *were* loaded. Never a generalisation score.
    #: If this is near zero the corpus is stored and unreachable, and every delta above it is
    #: noise about a store nothing can read.
    recall: Dict[str, Any] = field(default_factory=dict)
    stages: List[StageResult] = field(default_factory=list)
    control: Dict[str, Any] = field(default_factory=dict)
    final: Dict[str, Any] = field(default_factory=dict)
    curriculum: Dict[str, Any] = field(default_factory=dict)
    ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"exam_size": self.exam_size,
                "fact_questions": self.fact_questions,
                "stages": [s.to_dict() for s in self.stages],
                "control": self.control, "final": self.final,
                "recall": self.recall,
                "curriculum": self.curriculum, "ms": round(self.ms, 1)}

    def table(self) -> str:
        """The attribution table. The point of the whole module, in one block of text."""
        # `prec` and `hit` are both printed, for the reason `study._hit` gives: F1 asks whether
        # the reply was right *and* complete, and a lookup gold answer here lists up to eight
        # objects, so a correct one-word answer scores about 0.22 and reads as wrong. Neither is
        # the score alone and the gap between them is itself the reading.
        head = (f"{'#':>2}  {'source':<13} {'form':<8} {'rows':>8}  "
                f"{'cover':>6} {'Δ':>7}  {'prec':>6} {'Δ':>7}  {'hit':>6}")
        lines = [head, "-" * len(head)]
        lines.append(f"{'':>2}  {'(control)':<13} {'':<8} {'':>8}  "
                     f"{self.control.get('coverage', 0.0):>6.3f} {'':>7}  "
                     f"{self.control.get('precision_when_answered', 0.0):>6.3f} {'':>7}  "
                     f"{self.control.get('on_topic_when_answered', 0.0):>6.3f}")
        for stage in self.stages:
            if stage.skipped:
                lines.append(f"{stage.priority:>2}  {stage.key:<13} {stage.form:<8} "
                             f"{'—':>8}  {stage.skipped}")
                continue
            lines.append(
                f"{stage.priority:>2}  {stage.key:<13} {stage.form:<8} {stage.rows:>8}  "
                f"{stage.coverage:>6.3f} {stage.d_coverage:>+7.3f}  "
                f"{stage.precision:>6.3f} {stage.d_precision:>+7.3f}  "
                f"{stage.on_topic:>6.3f}")
        return "\n".join(lines)


def discover(directory: Any) -> Dict[str, Path]:
    """Which of the ten sources have a prepared file in ``directory``.

    Matched on :attr:`nyxara.njp.corpora.Source.filename` rather than on a glob, so a stray file
    cannot be loaded as a corpus and a corpus in the wrong shape — ``wikipedia.pairs.jsonl.gz``,
    say — is simply not found. The form is part of the name because the form decides the loader,
    and a file that lies about its form would be read by the wrong one and quietly yield nothing.
    """
    root = Path(str(directory))
    found: Dict[str, Path] = {}
    for source in corpora.SOURCES:
        for name in (source.filename, source.filename[:-3]):   # .gz and plain
            candidate = root / name
            if candidate.exists():
                found[source.key] = candidate
                break
    return found


def _count_rows(path: Path) -> int:
    """How many rows a prepared file holds. Cheap enough to be worth having in the table."""
    from nyxara.njp.ingest import stream_rows
    try:
        return sum(1 for _ in stream_rows(path))
    except Exception:  # noqa: BLE001
        return 0


def _exam_corpus(files: Dict[str, Path], *, bundled: Any, pool: int, seed: int,
                 extra: Optional[Sequence[List[Any]]] = None
                 ) -> Tuple[List[Any], Dict[str, List[Any]]]:
    """The exam set, and the study half of every pair corpus, split before anything is taught.

    One exam set for the whole run, drawn from **every** conversational source plus the bundled
    corpus, and drawn *once*. Both halves of that are deliberate:

    *One set* — a stage that changed the questions could not be compared with the stage before it,
    and comparing stages is the only thing this module does.

    *Every source* — an exam made only of the bundled instruction corpus would ask nothing about
    theorems, arithmetic or conversation, so ProofNet, GSM8K and OASST1 would each be measured
    against questions from a different domain and would each read as having contributed nothing.

    The split itself is :meth:`nyxara.njp.study.Pair.held_out`'s, which hashes the question. That
    is what keeps a corpus that grew between two runs from reshuffling the held-out set — a
    reshuffled holdout is a leaked one.
    """
    from nyxara.njp.study import Corpus

    per_source: List[List[Any]] = []
    study: Dict[str, List[Any]] = {}
    sources = [("bundled", bundled)] + [
        (s.key, files[s.key]) for s in corpora.in_priority_order()
        if s.form == "pairs" and s.key in files]
    for key, path in sources:
        if path is None:
            continue
        pairs = Corpus.load(path, limit=(pool or None), sample=True, seed=seed)
        kept, held = Corpus.split(pairs)
        if key != "bundled":
            study[key] = kept
        if held:
            per_source.append(held)
    # The lookup questions from `_fact_exam` are already held out — their subjects are refused by
    # the loaders — so they are not split again here, only interleaved with everything else.
    per_source.extend([block for block in (extra or ()) if block])

    # Round-robin, not concatenated, and this is the difference between a real measurement and a
    # flattering one. `Tutor.exam` applies its limit by taking the **head** of what it is given,
    # so a concatenated list means `--exam 300` asks 300 questions from whichever corpus happened
    # to be loaded first — the bundled one — and ProofNet, GSM8K, OASST1 and MATH are then each
    # measured on questions from somebody else's domain and each read as having contributed
    # nothing. Interleaving makes every prefix balanced, so the limit shortens the exam instead of
    # narrowing it. Deterministic rather than shuffled: the same run twice must ask the same
    # questions in the same order, or two runs are not comparable.
    exam: List[Any] = []
    for row in range(max((len(block) for block in per_source), default=0)):
        for block in per_source:
            if row < len(block):
                exam.append(block[row])
    return exam, study


def _fact_exam(files: Dict[str, Path], *, per_source: int = 400, recall: int = 200,
               seed: int = 0) -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]]]:
    """``(withheld, recall)`` lookup questions, built from each fact corpus's own rows.

    **Why this exists, and why it is the third version of it.** The first full run measured
    ConceptNet's 195,897 facts at ``coverage +0.008``, and the natural reading was that two
    hundred thousand facts had taught her nothing. Asking her directly said otherwise — four of
    ten spot-check questions she had abstained on before came back correct. The exam had no
    instrument: every question in it came from the bundled instruction corpus, GSM8K, MATH, OASST1
    or ProofNet, and essentially none of those is a lookup.

    The second version held out whole **subjects** and measured **0 of 60** — correctly, and
    uselessly. A fact store asked about an entity it has never been given a single fact about has
    nothing to answer from, so that number was a restatement of the holdout rather than a
    measurement of anything. It is the more dangerous of the two failures, because it looks like a
    result.

    So two sets, and the pair of them is the reading:

    * **withheld** — questions about a ``(subject, predicate)`` this load will refuse, where the
      subject is otherwise present. She has ``dog``; ``dog capable_of`` is gone entirely. Only
      inheritance or composition can answer it, which is what :mod:`nyxara.njp.core` claims.
    * **recall** — questions about pairs that *were* loaded. Not a generalisation score and never
      to be read as one; it is the control, in the sense ``nyxara.eval.intelligence`` uses the
      word. If recall is near zero the corpus is stored and unreachable and every other number in
      the table is noise; if it is high, the withheld number means what it says.

    The gold answer is every object the corpus states for that pair — she is right if she says any
    of them. ConceptNet lists fifteen things a dog is, and demanding a particular one would score
    her against an arbitrary row.
    """
    from nyxara.njp.study import Pair

    withheld: Dict[str, List[Any]] = {}
    kept: Dict[str, List[Any]] = {}
    for source in corpora.in_priority_order():
        if source.form not in ("triples", "text") or source.key not in files:
            continue
        out_gold: Dict[Tuple[str, str], List[str]] = {}
        in_gold: Dict[Tuple[str, str], List[str]] = {}
        subjects_seen: Set[str] = set()
        cap = max(per_source, recall) * 8
        for subject, predicate, obj in _corpus_triples(source, files[source.key]):
            if not corpora.question_for(subject, predicate):
                continue
            subjects_seen.add(subject)
            bucket = (out_gold if corpora.held_out_relation(subject, predicate) else in_gold)
            values = bucket.setdefault((subject, predicate), [])
            if len(values) < 8:
                values.append(obj)
            if len(out_gold) >= cap and len(in_gold) >= cap:
                break

        # A withheld pair is only a fair question if the subject survives the load with something
        # else attached. Where it does not, the question is the subject-level holdout again under
        # another name, and it would read as a failure to generalise rather than as an absence.
        with_other = {s for (s, _p) in in_gold}
        out_gold = {(s, p): v for (s, p), v in out_gold.items() if s in with_other}

        withheld[source.key] = _sample_questions(Pair, out_gold, per_source, seed)
        kept[source.key] = _sample_questions(Pair, in_gold, recall, seed)
    return ({k: v for k, v in withheld.items() if v},
            {k: v for k, v in kept.items() if v})


def _sample_questions(Pair: Any, gold: Dict[Tuple[str, str], List[str]],
                      limit: int, seed: int) -> List[Any]:
    """Sampled rather than truncated, for the reason :meth:`Corpus.load` gives: a prepared corpus
    is written in whatever order its source had, so the first N are one corner of it."""
    pairs = [Pair(question=corpora.question_for(subject, predicate),
                  answer=", ".join(objects))
             for (subject, predicate), objects in gold.items()]
    random.Random(seed).shuffle(pairs)
    return pairs[:limit] if limit else pairs


def _corpus_triples(source: Any, path: Any) -> Iterator[Tuple[str, str, str]]:
    """Every ``(subject, predicate, object)`` a fact corpus states, held out or not.

    A triples corpus states them directly, and the predicate is folded here because the loader
    folds it — an unfolded name would put the exam's holdout on a different key from the load's,
    which is a leak that reports as a generalisation.

    A prose corpus states sentences, and what a sentence yields is whatever the extractor makes of
    it, so the extractor is run here on a throwaway grounder that asserts nothing. Guessing with a
    regex instead would be the second parser :mod:`nyxara.njp.corpora` exists to refuse, and it
    would guess differently from the loader.
    """
    from nyxara.njp.grounding import Grounder
    from nyxara.njp.ingest import stream_statements, stream_triples

    reader = Grounder()
    if source.form == "triples":
        for subject, predicate, obj, _conf in stream_triples(path):
            yield subject.strip().lower(), reader._predicate(predicate), obj
        return

    for statement, _provenance in stream_statements(path):
        try:
            extracted = reader._extract(statement)
        except Exception:  # noqa: BLE001
            continue
        for triple in extracted:
            yield triple.subject.strip().lower(), triple.predicate, triple.object


def train(brain: Any, directory: Any, *,
          only: Optional[Sequence[str]] = None,
          skip: Optional[Sequence[str]] = None,
          exam_limit: int = 300,
          pool: int = 4000,
          fact_pool: int = 400,
          recall_pool: int = 200,
          seed: int = 0,
          bundled: Any = None,
          max_facts: int = 250_000,
          max_statements: int = 0,
          study_limit: int = 0,
          seed_kinds: bool = True,
          f1_threshold: float = 0.4,
          checkpoint: Any = None,
          checkpoint_every: int = 0,
          on_stage: Any = None) -> TrainReport:
    """Run every prepared corpus in priority order, exam between each, report the attribution.

    ``bundled`` defaults to :data:`nyxara.njp.study.DEFAULT_CORPUS`. It is in the exam set but
    never in a stage: it is the corpus ``nyxara-study`` already trains on, and studying it here
    would put a tenth stage in the table that is not one of the Master's ten.
    """
    from nyxara.njp.study import DEFAULT_CORPUS, Tutor

    report = TrainReport()
    started = time.perf_counter()
    files = discover(directory)
    chosen = corpora.in_priority_order(only)
    denied = {str(k).strip().lower() for k in (skip or ())}

    # The conversational half and the lookup half are built separately and interleaved together,
    # so a shortened exam stays balanced across both — see `_exam_corpus` on why the limit takes
    # the head of the list.
    fact_sets, recall_sets = (_fact_exam(files, per_source=fact_pool, recall=recall_pool,
                                         seed=seed) if fact_pool else ({}, {}))
    exam, study_sets = _exam_corpus(
        files, bundled=(bundled if bundled is not None else DEFAULT_CORPUS),
        pool=pool, seed=seed, extra=list(fact_sets.values()))
    report.exam_size = min(len(exam), exam_limit) if exam_limit else len(exam)
    report.fact_questions = sum(len(v) for v in fact_sets.values())
    recall_exam = [p for block in recall_sets.values() for p in block]

    tutor = Tutor(brain, seed=seed_kinds, f1_threshold=f1_threshold)

    # Seeded here, before the control, rather than left to `Tutor.study`.
    #
    # `Tutor` seeds the kind layer lazily, on its first `study` call. In a one-corpus run that is
    # the same thing as seeding at the start. In this ladder it is not: the first `study` call is
    # stage 4, so 137 kind-level facts — the ones `core._inherit` needs before it can inherit
    # anything — would land inside ProofNet's row and be measured as ProofNet's contribution.
    # `Tutor.study`'s own comment gives the reason to move it: the kind layer should be part of
    # what she starts from, not growth a corpus produced.
    if seed_kinds:
        from nyxara.njp.study import seed_kinds as _seed_kinds
        tutor.seeded = _seed_kinds(brain)

    control = tutor.exam(exam, limit=exam_limit)
    report.control = control.to_dict()
    last_coverage, last_precision = control.coverage, control.precision

    for source in chosen:
        stage = StageResult(key=source.key, priority=source.priority, form=source.form)
        began = time.perf_counter()
        path = files.get(source.key)
        if source.key in denied:
            stage.skipped = "skipped by request"
        elif path is None:
            stage.skipped = f"not prepared ({source.filename})"
        else:
            stage.path = str(path)
            stage.rows = _count_rows(path)
            _run_stage(brain, tutor, source, path, stage,
                       study_sets.get(source.key, []),
                       max_facts=max_facts, max_statements=max_statements,
                       study_limit=study_limit,
                       exclude=(corpora.held_out_relation if fact_pool else None),
                       checkpoint=checkpoint, checkpoint_every=checkpoint_every)
            got = tutor.exam(exam, limit=exam_limit)
            stage.coverage, stage.precision = got.coverage, got.precision
            stage.accuracy, stage.on_topic = got.accuracy, got.on_topic_rate
            stage.d_coverage = got.coverage - last_coverage
            stage.d_precision = got.precision - last_precision
            last_coverage, last_precision = got.coverage, got.precision
            report.final = got.to_dict()
        stage.ms = (time.perf_counter() - began) * 1000.0
        report.stages.append(stage)
        if on_stage is not None:
            try:
                on_stage(stage)
            except Exception:  # noqa: BLE001 — a reporting callback cannot fail a run
                pass

    if not report.final:
        report.final = dict(report.control)

    # Run once, after every stage, rather than between them. It is a control rather than a
    # measurement of any one corpus — "is what she was given reachable at all" — and asking it
    # eleven times would double the wall-clock to answer the same question eleven times.
    if recall_exam:
        got = tutor.exam(recall_exam, limit=exam_limit)
        report.recall = got.to_dict()

    report.curriculum = _assess(brain)
    report.ms = (time.perf_counter() - started) * 1000.0
    return report


def _run_stage(brain: Any, tutor: Any, source: Any, path: Any, stage: StageResult,
               study_pairs: Sequence[Any], *, max_facts: int, max_statements: int,
               study_limit: int, exclude: Any, checkpoint: Any,
               checkpoint_every: int) -> None:
    """Load one corpus by whichever of the three doors its form names.

    ``source=source.key`` on every loader, and that is not cosmetic. ``Grounder._regenerable``
    matches the ingest manifest on the fact's ``source`` string, so two corpora sharing a tag
    would have one manifest row between them and the second would be written to the sidecar in
    full on every save. It is also what makes a fact's provenance answerable: "where did she learn
    that" has an answer that names a dataset.
    """
    if source.form == "triples":
        from nyxara.njp.ingest import ingest_triples
        got = ingest_triples(brain, path, source=source.key, max_facts=max_facts,
                             exclude=exclude,
                             checkpoint=checkpoint, checkpoint_every=checkpoint_every)
        stage.report = got.to_dict()
    elif source.form == "text":
        from nyxara.njp.ingest import ingest_text
        got = ingest_text(brain, path, source=source.key, max_facts=max_facts,
                          max_statements=max_statements, exclude=exclude,
                          checkpoint=checkpoint, checkpoint_every=checkpoint_every)
        stage.report = got.to_dict()
    elif source.form == "pairs":
        pairs = list(study_pairs)
        if study_limit and len(pairs) > study_limit:
            pairs = pairs[:study_limit]
        # The held-out tenth was removed by `_exam_corpus` before this ran, not here. Splitting at
        # the point of study would mean the exam set depended on which stages were run, and two
        # runs with different `--only` flags would not be comparable.
        got = tutor.study(pairs, checkpoint=checkpoint, checkpoint_every=checkpoint_every)
        stage.report = got.to_dict()
        stage.rows = len(pairs)
    else:
        stage.skipped = f"unknown form {source.form!r}"


def _assess(brain: Any) -> Dict[str, Any]:
    """What the curriculum reads after the run, or an empty mapping if it cannot run.

    Reported rather than acted on. :mod:`nyxara.njp.curriculum` gates *stages of development* on
    measurements of real organs, and a training run is exactly the event after which the answer to
    "what is she now blocked on" changes. Printing it here is how the next corpus gets chosen by a
    number instead of by a hunch.
    """
    try:
        from nyxara.njp.curriculum import Curriculum
        return Curriculum().assess(brain).to_dict()
    except Exception:  # noqa: BLE001 — the training result stands whether or not this reads
        return {}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m nyxara.njp.train --data DIR [--only KEY,KEY] [--save PATH]``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train NJP on the prepared corpora, in priority order, exam between each.")
    parser.add_argument("--data", default="data/njp",
                        help="directory holding the files prepare_datasets.py wrote")
    parser.add_argument("--only", default="",
                        help="comma-separated source keys to run; default is all ten")
    parser.add_argument("--skip", default="", help="comma-separated source keys to leave out")
    parser.add_argument("--exam", type=int, default=300, help="held-out questions per exam")
    parser.add_argument("--pool", type=int, default=4000,
                        help="pairs sampled from each conversational corpus (study + exam)")
    parser.add_argument("--recall-pool", type=int, default=200,
                        help="control questions per fact corpus, on pairs that WERE loaded")
    parser.add_argument("--fact-pool", type=int, default=400,
                        help="lookup questions held out of each fact corpus; 0 disables the "
                             "holdout entirely and leaves the fact stages unmeasured")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--max-facts", type=int, default=250_000,
                        help="hard cap on facts taken from any one corpus")
    parser.add_argument("--max-statements", type=int, default=0,
                        help="hard cap on statements read from any one prose corpus; 0 = all")
    parser.add_argument("--study-limit", type=int, default=0,
                        help="cap on pairs studied per conversational corpus; 0 = all")
    parser.add_argument("--f1-threshold", type=float, default=0.4)
    parser.add_argument("--no-seed", action="store_true",
                        help="skip the kind-level seed and measure the corpora alone")
    parser.add_argument("--save", default="", help="write the trained brain's state here")
    parser.add_argument("--load", default="", help="start from a saved state")
    parser.add_argument("--resume", action="store_true",
                        help="load --save if it exists, then keep training into it")
    parser.add_argument("--checkpoint-every", type=int, default=25_000)
    parser.add_argument("--json", default="", help="write the full report here as JSON")
    parser.add_argument("--plan", action="store_true",
                        help="print what would run against --data, and load nothing")
    args = parser.parse_args(list(argv) if argv is not None else None)

    only = [k for k in args.only.split(",") if k.strip()] or None
    skip = [k for k in args.skip.split(",") if k.strip()]
    files = discover(args.data)

    if args.plan:
        print(f"data directory: {args.data}\n")
        for source in corpora.in_priority_order(only):
            where = files.get(source.key)
            mark = "ready " if where else "MISSING"
            print(f"  {source.priority:>2} {source.tier:<6} {source.key:<13} "
                  f"{source.form:<8} {mark} {where or source.filename}")
            print(f"     {source.gets}")
        missing = [s.key for s in corpora.in_priority_order(only) if s.key not in files]
        if missing:
            print(f"\nprepare the missing ones with:\n"
                  f"  python scripts/prepare_datasets.py --out {args.data} "
                  f"--only {','.join(missing)}")
        return 0

    if not files:
        print(f"no prepared corpora in {args.data}")
        print("run scripts/prepare_datasets.py first, or pass --plan to see what is expected")
        return 1

    from nyxara.njp.brain import NJPBrain
    from nyxara.njp.study import load_state, save_state

    brain = NJPBrain()
    restore_from = args.load or (args.save if args.resume else "")
    if restore_from:
        try:
            if load_state(brain, restore_from):
                print(f"resumed from {restore_from}")
            elif args.load:
                print(f"nothing to load at {args.load}")
                return 1
            else:
                print(f"no state at {restore_from} yet — starting fresh")
        except Exception as exc:  # noqa: BLE001 — a corrupt state must not look like a fresh start
            print(f"could not read state at {restore_from}: {exc}")
            return 1

    def _checkpoint(n: int) -> None:
        if not args.save:
            return
        try:
            save_state(brain, args.save)
        except Exception as exc:  # noqa: BLE001
            print(f"  CHECKPOINT FAILED at {n}: {exc}", flush=True)
            return
        print(f"  checkpoint at {n} → {args.save}", flush=True)

    def _announce(stage: StageResult) -> None:
        if stage.skipped:
            print(f"  {stage.priority:>2} {stage.key:<13} {stage.skipped}", flush=True)
            return
        print(f"  {stage.priority:>2} {stage.key:<13} {stage.rows:>7} rows  "
              f"cover {stage.coverage:.3f} ({stage.d_coverage:+.3f})  "
              f"prec {stage.precision:.3f} ({stage.d_precision:+.3f})  "
              f"{stage.ms / 1000.0:.1f}s", flush=True)

    print(f"data directory: {args.data} — {len(files)} of {len(corpora.SOURCES)} corpora ready\n")
    report = train(brain, args.data, only=only, skip=skip,
                   exam_limit=args.exam, pool=args.pool, fact_pool=args.fact_pool,
                   recall_pool=args.recall_pool,
                   seed=args.sample_seed,
                   max_facts=args.max_facts, max_statements=args.max_statements,
                   study_limit=args.study_limit, seed_kinds=not args.no_seed,
                   f1_threshold=args.f1_threshold,
                   checkpoint=_checkpoint, checkpoint_every=args.checkpoint_every,
                   on_stage=_announce)

    print(f"\n— attribution, on {report.exam_size} held-out questions asked after every stage "
          f"({report.fact_questions} of them lookups from the fact corpora's own withheld "
          f"subjects) —")
    print(report.table())
    print("\nRead precision down the column, not coverage: every load raises coverage by turning "
          "abstentions\ninto answers, and only a load that held precision up while doing it "
          "added knowledge rather than noise.")

    if report.recall:
        print(f"\ncontrol — the same lookups on pairs that WERE loaded, {report.recall['asked']} "
              f"asked:\n  coverage {report.recall['coverage']:.3f}   "
              f"precision {report.recall['precision_when_answered']:.3f}   "
              f"hit {report.recall['on_topic_when_answered']:.3f}")
        print("  Not a generalisation score. It answers only 'is what she was given reachable',\n"
              "  and if it is near zero every delta above it is noise about an unreadable store.")

    blocked = (report.curriculum or {}).get("blocked_by")
    if blocked:
        print(f"\ncurriculum: blocked at {blocked}")

    if args.json:
        try:
            Path(args.json).write_text(json.dumps(report.to_dict(), indent=1), encoding="utf-8")
            print(f"\nfull report → {args.json}")
        except OSError as exc:
            print(f"\ncould not write {args.json}: {exc}")

    if args.save:
        try:
            save_state(brain, args.save)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED to write state to {args.save}: {exc}")
            return 1
        print(f"state written to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
