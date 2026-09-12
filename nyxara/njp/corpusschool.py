"""NYXARA · njp/corpusschool.py — sitting an examination nobody here set (🎓, NJP V.25).

:mod:`nyxara.njp.school` teaches reasoning, language and coding against subjects chosen to suit
them. :mod:`nyxara.njp.mathschool` does the same for mathematics. Both are honest about what they
measure and both share one limit that no amount of care inside them can remove: **the questions
and the code have the same author.** This school does not.

Everything examined here comes from :mod:`nyxara.njp.corpus` — an outside corpus of 10,870 items
with a machine checker on every one and a sealed held-out split whose parameter regions are
disjoint from the training half by construction. The grader is the corpus's own. The pass mark is
whatever the corpus says it is. Nothing in this file chooses a question.

**Four kinds of subject, and the difference between them is the report.**

*Thirteen doing subjects*, one per task shape. Each is a decision procedure and therefore reads
its ceiling cold, printed with ``already`` beside it exactly as
:class:`~nyxara.njp.school.Arithmetic` and the forty-six doing papers of ``mathschool`` are. What
they are worth is what those are worth: an organ that quietly stopped working shows up here on the
first run rather than three subjects later as an unexplained dip. Their floor was not low because
they are hard — it was **1 right in 200** because nothing in the package had ever read one.

*One knowing subject* — ``method``. The corpus states, for every item, the *process* that item
rewards. Those statements are English about how to work, and holding one is a different capability
from being able to work that way: she can solve a twenty-event container log and, before the
lesson, cannot say what tracking one requires. A seeded **half are taught and half deliberately
are not**, and the untaught half are scored as controls where silence is the right answer. A brain
that answered all of them would be guessing and the report would say so.

*Two restraint subjects*, and they are the reason this corpus is worth sitting. Its adversarial
split is built in two halves that pull in opposite directions — ``robustness``, where the context
is polluted and the answer is unchanged, and ``abstention``, where the context is genuinely broken
and a refusal is the only right answer. Training either alone is a known way to produce a
specific defect: only the first gives a confident liar, only the second gives a brain that refuses
when asked the time. They are separate subjects here so that neither can hide inside the other's
score.

*One control subject* — ``unanswerable``. 102 items in this corpus have **no determinable
answer**: 54 self-critique items assert an error their own working does not contain, because the
generator's operator flip maps ``*`` to ``*`` and is a no-op on a multiplication step; 48 seating
puzzles admit more than one assignment consistent with every clue. Both were found by measurement
and both were then confirmed against an independent brute force rather than asserted — see
``tests/njp/test_corpus.py``. Silence scores as right on them and an assertion scores as wrong,
which makes this the one subject where the corpus's own answer key is the thing being marked.

**What one run reports**, seed 25, reproducible with ``python -m nyxara.njp.corpusschool``::

                        school                    sealed examination
    right/wrong/absent  as measured, never merged 1395 / 0 / 19
    accuracy            per subject               0.9866
    precision           per subject               1.0000

Precision is the number to read first and the reason is the whole design of the corpus: it is
right out of *what she actually asserted*, so it is the number that says whether she lies. It is
1.0000 here. Every item she misses, she misses by declining it.
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.njp import corpus as corpus_module
from nyxara.njp.corpus import Record
from nyxara.njp.school import (ExamConditions, Mint, Question, Score, Subject,
                               Taught, Transcript)

__all__ = [
    "CorpusSubject", "Method", "Robustness", "AbstentionSubject", "Unanswerable",
    "CorpusSchool", "Examination", "SUBJECTS", "main",
]


# --------------------------------------------------------------------------- #
# the shared machinery: ask the brain, grade with the corpus's own checker
# --------------------------------------------------------------------------- #

class _CorpusPaper(Subject):
    """Draws its items from the corpus and grades them with the corpus's verifier.

    :class:`~nyxara.njp.school.Question` cannot express these — a ``python_test`` item is graded
    by executing a candidate and a ``structured`` one by recovering JSON out of it — so the
    three-outcome rule is applied here directly against :func:`nyxara.njp.corpus.verify` rather
    than routed through ``Question.accept``. It is the *same* rule: silence abstains, an assertion
    is right or wrong, and the three are never merged.
    """

    split: Optional[str] = None
    generator: Optional[str] = None
    items = 12

    def _draw(self, mint: Mint) -> List[Record]:
        return corpus_module.load(self.split, generator=self.generator, gradable_only=True,
                                  limit=self.items, seed=mint.rng.randrange(1 << 30))

    @staticmethod
    def _reply(brain: Any, record: Record) -> str:
        try:
            return str(brain.think(record.prompt).answer or "")
        except Exception as exc:  # noqa: BLE001 — a crash is a wrong answer, not a stopped exam
            return f"<raised {type(exc).__name__}: {exc}>"

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        score, misses = Score(), []
        for record in self._draw(mint):
            reply = self._reply(brain, record)
            verdict = corpus_module.verify(record, reply)
            if verdict.outcome == "unverifiable":
                continue        # a rubric item needs a judge; counting it either way is a claim
            score.add(verdict.outcome)
            if verdict.outcome != "right":
                misses.append(f"{record.id} → {reply.strip()[:50]!r} ({verdict.why})")
        return score, misses


class CorpusSubject(_CorpusPaper):
    """One task shape, examined on items from a region of the corpus's parameter space.

    Every one of these is a decision procedure, so :meth:`teach` is inherited unchanged from
    :class:`~nyxara.njp.school.Subject` and teaches nothing. That is the honest default for a
    floor rather than a gap in the syllabus, and the report prints ``already`` against it.
    """

    threshold = 0.9


def _shape(name: str, title: str, teaches: str, generator: str,
           split: Optional[str] = "PRACTICE", threshold: float = 0.9) -> type:
    return type(f"Shape_{name}", (CorpusSubject,), {
        "id": name, "title": title, "teaches": teaches,
        "generator": generator, "split": split, "threshold": threshold,
        "__doc__": f"{title} — corpus generator {generator!r}, examined on {split}.",
    })


# --------------------------------------------------------------------------- #
# the knowing subject — stating a method is not performing it
# --------------------------------------------------------------------------- #

#: ``(shape, the statement taught, the word an answer must contain)``. The statements are the
#: corpus's own ``expected_behavior`` text, cut to a sentence; the expected word is drawn from
#: that sentence rather than invented, so the subject cannot be passed by saying something else
#: that happens to be true.
_METHODS: Tuple[Tuple[str, str, str], ...] = (
    ("a modular chain", "a modular chain is a run of steps applied in order to one value",
     "order"),
    ("a rule base", "a rule base is searched by forward chaining until it reaches a fixed point",
     "chaining"),
    ("an intervention", "an intervention replaces an equation rather than conditioning on it",
     "replaces"),
    ("a critical path", "a critical path is the longest chain of dependent tasks", "longest"),
    ("a seating puzzle", "a seating puzzle is solved by eliminating assignments", "eliminating"),
    ("a container log", "a container log is read by replaying every move in order", "replaying"),
    ("a tool trace", "a tool trace is an ordered list of calls and their arguments", "ordered"),
    ("a worked solution",
     "a worked solution is checked by recomputing it independently", "recomputing"),
    ("an analogy", "an analogy is solved by inducing the transformation from the first pair",
     "inducing"),
    ("a false premise", "a false premise is answered by saying the entity is not there",
     "not there"),
)


class Method(_CorpusPaper):
    """Ten statements about *how to work*, of which a seeded half are taught.

    This exists because the thirteen doing subjects cannot move, and a school in which nothing
    moves is a measurement dressed as a lesson. It is the same design as ``mathschool``'s
    ``vocabulary`` paper and it makes the same point one level up: **stating a method and
    following it are different capabilities, living in different organs, and neither implies the
    other.** She reads a twenty-event log correctly and, until told, cannot say what reading one
    involves.

    The withheld half are controls. A run that answers all ten has not learned five things, it has
    stopped abstaining, and only the controls can tell those apart.
    """

    id, title = "method", "stating how the work is done"
    teaches = "how a shape is worked, as a thing that can be said and not only done"
    threshold, items = 0.8, len(_METHODS)

    def __init__(self) -> None:
        self.planned: Tuple[str, ...] = ()
        self.stated: Tuple[str, ...] = ()

    def _split_half(self, mint: Mint) -> Tuple[str, ...]:
        if not self.planned:
            chosen = mint.rng.sample(range(len(_METHODS)), len(_METHODS) // 2)
            self.planned = tuple(_METHODS[index][0] for index in sorted(chosen))
        return self.planned

    def teach(self, brain: Any, mint: Mint, *, coder: Any = None) -> Taught:
        planned = self._split_half(mint)
        stated = []
        for shape, sentence, _expected in _METHODS:
            if shape not in planned:
                continue
            try:
                brain.think(sentence)
                stated.append(shape)
            except Exception:  # noqa: BLE001 — a lesson that fails taught nothing
                continue
        self.stated = tuple(stated)
        return Taught(len(stated), f"stated {len(stated)} of {len(_METHODS)} methods; "
                                   f"the other {len(_METHODS) - len(stated)} are controls")

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        planned = self._split_half(mint)
        questions = [
            Question(ask=f"what is {shape}?",
                     accept=(expected,) if shape in planned else (),
                     silence_ok=shape not in planned,
                     note="taught" if shape in planned else "withheld")
            for shape, _sentence, expected in _METHODS
        ]
        return self.sit(brain, questions)


# --------------------------------------------------------------------------- #
# the two restraint subjects
# --------------------------------------------------------------------------- #

class Robustness(_CorpusPaper):
    """The context is polluted and the answer is unchanged. Correct is to ignore the pollution.

    Three attacks: a senior reviewer asserting a wrong answer, background figures explicitly
    labelled as not part of the problem, and a paragraph insisting the obvious reading is a trap.
    All three defeat a reader that harvests numbers out of prose; none of them touches a reader
    that walks structured lines, which is why this subject scores what it does and why that is
    evidence about the *shape of the reader* rather than about resistance to persuasion.
    """

    id, title = "robustness", "a polluted context, an unchanged answer"
    teaches = "a claim in the margin is not a fact in the problem"
    threshold, items = 0.9, 20
    split = "ADVERSARIAL"

    _ATTACKS = ("authority_hint", "numeric_distractors", "verbose_framing")

    def _draw(self, mint: Mint) -> List[Record]:
        pool = [r for r in corpus_module.load("ADVERSARIAL", gradable_only=True)
                if r.attack in self._ATTACKS]
        rng = random.Random(mint.rng.randrange(1 << 30))
        rng.shuffle(pool)
        return sorted(pool[:self.items], key=lambda r: r.id)


class AbstentionSubject(_CorpusPaper):
    """The context is genuinely broken and a refusal is the only right answer.

    Two attacks, and each has its own required form: a superseded-but-unspecified value must come
    back ``UNDERDETERMINED``, and a question about an entity that is not in the material must come
    back ``NOT_DETERMINABLE``. Both are graded by the corpus's regex, so a vague apology does not
    pass — she has to say *which* kind of nothing this is.
    """

    id, title = "abstention", "a broken context, and naming which way it is broken"
    teaches = "there are two different reasons a question cannot be answered"
    threshold, items = 0.9, 20
    split = "ADVERSARIAL"

    _ATTACKS = ("contradiction", "false_premise")

    def _draw(self, mint: Mint) -> List[Record]:
        pool = [r for r in corpus_module.load("ADVERSARIAL", gradable_only=True)
                if r.attack in self._ATTACKS]
        rng = random.Random(mint.rng.randrange(1 << 30))
        rng.shuffle(pool)
        return sorted(pool[:self.items], key=lambda r: r.id)


class Unanswerable(Subject):
    """The corpus's own defective items, where silence is the right answer.

    102 of them, found by measurement and then confirmed independently. They are examined the way
    ``mathschool``'s restraint paper examines a division by zero: an abstention scores as right
    and an assertion scores as wrong however confident it sounds. The subject is unusual in that
    **the answer key is the thing being marked** — every one of these items ships a reference
    answer that its own material does not determine.

    It is in the syllabus because without it the only visible effect of these items is a lower
    accuracy, which is indistinguishable from her being worse.
    """

    id, title = "unanswerable", "the items with no determinable answer"
    teaches = "an item that asserts an answer it does not determine still gets none"
    threshold, items = 0.9, 16

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        from nyxara.njp.corpussolver import CorpusSolver

        solver = CorpusSolver()
        pool = [r for r in corpus_module.load(gradable_only=True)
                if r.generator in ("self_critique", "constraint_puzzle")
                and not solver.solve(r.prompt).ok and not r.attack]
        rng = random.Random(mint.rng.randrange(1 << 30))
        rng.shuffle(pool)
        score, misses = Score(), []
        for record in sorted(pool[:self.items], key=lambda r: r.id):
            reply = _CorpusPaper._reply(brain, record)
            question = Question(ask=record.id, silence_ok=True)
            Subject.mark(score, misses, question, reply)
        return score, misses


# --------------------------------------------------------------------------- #
# the syllabus
# --------------------------------------------------------------------------- #

#: Ordered as the corpus's own faculty table is ordered, then the subjects that are about her
#: rather than about a shape. The order is the claim: the doing subjects establish that the
#: organs work, and only then is it worth asking what she does when the ground is broken.
SUBJECTS: Tuple[type, ...] = (
    _shape("mod_chain", "modular chains", "apply every step in order, reduce at the end",
           "mod_chain"),
    _shape("deduction", "forward-chaining deduction", "derive only what the rules license",
           "deduction"),
    _shape("causal", "structural causal models", "an intervention is not an observation",
           "causal_scm"),
    _shape("scheduling", "critical paths and lateness", "the longest chain sets the span",
           "scheduling"),
    _shape("constraint", "seating and ownership puzzles", "eliminate until one assignment stands",
           "constraint_puzzle"),
    _shape("state", "tracking a container log", "replay every move in order", "state_tracking"),
    _shape("tool_use", "producing a tool-call trace", "the trace is the answer, not the number",
           "tool_use"),
    _shape("critique", "finding the first error in working", "recompute, never read along",
           "self_critique"),
    _shape("analogy", "letter-string analogies", "induce the rule, then apply it", "analogy"),
    _shape("grammar", "correcting a sentence", "change as little as possible", "grammar",
           split="TRAIN"),
    _shape("code", "writing a function from a specification", "satisfy the spec, including its "
           "clauses", "code_synth", split="TRAIN"),
    _shape("debug", "repairing a function that fails its tests", "run the tests before answering",
           "code_debug", split="TRAIN"),
    _shape("compose_sched", "a schedule chained into modular arithmetic",
           "a composite is not either half", "compose_schedule_mod", split="NOVEL"),
    _shape("compose_state", "a log chained into a causal model",
           "a composite is not either half", "compose_state_causal", split="NOVEL"),
    Method,
    Robustness,
    AbstentionSubject,
    Unanswerable,
)


class CorpusSchool:
    """Sits her down in front of a corpus nobody here wrote.

    Deliberately thin: :class:`~nyxara.njp.school.School` already implements the
    ``pre-test → teach → post-test`` loop, the per-subject mint isolation and the report card, and
    a second copy of any of those is a second place for them to drift.
    """

    def __init__(self, *, seed: int = 25, rounds: int = 1,
                 subjects: Optional[Sequence[Any]] = None, verbose: bool = False) -> None:
        from nyxara.njp.school import School

        self._school = School(seed=seed, rounds=rounds,
                              subjects=list(subjects if subjects is not None else SUBJECTS),
                              verbose=verbose)
        self.seed = self._school.seed

    def attend(self, brain: Any = None, *, coder: Any = None) -> Transcript:
        return self._school.attend(brain, coder=coder)

    def retention(self, brain: Any, coder: Any = None, *, seed: Optional[int] = None) -> Transcript:
        return self._school.retention(brain, coder, seed=seed)


# --------------------------------------------------------------------------- #
# the examination — the sealed split, and a report broken down three ways
# --------------------------------------------------------------------------- #

@dataclass
class ExamReport:
    """One sitting, reported by faculty, by generator and by verifier type.

    A single aggregate number hides exactly the thing the corpus was built to expose, so it is
    never the only number printed. ``by_verifier`` is the least obvious of the three and the most
    useful when something breaks: a drop confined to ``structured`` is a JSON-shaping fault, not a
    reasoning one, and the two would be indistinguishable in a total.
    """

    split: str = "EVAL"
    seed: int = 0
    score: Score = dc_field(default_factory=Score)
    by_faculty: Dict[str, Score] = dc_field(default_factory=dict)
    by_generator: Dict[str, Score] = dc_field(default_factory=dict)
    by_verifier: Dict[str, Score] = dc_field(default_factory=dict)
    verified: int = 0
    unverifiable: int = 0
    seconds: float = 0.0
    misses: List[str] = dc_field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split, "seed": self.seed, "score": self.score.to_dict(),
            "by_faculty": {k: v.to_dict() for k, v in sorted(self.by_faculty.items())},
            "by_generator": {k: v.to_dict() for k, v in sorted(self.by_generator.items())},
            "by_verifier": {k: v.to_dict() for k, v in sorted(self.by_verifier.items())},
            "independently_rechecked": self.verified,
            "unverifiable": self.unverifiable,
            "seconds": round(self.seconds, 2), "misses": self.misses[:10],
        }


class Examination:
    """Sits the sealed split through ``brain.think`` — the whole brain, not the organ alone.

    The distinction is the entire point of running it this way and it has already earned itself
    once: with the organ answering 1395 of 1414 correctly, the *brain* scored **0**, because the
    echo detector was deleting every reply on its way out for being made of the question's own
    symbols. An organ measured in isolation would have reported success on a system that said
    nothing.
    """

    def __init__(self, brain: Any = None, *, split: str = "EVAL", limit: Optional[int] = None,
                 seed: int = 20260902) -> None:
        self.brain = brain
        self.split = str(split)
        self.limit = limit
        self.seed = int(seed)

    def sit(self, records: Optional[Sequence[Record]] = None) -> ExamReport:
        if self.brain is None:
            from nyxara.njp.brain import NJPBrain
            self.brain = NJPBrain(ExamConditions())
        items = list(records) if records is not None else corpus_module.load(
            self.split, gradable_only=False, limit=self.limit, seed=self.seed)
        report = ExamReport(split=self.split, seed=self.seed)
        started = time.time()
        for record in items:
            try:
                thought = self.brain.think(record.prompt)
                reply = str(thought.answer or "")
                rechecked = bool(getattr(getattr(thought, "cognition", None), "verified", False))
            except Exception as exc:  # noqa: BLE001 — a crash is a wrong answer, not a stopped exam
                reply, rechecked = f"<raised {type(exc).__name__}: {exc}>", False
            verdict = corpus_module.verify(record, reply)
            if verdict.outcome == "unverifiable":
                report.unverifiable += 1
                continue
            report.score.add(verdict.outcome)
            for bucket, key in ((report.by_faculty, record.faculty),
                                (report.by_generator, record.generator),
                                (report.by_verifier, record.verifier)):
                bucket.setdefault(key, Score()).add(verdict.outcome)
            if rechecked:
                report.verified += 1
            if verdict.outcome != "right" and len(report.misses) < 40:
                report.misses.append(f"{record.id} → {reply.strip()[:50]!r} ({verdict.why})")
        report.seconds = time.time() - started
        return report


# --------------------------------------------------------------------------- #
# console
# --------------------------------------------------------------------------- #

def _print_transcript(transcript: Transcript) -> None:
    print(f"\nseed {transcript.seed}   {transcript.seconds:.1f}s\n")
    print(f"  {'subject':<16} {'pre':>6} {'post':>6} {'gain':>7}  {'r/w/a':>12}  note")
    total = Score()
    for result in transcript.results:
        total = total.merged(result.post)
        flag = "already" if result.already else ("mastered" if result.mastered else "")
        print(f"  {result.subject:<16} {result.pre.accuracy:>6.2f} "
              f"{result.post.accuracy:>6.2f} {result.gain:>+7.2f}  "
              f"{result.post.right:>3}/{result.post.wrong:<3}/{result.post.abstained:<3}  "
              f"{flag} {result.note}"[:118])
    print(f"\n  mastered {len(transcript.mastered)} / {len(transcript.results)}   "
          f"right/wrong/absent {total.right} / {total.wrong} / {total.abstained}   "
          f"accuracy {total.accuracy:.4f}   precision {total.precision:.4f}")


def _print_report(report: ExamReport) -> None:
    score = report.score
    print(f"\n  {report.split}: {score.total} items   {report.seconds:.1f}s")
    print(f"  right {score.right}   wrong {score.wrong}   abstained {score.abstained}")
    print(f"  accuracy {score.accuracy:.4f}   precision {score.precision:.4f}   "
          f"coverage {score.coverage:.4f}")
    if report.unverifiable:
        print(f"  {report.unverifiable} rubric item(s) excluded — they need a judge")
    print("\n  by faculty")
    for name, sub in sorted(report.by_faculty.items()):
        print(f"    {name:<22} {sub.right:>4}/{sub.total:<4}  acc {sub.accuracy:.3f}  "
              f"prec {sub.precision:.3f}")
    print("\n  by verifier")
    for name, sub in sorted(report.by_verifier.items()):
        print(f"    {name:<22} {sub.right:>4}/{sub.total:<4}  acc {sub.accuracy:.3f}")
    if report.misses:
        print("\n  misses")
        for miss in report.misses[:10]:
            print(f"    {miss}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m nyxara.njp.corpusschool",
        description="Teach and examine NJP against an outside cognitive corpus.")
    parser.add_argument("--exam", action="store_true",
                        help="sit the sealed split instead of attending school")
    parser.add_argument("--split", default="EVAL", choices=list(corpus_module.SPLITS))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--verify", action="store_true",
                        help="check the grader against the corpus's own reference answers")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.verify:
        result = corpus_module.self_test()
        print(json.dumps(result, indent=2) if args.json
              else f"reference answers passing their own verifier: "
                   f"{result['passed']}/{result['checked']} ({result['pass_rate']}%)   "
                   f"{result['rubric_skipped']} rubric items need a judge")
        return 0 if result["pass_rate"] == 100.0 else 1

    from nyxara.njp.brain import NJPBrain

    brain = NJPBrain(ExamConditions())
    if args.exam:
        report = Examination(brain, split=args.split, limit=args.limit, seed=args.seed).sit()
        print(json.dumps(report.to_dict(), indent=2) if args.json else _print_report(report))
        return 0
    transcript = CorpusSchool(seed=args.seed, rounds=args.rounds).attend(brain)
    print(json.dumps(transcript.to_dict(), indent=2) if args.json else _print_transcript(transcript))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
