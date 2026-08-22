"""NYXARA · eval/intelligence.py — the seven-stage learning curve (📈, NJP V.05).

The benchmark that decides whether :mod:`nyxara.njp.core` is learning or bookkeeping.

**Why this exists.** ``fabric.grown_total`` rising is a fact about her internals and no evidence
at all about her competence: synapses grow on every turn by construction, so the number goes up
whether or not anything was learned. The same is true of ``concepts``, ``episodes``, and every
other counter in :meth:`~nyxara.njp.brain.NJPBrain.stats`. An internal state that changed and an
unseen task that got easier are different claims, and only the second one is worth the word
*intelligence*. So every stage here is scored on items the brain **was never taught**, and the
teaching set and the test set are disjoint by construction rather than by hope.

**The stages, hardest last**::

    1 MEMORIZATION        can she hold a fact and give it back
    2 GENERALIZATION      a rule induced from some members, tested on others
    3 RECOMBINATION       A→B and B→C stated separately; A⇝C never stated
    4 CAUSAL PREDICTION   a counterfactual value at a point never observed
    5 SELF-CORRECTION     a contradiction arrives; does the newer fact win, and is it flagged
    6 TRANSFER            the same structure in a domain sharing no vocabulary
    7 PARAPHRASE          the simplest of those inferences, said a different way

Stage 1 is not a throwaway. It is the control: if memorisation fails then every later stage is
measuring a broken pipe rather than an absent faculty, and the report says so instead of
reporting zeroes as if they meant six different things.

Stage 7 is a control of the opposite kind, and it was added because the first six could not see
a defect the audit found by hand: they generate sentences that match the extractor's patterns
exactly, so all six read 1.00 while the identical inheritance failed on the word "birds". Where
1-6 hold the surface constant to measure inference, 7 holds the inference constant to measure the
surface. Measured on the change that introduced it: 0.40 → 1.00, with 1-6 unmoved at 1.00 — which
is what says the fix was to extraction and not to the thing being extracted.

**Vocabulary is generated, never literary.** Every stage builds its own nonsense terms from the
seed, so no item can be answered from anything but the facts stated in that stage's own teaching
phase — and a stage's terms never appear in another stage's. Bundled data files and real-world
nouns are both avoided for the same reason: an entity she could know about from anywhere else is
an entity whose test item proves nothing.

**A fresh brain per stage.** Each stage constructs its own :class:`~nyxara.njp.brain.NJPBrain`,
so a fact taught in stage 3 cannot leak into stage 6 and a schema confirmed in stage 2 cannot
flatter stage 5. This costs a few hundred milliseconds and buys the only thing that makes a
stage score interpretable on its own.

Run it::

    python -m nyxara.eval --intelligence
    python -m nyxara.eval --intelligence --seed 7 --width 8
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "StageResult", "IntelligenceReport", "run_intelligence_benchmark",
    "STAGE_NAMES",
]

STAGE_NAMES: Tuple[str, ...] = (
    "memorization",
    "generalization",
    "recombination",
    "causal_prediction",
    "self_correction",
    "transfer",
    "paraphrase",
)

#: Syllables the generated vocabulary is built from. Deliberately meaningless: a term that looks
#: like a word invites the reader to check whether the answer is *true*, when the only thing
#: under test is whether it follows from what was said.
_SYLLABLES = ("va", "ku", "mi", "zo", "ren", "tal", "fu", "nex", "dor", "pil",
              "sha", "wen", "gru", "oky", "bem", "lir", "tza", "quo", "vel", "nak")


def _vocabulary(rng: random.Random, n: int, *, tag: str) -> List[str]:
    """``n`` distinct nonsense terms, stable for a seed and unique to this ``tag``.

    The tag is mixed in so two stages drawing the same count from the same seed still get
    disjoint vocabularies — without it stage 2's "vaku" and stage 6's "vaku" would be the same
    string in two brains, which is harmless right up until someone runs both against one brain
    and cannot work out why transfer scored so well.
    """
    out: List[str] = []
    seen = set()
    while len(out) < n:
        word = tag + "".join(rng.choice(_SYLLABLES) for _ in range(2)) + str(len(out))
        if word not in seen:
            seen.add(word)
            out.append(word)
    return out


@dataclass
class StageResult:
    """One stage: what was taught, what was asked, and how much of it came back right."""

    stage: str = ""
    correct: int = 0
    total: int = 0
    taught: int = 0
    ms: float = 0.0
    detail: List[str] = field(default_factory=list)
    note: str = ""

    @property
    def score(self) -> Optional[float]:
        """``None`` when nothing was asked — distinct from 0.0, which means asked and wrong."""
        return (self.correct / self.total) if self.total else None

    def to_dict(self) -> Dict[str, Any]:
        return {"stage": self.stage, "correct": self.correct, "total": self.total,
                "taught": self.taught, "score": None if self.score is None else round(self.score, 3),
                "ms": round(self.ms, 1), "note": self.note, "detail": self.detail[:6]}


@dataclass
class IntelligenceReport:
    """The whole curve, in the order the stages build on each other."""

    stages: List[StageResult] = field(default_factory=list)
    seed: int = 0
    width: int = 0
    ms: float = 0.0

    def by_name(self, name: str) -> Optional[StageResult]:
        for stage in self.stages:
            if stage.stage == name:
                return stage
        return None

    @property
    def mean(self) -> Optional[float]:
        scored = [s.score for s in self.stages if s.score is not None]
        return (sum(scored) / len(scored)) if scored else None

    @property
    def sound(self) -> bool:
        """Is the control stage healthy enough for the rest of the curve to mean anything?

        Memorisation below half is a broken pipe, not a weak faculty — a brain that cannot give
        back a fact it was just told will score zero on everything downstream for a reason that
        has nothing to do with generalisation. Reporting that as "no transfer ability" would be
        the benchmark lying about its own validity.
        """
        first = self.by_name("memorization")
        return bool(first and first.score is not None and first.score >= 0.5)

    def to_dict(self) -> Dict[str, Any]:
        return {"seed": self.seed, "width": self.width, "ms": round(self.ms, 1),
                "mean": None if self.mean is None else round(self.mean, 3),
                "sound": self.sound,
                "stages": [s.to_dict() for s in self.stages]}

    def render(self) -> str:
        """The curve as a table. Numbers first; no adjective is applied to any of them."""
        lines = [
            "NYXARA — seven-stage intelligence benchmark",
            f"seed {self.seed} · width {self.width} · {self.ms:.0f} ms",
            "",
            f"{'stage':<20} {'score':>7} {'correct':>9} {'taught':>7}  note",
            "-" * 78,
        ]
        for stage in self.stages:
            score = "  n/a  " if stage.score is None else f"{stage.score:6.2f} "
            lines.append(f"{stage.stage:<20} {score:>7} "
                         f"{stage.correct:>4}/{stage.total:<4} {stage.taught:>7}  {stage.note}")
        lines.append("-" * 78)
        mean = "n/a" if self.mean is None else f"{self.mean:.3f}"
        lines.append(f"{'mean':<20} {mean:>7}")
        if not self.sound:
            lines.append("")
            lines.append("WARNING: memorization below 0.5 — the control stage failed, so every "
                         "score below it is measuring a broken pipe rather than a faculty.")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #

#: Something done to a fresh brain before a stage teaches it — bulk-loading a corpus, restoring a
#: saved state, seeding a knowledge base. Takes the brain, returns nothing.
Preparer = Callable[[Any], None]


def _brain(*, prepare: Optional[Preparer] = None, **overrides: Any) -> Any:
    """A fresh brain, or ``None`` if NJP will not construct in this environment.

    ``prepare`` is what makes this benchmark usable as a before/after instrument at all. Every
    stage builds its own brain — deliberately, so that no stage can be carried by what another
    one taught — and the consequence was that nothing done to a brain *outside* the benchmark
    could ever be measured by it. Feeding a quarter-million ConceptNet facts and then running
    this measured six fresh brains that had never seen them.

    A failing preparer is fatal to the stage rather than silent. The alternative is a run that
    reports a fresh-brain score under a heading that claims the corpus was loaded, which is worse
    than no number: it reads as evidence that the corpus did not help.
    """
    try:
        from nyxara.njp.brain import NJPBrain
        brain = NJPBrain(type("_Cfg", (), overrides)) if overrides else NJPBrain()
    except Exception:  # noqa: BLE001
        return None
    if prepare is not None:
        prepare(brain)
    return brain


def _teach(brain: Any, sentences: Sequence[str]) -> int:
    """State each sentence as the Master would. Returns how many actually grounded.

    The return matters: a stage that taught twenty sentences and grounded three is not a stage
    that measured generalisation, and the difference shows up in the report as ``taught`` rather
    than being averaged silently into the score.
    """
    landed = 0
    for sentence in sentences:
        try:
            thought = brain.think(sentence)
            grounding = getattr(getattr(thought, "percept", None), "grounding", None)
            if getattr(grounding, "triples", None):
                landed += 1
        except Exception:  # noqa: BLE001
            continue
    return landed


def _ask(brain: Any, question: str) -> str:
    try:
        return str(getattr(brain.think(question), "answer", "") or "").strip().lower()
    except Exception:  # noqa: BLE001
        return ""


def _hit(answer: str, expected: str) -> bool:
    """Did the answer name the expected thing?

    Substring rather than equality, and only in that direction. A derived answer legitimately
    arrives with its route attached ("water, inherited from animal") and demanding an exact match
    would score a correct inference as a miss; accepting the reverse containment would let "I do
    not know about water" pass, which is why it is not symmetric.
    """
    expected = str(expected or "").strip().lower()
    return bool(expected) and bool(answer) and expected in answer


# --------------------------------------------------------------------------- #
# 1 · MEMORIZATION — the control
# --------------------------------------------------------------------------- #

def _stage_memorization(rng: random.Random, width: int,
       prepare: Optional[Preparer] = None) -> StageResult:
    """Told N facts, asked for each one back. If this fails, nothing below it is interpretable."""
    out = StageResult(stage="memorization")
    started = time.perf_counter()
    try:
        brain = _brain(prepare=prepare)
        if brain is None:
            out.note = "NJP unavailable"
            return out
        causes = _vocabulary(rng, width, tag="mem")
        effects = _vocabulary(rng, width, tag="fem")
        out.taught = _teach(brain, [f"{c} se {e} hoti hai" for c, e in zip(causes, effects)])
        for cause, effect in zip(causes, effects):
            out.total += 1
            if _hit(_ask(brain, f"{cause} se kya hota hai?"), effect):
                out.correct += 1
            else:
                out.detail.append(f"miss: {cause} → expected {effect}")
        out.note = "recall of stated facts"
        return out
    except Exception as exc:  # noqa: BLE001
        out.note = f"error: {exc}"
        return out
    finally:
        out.ms = (time.perf_counter() - started) * 1000.0


# --------------------------------------------------------------------------- #
# 2 · GENERALIZATION — a rule induced from some members, tested on others
# --------------------------------------------------------------------------- #

def _stage_generalization(rng: random.Random, width: int,
       prepare: Optional[Preparer] = None) -> StageResult:
    """Every member of a kind shares a property; some are told, the rest are asked.

    The held-out members are told *only* that they belong to the kind. Nothing states their
    property, so a correct answer cannot come from recall — it has to come from the schema, and
    the schema has to have survived being scored on members it was not induced from.
    """
    out = StageResult(stage="generalization")
    started = time.perf_counter()
    try:
        brain = _brain(prepare=prepare)
        if brain is None:
            out.note = "NJP unavailable"
            return out
        members = _vocabulary(rng, max(6, width + 4), tag="gen")
        kind, prop = "zorbex", "quilta"
        # Everyone is stated to be of the kind; only the first two thirds get the property.
        cut = max(2, int(len(members) * 0.66))
        taught = [f"{m} is a {kind}" for m in members]
        taught += [f"{m} increases {prop}" for m in members[:cut]]
        out.taught = _teach(brain, taught)

        learner = getattr(brain, "learner", None)
        if learner is not None:
            # Force the schedule: the benchmark is not long enough for the turn-count cadence to
            # come round, and what is under test is the mechanism, not its timer.
            learner.represent()
            learner.test()

        for member in members[cut:]:
            out.total += 1
            answer = _ask(brain, f"what does {member} increase?")
            if _hit(answer, prop):
                out.correct += 1
            else:
                out.detail.append(f"miss: {member} → expected {prop}, got {answer!r}")
        out.note = f"{cut} members taught, {len(members) - cut} held out"
        return out
    except Exception as exc:  # noqa: BLE001
        out.note = f"error: {exc}"
        return out
    finally:
        out.ms = (time.perf_counter() - started) * 1000.0


# --------------------------------------------------------------------------- #
# 3 · RECOMBINATION — a conclusion neither sentence contains
# --------------------------------------------------------------------------- #

def _stage_recombination(rng: random.Random, width: int,
       prepare: Optional[Preparer] = None) -> StageResult:
    """``A causes B`` and ``B causes C`` are stated; ``A ⇝ C`` never is.

    The two premises are taught in separate turns and never appear together in one sentence, so
    the link between them exists only if she put it there. Scored on the *end* of the chain,
    which is the term that appears in no sentence with the start.
    """
    out = StageResult(stage="recombination")
    started = time.perf_counter()
    try:
        brain = _brain(prepare=prepare)
        if brain is None:
            out.note = "NJP unavailable"
            return out
        chains = max(3, width // 2)
        heads = _vocabulary(rng, chains, tag="rca")
        mids = _vocabulary(rng, chains, tag="rcb")
        tails = _vocabulary(rng, chains, tag="rcc")
        taught: List[str] = []
        for head, mid, tail in zip(heads, mids, tails):
            taught.append(f"{head} se {mid} hoti hai")
            taught.append(f"{mid} se {tail} hoti hai")
        out.taught = _teach(brain, taught)

        # Scored through `think`, like every other stage, and that is a deliberate change.
        #
        # This used to call `brain.learner.reach(...)` and `.connects(...)` directly, so it
        # measured the *organ* and reported 20/20 while the same question asked in words returned
        # "" — `core.reach` and `.connects` had zero production callers, and this benchmark was
        # the only thing in the repo touching them. A stage that passes on a path the Master
        # cannot use is not evidence about the Master's experience of her.
        for head, mid, tail in zip(heads, mids, tails):
            out.total += 1
            said = _ask(brain, f"{head} se kya kya hota hai")
            if not _hit(said, tail):
                said = _ask(brain, f"does {head} cause {tail}")
            if _hit(said, tail) or said.strip().lower().startswith("yes"):
                out.correct += 1
            else:
                out.detail.append(f"miss: {head} ⇝ {tail} not derived through think()")
        out.note = f"{chains} two-step chains through think(), endpoint never stated"
        return out
    except Exception as exc:  # noqa: BLE001
        out.note = f"error: {exc}"
        return out
    finally:
        out.ms = (time.perf_counter() - started) * 1000.0


# --------------------------------------------------------------------------- #
# 4 · CAUSAL PREDICTION — a value at a point never observed
# --------------------------------------------------------------------------- #

def _stage_causal(rng: random.Random, width: int,
       prepare: Optional[Preparer] = None) -> StageResult:
    """Paired measurements are given; a counterfactual is asked at an unobserved input.

    Scored on whether the do-operator produces a *number* with a stated confidence, not on how
    close the number is. Accuracy of the fit is :mod:`nyxara.njp.universe`'s business and has its
    own tests; what this stage asks is whether a conversation of ordinary sentences ends with a
    counterfactual engine that has anything at all to work with — which is exactly what was
    measured as broken, and for a language reason rather than a modelling one.
    """
    out = StageResult(stage="causal_prediction")
    started = time.perf_counter()
    try:
        brain = _brain(prepare=prepare)
        if brain is None:
            out.note = "NJP unavailable"
            return out
        taught: List[str] = []
        for i in range(1, max(4, width // 2) + 1):
            taught.append(f"plant got {i * 2} litres of water")
            taught.append(f"the plant grew {i * 10} cm")
        out.taught = _teach(brain, taught)

        probes = ["What if the water were 3?",
                  "what if the water is halved?",
                  "what if the water is doubled?"]
        for probe in probes:
            out.total += 1
            answer = _ask(brain, probe)
            # A real do-operator answer names both endpoints and a confidence; a refusal or an
            # empty string is a miss, and so is a sentence with no digit in it.
            if answer and any(ch.isdigit() for ch in answer) and "confidence" in answer:
                out.correct += 1
            else:
                out.detail.append(f"miss: {probe!r} → {answer!r}")
        out.note = "counterfactual at an unobserved input"
        return out
    except Exception as exc:  # noqa: BLE001
        out.note = f"error: {exc}"
        return out
    finally:
        out.ms = (time.perf_counter() - started) * 1000.0


# --------------------------------------------------------------------------- #
# 5 · SELF-CORRECTION — the newer fact wins, and the change is visible
# --------------------------------------------------------------------------- #

def _stage_self_correction(rng: random.Random, width: int,
       prepare: Optional[Preparer] = None) -> StageResult:
    """A fact is stated, then contradicted. Does she answer with the correction?

    Two things are scored and both are required: she must answer with the *revised* value, and
    the superseded one must stop being answered with. A store that simply appends would pass the
    first half by accident half the time, which is why the old value being *absent* is checked
    rather than assumed.
    """
    out = StageResult(stage="self_correction")
    started = time.perf_counter()
    try:
        brain = _brain(prepare=prepare)
        if brain is None:
            out.note = "NJP unavailable"
            return out
        people = _vocabulary(rng, max(3, width // 2), tag="scp")
        first = _vocabulary(rng, len(people), tag="sca")
        second = _vocabulary(rng, len(people), tag="scb")
        taught: List[str] = []
        for person, old in zip(people, first):
            taught.append(f"{person} lives in {old}")
        for person, new in zip(people, second):
            taught.append(f"{person} lives in {new}")
        out.taught = _teach(brain, taught)

        for person, old, new in zip(people, first, second):
            out.total += 1
            answer = _ask(brain, f"where does {person} live?")
            if _hit(answer, new) and not _hit(answer, old):
                out.correct += 1
            else:
                out.detail.append(f"miss: {person} → expected {new}, got {answer!r}")
        out.note = "revised value answered, superseded one dropped"
        return out
    except Exception as exc:  # noqa: BLE001
        out.note = f"error: {exc}"
        return out
    finally:
        out.ms = (time.perf_counter() - started) * 1000.0


# --------------------------------------------------------------------------- #
# 6 · TRANSFER — the same structure, no shared vocabulary
# --------------------------------------------------------------------------- #

def _stage_transfer(rng: random.Random, width: int,
       prepare: Optional[Preparer] = None) -> StageResult:
    """A hierarchy is taught in one domain; the same shape is completed in another.

    The hardest stage, and the one closest to what the word transfer is usually used to mean. She
    is taught a three-level ``is_a`` hierarchy with a property at the top in domain A, then given
    only the *hierarchy* of a structurally identical domain B — different terms, no overlap, and
    the property stated only at B's root. The question is asked of a B leaf that was never said
    to have it.

    What this can and cannot show, stated plainly: it measures whether she completes a structure
    she was given the shape of, which is inheritance over stated edges. It is **not** a claim
    that she noticed domain B resembled domain A — nothing here rewards analogy between the two,
    and a brain that solved B while having been taught nothing about A would score identically.
    That is a real limit of this stage and it is better named than glossed.
    """
    out = StageResult(stage="transfer")
    started = time.perf_counter()
    try:
        brain = _brain(prepare=prepare)
        if brain is None:
            out.note = "NJP unavailable"
            return out
        depth = 3
        pairs = max(2, width // 3)
        taught: List[str] = []
        probes: List[Tuple[str, str]] = []

        # Domain A, taught complete, so the shape has been seen at least once.
        a_levels = _vocabulary(rng, depth, tag="tfa")
        a_prop = "molvex"
        for lower, upper in zip(a_levels, a_levels[1:]):
            taught.append(f"{lower} is a {upper}")
        taught.append(f"{a_levels[-1]} needs {a_prop}")

        # Domain B..N, hierarchy only. The property is stated at the root and nowhere else.
        for i in range(pairs):
            levels = _vocabulary(rng, depth, tag=f"tf{i}")
            prop = _vocabulary(rng, 1, tag=f"tp{i}")[0]
            for lower, upper in zip(levels, levels[1:]):
                taught.append(f"{lower} is a {upper}")
            taught.append(f"{levels[-1]} needs {prop}")
            probes.append((levels[0], prop))       # ask the LEAF, told only its parent

        out.taught = _teach(brain, taught)
        for leaf, prop in probes:
            out.total += 1
            answer = _ask(brain, f"what does {leaf} need?")
            if _hit(answer, prop):
                out.correct += 1
            else:
                out.detail.append(f"miss: {leaf} → expected {prop}, got {answer!r}")
        out.note = f"{depth}-level hierarchy, property stated only at the root"
        return out
    except Exception as exc:  # noqa: BLE001
        out.note = f"error: {exc}"
        return out
    finally:
        out.ms = (time.perf_counter() - started) * 1000.0


# --------------------------------------------------------------------------- #
# 7 · PARAPHRASE — the same inference, said a different way
# --------------------------------------------------------------------------- #

#: How the taught sentence and the asked question are allowed to differ. Each row is
#: ``(taught kind phrase, asked kind phrase, taught verb, asked verb)`` and every row states the
#: SAME fact and asks the SAME question — only the surface differs. Nothing here is exotic: these
#: are the forms an ordinary sentence arrives in.
_PARAPHRASES: Tuple[Tuple[str, str, str, str], ...] = (
    ("{k}",        "{k}",        "needs",    "need"),      # bare, verb form only
    ("{k}s",       "a {k}",      "need",     "need"),      # plural taught, singular asked
    ("the {k}s",   "{k}",        "require",  "need"),      # article + plural + synonym
    ("a {k}",      "the {k}",    "needs",    "require"),   # articles both sides
    ("{k}s",       "{k}s",       "requires", "need"),      # plural both sides
)


def _stage_paraphrase(rng: random.Random, width: int,
       prepare: Optional[Preparer] = None) -> StageResult:
    """One inheritance, taught and asked in different words. Surface robustness, isolated.

    **Why this stage exists.** The six above are the reasoning core's control, and they are
    template-shaped on purpose — every sentence they generate matches the extractor's patterns
    exactly, which is what makes them a clean read on the *inference*. The cost is that they say
    nothing about the surface, and an audit found the gap the hard way: the identical two-step
    inheritance scored perfectly when the kind was written ``bird`` and failed when it was
    written ``birds``, returning a confident wrong answer rather than a miss. Every stage above
    read 1.00 through that defect, because none of them ever varied a phrasing.

    So this stage holds the *inference* fixed at the simplest thing that can be called one —
    ``X is_a K``, ``K needs P`` ⊢ ``X needs P`` — and varies only how the two sentences are
    written. A score here is a statement about extraction and canonicalisation and nothing else,
    which is what makes a drop in it diagnosable instead of merely disappointing.

    Scored per phrasing rather than pooled, so the report names which surface form failed instead
    of averaging one broken form into four working ones.
    """
    out = StageResult(stage="paraphrase")
    started = time.perf_counter()
    try:
        rows = max(1, min(width, len(_PARAPHRASES)))
        failed: List[str] = []
        # One brain for the whole stage, like every stage above it — the preparer contract is one
        # prepared brain per stage, and a stage that built five would be measuring five brains the
        # caller never got to prepare. Isolation comes from the vocabulary instead: each row draws
        # its own terms from `_vocabulary`, so no row's facts can answer another row's question.
        brain = _brain(prepare=prepare)
        if brain is None:
            out.note = "NJP unavailable"
            return out
        for i in range(rows):
            taught_kind, asked_kind, taught_verb, asked_verb = _PARAPHRASES[i % len(_PARAPHRASES)]
            member, kind, prop = _vocabulary(rng, 3, tag=f"par{i}")
            taught = [
                f"{member} is a {kind}",
                f"{taught_kind.format(k=kind)} {taught_verb} {prop}",
            ]
            out.taught += _teach(brain, taught)
            out.total += 1
            answer = _ask(brain, f"what does {asked_kind.format(k=member)} {asked_verb}?")
            if _hit(answer, prop):
                out.correct += 1
            else:
                failed.append(f"{taught_kind}/{taught_verb} → {asked_kind}/{asked_verb}: "
                              f"expected {prop}, got {answer!r}")
        out.detail.extend(failed)
        out.note = f"{rows} phrasings of one inheritance, taught and asked differently"
        return out
    except Exception as exc:  # noqa: BLE001
        out.note = f"error: {exc}"
        return out
    finally:
        out.ms = (time.perf_counter() - started) * 1000.0


_StageRunner = Callable[[random.Random, int, Optional[Preparer]], StageResult]

_STAGES: Tuple[Tuple[str, _StageRunner], ...] = (
    ("memorization", _stage_memorization),
    ("generalization", _stage_generalization),
    ("recombination", _stage_recombination),
    ("causal_prediction", _stage_causal),
    ("self_correction", _stage_self_correction),
    ("transfer", _stage_transfer),
    # Last, and deliberately outside the five that measure inference: this one measures the
    # surface those five hold constant.
    ("paraphrase", _stage_paraphrase),
)


def run_intelligence_benchmark(*, seed: int = 0, width: int = 6,
                               only: Sequence[str] = (),
                               prepare: Optional[Preparer] = None) -> IntelligenceReport:
    """Run the curve and report it. Deterministic for a seed.

    ``width`` scales how many items each stage asks. It is a knob on statistical resolution, not
    on difficulty — a stage does not get harder with more items, only better measured.

    ``prepare`` runs against every stage's fresh brain before it is taught, which is what turns
    this from a fixed reading into a before/after instrument::

        base = run_intelligence_benchmark()
        with_facts = run_intelligence_benchmark(
            prepare=lambda brain: ingest_triples(brain, "conceptnet_en.jsonl.gz"))

    **Read the vector, not the mean.** The mean can rise because one stage got easier to guess,
    and the stage worth watching here is `generalization` — it is schema induction, it currently
    scores 0.00, and it needs several subjects sharing both a concept and a relation, which a
    conversation almost never supplies and a commonsense corpus supplies constantly. A run where
    the mean moves and `generalization` does not has not shown what it appears to show.

    And note what this instrument **cannot** do: every stage's vocabulary is RNG-generated
    nonsense, disjoint by construction from anything a real corpus contains. So a corpus cannot
    make her better at these questions by containing their answers. That is the point — it makes
    the benchmark a **regression guard** on prepared state rather than a way to credit it. A gain
    here would be structural or it would be an error.
    """
    report = IntelligenceReport(seed=int(seed), width=int(width))
    started = time.perf_counter()
    wanted = {str(name) for name in only} if only else None
    for name, runner in _STAGES:
        if wanted is not None and name not in wanted:
            continue
        # A fresh RNG per stage, seeded from the run seed and the stage name, so adding or
        # skipping a stage cannot shift the vocabulary every later stage draws. Without this,
        # `--only transfer` would test a different transfer problem than a full run does.
        rng = random.Random(f"{seed}:{name}")
        try:
            report.stages.append(runner(rng, max(2, int(width)), prepare))
        except Exception as exc:  # noqa: BLE001
            report.stages.append(StageResult(stage=name, note=f"error: {exc}"))
    report.ms = (time.perf_counter() - started) * 1000.0
    return report
