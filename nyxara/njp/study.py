"""NYXARA · njp/study.py — teaching NJP from a corpus, and measuring what she actually took (📚).

Every other module in :mod:`nyxara.njp` learns from the Master a turn at a time. This one learns
from a *corpus*, and the difference that matters is not volume — it is that a corpus can be split.
Thirty-five thousand question/answer pairs give her something no conversation can: a held-out set
she has provably never seen, which is the only way "she learned it" stops being a claim.

**What "training" means for this brain.** There are no gradients on a language model here, because
there is no language model. Teaching NJP one pair does five separate things, and each is a
different organ getting evidence:

* **Grounding** — the answer is a statement, so :mod:`nyxara.njp.grounding` extracts relations
  from it and they enter the fact store. This is the part that later answers a question.
* **Association** — the question is bound to its answer in :mod:`nyxara.njp.memory`, so a question
  she has never seen can still reach a neighbouring one.
* **Concepts** — the answer's relations feed :mod:`nyxara.njp.concepts`, which invents kinds over
  the corpus's vocabulary and compresses them.
* **The fabric** — every turn potentiates what fired together and grows new synapses where a
  causal pair had none.
* **The readout** — :mod:`nyxara.njp.learn` takes a real gradient step per turn.

**The exam is the point.** :meth:`Tutor.exam` asks held-out questions and scores the reply against
the gold answer by content-word F1. Three outcomes are reported separately and never merged:
answered-and-right, answered-and-wrong, and **abstained**. Abstention is not failure here — this
brain is built to refuse what it cannot ground, and folding a principled "I don't know" into the
error rate would punish exactly the behaviour the rest of the codebase works to guarantee. A
system that abstains 90% of the time and is right on the rest is a different animal from one that
guesses 90% of the time, and one number cannot tell them apart.

**On honesty about the ceiling.** This is a fact store, a growing automaton and a gradient-trained
readout — not a generative model. It will answer questions whose facts it extracted and abstain on
the rest, and the measured numbers will say so. Nothing here will make her fluent; it makes her
*grounded in a domain*, which is a smaller and checkable claim.

**And about the denominator.** The bundled corpus is an instruction-following dataset, not the
"AI question/answer" set this file used to claim (see :data:`DEFAULT_CORPUS`): about 36% of it
asks her to *write* something and another 38% to compute or transform something. Neither has an
answer a fact store can hold. So a coverage number over the whole corpus is mostly measuring
items that are unreachable by construction, and it will stay low no matter how good extraction
becomes. Two figures are worth more than one here: extraction quality (:meth:`Grounder.stats`
reports ``unparsed`` and ``extraction_rate``) and answering quality on the factual subset.

Scoring has the mirror-image problem. Gold answers average 25.3 content words of prose while a
fact store answers in three or four, so symmetric F1 fails a *correct* short answer about two
times in three. :func:`_f1` and :func:`_hit` are therefore both reported: F1 asks whether the
reply was right **and** complete, ``_hit`` asks only whether it was right. Neither is the score
on its own, and the gap between them is itself the reading.

Pure standard library. No LLM anywhere in the path.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

__all__ = ["Pair", "Corpus", "StudyReport", "ExamReport", "Tutor", "SeedReport",
           "seed_kinds", "save_state", "load_state", "main",
           "DEFAULT_CORPUS", "DEFAULT_KINDS"]

#: The bundled corpus: 35,693 deduplicated instruction/response pairs, general-domain.
#:
#: It was described here as "question/answer pairs on artificial intelligence", and measurement
#: says it is neither. Classified over an 8,000-pair sample:
#:
#:   * **35.8% generative** — "Design a poster", "Summarize this book", "Recommend a poem".
#:     There is no fact of the matter to store, so a fact store cannot answer these at all.
#:   * **37.6% other** — "Find the volume of a cone with height 10cm", "Arrange these words
#:     alphabetically". Computation and transformation, mostly not lookup either.
#:   * **26.6% factual** — "What is X", "Name two advantages of Y". These are the answerable part.
#:
#: Head words are `generate 3461 · create 2679 · describe 2523 · write 1982`, and the sampled
#: content spans cafés, Olympic swimmers and dinosaurs. It is an instruction-following dataset.
#:
#: This matters for how the exam is read, not just for accuracy of description: **roughly
#: three-quarters of it is not answerable by lookup at any quality of extraction**, so a coverage
#: figure over the whole corpus has a denominator that is mostly unreachable by construction.
#: Report against the factual subset, or say which denominator is in use.
DEFAULT_CORPUS = Path(__file__).with_name("data") / "ai_qa.jsonl.gz"

#: Kind-level facts, which no corpus of instances ever states.
#:
#: Measured on the bundled corpus twice over: of 92 distinct kinds she extracts, **not one** has a
#: single property stated about it, and across every kind with two or more members there is **not
#: one** (predicate, object) pair two members agree on. So a corpus of instances can teach her
#: that a blue whale is an animal and can never teach her that animals need food — and without the
#: second, `core._inherit` has nothing to inherit and `answers_given` is structurally zero however
#: correct the machinery is.
#:
#: Shipped for the same reason `eval/data/holdout_realworld.jsonl` is: some evidence has to come
#: from outside the generator. Every line is a plain, checkable claim about a *kind*, it enters as
#: **testimony** with a confidence below a direct observation, and it carries ``source="seed"`` so
#: nothing can mistake it for something she worked out. Checked against the held-out set: the
#: closest any seed fact comes to answering a held-out question is F1 0.364, under the 0.4 mark.
DEFAULT_KINDS = Path(__file__).with_name("data") / "kinds.jsonl"

#: Below what a parsed statement earns. A claim about a whole kind is defeasible by construction —
#: most birds fly, and a penguin is still a bird — so anything inheriting from one should start
#: from a number that already says the claim admits exceptions.
_SEED_CONFIDENCE = 0.8

# Function words carry no topic, so they are excluded from scoring. Grading on them would let a
# reply score well by containing "the" and "of" — which is how a bag-of-words metric flatters a
# system that has learned nothing.
_STOP = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "of", "to", "in", "on",
    "at", "for", "with", "and", "or", "but", "it", "its", "this", "that", "these", "those", "as",
    "by", "from", "can", "could", "will", "would", "may", "might", "such", "which", "who", "what",
    "when", "where", "how", "why", "there", "their", "them", "they", "has", "have", "had", "do",
    "does", "did", "not", "no", "if", "then", "than", "so", "also", "more", "most", "other",
})


def _content(text: Any) -> Set[str]:
    out = {w for w in "".join(c if c.isalnum() else " " for c in str(text or "").lower()).split()
           if w not in _STOP and len(w) > 2}
    return out


def _round_or_none(value: Optional[float], digits: int) -> Optional[float]:
    """Round, but keep None as None — an absent organ must not report a number."""
    return None if value is None else round(value, digits)


def _f1(gold: Set[str], said: Set[str]) -> float:
    """Content-word F1. Symmetric, so neither a terse nor a rambling reply is rewarded for it."""
    if not gold or not said:
        return 0.0
    shared = len(gold & said)
    if not shared:
        return 0.0
    precision = shared / len(said)
    recall = shared / len(gold)
    return 2 * precision * recall / (precision + recall)


def _hit(gold: Set[str], said: Set[str]) -> float:
    """How much of what she *said* is in the gold answer. Precision only, no length penalty.

    :func:`_f1` is symmetric, and against this corpus that quietly makes it a test of verbosity.
    Gold answers here average **25.3 content words** of prose. A fact store answers in three or
    four. Measured over 200 held-out pairs, a reply consisting of four correct content words
    drawn from the gold answer scores mean F1 **0.369** and clears the 0.4 bar only **36%** of
    the time — and even quoting the gold answer's own first sentence verbatim clears it only 70%.
    So a *perfectly correct* short answer fails the symmetric metric roughly two times in three,
    and `accuracy_overall` is capped near 0.36 however good the extraction gets.

    This is the other half of the picture, not a replacement: it asks whether what she said was
    *right*, where F1 asks whether it was right **and** complete. A reply that pads itself with
    noise loses here exactly as it does there, because noise words are not in the gold set. What
    it does not do is punish her for being brief when brevity is correct.

    Both are reported. Neither is the score on its own: F1 alone calls a correct short answer
    wrong, and this alone calls "water" a complete answer to a question about the water cycle.
    """
    if not gold or not said:
        return 0.0
    return len(gold & said) / len(said)


# --------------------------------------------------------------------------- #
# The corpus
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Pair:
    """One question and its answer."""

    question: str = ""
    answer: str = ""

    @property
    def key(self) -> str:
        return hashlib.blake2b(self.question.lower().encode("utf-8"),
                               digest_size=8).hexdigest()

    @property
    def held_out(self) -> bool:
        """Deterministic split, by a hash of the question.

        Hashed rather than counted so the split is stable across runs, across restarts, and
        across a corpus that grew — a held-out set that reshuffles is a held-out set that has
        already leaked.
        """
        return int(self.key[:4], 16) % 100 < Corpus.HOLDOUT_PERCENT

    def to_dict(self) -> Dict[str, Any]:
        return {"question": self.question, "answer": self.answer}


class Corpus:
    """Loads question/answer pairs from ``.jsonl``, ``.jsonl.gz`` or a plain ``.json`` list."""

    #: Share of the corpus reserved for the exam and never studied.
    HOLDOUT_PERCENT = 10

    @staticmethod
    def _rows(path: Path) -> Iterator[Dict[str, Any]]:
        """Rows of the corpus file, with a corrupt line treated as fatal.

        The format handling lives in :func:`nyxara.njp.ingest.stream_rows` so there is one
        implementation of it rather than two. The error policy stays here because it genuinely
        differs: a bad line in *this* corpus means the bundled file is wrong and the run should
        stop, where a bad line in a harvested third-party dump means one row of a quarter-million
        is wrong and the rest are fine.
        """
        from nyxara.njp.ingest import stream_rows
        yield from stream_rows(path, on_error="raise")

    @classmethod
    def load(cls, path: Any = DEFAULT_CORPUS, *, limit: Optional[int] = None,
             sample: bool = True, seed: int = 0) -> List[Pair]:
        """Read pairs, tolerating both this corpus's key names and the original's.

        ``limit`` **samples** rather than truncating. Taking the first N lines of a corpus that
        was written in some order is not a smaller corpus, it is a different one: the bundled
        corpus is grouped by topic, so ``--limit 1500`` used to mean "everything she knows is
        about whatever the file happens to open with". Reservoir sampling (Vitter R) draws a
        uniform subset in one pass without holding the file in memory, seeded so the same
        ``(limit, seed)`` yields the same subset on every machine and every restart — which is
        what makes a before/after comparison across two runs mean anything.

        ``sample=False`` restores the old head-of-file behaviour for a caller that genuinely
        wants the first N rows.
        """
        out: List[Pair] = []
        try:
            if limit is None or limit <= 0 or not sample:
                for row in cls._rows(Path(path)):
                    pair = cls._pair(row)
                    if pair is None:
                        continue
                    out.append(pair)
                    if limit is not None and limit > 0 and len(out) >= limit:
                        break
                return out

            rng = random.Random(seed)
            seen = 0
            for row in cls._rows(Path(path)):
                pair = cls._pair(row)
                if pair is None:
                    continue
                seen += 1
                if len(out) < limit:
                    out.append(pair)
                    continue
                j = rng.randrange(seen)          # uniform over everything read so far
                if j < limit:
                    out[j] = pair
        except FileNotFoundError:
            return out
        return out

    @staticmethod
    def _pair(row: Dict[str, Any]) -> Optional[Pair]:
        """One row → one pair, or None when either half is missing."""
        question = " ".join(str(row.get("question") or row.get("Question") or "").split())
        answer = " ".join(str(row.get("answer") or row.get("Answer") or "")
                          .replace("\\n", " ").split())
        if not question or not answer:
            return None
        return Pair(question=question, answer=answer)

    @staticmethod
    def split(pairs: Sequence[Pair]) -> Tuple[List[Pair], List[Pair]]:
        """``(study, exam)``. The exam half is never fed to any organ."""
        study = [p for p in pairs if not p.held_out]
        exam = [p for p in pairs if p.held_out]
        return study, exam


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
@dataclass
class StudyReport:
    """What a study pass moved. Before/after on every organ that should have changed."""

    studied: int = 0
    skipped: int = 0
    facts_before: int = 0
    facts_after: int = 0
    concepts_before: int = 0
    concepts_after: int = 0
    compression_before: float = 1.0
    compression_after: float = 1.0
    beliefs_after: int = 0
    cells_before: int = 0
    cells_after: int = 0
    synapses_before: int = 0
    synapses_after: int = 0
    consolidations: int = 0
    ms: float = 0.0
    # The gradient learner. It trains one step per pair (`integrate.py`), and until now its
    # progress was invisible to the only command that drives it — the study report described
    # every organ that moved *except* the one actually doing gradient descent.
    readout_loss_before: Optional[float] = None
    readout_loss_after: Optional[float] = None
    readout_steps: int = 0
    readout_width_before: int = 0
    readout_width: int = 0

    @staticmethod
    def _per_slot(loss: Optional[float], width: int) -> Optional[float]:
        """Loss per slot — the only form of this number that survives the readout growing.

        ``Readout._loss`` is binary cross-entropy **summed over slots** and divided by the batch
        size, so it scales with ``width``. The readout doubles its width and rehashes whenever the
        cells it has mapped pass a load factor, which on a corpus happens repeatedly: measured over
        500 pairs the width went 512 → 8192 and the raw loss went 33.50 → 197.66, which reads as
        training having made it six times worse. Per slot those same two numbers are 0.0654 →
        0.0241 — it improved by a factor of nearly three. Dividing by width is what makes the two
        ends of a pass comparable at all.
        """
        if loss is None or width <= 0:
            return None
        return loss / width

    @property
    def readout_learned(self) -> Optional[float]:
        """Per-slot loss drop over the pass. Positive means it got better."""
        before = self._per_slot(self.readout_loss_before, self.readout_width_before)
        after = self._per_slot(self.readout_loss_after, self.readout_width)
        if before is None or after is None:
            return None
        return before - after

    @property
    def facts_learned(self) -> int:
        return self.facts_after - self.facts_before

    @property
    def grew(self) -> bool:
        """Did anything durable actually change? A pass that moved nothing says so."""
        return bool(self.facts_learned > 0 or self.concepts_after > self.concepts_before
                    or self.synapses_after > self.synapses_before)

    def to_dict(self) -> Dict[str, Any]:
        return {"studied": self.studied, "skipped": self.skipped,
                "facts": [self.facts_before, self.facts_after],
                "facts_learned": self.facts_learned,
                "concepts": [self.concepts_before, self.concepts_after],
                "compression": [round(self.compression_before, 4),
                                round(self.compression_after, 4)],
                "beliefs": self.beliefs_after,
                "cells": [self.cells_before, self.cells_after],
                "synapses": [self.synapses_before, self.synapses_after],
                "consolidations": self.consolidations,
                "readout": {
                    # Raw first, because it is what the organ reports; per-slot second, because
                    # it is the one that means anything once the width has changed under it.
                    "loss": [self.readout_loss_before, self.readout_loss_after],
                    "loss_per_slot": [
                        _round_or_none(self._per_slot(self.readout_loss_before,
                                                      self.readout_width_before), 6),
                        _round_or_none(self._per_slot(self.readout_loss_after,
                                                      self.readout_width), 6)],
                    "learned_per_slot": _round_or_none(self.readout_learned, 6),
                    "steps": self.readout_steps,
                    "width": [self.readout_width_before, self.readout_width]},
                "grew": self.grew, "ms": round(self.ms, 1),
                "ms_per_pair": round(self.ms / self.studied, 2) if self.studied else None}


@dataclass
class ExamReport:
    """What she could actually answer, on questions she was never taught."""

    asked: int = 0
    answered: int = 0
    correct: int = 0
    abstained: int = 0
    total_f1: float = 0.0
    # The brevity-aware half. `correct` is F1 ≥ threshold and stays the headline; `on_topic` is
    # the same answers judged by whether what she said was *right* rather than also complete.
    # Reported together because the gap between them is itself the reading: a wide gap means she
    # is answering correctly and tersely, a narrow one means she is genuinely missing.
    on_topic: int = 0
    total_hit: float = 0.0
    best: List[Tuple[str, float]] = field(default_factory=list)
    threshold: float = 0.4
    ms: float = 0.0

    @property
    def coverage(self) -> float:
        """Share of questions she was willing to answer at all."""
        return (self.answered / self.asked) if self.asked else 0.0

    @property
    def precision(self) -> float:
        """Of the ones she answered, how many were right. The number that matters most.

        Kept separate from coverage on purpose: a brain that abstains honestly and is right when
        it speaks is the design goal, and averaging the two into one 'accuracy' hides exactly
        that distinction.
        """
        return (self.correct / self.answered) if self.answered else 0.0

    @property
    def accuracy(self) -> float:
        """Right answers over *all* questions asked, abstentions counted as not-right."""
        return (self.correct / self.asked) if self.asked else 0.0

    @property
    def mean_f1(self) -> float:
        return (self.total_f1 / self.answered) if self.answered else 0.0

    @property
    def on_topic_rate(self) -> float:
        """Of the ones she answered, how many were *right* — brevity not penalised."""
        return (self.on_topic / self.answered) if self.answered else 0.0

    @property
    def mean_hit(self) -> float:
        return (self.total_hit / self.answered) if self.answered else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"asked": self.asked, "answered": self.answered, "correct": self.correct,
                "abstained": self.abstained,
                "coverage": round(self.coverage, 4),
                "precision_when_answered": round(self.precision, 4),
                "accuracy_overall": round(self.accuracy, 4),
                "mean_f1_when_answered": round(self.mean_f1, 4),
                # Brevity-aware. A correct three-word answer to a twenty-five-word gold answer
                # scores near 1.0 here and near 0.3 on F1; the pair is what tells a terse right
                # answer from a wrong one, which neither number does alone.
                "on_topic_when_answered": round(self.on_topic_rate, 4),
                "mean_hit_when_answered": round(self.mean_hit, 4),
                "threshold": self.threshold, "ms": round(self.ms, 1),
                "best": [[q[:70], round(s, 3)] for q, s in self.best[:5]]}


# --------------------------------------------------------------------------- #
# The tutor
# --------------------------------------------------------------------------- #
class Tutor:
    """Feeds a corpus into NJP's organs, then examines her on what was held back.

    The cadence knobs exist because a corpus is not a conversation. Concept crystallisation is
    ``O(observations)`` and re-running it every fourth turn — right for a chat, where the
    observation count is small and the Master is waiting — costs hours over thirty-five thousand
    pairs and gains nothing, since the concepts barely move between adjacent examples.
    """

    def __init__(self, brain: Any, *, seed: bool = True, crystallise_every: int = 250,
                 consolidate_every: int = 1000, max_chars: int = 400,
                 f1_threshold: float = 0.4) -> None:
        self.brain = brain
        self.crystallise_every = max(1, int(crystallise_every))
        self.consolidate_every = max(1, int(consolidate_every))
        self.max_chars = max(80, int(max_chars))
        self.f1_threshold = float(f1_threshold)
        self.studied = 0
        # Seeded once, before the first pair, and on by default.
        #
        # A corpus of instances cannot teach the properties of kinds — measured, not assumed: of
        # 92 kinds extracted from the bundled corpus, none has a property stated about it, and no
        # two members of any kind agree on one. Teaching from such a corpus with no kind layer
        # leaves `core._inherit` with nothing to inherit, which is exactly the state it was in.
        #
        # Off by `seed=False` for a caller that wants to measure the corpus alone.
        self.seed = bool(seed)
        self.seeded: Optional[SeedReport] = None

    # ---- reading the organs ------------------------------------------------- #
    def _snapshot(self) -> Dict[str, Any]:
        stats = self.brain.stats()
        genesis = stats.get("concepts") or {}
        # Absent when numpy is missing or the organ is gated off. Reported as None rather than
        # 0.0, so "no readout" cannot be mistaken for "trained to zero loss".
        readout = stats.get("readout") or {}
        last = readout.get("last") or {}
        loss = last.get("loss_after")
        return {
            "facts": int((stats.get("grounding") or {}).get("facts", 0)),
            "concepts": int(genesis.get("concepts", 0)),
            "compression": float(genesis.get("compression", 1.0)),
            "beliefs": int((stats.get("beliefs") or {}).get("beliefs", 0)),
            "cells": int((stats.get("fabric") or {}).get("cells", 0)),
            "synapses": int((stats.get("fabric") or {}).get("synapses", 0)),
            "readout_loss": float(loss) if isinstance(loss, (int, float)) else None,
            "readout_steps": int(readout.get("steps", 0) or 0),
            "readout_width": int(readout.get("width", 0) or 0),
        }

    # ---- study --------------------------------------------------------------- #
    def study(self, pairs: Iterable[Pair], *, progress: Any = None,
              checkpoint: Any = None, checkpoint_every: int = 0) -> StudyReport:
        """Teach every pair. The answer is grounded; the question is bound to it.

        ``checkpoint`` is called with the count so far every ``checkpoint_every`` pairs. A long
        corpus is hours of work and the state only becomes durable when someone writes it down —
        measured the hard way: a 30-minute run was cut off by its own time budget and lost
        everything, because saving happened once, at an end it never reached.
        """
        rep = StudyReport()
        # Before the snapshot, so the kind layer is part of what she starts from rather than
        # showing up as growth the corpus produced.
        if self.seed and self.seeded is None:
            self.seeded = seed_kinds(self.brain)
        before = self._snapshot()
        rep.facts_before = before["facts"]
        rep.concepts_before = before["concepts"]
        rep.compression_before = before["compression"]
        rep.cells_before = before["cells"]
        rep.synapses_before = before["synapses"]
        rep.readout_loss_before = before["readout_loss"]
        rep.readout_width_before = before["readout_width"]
        steps_before = before["readout_steps"]

        # Crystallisation is throttled for the whole pass and restored afterwards, so a tutored
        # brain is left in exactly the conversational posture it had before being tutored.
        field_organ = getattr(self.brain, "field", None)
        restore = getattr(field_organ, "crystallise_every", None)
        if field_organ is not None:
            field_organ.crystallise_every = self.crystallise_every

        t0 = time.perf_counter()
        try:
            for pair in pairs:
                if not pair.question or not pair.answer:
                    rep.skipped += 1
                    continue
                if pair.held_out:
                    # Never studied, whatever the caller passed. The split is enforced here as
                    # well as at the caller, because a leak into the exam set is silent and
                    # makes every number after it meaningless.
                    rep.skipped += 1
                    continue
                self._teach(pair)
                self.studied += 1
                rep.studied += 1
                if self.studied % self.consolidate_every == 0:
                    rep.consolidations += self._consolidate()
                if progress is not None and rep.studied % 500 == 0:
                    try:
                        progress(rep.studied)
                    except Exception:  # noqa: BLE001
                        pass
                if (checkpoint is not None and checkpoint_every > 0
                        and rep.studied % checkpoint_every == 0):
                    try:
                        checkpoint(rep.studied)
                    except Exception:  # noqa: BLE001 — a failed save costs durability, not the run
                        pass
        finally:
            if field_organ is not None and restore is not None:
                field_organ.crystallise_every = restore
            rep.ms = (time.perf_counter() - t0) * 1000.0

        # One final crystallisation, so the reported concept count reflects everything studied
        # rather than whatever the last throttled pass happened to catch.
        genesis = getattr(self.brain, "genesis", None)
        if genesis is not None:
            try:
                genesis.crystallise()
            except Exception:  # noqa: BLE001
                pass

        after = self._snapshot()
        rep.facts_after = after["facts"]
        rep.concepts_after = after["concepts"]
        rep.compression_after = after["compression"]
        rep.beliefs_after = after["beliefs"]
        rep.cells_after = after["cells"]
        rep.synapses_after = after["synapses"]
        rep.readout_loss_after = after["readout_loss"]
        rep.readout_steps = max(0, after["readout_steps"] - steps_before)
        rep.readout_width = after["readout_width"]
        return rep

    def _teach(self, pair: Pair) -> None:
        """One pair through every organ that can learn from it."""
        answer = pair.answer[: self.max_chars]
        try:
            self.brain.think(answer)
        except Exception:  # noqa: BLE001 — one bad example never stops a corpus
            return
        # Bind the question to its answer. Without this a held-out question can only be reached
        # through the facts the answer happened to yield; with it, a near-miss question finds the
        # answer it belongs to.
        try:
            memory = getattr(self.brain, "memory", None)
            if memory is not None:
                memory.remember(pair.key, answer, kind="fact", cue=pair.question)
        except Exception:  # noqa: BLE001
            pass
        try:
            levels = getattr(self.brain, "levels", None)
            if levels is not None:
                levels.remember(pair.key, answer, cue=pair.question, source="corpus")
        except Exception:  # noqa: BLE001
            pass

    def _consolidate(self) -> int:
        try:
            levels = getattr(self.brain, "levels", None)
            if levels is None:
                return 0
            report = levels.consolidate()
            return 1 if getattr(report, "changed", False) else 0
        except Exception:  # noqa: BLE001
            return 0

    # ---- exam ----------------------------------------------------------------- #
    def exam(self, pairs: Iterable[Pair], *, limit: Optional[int] = None) -> ExamReport:
        """Ask held-out questions and score what comes back. Abstention counted, not punished."""
        rep = ExamReport(threshold=self.f1_threshold)
        t0 = time.perf_counter()
        scored: List[Tuple[str, float]] = []
        for pair in pairs:
            if limit is not None and rep.asked >= limit:
                break
            rep.asked += 1
            said = ""
            try:
                thought = self.brain.think(pair.question, remember=False)
                said = str(getattr(thought, "answer", "") or "")
            except Exception:  # noqa: BLE001
                said = ""
            if not said.strip():
                rep.abstained += 1
                continue
            rep.answered += 1
            gold, spoken = _content(pair.answer), _content(said)
            score = _f1(gold, spoken)
            rep.total_f1 += score
            hit = _hit(gold, spoken)
            rep.total_hit += hit
            if hit >= self.f1_threshold:
                rep.on_topic += 1
            scored.append((pair.question, score))
            if score >= self.f1_threshold:
                rep.correct += 1
        rep.best = sorted(scored, key=lambda kv: kv[1], reverse=True)[:8]
        rep.ms = (time.perf_counter() - t0) * 1000.0
        return rep


# --------------------------------------------------------------------------- #
# Persisting a run
# --------------------------------------------------------------------------- #
def save_state(brain: Any, path: Any) -> Path:
    """Write the brain's state so that an interrupted write cannot destroy the last good one.

    A checkpoint exists for exactly one situation: the process dies mid-run. Writing straight
    into the destination means the process can die *during that write*, leaving a truncated file
    where nine hours of training used to be — the one moment the checkpoint was for is the one
    moment it is most likely to be half-written. So: write a temp file in the same directory,
    flush it to the platter, then ``os.replace``, which is atomic on POSIX and Windows alike.

    This raises on failure. The caller decides whether a failed save is worth stopping for; what
    it must not do is what the old code did — swallow the exception and print a success line for
    a file that was never written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(brain.to_dict(), fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def load_state(brain: Any, path: Any) -> bool:
    """Restore a saved state into ``brain``. False when there is nothing to restore."""
    source = Path(path)
    if not source.exists():
        return False
    brain.load_dict(json.loads(source.read_text(encoding="utf-8")))
    return True


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m nyxara.njp.study [--limit N] [--exam N] [--corpus PATH] [--save PATH]``."""
    import argparse

    parser = argparse.ArgumentParser(description="Teach NJP from a question/answer corpus.")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--limit", type=int, default=2000,
                        help="pairs to SAMPLE from the corpus (study + exam); 0 = all")
    parser.add_argument("--sample-seed", type=int, default=0,
                        help="seed for the corpus subsample, so a run is reproducible")
    parser.add_argument("--head", action="store_true",
                        help="take the corpus's first --limit rows instead of sampling them")
    parser.add_argument("--exam", type=int, default=200, help="held-out questions to ask")
    parser.add_argument("--crystallise-every", type=int, default=250)
    parser.add_argument("--consolidate-every", type=int, default=1000)
    parser.add_argument("--max-chars", type=int, default=400,
                        help="how much of an answer is fed to the organs")
    parser.add_argument("--f1-threshold", type=float, default=0.4,
                        help="content-word F1 at which an exam answer counts as correct")
    parser.add_argument("--no-seed", action="store_true",
                        help="skip the kind-level seed and measure the corpus alone")
    parser.add_argument("--save", default="", help="write the trained brain's state here")
    parser.add_argument("--load", default="",
                        help="start from a saved state instead of a fresh brain")
    parser.add_argument("--resume", action="store_true",
                        help="load --save if it exists, then keep training into it")
    parser.add_argument("--checkpoint-every", type=int, default=2000,
                        help="re-save the state every N pairs, so a long run cannot lose it all")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from nyxara.njp.brain import NJPBrain

    pairs = Corpus.load(args.corpus, limit=(args.limit or None),
                        sample=not args.head, seed=args.sample_seed)
    if not pairs:
        print(f"no pairs loaded from {args.corpus}")
        return 1
    study_set, exam_set = Corpus.split(pairs)
    how = "first" if args.head else f"sampled (seed {args.sample_seed})"
    print(f"corpus {args.corpus}: {len(pairs)} pairs {how} "
          f"→ {len(study_set)} study / {len(exam_set)} held out")

    brain = NJPBrain()

    # `--resume` reads the file `--save` will be written back to; `--load` is an explicit source.
    # Resuming a run that has not started yet is not an error — it is the first run.
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

    tutor = Tutor(brain, seed=not args.no_seed,
                  crystallise_every=args.crystallise_every,
                  consolidate_every=args.consolidate_every,
                  max_chars=args.max_chars,
                  f1_threshold=args.f1_threshold)

    print("\n— exam BEFORE studying (the control) —")
    before = tutor.exam(exam_set, limit=args.exam)
    print(json.dumps(before.to_dict(), indent=1))

    print("\n— studying —")
    started = time.perf_counter()

    def _progress(n: int) -> None:
        rate = (time.perf_counter() - started) / n * 1000.0
        print(f"  {n}/{len(study_set)} pairs  ({rate:.0f} ms/pair)", flush=True)

    def _checkpoint(n: int) -> None:
        if not args.save:
            return
        # Loud on failure. A checkpoint that silently does nothing is worse than none at all,
        # because the run continues believing it is safe to be interrupted.
        try:
            save_state(brain, args.save)
        except Exception as exc:  # noqa: BLE001
            print(f"  CHECKPOINT FAILED at {n}: {exc}", flush=True)
            return
        print(f"  checkpoint at {n} → {args.save}", flush=True)

    report = tutor.study(study_set, progress=_progress, checkpoint=_checkpoint,
                         checkpoint_every=args.checkpoint_every)
    print(json.dumps(report.to_dict(), indent=1))

    print("\n— exam AFTER studying (same held-out questions) —")
    after = tutor.exam(exam_set, limit=args.exam)
    print(json.dumps(after.to_dict(), indent=1))

    print(f"\ncoverage {before.coverage:.3f} → {after.coverage:.3f}   "
          f"precision {before.precision:.3f} → {after.precision:.3f}   "
          f"overall {before.accuracy:.3f} → {after.accuracy:.3f}")

    if args.save:
        try:
            save_state(brain, args.save)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED to write state to {args.save}: {exc}")
            return 1
        print(f"state written to {args.save}")
    return 0


# --------------------------------------------------------------------------- #
# The half a corpus of instances cannot teach
# --------------------------------------------------------------------------- #
@dataclass
class SeedReport:
    """What seeding the kind-level facts actually put in the store."""

    read: int = 0
    asserted: int = 0
    skipped: int = 0
    kinds: int = 0
    ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"read": self.read, "asserted": self.asserted, "skipped": self.skipped,
                "kinds": self.kinds, "ms": round(self.ms, 3)}


def seed_kinds(brain: Any, path: Any = DEFAULT_KINDS) -> SeedReport:
    """Teach the properties of kinds, which a corpus of instances never states.

    Asserted straight into the fact store rather than pushed through ``think``. These are not
    turns — nobody said them to her, and running them as conversation would put 137 fabricated
    exchanges into her episodic memory and her fabric's growth record, which would make the
    session log a lie about what happened.

    Marked ``source="seed"`` throughout, so provenance separates it from anything she extracted,
    and held below the confidence a parsed statement earns: a claim about a whole kind is
    defeasible by construction — most birds fly, and a penguin is still a bird — and the store
    should say so before anything inherits from it.

    Fed to the concept layer as well as the fact store, because a kind with properties is exactly
    the material :mod:`nyxara.njp.concepts` needs and has never had.
    """
    report = SeedReport()
    t0 = time.perf_counter()
    try:
        from nyxara.njp.ingest import ingest_triples

        # The mechanism is `njp/ingest.py`, which generalised what this function did at 137 rows
        # to a corpus that does not fit in memory. Delegating rather than keeping a second copy
        # is the point: there is one bulk path into the fact store, so a fix to folding, dedup or
        # batching reaches the seed file and the corpus at the same time.
        #
        # `to_world=False`: these are properties of kinds, not statements of mechanism. "bird
        # is_a animal" is not a law about how the world moves and does not belong on the causal
        # skeleton, and the seed file never routed to the world model before this either.
        got = ingest_triples(brain, path, source="seed", min_confidence=0.0,
                             to_world=False, default_confidence=_SEED_CONFIDENCE)
        report.read = got.read
        report.asserted = got.asserted
        report.skipped = got.skipped + got.duplicate
        report.kinds = got.subjects
        return report
    except Exception:  # noqa: BLE001 — a failed seed leaves the store exactly as it was
        return report
    finally:
        report.ms = (time.perf_counter() - t0) * 1000.0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
#
# This guard belongs at the *end* of the module, below every name `main` reaches.
#
# It used to sit directly under `main` (above `SeedReport`/`seed_kinds`), and under
# `python -m nyxara.njp.study` that is not a style question — it is a crash. `-m` executes the
# module body top to bottom, so the guard fired while the body was still running and `main()` was
# called before `seed_kinds` had been bound:
#
#     File "nyxara/njp/study.py", line 336, in study
#         self.seeded = seed_kinds(self.brain)
#     NameError: name 'seed_kinds' is not defined
#
# Importing the module as a library ran the whole body first, so `seed_kinds` existed and every
# test passed. That is exactly why the suite never caught it, and why the regression test for this
# invokes the CLI through `subprocess` rather than by importing `main`.
if __name__ == "__main__":
    raise SystemExit(main())
