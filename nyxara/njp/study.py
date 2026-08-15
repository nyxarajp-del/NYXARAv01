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

Pure standard library. No LLM anywhere in the path.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

__all__ = ["Pair", "Corpus", "StudyReport", "ExamReport", "Tutor", "DEFAULT_CORPUS"]

#: The bundled corpus: 35,693 deduplicated question/answer pairs on artificial intelligence.
DEFAULT_CORPUS = Path(__file__).with_name("data") / "ai_qa.jsonl.gz"

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
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
            head = fh.read(1)
            fh.seek(0)
            if head == "[":                      # a plain JSON list
                for row in json.load(fh):
                    yield row
                return
            for line in fh:                      # JSON lines
                line = line.strip()
                if line:
                    yield json.loads(line)

    @classmethod
    def load(cls, path: Any = DEFAULT_CORPUS, *, limit: Optional[int] = None) -> List[Pair]:
        """Read pairs, tolerating both this corpus's key names and the original's."""
        out: List[Pair] = []
        try:
            for row in cls._rows(Path(path)):
                question = " ".join(str(row.get("question") or row.get("Question") or "").split())
                answer = " ".join(str(row.get("answer") or row.get("Answer") or "")
                                  .replace("\\n", " ").split())
                if not question or not answer:
                    continue
                out.append(Pair(question=question, answer=answer))
                if limit is not None and len(out) >= limit:
                    break
        except FileNotFoundError:
            return out
        return out

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

    def to_dict(self) -> Dict[str, Any]:
        return {"asked": self.asked, "answered": self.answered, "correct": self.correct,
                "abstained": self.abstained,
                "coverage": round(self.coverage, 4),
                "precision_when_answered": round(self.precision, 4),
                "accuracy_overall": round(self.accuracy, 4),
                "mean_f1_when_answered": round(self.mean_f1, 4),
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

    def __init__(self, brain: Any, *, crystallise_every: int = 250,
                 consolidate_every: int = 1000, max_chars: int = 400,
                 f1_threshold: float = 0.4) -> None:
        self.brain = brain
        self.crystallise_every = max(1, int(crystallise_every))
        self.consolidate_every = max(1, int(consolidate_every))
        self.max_chars = max(80, int(max_chars))
        self.f1_threshold = float(f1_threshold)
        self.studied = 0

    # ---- reading the organs ------------------------------------------------- #
    def _snapshot(self) -> Dict[str, Any]:
        stats = self.brain.stats()
        genesis = stats.get("concepts") or {}
        return {
            "facts": int((stats.get("grounding") or {}).get("facts", 0)),
            "concepts": int(genesis.get("concepts", 0)),
            "compression": float(genesis.get("compression", 1.0)),
            "beliefs": int((stats.get("beliefs") or {}).get("beliefs", 0)),
            "cells": int((stats.get("fabric") or {}).get("cells", 0)),
            "synapses": int((stats.get("fabric") or {}).get("synapses", 0)),
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
        before = self._snapshot()
        rep.facts_before = before["facts"]
        rep.concepts_before = before["concepts"]
        rep.compression_before = before["compression"]
        rep.cells_before = before["cells"]
        rep.synapses_before = before["synapses"]

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
            score = _f1(_content(pair.answer), _content(said))
            rep.total_f1 += score
            scored.append((pair.question, score))
            if score >= self.f1_threshold:
                rep.correct += 1
        rep.best = sorted(scored, key=lambda kv: kv[1], reverse=True)[:8]
        rep.ms = (time.perf_counter() - t0) * 1000.0
        return rep


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m nyxara.njp.study [--limit N] [--exam N] [--corpus PATH] [--save PATH]``."""
    import argparse

    parser = argparse.ArgumentParser(description="Teach NJP from a question/answer corpus.")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--limit", type=int, default=2000,
                        help="pairs to READ from the corpus (study + exam); 0 = all")
    parser.add_argument("--exam", type=int, default=200, help="held-out questions to ask")
    parser.add_argument("--crystallise-every", type=int, default=250)
    parser.add_argument("--save", default="", help="write the trained brain's state here")
    parser.add_argument("--checkpoint-every", type=int, default=2000,
                        help="re-save the state every N pairs, so a long run cannot lose it all")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from nyxara.njp.brain import NJPBrain

    pairs = Corpus.load(args.corpus, limit=(args.limit or None))
    if not pairs:
        print(f"no pairs loaded from {args.corpus}")
        return 1
    study_set, exam_set = Corpus.split(pairs)
    print(f"corpus {args.corpus}: {len(pairs)} pairs "
          f"→ {len(study_set)} study / {len(exam_set)} held out")

    brain = NJPBrain()
    tutor = Tutor(brain, crystallise_every=args.crystallise_every)

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
        Path(args.save).write_text(json.dumps(brain.to_dict()), encoding="utf-8")
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
        Path(args.save).write_text(json.dumps(brain.to_dict()), encoding="utf-8")
        print(f"state written to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
