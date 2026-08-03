"""NYXARA · growth/entropy.py — L-ENTROPY: adaptive curriculum noise (🌀, chaos-proofing).

A brain trained only on clean text learns to depend on the text being clean. Real input is not:
it arrives with typos, truncations, dropped words, reordered clauses and numbers that drifted. So
this module perturbs the *training* corpus on purpose, on a schedule, and measures whether the
resulting brain is genuinely more robust.

Every operator is deterministic given the seed and built from the standard library — token dropout,
character-level typos, span masking, sentence permutation, and numeric jitter. The perturbation
level anneals from zero toward ``entropy_max`` across training and scales with her measured
competence, so a weak brain sees clean data first and earns its noise.

**Two rules are not negotiable, because breaking either inverts the point of the layer.**

1. *The training split only — never the evaluation split.* The Foundry promotes a candidate on a
   strictly-lower perplexity gate. Perturb the held-out text and that gate is measuring noise
   instead of skill, and the whole promotion ladder quietly stops meaning anything. Noise is
   applied in :meth:`~nyxara.growth.foundry.Foundry.train_candidate` *after* the holdout split.
2. *In a supervised pair, only the prompt is ever corrupted — never the answer.* The training
   document is ``### User:\\n<prompt>\\n\\n### NYXARA:\\n<answer>``. Corrupting the answer would
   teach her to produce damaged answers, which is the exact opposite of robustness. Corrupting the
   prompt teaches the thing worth learning: give the right answer even when the question arrived
   broken.

The honest framing: this is data augmentation, and augmentation is a well-understood, measurable
technique. It buys real robustness to degraded input. It does not confer reasoning ability the
brain does not otherwise have — and the verification deliberately measures *both* the noisy and
the clean holdout, because an augmentation that improves noisy scores while quietly wrecking clean
ones is not a win.

Depends on ``senses/nlp`` (sentence splitting) and ``kernel/config``. Nothing imports back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from random import Random
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["PerturbSpec", "EntropyReport", "EntropyCurriculum", "perturb_corpus"]

# The supervised template's answer marker (mind/llm._SELF_ASSISTANT_TAG). Everything from here on
# is the target the model must learn to produce, and is therefore never touched.
_ANSWER_TAG = "### NYXARA:"
_MASK = "<mask>"

_WORD_RE = re.compile(r"\S+")
_NUM_RE = re.compile(r"(?<![\w.])[+-]?\d+\.?\d*(?![\w.])")

# Every operator this module knows. Listed once so config, report and dispatch cannot drift apart.
OPERATORS: Tuple[str, ...] = ("token_dropout", "char_noise", "span_mask",
                              "sentence_permute", "numeric_jitter")


@dataclass
class PerturbSpec:
    """How hard to perturb one document, and with which operators."""

    level: float = 0.0                      # 0.0 = untouched, 1.0 = maximum damage
    operators: Tuple[str, ...] = OPERATORS

    def enabled(self) -> bool:
        return self.level > 0.0 and bool(self.operators)


@dataclass
class EntropyReport:
    """What the curriculum actually did — countable, so a claim about it can be checked."""

    documents: int = 0
    perturbed: int = 0
    answers_protected: int = 0              # pairs whose answer side was held back from every op
    applied: Dict[str, int] = field(default_factory=dict)
    mean_level: float = 0.0

    def note_op(self, name: str) -> None:
        self.applied[name] = self.applied.get(name, 0) + 1

    def summary(self) -> str:
        ops = ", ".join(f"{k}×{v}" for k, v in sorted(self.applied.items())) or "none"
        return (f"· entropy: perturbed {self.perturbed}/{self.documents} training doc(s) "
                f"at mean level {self.mean_level:.2f} ({ops}); "
                f"{self.answers_protected} answer(s) protected; eval split untouched")

    def to_dict(self) -> Dict[str, Any]:
        return {"documents": self.documents, "perturbed": self.perturbed,
                "answers_protected": self.answers_protected, "applied": dict(self.applied),
                "mean_level": round(self.mean_level, 4)}


class EntropyCurriculum:
    """Injects scheduled, seeded noise into training text — and only into training text."""

    def __init__(self, *, level: float = 0.15, operators: Optional[Sequence[str]] = None,
                 seed: int = 0) -> None:
        self.max_level = max(0.0, min(1.0, float(level)))
        chosen = tuple(operators) if operators else OPERATORS
        self.operators: Tuple[str, ...] = tuple(op for op in chosen if op in OPERATORS)
        self.seed = int(seed)

    # ------------------------------------------------------------------ #
    # schedule
    # ------------------------------------------------------------------ #
    def schedule(self, *, step: int = 0, total: int = 1, competence: float = 1.0) -> float:
        """The perturbation level for this point in training.

        Anneals 0 → ``max_level`` across the run and scales with ``competence`` (her measured
        mastery tier, 0..1). A brain that cannot yet model clean text is not helped by being handed
        broken text — it earns its noise as it gets stronger."""
        total = max(1, int(total))
        progress = min(1.0, max(0.0, (int(step) + 1) / total))
        competence = min(1.0, max(0.0, float(competence)))
        return self.max_level * progress * competence

    # ------------------------------------------------------------------ #
    # operators — each takes text + level + rng, returns text
    # ------------------------------------------------------------------ #
    @staticmethod
    def _token_dropout(text: str, level: float, rng: Random) -> str:
        words = text.split()
        if len(words) < 4:
            return text
        keep = [w for w in words if rng.random() >= level]
        return " ".join(keep) if len(keep) >= 2 else text

    @staticmethod
    def _char_noise(text: str, level: float, rng: Random) -> str:
        """Typo simulation: swap adjacent characters, drop one, or duplicate one."""
        chars = list(text)
        if len(chars) < 4:
            return text
        n_edits = max(1, int(len(chars) * level * 0.5))
        for _ in range(n_edits):
            i = rng.randrange(len(chars) - 1)
            roll = rng.random()
            if roll < 0.34:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
            elif roll < 0.67:
                chars[i] = ""
            else:
                chars[i] = chars[i] * 2
        return "".join(chars)

    @staticmethod
    def _span_mask(text: str, level: float, rng: Random) -> str:
        words = text.split()
        if len(words) < 6:
            return text
        span = max(1, int(len(words) * level))
        start = rng.randrange(0, max(1, len(words) - span))
        return " ".join(words[:start] + [_MASK] + words[start + span:])

    @staticmethod
    def _sentence_permute(text: str, level: float, rng: Random) -> str:
        """Reorder sentences — the model must recover meaning that no longer arrives in order."""
        try:
            from nyxara.senses.nlp import sentences
            parts = sentences(text)
        except Exception:  # noqa: BLE001 — fall back to a plain split rather than lose the operator
            parts = [s for s in re.split(r"(?<=[.!?])\s+", text) if s]
        if len(parts) < 2:
            return text
        n_swaps = max(1, int(len(parts) * level))
        parts = list(parts)
        for _ in range(n_swaps):
            i, j = rng.randrange(len(parts)), rng.randrange(len(parts))
            parts[i], parts[j] = parts[j], parts[i]
        return " ".join(parts)

    @staticmethod
    def _numeric_jitter(text: str, level: float, rng: Random) -> str:
        """Nudge numbers by a small relative amount, preserving magnitude and sign."""
        def _jitter(match: "re.Match[str]") -> str:
            raw = match.group(0)
            try:
                value = float(raw)
            except ValueError:  # pragma: no cover — the regex already guarantees a number
                return raw
            shifted = value * (1.0 + rng.uniform(-level, level))
            return str(int(round(shifted))) if "." not in raw else f"{shifted:.4g}"
        return _NUM_RE.sub(_jitter, text)

    def _apply_one(self, text: str, spec: PerturbSpec, rng: Random,
                   report: Optional[EntropyReport]) -> str:
        """Apply exactly one randomly chosen operator — composing them destroys the text."""
        if not spec.enabled() or not text.strip():
            return text
        name = rng.choice(list(spec.operators))
        fn = {"token_dropout": self._token_dropout, "char_noise": self._char_noise,
              "span_mask": self._span_mask, "sentence_permute": self._sentence_permute,
              "numeric_jitter": self._numeric_jitter}[name]
        try:
            out = fn(text, spec.level, rng)
        except Exception:  # noqa: BLE001 — a surprising document is left alone, never fatal
            return text
        if out != text and report is not None:
            report.note_op(name)
        return out

    # ------------------------------------------------------------------ #
    # documents
    # ------------------------------------------------------------------ #
    def perturb(self, text: str, *, level: float, rng: Optional[Random] = None,
                report: Optional[EntropyReport] = None) -> str:
        """Perturb one training document, holding a supervised answer back from every operator.

        A document containing the answer marker is split there: the prompt half is perturbed, the
        answer half is returned byte-for-byte. Teaching her to emit damaged answers would be the
        opposite of what this layer is for."""
        rng = rng if rng is not None else Random(self.seed)
        spec = PerturbSpec(level=max(0.0, min(1.0, level)), operators=self.operators)
        if not spec.enabled():
            return text

        index = text.find(_ANSWER_TAG)
        if index == -1:
            return self._apply_one(text, spec, rng, report)

        head, tail = text[:index], text[index:]      # tail holds the marker AND the answer
        if report is not None:
            report.answers_protected += 1
        return self._apply_one(head, spec, rng, report) + tail

    def perturb_corpus(self, corpus: Sequence[str], *, competence: float = 1.0,
                       report: Optional[EntropyReport] = None) -> List[str]:
        """Perturb a whole **training** split. Never call this on the evaluation split.

        The level anneals across the corpus, so the easy, clean material leads and the noisier
        material follows — the same easy→hard intent as the Foundry's difficulty curriculum."""
        docs = list(corpus)
        report = report if report is not None else EntropyReport()
        report.documents += len(docs)
        if self.max_level <= 0.0 or not self.operators or not docs:
            return docs

        rng = Random(self.seed)
        out: List[str] = []
        total_level = 0.0
        for i, doc in enumerate(docs):
            level = self.schedule(step=i, total=len(docs), competence=competence)
            total_level += level
            perturbed = self.perturb(doc, level=level, rng=rng, report=report)
            if perturbed != doc:
                report.perturbed += 1
            out.append(perturbed)
        report.mean_level = total_level / len(docs)
        return out


def perturb_corpus(corpus: Sequence[str], *, settings: Any = None, competence: float = 1.0,
                   level: Optional[float] = None, seed: int = 0,
                   report: Optional[EntropyReport] = None) -> List[str]:
    """Convenience wrapper reading the configured level/operators. Never raises.

    Returns the corpus untouched when the layer is disabled, so callers can invoke it
    unconditionally."""
    cfg = getattr(settings, "foundry", None) if settings is not None else None
    if cfg is not None and not getattr(cfg, "dataset_entropy", True):
        return list(corpus)
    try:
        curriculum = EntropyCurriculum(
            level=level if level is not None else float(getattr(cfg, "entropy_max", 0.15)),
            operators=getattr(cfg, "entropy_ops", None), seed=seed)
        return curriculum.perturb_corpus(corpus, competence=competence, report=report)
    except Exception:  # noqa: BLE001 — augmentation is additive; its failure never costs a forge
        return list(corpus)
