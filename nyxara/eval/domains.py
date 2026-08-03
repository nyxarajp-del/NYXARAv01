"""NYXARA · eval/domains.py — one ruler that works across vocabularies, and four domain scores.

Two jobs, and the first one is the reason this module exists at all.

**1. ``bits_per_byte`` — a metric that survives a change of tokenizer.**

:meth:`~nyxara.growth.foundry_models.BaseLanguageModel.perplexity` is per *token*, and the
foundry's promotion gate (``growth/foundry.py::_gauntlet``) promotes a candidate only when its
perplexity beats the active model's. That comparison is sound while every brain speaks bytes.
It stops being sound the moment one of them speaks a trained sub-word vocabulary, and it fails
in the **worst possible direction**:

    A byte-level model predicting the ``e`` after ``th`` is solving a far easier problem than a
    sub-word model predicting an entire word. The byte model therefore reports a *much lower*
    per-token perplexity while being a *much worse* model. Under a naive gate, a genuinely
    better 300M brain with a real tokenizer would be refused promotion — forever — in favour of
    a 137k byte-level one.

Bits-per-byte fixes this by normalising the total cross-entropy by the **UTF-8 bytes** of the
text rather than by however many tokens a particular vocabulary happened to split it into.
Bytes are the one unit every backend here agrees on, so a Kneser-Ney word model, a byte-level
n-gram and a 32k-vocabulary transformer can all be put on the same axis and honestly ranked.
It is the standard metric for exactly this reason.

**2. Four domain scores, because "good at everything" has to be measurable.**

``general`` (held-out bits-per-byte), ``code``, ``math`` and ``conversation`` are scored
separately, so a model that quietly traded arithmetic for prose is *visible* rather than hidden
inside one averaged number. Each reuses a battery that already exists — ``hard_benchmark``'s
math and code builders and the shipped ``holdout_realworld.jsonl`` — rather than inventing a
fresh ruler this module would then be grading itself against.

Depends on ``eval/benchmark``, ``eval/hard_benchmark`` and ``eval/datasets``. Nothing imports
back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

__all__ = [
    "LN2",
    "bits_per_byte",
    "corpus_bits_per_byte",
    "DomainScores",
    "evaluate_domains",
    "DOMAINS",
]

LN2 = math.log(2.0)

# The four things the Master asked her to be good at. Order is fixed so a report always reads
# the same way, and so a regression in one is obvious next to the others.
DOMAINS = ("general", "code", "math", "conversation")


# --------------------------------------------------------------------------- #
# The cross-vocabulary metric
# --------------------------------------------------------------------------- #
def bits_per_byte(model: Any, text: str) -> float:
    """Bits of cross-entropy per UTF-8 byte of ``text`` — comparable across every backend.

    Lower is better, and the scale is interpretable on its own: ~8.0 is a model that has learned
    nothing (a uniform distribution over bytes), ~1.0 is a decent small language model, and
    below ~0.8 is genuinely strong. Because the denominator is bytes, a byte-level model and a
    32k-vocabulary model can be compared directly — which per-token perplexity cannot do.

    Returns ``inf`` for empty text or a model that cannot score it, never raises: this is used
    inside a promotion gate, and a metric that throws would take the whole forge down.
    """
    if not text:
        return float("inf")
    n_bytes = len(text.encode("utf-8", "replace"))
    if n_bytes <= 0:
        return float("inf")
    try:
        total_nats, n_tokens = model.cross_entropy_nats(text)
    except Exception:  # noqa: BLE001 — an unscorable model is infinitely bad, not a crash
        return float("inf")
    if not n_tokens or not math.isfinite(total_nats):
        return float("inf")
    return total_nats / (n_bytes * LN2)


def corpus_bits_per_byte(model: Any, texts: Sequence[str]) -> float:
    """Bits-per-byte over a whole corpus, weighted by bytes.

    Weighting by bytes rather than averaging per-document scores is deliberate: a mean of
    per-document numbers lets a hundred one-line documents outvote a long one, which makes the
    metric a function of how the corpus happens to be chunked instead of of the model.
    """
    total_nats, total_bytes = 0.0, 0
    for text in texts:
        if not text:
            continue
        n_bytes = len(text.encode("utf-8", "replace"))
        try:
            nats, n_tokens = model.cross_entropy_nats(text)
        except Exception:  # noqa: BLE001
            continue
        if not n_tokens or not math.isfinite(nats):
            continue
        total_nats += nats
        total_bytes += n_bytes
    if total_bytes <= 0:
        return float("inf")
    return total_nats / (total_bytes * LN2)


# --------------------------------------------------------------------------- #
# Domain scores
# --------------------------------------------------------------------------- #
@dataclass
class DomainScores:
    """Per-domain capability, plus the held-out bits-per-byte that gates promotion."""

    general_bpb: float = float("inf")     # held-out bits per byte — lower is better
    code: float = 0.0                     # accuracy in [0, 1] — higher is better
    math: float = 0.0
    conversation: float = 0.0
    counts: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        if message and message not in self.notes:
            self.notes.append(message)

    @property
    def weakest(self) -> str:
        """The capability domain scoring lowest — what the self-expanding MoE should target.

        ``general`` is excluded because it is measured on a different scale (bits, not
        accuracy) and comparing the two would be meaningless.
        """
        scored = {"code": self.code, "math": self.math, "conversation": self.conversation}
        return min(scored, key=lambda k: scored[k])

    def summary(self) -> str:
        bpb = ("n/a" if not math.isfinite(self.general_bpb)
               else f"{self.general_bpb:.3f} bits/byte")
        parts = [f"general {bpb}", f"code {self.code:.1%}", f"math {self.math:.1%}",
                 f"conversation {self.conversation:.1%}"]
        notes = ("\n  · " + "\n  · ".join(self.notes)) if self.notes else ""
        return "· " + " | ".join(parts) + notes

    def to_dict(self) -> Dict[str, Any]:
        return {"general_bpb": (None if not math.isfinite(self.general_bpb)
                                else round(self.general_bpb, 5)),
                "code": round(self.code, 5), "math": round(self.math, 5),
                "conversation": round(self.conversation, 5),
                "counts": dict(self.counts), "notes": list(self.notes)}


def _solver_for(model: Any, *, max_tokens: int = 96) -> Callable[[str], str]:
    """Adapt a language model to the ``Callable[[str], str]`` shape the batteries expect."""
    def solve(prompt: str) -> str:
        try:
            return model.generate(prompt, max_tokens=max_tokens) or ""
        except Exception:  # noqa: BLE001 — a generation failure is a zero, never a crash
            return ""
    return solve


def _run_battery(bench: Any, solver: Callable[[str], str], scores: DomainScores,
                 label: str) -> float:
    """Run one battery, recording its size, and never letting a failure abort the others."""
    try:
        report = bench.run(solver)
    except Exception as exc:  # noqa: BLE001
        scores.note(f"{label}: battery failed ({exc})")
        return 0.0
    scores.counts[label] = len(report)
    return report.accuracy


def evaluate_domains(model: Any, *, holdout: Optional[Sequence[str]] = None,
                     settings: Any = None, max_tokens: int = 96,
                     skip_generation: bool = False) -> DomainScores:
    """Score ``model`` on all four domains.

    ``holdout`` is the held-out text for the ``general`` bits-per-byte measurement — pass the
    same split ``Foundry._holdout`` produced, so the number is comparable with the perplexity
    the gauntlet already records.

    ``skip_generation`` scores only ``general``, which is what a promotion gate wants when the
    three generative batteries would cost more than the forge itself. Every failure degrades to
    a note and a zero rather than an exception: this runs inside the forge, and a metric that
    raises takes the whole thing down.
    """
    scores = DomainScores()

    if holdout:
        scores.general_bpb = corpus_bits_per_byte(model, holdout)
        scores.counts["general"] = len(holdout)
    else:
        scores.note("general: no holdout supplied — bits-per-byte not measured")

    if skip_generation:
        return scores

    solver = _solver_for(model, max_tokens=max_tokens)

    try:
        from nyxara.eval.hard_benchmark import build_code_benchmark, build_math_benchmark
        scores.math = _run_battery(build_math_benchmark(), solver, scores, "math")
        scores.code = _run_battery(build_code_benchmark(), solver, scores, "code")
    except Exception as exc:  # noqa: BLE001 — an unavailable battery is a note, not a failure
        scores.note(f"math/code batteries unavailable ({exc})")

    try:
        from nyxara.eval.datasets import build_realworld_benchmark
        scores.conversation = _run_battery(build_realworld_benchmark(settings), solver,
                                           scores, "conversation")
    except Exception as exc:  # noqa: BLE001
        scores.note(f"conversation battery unavailable ({exc})")

    return scores
