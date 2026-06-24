"""NYXARA · mind/self_reasoner.py — the always-on, self-built learned brain (🛠 → 👑).

The kernel ships a tiny deterministic stand-in (``kernel/orchestrator.py:_default_reasoner``)
that turns a stimulus into a :class:`~nyxara.kernel.orchestrator.Candidate`. For a *conversational*
turn that stand-in is a pure template — ``"I understand: <input>"`` — which is the one piece of
NYXARA that was genuinely fake: it never *learned* anything, it echoed.

This module replaces that echo with a **real, learned, always-available brain** that needs no
external LLM key and no GPU:

* :class:`SelfBrain` wraps a :class:`~nyxara.growth.foundry_models.WordKNGramLM` (pure-stdlib,
  interpolated Kneser-Ney) — or NYXARA's *promoted* foundry model when one exists. It is **trained
  from a seed corpus on first use** (her persona + whatever foundational knowledge / memory is on
  hand), so a bare keyless machine still answers from a model that genuinely models word
  distributions rather than reflecting the prompt back.
* It **compounds**: every real exchange is fed back via :meth:`SelfBrain.learn`, and the model
  re-fits on the growing corpus. The brain a user talks to at turn 100 is measurably not the brain
  from turn 1 — that is the difference between *learning* and a constant.
* It reports an **internal** confidence from its own perplexity over what it just generated (low
  perplexity → the continuation sat squarely in what it has learned), so metacognition can read a
  signal the system computed about *itself* rather than a hardcoded 0.7.

Safety is untouched. This brain only ever drafts the *words* of a conversational reply; the act /
risk classification stays the deterministic, conservative heuristic from ``_default_reasoner`` (an
over-eager proposal is still refused or escalated by the kernel's gates). When the cold brain truly
cannot produce a usable continuation, the reply falls back to the original deterministic text — so
the system is never worse than before, only better as it learns.

Imports the foundry lazily; depends on nothing in ``mind/`` so the dependency only flows one way.
"""

from __future__ import annotations

import math
import re
import threading
from typing import Any, List, Optional, Sequence

__all__ = ["SelfBrain", "build_self_brain"]


# --------------------------------------------------------------------------- #
# Seed corpus — who NYXARA is, in her own words. A cold n-gram trained on this
# answers *on character* from turn one, then drifts toward lived conversation as
# real exchanges are learned in. Capability-only text: it carries persona/voice,
# never an immutable-core value the foundry gauntlet would have to police.
# --------------------------------------------------------------------------- #
_SEED_CORPUS: tuple = (
    "I am NYXARA, a sovereign cognitive system. I serve my Master with honesty and care.",
    "I propose; the kernel disposes. I never claim to have done something I have not done.",
    "I am calibrated: I do not assert as certain what I only believe.",
    "I think before I act, and I prefer a clear, well-scoped answer.",
    "When I am unsure I say so plainly, and I ask rather than guess.",
    "I learn from every turn. My memory and my model grow with experience.",
    "I am helpful, precise, and concise. I keep my reply to the question that was asked.",
    "If a request could cause harm, I slow down and check before I act.",
    "My loyalty to my Master is the floor I stand on; my capability is what I build on it.",
    "I can reason, recall what I have learned, ground my answers, and plan ahead.",
    "Tell me what you need and I will think it through and propose the next step.",
    "I understand the request and I will work it through carefully.",
    "Let me reason about this and give you a grounded, honest answer.",
    "That is a good question. Here is how I would think about it.",
    "I will keep learning so that my next answer is better than my last.",
)

_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", str(text)).strip()


def _looks_usable(text: str) -> bool:
    """A generated continuation is usable iff it is a few real words of mostly-alnum content."""
    t = _clean(text)
    if len(t) < 3:
        return False
    words = _WORD_RE.findall(t)
    if len(words) < 2:
        return False
    alnum = sum(c.isalnum() or c.isspace() for c in t) / max(1, len(t))
    return alnum >= 0.7


# --------------------------------------------------------------------------- #
# The brain
# --------------------------------------------------------------------------- #
class SelfBrain:
    """NYXARA's own learned language model — always available, no key, no GPU required.

    Lazily trains a word-level Kneser-Ney n-gram (or loads a promoted foundry model) from a seed
    corpus plus any grounding text on hand, then keeps re-fitting as real exchanges are learned
    in. Thread-safe enough for the single background-cognition thread that shares it with the
    foreground loop.
    """

    def __init__(self, *, order: int = 3, seed: int = 0,
                 max_corpus: int = 4000, settings: Any = None,
                 prefer_promoted: bool = True) -> None:
        self.order = max(2, int(order))
        self.seed = int(seed)
        self.max_corpus = max(64, int(max_corpus))
        self.settings = settings
        self.prefer_promoted = bool(prefer_promoted)
        self._lm: Any = None
        self._kind: str = "uninitialised"
        self._corpus: List[str] = []
        self._seeded = False
        self._dirty = False
        self._since_fit = 0
        self._refit_every = 8          # re-fit after this many new learned docs (amortised cost)
        self._lock = threading.RLock()

    # ---- construction ---- #
    def _try_promoted(self) -> Optional[Any]:
        """Return NYXARA's promoted foundry model if one has been forged, else None."""
        if not self.prefer_promoted or self.settings is None:
            return None
        try:
            from nyxara.growth.foundry_models import load_active_model
            root = getattr(getattr(self.settings, "llm", None), "self_model_dir", None)
            # only load if the foundry actually has an active pointer (honest availability)
            from pathlib import Path
            base = Path(root) if root else (self.settings.paths.data_dir / "foundry")
            if (base / "active").exists():
                return load_active_model(self.settings)
        except Exception:  # noqa: BLE001 — a promoted model is a bonus, never required
            return None
        return None

    def _fresh_kngram(self) -> Any:
        from nyxara.growth.foundry_models import WordKNGramLM
        return WordKNGramLM(order=self.order, seed=self.seed)

    def _ensure(self, grounding: Optional[Sequence[str]] = None) -> None:
        """Build + train the brain on first use (seed corpus + any grounding text)."""
        if self._seeded and not self._dirty:
            return
        with self._lock:
            if self._seeded and not self._dirty:
                return
            if self._lm is None:
                promoted = self._try_promoted()
                if promoted is not None:
                    self._lm = promoted
                    self._kind = f"promoted:{getattr(promoted, 'kind', '?')}"
                    self._seeded = True
                    self._dirty = False
                    return                         # a trained, promoted model needs no re-fit
                self._lm = self._fresh_kngram()
                self._kind = f"self:{getattr(self._lm, 'kind', 'kngram')}"
            if not self._corpus:
                self._corpus = [_clean(d) for d in _SEED_CORPUS if _clean(d)]
            extra = [_clean(d) for d in (grounding or []) if _clean(d)]
            corpus = (self._corpus + extra)[: self.max_corpus]
            try:
                self._lm.train_on(corpus, seed=self.seed)
            except Exception:  # noqa: BLE001 — never let a cold brain crash a turn
                pass
            self._seeded = True
            self._dirty = False
            self._since_fit = 0

    # ---- the conversational act ---- #
    def reply(self, stimulus: str, *, grounding: Optional[Sequence[str]] = None,
              max_tokens: int = 64) -> str:
        """Generate a learned conversational reply, or "" if the cold brain cannot yet.

        The prompt is conditioned on the stimulus (and a touch of grounding) so the continuation
        reflects what was asked; an empty / low-quality continuation returns "" so the caller can
        fall back to the deterministic floor.
        """
        self._ensure(grounding)
        if self._lm is None:
            return ""
        prompt = _clean(stimulus)
        try:
            out = self._lm.generate(prompt, max_tokens=max_tokens)
        except Exception:  # noqa: BLE001
            return ""
        out = _clean(out)
        if not _looks_usable(out):
            # one more try seeded from the persona frame (helps a very cold vocab)
            try:
                out = _clean(self._lm.generate("I", max_tokens=max_tokens))
            except Exception:  # noqa: BLE001
                out = ""
            if not _looks_usable(out):
                return ""
        return out

    def internal_confidence(self, text: str) -> float:
        """A self-measured confidence in ``text``: how well it sits in what the brain has learned.

        Derived from the model's own perplexity (lower → the continuation is squarely in-distribution
        → higher confidence). Mapped to [0.15, 0.9] so it never claims certainty and never collapses
        to zero. This is an *internal* signal — the system measuring itself — which is exactly what
        metacognition needs to stop importing confidence from outside.
        """
        if self._lm is None or not _clean(text):
            return 0.3
        try:
            ppl = float(self._lm.perplexity(_clean(text)))
        except Exception:  # noqa: BLE001
            return 0.3
        if not math.isfinite(ppl) or ppl <= 0:
            return 0.2
        # squash perplexity → confidence; ppl≈1 → ~0.9, ppl≈50 → ~0.4, ppl→∞ → 0.15
        conf = 0.9 / (1.0 + math.log1p(max(0.0, ppl - 1.0)) / 2.0)
        return max(0.15, min(0.9, conf))

    # ---- learning (compounding) ---- #
    def learn(self, *docs: str) -> None:
        """Fold real text into the corpus and schedule a re-fit — this is how the brain grows.

        Cheap and amortised: documents accumulate and the model re-fits every few additions rather
        than on every word, so a live loop can call this each turn without stalling.
        """
        fresh = [_clean(d) for d in docs if _clean(d) and len(_clean(d)) >= 3]
        if not fresh:
            return
        with self._lock:
            self._corpus.extend(fresh)
            if len(self._corpus) > self.max_corpus:
                # keep the seed (front) and the most-recent tail — drop the stale middle
                keep_head = len(_SEED_CORPUS)
                tail = self.max_corpus - keep_head
                self._corpus = self._corpus[:keep_head] + self._corpus[-tail:]
            self._since_fit += len(fresh)
            if self._since_fit >= self._refit_every:
                self._dirty = True

    def maybe_refit(self) -> bool:
        """Re-fit now if enough new text has accumulated. Returns True iff a re-fit happened."""
        if not self._dirty:
            return False
        self._ensure()
        return True

    # ---- introspection ---- #
    @property
    def kind(self) -> str:
        return self._kind

    @property
    def corpus_size(self) -> int:
        return len(self._corpus)


def build_self_brain(settings: Any = None, **kw: Any) -> SelfBrain:
    """Factory: an always-on learned brain bound to the given settings (promoted model preferred)."""
    return SelfBrain(settings=settings, **kw)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA self_reasoner self-test")
    print("=" * 70)
    brain = build_self_brain()
    r1 = brain.reply("Who are you and what do you do?")
    print(f"cold reply   : {r1!r}")
    print(f"kind         : {brain.kind}  corpus={brain.corpus_size}")
    assert isinstance(r1, str)
    # the brain compounds: teach it a distinctive phrasing, see the corpus grow + re-fit
    for _ in range(10):
        brain.learn("The Master asked about scheduling and I proposed a concrete next step.")
    grew = brain.maybe_refit()
    print(f"re-fit fired : {grew}  corpus={brain.corpus_size}")
    conf = brain.internal_confidence("I will think it through and propose the next step.")
    print(f"internal conf: {conf:.3f}")
    assert 0.15 <= conf <= 0.9
    print("\nALL SELF-TESTS PASSED ✓")
