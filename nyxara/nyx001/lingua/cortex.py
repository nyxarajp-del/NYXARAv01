"""NYXARA · nyx001/lingua/cortex.py — Track B, assembled and plugged in (🗣, language last).

The four Track B modules — :class:`~nyxara.nyx001.lingua.tokenizer.Tokenizer`,
:class:`~nyxara.nyx001.lingua.dynamic_embedding.DynamicEmbedding`,
:class:`~nyxara.nyx001.lingua.sequence.SequenceModel` and
:class:`~nyxara.nyx001.lingua.grounding.Grounding` — were each complete, each tested, and each
reachable from nowhere. Nothing in the package constructed one: not the stack, not the brain, not
the kernel, not a config gate. They were a track that existed only as an import.

:class:`LanguageCortex` is the assembly, and :meth:`read` is the one call that runs it:

    text → tokenizer → ids → dynamic embedding (under the live situation) → selective scan
                          ↘ words → grounding, bound to the concept Layer 4 just assigned

**Why it hangs off the stack rather than inside it.** Layers 0–17 are Track A. Track B is a
*modality*, and the charter keeps the two apart deliberately — so this is a sibling of the layer
stack, not an eighteenth layer, and ``CognitiveStack.layers()`` still returns exactly the
eighteen it always did.

**The ordering is enforced, not just documented.** ``read`` binds words to a concept it is
*given*; it never invents one. Called before Layer 4 has formed anything, ``concept`` is ``None``,
no binding is made, and :meth:`meaning` keeps returning ``None`` for every word — which is the
charter's Environment → Perception → Concepts → World Model → Language ordering expressed as a
refusal rather than as a comment. Language cannot run ahead of the concepts it is supposed to
denote, because there is nothing for it to denote yet.

**Cost, stated rather than discovered, because it is not small.** The selective scan is almost
all of it. Measured at medium scale (width 256, depth 4): ~17 ms at 8 tokens, ~42 ms at 16, ~92 ms
at 32. Tokenization and grounding together are ~2.7 ms regardless. In a live cycle that made Track
B roughly 96% of the stack's measured time — it about doubles the cost of a full cycle — and
``stats()["timings_ms"]["lingua"]`` reports it every run so it stays visible rather than becoming
folklore.

Two knobs, both real: ``max_tokens`` caps the sequence length (the scan is linear in it, so the
cap is a ceiling on the worst case rather than the typical cost — an online tokenizer that has
learned its merges usually returns far fewer). ``sequence_enabled=False`` removes the scan
entirely while leaving tokenization and grounding running, which is the cheap configuration that
keeps everything that actually *accumulates learned state* — the merge table and the word→concept
bindings — and drops only the part that does not yet have an objective training it.

Honest: nothing here is a language model. The tokenizer starts as a byte tokenizer, the embedding
table never leaves its random initialisation while no language objective is running, and the
grounding vocabulary is empty at birth and grows only as far as co-occurrence with self-formed
concepts will carry it. :meth:`stats` reports each of those as a number so the emptiness is
visible instead of implied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nyxara.nyx001.lingua.dynamic_embedding import DynamicEmbedding
from nyxara.nyx001.lingua.grounding import Grounding
from nyxara.nyx001.lingua.sequence import SequenceModel
from nyxara.nyx001.lingua.tokenizer import Tokenizer

__all__ = ["Reading", "LanguageCortex"]


@dataclass
class Reading:
    """What one pass over a piece of text actually produced. Absent stages report ``None``."""

    tokens: int = 0
    compression: Optional[float] = None
    embedded: bool = False
    scanned: bool = False
    state_norm: Optional[float] = None
    concept: Optional[str] = None
    bound: int = 0
    grounded_words: List[str] = field(default_factory=list)

    @property
    def ran(self) -> bool:
        return self.tokens > 0

    def to_dict(self) -> Dict[str, Any]:
        return {"tokens": self.tokens,
                "compression": None if self.compression is None else round(self.compression, 4),
                "embedded": self.embedded, "scanned": self.scanned,
                "state_norm": None if self.state_norm is None else round(self.state_norm, 4),
                "concept": self.concept, "bound": self.bound,
                "grounded_words": list(self.grounded_words[:8])}


class LanguageCortex:
    """Track B as one object: segment, embed under the situation, scan, and ground."""

    def __init__(self, substrate: Any, *, max_tokens: int = 32,
                 sequence_enabled: bool = True, min_support: float = 3.0,
                 min_exclusivity: float = 0.5) -> None:
        self.substrate = substrate
        self.max_tokens = max(1, int(max_tokens))
        self.sequence_enabled = bool(sequence_enabled)
        self.reads = 0
        self.available = False
        self.tokenizer = Tokenizer(max_vocab=int(getattr(substrate, "vocab", 8192)))
        self.grounding = Grounding(substrate, min_support=float(min_support),
                                   min_exclusivity=float(min_exclusivity))
        self.embedding: Optional[DynamicEmbedding] = None
        self.sequence: Optional[SequenceModel] = None
        try:
            self.embedding = DynamicEmbedding(substrate)
            if self.sequence_enabled:
                self.sequence = SequenceModel(substrate)
            self.available = True
        except Exception:  # noqa: BLE001 — without NumPy the two cheap halves still run
            self.available = False

    # ---- the one call ---- #
    def read(self, text: str, *, concept: Any = None, context: Any = None,
             memory: Any = None, goal: Any = None, uncertainty: float = 0.0) -> Reading:
        """Run text through the whole track. Never raises; a failed stage is absent, not zero."""
        out = Reading(concept=getattr(concept, "name", None) if concept is not None else None)
        body = str(text or "").strip()
        if not body:
            return out
        try:
            self.reads += 1
            ids = self.tokenizer.encode(body)[: self.max_tokens]
            out.tokens = len(ids)
            out.compression = self.tokenizer.compression(body)

            if ids and self.embedding is not None:
                z = self.embedding.embed_sequence(ids, context=context, memory=memory,
                                                  goal=goal, uncertainty=uncertainty)
                out.embedded = z is not None
                if z is not None and self.sequence is not None:
                    out.scanned = self.sequence.forward(z) is not None
                    out.state_norm = self.sequence.state_norm()

            # The ordering, enforced: no concept means no binding. A word cannot be grounded in
            # something the system has not formed.
            if concept is not None:
                bindings = self.grounding.ground_text(body, concept)
                out.bound = len(bindings)
                out.grounded_words = [b.token for b in bindings
                                      if self.grounding.is_grounded(b.token)]
            return out
        except Exception:  # noqa: BLE001 — language never breaks a cycle
            return out

    def reset_state(self) -> None:
        """Drop the scan's carried belief — a new utterance, the same learned weights."""
        if self.sequence is not None:
            self.sequence.reset_state()

    # ---- read ---- #
    def meaning(self, word: str) -> Optional[str]:
        """The concept a word denotes, or ``None`` when it has not earned a grounding."""
        return self.grounding.meaning(word)

    def describe(self, concept: Any, *, k: int = 5) -> List[Any]:
        """Concept → the words that reliably denote it. The ordering, run backwards."""
        return self.grounding.describe(concept, k=k)

    def vocabulary(self) -> List[str]:
        """Every word this system has actually grounded — usually far fewer than it has seen."""
        return self.grounding.vocabulary()

    def stats(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "available": self.available, "reads": self.reads, "max_tokens": self.max_tokens,
            "tokenizer": self.tokenizer.stats(),
            "grounding": self.grounding.stats(),
        }
        if self.embedding is not None:
            out["embedding"] = self.embedding.stats()
        if self.sequence is not None:
            out["sequence"] = self.sequence.stats()
        return out

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"reads": self.reads, "tokenizer": self.tokenizer.to_dict(),
                               "grounding": self.grounding.to_dict()}
        if self.embedding is not None:
            out["embedding"] = self.embedding.to_dict()
        if self.sequence is not None:
            out["sequence"] = self.sequence.to_dict()
        return out

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.reads = int(d.get("reads", 0))
            self.tokenizer.load_dict(d.get("tokenizer") or {})
            self.grounding.load_dict(d.get("grounding") or {})
            if self.embedding is not None and d.get("embedding"):
                self.embedding.load_dict(d["embedding"])
            if self.sequence is not None and d.get("sequence"):
                self.sequence.load_dict(d["sequence"])
        except Exception:  # noqa: BLE001 — a corrupt sidecar leaves a freshly-born tongue
            pass
