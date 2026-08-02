"""NYXARA · memory/holographic_field.py — Holographic Memory Field (🧠🌌, Cognition/Memory).

The critique this module answers, in the Master's words: *"AI vector databases (RAG) ka use karti hai
jo text ko chunks mein tod kar store karte hain. Naya: ek continuous holographic memory field jahan har
memory poore system ke saath entangled ho — forgetting aur search-latency kam."*

Her :mod:`nyxara.cognition.hyper_dimensional_vectors` already provides the real substrate — a
Vector-Symbolic Architecture where ``bundle`` is a **holographic superposition** ("the whole is
distributed across every bit"), ``bind`` ties a role to a filler, and an :class:`ItemMemory` cleans a
noisy vector back to the nearest clean symbol (content-addressable recall). It was a *cognition*
faculty; this module makes it a **memory layer**.

:class:`HolographicMemoryField` stores every memory as a **key ⊗ value binding bundled into one
continuous field hypervector**. Recall is a single **unbind + cleanup** — an associative,
near-constant-work operation over the whole field, *not* a per-chunk linear scan of stored text. Because
every memory shares the one field vector, memories are genuinely **entangled**: recalling one is a
projection of the whole. This is the honest reading of "zero search latency": recall does O(dim) work
plus a cleanup against a bounded symbol set, independent of how many memories were folded in.

Honest about physics (no fake claims): a bundle **saturates** — crosstalk grows with the number of
folded bindings, so the field holds a bounded *working horizon* at a given width, with cleanup-bounded
fidelity. Beyond capacity, older bindings fade (graceful forgetting) and the layer **spills** to the
ordinary :class:`~nyxara.memory.store.MemoryStore` for the long tail. So: instant associative blend for
the live horizon, honest degradation past it — not literal infinite, latency-free, loss-free memory,
which is physically impossible.

Pure composition of the HDC faculty; **no LLM**; degrades to a small pure-python space when NumPy is
absent (correct, just slower).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from nyxara.cognition.hyper_dimensional_vectors import HyperSpace, ItemMemory

__all__ = ["HoloRecall", "HolographicMemoryField"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str, *, cap: int = 64) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())[:cap]


@dataclass
class HoloRecall:
    """The result of an associative recall from the holographic field."""

    key: Optional[str]
    text: Optional[str]
    score: float                     # cosine of the cleaned recall to the nearest stored value
    decided: bool                    # True when the recall is confident (score ≥ threshold)
    runner_up: Optional[str] = None
    margin: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "text": self.text, "score": round(self.score, 6),
                "decided": self.decided, "runner_up": self.runner_up,
                "margin": round(self.margin, 6)}


class HolographicMemoryField:
    """A continuous, entangled, associative memory over one HDC field vector (real HRR key→value store)."""

    def __init__(self, *, dim: int = 10000, capacity: int = 512, seed: int = 42,
                 recall_threshold: float = 0.15) -> None:
        self.space = HyperSpace(dim, seed=seed)
        self.keys = ItemMemory(self.space)           # atomic key symbols
        self.values = ItemMemory(self.space)         # stored value vectors (the cleanup memory)
        self.tokens = ItemMemory(self.space)         # atomic token symbols for encoding text
        self.field = self.space.zero()               # THE continuous holographic field
        self.capacity = max(1, int(capacity))
        self.recall_threshold = float(recall_threshold)
        self._texts: Dict[str, str] = {}             # value-id → original text
        self._key_to_vid: Dict[str, str] = {}        # key → its current value-id
        self._order: List[str] = []                  # value-ids oldest→newest (for graceful fade)
        self._count = 0
        self.spilled: List[Tuple[str, str]] = []     # (key, text) evicted past capacity (spill to store)

    # ---- encoding -------------------------------------------------------- #
    def _encode_text(self, text: str) -> Any:
        """Encode text into a bipolar value hypervector: a bundle of its token symbols, signed.

        This is a *bag of tokens* on purpose — so a content cue that shares vocabulary with a stored
        memory produces a *similar* vector (content-addressable recall). The sign step quantises the
        superposition back to a clean ±1 vector suitable for binding and cleanup. (Exact key→value
        recall works regardless of the encoding, since unbind recovers the stored vector itself.)"""
        toks = _tokens(text)
        if not toks:
            return self.space.random()
        vecs = [self.tokens.symbol(t) for t in toks]
        return self.space.sign(self.space.bundle(*vecs))

    # ---- write ----------------------------------------------------------- #
    def remember(self, key: str, text: str) -> None:
        """Fold ``key → text`` into the field as a bound, bundled hypervector.

        Rewriting an existing key superposes the new binding (the freshest recall wins on cleanup).
        Past capacity the oldest binding is dropped from the cleanup memory (graceful forgetting) and
        surfaced on :attr:`spilled` so a caller can persist it to the ordinary store."""
        key = str(key)
        key_hv = self.keys.symbol(key)
        val_hv = self._encode_text(text)
        # rewriting a key: drop the previous binding from the cleanup memory so the freshest value
        # wins on recall (the stale binding remains only as bounded crosstalk in the field sum).
        prev_vid = self._key_to_vid.get(key)
        if prev_vid is not None:
            self.values._items.pop(prev_vid, None)      # noqa: SLF001 — bounded internal vocabulary
            self.values._matrix = None                  # noqa: SLF001 — invalidate the cleanup cache
            self._texts.pop(prev_vid, None)
            if prev_vid in self._order:
                self._order.remove(prev_vid)
        vid = f"v{self._count}"
        self.values.add(vid, val_hv)
        self._texts[vid] = text
        self._key_to_vid[key] = vid
        self._order.append(vid)
        self.field = self.space.bundle(self.field, self.space.bind(key_hv, val_hv))
        self._count += 1
        self._enforce_capacity()

    def _enforce_capacity(self) -> None:
        while len(self._order) > self.capacity:
            old_vid = self._order.pop(0)
            text = self._texts.pop(old_vid, None)
            # drop it from the cleanup memory so recall no longer resolves to a faded binding
            self.values._items.pop(old_vid, None)       # noqa: SLF001 — bounded internal vocabulary
            self.values._matrix = None                  # noqa: SLF001 — invalidate the cleanup cache
            key = next((k for k, v in self._key_to_vid.items() if v == old_vid), None)
            if key is not None and text is not None:
                self.spilled.append((key, text))

    # ---- read (associative, near-constant work over the field) ----------- #
    def recall(self, key: str) -> HoloRecall:
        """Recall the value bound to ``key``: unbind the key from the field, then clean the noisy
        result to the nearest stored value. One O(dim) unbind + a cleanup over the bounded value set —
        no scan of the stored texts."""
        key = str(key)
        if key not in self.keys:
            return HoloRecall(key=key, text=None, score=0.0, decided=False)
        est = self.space.unbind(self.field, self.keys.symbol(key))
        res = self.values.cleanup(est)
        text = self._texts.get(res.name) if res.name else None
        return HoloRecall(key=key, text=text, score=res.score,
                          decided=res.score >= self.recall_threshold,
                          runner_up=res.runner_up, margin=res.margin)

    def recall_by_content(self, cue: str) -> HoloRecall:
        """Content-addressable recall: encode ``cue`` and clean it to the nearest stored value — the
        memory most similar to the cue, blended out of the whole field."""
        cue_hv = self._encode_text(cue)
        res = self.values.cleanup(cue_hv)
        if res.name is None:
            return HoloRecall(key=None, text=None, score=0.0, decided=False)
        text = self._texts.get(res.name)
        key = next((k for k, v in self._key_to_vid.items() if v == res.name), None)
        return HoloRecall(key=key, text=text, score=res.score,
                          decided=res.score >= self.recall_threshold,
                          runner_up=res.runner_up, margin=res.margin)

    # ---- introspection --------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._order)

    def report(self) -> Dict[str, Any]:
        return {"dim": self.space.dim, "held": len(self._order), "capacity": self.capacity,
                "written": self._count, "spilled": len(self.spilled),
                "saturation": round(len(self._order) / self.capacity, 4)}


# --------------------------------------------------------------------------- #
# Self-test / demo — real holographic key→value recall (no LLM, no network)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA holographic-memory-field self-test — one field, associative recall")
    print("=" * 70)

    field = HolographicMemoryField(dim=10000, capacity=64)

    facts = {
        "master": "the master is jp whom nyxara serves with loyalty",
        "rule4": "capability may grow but character never changes she stays corrigible",
        "physics": "force equals mass times acceleration in classical mechanics",
        "colour": "the sky appears blue because of rayleigh scattering of sunlight",
        "memory": "holographic memory distributes every fact across the whole field vector",
    }
    for k, v in facts.items():
        field.remember(k, v)

    # 1) recall each fact by its key — associative, from the ONE entangled field ----------------
    hits = 0
    for k, v in facts.items():
        r = field.recall(k)
        ok = r.text == v
        hits += int(ok)
        print(f"  recall[{k:8}] score={r.score:.3f} decided={r.decided} {'✓' if ok else '✗ '+str(r.text)[:30]}")
    assert hits == len(facts), f"only {hits}/{len(facts)} keys recalled cleanly"
    print(f"key recall          : {hits}/{len(facts)} exact from one holographic field ✓")

    # 2) content-addressable recall: a cue blends out the most-similar memory ------------------
    r = field.recall_by_content("why is the sky blue")
    print(f"\ncontent recall      : cue 'why is the sky blue' → key={r.key} score={r.score:.3f}")
    assert r.key == "colour", r.to_dict()
    print("content recall      : cue landed on the matching memory ✓")

    # 3) many facts still recall well below capacity, and saturation is honest -----------------
    for i in range(40):
        field.remember(f"fact{i}", f"synthetic memory number {i} about topic {i % 7}")
    good = sum(1 for k, v in facts.items() if field.recall(k).text == v)
    print(f"\nafter 45 memories   : {good}/{len(facts)} original facts still recalled; "
          f"report={field.report()}")
    assert good >= 4, "the field should still hold most of the working horizon"
    print("capacity            : graceful — most memories survive under the horizon ✓")

    # 4) past capacity the oldest binding fades and spills (graceful forgetting) ----------------
    small = HolographicMemoryField(dim=4096, capacity=4)
    for i in range(8):
        small.remember(f"k{i}", f"memory {i}")
    assert len(small) == 4 and len(small.spilled) == 4, small.report()
    print("forgetting          : oldest bindings faded + spilled past capacity ✓")

    print("\nALL SELF-TESTS PASSED ✓")
