"""NYXARA · nyx001/lingua/tokenizer.py — a segmentation learned from experience (🔤, no corpus).

Track B's first requirement: a **new tokenizer**, not a borrowed one. The charter's prohibition
list rules out inheriting anything pretrained, and a BPE vocabulary trained on a human corpus is
exactly that — a compressed artefact of somebody else's text, carrying their word boundaries,
their language mix, and their idea of what a subword is.

So this learns its segmentation from *what the system actually encounters*, by the same principle
BPE uses but with no external corpus and no offline training phase: adjacent-symbol pairs that keep
recurring get merged into one symbol. The merge table starts **empty** and grows online.

    1. start from bytes — the only alphabet available without inheriting a vocabulary
    2. count adjacent pairs in the stream as it arrives
    3. when a pair's count clears ``merge_at``, promote it to a symbol
    4. bounded: at ``max_vocab`` the least-used learned symbols are evicted

Two consequences are honest rather than convenient. At birth this tokenizer is a **byte
tokenizer** — it segments nothing, because it has seen nothing. And its vocabulary is a fingerprint
of one system's history, so two NYX V.001 instances fed different experience will not share a
vocabulary. That is the correct behaviour for a system forbidden from inheriting one, and it is
also why :meth:`Tokenizer.to_dict` carries the merge table: the vocabulary is *learned state*, like
any weight.

Honest: this is online BPE over bytes. The mechanism is not novel — what is different is the
absence of a pretraining corpus and the online, bounded, evictable merge table. It will be worse
than a corpus-trained tokenizer at compressing English, and better at compressing whatever this
particular system actually sees.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

__all__ = ["Tokenizer"]


class Tokenizer:
    """Byte-level segmentation that learns its own merges from the stream it is shown."""

    def __init__(self, *, max_vocab: int = 8192, merge_at: int = 8,
                 max_pairs: int = 65536, decay: float = 0.999) -> None:
        self.max_vocab = int(max_vocab)
        self.merge_at = int(merge_at)
        self.max_pairs = int(max_pairs)
        self.decay = float(decay)
        # ids 0-255 are the raw bytes; 256+ are learned merges. Nothing is seeded beyond that.
        self._merges: Dict[Tuple[int, int], int] = {}
        self._inverse: Dict[int, Tuple[int, int]] = {}
        self._counts: Dict[Tuple[int, int], float] = {}
        self._uses: Dict[int, float] = {}
        self._next_id = 256
        # Ids freed by eviction, held for exact reuse. `_next_id` only ever moves FORWARD over
        # ids that were never issued; a freed id comes back through here. Rewinding `_next_id`
        # to the victim instead made the counter walk back over ids that were still live, so two
        # different pairs ended up mapped to one id and `decode` stopped being the inverse of
        # `encode` — measured at 100% round-trip failure once eviction started.
        self._free: List[int] = []
        self.observed = 0
        self.learned = 0
        self.evicted = 0

    # ---- encode ---- #
    def encode(self, text: str, *, learn: bool = True) -> List[int]:
        """Text → ids, applying every merge it has learned. Optionally learns from this text."""
        try:
            ids = list(str(text).encode("utf-8", "ignore"))
            if not ids:
                return ids
            ids = self._apply_merges(ids)
            if learn:
                self._observe(ids)
            for i in ids:
                if i >= 256:
                    self._uses[i] = self._uses.get(i, 0.0) + 1.0
            return ids
        except Exception:  # noqa: BLE001 — a tokenizer that fails returns bytes, never nothing
            return list(str(text).encode("utf-8", "ignore"))

    def _apply_merges(self, ids: List[int]) -> List[int]:
        """Repeatedly apply the highest-priority applicable merge. Bounded by vocabulary size."""
        if not self._merges:
            return ids
        for _ in range(len(ids)):
            best: Optional[Tuple[int, int]] = None
            best_id = 0
            for pair in zip(ids, ids[1:]):
                mid = self._merges.get(pair)
                # lowest id = learned earliest = most frequent when learned; apply that first
                if mid is not None and (best is None or mid < best_id):
                    best, best_id = pair, mid
            if best is None:
                break
            out: List[int] = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best:
                    out.append(best_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        return ids

    # ---- learn ---- #
    def _observe(self, ids: List[int]) -> None:
        """Count adjacent pairs; promote any that clears the threshold."""
        self.observed += 1
        for pair in zip(ids, ids[1:]):
            self._counts[pair] = self._counts.get(pair, 0.0) + 1.0
            if self._counts[pair] >= self.merge_at and pair not in self._merges:
                self._promote(pair)
        if len(self._counts) > self.max_pairs:
            keep = sorted(self._counts.items(), key=lambda kv: kv[1], reverse=True)
            self._counts = dict(keep[: self.max_pairs // 2])

    def _promote(self, pair: Tuple[int, int]) -> None:
        if not self._free and self._next_id >= self.max_vocab:
            if not self._evict():
                return
        if self._free:
            mid = self._free.pop()
        else:
            mid = self._next_id
            self._next_id += 1
        self._merges[pair] = mid
        self._inverse[mid] = pair
        self._uses[mid] = float(self.merge_at)
        self.learned += 1

    def _evict(self) -> bool:
        """Drop the least-used learned symbol. Bytes are never evictable.

        A symbol that another live merge is *built out of* is never chosen: expanding a merge
        whose component has been dropped cannot recover the original bytes, so evicting one
        would make every symbol above it decode to rubbish. Only **roots** of the merge forest
        are evictable — symbols nothing else expands into — which is what keeps :meth:`decode`
        the exact inverse of :meth:`encode`.

        This cannot wedge: a non-empty forest always has at least one root, so as long as any
        learned symbol exists there is always something evictable.
        """
        learned = [i for i in self._inverse if i >= 256]
        if not learned:
            return False
        component_of = {c for p in self._inverse.values() for c in p if c >= 256}
        roots = [i for i in learned if i not in component_of]
        if not roots:
            return False
        victim = min(roots, key=lambda i: self._uses.get(i, 0.0))
        pair = self._inverse.pop(victim, None)
        if pair is not None:
            self._merges.pop(pair, None)
        self._uses.pop(victim, None)
        self.evicted += 1
        self._free.append(victim)       # freed for exact reuse; never rewind _next_id
        return True

    # ---- decode ---- #
    def decode(self, ids: List[int]) -> str:
        """ids → text, expanding learned symbols back to their bytes. Lossless by construction."""
        try:
            out: List[int] = []
            for i in ids:
                out.extend(self._expand(int(i)))
            return bytes(b & 0xFF for b in out).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return ""

    def _expand(self, i: int, depth: int = 0) -> List[int]:
        if i < 256 or depth > 64:
            return [i]
        pair = self._inverse.get(i)
        if pair is None:
            return [0]
        return self._expand(pair[0], depth + 1) + self._expand(pair[1], depth + 1)

    # ---- read ---- #
    @property
    def vocab_size(self) -> int:
        return 256 + len(self._inverse)

    def compression(self, text: str) -> Optional[float]:
        """Bytes per token on ``text``. 1.0 means it has learned nothing yet — honestly 1.0."""
        try:
            raw = len(str(text).encode("utf-8", "ignore"))
            if raw == 0:
                return None
            return float(raw / max(1, len(self.encode(text, learn=False))))
        except Exception:  # noqa: BLE001
            return None

    def stats(self) -> Dict[str, Any]:
        return {"vocab_size": self.vocab_size, "learned": self.learned,
                "evicted": self.evicted, "observed": self.observed,
                "pairs_tracked": len(self._counts), "max_vocab": self.max_vocab}

    def to_dict(self) -> Dict[str, Any]:
        return {"observed": self.observed, "learned": self.learned, "evicted": self.evicted,
                "next_id": self._next_id,
                "merges": [[a, b, mid] for (a, b), mid in self._merges.items()]}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.observed = int(d.get("observed", 0))
            self.learned = int(d.get("learned", 0))
            self.evicted = int(d.get("evicted", 0))
            self._next_id = int(d.get("next_id", 256))
            for a, b, mid in d.get("merges") or []:
                pair = (int(a), int(b))
                self._merges[pair] = int(mid)
                self._inverse[int(mid)] = pair
            # Rebuild the free list from the gap between what was issued and what survived, so a
            # restored tokenizer allocates exactly like the one that was saved.
            self._free = [i for i in range(256, self._next_id) if i not in self._inverse]
        except Exception:  # noqa: BLE001
            pass
