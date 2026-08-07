"""NYXARA · nyx/holomem.py — content-addressed associative memory (🌐, NYX V.01 pillar 1).

The point of this module is what it *replaces*: a token context window. An LLM can only attend
to what fits in its window, so turn 500 pushes turn 1 out of existence. Here, recall is
**content-addressed** — a cue is encoded into the same 10,000-D hyperdimensional space as
everything stored, and whatever is *similar* comes back, no matter how long ago it was written.
A memory from an hour ago and one from a thousand turns ago are equally reachable; what decides
is similarity, not recency-in-a-buffer.

On top of the field this adds a **link layer**: memories that are recalled together, or written
in the same moment, become associated, so recall returns not just the nearest memory but the
small constellation around it. That is the "multidimensional links" part, and it is ordinary
bookkeeping — no magic.

Honest framing:

* **Not infinite.** ``capacity`` is real. Past it, the oldest binding is evicted from the
  cleanup memory (and surfaced on ``spilled`` so a caller can persist it to the ordinary
  ``memory/store.py``). She forgets — gracefully, and measurably.
* **Not lossless.** Holographic superposition means recall is a *cleanup* against stored
  vectors, with crosstalk that grows with load. ``HoloRecall.decided`` reports honestly when
  the recall is not confident; a low-confidence recall is a real "I'm not sure", not a guess
  dressed as an answer.
* **No token window** is a true claim; **"infinite memory"** is not, and is not made here.

Composes the audited :class:`~nyxara.memory.holographic_field.HolographicMemoryField` (which
in turn uses ``cognition/hyper_dimensional_vectors.py``). Pure standard library beyond that.
Fail-soft throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.memory.holographic_field import HolographicMemoryField, HoloRecall

__all__ = ["Trace", "Recall", "HoloMemory"]


@dataclass
class Trace:
    """One remembered thing, with the links it has grown to other memories.

    ``text`` is the part that is *encoded and searchable* — what she knows. ``cue`` is the
    context that produced it (the question that was asked), kept for provenance but deliberately
    **not** encoded: a stored question matches itself almost perfectly, and would otherwise echo
    back as the answer to its own re-asking.
    """

    key: str
    text: str
    kind: str = "episode"            # episode | fact | conclusion | percept | conjecture
    cue: str = ""
    written_tick: int = 0
    recalls: int = 0
    links: Dict[str, float] = field(default_factory=dict)   # other key → association weight

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "text": self.text, "kind": self.kind, "cue": self.cue,
                "written_tick": self.written_tick, "recalls": self.recalls,
                "links": {k: round(v, 6) for k, v in self.links.items()}}


@dataclass
class Recall:
    """What came back from a cue: the nearest trace plus the constellation linked to it."""

    hit: Optional[Trace] = None
    score: float = 0.0
    decided: bool = False            # False means "I am not confident" — say so, don't guess
    associated: List[Tuple[str, float]] = field(default_factory=list)
    considered: int = 0

    @property
    def text(self) -> str:
        return self.hit.text if self.hit is not None else ""

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.hit.key if self.hit else None, "text": self.text,
                "score": round(self.score, 6), "decided": self.decided,
                "associated": [[k, round(w, 6)] for k, w in self.associated],
                "considered": self.considered}


class HoloMemory:
    """Content-addressed associative memory with a link layer — NYX's long-term recall."""

    def __init__(self, *, dim: int = 10000, capacity: int = 4096, seed: int = 42,
                 recall_threshold: float = 0.15, link_rate: float = 0.2,
                 max_links_per_trace: int = 16) -> None:
        self.dim = int(dim)
        self.seed = int(seed)
        self.capacity = max(1, int(capacity))
        self.recall_threshold = float(recall_threshold)
        self.link_rate = float(link_rate)
        self.max_links_per_trace = max(0, int(max_links_per_trace))

        self.field = HolographicMemoryField(dim=self.dim, capacity=self.capacity,
                                            seed=self.seed,
                                            recall_threshold=self.recall_threshold)
        self.traces: Dict[str, Trace] = {}
        self.spilled: List[Tuple[str, str]] = []   # evicted (key, text) — persist these elsewhere
        self.ticks = 0
        self._recent: List[str] = []               # keys written/recalled in the current moment

    def __len__(self) -> int:
        return len(self.traces)

    def __contains__(self, key: str) -> bool:
        return str(key) in self.traces

    # ---- write ----------------------------------------------------------- #
    def remember(self, key: str, text: str, *, kind: str = "episode", cue: str = "",
                 link_to: Sequence[str] = ()) -> Optional[Trace]:
        """Store ``text`` under ``key`` and link it to whatever else is live in this moment.

        Only ``text`` is encoded into the holographic field. ``cue`` (the question that produced
        it) is kept as provenance only — see :class:`Trace`.
        """
        try:
            if key is None or text is None:
                return None                    # str(None) == "None" would store a literal lie
            key, text = str(key), str(text)
            if not key.strip() or not text.strip():
                return None
            self.ticks += 1
            self.field.remember(key, text)
            trace = self.traces.get(key)
            if trace is None:
                trace = Trace(key=key, text=text, kind=kind, cue=str(cue or ""),
                              written_tick=self.ticks)
                self.traces[key] = trace
            else:
                trace.text, trace.kind, trace.written_tick = text, kind, self.ticks
                trace.cue = str(cue or trace.cue)

            partners = list(link_to) + [k for k in self._recent if k != key]
            for other in partners:
                self._associate(key, str(other))
            self._note_recent(key)
            self._drain_spill()
            return trace
        except Exception:  # noqa: BLE001 — a write failure never breaks a turn
            return None

    def associate(self, a: str, b: str, *, weight: Optional[float] = None) -> None:
        """Explicitly link two memories (used when a cycle concludes that two things belong)."""
        try:
            self._associate(str(a), str(b), weight=weight)
        except Exception:  # noqa: BLE001
            pass

    def _associate(self, a: str, b: str, *, weight: Optional[float] = None) -> None:
        if a == b or a not in self.traces or b not in self.traces:
            return
        rate = self.link_rate if weight is None else float(weight)
        for src, dst in ((a, b), (b, a)):
            links = self.traces[src].links
            w = links.get(dst, 0.0)
            links[dst] = w + rate * (1.0 - w)          # saturating, like the graph's edges
            if len(links) > self.max_links_per_trace:
                weakest = min(links, key=links.get)
                links.pop(weakest, None)

    def _note_recent(self, key: str) -> None:
        """Keep a short 'what is live right now' list — this is what new writes link against."""
        if key in self._recent:
            self._recent.remove(key)
        self._recent.append(key)
        del self._recent[:-8]

    def _drain_spill(self) -> None:
        """The field evicts past capacity; mirror that here so traces cannot outlive the field."""
        for key, text in self.field.spilled:
            self.traces.pop(key, None)
            self.spilled.append((key, text))
        self.field.spilled = []
        del self.spilled[:-256]

    # ---- read ------------------------------------------------------------ #
    def recall(self, cue: str, *, k: int = 5) -> Recall:
        """Content-addressed recall: what does this cue bring back, and what hangs off it?

        No token window is consulted — the cue is matched against everything stored.
        """
        out = Recall(considered=len(self.traces))
        try:
            cue = str(cue or "").strip()
            if not cue or not self.traces:
                return out
            hit: HoloRecall = self.field.recall_by_content(cue)
            trace = self.traces.get(hit.key) if hit.key else None
            out.score = float(hit.score)
            out.decided = bool(hit.decided)
            if trace is None:
                return out
            out.hit = trace
            trace.recalls += 1
            self._note_recent(trace.key)
            links = sorted(trace.links.items(), key=lambda p: -p[1])
            out.associated = [(kk, w) for kk, w in links if kk in self.traces][: max(0, int(k))]
            return out
        except Exception:  # noqa: BLE001 — recall degrades to "nothing came back", never crashes
            return out

    def recall_key(self, key: str) -> Optional[Trace]:
        """Exact lookup by key — no similarity involved."""
        return self.traces.get(str(key))

    def candidates(self, cue: str, *, k: int = 5) -> List[Tuple[Trace, float]]:
        """The ``k`` closest traces to ``cue``, most similar first.

        :meth:`recall` returns only the single nearest memory, which is the right default but a
        poor way to *answer* something: a passing episode can sit closer to the wording of a
        question than the fact that actually answers it. Callers that need to weigh what kind of
        memory they are looking at (see ``nyx/modules.MemorySpecialist``) rank these instead.
        """
        try:
            cue = str(cue or "").strip()
            if not cue or not self.traces:
                return []
            field = self.field
            probe = field._encode_text(cue)              # noqa: SLF001 — same encoder as writes
            space, values = field.space, field.values
            # one reverse pass, not a scan per vector
            vid_to_key = {vid: key
                          for key, vid in field._key_to_vid.items()}  # noqa: SLF001
            scored: List[Tuple[Trace, float]] = []
            for vid, vec in values._items.items():       # noqa: SLF001 — bounded value vocabulary
                trace = self.traces.get(vid_to_key.get(vid, ""))
                if trace is None:
                    continue
                scored.append((trace, float(space.similarity(vec, probe))))
            scored.sort(key=lambda p: -p[1])
            return scored[: max(1, int(k))]
        except Exception:  # noqa: BLE001 — ranking degrades to "no candidates", never crashes
            return []

    def context(self, cue: str, *, k: int = 5) -> List[Trace]:
        """The working context for a turn: the recalled trace plus its constellation.

        This is what replaces "the last N tokens" — it is selected by relevance, not recency.
        """
        try:
            rec = self.recall(cue, k=k)
            out: List[Trace] = []
            if rec.hit is not None:
                out.append(rec.hit)
            for key, _w in rec.associated:
                trace = self.traces.get(key)
                if trace is not None and trace not in out:
                    out.append(trace)
            return out[: max(1, int(k))]
        except Exception:  # noqa: BLE001
            return []

    def stats(self) -> Dict[str, Any]:
        try:
            links = sum(len(t.links) for t in self.traces.values())
            return {"traces": len(self.traces), "capacity": self.capacity, "dim": self.dim,
                    "links": links, "spilled": len(self.spilled), "ticks": self.ticks,
                    "mean_links": round(links / len(self.traces), 3) if self.traces else 0.0}
        except Exception:  # noqa: BLE001
            return {}

    # ---- persistence ----------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        """Store the traces (the recipe). The field rebuilds itself by replaying them."""
        return {"dim": self.dim, "seed": self.seed, "capacity": self.capacity,
                "recall_threshold": self.recall_threshold, "link_rate": self.link_rate,
                "max_links_per_trace": self.max_links_per_trace, "ticks": self.ticks,
                "traces": [t.to_dict() for t in self.traces.values()]}

    def load_dict(self, data: Any) -> bool:
        """Rebuild from saved traces, oldest first. A corrupt sidecar is an empty memory."""
        try:
            if not isinstance(data, dict):
                return False
            if int(data.get("dim", self.dim)) != self.dim:
                return False                       # a different width would recall wrongly
            if int(data.get("seed", self.seed)) != self.seed:
                return False
            self.capacity = max(1, int(data.get("capacity", self.capacity)))
            self.recall_threshold = float(data.get("recall_threshold", self.recall_threshold))
            self.link_rate = float(data.get("link_rate", self.link_rate))
            self.max_links_per_trace = int(data.get("max_links_per_trace",
                                                    self.max_links_per_trace))
            self.field = HolographicMemoryField(dim=self.dim, capacity=self.capacity,
                                                seed=self.seed,
                                                recall_threshold=self.recall_threshold)
            self.traces, self.spilled, self._recent = {}, [], []
            raws = sorted((r for r in data.get("traces", []) or [] if isinstance(r, dict)),
                          key=lambda r: int(r.get("written_tick", 0)))
            for raw in raws:
                key, text = str(raw.get("key", "")), str(raw.get("text", ""))
                if not key or not text:
                    continue
                self.field.remember(key, text)
                self.traces[key] = Trace(
                    key=key, text=text, kind=str(raw.get("kind", "episode")),
                    cue=str(raw.get("cue", "")),
                    written_tick=int(raw.get("written_tick", 0)),
                    recalls=int(raw.get("recalls", 0)),
                    links={str(k): float(v) for k, v in (raw.get("links") or {}).items()})
            # links may point at traces that were evicted before the save — drop those
            for trace in self.traces.values():
                trace.links = {k: v for k, v in trace.links.items() if k in self.traces}
            self.ticks = int(data.get("ticks", len(self.traces)))
            self.field.spilled = []
            return True
        except Exception:  # noqa: BLE001 — a corrupt sidecar must never block boot
            return False
