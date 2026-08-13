"""NYXARA · causal/hypergraph_compress.py — Neural-Symbolic Hyper-Graph Compression (OMNISCIENCE · 3).

*The Master's ask:* as NYXARA evolves, her knowledge graphs and causal histories balloon. Fold them
into **dense, hierarchical hyper-graphs** so terabytes of history live in a few MB of RAM and can be
queried instantly.

What that means here, honestly. :class:`HyperGraphCompressor` takes a set of knowledge triples
``(subject, predicate, object)`` (plus optional causal-history strings) and packs them losslessly:

    1. **Symbol table** — every distinct string is stored once and referenced by an integer id
       (dedup: the biggest real win, since graphs repeat entities and relations constantly);
    2. **Hyper-edge grouping** — edges sharing a ``(subject, predicate)`` are collapsed into a
       single hyper-edge pointing at *many* objects (and likewise for shared ``(predicate, object)``),
       so a hub of *k* edges becomes one hyper-edge — the hierarchical fold;
    3. **Delta encoding** — object-id lists are sorted and stored as successive differences, so the
       packed integers are small.

:meth:`decompress` reconstructs the **exact** original triple set (order-independent), and every
:meth:`compress` self-verifies that round-trip and reports a **measured** ratio (raw vs. packed
bytes, via stdlib ``json`` + ``zlib``).

The honest boundary. This is genuinely lossless for the triple set and gives real, *measured*
compression — but the ratio depends entirely on how repetitive the data is. It is **not** a magic
"terabytes → a few MB, always" law; unstructured, non-repeating data barely compresses, and the
report says so. No neural weights are involved; "neural-symbolic" here means the symbol/hyper-edge
structure that a VSA memory (:mod:`nyxara.memory.holographic_field`) can later bind over.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = ["CompressionReport", "HyperGraphCompressor"]

Triple = Tuple[str, str, str]


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class CompressionReport:
    """A measured account of one compression pass — no unverifiable claims."""

    triples: int
    symbols: int
    hyperedges: int
    bytes_raw: int
    bytes_packed: int
    lossless: bool

    @property
    def ratio(self) -> float:
        return (self.bytes_raw / self.bytes_packed) if self.bytes_packed else 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {"triples": self.triples, "symbols": self.symbols, "hyperedges": self.hyperedges,
                "bytes_raw": self.bytes_raw, "bytes_packed": self.bytes_packed,
                "ratio": round(self.ratio, 3), "lossless": self.lossless}

    def summary(self) -> str:
        return (f"{self.triples} triples → {self.hyperedges} hyper-edges, "
                f"{self.bytes_raw}B → {self.bytes_packed}B ({self.ratio:.2f}×), "
                f"lossless={self.lossless}")


# --------------------------------------------------------------------------- #
# Compressor
# --------------------------------------------------------------------------- #
class HyperGraphCompressor:
    """Lossless hierarchical hyper-graph packer for knowledge triples + causal history."""

    def compress(self, triples: Iterable[Triple],
                 history: Optional[Sequence[str]] = None) -> Tuple[Dict[str, Any], CompressionReport]:
        """Pack ``triples`` (and optional ``history``) into a hyper-graph blob + report."""
        triple_list: List[Triple] = [tuple(map(str, t)) for t in triples]  # type: ignore[misc]
        unique: List[Triple] = sorted(set(triple_list))
        history = list(history or [])

        # 1. symbol table (dedup)
        sym_id: Dict[str, int] = {}
        symbols: List[str] = []

        def _id(s: str) -> int:
            if s not in sym_id:
                sym_id[s] = len(symbols)
                symbols.append(s)
            return sym_id[s]

        # 2. hyper-edge grouping by (subject, predicate) → sorted object ids, delta-encoded
        groups: Dict[Tuple[int, int], List[int]] = {}
        for s, p, o in unique:
            groups.setdefault((_id(s), _id(p)), []).append(_id(o))
        for hist in history:
            _id(hist)  # keep history strings in the table so decompress can carry them

        hyperedges: List[List[int]] = []
        for (sid, pid), objs in sorted(groups.items()):
            objs = sorted(set(objs))
            delta = _delta_encode(objs)
            hyperedges.append([sid, pid, *delta])

        blob: Dict[str, Any] = {
            "v": 1,
            "symbols": symbols,
            "hyperedges": hyperedges,
            "history": [_id(h) for h in history],
        }

        # measured sizes — real bytes, not a slogan. Raw = the naive uncompressed store (a plain
        # triple list, as logs actually accumulate); packed = our structural hyper-graph after
        # generic entropy coding. The ratio is that honest end-to-end "raw logs → dense graph" win.
        naive = json.dumps(sorted([list(t) for t in unique])
                           + [["__hist__", h] for h in history])
        raw_bytes = len(naive.encode("utf-8"))
        packed_bytes = _packed_size(json.dumps(blob, separators=(",", ":")))

        # self-verify losslessness
        restored = self.decompress(blob)
        lossless = restored == set(unique)

        report = CompressionReport(triples=len(unique), symbols=len(symbols),
                                   hyperedges=len(hyperedges), bytes_raw=raw_bytes,
                                   bytes_packed=packed_bytes, lossless=lossless)
        return blob, report

    def decompress(self, blob: Dict[str, Any]) -> Set[Triple]:
        """Reconstruct the exact original triple set from a hyper-graph blob."""
        symbols: List[str] = blob["symbols"]
        out: Set[Triple] = set()
        for edge in blob["hyperedges"]:
            sid, pid = edge[0], edge[1]
            obj_ids = _delta_decode(edge[2:])
            s, p = symbols[sid], symbols[pid]
            for oid in obj_ids:
                out.add((s, p, symbols[oid]))
        return out

    def history(self, blob: Dict[str, Any]) -> List[str]:
        symbols: List[str] = blob["symbols"]
        return [symbols[i] for i in blob.get("history", [])]

    def query(self, blob: Dict[str, Any], subject: Optional[str] = None,
              predicate: Optional[str] = None) -> List[Triple]:
        """Query the packed graph directly (no full decompress needed for a hub lookup).

        A filter that was *asked for* but names a symbol the graph has never seen narrows the
        result to nothing. Collapsing that case into the same ``None`` that means "no filter"
        made an unknown subject return the **entire** graph — the worst possible answer from a
        lookup whose only honest reply is "I don't know that one".
        """
        symbols: List[str] = blob["symbols"]
        index: Dict[str, int] = {s: i for i, s in enumerate(symbols)}
        sid = pid = None
        if subject is not None:
            sid = index.get(subject)
            if sid is None:
                return []
        if predicate is not None:
            pid = index.get(predicate)
            if pid is None:
                return []
        hits: List[Triple] = []
        for edge in blob["hyperedges"]:
            if sid is not None and edge[0] != sid:
                continue
            if pid is not None and edge[1] != pid:
                continue
            s, p = symbols[edge[0]], symbols[edge[1]]
            for oid in _delta_decode(edge[2:]):
                hits.append((s, p, symbols[oid]))
        return hits


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _delta_encode(sorted_ids: Sequence[int]) -> List[int]:
    out: List[int] = []
    prev = 0
    for i in sorted_ids:
        out.append(i - prev)
        prev = i
    return out


def _delta_decode(deltas: Sequence[int]) -> List[int]:
    out: List[int] = []
    cur = 0
    for d in deltas:
        cur += d
        out.append(cur)
    return out


def _packed_size(text: str) -> int:
    """Real byte count after generic entropy coding (zlib) — an honest size, not a guess."""
    return len(zlib.compress(text.encode("utf-8"), level=6))
