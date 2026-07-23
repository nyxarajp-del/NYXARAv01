"""NYXARA · memory/holo_shard.py — holographic distributed state reconstruction (🌐, O).

Split NYXARA's memory into ``n`` redundant shards such that **any ``k`` surviving shards rebuild the
whole** — so losing or corrupting nodes below the redundancy budget does not lose her mind. This is real
**k-of-n erasure coding** (Reed–Solomon over GF(2⁸) with a Vandermonde generator matrix), the same family
of codes RAID/object stores use — not "1% rebuilds 100%" magic and not quantum entanglement. Reconstruction
needs at least ``k`` shards; below ``k`` it fails honestly (returns ``None``) rather than fabricating data.

Pairs with the gossip transport (:mod:`nyxara.growth.patch_gossip`) to distribute shards, and with
:mod:`nyxara.growth.self_repair` to verify a rebuilt bundle. Pure standard library; no LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

__all__ = ["Shard", "HoloSharder"]

# --- GF(2^8) with the standard AES/RS primitive polynomial 0x11d --- #
_EXP = [0] * 512
_LOG = [0] * 256
_x = 1
for _i in range(255):
    _EXP[_i] = _x
    _LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]


def _mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _EXP[_LOG[a] + _LOG[b]]


def _div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("GF division by zero")
    if a == 0:
        return 0
    return _EXP[(_LOG[a] - _LOG[b]) % 255]


@dataclass
class Shard:
    index: int          # evaluation point id (1..n), stable across the shard's life
    data: bytes

    def to_dict(self) -> Dict[str, object]:
        return {"index": self.index, "len": len(self.data)}


class HoloSharder:
    """k-of-n Reed–Solomon erasure coding over the byte stream of NYXARA's memory."""

    def __init__(self, *, k: int, n: int) -> None:
        if not (1 <= k <= n <= 255):
            raise ValueError("require 1 <= k <= n <= 255")
        self.k = k
        self.n = n
        # distinct non-zero evaluation points 1..n; row i is the Vandermonde row [alpha_i^0..^(k-1)]
        self._alpha = list(range(1, n + 1))

    def _row(self, alpha: int) -> List[int]:
        row = [1] * self.k
        for j in range(1, self.k):
            row[j] = _mul(row[j - 1], alpha)
        return row

    # ---- encode ---- #
    def encode(self, data: bytes) -> List[Shard]:
        """Encode ``data`` into ``n`` shards. Prepends a 4-byte length so padding is exactly undone."""
        payload = len(data).to_bytes(4, "big") + data
        pad = (-len(payload)) % self.k
        payload += b"\x00" * pad
        stripes = [payload[i:i + self.k] for i in range(0, len(payload), self.k)]
        shards: List[List[int]] = [[] for _ in range(self.n)]
        for stripe in stripes:
            for si, alpha in enumerate(self._alpha):
                row = self._row(alpha)
                acc = 0
                for j in range(self.k):
                    acc ^= _mul(row[j], stripe[j])
                shards[si].append(acc)
        return [Shard(index=self._alpha[i], data=bytes(shards[i])) for i in range(self.n)]

    # ---- decode ---- #
    def decode(self, shards: List[Shard]) -> Optional[bytes]:
        """Rebuild the original bytes from any ``k`` (or more) shards. Fewer than ``k`` ⇒ None."""
        uniq: Dict[int, bytes] = {}
        for s in shards:
            if 1 <= s.index <= self.n:
                uniq.setdefault(s.index, s.data)
        if len(uniq) < self.k:
            return None
        chosen = sorted(uniq.items())[: self.k]
        rows = [self._row(idx) for idx, _ in chosen]
        inv = _invert(rows)
        if inv is None:
            return None
        n_stripes = len(chosen[0][1])
        out = bytearray()
        for st in range(n_stripes):
            code = [data[st] for _, data in chosen]
            for i in range(self.k):
                acc = 0
                for j in range(self.k):
                    acc ^= _mul(inv[i][j], code[j])
                out.append(acc)
        length = int.from_bytes(bytes(out[:4]), "big")
        body = bytes(out[4:])
        if length > len(body):
            return None
        return body[:length]


def _invert(matrix: List[List[int]]) -> Optional[List[List[int]]]:
    """Invert a k×k GF(2⁸) matrix by Gauss–Jordan elimination (None if singular)."""
    n = len(matrix)
    m = [list(row) + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = next((r for r in range(col, n) if m[r][col] != 0), None)
        if piv is None:
            return None
        m[col], m[piv] = m[piv], m[col]
        inv_p = _div(1, m[col][col])
        m[col] = [_mul(v, inv_p) for v in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0:
                factor = m[r][col]
                m[r] = [a ^ _mul(factor, b) for a, b in zip(m[r], m[col])]
    return [row[n:] for row in m]
