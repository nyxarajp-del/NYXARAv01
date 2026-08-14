"""NYXARA · njp/manifold.py — the Hyper-Dimensional Latent Manifold (🌌, NJP V.01).

The fabric thinks in **cells**: a settle produces a set of cell ids that fired, and that set is a
sparse, discrete, awkward thing to reason *about*. This module lifts it into a single
high-dimensional vector — one **snapshot** — where a thousand simultaneously-active variables are
one object rather than a thousand objects, and where relations between whole configurations
become geometry you can measure.

That is the actual content of "seeing many variables at once". A snapshot of 1000 co-active cells
is *one* vector of ``dim`` numbers, and comparing two whole world-states is one dot product
instead of a set intersection. Structure survives the lift because binding is invertible: a
``role ⊗ filler`` pair can be unbound later, so "the engine turns the flywheel" and "the flywheel
turns the engine" are different snapshots rather than the same bag.

**The predictive half — how a result is known before the event finishes.** :class:`Manifold` holds
an associative map ``W`` learned online by Hebbian outer product over *observed transitions*
(snapshot at t → snapshot at t+1). Once ``W`` has seen a regularity, :meth:`predict` returns the
next snapshot **without settling the fabric**, and :meth:`precognition` decodes it back to the
cells most likely to fire. The fabric can therefore answer from a prediction and spend a full
settle only when the prediction is untrusted. This is genuine forward inference over learned
dynamics.

Honest framing, kept the way this repo keeps it:

* **Prediction, not clairvoyance.** The map can only anticipate regularities it has actually
  observed, and it is guarded on **two** independent counts, because either alone lets a fluent
  wrong answer through. *Margin* asks whether the winners separate from the field. *Familiarity*
  asks whether this query resembles anything ever learned — a heteroassociative map returns
  something for any input, so a clean-looking answer to a question it has no experience of is its
  natural failure mode, and separation cannot catch that. Either floor unmet ⇒ ``trusted=False``
  **with the reason**: an admission, not a guess dressed up. Accuracy is recorded in
  :mod:`nyxara.njp.ledger` and is whatever it measures, never asserted.
* **Capacity is measured, not quoted.** :meth:`capacity_probe` finds where cleanup actually breaks
  *on this machine*. Superposing more than that degrades retrieval, and the probe says where.
* **No backpropagation in this map.** It is a Hebbian outer product with decay — one pass, online,
  local. Same seed ⇒ same manifold on every machine.
* This is classical high-dimensional computing (HDC/VSA). Nothing quantum, no extra physical
  dimensions: ``dim`` is a number of components in RAM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["Snapshot", "Prediction", "Manifold"]

try:  # numpy makes a 10,000-D manifold cheap; the pure-Python floor keeps it honest without it.
    import numpy as _np
except Exception:  # noqa: BLE001 — numpy is a capability, never a dependency
    _np = None


def _splitmix64(x: int) -> int:
    """A deterministic 64-bit mixer — the whole manifold's randomness, seeded and portable.

    Deliberately not :mod:`random`: every cell's hypervector must be reproducible from its id
    alone, on every machine and in any order, without holding a table of them all.
    """
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


@dataclass
class Snapshot:
    """One whole configuration of the fabric, as a single point on the manifold."""

    vector: Any = None                       # bipolar {-1,+1}, length `dim`
    cells: Tuple[int, ...] = ()              # what it was built from
    tick: int = 0

    @property
    def width(self) -> int:
        return len(self.cells)

    def to_dict(self) -> Dict[str, Any]:
        return {"cells": list(self.cells), "tick": self.tick, "width": self.width}


@dataclass
class Prediction:
    """What the manifold expects next — with the margin that says whether to believe it."""

    cells: List[int] = field(default_factory=list)
    margin: float = 0.0                      # separation of the winners from the field
    familiarity: float = 0.0                 # how like anything it has actually seen the query is
    trusted: bool = False                    # margin AND familiarity both cleared their floors
    samples: int = 0                         # how many transitions W has actually learned
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"cells": self.cells[:32], "margin": round(self.margin, 6),
                "familiarity": round(self.familiarity, 6), "trusted": self.trusted,
                "samples": self.samples, "reason": self.reason}


class Manifold:
    """Cells → one high-dimensional snapshot, plus an online-learned map from now to next."""

    def __init__(self, *, dim: int = 10000, seed: int = 42,
                 min_margin: float = 0.08, learn_rate: float = 1.0,
                 decay: float = 0.999, min_samples: int = 8,
                 max_pairs: int = 512, min_familiarity: float = 0.15) -> None:
        self.dim = max(16, int(dim))
        self.seed = int(seed)
        self.min_margin = float(min_margin)
        self.learn_rate = float(learn_rate)
        self.decay = float(decay)
        self.min_samples = max(1, int(min_samples))
        self.max_pairs = max(8, int(max_pairs))
        self.min_familiarity = float(min_familiarity)

        self._atoms: Dict[int, Any] = {}      # cell id -> its fixed hypervector (memoised)
        # The transition map is held in LOW-RANK form: the pairs (b, a) whose outer products it is
        # the sum of. W = Σ aᵢbᵢᵀ, so Wx = Σ aᵢ(bᵢ·x) — exactly the same operator, computed from
        # the pairs instead of from a materialised matrix. At the default dim the dense form is a
        # 10000×10000 float32 (400 MB) that has to be allocated and swept on every single turn;
        # this is ~10 MB and one small matmul. Same mathematics, three orders of magnitude less.
        self._before: List[Any] = []
        self._after: List[Any] = []
        self._weights: List[float] = []
        self.samples = 0                      # transitions actually learned
        self.hits = 0                         # predictions that later proved right
        self.misses = 0
        self._atom_rows: Dict[int, int] = {}
        # A growing store with spare capacity, not a matrix reallocated per append.
        self._atom_store: Any = None
        self._atom_used = 0

    # ---- atoms ----------------------------------------------------------- #
    def atom(self, cell_id: int) -> Any:
        """The fixed hypervector for one cell. Derived from its id, so it never needs storing."""
        got = self._atoms.get(cell_id)
        if got is not None:
            return got
        vec = self._mint(cell_id)
        # Memoise, but never without bound: a fabric that grows for months would otherwise keep
        # every atom it ever saw. Atoms are pure functions of the id, so dropping one costs a
        # recompute and nothing else.
        if len(self._atoms) > 65536:
            self._atoms.clear()
            # The stack is built from these and must go with them, or a row would point at an
            # atom that no longer exists.
            self._atom_store = None
            self._atom_used = 0
            self._atom_rows = {}
        self._atoms[cell_id] = vec
        return vec

    def _mint(self, cell_id: int) -> Any:
        """Deterministic bipolar vector for ``cell_id``. Same on every machine, forever."""
        base = _splitmix64((int(cell_id) << 1) ^ self.seed)
        if _np is not None:
            rng = _np.random.default_rng(base % (2**63))
            return rng.integers(0, 2, size=self.dim, dtype=_np.int8) * 2 - 1
        out: List[int] = []
        x = base
        for _ in range(self.dim):
            x = _splitmix64(x)
            out.append(1 if (x >> 17) & 1 else -1)
        return out

    # ---- lifting --------------------------------------------------------- #
    def encode(self, cells: Iterable[int], *, tick: int = 0) -> Snapshot:
        """Bundle a set of co-active cells into ONE snapshot vector.

        Bundling is sum-then-sign: the result is similar to every component and to nothing else,
        which is exactly the property that lets a thousand active variables be carried as a single
        object without any of them being privileged.
        """
        ids = tuple(dict.fromkeys(int(c) for c in cells))
        if not ids:
            return Snapshot(vector=self._zeros(), cells=(), tick=tick)
        if _np is not None:
            acc = _np.zeros(self.dim, dtype=_np.int32)
            for cid in ids:
                acc += self.atom(cid)
            # Ties (an even bundle summing to 0) break to +1 deterministically, so the same set
            # always yields the same snapshot — a coin-flip here would break reproducibility.
            vec = _np.where(acc >= 0, 1, -1).astype(_np.int8)
            return Snapshot(vector=vec, cells=ids, tick=tick)
        acc = [0] * self.dim
        for cid in ids:
            atom = self.atom(cid)
            for i in range(self.dim):
                acc[i] += atom[i]
        return Snapshot(vector=[1 if v >= 0 else -1 for v in acc], cells=ids, tick=tick)

    def bind(self, a: Any, b: Any) -> Any:
        """Bind two vectors (elementwise product) — invertible, so structure survives the lift.

        Binding is its own inverse over bipolar vectors, which is why ``role ⊗ filler`` can be
        unbound with the same operation. This is what keeps "A turns B" distinct from "B turns A".
        """
        if _np is not None:
            return (_np.asarray(a) * _np.asarray(b)).astype(_np.int8)
        return [x * y for x, y in zip(a, b)]

    def similarity(self, a: Any, b: Any) -> float:
        """Cosine over bipolar vectors — for {-1,+1} this is just the normalised dot product."""
        try:
            if _np is not None:
                return float(_np.dot(_np.asarray(a, dtype=_np.int32),
                                     _np.asarray(b, dtype=_np.int32))) / float(self.dim)
            return sum(x * y for x, y in zip(a, b)) / float(self.dim)
        except Exception:  # noqa: BLE001
            return 0.0

    def _zeros(self) -> Any:
        return _np.zeros(self.dim, dtype=_np.int8) if _np is not None else [0] * self.dim

    def _atoms_matrix(self, pool: Sequence[int]) -> Any:
        """Stacked atoms for ``pool``, grown **incrementally** as the fabric grows.

        The previous form rebuilt the whole stack whenever the pool changed by anything at all,
        and checked that with a full element-wise list comparison. Both halves were expensive in
        exactly the situation this fabric is always in: a cell set that grows by a few cells per
        turn. Measured on a 20k-cell fabric, the rebuild dominated the entire settle — 3.7 s of a
        6.6 s profile, against 0.65 s for the actual membrane arithmetic. The substrate was not
        the slow part; this was.

        Now each atom keeps a permanent row, new cells append, and a pool that is exactly the
        known rows in order is returned without copying at all.
        """
        ids = list(pool)
        missing = [c for c in ids if c not in self._atom_rows]
        if missing:
            self._grow(len(missing))
            for cid in missing:
                self._atom_store[self._atom_used] = self.atom(cid)
                self._atom_rows[cid] = self._atom_used
                self._atom_used += 1

        if self._atom_store is None or not self._atom_used:
            return _np.zeros((0, self.dim), dtype=_np.float32)
        rows = [self._atom_rows[c] for c in ids]
        # The overwhelmingly common case: the caller wants every atom, in row order. A view of the
        # filled prefix avoids a fancy-index copy of the whole thing on every single turn.
        if len(rows) == self._atom_used and rows == list(range(self._atom_used)):
            return self._atom_store[: self._atom_used]
        return self._atom_store[rows]

    def _grow(self, needed: int) -> None:
        """Make room for ``needed`` more atoms, doubling capacity when the buffer is full.

        The obvious implementation appends with ``vstack``, and it is quadratic: every append
        copies the entire matrix, so filling N rows copies O(N²) floats. On a 20k-cell fabric that
        single line was 3.6 s of a 7 s profile — worse than the rebuild it replaced. Doubling
        makes appends amortised O(1), which is what a growing store has to be when the whole
        premise of this substrate is that it keeps growing.
        """
        capacity = 0 if self._atom_store is None else self._atom_store.shape[0]
        if self._atom_used + needed <= capacity:
            return
        target = max(1024, capacity * 2, self._atom_used + needed)
        grown = _np.zeros((target, self.dim), dtype=_np.float32)
        if self._atom_store is not None and self._atom_used:
            grown[: self._atom_used] = self._atom_store[: self._atom_used]
        self._atom_store = grown

    # ---- the predictive half --------------------------------------------- #
    def learn_transition(self, before: Snapshot, after: Snapshot) -> None:
        """Teach the map that ``before`` was followed by ``after``. One online Hebbian step.

        ``W += η · after ⊗ beforeᵀ`` with a slow decay, so a regularity that stops holding fades
        instead of being defended forever. No gradient, no epochs, no stored dataset.

        Stored as the pair rather than as a materialised matrix — see :meth:`__init__`. The oldest
        pair is evicted past ``max_pairs``, which is a real bound: the map remembers a window of
        recent dynamics, not everything it has ever seen, and :meth:`stats` reports the window.
        """
        try:
            if before is None or after is None or not before.cells or not after.cells:
                return
            if _np is None:
                self.samples += 1     # the pure-Python floor records evidence but does not predict
                return
            if self.decay < 1.0 and self._weights:
                self._weights = [w * self.decay for w in self._weights]
            self._before.append(_np.asarray(before.vector, dtype=_np.int8))
            self._after.append(_np.asarray(after.vector, dtype=_np.int8))
            self._weights.append(self.learn_rate)
            if len(self._before) > self.max_pairs:
                # Drop the weakest, not simply the oldest: decay already means an old pair that
                # kept being reinforced outranks a recent one that did not.
                worst = min(range(len(self._weights)), key=lambda i: self._weights[i])
                for seq in (self._before, self._after, self._weights):
                    del seq[worst]
            self.samples += 1
        except Exception:  # noqa: BLE001 — learning never breaks a beat
            return

    def familiarity(self, before: Snapshot) -> float:
        """How like anything the map has actually learned this query is — 0.0 … 1.0.

        The best cosine against any stored ``before`` vector. This is the difference between
        "I have seen this situation" and "I can produce an output for this situation", and only
        the first of those licenses a prediction.
        """
        try:
            if _np is None or not self._before or before is None or not before.cells:
                return 0.0
            x = _np.asarray(before.vector, dtype=_np.float32)
            B = _np.asarray(self._before, dtype=_np.float32)
            return float(_np.max(B @ x)) / float(self.dim)
        except Exception:  # noqa: BLE001
            return 0.0

    def predict(self, before: Snapshot) -> Any:
        """The snapshot the map expects to follow ``before``. ``None`` when it cannot say.

        ``Wx = Σ aᵢ(bᵢ·x)`` — one matmul against the stored ``before`` vectors to get the
        coefficients, one weighted sum of the ``after`` vectors to get the result.
        """
        try:
            if _np is None or not self._before or before is None or not before.cells:
                return None
            x = _np.asarray(before.vector, dtype=_np.float32)
            B = _np.asarray(self._before, dtype=_np.float32)      # (n, dim)
            A = _np.asarray(self._after, dtype=_np.float32)       # (n, dim)
            w = _np.asarray(self._weights, dtype=_np.float32)     # (n,)
            coeff = (B @ x) * w / float(self.dim)                 # (n,)
            return A.T @ coeff                                    # (dim,)
        except Exception:  # noqa: BLE001
            return None

    def precognition(self, before: Snapshot, *, k: int = 16,
                     candidates: Optional[Sequence[int]] = None) -> Prediction:
        """Which cells are about to fire — **without settling the fabric**.

        The raw prediction is a point in the manifold; turning it back into cell ids means asking
        which known atoms it is closest to. ``candidates`` restricts that search to the cells the
        fabric actually holds, which is both faster and the only correct thing to do: a cell that
        does not exist cannot be about to fire.

        Returns ``trusted=False`` — with the reason — when the map has too little evidence or the
        winners are not separated from the field. An untrusted prediction is a real result: it is
        the fabric's signal to spend a full settle instead of guessing.
        """
        out = Prediction(samples=self.samples)
        try:
            if _np is None:
                out.reason = "numpy absent — the manifold does not predict without it"
                return out
            # Familiarity is measured FIRST and on every path, including the ones that decline.
            # It is not only a gate here: callers read it as a graded "how well do I know this"
            # signal, and a version that stayed 0.0 until the evidence floor was crossed would
            # make that signal a step function — indistinguishable from total ignorance right up
            # to the moment it isn't.
            out.familiarity = self.familiarity(before)
            if self.samples < self.min_samples:
                out.reason = f"only {self.samples} transitions learned, need {self.min_samples}"
                return out
            # Has anything like this ever been seen? A heteroassociative map returns *something*
            # for any input, so a confident-looking answer to a question it has no experience of
            # is its natural failure mode. Separation of the winners cannot catch that — the
            # winners separate cleanly from a field the map has simply never had reason to
            # distinguish. Familiarity of the QUERY is the honest guard, and it is what turns a
            # genuinely novel input into "I do not know this" rather than a fluent guess.
            if out.familiarity < self.min_familiarity:
                out.reason = (f"nothing like this has been seen "
                              f"(familiarity {out.familiarity:.3f} "
                              f"below {self.min_familiarity:.3f})")
                return out

            raw = self.predict(before)
            if raw is None:
                out.reason = "no transition map yet"
                return out
            pool = list(candidates) if candidates is not None else list(self._atoms.keys())
            if not pool:
                out.reason = "no candidate cells to decode onto"
                return out
            norm = float(_np.linalg.norm(raw)) or 1.0
            # One matmul over the stacked atoms rather than a dot product per cell in Python.
            # Decoding is the hot path — it runs on every anticipate, against every cell the
            # fabric holds — and the loop form made it the dominant cost of a whole turn.
            matrix = self._atoms_matrix(pool)
            sims = (matrix @ raw) / (norm * math.sqrt(self.dim))
            order = _np.argsort(-sims)
            scored: List[Tuple[float, int]] = [(float(sims[i]), pool[i]) for i in order]
            k = max(1, min(int(k), len(scored)))
            top = scored[:k]
            out.cells = [cid for _s, cid in top]
            # The margin is how far the winners stand off from the field: mean(winners) −
            # mean(everyone else). Deliberately NOT the gap between the weakest winner and the
            # strongest loser — that measures one adjacent pair, so it collapses toward zero
            # whenever k approaches the pool size, and reported an untrustworthy 0.0014 on a
            # prediction that was in fact clean. A cluster that genuinely separates scores high
            # however many cells the fabric happens to hold.
            rest = scored[k:]
            if rest:
                top_mean = sum(s for s, _c in top) / float(len(top))
                rest_mean = sum(s for s, _c in rest) / float(len(rest))
                out.margin = float(top_mean - rest_mean)
            else:
                out.margin = float(sum(s for s, _c in top) / float(len(top)))
            out.trusted = out.margin >= self.min_margin
            if not out.trusted:
                out.reason = f"margin {out.margin:.4f} below floor {self.min_margin:.4f}"
            return out
        except Exception:  # noqa: BLE001 — a failed foresight is no foresight, never a crash
            out.reason = "prediction failed"
            return out

    def score_prediction(self, predicted: Sequence[int], actual: Sequence[int]) -> float:
        """Jaccard overlap of predicted vs actual firing. Feeds the ledger's accuracy column."""
        try:
            p, a = set(int(x) for x in predicted), set(int(x) for x in actual)
            if not p and not a:
                return 1.0
            union = p | a
            if not union:
                return 0.0
            score = len(p & a) / float(len(union))
            if score >= 0.5:
                self.hits += 1
            else:
                self.misses += 1
            return score
        except Exception:  # noqa: BLE001
            return 0.0

    # ---- measured capacity ----------------------------------------------- #
    def capacity_probe(self, *, start: int = 2, stop: int = 512) -> int:
        """How many cells this manifold can bundle and still recover — **on this machine**.

        Not a quoted figure: it bundles `n` atoms, checks every one is still the nearest match,
        and returns the largest `n` that held. This is the honest ceiling on "many variables at
        once", and it is why that phrase is allowed to appear in this module's docstring.
        """
        try:
            if _np is None:
                return 0
            best = 0
            n = max(2, int(start))
            while n <= int(stop):
                ids = list(range(900_000, 900_000 + n))
                snap = self.encode(ids)
                ok = all(self.similarity(snap.vector, self.atom(c)) > 0.0 for c in ids)
                if not ok:
                    break
                best = n
                n *= 2
            return best
        except Exception:  # noqa: BLE001
            return 0

    # ---- reporting / persistence ----------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {"dim": self.dim, "samples": self.samples, "atoms_cached": len(self._atoms),
                "min_familiarity": self.min_familiarity,
                "pairs": len(self._before), "max_pairs": self.max_pairs, "hits": self.hits, "misses": self.misses,
                "accuracy": round(self.hits / total, 4) if total else None,
                "numpy": _np is not None}

    def to_dict(self) -> Dict[str, Any]:
        """Counters only — ``W`` is ``dim × dim`` floats and does not belong in a JSON sidecar.

        At the default dim that matrix is 400 MB; writing it into ``njp.json`` would make every
        save a minutes-long stall. The map is rebuilt by living, which is the point of an online
        learner, and the counters below keep the ledger's history honest across a restart.
        """
        return {"dim": self.dim, "seed": self.seed, "samples": self.samples,
                "hits": self.hits, "misses": self.misses}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.samples = int(d.get("samples", 0))
            self.hits = int(d.get("hits", 0))
            self.misses = int(d.get("misses", 0))
        except Exception:  # noqa: BLE001 — a corrupt sidecar leaves a fresh manifold
            pass
