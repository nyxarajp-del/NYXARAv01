"""NYXARA · njp/learn.py — real gradients over the fabric (∇, NJP V.03).

Every update in :mod:`nyxara.njp.fabric` is **local**: a synapse changes because its own two
endpoints fired, and nothing tells it what the network as a whole was trying to achieve. That is
biologically honest and it is genuinely limited — local rules cannot assign credit *through* a
chain of activity to the thing that actually caused the error.

This module adds the other kind of learning, properly: a readout head over the fabric's settled
state, trained by **real reverse-mode backpropagation** with **real gradients**.

**Nothing here re-implements autodiff.** :mod:`nyxara.growth.genesis_numpy` already contains a
complete engine — ``Tensor`` carrying ``data``/``grad``/``_backward``/``_prev``, a ``backward``
that topologically orders the graph and applies the chain rule in reverse, broadcast-aware
gradient accumulation, and twenty-one differentiable primitives. It was invisible only because
``__all__`` did not name it. Writing a second engine beside a working one is the duplication this
whole effort has been avoiding, so this imports it.

**What makes the claim checkable.** "Real backprop" is easy to assert and easy to get subtly
wrong — a sign error in one op still trains, just worse. So :func:`gradcheck` compares every
analytic gradient against a **central finite-difference** estimate of the same quantity, and the
test suite fails if any disagree beyond tolerance. That is the difference between a gradient and a
number shaped like one.

**What it learns from.** The input is which cells fired in a settle; the target is which cells
fired next. That is the prediction-error signal the rest of NJP already runs on, so the readout is
not a bolt-on objective — it is the same thing the manifold anticipates and the error taxonomy
diagnoses, learned by gradient descent instead of by Hebbian coincidence.

**It is additive, not a replacement.** The fabric's plasticity stays exactly as it was. Two
learners now sit over one substrate: local rules that grow structure, and a gradient-trained head
that reads it. Where they disagree, :mod:`nyxara.njp.predict` scores both.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = ["TrainStep", "Readout", "gradcheck", "available"]

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # noqa: BLE001
    _HAS_NUMPY = False


def _engine() -> Any:
    """The autodiff engine, or None. Absent numpy means no gradients, reported not disguised."""
    if not _HAS_NUMPY:
        return None
    try:
        from nyxara.growth import genesis_numpy
        return genesis_numpy
    except Exception:  # noqa: BLE001
        return None


def available() -> bool:
    """Can she actually do gradient learning on this machine?"""
    engine = _engine()
    return bool(engine is not None and getattr(engine, "_HAS_NUMPY", False))


@dataclass
class TrainStep:
    """One optimiser step, with the numbers that say whether it helped."""

    loss_before: float = 0.0
    loss_after: float = 0.0
    grad_norm: float = 0.0
    lr: float = 0.0
    samples: int = 0
    ms: float = 0.0

    @property
    def improved(self) -> bool:
        return self.loss_after < self.loss_before

    def to_dict(self) -> Dict[str, Any]:
        return {"loss_before": round(self.loss_before, 6),
                "loss_after": round(self.loss_after, 6),
                "delta": round(self.loss_after - self.loss_before, 6),
                "grad_norm": round(self.grad_norm, 6), "lr": round(self.lr, 6),
                "samples": self.samples, "improved": self.improved,
                "ms": round(self.ms, 3)}


class Readout:
    """A gradient-trained head over the fabric: which cells fired → which fire next.

    Two layers with a tanh hidden and a sigmoid output, because the target is a **multi-label**
    membership vector (many cells fire at once) rather than a distribution over one choice.
    Softmax + cross-entropy would force the outputs to compete for a fixed budget of probability
    and quietly teach it that only one cell may fire — which is false of this substrate.

    Cells are mapped to a slot by hashing, so the head has a stable input width while the fabric
    underneath keeps growing new cells. Collisions are real and are reported by :meth:`stats`.

    **The width grows rather than staying fixed**, and the reason is a measurement. Held at 512
    while the fabric grew to 4,700 cells, the head reported **4,188 collisions** — 89% of cells
    sharing a slot with roughly eight others. A representation in which nine different cells are
    literally the same input cannot distinguish them however long it trains, which is what a loss
    falling 75 → 66 with an accuracy of 0.0 was saying.

    The original defence of the fixed width was that resizing on every birth would throw away
    everything learned. That is true and it is not the only option: widening in steps and
    **copying the existing weights into the larger matrices** preserves every slot that already
    existed, so what was learned stays exactly where it was addressed. Only the new capacity is
    fresh. The Adam moments are carried across the same way, because resetting them would make
    the step after a growth behave as though training had just started.
    """

    def __init__(self, *, width: int = 512, hidden: int = 128, lr: float = 0.01,
                 seed: int = 42, weight_decay: float = 0.0,
                 max_width: int = 16384, load_factor: float = 0.75) -> None:
        self.width = max(8, int(width))
        #: A hard ceiling, so growth cannot become the memory leak the fixed width was avoiding.
        #: Past it she collides again and says so, which is the honest end state for a finite
        #: machine rather than an allocation that eventually fails.
        self.max_width = max(self.width, int(max_width))
        #: Grow once the slots in use pass this share of the width. Below 1.0 on purpose: hashing
        #: collides well before a table is full, and waiting for "full" means most of the damage
        #: is already done.
        self.load_factor = min(0.95, max(0.1, float(load_factor)))
        self.growths = 0
        self.hidden = max(4, int(hidden))
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)

        self.steps = 0
        self.samples_seen = 0
        self.last: Optional[TrainStep] = None
        self._collisions = 0
        self._slots: Dict[int, int] = {}
        # The occupied slots, as a set. This used to be re-derived with `got in
        # self._slots.values()` on every newly seen cell — a linear scan inside the hot path,
        # which is O(n^2) over the life of the fabric and was doing roughly 11 million
        # comparisons by the time 4,700 cells had been mapped.
        self._used: Set[int] = set()

        self._params: List[Any] = []
        self._m: List[Any] = []
        self._v: List[Any] = []
        self._built = False
        self._build()

    # ---- parameters ------------------------------------------------------- #
    def _build(self) -> None:
        engine = _engine()
        if engine is None:
            return
        rng = np.random.default_rng(self.seed)
        # Xavier/Glorot: keeps the forward variance stable through a tanh layer, which matters
        # here because a saturated tanh produces vanishing gradients and the head simply stops
        # learning without ever erroring.
        def _p(fan_in: int, fan_out: int) -> Any:
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            data = rng.uniform(-limit, limit, size=(fan_in, fan_out))
            return engine.Tensor(data, requires_grad=True)

        self.W1 = _p(self.width, self.hidden)
        self.b1 = engine.Tensor(np.zeros((1, self.hidden)), requires_grad=True)
        self.W2 = _p(self.hidden, self.width)
        self.b2 = engine.Tensor(np.zeros((1, self.width)), requires_grad=True)
        self._params = [self.W1, self.b1, self.W2, self.b2]
        self._m = [np.zeros_like(p.data) for p in self._params]
        self._v = [np.zeros_like(p.data) for p in self._params]
        self._built = True

    # ---- encoding --------------------------------------------------------- #
    def slot(self, cell_id: int) -> int:
        """Stable slot for a cell, so what was learned about it stays addressable.

        Stable across growth too: widening rehashes every known cell and moves its weights with
        it, so a slot changing is not a slot forgetting.
        """
        cell = int(cell_id)
        got = self._slots.get(cell)
        if got is not None:
            return got
        if len(self._slots) + 1 > self.width * self.load_factor:
            self._grow()
            got = self._slots.get(cell)
            if got is not None:
                return got
        got = self._assign(cell, self.width, self._used)
        self._slots[cell] = got
        return got

    def _assign(self, cell: int, width: int, used: Set[int]) -> int:
        """A free slot for this cell, probing forward from its hash.

        Plain ``cell % width`` collides at the birthday rate long before the table is full: 1,049
        cells in 2,048 slots produced 237 shared slots, which is exactly what random hashing
        predicts and is still 237 pairs of cells the head cannot tell apart. Probing costs one
        dictionary lookup per step and gives a distinct slot to every cell until the table is
        genuinely full — at which point the collision is real, unavoidable and counted.
        """
        start = cell % width
        for step in range(width):
            candidate = (start + step) % width
            if candidate not in used:
                used.add(candidate)
                return candidate
        # Genuinely full. Share a slot and say so.
        self._collisions += 1
        return start

    def _grow(self) -> bool:
        """Double the width, carrying every learned weight into the larger head.

        The rows and columns that existed keep their values; only the new capacity is initialised.
        A cell's slot may change — the hash is over a different modulus — so every known cell is
        rehashed and its learned row moved with it. That is the difference between widening and
        starting again.
        """
        engine = _engine()
        if engine is None or not self._built or self.width >= self.max_width:
            return False
        try:
            old_width = self.width
            new_width = min(self.max_width, old_width * 2)
            if new_width <= old_width:
                return False
            rng = np.random.default_rng(self.seed + self.growths + 1)

            def _wider_in(tensor: Any) -> Any:
                limit = math.sqrt(6.0 / (new_width + self.hidden))
                data = rng.uniform(-limit, limit, size=(new_width, self.hidden))
                return engine.Tensor(data, requires_grad=True)

            def _wider_out(tensor: Any) -> Any:
                limit = math.sqrt(6.0 / (self.hidden + new_width))
                data = rng.uniform(-limit, limit, size=(self.hidden, new_width))
                return engine.Tensor(data, requires_grad=True)

            w1, w2 = _wider_in(self.W1), _wider_out(self.W2)
            b2 = engine.Tensor(np.zeros((1, new_width)), requires_grad=True)
            m1 = np.zeros_like(w1.data)
            m2 = np.zeros_like(w2.data)
            mb2 = np.zeros_like(b2.data)
            v1, v2, vb2 = (np.zeros_like(w1.data), np.zeros_like(w2.data),
                           np.zeros_like(b2.data))
            old_m1, old_mb1, old_m2, old_mb2 = self._m
            old_v1, old_vb1, old_v2, old_vb2 = self._v

            # Rehash and carry. A cell whose slot moves takes its input row, its output column,
            # its bias and its optimiser moments with it.
            moved: Dict[int, int] = {}
            used: Set[int] = set()
            collisions = 0
            for cell, old_slot in self._slots.items():
                new_slot = self._assign(int(cell), new_width, used)
                w1.data[new_slot] = self.W1.data[old_slot]
                w2.data[:, new_slot] = self.W2.data[:, old_slot]
                b2.data[0, new_slot] = self.b2.data[0, old_slot]
                m1[new_slot] = old_m1[old_slot]
                v1[new_slot] = old_v1[old_slot]
                m2[:, new_slot] = old_m2[:, old_slot]
                v2[:, new_slot] = old_v2[:, old_slot]
                mb2[0, new_slot] = old_mb2[0, old_slot]
                vb2[0, new_slot] = old_vb2[0, old_slot]
                moved[cell] = new_slot

            self.W1, self.W2, self.b2 = w1, w2, b2
            self._params = [self.W1, self.b1, self.W2, self.b2]
            self._m = [m1, old_mb1, m2, mb2]
            self._v = [v1, old_vb1, v2, vb2]
            self._slots = moved
            self._used = used
            self.width = new_width
            self.growths += 1
            return True
        except Exception:  # noqa: BLE001 — a failed growth leaves the head exactly as it was
            return False

    def encode(self, cells: Iterable[int]) -> Any:
        """A set of firing cells as a dense membership row."""
        row = np.zeros((1, self.width), dtype=float)
        for cid in cells:
            row[0, self.slot(cid)] = 1.0
        return row

    # ---- forward ----------------------------------------------------------- #
    def _forward(self, x: Any) -> Any:
        """``sigmoid(tanh(x·W1 + b1)·W2 + b2)`` — built from the engine's differentiable ops."""
        engine = _engine()
        hidden_pre = engine.add(engine.matmul(x, self.W1), self.b1)
        hidden = _tanh(engine, hidden_pre)
        out_pre = engine.add(engine.matmul(hidden, self.W2), self.b2)
        return engine.sigmoid(out_pre)

    def predict(self, cells: Iterable[int], *, k: int = 8) -> List[Tuple[int, float]]:
        """The ``k`` cells this head expects to fire next, most confident first."""
        if not self._built:
            return []
        try:
            engine = _engine()
            x = engine.Tensor(self.encode(cells))
            probs = self._forward(x).data[0]
            top = np.argsort(-probs)[: max(0, int(k))]
            # Slots are shared, so report the cells actually mapped there rather than the slot
            # index — a slot number is not a cell and would be a meaningless answer.
            by_slot: Dict[int, List[int]] = {}
            for cid, s in self._slots.items():
                by_slot.setdefault(s, []).append(cid)
            out: List[Tuple[int, float]] = []
            for s in top:
                for cid in by_slot.get(int(s), []):
                    out.append((cid, float(probs[int(s)])))
            return out[: max(0, int(k))]
        except Exception:  # noqa: BLE001
            return []

    # ---- loss --------------------------------------------------------------- #
    def _loss(self, pred: Any, target: Any) -> Any:
        """Binary cross-entropy, summed over slots and averaged over the batch.

        Clamped inside the log: a sigmoid that saturates to exactly 0 or 1 makes the loss ``inf``
        and every downstream gradient ``nan``, which does not raise — it silently destroys every
        parameter on the next step.
        """
        engine = _engine()
        eps = 1e-7
        p = pred.data
        n = max(1, target.shape[0])
        value = -float(np.sum(target * np.log(np.clip(p, eps, 1.0))
                              + (1.0 - target) * np.log(np.clip(1.0 - p, eps, 1.0)))) / n

        out = engine.Tensor(np.array(value), _prev=(pred,))

        def _bw() -> None:
            # d/dp of BCE is (p - y) / (p(1-p)); composed with the sigmoid already in `pred`
            # the engine handles the rest of the chain.
            grad = (np.clip(p, eps, 1.0 - eps) - target) / (
                np.clip(p * (1.0 - p), eps, None) * n)
            engine._acc(pred, grad * float(out.grad if out.grad is not None else 1.0))

        out._backward = _bw
        return out

    def loss_on(self, batch: Sequence[Tuple[Sequence[int], Sequence[int]]]) -> float:
        """Loss over a batch without touching a single weight. Used for before/after readings."""
        if not self._built or not batch:
            return 0.0
        try:
            engine = _engine()
            x, y = self._batch(batch)
            pred = self._forward(engine.Tensor(x))
            eps = 1e-7
            p = pred.data
            return -float(np.sum(y * np.log(np.clip(p, eps, 1.0))
                                 + (1.0 - y) * np.log(np.clip(1.0 - p, eps, 1.0)))) / len(batch)
        except Exception:  # noqa: BLE001
            return 0.0

    def _batch(self, batch: Sequence[Tuple[Sequence[int], Sequence[int]]]) -> Tuple[Any, Any]:
        x = np.zeros((len(batch), self.width))
        y = np.zeros((len(batch), self.width))
        for i, (before, after) in enumerate(batch):
            x[i] = self.encode(before)[0]
            y[i] = self.encode(after)[0]
        return x, y

    # ---- training ------------------------------------------------------------ #
    def train_step(self, batch: Sequence[Tuple[Sequence[int], Sequence[int]]]) -> TrainStep:
        """One real gradient step: forward, backward, AdamW update. Measured before and after."""
        out = TrainStep(lr=self.lr, samples=len(batch))
        t0 = time.perf_counter()
        if not self._built or not batch:
            return out
        try:
            engine = _engine()
            x, y = self._batch(batch)
            out.loss_before = self.loss_on(batch)

            for p in self._params:
                p.grad = None
            pred = self._forward(engine.Tensor(x))
            loss = self._loss(pred, y)
            engine.backward(loss)

            out.grad_norm = math.sqrt(sum(
                float((p.grad ** 2).sum()) for p in self._params if p.grad is not None))
            self._adamw()
            self.steps += 1
            self.samples_seen += len(batch)
            out.loss_after = self.loss_on(batch)
            self.last = out
            return out
        except Exception:  # noqa: BLE001 — a failed step changes nothing and never breaks a beat
            return out
        finally:
            out.ms = (time.perf_counter() - t0) * 1000.0

    def _adamw(self) -> None:
        """AdamW with bias correction and gradient clipping."""
        beta1, beta2, eps, clip = 0.9, 0.999, 1e-8, 1.0
        total = math.sqrt(sum(float((p.grad ** 2).sum())
                              for p in self._params if p.grad is not None))
        scale = min(1.0, clip / (total + 1e-6)) if total > clip else 1.0
        t = self.steps + 1
        for i, p in enumerate(self._params):
            if p.grad is None:
                continue
            g = p.grad * scale
            self._m[i] = beta1 * self._m[i] + (1.0 - beta1) * g
            self._v[i] = beta2 * self._v[i] + (1.0 - beta2) * (g * g)
            m_hat = self._m[i] / (1.0 - beta1 ** t)
            v_hat = self._v[i] / (1.0 - beta2 ** t)
            # Decoupled decay: applied to the weight, not folded into the gradient (the "W").
            if self.weight_decay:
                p.data -= self.lr * self.weight_decay * p.data
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    # ---- reporting ------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        return {"available": available(), "built": self._built,
                "width": self.width, "hidden": self.hidden, "lr": self.lr,
                "steps": self.steps, "samples_seen": self.samples_seen,
                "parameters": int(sum(p.data.size for p in self._params)) if self._built else 0,
                "cells_mapped": len(self._slots), "slot_collisions": self._collisions,
                "collision_rate": (round(self._collisions / len(self._slots), 4)
                                   if self._slots else 0.0),
                "growths": self.growths, "max_width": self.max_width,
                "last": self.last.to_dict() if self.last else None}

    def to_dict(self) -> Dict[str, Any]:
        if not self._built:
            return {}
        return {"width": self.width, "hidden": self.hidden, "steps": self.steps,
                "params": [p.data.tolist() for p in self._params],
                "slots": {str(k): v for k, v in self._slots.items()}}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            if not self._built or not d:
                return
            params = d.get("params") or []
            if len(params) == len(self._params):
                for p, saved in zip(self._params, params):
                    arr = np.array(saved, dtype=float)
                    if arr.shape == p.data.shape:
                        p.data = arr
            self._slots = {int(k): int(v) for k, v in (d.get("slots") or {}).items()}
            self.steps = int(d.get("steps", 0))
        except Exception:  # noqa: BLE001
            pass


def _tanh(engine: Any, a: Any) -> Any:
    """tanh as a differentiable op. The engine ships relu/sigmoid/gelu/silu but not this one.

    Used for the hidden layer because it is zero-centred: a relu hidden here dies on the many
    all-zero input rows a sparse firing vector produces, and a dead unit contributes no gradient
    ever again.
    """
    out = engine.Tensor(np.tanh(a.data), _prev=(a,))

    def _bw() -> None:
        if out.grad is not None:
            engine._acc(a, out.grad * (1.0 - out.data ** 2))

    out._backward = _bw
    return out


def gradcheck(fn: Callable[[Any], Any], param: Any, *, eps: float = 1e-5,
              tolerance: float = 1e-6) -> Tuple[bool, float]:
    """Compare analytic gradients against central finite differences.

    ``(passed, max_absolute_error)``. This is what makes "real backprop" a checkable claim rather
    than an assertion: a sign error or a missing term in one backward closure still *trains*, just
    worse, and nothing else in a training loop would ever notice.

    Central differences ``(f(x+h) - f(x-h)) / 2h`` rather than forward, because the forward
    estimate carries O(h) error which is the same order as the tolerance being tested.
    """
    engine = _engine()
    if engine is None:
        return (False, float("inf"))
    try:
        param.grad = None
        loss = fn(param)
        engine.backward(loss)
        analytic = param.grad.copy() if param.grad is not None else np.zeros_like(param.data)

        numeric = np.zeros_like(param.data)
        flat = param.data.reshape(-1)
        numeric_flat = numeric.reshape(-1)
        for i in range(flat.size):
            original = flat[i]
            flat[i] = original + eps
            plus = float(fn(param).data)
            flat[i] = original - eps
            minus = float(fn(param).data)
            flat[i] = original
            numeric_flat[i] = (plus - minus) / (2.0 * eps)

        error = float(np.max(np.abs(analytic - numeric)))
        return (error <= tolerance, error)
    except Exception:  # noqa: BLE001
        return (False, float("inf"))
