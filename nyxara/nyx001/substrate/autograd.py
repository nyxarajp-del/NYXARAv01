"""NYXARA · nyx001/substrate/autograd.py — Layer 0's differentiable computation (∂, borrowed).

The NYX V.001 charter forbids inheriting pretrained intelligence and permits exactly one kind of
foundation: *mathematics and differentiable computation*. This module is that foundation — and it
is deliberately **not new code**.

:mod:`nyxara.growth.genesis_numpy` already carries a correct reverse-mode autograd over NumPy:
a :class:`Tensor` that records a local backward per op, a :func:`backward` that walks the graph
once in reverse-topological order, and the ops a cognitive substrate actually needs — ``matmul``,
``add``, ``mul``, ``relu``, ``sigmoid``, ``silu``, ``gelu``, ``softmax_lastdim``, ``layernorm``,
``rmsnorm``, ``embed``, ``shift_time``, ``cross_entropy``, ``mean_entropy`` — plus ``diag_scan``,
a genuine diagonal gated linear recurrence (``h_t = a ⊙ h_{t-1} + x_t``). That engine is already
exercised by the Genesis search and its own tests. Re-implementing it here would have produced a
second, less-tested copy of the same arithmetic, so this module **re-exports** it and adds only
the handful of ops the eighteen layers need and it did not have: ``sub``, ``div``, ``neg``,
``exp``, ``log``, ``tanh``, ``square``, ``sum_all``, ``mean_all``, ``sum_lastdim``,
``mean_lastdim``, ``concat_last``, ``mse`` and ``cosine``.

``diag_scan`` matters more than it looks. The charter forbids copying the Transformer recipe, and
a selective state-space scan is a different mechanism with a different inductive bias — linear in
sequence length, carrying a recurrent state rather than an all-pairs attention matrix. Track B's
sequence model (:mod:`nyxara.nyx001.lingua.sequence`) is built on it for exactly that reason.

Honest about the ceiling: this is float64 NumPy with a Python-level tape. It is *real* automatic
differentiation — gradients are exact, not estimated — but it is not a production tensor library.
There is no GPU path, no kernel fusion, no graph optimisation, and the tape is rebuilt every
forward pass. That is a deliberate trade: the charter asks for a substrate that is inspectable and
owned, not one that is fast.

Without NumPy this module reports ``HAS_NUMPY = False`` and every layer above it declines cleanly
rather than half-working — the same posture ``genesis_numpy`` already takes.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from nyxara.growth.genesis_numpy import (  # noqa: F401 — re-exported as Layer 0's op set
    Tensor,
    _acc,
    _HAS_NUMPY as HAS_NUMPY,
    add,
    add_const,
    backward,
    cross_entropy,
    diag_scan,
    embed,
    gelu,
    layernorm,
    matmul,
    mean_entropy,
    mul,
    relu,
    reshape,
    rmsnorm,
    scale,
    shift_time,
    sigmoid,
    silu,
    softmax_lastdim,
    transpose_last2,
)

try:
    import numpy as np
except Exception:  # noqa: BLE001 — mirrored from genesis_numpy: declare, never half-work
    np = None  # type: ignore[assignment]

__all__ = [
    "HAS_NUMPY", "Tensor", "backward",
    # re-exported primitives
    "matmul", "add", "add_const", "mul", "scale", "reshape", "transpose_last2",
    "relu", "sigmoid", "silu", "gelu", "softmax_lastdim", "layernorm", "rmsnorm",
    "embed", "shift_time", "diag_scan", "cross_entropy", "mean_entropy",
    # added here
    "neg", "sub", "div", "exp", "log", "tanh", "square", "abs_val",
    "sum_all", "mean_all", "sum_lastdim", "mean_lastdim", "concat_last",
    "mse", "cosine", "constant", "parameter",
]


# --------------------------------------------------------------------------- #
# Constructors
# --------------------------------------------------------------------------- #
def constant(data: "np.ndarray") -> Tensor:
    """A leaf that carries no gradient — data entering the graph from the world."""
    return Tensor(np.asarray(data, dtype=np.float64))


def parameter(data: "np.ndarray") -> Tensor:
    """A leaf that *is* learned — the only kind of tensor an optimizer may touch."""
    return Tensor(np.asarray(data, dtype=np.float64), requires_grad=True)


# --------------------------------------------------------------------------- #
# Elementwise ops the eighteen layers need and genesis_numpy did not export
# --------------------------------------------------------------------------- #
def neg(a: Tensor) -> Tensor:
    return scale(a, -1.0)


def sub(a: Tensor, b: Tensor) -> Tensor:
    """``a - b``. The world model's prediction error is literally this op."""
    out = Tensor(a.data - b.data, _prev=(a, b))

    def _bw() -> None:
        _acc(a, out.grad)
        _acc(b, -out.grad)
    out._backward = _bw
    return out


def div(a: Tensor, b: Tensor, *, eps: float = 1e-12) -> Tensor:
    """``a / b``, guarded so a zero denominator degrades instead of producing NaN."""
    denom = np.where(np.abs(b.data) < eps, eps, b.data)
    out = Tensor(a.data / denom, _prev=(a, b))

    def _bw() -> None:
        _acc(a, out.grad / denom)
        _acc(b, -out.grad * a.data / (denom * denom))
    out._backward = _bw
    return out


def exp(a: Tensor) -> Tensor:
    e = np.exp(np.clip(a.data, -60.0, 60.0))    # clipped: an overflow here poisons a whole cycle
    out = Tensor(e, _prev=(a,))

    def _bw() -> None:
        _acc(a, out.grad * e)
    out._backward = _bw
    return out


def log(a: Tensor, *, eps: float = 1e-12) -> Tensor:
    safe = np.maximum(a.data, eps)
    out = Tensor(np.log(safe), _prev=(a,))

    def _bw() -> None:
        _acc(a, out.grad / safe)
    out._backward = _bw
    return out


def tanh(a: Tensor) -> Tensor:
    t = np.tanh(a.data)
    out = Tensor(t, _prev=(a,))

    def _bw() -> None:
        _acc(a, out.grad * (1.0 - t * t))
    out._backward = _bw
    return out


def square(a: Tensor) -> Tensor:
    out = Tensor(a.data * a.data, _prev=(a,))

    def _bw() -> None:
        _acc(a, out.grad * 2.0 * a.data)
    out._backward = _bw
    return out


def abs_val(a: Tensor) -> Tensor:
    """``|a|`` — the op an L1 sparsity penalty is actually made of.

    Subgradient 0 at the origin (rather than ±1), which is the convention that lets an L1 term
    park a coordinate *at* zero instead of oscillating across it. That is the whole point of
    preferring L1 over L2 for sparsity, so the choice is not incidental.
    """
    out = Tensor(np.abs(a.data), _prev=(a,))

    def _bw() -> None:
        _acc(a, out.grad * np.sign(a.data))
    out._backward = _bw
    return out


# --------------------------------------------------------------------------- #
# Reductions
# --------------------------------------------------------------------------- #
def sum_all(a: Tensor) -> Tensor:
    out = Tensor(np.array(a.data.sum()), _prev=(a,))

    def _bw() -> None:
        _acc(a, np.ones_like(a.data) * out.grad)
    out._backward = _bw
    return out


def mean_all(a: Tensor) -> Tensor:
    n = max(1, a.data.size)
    out = Tensor(np.array(a.data.mean()), _prev=(a,))

    def _bw() -> None:
        _acc(a, np.ones_like(a.data) * (out.grad / n))
    out._backward = _bw
    return out


def sum_lastdim(a: Tensor) -> Tensor:
    out = Tensor(a.data.sum(axis=-1), _prev=(a,))

    def _bw() -> None:
        _acc(a, np.expand_dims(out.grad, -1) * np.ones_like(a.data))
    out._backward = _bw
    return out


def mean_lastdim(a: Tensor) -> Tensor:
    n = max(1, a.data.shape[-1])
    out = Tensor(a.data.mean(axis=-1), _prev=(a,))

    def _bw() -> None:
        _acc(a, np.expand_dims(out.grad, -1) * np.ones_like(a.data) / n)
    out._backward = _bw
    return out


def concat_last(parts: Sequence[Tensor]) -> Tensor:
    """Join tensors along the last axis — how a layer fuses several evidence streams."""
    items: List[Tensor] = list(parts)
    if not items:
        raise ValueError("concat_last needs at least one tensor")
    if len(items) == 1:
        return items[0]
    sizes = [t.data.shape[-1] for t in items]
    out = Tensor(np.concatenate([t.data for t in items], axis=-1), _prev=tuple(items))

    def _bw() -> None:
        start = 0
        for t, width in zip(items, sizes):
            _acc(t, out.grad[..., start:start + width])
            start += width
    out._backward = _bw
    return out


# --------------------------------------------------------------------------- #
# Objective primitives
# --------------------------------------------------------------------------- #
def mse(pred: Tensor, target: "np.ndarray") -> Tensor:
    """Mean squared error against a fixed target — the world model's ``e_t = D(s, ŝ)``.

    ``target`` is plain NumPy on purpose: it is an observation of the world, not something the
    optimizer is allowed to move toward the prediction.
    """
    tgt = np.asarray(target, dtype=np.float64)
    diff = pred.data - tgt
    n = max(1, diff.size)
    out = Tensor(np.array((diff * diff).mean()), _prev=(pred,))

    def _bw() -> None:
        _acc(pred, out.grad * (2.0 / n) * diff)
    out._backward = _bw
    return out


def cosine(a: Tensor, b: Tensor, *, eps: float = 1e-12) -> Tensor:
    """Cosine similarity over the last axis — the similarity concept formation is built on."""
    dot = (a.data * b.data).sum(axis=-1)
    na = np.sqrt((a.data * a.data).sum(axis=-1)) + eps
    nb = np.sqrt((b.data * b.data).sum(axis=-1)) + eps
    out = Tensor(dot / (na * nb), _prev=(a, b))

    def _bw() -> None:
        g = np.expand_dims(out.grad, -1)
        cos = np.expand_dims(out.data, -1)
        _acc(a, g * (b.data / np.expand_dims(na * nb, -1) - cos * a.data / np.expand_dims(na * na, -1)))
        _acc(b, g * (a.data / np.expand_dims(na * nb, -1) - cos * b.data / np.expand_dims(nb * nb, -1)))
    out._backward = _bw
    return out


def shape_of(t: Tensor) -> Tuple[int, ...]:
    return tuple(t.data.shape)
