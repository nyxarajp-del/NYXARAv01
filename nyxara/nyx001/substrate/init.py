"""NYXARA · nyx001/substrate/init.py — Layer 0's random birth (🎲, the only inheritance allowed).

The charter's prohibition list bans pretrained weights outright and its allow-list names *random
initialization* as a permitted foundation. This module is that line in the sand: every parameter
NYX V.001 owns is born here, from a seeded pseudo-random generator, and from nothing else. No
tensor in :mod:`nyxara.nyx001` is ever loaded from a checkpoint that some other model trained.

Four schemes, chosen because they are what the layers above actually need:

* :func:`xavier` — Glorot: variance ``2/(fan_in+fan_out)``, for the saturating paths (tanh,
  sigmoid) where keeping forward *and* backward variance flat both ways matters.
* :func:`he` — Kaiming: variance ``2/fan_in``, for the ReLU/GELU paths, where half the units are
  dead at init and the surviving half must carry the signal.
* :func:`orthogonal` — a QR-factored orthonormal matrix, for recurrent state transitions. This is
  the one that earns its keep: a gated recurrence with a badly-scaled transition either explodes
  or forgets everything within a few steps, and the world model's rollout depends on neither.
* :func:`uniform_gate` — small positive values in ``(lo, hi)`` for the decay gates of
  :func:`~nyxara.nyx001.substrate.autograd.diag_scan`, which must start inside ``(0, 1)`` or the
  scan is unstable on its first pass.

Everything takes an explicit :class:`numpy.random.Generator`, so a run is reproducible from its
seed alone — which is what makes an ablation honest: two configurations that differ only in a
flag really do differ only in that flag.

Honest: seeded reproducibility is exact for a fixed NumPy version and a fixed op order. It is not
a cryptographic guarantee, and it does not survive changing the number of layers built before a
given tensor, because that changes how far the generator has been advanced.
"""

from __future__ import annotations

from typing import Tuple

from nyxara.nyx001.substrate.autograd import HAS_NUMPY, Tensor, parameter

try:
    import numpy as np
except Exception:  # noqa: BLE001 — the whole substrate declines without NumPy
    np = None  # type: ignore[assignment]

__all__ = [
    "HAS_NUMPY", "rng_for", "xavier", "he", "orthogonal", "uniform_gate", "zeros", "ones",
    "count_params",
]


def rng_for(seed: int, *, stream: int = 0) -> "np.random.Generator":
    """A generator for one named stream of a run.

    ``stream`` lets each layer draw from its own sub-generator, so adding a layer does not shift
    every later layer's weights — the reproducibility caveat in the module docstring, mitigated.
    """
    return np.random.default_rng([int(seed), int(stream)])


def _fan(shape: Tuple[int, ...]) -> Tuple[int, int]:
    if len(shape) == 1:
        return int(shape[0]), int(shape[0])
    fan_in = int(shape[-2])
    fan_out = int(shape[-1])
    return fan_in, fan_out


def xavier(rng: "np.random.Generator", *shape: int, gain: float = 1.0) -> Tensor:
    """Glorot-normal init — variance ``2/(fan_in+fan_out)``, scaled by ``gain``."""
    fan_in, fan_out = _fan(shape)
    std = gain * np.sqrt(2.0 / max(1, fan_in + fan_out))
    return parameter(rng.normal(0.0, std, size=shape))


def he(rng: "np.random.Generator", *shape: int, gain: float = 1.0) -> Tensor:
    """Kaiming-normal init — variance ``2/fan_in``, for rectified paths."""
    fan_in, _ = _fan(shape)
    std = gain * np.sqrt(2.0 / max(1, fan_in))
    return parameter(rng.normal(0.0, std, size=shape))


def orthogonal(rng: "np.random.Generator", rows: int, cols: int, *, gain: float = 1.0) -> Tensor:
    """An orthonormal (or semi-orthonormal) matrix via QR — for recurrent transitions.

    The sign correction on ``R``'s diagonal is not cosmetic: without it NumPy's QR returns a
    factorisation whose ``Q`` is orthonormal but not uniformly distributed, which quietly biases
    the spectrum of every recurrence initialised from it.
    """
    flat = rng.normal(0.0, 1.0, size=(max(rows, cols), min(rows, cols)))
    q, r = np.linalg.qr(flat)
    q = q * np.sign(np.diag(r))
    if rows < cols:
        q = q.T
    return parameter(gain * q[:rows, :cols])


def uniform_gate(rng: "np.random.Generator", width: int, *, lo: float = 0.85,
                 hi: float = 0.999) -> Tensor:
    """Decay gates for ``diag_scan``, drawn strictly inside ``(0, 1)``.

    The default band is deliberately near the top of the range: a gate at 0.9 has a memory
    half-life of about seven steps, one at 0.999 about seven hundred. Starting the population
    spread across that band gives the scan both fast and slow channels before it has learned
    anything, which is what lets a world model separate what just happened from what persists.
    """
    lo = float(min(max(lo, 1e-4), 1.0 - 1e-4))
    hi = float(min(max(hi, lo + 1e-4), 1.0 - 1e-6))
    return parameter(rng.uniform(lo, hi, size=(int(width),)))


def zeros(*shape: int) -> Tensor:
    """A learned bias starts at zero — the one place where "no prior" is literally true."""
    return parameter(np.zeros(shape))


def ones(*shape: int) -> Tensor:
    """A learned scale starts at one (layernorm/rmsnorm gain)."""
    return parameter(np.ones(shape))


def count_params(*tensors: Tensor) -> int:
    """How many numbers NYX V.001 actually owns — reported, never estimated."""
    return int(sum(int(t.data.size) for t in tensors if t is not None))
