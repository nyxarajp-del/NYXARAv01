"""NYXARA · njp/graft.py — a trained network's weights, converted into fabric cells (🧬, NJP V.07).

Every other organ in :mod:`nyxara.njp` changes what she knows by *growing* structure. This one
does the opposite and is honest about it: it takes weights somebody else trained and **converts**
them into cells and synapses the fabric can compute with.

**Why this is possible at all.** :class:`~nyxara.njp.cell.Cell` carries a ``threshold`` and a
``bias``; :meth:`~nyxara.njp.fabric.Fabric.connect` carries a weight. Those three numbers are
exactly what ANN→SNN conversion needs (Diehl et al. 2015; Rueckauer et al. 2017): one artificial
neuron becomes one integrate-and-fire cell, one trained weight becomes one synapse, and the cell's
threshold is set so that its **firing rate is proportional to the artificial neuron's activation**.
The substrate was already right. What was missing was the converter and somewhere to put the
numbers, and that is this module.

**The rate code, exactly.** For an integrate-and-fire cell with threshold ``θ`` driven by a
constant current ``I`` and reset by *subtraction* (``V -= θ``, never ``V = 0``)::

    V ← V + I ;  if V ≥ θ:  spike,  V ← V − θ
    ⇒  firing rate over T steps  r ≈ clamp(I / θ, 0, 1)

So with ``I = Wx + b`` and ``θ`` set to the layer's activation scale ``λ``, ``r ≈ ReLU(Wx+b)/λ``.
The spike rate *is* the normalised activation. ``λ`` is picked by **threshold balancing**: the
99.9th percentile of the layer's activations over a calibration set, not the maximum — one
outlier activation would set ``θ`` so high that every other neuron's rate collapses to a few
spikes and the whole layer quantises to noise.

**Three deviations from the fabric's own law, stated here rather than discovered later.**

1. **Grafted cells do not leak and do not go refractory** (``law="if_soft_reset"``, the default).
   The fabric's law is *leaky* integrate-and-fire with a refractory period, and both destroy a
   rate code: the leak discards charge that the code is counting, and the refractory period caps
   the rate at ``1/τ_ref`` regardless of how strong the drive is. Passing ``law="fabric_lif"``
   runs the fabric's real law instead and :func:`verify_region` will report what that costs —
   the option exists so the claim is checkable rather than asserted.
2. **Grafted structure is declared, not grown.** The whole point of the fabric is that structure
   is an outcome. A graft is the opposite, so it is counted separately everywhere
   (:meth:`Grafter.stats`, ``Fabric.stats()["grafted"]``). Folding 5 billion declared synapses
   into the same counter as 40 grown ones would not make the fabric look bigger; it would make
   ``growing`` unfalsifiable, and that counter is one of the few honest things it reports.
3. **Grafted regions are frozen by default** (``plastic=False``). ``Fabric.expand`` depresses and
   prunes synapses that fire without their target following, which on the first turn after a
   graft would delete imported weights wholesale. Freezing is what makes the graft survive; the
   cost is that a frozen region does not learn, and that is the honest trade rather than a bug.

**Nothing is claimed without measurement.** :func:`verify_region` runs the same input through the
original arithmetic and through the converted cells and returns a cosine and a relative L2 error.
:meth:`Grafter.graft_linear` **refuses** a region below ``min_fidelity`` and leaves the fabric
untouched — a partial graft that silently computes something else is worse than no graft, because
every later measurement would inherit the error without knowing it.

Pure numpy. The GGUF reader is optional: without the ``gguf`` package the converter still works
on arrays handed to it directly, which is also how the test suite exercises it without a 5.6 GB
download.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as _np

__all__ = [
    "QuantizedWeights", "GraftedRegion", "Fidelity", "GraftCertificate",
    "Grafter", "balance_threshold", "verify_region",
]

#: Default activation percentile for threshold balancing. Rueckauer et al. report 99.9 as the
#: sweet spot: high enough that almost nothing saturates, low enough that a single outlier
#: activation cannot set the scale for the whole layer.
DEFAULT_PERCENTILE = 99.9

#: Default simulation length for the rate code. The quantisation error of a rate code is O(1/T),
#: so this is a direct accuracy/latency dial and :func:`verify_region` measures where it lands.
DEFAULT_TIMESTEPS = 64

#: A region below this cosine against the original arithmetic is rejected outright.
DEFAULT_MIN_FIDELITY = 0.95


# --------------------------------------------------------------------------- #
# Quantised weight storage — the reason a 9B graft fits in RAM at all
# --------------------------------------------------------------------------- #
#: Weights are quantised in groups of this many consecutive fan-in entries, each with its own
#: scale — the same thing GGUF's own K-quants do, and for the same reason.
#:
#: Measured on a 256x512 layer at T=64, end-to-end cosine against the original arithmetic:
#:
#:     bits=8, group=128 -> 0.99974,  1.031 B/param  (5.4B params ~ 5.6 GB)
#:     bits=8, group=32  -> 0.99975,  1.125 B/param  (5.4B params ~ 6.1 GB)
#:     bits=4, group=32  -> 0.99456,  0.625 B/param  (5.4B params ~ 3.4 GB)
#:
#: Worth stating plainly because the per-weight numbers look much worse than the end-to-end ones:
#: 4-bit symmetric quantisation carries ~11% relative error *per weight* at any group size — the
#: resolution of 15 levels is what it is, and grouping barely moves it. What grouping and the rate
#: code together buy is that those errors are independent across a wide fan-in, so they largely
#: cancel in the sum the neuron actually computes. Which is exactly why the fidelity gate measures
#: the layer's OUTPUT rather than its weights: the weight error alone would have rejected a
#: representation that works.
DEFAULT_GROUP = 32


@dataclass
class QuantizedWeights:
    """A dense weight matrix kept small enough to actually hold.

    ``Fabric.out`` is ``Dict[int, Dict[int, float]]``, which is the right structure for growth
    (inserting one synapse is O(1)) and the wrong one for a graft: a Python dict entry plus a
    float object runs about 100 bytes per synapse, so the fabric's own 4M ceiling is already
    ~400 MB and a 5.4B-parameter graft would be **~540 GB**. Nothing about that is fixable by
    tuning; it needs a different representation.

    Here one weight is one ``int8`` — or half a byte at ``bits=4`` — plus one ``float32`` scale
    per **group** of ``group`` consecutive fan-in entries. At ``bits=4, group=32`` that is
    ``0.5 + 4/32 = 0.625`` bytes per parameter, which puts a 5.4B-parameter graft at ~3.4 GB.

    Quantisation is symmetric and grouped along the fan-in because a row is one neuron's input
    and its weights are not uniformly scaled across it. A single scale for the whole row lets one
    large weight crush the resolution of every other input to the same neuron; a scale per 32
    entries costs an eighth of a byte per weight and keeps them all.
    """

    q: Any                      # int8 [n_post, n_pre_padded]  (packed to half-width at bits=4)
    scale: Any                  # float32 [n_post, n_groups] — symmetric, per group
    n_pre: int
    n_post: int
    bits: int = 8
    group: int = DEFAULT_GROUP

    @property
    def n_params(self) -> int:
        return int(self.n_pre) * int(self.n_post)

    @property
    def n_groups(self) -> int:
        return int(math.ceil(self.n_pre / self.group))

    @property
    def padded(self) -> int:
        return self.n_groups * self.group

    @property
    def nbytes(self) -> int:
        return int(self.q.nbytes) + int(self.scale.nbytes)

    @property
    def bytes_per_param(self) -> float:
        n = self.n_params
        return round(self.nbytes / n, 4) if n else 0.0

    @classmethod
    def from_dense(cls, w: Any, *, bits: int = 8,
                   group: int = DEFAULT_GROUP) -> "QuantizedWeights":
        """Quantise ``w`` of shape ``[n_post, n_pre]``. Symmetric, grouped along fan-in."""
        w = _np.asarray(w, dtype=_np.float32)
        if w.ndim != 2:
            raise ValueError(f"weights must be 2-D [n_post, n_pre], got shape {w.shape}")
        if bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {bits}")
        n_post, n_pre = w.shape
        group = max(1, int(group))
        n_groups = int(math.ceil(n_pre / group))
        pad = n_groups * group - n_pre
        if pad:
            # Zero-padding the tail group is exact: a zero weight contributes nothing to the dot
            # product, so the padded columns cannot change the result.
            w = _np.concatenate([w, _np.zeros((n_post, pad), dtype=_np.float32)], axis=1)
        blocks = w.reshape(n_post, n_groups, group)
        qmax = 7.0 if bits == 4 else 127.0
        peak = _np.max(_np.abs(blocks), axis=2)
        # An all-zero group has no scale; leaving it at 0 would divide by zero on dequantise.
        scale = _np.where(peak > 0.0, peak / qmax, 1.0).astype(_np.float32)
        q = _np.rint(blocks / scale[:, :, None]).clip(-qmax, qmax).astype(_np.int8)
        q = q.reshape(n_post, n_groups * group)
        if bits == 4:
            q = _pack_int4(q)
        return cls(q=q, scale=scale, n_pre=int(n_pre), n_post=int(n_post),
                   bits=int(bits), group=group)

    def _int_blocks(self) -> Any:
        """The integer weights as ``[n_post, n_groups, group]``, unpacked if they were 4-bit."""
        q = _unpack_int4(self.q, self.padded) if self.bits == 4 else self.q
        return q.reshape(self.n_post, self.n_groups, self.group)

    def dequantize(self) -> Any:
        """The full float32 matrix back. Allocates ``n_params * 4`` bytes — callers that only
        need a product should use :meth:`matvec`, which never materialises this."""
        blocks = self._int_blocks().astype(_np.float32) * self.scale[:, :, None]
        return blocks.reshape(self.n_post, self.padded)[:, :self.n_pre]

    def matvec(self, x: Any) -> Any:
        """``W @ x`` without materialising ``W`` in float.

        Each group's integer dot product is exact and its scale is applied once, so the only
        error in the result is the quantisation of the weights themselves rather than rounding
        accumulated across the length of the row.
        """
        x = _np.asarray(x, dtype=_np.float32).ravel()
        if x.size != self.n_pre:
            raise ValueError(f"input has {x.size} entries, layer expects {self.n_pre}")
        if self.padded != self.n_pre:
            x = _np.concatenate([x, _np.zeros(self.padded - self.n_pre, dtype=_np.float32)])
        xg = x.reshape(self.n_groups, self.group)
        partial = _np.einsum("pgk,gk->pg", self._int_blocks().astype(_np.float32), xg)
        return (partial * self.scale).sum(axis=1)

    def to_dict(self) -> Dict[str, Any]:
        return {"n_pre": self.n_pre, "n_post": self.n_post, "bits": self.bits,
                "group": self.group, "n_params": self.n_params, "bytes": self.nbytes,
                "bytes_per_param": self.bytes_per_param}


def _pack_int4(q: Any) -> Any:
    """Two int4 values per byte, low nibble first. ``q`` must already be clipped to [-7, 7]."""
    q = _np.asarray(q, dtype=_np.int8)
    n_post, n_pre = q.shape
    if n_pre % 2:                                   # odd fan-in: pad the tail with a zero
        q = _np.concatenate([q, _np.zeros((n_post, 1), dtype=_np.int8)], axis=1)
    lo = (q[:, 0::2] & 0x0F).astype(_np.uint8)
    hi = (q[:, 1::2] & 0x0F).astype(_np.uint8)
    return ((hi << 4) | lo).view(_np.int8)


def _unpack_int4(packed: Any, n_pre: int) -> Any:
    """Inverse of :func:`_pack_int4`. Sign-extends each nibble back to int8."""
    b = _np.asarray(packed).view(_np.uint8)
    lo = (b & 0x0F).astype(_np.int8)
    hi = ((b >> 4) & 0x0F).astype(_np.int8)
    lo = _np.where(lo > 7, lo - 16, lo).astype(_np.int8)
    hi = _np.where(hi > 7, hi - 16, hi).astype(_np.int8)
    out = _np.empty((b.shape[0], b.shape[1] * 2), dtype=_np.int8)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return out[:, :n_pre]


# --------------------------------------------------------------------------- #
# Threshold balancing
# --------------------------------------------------------------------------- #
def balance_threshold(activations: Any, *, percentile: float = DEFAULT_PERCENTILE) -> float:
    """The firing threshold ``λ`` for a layer, from what it actually activates to.

    The percentile rather than the maximum, and this is the whole reason the technique works.
    ``θ = max`` is defensible in a proof and catastrophic in practice: activations are heavy-
    tailed, so one outlier sets ``θ`` an order of magnitude above the bulk, every other neuron
    lands at ``I/θ ≈ 0.01``, and over ``T`` steps almost all of them emit zero spikes. The layer
    then "works" — it just carries no information. Clipping the top 0.1% costs a small saturation
    error on a few neurons and keeps the resolution of all the rest.
    """
    a = _np.asarray(activations, dtype=_np.float32).ravel()
    a = a[_np.isfinite(a)]
    if a.size == 0:
        return 1.0
    lam = float(_np.percentile(_np.abs(a), float(percentile)))
    # A layer that never activates would give θ=0 and divide by zero on every drive.
    return lam if lam > 1e-8 else 1.0


# --------------------------------------------------------------------------- #
# One grafted region
# --------------------------------------------------------------------------- #
@dataclass
class GraftedRegion:
    """One converted layer: a block of cells, the weights between them, and how it computes.

    Cell ids are a contiguous range so the region can be addressed as a slice rather than as a
    set — a graft is dense by construction, which is exactly the case the fabric's sparse dicts
    are bad at and a range is good at.
    """

    name: str
    pre_lo: int                       # presynaptic cell id range [pre_lo, pre_lo + n_pre)
    post_lo: int                      # postsynaptic cell id range [post_lo, post_lo + n_post)
    weights: QuantizedWeights
    bias: Any                         # float32 [n_post]
    threshold: float                  # λ from balance_threshold
    law: str = "if_soft_reset"        # or "fabric_lif" — see the module docstring
    #: The membrane decay this region runs under when ``law="fabric_lif"``, taken from the fabric
    #: it was grafted into. It lives on the region rather than being passed per call because
    #: otherwise the alternative law is INERT during verification: `verify_region` would simulate
    #: it at the default `decay=1.0`, both laws would score identically, and the option that
    #: exists to make the fabric's leak measurable would measure nothing — while reporting a
    #: number that looks like a measurement. That is the failure `eval/ablation.py` calls out by
    #: name, and it is just as easy to write here.
    decay: float = 1.0
    plastic: bool = False             # frozen by default; expand() skips frozen regions
    fidelity: float = float("nan")    # measured, never assumed
    timesteps: int = DEFAULT_TIMESTEPS
    source: str = ""                  # which tensor this came from

    @property
    def n_pre(self) -> int:
        return self.weights.n_pre

    @property
    def n_post(self) -> int:
        return self.weights.n_post

    @property
    def n_params(self) -> int:
        return self.weights.n_params

    @property
    def nbytes(self) -> int:
        return self.weights.nbytes + int(_np.asarray(self.bias).nbytes)

    def owns_cell(self, cell_id: int) -> bool:
        """True when ``cell_id`` belongs to this region — used by ``expand`` to leave it alone."""
        return (self.pre_lo <= cell_id < self.pre_lo + self.n_pre
                or self.post_lo <= cell_id < self.post_lo + self.n_post)

    def current(self, pre_rates: Any) -> Any:
        """The input current onto every post cell: ``W @ r + b``. One matvec, no float matrix."""
        return self.weights.matvec(pre_rates) + _np.asarray(self.bias, dtype=_np.float32)

    def forward(self, pre_rates: Any, *, timesteps: Optional[int] = None,
                decay: Optional[float] = None, tau_ref_steps: int = 0) -> Any:
        """Simulate the region and return each post cell's firing **rate** in ``[0, 1]``.

        ``pre_rates`` are normalised activations in ``[0, 1]`` — the previous layer's rates, or an
        analog-encoded input. Analog input rather than Poisson sampling on purpose: Poisson
        encoding adds sampling noise of order ``1/√T`` on top of the ``1/T`` quantisation the rate
        code already has, which is a large accuracy loss bought for nothing when the input is
        available as a number.

        ``decay``/``tau_ref_steps`` are ignored under ``law="if_soft_reset"`` and applied under
        ``law="fabric_lif"``, which is what makes the cost of the fabric's own law measurable
        instead of arguable. ``decay`` defaults to the region's own :attr:`decay` rather than to
        ``1.0``, so a caller that does not pass one — :func:`verify_region` is the important one —
        still simulates the law the region actually runs under.
        """
        T = int(timesteps if timesteps is not None else self.timesteps)
        T = max(1, T)
        decay = float(self.decay if decay is None else decay)
        I = self.current(pre_rates)
        theta = float(self.threshold)
        v = _np.zeros(self.n_post, dtype=_np.float32)
        spikes = _np.zeros(self.n_post, dtype=_np.int32)
        leaky = (self.law == "fabric_lif")
        refr = _np.zeros(self.n_post, dtype=_np.int32) if tau_ref_steps > 0 else None
        for _t in range(T):
            if leaky:
                v *= decay
            v += I
            fired = v >= theta
            if refr is not None:
                fired &= (refr <= 0)
                refr = _np.maximum(refr - 1, 0)
                refr = _np.where(fired, int(tau_ref_steps), refr)
            if fired.any():
                spikes += fired.astype(_np.int32)
                # Reset by SUBTRACTION, not to zero. A hard reset throws away the charge above
                # threshold, and that residue is precisely the fractional part the rate code is
                # trying to represent — hard reset is the single commonest reason a converted
                # network loses accuracy.
                v = _np.where(fired, v - theta, v)
            # A cell driven negative would otherwise accumulate an arbitrarily deep debt and stay
            # silent long after its input turned positive. ReLU has no such memory.
            v = _np.maximum(v, -theta)
        return spikes.astype(_np.float32) / float(T)

    def reference(self, pre_rates: Any) -> Any:
        """This region's **analog** output — its own arithmetic, without the spiking simulation.

        Read the scope carefully, because it is easy to mistake for a stronger check than it is.
        This uses the region's *quantised* weights, so comparing :meth:`forward` against it
        isolates exactly one error term: what the **rate code** costs. It says nothing about what
        quantisation cost, because both sides carry the same quantised weights.

        The full check — quantisation *and* rate code, measured against the arithmetic the graft
        was supposed to reproduce — needs the original float weights, which the region no longer
        holds. :meth:`Grafter.graft_linear` therefore builds that reference from ``w`` before
        quantising and passes it to :func:`verify_region` explicitly, and that is the number the
        fidelity gate and the certificate report. Use this one to attribute an error, not to
        accept a region.
        """
        I = self.current(pre_rates)
        return _np.clip(I / float(self.threshold), 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "source": self.source,
                "pre": [self.pre_lo, self.pre_lo + self.n_pre],
                "post": [self.post_lo, self.post_lo + self.n_post],
                "n_params": self.n_params, "bytes": self.nbytes,
                "threshold": round(float(self.threshold), 6),
                "law": self.law, "plastic": self.plastic,
                "decay": round(float(self.decay), 6),
                "timesteps": self.timesteps,
                "fidelity": (None if _np.isnan(self.fidelity) else round(float(self.fidelity), 6)),
                "weights": self.weights.to_dict()}


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
@dataclass
class Fidelity:
    """How close the converted cells came to the arithmetic they replaced. Measured, per region."""

    cosine: float = 0.0
    rel_l2: float = float("inf")
    samples: int = 0
    timesteps: int = 0
    ms: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(_np.isfinite(self.cosine))

    def to_dict(self) -> Dict[str, Any]:
        return {"cosine": round(float(self.cosine), 6),
                "rel_l2": (None if not _np.isfinite(self.rel_l2) else round(float(self.rel_l2), 6)),
                "samples": self.samples, "timesteps": self.timesteps,
                "ms": round(self.ms, 3)}


def verify_region(region: GraftedRegion, samples: Sequence[Any], *,
                  timesteps: Optional[int] = None,
                  reference: Optional[Callable[[Any], Any]] = None) -> Fidelity:
    """Run the same inputs through the original arithmetic and the converted cells.

    Returns a cosine and a relative L2 error over the concatenated outputs. Concatenated rather
    than averaged per-sample because a mean of per-sample cosines hides the case this is built to
    catch: a region that is excellent on most inputs and produces garbage on a few.

    ``reference`` defaults to :meth:`GraftedRegion.reference`, which is the region's own analog
    output and therefore measures the **rate code alone**. That default is for attribution. To
    accept a region, pass the original layer's float arithmetic — which is what
    :meth:`Grafter.graft_linear` does, so the gate sees quantisation error too rather than
    cancelling it out of both sides.
    """
    t0 = time.perf_counter()
    out = Fidelity(samples=len(samples),
                   timesteps=int(timesteps if timesteps is not None else region.timesteps))
    if not samples:
        return out
    ref_fn = reference if reference is not None else region.reference
    got: List[Any] = []
    want: List[Any] = []
    for x in samples:
        x = _np.asarray(x, dtype=_np.float32)
        got.append(_np.asarray(region.forward(x, timesteps=out.timesteps), dtype=_np.float64))
        want.append(_np.asarray(ref_fn(x), dtype=_np.float64))
    a = _np.concatenate(got).ravel()
    b = _np.concatenate(want).ravel()
    na, nb = float(_np.linalg.norm(a)), float(_np.linalg.norm(b))
    if na > 0.0 and nb > 0.0:
        out.cosine = float(_np.dot(a, b) / (na * nb))
    elif na == 0.0 and nb == 0.0:
        # Both silent is agreement, not failure — a layer that should emit nothing and emits
        # nothing has converted correctly, and scoring it 0.0 would reject a working region.
        out.cosine = 1.0
    else:
        out.cosine = 0.0
    out.rel_l2 = float(_np.linalg.norm(a - b) / nb) if nb > 0.0 else float("inf")
    out.ms = (time.perf_counter() - t0) * 1000.0
    return out


@dataclass
class GraftCertificate:
    """What was imported, how it was measured, and what was refused.

    Modelled on :mod:`nyxara.njp.forge`, which keeps a certificate for every C kernel it swaps in
    and records the domain the equivalence was established over. The same discipline applies here
    and for the same reason: an unverified import is a claim, and this package does not store
    claims as facts. Rejections are recorded as carefully as acceptances — a certificate that only
    lists successes is a brochure.
    """

    regions: List[Dict[str, Any]] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    params_grafted: int = 0
    params_available: int = 0
    cells_minted: int = 0
    bytes_resident: int = 0
    percentile: float = DEFAULT_PERCENTILE
    min_fidelity: float = DEFAULT_MIN_FIDELITY
    timesteps: int = DEFAULT_TIMESTEPS
    started: float = field(default_factory=time.time)

    @property
    def coverage(self) -> float:
        """Fraction of the available parameters that actually landed. The headline number, and
        the one a summary is most tempted to round up."""
        return (self.params_grafted / self.params_available) if self.params_available else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "params_grafted": self.params_grafted,
            "params_available": self.params_available,
            "coverage": round(self.coverage, 6),
            "cells_minted": self.cells_minted,
            "bytes_resident": self.bytes_resident,
            "gb_resident": round(self.bytes_resident / 1e9, 4),
            "regions": self.regions,
            "rejected": self.rejected,
            "n_regions": len(self.regions), "n_rejected": len(self.rejected),
            "percentile": self.percentile, "min_fidelity": self.min_fidelity,
            "timesteps": self.timesteps,
            "started": self.started,
        }


# --------------------------------------------------------------------------- #
# The grafter
# --------------------------------------------------------------------------- #
#: Grafted cell ids start here. Grown ids come from ``Fabric.next_id``, which counts up from 1 and
#: is driven by neurogenesis and by token hashing, so anything below this base is hers. Keeping
#: the two ranges disjoint by construction means a graft can never silently overwrite a cell she
#: grew, and ``owns_cell`` stays a range check instead of a set membership test.
GRAFT_ID_BASE = 1 << 40


class Grafter:
    """Converts trained layers into fabric cells, verifies each one, and refuses the bad ones.

    The refusal is the important half. A converter that always succeeds is a converter that has
    stopped measuring: ANN→SNN conversion has real failure modes — a layer whose activations are
    so heavy-tailed that no percentile works, a fan-in large enough that int4 quantisation loses
    the signal, a rate code too short for the dynamic range the layer needs — and each of them
    produces a region that computes *something*, plausibly, wrongly. So every region is scored
    against the arithmetic it replaces before it is attached, and one that does not clear
    ``min_fidelity`` is rolled back to nothing: the cells it minted are removed and the fabric is
    exactly as it was.
    """

    def __init__(self, fabric: Any, *, percentile: float = DEFAULT_PERCENTILE,
                 timesteps: int = DEFAULT_TIMESTEPS,
                 min_fidelity: float = DEFAULT_MIN_FIDELITY,
                 bits: int = 8, group: int = DEFAULT_GROUP,
                 law: str = "if_soft_reset",
                 id_base: int = GRAFT_ID_BASE) -> None:
        self.fabric = fabric
        self.percentile = float(percentile)
        self.timesteps = max(1, int(timesteps))
        self.min_fidelity = float(min_fidelity)
        self.bits = int(bits)
        self.group = int(group)
        self.law = str(law)
        self._next_graft_id = int(id_base)
        self.cert = GraftCertificate(percentile=self.percentile,
                                     min_fidelity=self.min_fidelity,
                                     timesteps=self.timesteps)

    # ---- cell allocation ------------------------------------------------- #
    def _alloc(self, n: int) -> int:
        lo = self._next_graft_id
        self._next_graft_id += int(n)
        return lo

    def _mint(self, lo: int, n: int, *, threshold: float,
              bias: Optional[Any] = None) -> int:
        """Mint ``n`` cells at ``[lo, lo+n)`` with a converted threshold and bias.

        ``Fabric.ensure`` pushes ``next_id`` past any id it is handed, so minting at 2^40 would
        drag the fabric's own counter up there and every cell she grows afterwards would be
        allocated out of the graft range. The counter is therefore saved and restored around the
        mint: grown ids keep counting from where they were, and the two ranges stay disjoint.
        """
        fab = self.fabric
        saved_next = getattr(fab, "next_id", 1)
        saved_born = getattr(fab, "born_total", 0)
        made = 0
        for k in range(int(n)):
            cell = fab.ensure(lo + k)
            cell.threshold = float(threshold)
            if bias is not None:
                cell.bias = float(_np.asarray(bias).ravel()[k])
            made += 1
        fab.next_id = saved_next
        # Grafted cells are not born, they are placed. Letting them into ``born_total`` would
        # inflate the one counter that answers "how much has she grown by herself".
        fab.born_total = saved_born
        return made

    def _unmint(self, lo: int, n: int) -> None:
        """Undo a mint — used when a region fails verification and must leave no trace."""
        fab = self.fabric
        for k in range(int(n)):
            cid = lo + k
            fab.cells.pop(cid, None)
            fab.out.pop(cid, None)
            fab.inn.pop(cid, None)
        fab._dirty = True

    # ---- the conversion -------------------------------------------------- #
    def graft_linear(self, name: str, w: Any, b: Optional[Any] = None, *,
                     calibration: Optional[Sequence[Any]] = None,
                     verify_samples: Optional[Sequence[Any]] = None,
                     source: str = "", pre_lo: Optional[int] = None,
                     plastic: bool = False) -> Optional[GraftedRegion]:
        """Convert one ``ReLU(Wx + b)`` layer into cells, verify it, and attach it — or refuse.

        ``calibration`` sets the threshold and ``verify_samples`` scores the result. They default
        to each other and then to a random battery, but a caller with real activations should pass
        them: a threshold balanced on noise is balanced on nothing, and this is the one number the
        whole conversion rests on.
        """
        w = _np.asarray(w, dtype=_np.float32)
        if w.ndim != 2:
            raise ValueError(f"graft_linear expects [n_post, n_pre], got {w.shape}")
        n_post, n_pre = w.shape
        bias = (_np.zeros(n_post, dtype=_np.float32) if b is None
                else _np.asarray(b, dtype=_np.float32).ravel())
        if bias.size != n_post:
            raise ValueError(f"bias has {bias.size} entries, layer has {n_post} outputs")

        self.cert.params_available += int(n_pre) * int(n_post)

        cal = list(calibration) if calibration is not None else None
        if cal is None:
            cal = list(verify_samples) if verify_samples is not None else _default_probe(n_pre)
        acts = _np.concatenate([_np.maximum(w @ _np.asarray(x, dtype=_np.float32) + bias, 0.0)
                                for x in cal]) if cal else _np.asarray([1.0], dtype=_np.float32)
        theta = balance_threshold(acts, percentile=self.percentile)

        qw = QuantizedWeights.from_dense(w, bits=self.bits, group=self.group)
        post_lo = self._alloc(n_post)
        pre_base = int(pre_lo) if pre_lo is not None else self._alloc(n_pre)

        region = GraftedRegion(
            name=str(name), pre_lo=pre_base, post_lo=post_lo, weights=qw, bias=bias,
            threshold=theta, law=self.law, plastic=bool(plastic),
            decay=float(getattr(self.fabric, "decay", 1.0)),
            timesteps=self.timesteps, source=str(source or name))

        # The float reference — the arithmetic the graft is supposed to reproduce. Deliberately
        # built from the ORIGINAL ``w``, not from the quantised copy, so quantisation error is
        # inside what is being measured rather than cancelled out of it.
        def _ref(x: Any) -> Any:
            return _np.clip((w @ _np.asarray(x, dtype=_np.float32) + bias) / theta, 0.0, 1.0)

        probe = list(verify_samples) if verify_samples is not None else cal
        fid = verify_region(region, probe, timesteps=self.timesteps, reference=_ref)
        region.fidelity = fid.cosine

        entry = {**region.to_dict(), "fidelity_detail": fid.to_dict()}
        if fid.cosine < self.min_fidelity:
            entry["reason"] = (f"fidelity {fid.cosine:.4f} < floor {self.min_fidelity:.4f} "
                               f"at T={fid.timesteps}")
            self.cert.rejected.append(entry)
            return None

        minted = self._mint(post_lo, n_post, threshold=theta, bias=bias)
        if pre_lo is None:
            minted += self._mint(pre_base, n_pre, threshold=theta)

        # `attach_graft` refreshes the fabric's frozen-id index, which is what keeps plasticity
        # and apoptosis off the region. Appending to the list without it would attach a region
        # the freeze does not know about — wired, computing, and quietly prunable.
        self.fabric.attach_graft(region)

        self.cert.regions.append(entry)
        self.cert.params_grafted += region.n_params
        self.cert.cells_minted += minted
        self.cert.bytes_resident += region.nbytes
        return region

    # ---- reporting ------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        return self.cert.to_dict()


def _default_probe(n_pre: int, *, n: int = 24, seed: int = 0) -> List[Any]:
    """A stand-in calibration battery: uniform rates in [0, 1], the range a rate code carries.

    Only used when the caller supplies nothing. It is enough to keep the converter honest on
    synthetic tests and it is *not* enough for a real model — which is why the threshold that
    came out of it is recorded in the certificate rather than left implicit.
    """
    rng = _np.random.default_rng(seed)
    return [rng.random(n_pre).astype(_np.float32) for _ in range(n)]


# --------------------------------------------------------------------------- #
# Reading a GGUF file — optional, and absent is a normal answer
# --------------------------------------------------------------------------- #
#: The tensor families a graft can and cannot take, and why. Kept as data rather than as prose in
#: a commit message, because the coverage number the certificate reports is derived from it and
#: should be traceable to a reason.
#: Order matters: the first entry whose key appears in the tensor name wins, so the longer, more
#: specific names come first. Measured against the real file this is not pedantry — ``attn_qkv``
#: contains ``attn_q`` as a substring, so an unordered table classified a *fused* projection as a
#: separate one. It happened to land on the right verdict and for entirely the wrong reason, which
#: is the kind of accident that stops being harmless the moment the verdicts differ.
TENSOR_POLICY: Tuple[Tuple[str, str], ...] = (
    # --- the conversion targets: SwiGLU's gate/up/down are the ReLU-family layers ------------- #
    ("ffn_gate", "graft"),
    ("ffn_up", "graft"),
    ("ffn_down", "graft"),
    # --- attention: softmax is not a rate-coded nonlinearity; see the module docstring --------- #
    ("attn_qkv", "hybrid"),      # this model fuses q/k/v in 24 of its 32 blocks
    ("attn_gate", "hybrid"),
    ("attn_output", "hybrid"),
    ("attn_q", "hybrid"),
    ("attn_k", "hybrid"),
    ("attn_v", "hybrid"),
    # --- state space: this is a HYBRID transformer/SSM, not a pure transformer ----------------- #
    # 24 of its blocks carry ssm_* tensors (gated-deltanet). A recurrent state update is not a
    # feedforward activation, so rate-coded conversion does not apply to it — and it is counted
    # rather than skipped, because 400M+ parameters silently outside the denominator would inflate
    # the coverage figure this module publishes.
    ("ssm_out", "hybrid"),
    ("ssm_in", "hybrid"),
    ("ssm_conv1d", "hybrid"),
    ("ssm_norm", "scale"),
    ("ssm_dt", "scale"),
    ("ssm_a", "scale"),
    ("ssm_alpha", "scale"),
    ("ssm_beta", "scale"),
    # --- tables, not layers: tokens drive cells directly (brain._cell_id) ---------------------- #
    ("token_embd", "lookup"),
    ("output_norm", "scale"),    # before "output", which would otherwise swallow it
    ("output", "lookup"),
    # --- norms: applied around a region, never converted into one ----------------------------- #
    ("attn_norm", "scale"),
    ("ffn_norm", "scale"),
    ("post_attention_norm", "scale"),
)


def classify_tensor(name: str) -> str:
    """``graft`` | ``hybrid`` | ``lookup`` | ``scale`` | ``skip`` for one GGUF tensor name."""
    for key, verdict in TENSOR_POLICY:
        if key in name:
            return verdict
    return "skip"


def read_gguf_tensors(path: Any, *, only: Optional[str] = None) -> List[Tuple[str, Any]]:
    """``[(name, array)]`` from a GGUF file, or an empty list when it cannot be read.

    Requires the optional ``gguf`` package (llama.cpp's own reader). Its absence is unavailability,
    not an error — the converter's whole numeric path is exercised by the test suite on arrays
    handed to it directly, so nothing here needs a 5.6 GB download to be known to work.
    """
    try:
        from gguf import GGUFReader  # type: ignore
    except Exception:  # noqa: BLE001 — not installed is the answer, not a failure
        return []
    try:
        reader = GGUFReader(str(path))
    except Exception:  # noqa: BLE001 — an unreadable file is unavailability too
        return []
    out: List[Tuple[str, Any]] = []
    for tensor in getattr(reader, "tensors", []):
        name = str(getattr(tensor, "name", ""))
        if only and classify_tensor(name) != only:
            continue
        try:
            out.append((name, dequantize_tensor(tensor)))
        except Exception:  # noqa: BLE001 — one unreadable tensor must not lose the rest
            continue
    return out


def dequantize_tensor(tensor: Any) -> Any:
    """One GGUF tensor as float32 weights in its **logical** shape.

    This is not a convenience wrapper. ``GGUFReader`` hands back ``tensor.data`` as the *packed
    quantisation bytes*: for a Q4_K tensor of logical shape ``(4096, 12288)`` that is a
    ``(12288, 2304)`` array of ``uint8`` in ``[0, 255]``, where the bytes are interleaved 4-bit
    values, block scales and block minima. Calling ``.astype(float32)`` on it produces a matrix of
    plausible-looking numbers that are not weights at all.

    That failure is silent all the way through. The converter would balance a threshold on it,
    quantise it, simulate it, and verify it against *itself* — every stage internally consistent —
    and issue a fidelity certificate for a layer computing nothing. It is exactly the case this
    module's docstring warns about, and the only defence is to never hand it raw bytes.
    """
    from gguf import quants  # local: the reader is optional and so is this

    data = getattr(tensor, "data", None)
    qtype = getattr(tensor, "tensor_type", None)
    arr = quants.dequantize(data, qtype) if qtype is not None else _np.asarray(data)
    arr = _np.asarray(arr, dtype=_np.float32)
    if arr.dtype != _np.float32:                      # pragma: no cover — belt and braces
        raise ValueError(f"{getattr(tensor, 'name', '?')} did not dequantise to float32")
    return arr


# --------------------------------------------------------------------------- #
# CLI — graft a real model, and report what actually landed
# --------------------------------------------------------------------------- #
def graft_gguf(fabric: Any, path: Any, *, layers: str = "", settings: Any = None,
               calibration: Optional[Sequence[Any]] = None,
               progress: Optional[Callable[[str], None]] = None) -> GraftCertificate:
    """Convert every graftable tensor in a GGUF into fabric cells. Returns the certificate.

    ``layers`` is a range like ``"0-35"`` over ``blk.N``; empty means all of them. Tensors the
    policy classes as ``hybrid``/``lookup``/``scale`` are counted toward ``params_available`` and
    not grafted, so the coverage figure is a statement about the whole model rather than about the
    subset that happened to be attempted.
    """
    def say(msg: str) -> None:
        if progress is not None:
            progress(msg)

    lo, hi = -1, 1 << 30
    if layers:
        try:
            parts = str(layers).split("-")
            lo, hi = int(parts[0]), int(parts[-1])
        except Exception:  # noqa: BLE001
            lo, hi = -1, 1 << 30

    cfg = getattr(settings, "llm", None)
    grafter = Grafter(
        fabric,
        percentile=float(getattr(cfg, "graft_percentile", DEFAULT_PERCENTILE)),
        timesteps=int(getattr(cfg, "graft_timesteps", DEFAULT_TIMESTEPS)),
        min_fidelity=float(getattr(cfg, "graft_min_fidelity", DEFAULT_MIN_FIDELITY)),
        bits=int(getattr(cfg, "graft_bits", 8)),
        group=int(getattr(cfg, "graft_group", DEFAULT_GROUP)),
    )
    # Plasticity is a per-region property, not a grafter-wide one, so it is passed at graft time.
    plastic = bool(getattr(cfg, "graft_plastic", False))

    tensors = read_gguf_tensors(path)
    if not tensors:
        say(f"no tensors readable from {path} — is the `gguf` package installed?")
        return grafter.cert

    for name, array in tensors:
        verdict = classify_tensor(name)
        m = re.search(r"blk\.(\d+)\.", name)
        if m and not (lo <= int(m.group(1)) <= hi):
            continue
        if verdict != "graft":
            # Counted, not converted. Reporting coverage only over what was attempted would turn
            # "60% of the model" into "100% of the 60% we tried", which is the same number
            # dressed as a much better one.
            if verdict in ("hybrid", "lookup"):
                grafter.cert.params_available += int(_np.asarray(array).size)
            continue
        arr = _np.asarray(array)
        if arr.ndim != 2:
            continue
        say(f"grafting {name} {arr.shape} …")
        region = grafter.graft_linear(name, arr.astype(_np.float32), None,
                                      calibration=calibration, source=name, plastic=plastic)
        say(f"  {'kept' if region else 'REJECTED'} "
            f"{'fidelity %.5f' % region.fidelity if region else ''}")
    return grafter.cert


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Graft Qwythos weights into NJP's fabric.")
    ap.add_argument("--layers", default="", help="blk range, e.g. 0-35 (default: all)")
    ap.add_argument("--model", default=None, help="path to the .gguf (default: from config)")
    ap.add_argument("--verify", action="store_true", help="print the fidelity certificate")
    args = ap.parse_args(argv)

    from nyxara.kernel.config import get_settings
    from nyxara.njp.fabric import Fabric

    settings = get_settings()
    path = args.model
    if path is None:
        from nyxara.mind.gguf_assets import model_path
        path = model_path(settings)

    fabric = Fabric(seed=0)
    cert = graft_gguf(fabric, path, layers=args.layers, settings=settings,
                      progress=lambda m: print(m, flush=True))
    report = cert.to_dict()
    print()
    print(f"grafted   : {report['params_grafted']:,} / {report['params_available']:,} params "
          f"({report['coverage'] * 100:.1f}%)")
    print(f"cells     : {report['cells_minted']:,}")
    print(f"resident  : {report['gb_resident']:.2f} GB")
    print(f"regions   : {report['n_regions']} kept, {report['n_rejected']} rejected")
    if report["n_rejected"]:
        print("rejected  :")
        for entry in report["rejected"][:10]:
            print(f"    {entry['name']}: {entry.get('reason', '')}")
    if args.verify:
        print()
        print(json.dumps(report, indent=2)[:4000])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
