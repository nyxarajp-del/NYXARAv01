"""NYXARA · nyx001/lingua/sequence.py — a sequence model that is not a Transformer (📜, SSM).

Track B's second requirement, and one of the charter's explicit prohibitions: *"Copied Transformer
recipe"* is on the forbidden list. This module is the alternative, and the difference is mechanical
rather than cosmetic.

A Transformer layer computes, for every position, a weighted read over *every other position*:
attention is an all-pairs operation, O(T²) in time and memory, and it carries no state between
calls — the whole history must be re-read on each forward pass.

This is a **selective state-space scan**:

    h_t = a ⊙ h_{t-1} + (W_in · z_t) ⊙ σ(W_sel · z_t)
    y_t = W_out · (GELU(W_hid · h_t) ⊙ z_t) + z_t

Three properties follow, and each is a real consequence of the choice:

* **Linear in sequence length.** One pass, no T×T matrix. A long context costs proportionally, not
  quadratically.
* **It carries state.** ``h`` persists across calls, so the model has a *belief about what is going
  on* rather than a re-derivation from scratch. That is the same object Layer 2's world model
  carries, which is why the two compose.
* **It is selective.** ``σ(W_sel · z_t)`` gates what enters the state per-token — the model learns
  what is worth remembering rather than attending to everything and weighting afterwards. The
  decay ``a`` is a learned logit through a sigmoid, so it stays in ``(0,1)`` and can learn to shut.

That last detail is the one that was got wrong once already in this package: in Layer 2 the decay
gate was originally passed detached, its gradient was identically zero, and the recurrence was
decorative. Here the gate is built the same corrected way, for the same reason.

Honest: an SSM is not "better than a Transformer" — it is a different trade. Attention retrieves
from arbitrary distance exactly; a fixed-width recurrent state cannot, and this model will lose to
attention on tasks needing precise long-range lookup. What it buys is linear cost, a persistent
state, and — the reason it is here — a mechanism this system derived rather than inherited.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from nyxara.nyx001.substrate import (
    HAS_NUMPY,
    Tensor,
    he,
    ones,
    parameter,
    uniform_gate,
    xavier,
    zeros,
)
from nyxara.nyx001.substrate import autograd as A

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]

__all__ = ["SequenceModel"]


class SequenceModel:
    """A stacked selective state-space model over dynamic embeddings."""

    def __init__(self, substrate: Any, *, width: Optional[int] = None,
                 depth: Optional[int] = None, hidden_mult: int = 2) -> None:
        self.substrate = substrate
        self.width = int(width or getattr(substrate, "width", 256))
        self.depth = int(depth or getattr(substrate, "depth", 4))
        self.hidden = int(self.width * max(1, hidden_mult))
        self.available = bool(HAS_NUMPY)
        self._built = False
        self._params: List[Tensor] = []
        self._blocks: List[Dict[str, Tensor]] = []
        self._state: List[Any] = []
        self.calls = 0
        self._build()

    def _build(self) -> None:
        if not self.available:
            return
        try:
            rng = self.substrate.stream("sequence")
            for _ in range(self.depth):
                g0 = uniform_gate(rng, self.width).data
                block = {
                    "w_in": xavier(rng, self.width, self.width),
                    "w_sel": xavier(rng, self.width, self.width, gain=0.5),
                    "b_sel": zeros(self.width),
                    # logit-parameterised decay: cannot leave (0,1), and CAN learn to shut.
                    "a_raw": parameter(np.log(g0 / (1.0 - g0))),
                    "w_hid": he(rng, self.width, self.hidden),
                    "b_hid": zeros(self.hidden),
                    "w_out": xavier(rng, self.hidden, self.width, gain=0.1),
                    "b_out": zeros(self.width),
                    "norm_g": ones(self.width),
                    "norm_b": zeros(self.width),
                }
                self._blocks.append(block)
                self._params.extend(block.values())
            self._state = [None] * self.depth
            if hasattr(self.substrate, "note_params"):
                self.substrate.note_params(*self._params)
            self._built = True
        except Exception:  # noqa: BLE001
            self._built = False

    def _block_forward(self, z: Tensor, block: Dict[str, Tensor], h_prev: Any) -> Any:
        """One selective-scan block. ``z`` is (1, T, W); returns (output tensor, new state)."""
        zn = A.layernorm(z, block["norm_g"], block["norm_b"])
        gate = A.sigmoid(block["a_raw"])                             # (W,) in (0,1), learned
        # selection: what is worth letting into the state, decided per token
        sel = A.sigmoid(A.add(A.matmul(zn, block["w_sel"]), block["b_sel"]))
        proj = A.mul(A.matmul(zn, block["w_in"]), sel)
        if h_prev is not None:
            # The carried state enters at t=0 ONLY. `diag_scan` starts from h=0 and computes
            # h_t = a⊙h_{t-1} + proj_t, so seeding it means adding a⊙h_prev to the FIRST
            # timestep — after which the scan propagates it forward on its own.
            #
            # Broadcasting the seed across all T steps instead re-injects the previous state at
            # every timestep, and the scan then sums those injections geometrically: with a≈0.93
            # and T=16 one call returns a state ~14× the one it was given, so `forward(carry=True)`
            # compounds without bound. Measured before this fix: state_norm 2.6e+139 after 120
            # calls, with GELU overflowing to inf. Layer 2 writes the same seed unmasked and is
            # correct there only because it scans a single timestep, where "all T" and "t=0" are
            # the same thing — which is why this was inherited without being noticed.
            #
            # The seed still goes through the GRAPH so the decay gate keeps receiving gradient;
            # zeroing t>0 removes injections, not the ∂loss/∂gate term that teaches it how long
            # to remember.
            tlen = int(z.data.shape[1])
            seed = np.zeros((1, tlen, self.width), dtype=np.float64)
            seed[:, 0, :] = np.asarray(h_prev, dtype=np.float64).reshape(1, -1)
            proj = A.add(proj, A.mul(A.constant(seed), gate))
        h = A.diag_scan(proj, gate)                                  # (1,T,W)
        hid = A.gelu(A.add(A.matmul(h, block["w_hid"]), block["b_hid"]))
        out = A.add(A.matmul(hid, block["w_out"]), block["b_out"])
        out = A.mul(out, zn)                                         # gated by the input itself
        return A.add(out, z), h.data[0, -1, :]                       # residual + last state

    def forward(self, z: Tensor, *, carry: bool = True) -> Optional[Tensor]:
        """Run the stack over a (1, T, W) embedding tensor. Keeps state across calls if ``carry``."""
        if not self._built or z is None:
            return None
        try:
            self.calls += 1
            x = z
            for i, block in enumerate(self._blocks):
                x, new_state = self._block_forward(x, block, self._state[i] if carry else None)
                if carry:
                    self._state[i] = new_state
            return x
        except Exception:  # noqa: BLE001
            return None

    def reset_state(self) -> None:
        """Drop the carried belief without touching the weights — a new sequence, same model."""
        self._state = [None] * self.depth

    def parameters(self) -> List[Tensor]:
        return list(self._params)

    def state_norm(self) -> Optional[float]:
        """How much the recurrent state currently holds. ``None`` before anything has run."""
        if np is None or not any(s is not None for s in self._state):
            return None
        try:
            return float(sum(float(np.linalg.norm(s)) for s in self._state if s is not None))
        except Exception:  # noqa: BLE001
            return None

    def decay_range(self) -> Optional[Dict[str, float]]:
        """The learned decay gates' spread — evidence the selection is doing something."""
        if not self._built or np is None:
            return None
        try:
            gates = np.concatenate([1.0 / (1.0 + np.exp(-b["a_raw"].data))
                                    for b in self._blocks])
            return {"min": round(float(gates.min()), 4), "max": round(float(gates.max()), 4),
                    "mean": round(float(gates.mean()), 4), "std": round(float(gates.std()), 4)}
        except Exception:  # noqa: BLE001
            return None

    def stats(self) -> Dict[str, Any]:
        return {"built": self._built, "width": self.width, "depth": self.depth,
                "hidden": self.hidden, "calls": self.calls,
                "state_norm": self.state_norm(), "decay": self.decay_range(),
                "n_params": int(sum(int(p.data.size) for p in self._params)),
                "mechanism": "selective state-space scan (linear in T, carries state)"}

    def to_dict(self) -> Dict[str, Any]:
        return {"calls": self.calls}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.calls = int(d.get("calls", 0))
        except Exception:  # noqa: BLE001
            pass
