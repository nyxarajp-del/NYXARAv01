"""NYXARA · nyx001/lingua/dynamic_embedding.py — z_t = f(x_t, c_t, m_t, g_t, u_t) (🌊, not a lookup).

The charter's mathematical rethinking, first item: *"Dynamic Embeddings — token representation
changes with context and time."*

A standard embedding is a lookup table: token ``x`` always maps to row ``E[x]``, whatever is
happening. This one is a function of five things:

    z_t = LayerNorm( E[x_t] + W_c·c_t + W_m·m_t + W_g·g_t + (1 - u_t)·E[x_t] · γ )

where ``c_t`` is the current context state, ``m_t`` a memory trace retrieved for this token,
``g_t`` the goal in play, and ``u_t`` the system's uncertainty. Each is a real signal from a real
layer — context from Layer 2's recurrent state, memory from Layer 3's recall, goal from Layer 8,
uncertainty from Layer 2 — so the representation genuinely moves with the system's situation.

``u_t`` enters differently from the rest, and deliberately. It **scales the token's own identity
down**: under high uncertainty the static row contributes less and the contextual terms contribute
relatively more. The reasoning is that when the system does not know what is going on, what a token
meant *last time* is exactly the least trustworthy thing it holds — so it should lean on the
present situation rather than on a memorised default.

:meth:`DynamicEmbedding.drift` measures whether this is doing anything at all: the mean cosine
distance between the same token's embeddings in different contexts. A drift of ~0 means the
dynamics are decorative and it is a lookup table wearing five weight matrices. The number is
reported so that failure is visible rather than assumed away.

Honest: the table ``E`` is randomly initialised and learned only through whatever gradient reaches
it — with no language modelling objective running, it stays near its initialisation and the
contextual terms do most of the work. That is the expected state early in a NYX V.001's life, and
:meth:`stats` reports ``table_moved`` so it is checkable rather than assumed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from nyxara.nyx001.substrate import HAS_NUMPY, Tensor, ones, xavier, zeros
from nyxara.nyx001.substrate import autograd as A

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]

__all__ = ["DynamicEmbedding"]


class DynamicEmbedding:
    """Token representation as a function of context, memory, goal and uncertainty."""

    def __init__(self, substrate: Any, *, vocab: Optional[int] = None,
                 width: Optional[int] = None, gamma: float = 0.5) -> None:
        self.substrate = substrate
        self.vocab = int(vocab or getattr(substrate, "vocab", 8192))
        self.width = int(width or getattr(substrate, "width", 256))
        self.gamma = float(gamma)
        self.available = bool(HAS_NUMPY)
        self._built = False
        self._params: List[Tensor] = []
        self._drift_samples: List[float] = []
        self._table_init: Any = None
        self.calls = 0
        self._build()

    def _build(self) -> None:
        if not self.available:
            return
        try:
            rng = self.substrate.stream("dynamic_embedding")
            self.table = xavier(rng, self.vocab, self.width, gain=0.5)
            self.w_context = xavier(rng, self.width, self.width, gain=0.3)
            self.w_memory = xavier(rng, self.width, self.width, gain=0.3)
            self.w_goal = xavier(rng, self.width, self.width, gain=0.3)
            self.gain = ones(self.width)
            self.bias = zeros(self.width)
            self._params = [self.table, self.w_context, self.w_memory, self.w_goal,
                            self.gain, self.bias]
            self._table_init = self.table.data.copy()
            if hasattr(self.substrate, "note_params"):
                self.substrate.note_params(*self._params)
            self._built = True
        except Exception:  # noqa: BLE001
            self._built = False

    def _vec(self, v: Any) -> Any:
        """Coerce any signal to a width-sized row, or zeros when it is genuinely absent."""
        if v is None or np is None:
            return np.zeros((1, self.width)) if np is not None else None
        arr = np.asarray(v, dtype=np.float64).ravel()[: self.width]
        if arr.size < self.width:
            arr = np.pad(arr, (0, self.width - arr.size))
        return arr.reshape(1, self.width)

    def embed(self, token_id: int, *, context: Any = None, memory: Any = None,
              goal: Any = None, uncertainty: float = 0.0) -> Optional[Tensor]:
        """``z_t = f(x_t, c_t, m_t, g_t, u_t)``. Returns a live Tensor so gradients can reach it."""
        if not self._built:
            return None
        try:
            self.calls += 1
            tid = int(token_id) % self.vocab
            static = A.embed(self.table, np.asarray([[tid]]))          # (1,1,W)
            static = A.reshape(static, (1, self.width))
            u = float(min(1.0, max(0.0, uncertainty)))
            # Uncertainty scales the token's OWN identity down: what it meant last time is the
            # least trustworthy thing held when the situation is unclear.
            z = A.scale(static, 1.0 + self.gamma * (1.0 - u))
            for signal, weight in ((context, self.w_context), (memory, self.w_memory),
                                   (goal, self.w_goal)):
                if signal is not None:
                    z = A.add(z, A.matmul(A.constant(self._vec(signal)), weight))
            z = A.layernorm(z, self.gain, self.bias)
            return z
        except Exception:  # noqa: BLE001
            return None

    def embed_sequence(self, ids: List[int], *, context: Any = None, memory: Any = None,
                       goal: Any = None, uncertainty: float = 0.0) -> Optional[Tensor]:
        """A whole sequence, each token embedded under the same situation. (1, T, W)."""
        if not self._built or not ids:
            return None
        try:
            rows = [self.embed(i, context=context, memory=memory, goal=goal,
                               uncertainty=uncertainty) for i in ids]
            rows = [r for r in rows if r is not None]
            if not rows:
                return None
            stacked = A.concat_last(rows)                       # (1, T*W)
            return A.reshape(stacked, (1, len(rows), self.width))
        except Exception:  # noqa: BLE001
            return None

    # ---- is it actually dynamic? ---- #
    def drift(self, token_id: int, contexts: List[Any]) -> Optional[float]:
        """Mean cosine distance between one token's embeddings across contexts.

        This is the number that says whether the dynamics do anything. ~0.0 means it is a lookup
        table with extra steps.
        """
        if not self._built or np is None or len(contexts or []) < 2:
            return None
        try:
            vecs = []
            for c in contexts:
                z = self.embed(token_id, context=c)
                if z is not None:
                    vecs.append(z.data.ravel())
            if len(vecs) < 2:
                return None
            dists = []
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    a, b = vecs[i], vecs[j]
                    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
                    if na <= 0 or nb <= 0:
                        continue
                    dists.append(1.0 - float(np.dot(a, b) / (na * nb)))
            if not dists:
                return None
            d = float(sum(dists) / len(dists))
            self._drift_samples.append(d)
            if len(self._drift_samples) > 128:
                self._drift_samples = self._drift_samples[-64:]
            return d
        except Exception:  # noqa: BLE001
            return None

    def table_moved(self) -> Optional[float]:
        """How far the static table has travelled from its random init. 0.0 = never trained."""
        if not self._built or self._table_init is None or np is None:
            return None
        try:
            return float(np.linalg.norm(self.table.data - self._table_init))
        except Exception:  # noqa: BLE001
            return None

    def parameters(self) -> List[Tensor]:
        return list(self._params)

    def stats(self) -> Dict[str, Any]:
        return {"built": self._built, "vocab": self.vocab, "width": self.width,
                "calls": self.calls, "table_moved": self.table_moved(),
                "mean_drift": round(float(sum(self._drift_samples) / len(self._drift_samples)), 6)
                if self._drift_samples else None,
                "n_params": int(sum(int(p.data.size) for p in self._params))}

    def to_dict(self) -> Dict[str, Any]:
        return {"calls": self.calls}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.calls = int(d.get("calls", 0))
        except Exception:  # noqa: BLE001
            pass
