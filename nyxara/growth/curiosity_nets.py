"""NYXARA · growth/curiosity_nets.py — curiosity that is LEARNED BY GRADIENT DESCENT (🌌).

:mod:`growth.frontier` directs open-ended exploration by *counting and measuring*: a MAP-Elites grid
and a kNN "novelty = mean distance to the nearest behaviours seen." Honest, but symbolic — nothing is
learned in a parameter space. This module gives the frontier a **learned** novelty signal via
**Random Network Distillation** (RND), the standard learned-curiosity method, trained by real
back-propagation with numpy the only dependency and no LLM in the loop:

* A **fixed random target** net ``t(b)`` maps a behaviour descriptor to an embedding and is *never*
  trained — it is a stable, arbitrary fingerprint of the behaviour space.
* A **learned predictor** ``p_θ(b)`` is trained by gradient descent (Adam) to reproduce the target's
  output on every behaviour NYXARA actually visits.
* **Novelty = the predictor's error** ``‖t(b) − p_θ(b)‖²``, squashed to ``[0, 1]`` by a self-calibrated
  running scale. Where she has been often, the predictor has caught up → low error → low novelty
  (**habituation is learned, not counted**); a genuinely unfamiliar region the predictor has never fit
  → high error → high novelty.

It reuses the same battle-tested numpy MLP the world model learns its dynamics with
(:class:`nyxara.mind.world_model._MLP`), persists as a JSON-safe dict, and degrades honestly: without
numpy it simply reports ``None`` and the frontier keeps its kNN floor.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

try:                                       # numpy is the substrate for the learned predictor; optional
    import numpy as np
    _HAS_NUMPY = True
except Exception:                          # noqa: BLE001 — the frontier keeps its kNN floor
    np = None  # type: ignore
    _HAS_NUMPY = False

__all__ = ["has_numpy", "RNDNovelty"]


def has_numpy() -> bool:
    return _HAS_NUMPY


class RNDNovelty:
    """Random Network Distillation: novelty learned by gradient descent (a fixed target + a trained
    predictor). ``score`` reads novelty without learning; ``update`` takes one gradient step."""

    def __init__(self, in_dim: int, *, emb: int = 8, hidden: Sequence[int] = (16,),
                 lr: float = 0.02, seed: int = 0) -> None:
        self.in_dim = max(1, int(in_dim))
        self.emb = max(1, int(emb))
        self.hidden = tuple(int(h) for h in hidden)
        self.lr = float(lr)
        self.seed = int(seed)
        self.scale = 1.0                                   # EMA of raw prediction error (self-calibration)
        self.updates = 0
        self._target: Any = None                          # frozen random net
        self._pred: Any = None                            # learned net
        if _HAS_NUMPY:
            self._build()

    def _build(self) -> None:
        from nyxara.mind.world_model import _MLP           # reuse the tested numpy MLP + Adam
        self._target = _MLP(self.in_dim, self.emb, self.hidden, lr=0.0, seed=self.seed + 9973)
        self._pred = _MLP(self.in_dim, self.emb, self.hidden, lr=self.lr, seed=self.seed)

    def _row(self, x: Sequence[float]) -> Any:
        v = [float(e) for e in x][: self.in_dim]
        if len(v) < self.in_dim:
            v = v + [0.0] * (self.in_dim - len(v))
        return np.asarray([v], dtype=np.float64)

    def _raw_error(self, X: Any) -> float:
        t, _ = self._target.forward(X)
        p, _ = self._pred.forward(X)
        return float(np.mean((t - p) ** 2))

    def score(self, x: Sequence[float]) -> Optional[float]:
        """Novelty in ``[0, 1)`` — high where the predictor has not yet fit this region. None w/o numpy."""
        if not (_HAS_NUMPY and self._pred is not None):
            return None
        err = self._raw_error(self._row(x))
        return err / (err + self.scale + 1e-8)

    def update(self, x: Sequence[float]) -> Optional[float]:
        """One gradient step of the predictor toward the target on behaviour ``x``; returns raw error."""
        if not (_HAS_NUMPY and self._pred is not None):
            return None
        X = self._row(x)
        err = self._raw_error(X)
        target_out, _ = self._target.forward(X)
        self._pred.train_batch(X, target_out)              # real Adam back-prop toward the frozen target
        self.scale += 0.02 * (err - self.scale)            # EMA calibration of the error scale
        self.updates += 1
        return err

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"in_dim": self.in_dim, "emb": self.emb, "hidden": list(self.hidden),
                             "lr": self.lr, "seed": self.seed, "scale": float(self.scale),
                             "updates": self.updates}
        if _HAS_NUMPY and self._pred is not None:
            d["target"] = self._target.to_dict()
            d["pred"] = self._pred.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RNDNovelty":
        obj = cls(int(d.get("in_dim", 3)), emb=int(d.get("emb", 8)),
                  hidden=d.get("hidden", (16,)), lr=float(d.get("lr", 0.02)),
                  seed=int(d.get("seed", 0)))
        obj.scale = float(d.get("scale", 1.0)) or 1.0
        obj.updates = int(d.get("updates", 0))
        if _HAS_NUMPY and obj._pred is not None and d.get("pred") and d.get("target"):
            obj._target.load_dict(d["target"])
            obj._pred.load_dict(d["pred"])
        return obj
