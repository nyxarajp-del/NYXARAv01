"""NYXARA · growth/forecaster.py — the payoff forecaster (🔮, learned credit, heavy-ML).

The :class:`~nyxara.growth.credit.ImprovementLedger` learns *per-arm* how often an intervention
pays off (a Beta posterior keyed by weakness-class / action). That is robust and dependency-free,
but it cannot *generalise across context*: it can't learn "edits of this class pay off **when the
benchmark is already high and the machine is strong**, but not otherwise." This module adds that
generalisation as an optional, genuinely-learned model.

``PayoffForecaster`` is a small torch MLP mapping a feature vector — the live cycle signals
(accuracy, handoff, weaknesses, knowledge, capacity, trend) concatenated with a hashed embedding
of the arm — to a predicted reward in ``[0, 1]`` (the index-gain the ledger would have recorded).
It trains online from observed ``(arm, features, reward)`` outcomes via SGD on a tiny replay
buffer, exactly the graceful-degradation contract used across the foundry:

* **torch present** → a real learned predictor sharpens the bandit's ranking with context.
* **torch absent** → :meth:`available` is ``False`` and :meth:`predict` returns ``None``; the
  caller transparently falls back to the ledger's Beta-Thompson scores. Nothing breaks; CI/CPU
  runs the dependency-free path.

It only ever *scores* candidate work — it never touches a gate, edits source, or widens authority.
"""

from __future__ import annotations

import hashlib
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

__all__ = ["PayoffForecaster", "FEATURE_KEYS"]

# the ordered, fixed cycle-signal features the model reads (stable layout = train/infer parity)
FEATURE_KEYS: Tuple[str, ...] = (
    "accuracy", "handoff", "weaknesses", "knowledge", "stability",
    "capacity", "trend", "index", "frontier", "rigor",
)
_ARM_BUCKETS = 16          # hashed arm embedding width — generalises across same-class arms


def _arm_bucket(key: str) -> int:
    h = hashlib.sha256((key or "").encode("utf-8")).hexdigest()
    return int(h, 16) % _ARM_BUCKETS


def _featurize(key: str, features: Dict[str, float]) -> List[float]:
    """Deterministic fixed-width vector: ordered signals + a one-hot hashed arm bucket."""
    vec = [float(features.get(k, 0.0) or 0.0) for k in FEATURE_KEYS]
    onehot = [0.0] * _ARM_BUCKETS
    onehot[_arm_bucket(key)] = 1.0
    return vec + onehot


class PayoffForecaster:
    """A small online MLP predicting an intervention's index-gain reward in ``[0, 1]``.

    Parameters
    ----------
    settings:  :class:`~nyxara.kernel.config.NyxaraSettings` (read for buffer/step bounds).
    hidden:    hidden width of the two-layer MLP.
    lr:        SGD learning rate for the online updates.
    """

    def __init__(self, *, settings: Any = None, hidden: int = 32, lr: float = 0.05,
                 buffer_size: int = 512) -> None:
        from nyxara.kernel.config import get_settings
        self.settings = settings or get_settings()
        self.hidden = int(hidden)
        self.lr = float(lr)
        self._in_dim = len(FEATURE_KEYS) + _ARM_BUCKETS
        self._buffer: Deque[Tuple[List[float], float]] = deque(maxlen=int(buffer_size))
        self._torch = None
        self._model = None
        self._opt = None
        self._ready = False
        self._build()

    # ---- construction (graceful: no torch → disabled, never raises) ---- #
    def _build(self) -> None:
        try:
            import torch
            from torch import nn
        except Exception:  # noqa: BLE001 — no torch → the forecaster is simply unavailable
            self._torch = None
            return
        self._torch = torch
        self._model = nn.Sequential(
            nn.Linear(self._in_dim, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, 1), nn.Sigmoid())
        self._opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        self._ready = True

    def available(self) -> bool:
        """True only when torch is present AND the model has seen enough data to be trusted."""
        cfg = self.settings.self_improvement
        if not bool(getattr(cfg, "use_payoff_forecaster", False)):
            return False
        warmup = int(getattr(cfg, "forecaster_warmup", 16))
        return bool(self._ready and len(self._buffer) >= warmup)

    # ---- observe one outcome (fills the replay buffer; optionally trains) ---- #
    def observe(self, key: str, features: Dict[str, float], reward: float, *,
                train: bool = True) -> None:
        if not self._ready:
            return
        x = _featurize(key, features or {})
        self._buffer.append((x, max(0.0, min(1.0, float(reward)))))
        if train:
            self.train_step()

    def train_step(self, *, batch: int = 32) -> Optional[float]:
        """One SGD step on a random minibatch from the replay buffer. Returns the MSE loss."""
        if not self._ready or len(self._buffer) < 2:
            return None
        torch = self._torch
        import random
        n = min(int(batch), len(self._buffer))
        sample = random.sample(self._buffer, n)
        try:
            xs = torch.tensor([x for x, _ in sample], dtype=torch.float32)
            ys = torch.tensor([[y] for _, y in sample], dtype=torch.float32)
            self._model.train()
            pred = self._model(xs)
            loss = torch.nn.functional.mse_loss(pred, ys)
            self._opt.zero_grad()
            loss.backward()
            self._opt.step()
            return float(loss.detach().item())
        except Exception:  # noqa: BLE001 — a failed step never sinks the cycle
            return None

    # ---- predict the payoff of an arm in the current context ---- #
    def predict(self, key: str, features: Dict[str, float]) -> Optional[float]:
        if not self._ready:
            return None
        torch = self._torch
        try:
            x = torch.tensor([_featurize(key, features or {})], dtype=torch.float32)
            self._model.eval()
            with torch.no_grad():
                return float(self._model(x).item())
        except Exception:  # noqa: BLE001
            return None


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA payoff-forecaster self-test")
    print("=" * 70)

    fc = PayoffForecaster()
    if fc._torch is None:
        print("· torch not installed — forecaster correctly UNAVAILABLE (ledger path stands)")
        assert not fc.available()
        assert fc.predict("edit:x:transform", {"accuracy": 0.5}) is None
        print("\nALL SELF-TESTS PASSED ✓ (degraded path)")
    else:
        # teach it a learnable rule: arm 'good' pays off when accuracy is high, 'bad' never does.
        fc.settings.self_improvement.use_payoff_forecaster = True
        import random
        for _ in range(400):
            acc = random.random()
            feats = {"accuracy": acc, "capacity": 0.8}
            fc.observe("edit:good:transform", feats, reward=acc)
            fc.observe("edit:bad:transform", feats, reward=0.05)
        assert fc.available()
        hi = fc.predict("edit:good:transform", {"accuracy": 0.95, "capacity": 0.8})
        lo = fc.predict("edit:bad:transform", {"accuracy": 0.95, "capacity": 0.8})
        print(f"\npredict good/bad @acc=0.95: {hi:.3f} / {lo:.3f}")
        assert hi is not None and lo is not None and hi > lo
        print("\nALL SELF-TESTS PASSED ✓ (torch path)")
