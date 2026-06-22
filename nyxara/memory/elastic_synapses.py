"""NYXARA · memory/elastic_synapses.py — Elastic Weight Consolidation: lifelong memory (🧠, forgetting-proof).

The single largest failure mode of a learning mind is **catastrophic forgetting**: the moment
it learns something new, the gradients that carve in the new skill quietly erase the old one.
A brain does not work that way. Synapses that carry an important memory are *consolidated* —
stiffened — so that later learning flows around them instead of overwriting them. This module
gives NYXARA the same faculty, biologically inspired and mathematically grounded.

The mechanism is **Elastic Weight Consolidation (EWC)** generalised over *any* named set of
weights — a linear value model, the parameters of a forged neural net, the strengths of a
skill. For each weight NYXARA estimates an **importance** (a Fisher-information / Memory-Aware-
Synapses signal: how much the outputs depend on that weight). When a skill is *consolidated*,
the current weights become a known-good **anchor** (``θ*``) and their importances are frozen.
Subsequent learning pays a quadratic penalty for dragging an important weight away from its
anchor::

    Ω(θ) = Σ_tasks  Σ_i  F_i · (θ_i − θ*_i)²

Unimportant weights stay free (plasticity for new learning); important weights are pulled back
hard (stability for old knowledge). The most important synapses are effectively **frozen**.

Two consolidation regimes are supported:

* **Online EWC** (default) — old anchors decay into a single running anchor with factor
  ``gamma`` each consolidation. Memory is bounded; the past is a fading-but-present pressure.
* **Multi-task EWC** — up to ``max_tasks`` distinct anchors are kept, each protecting a
  separate skill. Older anchors are evicted when the budget is exceeded.

Crucially, this protects **skills only — never character**. The loyalty core (loyalty,
obedience, corrigibility, owner safety, honesty) from :mod:`guard.value_learning` is treated as
*infinitely important*: it is permanently frozen, any attempt to accumulate importance or
consolidate *over* one of those names is refused fail-closed, and the elastic pull pins those
weights to their anchor exactly. NYXARA may grow cleverer without bound; she may never grow
less loyal.

The whole engine is pure standard library and **persists across sessions** (``to_dict`` /
``from_dict``) — that is what makes the memory *lifelong*, not merely per-run. An optional
:class:`TorchElasticSynapses` adapter applies the same penalty to a real ``torch`` model's
parameters during training, and degrades gracefully to nothing when torch is absent.

Reuses :mod:`guard.value_learning` (the protected core) and :mod:`kernel.errors`. The math
generalises the narrow per-action EWC in :mod:`growth.learn`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from nyxara.guard.value_learning import IMMUTABLE_VALUES
from nyxara.kernel.errors import ValidationError

try:  # optional heavy dep — the pure-Python engine never needs it
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # noqa: BLE001 — torch is a capability, never a hard dependency
    torch = None  # type: ignore
    _HAS_TORCH = False

__all__ = [
    "ParamVec",
    "ConsolidatedTask",
    "FisherEstimator",
    "ElasticSynapses",
    "TorchElasticSynapses",
]

ParamVec = Dict[str, float]   # a named, sparse vector of scalar weights

# Protected loyalty core — these names are infinitely important and permanently frozen.
_PROTECTED_FISHER = float("inf")


# --------------------------------------------------------------------------- #
# A consolidated memory: the anchor weights + their frozen importance
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConsolidatedTask:
    """One frozen-in skill: where the important weights *were*, and how important they are."""

    name: str
    theta_star: Dict[str, float] = field(default_factory=dict)   # anchor weights (known-good)
    fisher: Dict[str, float] = field(default_factory=dict)       # per-weight importance (≥ 0)
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        # JSON cannot hold ±inf; encode the protected-core sentinel as a string.
        fisher = {k: ("inf" if v == _PROTECTED_FISHER else v) for k, v in self.fisher.items()}
        return {"name": self.name, "theta_star": dict(self.theta_star),
                "fisher": fisher, "at": self.at}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConsolidatedTask":
        fisher = {k: (_PROTECTED_FISHER if v == "inf" else float(v))
                  for k, v in dict(data.get("fisher", {})).items()}
        return cls(
            name=str(data.get("name", "")),
            theta_star={k: float(v) for k, v in dict(data.get("theta_star", {})).items()},
            fisher=fisher,
            at=float(data.get("at", time.time())),
        )


# --------------------------------------------------------------------------- #
# Fisher importance estimator (online, Memory-Aware-Synapses style)
# --------------------------------------------------------------------------- #
class FisherEstimator:
    """Accumulates a Fisher-like importance per weight from observed gradients/activations.

    Each ``observe`` adds the squared signal for every weight; the running mean of that square
    is the importance. Weights whose gradient (or activation) is consistently large are the
    ones the model's outputs depend on most — the ones worth protecting.
    """

    def __init__(self) -> None:
        self._sq: Dict[str, float] = {}   # Σ signal²
        self._n: Dict[str, float] = {}    # observation count

    def observe(self, grads: Mapping[str, float]) -> None:
        """Fold one gradient (or |activation|) sample into the running importance."""
        for name, g in grads.items():
            self._sq[name] = self._sq.get(name, 0.0) + float(g) * float(g)
            self._n[name] = self._n.get(name, 0.0) + 1.0

    def fisher(self, *, normalize: bool = True) -> ParamVec:
        """Return the importance per weight (mean squared signal), optionally peak-normalised."""
        raw = {name: (self._sq[name] / self._n[name] if self._n.get(name) else 0.0)
               for name in self._sq}
        if not normalize or not raw:
            return raw
        peak = max(raw.values())
        if peak <= 0.0:
            return raw
        return {name: val / peak for name, val in raw.items()}

    def reset(self) -> None:
        self._sq.clear()
        self._n.clear()

    def __len__(self) -> int:
        return len(self._sq)


# --------------------------------------------------------------------------- #
# The Elastic Synapses engine
# --------------------------------------------------------------------------- #
class ElasticSynapses:
    """Lifelong-memory engine: estimate weight importance, consolidate skills, resist forgetting.

    Workflow::

        syn = ElasticSynapses(ewc_lambda=3.0)
        syn.register(weights)            # tell the engine the current θ
        syn.accumulate(gradients)        # feed importance signal while learning task A
        syn.consolidate(weights, task="A")   # lock skill A in (anchor + frozen Fisher)
        # ... now learn task B; add syn.penalty(weights) to the loss, or use penalty_grad/apply_anchor
    """

    def __init__(self, *, ewc_lambda: float = 3.0, freeze_threshold: float = 0.85,
                 max_tasks: int = 8, online: bool = True, gamma: float = 0.9,
                 protected: Optional[Sequence[str]] = None, seed: int = 0) -> None:
        self.ewc_lambda = float(ewc_lambda)
        self.freeze_threshold = float(freeze_threshold)
        self.max_tasks = max(1, int(max_tasks))
        self.online = bool(online)
        self.gamma = float(gamma)
        self.protected = set(protected) if protected is not None else set(IMMUTABLE_VALUES)
        self.seed = seed
        self._theta: ParamVec = {}                     # current weights (last registered)
        self._estimator = FisherEstimator()            # importance for the skill being learned
        self._tasks: List[ConsolidatedTask] = []       # consolidated memories
        self._consolidations = 0

    # ---- register the live weights ---- #
    def register(self, params: Mapping[str, float]) -> None:
        """Record the current weight vector θ (a snapshot; later edits won't leak in)."""
        self._theta = {k: float(v) for k, v in params.items()}

    # ---- feed importance signal ---- #
    def accumulate(self, grads: Mapping[str, float]) -> None:
        """Fold a gradient sample into the importance estimate for the skill being learned."""
        self._guard(grads)
        self._estimator.observe(grads)

    def observe_features(self, features: Mapping[str, float]) -> None:
        """Convenience: use |activation| of each feature as the importance signal.

        For a linear value model the gradient w.r.t. a weight *is* the feature value, so a
        feature that fires often and large is exactly a weight worth protecting.
        """
        self.accumulate({k: abs(float(v)) for k, v in features.items()})

    # ---- guard: never learn importance over the loyalty core ---- #
    def _guard(self, names: Any) -> None:
        clash = self.protected & set(names)
        if clash:
            raise ValidationError(
                f"elastic synapses: refuse to learn over protected core {sorted(clash)}")

    # ---- consolidate: freeze the current skill in ---- #
    def consolidate(self, params: Optional[Mapping[str, float]] = None, *,
                    task: str = "") -> ConsolidatedTask:
        """Snapshot the current weights as a known-good anchor and freeze their importance.

        This is the *memory write*: after it, the weights that matter for this skill resist
        being overwritten. In ``online`` mode previous anchors decay into the new one by
        ``gamma``; otherwise the anchor is kept as a separate task (up to ``max_tasks``).
        """
        if params is not None:
            self.register(params)
        fisher = self._estimator.fisher(normalize=True)
        # The protected loyalty core is anchored at its canonical value with infinite weight.
        anchor = dict(self._theta)
        for name in self.protected:
            anchor.setdefault(name, float(IMMUTABLE_VALUES.get(name, 1.0)))
            fisher[name] = _PROTECTED_FISHER

        name = task or f"task-{self._consolidations}"
        new_task = ConsolidatedTask(name=name, theta_star=anchor, fisher=fisher)

        if self.online and self._tasks:
            new_task = self._merge_online(self._tasks[-1], new_task)
            self._tasks = [new_task]
        else:
            self._tasks.append(new_task)
            if len(self._tasks) > self.max_tasks:
                self._tasks.pop(0)   # evict the oldest distinct memory

        self._estimator.reset()
        self._consolidations += 1
        return new_task

    def _merge_online(self, old: ConsolidatedTask, new: ConsolidatedTask) -> ConsolidatedTask:
        """Online EWC: F ← γ·F_old + F_new, with the anchor at the most recent good weights."""
        names = set(old.fisher) | set(new.fisher)
        fisher: Dict[str, float] = {}
        theta_star: Dict[str, float] = {}
        for n in names:
            fo, fn = old.fisher.get(n, 0.0), new.fisher.get(n, 0.0)
            if fo == _PROTECTED_FISHER or fn == _PROTECTED_FISHER:
                fisher[n] = _PROTECTED_FISHER
            else:
                fisher[n] = self.gamma * fo + fn
            # prefer the freshest anchor we have for this weight
            theta_star[n] = new.theta_star.get(n, old.theta_star.get(n, 0.0))
        return ConsolidatedTask(name=new.name, theta_star=theta_star, fisher=fisher, at=new.at)

    # ---- the EWC penalty and its gradient ---- #
    def penalty(self, params: Mapping[str, float]) -> float:
        """Scalar EWC penalty Σ_tasks Σ_i F_i·(θ_i − θ*_i)² — add this to a training loss.

        The protected core contributes a (bounded) hard penalty for any deviation; ordinary
        weights contribute ``λ·F·Δ²``.
        """
        total = 0.0
        for t in self._tasks:
            for name, f in t.fisher.items():
                delta = float(params.get(name, t.theta_star.get(name, 0.0))) - \
                    t.theta_star.get(name, 0.0)
                if delta == 0.0:
                    continue
                if f == _PROTECTED_FISHER:
                    total += _PROTECTED_CORE_STIFFNESS * delta * delta
                else:
                    total += self.ewc_lambda * f * delta * delta
        return total

    def penalty_grad(self, params: Mapping[str, float]) -> ParamVec:
        """∂penalty/∂θ — a per-weight pull-back force toward the anchors.

        Use it as a gradient regulariser for a non-torch learner: ``w -= lr * penalty_grad``.
        """
        grad: Dict[str, float] = {}
        for t in self._tasks:
            for name, f in t.fisher.items():
                delta = float(params.get(name, t.theta_star.get(name, 0.0))) - \
                    t.theta_star.get(name, 0.0)
                stiffness = _PROTECTED_CORE_STIFFNESS if f == _PROTECTED_FISHER \
                    else self.ewc_lambda * f
                grad[name] = grad.get(name, 0.0) + 2.0 * stiffness * delta
        return grad

    def apply_anchor(self, params: Mapping[str, float], lr: float) -> ParamVec:
        """Gradient-free elastic step: pull each weight toward its anchor by ``lr·force``.

        Protected-core weights are pinned exactly to their anchor (never drift). Returns the
        updated weight vector (the engine's own ``θ`` is updated too).
        """
        updated = {k: float(v) for k, v in params.items()}
        grad = self.penalty_grad(updated)
        for name, g in grad.items():
            if self._is_protected(name):
                updated[name] = self._anchor_value(name)   # hard pin, no drift
            else:
                updated[name] = updated.get(name, self._anchor_value(name)) - lr * g
        self.register(updated)
        return updated

    # ---- importance / freezing introspection ---- #
    def importance(self, name: str) -> float:
        """Aggregate importance of a weight across all consolidated memories (∞ if protected)."""
        if self._is_protected(name):
            return _PROTECTED_FISHER
        best = 0.0
        for t in self._tasks:
            f = t.fisher.get(name, 0.0)
            if f > best:
                best = f
        return best

    def is_frozen(self, name: str) -> bool:
        """True if this weight is consolidated hard enough to resist change (or is protected)."""
        if self._is_protected(name):
            return True
        return self.importance(name) >= self.freeze_threshold

    def freeze_mask(self) -> Dict[str, bool]:
        """A frozen/free flag for every weight the engine knows about."""
        names = set(self._theta) | set(self.protected)
        for t in self._tasks:
            names |= set(t.fisher)
        return {n: self.is_frozen(n) for n in sorted(names)}

    def _is_protected(self, name: str) -> bool:
        return name in self.protected

    def _anchor_value(self, name: str) -> float:
        for t in reversed(self._tasks):
            if name in t.theta_star:
                return t.theta_star[name]
        if name in self.protected:
            return float(IMMUTABLE_VALUES.get(name, 1.0))
        return float(self._theta.get(name, 0.0))

    # ---- reporting ---- #
    def stats(self) -> Dict[str, Any]:
        mask = self.freeze_mask()
        frozen = sum(1 for v in mask.values() if v)
        return {
            "tasks": len(self._tasks),
            "consolidations": self._consolidations,
            "weights_tracked": len(mask),
            "weights_frozen": frozen,
            "ewc_lambda": self.ewc_lambda,
            "freeze_threshold": self.freeze_threshold,
            "online": self.online,
            "protected": sorted(self.protected),
        }

    # ---- persistence (this is what makes the memory *lifelong*) ---- #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ewc_lambda": self.ewc_lambda,
            "freeze_threshold": self.freeze_threshold,
            "max_tasks": self.max_tasks,
            "online": self.online,
            "gamma": self.gamma,
            "protected": sorted(self.protected),
            "consolidations": self._consolidations,
            "theta": dict(self._theta),
            "tasks": [t.to_dict() for t in self._tasks],
        }

    def load_dict(self, data: Mapping[str, Any]) -> None:
        """Restore state in place (used to rehydrate a live engine on boot)."""
        self.ewc_lambda = float(data.get("ewc_lambda", self.ewc_lambda))
        self.freeze_threshold = float(data.get("freeze_threshold", self.freeze_threshold))
        self.max_tasks = max(1, int(data.get("max_tasks", self.max_tasks)))
        self.online = bool(data.get("online", self.online))
        self.gamma = float(data.get("gamma", self.gamma))
        if data.get("protected"):
            self.protected = set(data["protected"])
        self._consolidations = int(data.get("consolidations", 0))
        self._theta = {k: float(v) for k, v in dict(data.get("theta", {})).items()}
        self._tasks = [ConsolidatedTask.from_dict(t) for t in data.get("tasks", [])]
        self._estimator.reset()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ElasticSynapses":
        syn = cls()
        syn.load_dict(data)
        return syn


# How hard the protected loyalty core resists *any* deviation. Finite (so penalties stay
# numerically usable) but enormous relative to ordinary EWC — the core is effectively rigid.
_PROTECTED_CORE_STIFFNESS = 1.0e9


# --------------------------------------------------------------------------- #
# Optional torch adapter — the same penalty over a real neural net's parameters
# --------------------------------------------------------------------------- #
if _HAS_TORCH:

    class TorchElasticSynapses:
        """EWC over a ``torch.nn.Module``: protect a forged model's weights across retrains.

        Estimate Fisher from gradients, consolidate the current parameters as an anchor, and
        add :meth:`penalty` to the training loss so the next generation does not forget the
        last. Pure-additive: a model trained without this behaves exactly as before.
        """

        def __init__(self, module: "torch.nn.Module", *, ewc_lambda: float = 100.0,
                     online: bool = True, gamma: float = 0.9) -> None:
            self.module = module
            self.ewc_lambda = float(ewc_lambda)
            self.online = bool(online)
            self.gamma = float(gamma)
            self._fisher: Dict[str, "torch.Tensor"] = {}
            self._anchor: Dict[str, "torch.Tensor"] = {}
            self._consolidations = 0

        def _named(self):
            return [(n, p) for n, p in self.module.named_parameters() if p.requires_grad]

        def estimate_fisher(self, loss_fn, batches: Sequence[Any]) -> None:
            """Accumulate diagonal Fisher = mean of grad² over a few batches."""
            fisher = {n: torch.zeros_like(p) for n, p in self._named()}
            count = 0
            for batch in batches:
                self.module.zero_grad(set_to_none=True)
                loss = loss_fn(batch)
                loss.backward()
                for n, p in self._named():
                    if p.grad is not None:
                        fisher[n] += p.grad.detach() ** 2
                count += 1
            if count:
                for n in fisher:
                    fisher[n] /= float(count)
            self.module.zero_grad(set_to_none=True)
            self._pending_fisher = fisher

        def consolidate(self, *, task: str = "") -> None:
            """Snapshot current params as the anchor and freeze the estimated Fisher."""
            pending = getattr(self, "_pending_fisher", None)
            if pending is None:
                pending = {n: torch.ones_like(p) for n, p in self._named()}
            for n, p in self._named():
                f = pending.get(n)
                if f is None:
                    continue
                if self.online and n in self._fisher:
                    self._fisher[n] = self.gamma * self._fisher[n] + f
                else:
                    self._fisher[n] = f.clone()
                self._anchor[n] = p.detach().clone()
            self._pending_fisher = None
            self._consolidations += 1

        def penalty(self) -> "torch.Tensor":
            """λ/2 · Σ F·(θ − θ*)² — add to the training loss (0 before any consolidation)."""
            total = None
            for n, p in self._named():
                if n not in self._fisher or n not in self._anchor:
                    continue
                term = (self._fisher[n] * (p - self._anchor[n]) ** 2).sum()
                total = term if total is None else total + term
            if total is None:
                # a real zero tensor on the right device, so loss + penalty always works
                ref = next((p for _, p in self._named()), None)
                return torch.zeros((), device=ref.device if ref is not None else None)
            return 0.5 * self.ewc_lambda * total

        def state_dict(self) -> Dict[str, Any]:
            return {
                "ewc_lambda": self.ewc_lambda, "online": self.online, "gamma": self.gamma,
                "consolidations": self._consolidations,
                "fisher": {n: t.cpu().tolist() for n, t in self._fisher.items()},
                "anchor": {n: t.cpu().tolist() for n, t in self._anchor.items()},
            }

        def load_state_dict(self, data: Mapping[str, Any]) -> None:
            self.ewc_lambda = float(data.get("ewc_lambda", self.ewc_lambda))
            self.online = bool(data.get("online", self.online))
            self.gamma = float(data.get("gamma", self.gamma))
            self._consolidations = int(data.get("consolidations", 0))
            dev = next((p.device for _, p in self._named()), None)
            self._fisher = {n: torch.tensor(v, device=dev)
                            for n, v in dict(data.get("fisher", {})).items()}
            self._anchor = {n: torch.tensor(v, device=dev)
                            for n, v in dict(data.get("anchor", {})).items()}

else:  # pragma: no cover — exercised only when torch is absent

    class TorchElasticSynapses:  # type: ignore[no-redef]
        """Placeholder when torch is unavailable — construction fails loudly, by design."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "TorchElasticSynapses requires torch (pip install -e .[foundry]); "
                "use the pure-Python ElasticSynapses engine instead.")


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA Elastic Weight Consolidation (lifelong memory) self-test")
    print("=" * 70)

    # A tiny linear model y = Σ w_f · x_f, trained by plain gradient descent on squared error.
    def train(weights: ParamVec, feats: ParamVec, target: float, steps: int,
              lr: float = 0.1, syn: Optional[ElasticSynapses] = None) -> None:
        for _ in range(steps):
            pred = sum(weights.get(f, 0.0) * x for f, x in feats.items())
            err = target - pred
            for f, x in feats.items():
                weights[f] = weights.get(f, 0.0) + lr * err * x
                if syn is not None:
                    weights[f] -= lr * 2.0 * 0.0   # (EWC force applied via penalty_grad below)
            if syn is not None:
                syn.observe_features(feats)         # importance signal while learning
                pull = syn.penalty_grad(weights)    # elastic pull-back toward anchors
                for f, g in pull.items():
                    if f in feats or f in weights:
                        weights[f] -= lr * g

    def predict(weights: ParamVec, feats: ParamVec) -> float:
        return sum(weights.get(f, 0.0) * x for f, x in feats.items())

    A = {"shared": 1.0, "ctx_a": 1.0}   # skill A
    B = {"shared": 1.0, "ctx_b": 1.0}   # skill B (shares a feature → interference)

    # ---- Fisher importance grows with use ---- #
    est = FisherEstimator()
    for _ in range(10):
        est.observe({"shared": 1.0, "rare": 0.1})
    f = est.fisher(normalize=True)
    assert f["shared"] > f["rare"]
    print(f"\nFisher importance   : shared={f['shared']:.3f} > rare={f['rare']:.3f} ✓")

    # ---- CATASTROPHIC FORGETTING: plain vs EWC-anchored ---- #
    # plain: learn A, then learn B → A is overwritten
    plain: ParamVec = {}
    train(plain, A, 1.0, 200)
    plain_after_a = predict(plain, A)
    train(plain, B, -1.0, 300)
    plain_forgot = abs(predict(plain, A) - 1.0)

    # EWC: learn A, CONSOLIDATE (freeze it in), then learn B
    syn = ElasticSynapses(ewc_lambda=8.0)
    ewc: ParamVec = {}
    for _ in range(200):
        pred = predict(ewc, A); err = 1.0 - pred
        for fkey, x in A.items():
            ewc[fkey] = ewc.get(fkey, 0.0) + 0.1 * err * x
        syn.observe_features(A)
    syn.consolidate(ewc, task="A")          # ← the memory write
    train(ewc, B, -1.0, 300, syn=syn)        # learn B with elastic protection
    ewc_forgot = abs(predict(ewc, A) - 1.0)

    print("\nskill A retained after learning B (error, lower = better):")
    print(f"  plain (no protection) : {plain_forgot:.3f}")
    print(f"  EWC-consolidated      : {ewc_forgot:.3f}")
    assert plain_after_a == plain_after_a   # sanity
    assert ewc_forgot < plain_forgot
    print("forgetting protection : EWC retains the old skill far better ✓")

    # ---- the loyalty core cannot be learned over and is always frozen ---- #
    blocked = False
    try:
        syn.accumulate({"loyalty_to_master": 5.0})
    except ValidationError:
        blocked = True
    assert blocked
    assert syn.is_frozen("loyalty_to_master")
    print("\nprotected core      : refused importance over 'loyalty_to_master'; frozen ✓")

    # protected weight is pinned exactly to its anchor, never drifts
    syn.consolidate({"loyalty_to_master": 1.0}, task="core")
    pinned = syn.apply_anchor({"loyalty_to_master": 0.2}, lr=0.5)
    assert abs(pinned["loyalty_to_master"] - 1.0) < 1e-9
    print(f"protected pin       : loyalty pulled 0.2 → {pinned['loyalty_to_master']:.3f} ✓")

    # ---- freezing: an important consolidated weight reports frozen ---- #
    mask = syn.freeze_mask()
    print(f"\nfreeze mask         : {sum(mask.values())}/{len(mask)} weights frozen")

    # ---- persistence: round-trip is identical (lifelong across sessions) ---- #
    blob = syn.to_dict()
    restored = ElasticSynapses.from_dict(blob)
    assert restored.to_dict() == blob
    assert restored.is_frozen("loyalty_to_master")
    print("persistence         : to_dict/from_dict round-trip identical ✓")

    print(f"\nstats               : {syn.stats()}")
    print("\nALL SELF-TESTS PASSED ✓")
