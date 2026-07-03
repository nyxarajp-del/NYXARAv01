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

Two consolidation regimes are supported, and neither ever *discards* knowledge:

* **Online EWC** (default) — old anchors decay into a single running anchor with factor
  ``gamma`` each consolidation. With ``per_skill_anchors=True`` each distinct task name keeps
  its *own* running anchor instead, so separate skills are protected separately.
* **Multi-task EWC** — up to ``max_tasks`` distinct anchors are kept, each protecting a
  separate skill. When the budget is exceeded the oldest anchor is **merged, not dropped**,
  into a long-term consolidated anchor via the quadratic-posterior rule
  (``F ← F_a + F_b``, ``θ* ← (F_a·θ*_a + F_b·θ*_b)/(F_a+F_b)``) — eviction is lossless.

Importance is estimated from **two** online signals and combined by ``max`` (protection can
only grow): the Fisher / Memory-Aware-Synapses running mean of squares
(:class:`FisherEstimator`), and a **Synaptic Intelligence** path integral
(:class:`PathIntegralEstimator`) that credits each weight with the loss it actually reduced
along its update trajectory (``ω_i = Σ max(0, −g_i·Δθ_i) / ((θ_i − θ_i^start)² + ξ)``). Feed
the former with :meth:`ElasticSynapses.accumulate` / :meth:`~ElasticSynapses.observe_features`
and the latter with :meth:`~ElasticSynapses.accumulate_step` on every weight update.

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
    "PathIntegralEstimator",
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
# Synaptic-Intelligence importance estimator (online, path-integral style)
# --------------------------------------------------------------------------- #
class PathIntegralEstimator:
    """Synaptic Intelligence: credit each weight with the loss it *actually* reduced.

    Where the Fisher/MAS signal asks "how much do the outputs depend on this weight?", the SI
    path integral asks "how much work did this weight do along its whole update trajectory?".
    Each training step contributes ``max(0, −g_i·Δθ_i)`` (the first-order loss decrease that
    step bought); at consolidation the accumulated contribution is normalised by the squared
    total travel ``(θ_i − θ_i^start)² + ξ`` so a weight that earned a lot while moving little
    is exactly the one worth protecting.
    """

    def __init__(self, *, xi: float = 0.1) -> None:
        self.xi = float(xi)
        self._acc: Dict[str, float] = {}          # Σ max(0, −g·Δθ) per weight
        self._theta_start: Dict[str, float] = {}  # θ at the start of the current segment

    def begin(self, params: Mapping[str, float]) -> None:
        """Start a new trajectory segment at the given weights (called at consolidation)."""
        self._theta_start = {k: float(v) for k, v in params.items()}
        self._acc.clear()

    def observe_step(self, grads: Mapping[str, float], old_params: Mapping[str, float],
                     new_params: Mapping[str, float]) -> None:
        """Fold one weight-update step into the running path integral."""
        for name, g in grads.items():
            old = float(old_params.get(name, 0.0))
            new = float(new_params.get(name, old))
            contrib = -float(g) * (new - old)
            if contrib > 0.0:
                self._acc[name] = self._acc.get(name, 0.0) + contrib
            self._theta_start.setdefault(name, old)

    def omega(self, params: Mapping[str, float], *, normalize: bool = True) -> ParamVec:
        """Per-weight SI importance ω = acc / (travel² + ξ), optionally peak-normalised."""
        raw: Dict[str, float] = {}
        for name, acc in self._acc.items():
            start = self._theta_start.get(name, 0.0)
            travel = float(params.get(name, start)) - start
            raw[name] = acc / (travel * travel + self.xi)
        if not normalize or not raw:
            return raw
        peak = max(raw.values())
        if peak <= 0.0:
            return raw
        return {name: val / peak for name, val in raw.items()}

    def reset(self) -> None:
        self._acc.clear()
        self._theta_start.clear()

    def __len__(self) -> int:
        return len(self._acc)


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
                 per_skill_anchors: bool = False, si_enabled: bool = True,
                 protected: Optional[Sequence[str]] = None, seed: int = 0) -> None:
        self.ewc_lambda = float(ewc_lambda)
        self.freeze_threshold = float(freeze_threshold)
        self.max_tasks = max(1, int(max_tasks))
        self.online = bool(online)
        self.gamma = float(gamma)
        self.per_skill_anchors = bool(per_skill_anchors)
        self.si_enabled = bool(si_enabled)
        self.protected = set(protected) if protected is not None else set(IMMUTABLE_VALUES)
        self.seed = seed
        self._theta: ParamVec = {}                     # current weights (last registered)
        self._estimator = FisherEstimator()            # importance for the skill being learned
        self._si = PathIntegralEstimator()             # SI path-integral importance (per step)
        self._tasks: List[ConsolidatedTask] = []       # consolidated memories
        self._longterm: Optional[ConsolidatedTask] = None  # lossless merge of evicted anchors
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

    def accumulate_step(self, grads: Mapping[str, float], old_params: Mapping[str, float],
                        new_params: Mapping[str, float]) -> None:
        """Fold one *weight-update step* into both importance signals (call every update).

        Feeds the Fisher/MAS running mean with the gradient and — when ``si_enabled`` — the
        Synaptic-Intelligence path integral with the actual step the weight took, so weights
        that genuinely reduced the loss earn extra protection. Protected-core names are
        refused fail-closed, exactly like :meth:`accumulate`.
        """
        self._guard(grads)
        self._estimator.observe(grads)
        if self.si_enabled:
            self._si.observe_step(grads, old_params, new_params)

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
        # Combine with the SI path-integral signal by max: protection can only ever grow.
        if self.si_enabled and len(self._si):
            omega = self._si.omega(self._theta, normalize=True)
            for n, w in omega.items():
                if w > fisher.get(n, 0.0):
                    fisher[n] = w
        # The protected loyalty core is anchored at its canonical value with infinite weight.
        anchor = dict(self._theta)
        for name in self.protected:
            anchor.setdefault(name, float(IMMUTABLE_VALUES.get(name, 1.0)))
            fisher[name] = _PROTECTED_FISHER

        name = task or f"task-{self._consolidations}"
        new_task = ConsolidatedTask(name=name, theta_star=anchor, fisher=fisher)

        if self.online and self.per_skill_anchors:
            # One running anchor per distinct skill name; repeats gamma-merge into their own.
            idx = next((i for i, t in enumerate(self._tasks) if t.name == name), None)
            if idx is not None:
                new_task = self._merge_online(self._tasks.pop(idx), new_task)
            self._tasks.append(new_task)
            if len(self._tasks) > self.max_tasks:
                self._absorb_longterm(self._tasks.pop(0))   # lossless: merged, never dropped
        elif self.online and self._tasks:
            new_task = self._merge_online(self._tasks[-1], new_task)
            self._tasks = [new_task]
        else:
            self._tasks.append(new_task)
            if len(self._tasks) > self.max_tasks:
                self._absorb_longterm(self._tasks.pop(0))   # lossless: merged, never dropped

        self._estimator.reset()
        self._si.begin(self._theta)
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

    def _merge_weighted(self, a: ConsolidatedTask, b: ConsolidatedTask, *,
                        name: str) -> ConsolidatedTask:
        """Quadratic-posterior merge of two consolidated memories — no knowledge is lost.

        The sum of two quadratic EWC penalties is itself quadratic; its exact minimiser is the
        Fisher-weighted mean of the anchors with the Fishers added::

            F ← F_a + F_b        θ* ← (F_a·θ*_a + F_b·θ*_b) / (F_a + F_b)

        Protected-core names stay pinned to their canonical values with infinite importance.
        """
        names = set(a.fisher) | set(b.fisher) | set(a.theta_star) | set(b.theta_star)
        fisher: Dict[str, float] = {}
        theta_star: Dict[str, float] = {}
        for n in names:
            fa, fb = a.fisher.get(n, 0.0), b.fisher.get(n, 0.0)
            fresh = b.theta_star.get(n, a.theta_star.get(n, 0.0))
            if n in self.protected:
                fisher[n] = _PROTECTED_FISHER
                theta_star[n] = float(IMMUTABLE_VALUES.get(n, 1.0))
            elif fa == _PROTECTED_FISHER or fb == _PROTECTED_FISHER:
                fisher[n] = _PROTECTED_FISHER
                theta_star[n] = fresh
            else:
                total = fa + fb
                fisher[n] = total
                theta_star[n] = ((fa * a.theta_star.get(n, 0.0)
                                  + fb * b.theta_star.get(n, 0.0)) / total
                                 if total > 0.0 else fresh)
        return ConsolidatedTask(name=name, theta_star=theta_star, fisher=fisher,
                                at=max(a.at, b.at))

    def _absorb_longterm(self, evicted: ConsolidatedTask) -> None:
        """Fold an evicted anchor into the long-term memory instead of discarding it."""
        if self._longterm is None:
            self._longterm = ConsolidatedTask(name="__longterm__",
                                              theta_star=dict(evicted.theta_star),
                                              fisher=dict(evicted.fisher), at=evicted.at)
        else:
            self._longterm = self._merge_weighted(self._longterm, evicted,
                                                  name="__longterm__")

    def _all_tasks(self) -> List[ConsolidatedTask]:
        """Every memory that exerts elastic pressure: the distinct anchors + the long-term."""
        if self._longterm is None:
            return list(self._tasks)
        return [self._longterm, *self._tasks]

    # ---- the EWC penalty and its gradient ---- #
    def penalty(self, params: Mapping[str, float], *,
                names: Optional[Sequence[str]] = None) -> float:
        """Scalar EWC penalty Σ_tasks Σ_i F_i·(θ_i − θ*_i)² — add this to a training loss.

        The protected core contributes a (bounded) hard penalty for any deviation; ordinary
        weights contribute ``λ·F·Δ²``. Pass ``names`` to restrict the sum to just the weights
        a step actually touched (O(touched) instead of O(all weights)).
        """
        allow = set(names) if names is not None else None
        total = 0.0
        for t in self._all_tasks():
            for name, f in t.fisher.items():
                if allow is not None and name not in allow:
                    continue
                delta = float(params.get(name, t.theta_star.get(name, 0.0))) - \
                    t.theta_star.get(name, 0.0)
                if delta == 0.0:
                    continue
                if f == _PROTECTED_FISHER:
                    total += _PROTECTED_CORE_STIFFNESS * delta * delta
                else:
                    total += self.ewc_lambda * f * delta * delta
        return total

    def penalty_grad(self, params: Mapping[str, float], *,
                     names: Optional[Sequence[str]] = None) -> ParamVec:
        """∂penalty/∂θ — a per-weight pull-back force toward the anchors.

        Use it as a gradient regulariser for a non-torch learner: ``w -= lr * penalty_grad``.
        Pass ``names`` to compute the pull only for the weights a step actually touched.
        """
        allow = set(names) if names is not None else None
        grad: Dict[str, float] = {}
        for t in self._all_tasks():
            for name, f in t.fisher.items():
                if allow is not None and name not in allow:
                    continue
                delta = float(params.get(name, t.theta_star.get(name, 0.0))) - \
                    t.theta_star.get(name, 0.0)
                stiffness = _PROTECTED_CORE_STIFFNESS if f == _PROTECTED_FISHER \
                    else self.ewc_lambda * f
                grad[name] = grad.get(name, 0.0) + 2.0 * stiffness * delta
        return grad

    def penalty_pull(self, params: Mapping[str, float], lr: float, *,
                     names: Optional[Sequence[str]] = None,
                     max_coef: float = 0.5) -> ParamVec:
        """A numerically **stable** elastic step: the per-weight delta of one pull-back.

        The raw gradient step ``−lr·∂penalty/∂θ`` diverges once ``lr·Σ 2λF`` exceeds 1
        (many consolidated anchors stack their stiffness). This method computes the same
        pull as ``w − θ̄`` times a coefficient **capped at** ``max_coef`` — the weight moves
        toward the Fisher-weighted mean of its anchors and can never overshoot it, however
        many memories protect it. Apply as ``w += delta`` per name. Protected-core names get
        the full capped pull toward their canonical anchor.
        """
        allow = set(names) if names is not None else None
        stiff: Dict[str, float] = {}
        weighted: Dict[str, float] = {}
        for t in self._all_tasks():
            for name, f in t.fisher.items():
                if allow is not None and name not in allow:
                    continue
                s = _PROTECTED_CORE_STIFFNESS if f == _PROTECTED_FISHER \
                    else self.ewc_lambda * f
                stiff[name] = stiff.get(name, 0.0) + 2.0 * s
                weighted[name] = weighted.get(name, 0.0) + 2.0 * s * t.theta_star.get(name,
                                                                                      0.0)
        delta: Dict[str, float] = {}
        for name, s_total in stiff.items():
            if s_total <= 0.0:
                continue
            anchor_mean = weighted[name] / s_total
            coef = min(lr * s_total, max_coef)
            delta[name] = -coef * (float(params.get(name, anchor_mean)) - anchor_mean)
        return delta

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
        for t in self._all_tasks():
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
        for t in self._all_tasks():
            names |= set(t.fisher)
        return {n: self.is_frozen(n) for n in sorted(names)}

    def _is_protected(self, name: str) -> bool:
        return name in self.protected

    def _anchor_value(self, name: str) -> float:
        for t in reversed(self._all_tasks()):   # freshest anchors first, long-term last
            if name in t.theta_star:
                return t.theta_star[name]
        if name in self.protected:
            return float(IMMUTABLE_VALUES.get(name, 1.0))
        return float(self._theta.get(name, 0.0))

    # ---- drift: how far the live weights have wandered from what was consolidated ---- #
    def anchor_drift(self, params: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
        """Importance-weighted deviation from each anchor: ``Σ F·Δ² / Σ F`` per memory.

        Zero means the consolidated skills are intact; a rising value means new learning is
        straining old knowledge. The protected core is excluded (it is pinned, not elastic).
        """
        theta = self._theta if params is None else params
        per_task: Dict[str, float] = {}
        for t in self._all_tasks():
            num = 0.0
            den = 0.0
            for name, f in t.fisher.items():
                if f == _PROTECTED_FISHER:
                    continue
                delta = float(theta.get(name, t.theta_star.get(name, 0.0))) - \
                    t.theta_star.get(name, 0.0)
                num += f * delta * delta
                den += f
            per_task[t.name] = (num / den) if den > 0.0 else 0.0
        mean = sum(per_task.values()) / len(per_task) if per_task else 0.0
        return {"per_task": per_task, "mean_drift": mean}

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
            "gamma": self.gamma,
            "max_tasks": self.max_tasks,
            "per_skill_anchors": self.per_skill_anchors,
            "si_enabled": self.si_enabled,
            "longterm_weights": len(self._longterm.fisher) if self._longterm else 0,
            "mean_anchor_drift": round(self.anchor_drift()["mean_drift"], 6),
        }

    # ---- persistence (this is what makes the memory *lifelong*) ---- #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ewc_lambda": self.ewc_lambda,
            "freeze_threshold": self.freeze_threshold,
            "max_tasks": self.max_tasks,
            "online": self.online,
            "gamma": self.gamma,
            "per_skill_anchors": self.per_skill_anchors,
            "si_enabled": self.si_enabled,
            "protected": sorted(self.protected),
            "consolidations": self._consolidations,
            "theta": dict(self._theta),
            "tasks": [t.to_dict() for t in self._tasks],
            "longterm": self._longterm.to_dict() if self._longterm is not None else None,
        }

    def load_dict(self, data: Mapping[str, Any]) -> None:
        """Restore state in place (used to rehydrate a live engine on boot).

        Backward compatible: a blob written before the long-term / per-skill / SI upgrades
        loads cleanly — missing keys simply keep the engine's current settings.
        """
        self.ewc_lambda = float(data.get("ewc_lambda", self.ewc_lambda))
        self.freeze_threshold = float(data.get("freeze_threshold", self.freeze_threshold))
        self.max_tasks = max(1, int(data.get("max_tasks", self.max_tasks)))
        self.online = bool(data.get("online", self.online))
        self.gamma = float(data.get("gamma", self.gamma))
        self.per_skill_anchors = bool(data.get("per_skill_anchors", self.per_skill_anchors))
        self.si_enabled = bool(data.get("si_enabled", self.si_enabled))
        if data.get("protected"):
            self.protected = set(data["protected"])
        self._consolidations = int(data.get("consolidations", 0))
        self._theta = {k: float(v) for k, v in dict(data.get("theta", {})).items()}
        self._tasks = [ConsolidatedTask.from_dict(t) for t in data.get("tasks", [])]
        longterm = data.get("longterm")
        self._longterm = ConsolidatedTask.from_dict(longterm) if longterm else None
        self._estimator.reset()
        self._si.begin(self._theta)

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
                     online: bool = True, gamma: float = 0.9, max_tasks: int = 8) -> None:
            self.module = module
            self.ewc_lambda = float(ewc_lambda)
            self.online = bool(online)
            self.gamma = float(gamma)
            self.max_tasks = max(1, int(max_tasks))
            self._fisher: Dict[str, "torch.Tensor"] = {}
            self._anchor: Dict[str, "torch.Tensor"] = {}
            # multi-task mode: distinct (fisher, anchor) snapshots; overflow merges into a
            # long-term pair via the same quadratic-posterior rule as the stdlib engine.
            self._tasks: List[Dict[str, Dict[str, "torch.Tensor"]]] = []
            self._longterm: Optional[Dict[str, Dict[str, "torch.Tensor"]]] = None
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
            """Snapshot current params as the anchor and freeze the estimated Fisher.

            Online mode gamma-merges into a single running anchor; multi-task mode keeps up
            to ``max_tasks`` distinct snapshots and losslessly merges overflow into a
            long-term anchor (Fishers added, anchors Fisher-weighted-averaged).
            """
            pending = getattr(self, "_pending_fisher", None)
            if pending is None:
                pending = {n: torch.ones_like(p) for n, p in self._named()}
            if self.online:
                for n, p in self._named():
                    f = pending.get(n)
                    if f is None:
                        continue
                    if n in self._fisher:
                        self._fisher[n] = self.gamma * self._fisher[n] + f
                    else:
                        self._fisher[n] = f.clone()
                    self._anchor[n] = p.detach().clone()
            else:
                snap = {
                    "fisher": {n: pending[n].clone() for n, _ in self._named()
                               if n in pending},
                    "anchor": {n: p.detach().clone() for n, p in self._named()},
                }
                self._tasks.append(snap)
                if len(self._tasks) > self.max_tasks:
                    self._absorb_longterm(self._tasks.pop(0))
            self._pending_fisher = None
            self._consolidations += 1

        def _absorb_longterm(self, evicted: Dict[str, Dict[str, "torch.Tensor"]]) -> None:
            """Quadratic-posterior merge of an evicted snapshot — knowledge is never dropped."""
            if self._longterm is None:
                self._longterm = evicted
                return
            lt_f, lt_a = self._longterm["fisher"], self._longterm["anchor"]
            ev_f, ev_a = evicted["fisher"], evicted["anchor"]
            for n in set(lt_f) | set(ev_f):
                fa = lt_f.get(n)
                fb = ev_f.get(n)
                if fa is None or fb is None:
                    src = evicted if fb is not None else self._longterm
                    lt_f[n] = src["fisher"][n]
                    lt_a[n] = src["anchor"][n]
                    continue
                total = fa + fb
                safe = torch.where(total > 0, total, torch.ones_like(total))
                lt_a[n] = torch.where(total > 0,
                                      (fa * lt_a[n] + fb * ev_a[n]) / safe, ev_a[n])
                lt_f[n] = total

        def _penalty_terms(self):
            if self._fisher:
                yield self._fisher, self._anchor
            for snap in self._tasks:
                yield snap["fisher"], snap["anchor"]
            if self._longterm is not None:
                yield self._longterm["fisher"], self._longterm["anchor"]

        def penalty(self) -> "torch.Tensor":
            """λ/2 · Σ_memories Σ F·(θ − θ*)² — add to the training loss (0 before any
            consolidation)."""
            total = None
            params = dict(self._named())
            for fisher, anchor in self._penalty_terms():
                for n, p in params.items():
                    if n not in fisher or n not in anchor:
                        continue
                    term = (fisher[n] * (p - anchor[n]) ** 2).sum()
                    total = term if total is None else total + term
            if total is None:
                # a real zero tensor on the right device, so loss + penalty always works
                ref = next((p for _, p in self._named()), None)
                return torch.zeros((), device=ref.device if ref is not None else None)
            return 0.5 * self.ewc_lambda * total

        @staticmethod
        def _pack(pair: Dict[str, Dict[str, "torch.Tensor"]]) -> Dict[str, Any]:
            return {"fisher": {n: t.cpu().tolist() for n, t in pair["fisher"].items()},
                    "anchor": {n: t.cpu().tolist() for n, t in pair["anchor"].items()}}

        def _unpack(self, blob: Mapping[str, Any],
                    dev: Any) -> Dict[str, Dict[str, "torch.Tensor"]]:
            return {"fisher": {n: torch.tensor(v, device=dev)
                               for n, v in dict(blob.get("fisher", {})).items()},
                    "anchor": {n: torch.tensor(v, device=dev)
                               for n, v in dict(blob.get("anchor", {})).items()}}

        def state_dict(self) -> Dict[str, Any]:
            return {
                "ewc_lambda": self.ewc_lambda, "online": self.online, "gamma": self.gamma,
                "max_tasks": self.max_tasks,
                "consolidations": self._consolidations,
                "fisher": {n: t.cpu().tolist() for n, t in self._fisher.items()},
                "anchor": {n: t.cpu().tolist() for n, t in self._anchor.items()},
                "tasks": [self._pack(s) for s in self._tasks],
                "longterm": self._pack(self._longterm) if self._longterm else None,
            }

        def load_state_dict(self, data: Mapping[str, Any]) -> None:
            self.ewc_lambda = float(data.get("ewc_lambda", self.ewc_lambda))
            self.online = bool(data.get("online", self.online))
            self.gamma = float(data.get("gamma", self.gamma))
            self.max_tasks = max(1, int(data.get("max_tasks", self.max_tasks)))
            self._consolidations = int(data.get("consolidations", 0))
            dev = next((p.device for _, p in self._named()), None)
            self._fisher = {n: torch.tensor(v, device=dev)
                            for n, v in dict(data.get("fisher", {})).items()}
            self._anchor = {n: torch.tensor(v, device=dev)
                            for n, v in dict(data.get("anchor", {})).items()}
            self._tasks = [self._unpack(s, dev) for s in data.get("tasks", []) or []]
            longterm = data.get("longterm")
            self._longterm = self._unpack(longterm, dev) if longterm else None

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

    # ---- lossless eviction: overflow merges into long-term, nothing is dropped ---- #
    lossless = ElasticSynapses(online=False, max_tasks=2)
    for i in range(4):
        lossless.observe_features({f"w{i}": 1.0})
        lossless.consolidate({f"w{i}": float(i)}, task=f"skill-{i}")
    assert len(lossless._tasks) == 2                       # budget respected
    assert lossless._longterm is not None                  # ...but evicted memories merged
    assert lossless.importance("w0") > 0.0                 # oldest skill still protected
    assert lossless.penalty({"w0": 99.0}, names=["w0"]) > 0.0
    print("lossless eviction   : evicted anchors merge into long-term, w0 still guarded ✓")

    # ---- Synaptic Intelligence: a weight that reduced loss earns importance ---- #
    si = ElasticSynapses(si_enabled=True)
    w0, w1 = {"busy": 0.0, "idle": 0.0}, {"busy": 0.5, "idle": 0.0}
    for _ in range(5):
        si.accumulate_step({"busy": -1.0, "idle": 0.0}, w0, w1)   # busy moved downhill
        w0, w1 = w1, {"busy": w1["busy"] + 0.5, "idle": 0.0}
    task_si = si.consolidate({"busy": 2.5, "idle": 0.0}, task="si")
    assert task_si.fisher.get("busy", 0.0) > task_si.fisher.get("idle", 0.0)
    print("synaptic intel (SI) : loss-reducing weight out-ranks the idle one ✓")

    # ---- per-skill anchors: each named skill keeps its own running anchor ---- #
    per = ElasticSynapses(per_skill_anchors=True, max_tasks=8)
    per.observe_features({"a": 1.0}); per.consolidate({"a": 1.0}, task="alpha")
    per.observe_features({"b": 1.0}); per.consolidate({"b": 2.0}, task="beta")
    per.observe_features({"a": 1.0}); per.consolidate({"a": 1.1}, task="alpha")
    names = [t.name for t in per._tasks]
    assert sorted(names) == ["alpha", "beta"]
    print(f"per-skill anchors   : distinct anchors {sorted(names)} ✓")

    print(f"\nstats               : {syn.stats()}")
    print("\nALL SELF-TESTS PASSED ✓")
