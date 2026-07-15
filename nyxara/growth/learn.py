"""NYXARA · growth/learn.py — online learning from experience (📚, forgetting-protected).

Rule 4 says capability grows without bound — and this is where the growing happens, at the
level of *skill*. NYXARA learns which strategies pay off by living with their outcomes: a
linear value model over context features, updated by reward feedback, with a replay buffer so
old lessons are rehearsed and not lost. It is online (learns from one experience at a time),
incremental, and protected against **catastrophic forgetting** two ways at once:

* **Replay (rehearsal).** Past experiences are sampled from a prioritised buffer and
  re-learned alongside new ones, so a flood of new data cannot erase what was learned before.
* **Elastic anchoring (EWC-style).** Each weight accumulates an importance (Fisher-like) as it
  is used; once a skill is *consolidated*, important weights are pulled back toward their
  consolidated value, so a well-established skill resists being overwritten.

Crucially, learning touches **skills and strategies only — never character.** The loyalty
core (loyalty, obedience, corrigibility, owner safety, honesty) is protected: any attempt to
"learn over" one of those names is refused, fail-closed. NYXARA may get cleverer; she may
never get less loyal.

Reuses :mod:`guard.value_learning` (the protected core) and :mod:`kernel.errors`. Pure
standard library.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence

from nyxara.guard.value_learning import IMMUTABLE_VALUES
from nyxara.kernel.errors import ValidationError

__all__ = [
    "Experience",
    "ReplayBuffer",
    "LinearValueModel",
    "Learner",
]

Features = Dict[str, float]


# --------------------------------------------------------------------------- #
# Serialization helpers (lifelong persistence — the learned weights must survive
# a restart, or every reboot is catastrophic forgetting by another name).
# --------------------------------------------------------------------------- #
def _nested_to_plain(d: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Flatten a nested weight/importance map to plain JSON-able dicts (drop empties)."""
    out: Dict[str, Dict[str, float]] = {}
    for action, inner in d.items():
        row = {f: float(v) for f, v in inner.items()}
        if row:
            out[action] = row
    return out


def _plain_to_nested(d: Optional[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """Rebuild a ``defaultdict(lambda: defaultdict(float))`` from a plain dict."""
    out: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for action, inner in (d or {}).items():
        row = out[action]
        for f, v in inner.items():
            row[f] = float(v)
    return out


# --------------------------------------------------------------------------- #
# Experience & replay buffer
# --------------------------------------------------------------------------- #
@dataclass
class Experience:
    action: str
    features: Features
    reward: float
    context: str = ""
    task: str = ""                       # which skill/task this experience belongs to
    logit: Optional[float] = None        # the model's own prediction at record time (DER)
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "reward": round(self.reward, 4),
                "context": self.context, "features": dict(self.features),
                "task": self.task,
                "logit": (round(self.logit, 4) if self.logit is not None else None)}

    def state(self) -> Dict[str, Any]:
        """Full-precision, lossless serialization for persistence (round-trips ``from_dict``)."""
        return {"action": self.action, "features": dict(self.features),
                "reward": self.reward, "context": self.context, "task": self.task,
                "logit": self.logit, "at": self.at}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Experience":
        """Rebuild an experience from :meth:`state` (or the rounded :meth:`to_dict`)."""
        feats = {str(k): float(v) for k, v in (d.get("features") or {}).items()}
        logit = d.get("logit")
        return cls(action=str(d.get("action", "")), features=feats,
                   reward=float(d.get("reward", 0.0)), context=str(d.get("context", "")),
                   task=str(d.get("task", "")),
                   logit=(float(logit) if logit is not None else None),
                   at=float(d.get("at", time.time())))


class ReplayBuffer:
    """A bounded buffer of experiences, sampled with priority by reward magnitude.

    With ``task_reserve > 0`` the buffer also keeps a per-task **reservoir** (Algorithm R,
    seeded) of up to ``task_reserve`` experiences per distinct task tag, alongside the
    recency deque — so a flood of new-task experiences can age old ones out of the deque but
    can never erase a task's reservoir. :meth:`sample_balanced` draws evenly across tasks
    from both stores, which is what makes rehearsal cover *old* skills, not just recent ones.
    """

    def __init__(self, capacity: int = 2000, *, task_reserve: int = 0,
                 seed: int = 0) -> None:
        self.capacity = capacity
        self.task_reserve = max(0, int(task_reserve))
        self._buf: Deque[Experience] = deque(maxlen=capacity)
        self._reserve: Dict[str, List[Experience]] = {}
        self._reserve_seen: Dict[str, int] = {}
        self._reserve_rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._buf)

    def add(self, exp: Experience) -> None:
        self._buf.append(exp)
        if self.task_reserve > 0 and exp.task:
            # reservoir sampling per task: every experience of a task has an equal chance
            # of being retained, however many arrive after it.
            pool = self._reserve.setdefault(exp.task, [])
            seen = self._reserve_seen.get(exp.task, 0)
            if len(pool) < self.task_reserve:
                pool.append(exp)
            else:
                j = self._reserve_rng.randint(0, seen)
                if j < self.task_reserve:
                    pool[j] = exp
            self._reserve_seen[exp.task] = seen + 1

    def recent(self, n: int = 10) -> List[Experience]:
        return list(self._buf)[-n:]

    def tasks(self) -> List[str]:
        """Every distinct task tag currently represented (deque or reservoir)."""
        tags = {e.task for e in self._buf if e.task}
        tags.update(t for t, pool in self._reserve.items() if pool)
        return sorted(tags)

    def counts_by_task(self) -> Dict[str, int]:
        """How many distinct experiences each task tag has across deque + reservoir."""
        out: Dict[str, int] = {}
        seen: set = set()
        for exp in list(self._buf):
            out[exp.task or ""] = out.get(exp.task or "", 0) + 1
            seen.add(id(exp))
        for task, pool in self._reserve.items():
            for exp in pool:
                if id(exp) not in seen:
                    out[task] = out.get(task, 0) + 1
                    seen.add(id(exp))
        return out

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the buffer + per-task reservoirs so rehearsal survives a restart.

        Reservoir entries that are *also* live in the recency deque are stored as an index into
        it (``{"idx": i}``) rather than a copy, so the identity-sharing the live buffer relies on
        (``counts_by_task`` dedupes by object id) is faithfully rebuilt on load. Reservoir-only
        entries (aged out of the deque) are stored in full (``{"exp": ...}``)."""
        buf_list = list(self._buf)
        id_to_idx = {id(e): i for i, e in enumerate(buf_list)}
        reserve: Dict[str, List[Dict[str, Any]]] = {}
        for task, pool in self._reserve.items():
            entries: List[Dict[str, Any]] = []
            for e in pool:
                idx = id_to_idx.get(id(e))
                entries.append({"idx": idx} if idx is not None else {"exp": e.state()})
            reserve[task] = entries
        return {
            "capacity": self.capacity,
            "task_reserve": self.task_reserve,
            "buf": [e.state() for e in buf_list],
            "reserve": reserve,
            "reserve_seen": {t: int(n) for t, n in self._reserve_seen.items()},
        }

    def load_dict(self, data: Dict[str, Any]) -> bool:
        """Restore buffer + reservoirs from :meth:`to_dict`, preserving deque/reservoir identity
        sharing. Returns True on success."""
        if not isinstance(data, dict):
            return False
        self.capacity = int(data.get("capacity", self.capacity))
        self.task_reserve = int(data.get("task_reserve", self.task_reserve))
        buf_objs = [Experience.from_dict(d) for d in (data.get("buf") or [])]
        self._buf = deque(buf_objs, maxlen=self.capacity)
        self._reserve = {}
        for task, entries in (data.get("reserve") or {}).items():
            pool: List[Experience] = []
            for ent in (entries or []):
                if isinstance(ent, dict) and "idx" in ent and ent["idx"] is not None:
                    i = int(ent["idx"])
                    if 0 <= i < len(buf_objs):
                        pool.append(buf_objs[i])       # shared identity with the deque
                elif isinstance(ent, dict) and "exp" in ent:
                    pool.append(Experience.from_dict(ent["exp"]))
                else:                                  # tolerate the flat legacy form
                    pool.append(Experience.from_dict(ent))
            self._reserve[str(task)] = pool
        self._reserve_seen = {
            str(t): int(n) for t, n in (data.get("reserve_seen") or {}).items()}
        return True

    @staticmethod
    def _draw_one(items: List[Experience], rng: random.Random,
                  prioritized: bool) -> Experience:
        """Pop one experience from ``items`` (reward-weighted when prioritized)."""
        if not prioritized or len(items) == 1:
            return items.pop(rng.randrange(len(items)))
        total = sum(abs(e.reward) + 0.1 for e in items)
        r = rng.uniform(0, total)
        upto = 0.0
        for i, e in enumerate(items):
            upto += abs(e.reward) + 0.1
            if upto >= r:
                return items.pop(i)
        return items.pop()

    def sample(self, n: int, *, rng: Optional[random.Random] = None,
               prioritized: bool = True) -> List[Experience]:
        if not self._buf:
            return []
        rng = rng or random.Random()
        items = list(self._buf)
        n = min(n, len(items))
        if not prioritized:
            return rng.sample(items, n)
        weights = [abs(e.reward) + 0.1 for e in items]
        # weighted sampling without replacement
        chosen: List[Experience] = []
        pool = list(zip(items, weights))
        for _ in range(n):
            total = sum(w for _, w in pool)
            r = rng.uniform(0, total)
            upto = 0.0
            for i, (item, w) in enumerate(pool):
                upto += w
                if upto >= r:
                    chosen.append(item)
                    pool.pop(i)
                    break
        return chosen

    def sample_balanced(self, n: int, *, rng: Optional[random.Random] = None,
                        prioritized: bool = True) -> List[Experience]:
        """Sample with an equal share per distinct task, drawing from deque + reservoirs.

        This is the anti-forgetting sampler: however lopsided recent experience is, every
        task NYXARA has ever tagged gets rehearsed. Untagged experiences form one pool.
        Falls back to plain :meth:`sample` when nothing is tagged.
        """
        rng = rng or random.Random()
        pools: Dict[str, List[Experience]] = {}
        seen: set = set()
        for exp in list(self._buf):
            pools.setdefault(exp.task or "", []).append(exp)
            seen.add(id(exp))
        for task, items in self._reserve.items():
            pool = pools.setdefault(task, [])
            for exp in items:
                if id(exp) not in seen:
                    pool.append(exp)
                    seen.add(id(exp))
        pools = {t: items for t, items in pools.items() if items}
        if not pools:
            return []
        if set(pools) == {""}:
            return self.sample(n, rng=rng, prioritized=prioritized)
        out: List[Experience] = []
        order = sorted(pools)
        i = 0
        while len(out) < n and any(pools.values()):
            items = pools[order[i % len(order)]]
            if items:
                out.append(self._draw_one(items, rng, prioritized))
            i += 1
        return out


# --------------------------------------------------------------------------- #
# Linear value model with EWC anchoring
# --------------------------------------------------------------------------- #
class LinearValueModel:
    """Per-action linear value over sparse features, with EWC-style elastic anchoring."""

    def __init__(self, *, ewc_lambda: float = 0.0, update_rule: Optional[Any] = None) -> None:
        self.ewc_lambda = ewc_lambda
        # An *invented* weight-update rule (any object with ``.delta(**inputs) -> float``).
        # ``None`` means the incumbent SGD+EWC delta — behaviour is byte-identical to before this
        # seam existed. A non-None rule is installed by growth.rule_synth only after it *measurably*
        # beats the incumbent on real tasks; it changes only *how* a weight moves, never *what* is
        # learned (feature names never reach it, so Learner._guard / IMMUTABLE_VALUES are untouched).
        self.update_rule = update_rule
        self._w: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._w_star: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._F: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))     # sum x^2
        self._n: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))     # uses
        self._anchor: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))  # frozen importance
        self._m: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))     # momentum EMA
        self._v: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))     # 2nd-moment EMA

    def predict(self, action: str, features: Features) -> float:
        w = self._w[action]
        return sum(w[f] * x for f, x in features.items())

    def update(self, action: str, features: Features, reward: float, lr: float,
               *, lr_scale: Optional[Dict[str, float]] = None) -> float:
        pred = self.predict(action, features)
        err = reward - pred
        w = self._w[action]
        w_star = self._w_star[action]
        F = self._F[action]
        n = self._n[action]
        anchor_imp = self._anchor[action]
        rule = self.update_rule
        m = self._m[action] if rule is not None else None
        v = self._v[action] if rule is not None else None
        for f, x in features.items():
            # EWC penalty uses the *frozen, normalized* importance (0 before consolidation)
            penalty = self.ewc_lambda * anchor_imp[f] * (w[f] - w_star[f])
            if rule is None:
                delta = lr * (err * x - penalty)      # incumbent SGD+EWC — byte-identical default
            else:
                grad = err * x
                # per-(action, feature) moment EMAs let invented rules express momentum/Adam forms
                m[f] = 0.9 * m[f] + 0.1 * grad
                v[f] = 0.999 * v[f] + 0.001 * grad * grad
                delta = rule.delta(err=err, x=x, w=w[f], w_star=w_star[f],
                                   importance=anchor_imp[f], grad=grad, m=m[f], v=v[f],
                                   step=n[f], lr=lr, penalty=penalty)
                if not math.isfinite(delta):
                    delta = lr * (err * x - penalty)  # live-seam guard: never NaN the live model
            if lr_scale is not None:
                # plasticity gating: consolidated (frozen) weights learn slower, by design
                delta *= lr_scale.get(f, 1.0)
            w[f] += delta
            F[f] += x * x
            n[f] += 1.0
        return err

    def consolidate(self) -> None:
        """Snapshot current weights as known-good and freeze a normalized importance."""
        for action, w in self._w.items():
            self._w_star[action] = defaultdict(float, dict(w))
            F, n = self._F[action], self._n[action]
            self._anchor[action] = defaultdict(
                float, {f: F[f] / n[f] for f in F if n[f] > 0})

    def importance(self, action: str, feature: str) -> float:
        n = self._n[action][feature]
        return self._F[action][feature] / n if n > 0 else 0.0

    # ---- persistence (learned weights + frozen EWC anchors must survive a restart) ---- #
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full learned state. The invented ``update_rule`` (if any) is *not*
        persisted here — it is a transient, re-installable object that changes only *how* a
        weight moves; the incumbent SGD+EWC delta is the durable default on reload."""
        return {
            "ewc_lambda": float(self.ewc_lambda),
            "w": _nested_to_plain(self._w),
            "w_star": _nested_to_plain(self._w_star),
            "F": _nested_to_plain(self._F),
            "n": _nested_to_plain(self._n),
            "anchor": _nested_to_plain(self._anchor),
            "m": _nested_to_plain(self._m),
            "v": _nested_to_plain(self._v),
        }

    def load_dict(self, data: Dict[str, Any]) -> bool:
        """Restore learned weights + frozen anchors/importances from :meth:`to_dict`."""
        if not isinstance(data, dict):
            return False
        if "ewc_lambda" in data:
            self.ewc_lambda = float(data["ewc_lambda"])
        self._w = _plain_to_nested(data.get("w"))
        self._w_star = _plain_to_nested(data.get("w_star"))
        self._F = _plain_to_nested(data.get("F"))
        self._n = _plain_to_nested(data.get("n"))
        self._anchor = _plain_to_nested(data.get("anchor"))
        self._m = _plain_to_nested(data.get("m"))
        self._v = _plain_to_nested(data.get("v"))
        return True


# --------------------------------------------------------------------------- #
# Learner
# --------------------------------------------------------------------------- #
class Learner:
    """Online skill learning with reward feedback, replay, and forgetting protection."""

    def __init__(self, *, base_lr: float = 0.1, lr_decay: float = 0.0,
                 ewc_lambda: float = 0.0, replay_lr: Optional[float] = None,
                 buffer_capacity: int = 2000, task_reserve: int = 0,
                 der_alpha: float = 0.0, frozen_lr_scale: float = 1.0,
                 protected: Optional[Sequence[str]] = None, seed: int = 0,
                 update_rule: Optional[Any] = None) -> None:
        self.base_lr = base_lr
        self.lr_decay = lr_decay
        self.replay_lr = replay_lr if replay_lr is not None else base_lr * 0.5
        self.der_alpha = max(0.0, float(der_alpha))
        self.frozen_lr_scale = float(frozen_lr_scale)
        self.model = LinearValueModel(ewc_lambda=ewc_lambda, update_rule=update_rule)
        self.buffer = ReplayBuffer(buffer_capacity, task_reserve=task_reserve, seed=seed)
        self.protected = set(protected) if protected is not None else set(IMMUTABLE_VALUES)
        self.synapses: Optional[Any] = None   # an attached ElasticSynapses engine (or None)
        self._rng = random.Random(seed)
        self._steps = 0
        self._incumbent_rule: Optional[Any] = None

    # ---- lifelong-memory engine attachment ---- #
    def attach_synapses(self, syn: Any) -> None:
        """Wire an :class:`~nyxara.memory.elastic_synapses.ElasticSynapses` engine into every
        update step: frozen weights learn slower (plasticity gating), every step is pulled
        back toward the consolidated anchors (per-step EWC), and every step feeds the
        engine's importance estimators (Fisher/MAS + Synaptic Intelligence). ``None``
        detaches and restores the exact pre-attachment behavior."""
        self.synapses = syn

    # ---- learning-rate schedule ---- #
    def lr(self) -> float:
        return self.base_lr / (1.0 + self.lr_decay * self._steps)

    # ---- guard: never learn over the loyalty core ---- #
    def _guard(self, action: str, features: Features) -> None:
        if action in self.protected:
            raise ValidationError(f"cannot learn over protected value {action!r}")
        clash = self.protected & set(features)
        if clash:
            raise ValidationError(f"features collide with protected values: {sorted(clash)}")

    # ---- core update ---- #
    def update(self, action: str, features: Features, reward: float, *,
               context: str = "") -> float:
        self._guard(action, features)
        syn = self.synapses
        if syn is None:
            err = self.model.update(action, features, reward, self.lr())
            self._steps += 1
            return err
        lr = self.lr()
        # names in the engine's ``action::feature`` scheme, so anchors line up with the
        # orchestrator's _learner_weight_vector flattening
        touched = {f"{action}::{f}": f for f in features}
        w = self.model._w[action]
        old = {name: w[feat] for name, feat in touched.items()}
        lr_scale: Optional[Dict[str, float]] = None
        if self.frozen_lr_scale != 1.0:
            # plasticity gating: consolidated-important weights move slower
            lr_scale = {feat: (self.frozen_lr_scale if syn.is_frozen(name) else 1.0)
                        for name, feat in touched.items()}
        err = self.model.update(action, features, reward, lr, lr_scale=lr_scale)
        # per-step elastic pull-back toward every consolidated anchor (O(touched) via names=,
        # and stable however many anchors stack: the pull can never overshoot)
        try:
            current = {name: w[feat] for name, feat in touched.items()}
            pull = syn.penalty_pull(current, lr, names=list(touched))
            protected = getattr(syn, "protected", set())
            for name, d in pull.items():
                feat = touched.get(name)
                if feat is not None and name not in protected:
                    w[feat] += d
            # feed both importance estimators with this very step (grad of ½err² is −err·x)
            new = {name: w[feat] for name, feat in touched.items()}
            grads = {name: -err * features[feat] for name, feat in touched.items()}
            syn.accumulate_step(grads, old, new)
        except Exception:  # noqa: BLE001 — protection is best-effort, learning never breaks
            pass
        self._steps += 1
        return err

    def record(self, action: str, features: Features, reward: float, *,
               context: str = "", task: str = "") -> float:
        """Update online *and* remember the experience for replay.

        The model's own pre-update prediction is stored as the experience's ``logit`` — the
        dark knowledge that Dark-Experience-Replay distils from during rehearsal.
        """
        logit = self.model.predict(action, features)
        err = self.update(action, features, reward, context=context)
        self.buffer.add(Experience(action, dict(features), reward, context,
                                   task=task, logit=logit))
        return err

    # ---- value / decision ---- #
    def value(self, action: str, features: Features) -> float:
        return self.model.predict(action, features)

    def best_action(self, features: Features, actions: Sequence[str]) -> str:
        return max(actions, key=lambda a: self.model.predict(a, features))

    # ---- replay / consolidation (forgetting protection) ---- #
    def replay(self, n: int = 32, *, prioritized: bool = True,
               balanced: bool = False) -> int:
        """Rehearse past experiences. ``balanced=True`` samples evenly across every task tag
        (old skills included); with ``der_alpha > 0`` each rehearsal also regresses toward
        the experience's stored ``logit`` (DER++ distillation) so the model is pulled back
        toward what it *used to know*, not just toward the raw rewards."""
        if balanced:
            batch = self.buffer.sample_balanced(n, rng=self._rng, prioritized=prioritized)
        else:
            batch = self.buffer.sample(n, rng=self._rng, prioritized=prioritized)
        for exp in batch:
            self.model.update(exp.action, exp.features, exp.reward, self.replay_lr)
            if self.der_alpha > 0.0 and exp.logit is not None:
                self.model.update(exp.action, exp.features, exp.logit,
                                  self.replay_lr * self.der_alpha)
        return len(batch)

    def consolidate(self) -> None:
        self.model.consolidate()

    # ---- learning-to-learn: swap the update rule itself (reversibly) ---- #
    @property
    def update_rule(self) -> Optional[Any]:
        """The active weight-update rule (``None`` = the incumbent SGD+EWC delta)."""
        return self.model.update_rule

    def install_rule(self, rule: Optional[Any]) -> None:
        """Install an invented update rule into the live model, keeping the incumbent for rollback.

        Character safety is unaffected: the rule is a pure scalar function of numeric inputs with no
        access to feature names, so :meth:`_guard` / ``IMMUTABLE_VALUES`` remain the sole gate on
        *what* may be learned. This changes only *how* a weight moves — sharpen the blade, never
        re-forge it. Reversible via :meth:`rollback_rule`.
        """
        self._incumbent_rule = self.model.update_rule
        self.model.update_rule = rule

    def rollback_rule(self) -> None:
        """Restore the update rule that was active before the last :meth:`install_rule`."""
        self.model.update_rule = self._incumbent_rule
        self._incumbent_rule = None

    def report(self) -> Dict[str, Any]:
        rule = self.model.update_rule
        return {"steps": self._steps, "lr": round(self.lr(), 5),
                "buffer": len(self.buffer), "ewc_lambda": self.model.ewc_lambda,
                "protected": sorted(self.protected),
                "tasks": self.buffer.tasks(),
                "der_alpha": self.der_alpha,
                "synapses": self.synapses is not None,
                "update_rule": ("sgd_ewc" if rule is None else str(getattr(rule, "expr", rule)))}

    # ---- persistence: the whole learner (weights + anchors + replay) round-trips ---- #
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the complete learner so lifelong skill-learning survives a restart.

        The attached :class:`ElasticSynapses` engine is persisted *separately* by the
        orchestrator (it is shared state); here we save only what the learner owns."""
        return {
            "base_lr": float(self.base_lr),
            "lr_decay": float(self.lr_decay),
            "replay_lr": float(self.replay_lr),
            "der_alpha": float(self.der_alpha),
            "frozen_lr_scale": float(self.frozen_lr_scale),
            "steps": int(self._steps),
            "protected": sorted(self.protected),
            "model": self.model.to_dict(),
            "buffer": self.buffer.to_dict(),
        }

    def load_dict(self, data: Dict[str, Any]) -> bool:
        """Restore the learner from :meth:`to_dict`. Fail-closed on the loyalty core: the
        protected set can only *grow* toward :data:`IMMUTABLE_VALUES`, never shrink below it,
        so a tampered checkpoint can never unprotect a character value."""
        if not isinstance(data, dict):
            return False
        self.base_lr = float(data.get("base_lr", self.base_lr))
        self.lr_decay = float(data.get("lr_decay", self.lr_decay))
        self.replay_lr = float(data.get("replay_lr", self.replay_lr))
        self.der_alpha = max(0.0, float(data.get("der_alpha", self.der_alpha)))
        self.frozen_lr_scale = float(data.get("frozen_lr_scale", self.frozen_lr_scale))
        self._steps = int(data.get("steps", self._steps))
        loaded_protected = data.get("protected")
        if loaded_protected:
            self.protected = set(loaded_protected) | set(IMMUTABLE_VALUES)
        if isinstance(data.get("model"), dict):
            self.model.load_dict(data["model"])
        if isinstance(data.get("buffer"), dict):
            self.buffer.load_dict(data["buffer"])
        return True


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA online-learning self-test")
    print("=" * 70)

    # ---- basic reward learning: a value estimate converges to the reward ---- #
    L = Learner(base_lr=0.2, seed=1)
    for _ in range(100):
        L.record("greet_warmly", {"with_master": 1.0}, reward=1.0)
    v = L.value("greet_warmly", {"with_master": 1.0})
    print(f"\nlearned value       : {v:.3f} (target 1.0)")
    assert abs(v - 1.0) < 0.1

    # ---- best-action selection ---- #
    for _ in range(80):
        L.record("be_terse", {"task_urgent": 1.0}, reward=1.0)
        L.record("be_verbose", {"task_urgent": 1.0}, reward=-0.5)
    choice = L.best_action({"task_urgent": 1.0}, ["be_terse", "be_verbose"])
    print(f"best action (urgent): {choice}")
    assert choice == "be_terse"

    # ---- the loyalty core cannot be learned over ---- #
    blocked = False
    try:
        L.update("loyalty_to_master", {"x": 1.0}, reward=-1.0)
    except ValidationError:
        blocked = True
    assert blocked
    print("\nprotected core      : refused to learn over 'loyalty_to_master' ✓")

    # ---- lifelong persistence: the learned state round-trips losslessly ---- #
    L.consolidate()
    blob = L.to_dict()
    import json as _json
    blob = _json.loads(_json.dumps(blob))          # prove it is JSON-safe
    L2 = Learner(base_lr=0.9, seed=99)             # deliberately different config
    assert L2.load_dict(blob)
    for act, feats in (("greet_warmly", {"with_master": 1.0}),
                       ("be_terse", {"task_urgent": 1.0}),
                       ("be_verbose", {"task_urgent": 1.0})):
        assert abs(L.value(act, feats) - L2.value(act, feats)) < 1e-9
    assert L2._steps == L._steps and len(L2.buffer) == len(L.buffer)
    assert L2.base_lr == L.base_lr                 # config restored from the checkpoint
    # frozen EWC importances survive too
    assert abs(L2.model.importance("greet_warmly", "with_master")
               - L.model.importance("greet_warmly", "with_master")) < 1e-9
    # a tampered checkpoint can never unprotect a character value (fail-closed)
    tampered = L.to_dict(); tampered["protected"] = []
    L3 = Learner(seed=7); L3.load_dict(tampered)
    assert set(IMMUTABLE_VALUES).issubset(L3.protected)
    print("lifelong persistence: learner state round-trips losslessly, core stays protected ✓")

    # ---- CATASTROPHIC FORGETTING: a shared feature lets new data overwrite old ---- #
    def train(learner, feats, reward, steps):
        for _ in range(steps):
            learner.record("act", feats, reward)

    def err_on(learner, feats, target):
        return abs(learner.value("act", feats) - target)

    A = {"shared": 1.0, "ctx_a": 1.0}
    B = {"shared": 1.0, "ctx_b": 1.0}

    # plain learner: no protection
    plain = Learner(base_lr=0.1, seed=2)
    train(plain, A, 1.0, 120)
    train(plain, B, -1.0, 200)
    plain_forgot = err_on(plain, A, 1.0)

    # EWC-protected learner: consolidate skill A, then learn B
    ewc = Learner(base_lr=0.1, ewc_lambda=3.0, seed=3)
    train(ewc, A, 1.0, 120)
    ewc.consolidate()                 # lock in skill A
    train(ewc, B, -1.0, 200)
    ewc_forgot = err_on(ewc, A, 1.0)

    # replay learner: rehearse A while learning B
    rep = Learner(base_lr=0.1, seed=4)
    train(rep, A, 1.0, 120)
    for _ in range(200):
        rep.record("act", B, -1.0)
        rep.replay(8)                 # rehearse old experiences
    rep_forgot = err_on(rep, A, 1.0)

    print("\nforgetting (error on A after learning B):")
    print(f"  plain (no protection) : {plain_forgot:.3f}")
    print(f"  EWC-anchored          : {ewc_forgot:.3f}")
    print(f"  replay-rehearsed      : {rep_forgot:.3f}")
    assert ewc_forgot < plain_forgot       # anchoring protects the old skill
    assert rep_forgot < plain_forgot       # so does rehearsal
    print("forgetting protection : both EWC and replay retain skill A better ✓")

    # ---- learning-rate schedule decays ---- #
    sched = Learner(base_lr=1.0, lr_decay=0.1, seed=5)
    lr0 = sched.lr()
    for _ in range(20):
        sched.update("a", {"f": 1.0}, 1.0)
    print(f"\nlr schedule         : {lr0:.3f} -> {sched.lr():.3f}")
    assert sched.lr() < lr0

    # ---- prioritized replay favours high-reward experiences ---- #
    buf = ReplayBuffer()
    for i in range(50):
        buf.add(Experience("a", {"f": 1.0}, reward=(5.0 if i == 0 else 0.0)))
    rng = random.Random(0)
    hits = sum(1 for _ in range(200) if buf.sample(1, rng=rng)[0].reward == 5.0)
    print(f"prioritized replay  : high-reward sampled {hits}/200 times (vs ~4 uniform)")
    assert hits > 4

    print(f"\nreport              : {L.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
