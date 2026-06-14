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
# Experience & replay buffer
# --------------------------------------------------------------------------- #
@dataclass
class Experience:
    action: str
    features: Features
    reward: float
    context: str = ""
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "reward": round(self.reward, 4),
                "context": self.context, "features": dict(self.features)}


class ReplayBuffer:
    """A bounded buffer of experiences, sampled with priority by reward magnitude."""

    def __init__(self, capacity: int = 2000) -> None:
        self.capacity = capacity
        self._buf: Deque[Experience] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def add(self, exp: Experience) -> None:
        self._buf.append(exp)

    def recent(self, n: int = 10) -> List[Experience]:
        return list(self._buf)[-n:]

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


# --------------------------------------------------------------------------- #
# Linear value model with EWC anchoring
# --------------------------------------------------------------------------- #
class LinearValueModel:
    """Per-action linear value over sparse features, with EWC-style elastic anchoring."""

    def __init__(self, *, ewc_lambda: float = 0.0) -> None:
        self.ewc_lambda = ewc_lambda
        self._w: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._w_star: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._F: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))     # sum x^2
        self._n: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))     # uses
        self._anchor: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))  # frozen importance

    def predict(self, action: str, features: Features) -> float:
        w = self._w[action]
        return sum(w[f] * x for f, x in features.items())

    def update(self, action: str, features: Features, reward: float, lr: float) -> float:
        pred = self.predict(action, features)
        err = reward - pred
        w = self._w[action]
        w_star = self._w_star[action]
        F = self._F[action]
        n = self._n[action]
        anchor_imp = self._anchor[action]
        for f, x in features.items():
            # EWC penalty uses the *frozen, normalized* importance (0 before consolidation)
            penalty = self.ewc_lambda * anchor_imp[f] * (w[f] - w_star[f])
            w[f] += lr * (err * x - penalty)
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


# --------------------------------------------------------------------------- #
# Learner
# --------------------------------------------------------------------------- #
class Learner:
    """Online skill learning with reward feedback, replay, and forgetting protection."""

    def __init__(self, *, base_lr: float = 0.1, lr_decay: float = 0.0,
                 ewc_lambda: float = 0.0, replay_lr: Optional[float] = None,
                 buffer_capacity: int = 2000,
                 protected: Optional[Sequence[str]] = None, seed: int = 0) -> None:
        self.base_lr = base_lr
        self.lr_decay = lr_decay
        self.replay_lr = replay_lr if replay_lr is not None else base_lr * 0.5
        self.model = LinearValueModel(ewc_lambda=ewc_lambda)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.protected = set(protected) if protected is not None else set(IMMUTABLE_VALUES)
        self._rng = random.Random(seed)
        self._steps = 0

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
        err = self.model.update(action, features, reward, self.lr())
        self._steps += 1
        return err

    def record(self, action: str, features: Features, reward: float, *,
               context: str = "") -> float:
        """Update online *and* remember the experience for replay."""
        err = self.update(action, features, reward, context=context)
        self.buffer.add(Experience(action, dict(features), reward, context))
        return err

    # ---- value / decision ---- #
    def value(self, action: str, features: Features) -> float:
        return self.model.predict(action, features)

    def best_action(self, features: Features, actions: Sequence[str]) -> str:
        return max(actions, key=lambda a: self.model.predict(a, features))

    # ---- replay / consolidation (forgetting protection) ---- #
    def replay(self, n: int = 32, *, prioritized: bool = True) -> int:
        batch = self.buffer.sample(n, rng=self._rng, prioritized=prioritized)
        for exp in batch:
            self.model.update(exp.action, exp.features, exp.reward, self.replay_lr)
        return len(batch)

    def consolidate(self) -> None:
        self.model.consolidate()

    def report(self) -> Dict[str, Any]:
        return {"steps": self._steps, "lr": round(self.lr(), 5),
                "buffer": len(self.buffer), "ewc_lambda": self.model.ewc_lambda,
                "protected": sorted(self.protected)}


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
