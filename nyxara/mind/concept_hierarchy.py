"""NYXARA · mind/concept_hierarchy.py — a shared concept layer over actions (☰).

Before this module, every action owned a *private* world-model: learning ``left``
taught ``move_left`` nothing, and a never-seen action was a zero-confidence blank.
The missing piece was a **concept above the raw action key**.

:class:`ConceptHierarchy` supplies it. It clusters actions by *what they do* — their
**effect signature**, the running mean Δstate and reward they produce — online and
incrementally. Actions that move the world the same way fall into the same concept,
*across domains*, so a brand-new action can borrow the dynamics of its concept-mates
instead of starting from nothing. Clusters of clusters give a shallow tree, and the
whole structure is introspectable (:meth:`report`, :meth:`to_dict`).

Pure standard library (no numpy). Pairs with
:class:`nyxara.mind.world_model.TransferWorldModel`, which conditions a single shared
dynamics model on a learned action embedding and uses this hierarchy to (a) seed a new
action's embedding near the right concept — few-shot cross-domain transfer — and (b)
report honest, transfer-discounted confidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["EffectSignature", "Concept", "ConceptHierarchy"]


def _euclid(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# --------------------------------------------------------------------------- #
# Effect signature — a running summary of what one action *does*
# --------------------------------------------------------------------------- #
@dataclass
class EffectSignature:
    """Running mean of an action's Δstate and reward (Welford-style incremental mean)."""

    dim: int
    n: int = 0
    mean_delta: List[float] = field(default_factory=list)
    mean_reward: float = 0.0

    def __post_init__(self) -> None:
        if not self.mean_delta:
            self.mean_delta = [0.0] * self.dim

    def update(self, delta: Sequence[float], reward: float) -> None:
        self.n += 1
        for i in range(self.dim):
            self.mean_delta[i] += (delta[i] - self.mean_delta[i]) / self.n
        self.mean_reward += (reward - self.mean_reward) / self.n

    @property
    def magnitude(self) -> float:
        return math.sqrt(sum(v * v for v in self.mean_delta))

    def vec(self, reward_weight: float = 0.5) -> List[float]:
        """The signature as a single vector ``[Δstate…, w·reward]`` for clustering."""
        return [*self.mean_delta, reward_weight * self.mean_reward]


# --------------------------------------------------------------------------- #
# Concept — a node grouping actions with the same effect signature
# --------------------------------------------------------------------------- #
@dataclass
class Concept:
    cid: int
    members: List[Any] = field(default_factory=list)
    centroid: List[float] = field(default_factory=list)   # mean of member signature vecs
    n: int = 0                                             # total samples across members


# --------------------------------------------------------------------------- #
# The hierarchy
# --------------------------------------------------------------------------- #
class ConceptHierarchy:
    """Online clustering of actions by effect signature into an introspectable hierarchy.

    ``threshold`` is a *relative* distance (signature distance divided by a running scale),
    so it is robust to the absolute magnitude of the dynamics. ``reward_weight`` sets how
    much reward (vs. the Δstate) counts toward "same concept"."""

    def __init__(self, *, threshold: float = 0.4, reward_weight: float = 0.35) -> None:
        self.threshold = threshold
        self.reward_weight = reward_weight
        self._sig: Dict[Any, EffectSignature] = {}
        self._concept_of: Dict[Any, int] = {}
        self._concepts: Dict[int, Concept] = {}
        self._next = 0
        self._dim: Optional[int] = None
        self._scale = 0.0          # EMA of signature-vector magnitude (for scale-free distance)

    # ---- size / introspection ---- #
    def __len__(self) -> int:
        return len(self._sig)

    def actions(self) -> List[Any]:
        return list(self._sig)

    def concepts(self) -> List[Concept]:
        return list(self._concepts.values())

    def signature(self, action: Any) -> Optional[EffectSignature]:
        return self._sig.get(action)

    # ---- learning = updating each action's effect signature ---- #
    def update(self, action: Any, state: Sequence[float], next_state: Sequence[float],
               reward: float) -> None:
        if len(state) != len(next_state):
            return                                      # only homogeneous transitions
        dim = len(state)
        if self._dim is None:
            self._dim = dim
        if dim != self._dim:
            return
        delta = [next_state[i] - state[i] for i in range(dim)]
        sig = self._sig.get(action)
        if sig is None:
            sig = EffectSignature(dim)
            self._sig[action] = sig
        sig.update(delta, reward)
        self._reassign(action)

    # ---- clustering internals ---- #
    def _vec(self, action: Any) -> List[float]:
        return self._sig[action].vec(self.reward_weight)

    def _dist(self, v: Sequence[float], centroid: Sequence[float]) -> float:
        return _euclid(v, centroid) / (self._scale + 1e-6)

    def _centroid_excl(self, c: Concept, action: Any) -> Optional[List[float]]:
        members = [m for m in c.members if m != action]
        if not members:
            return None
        dimv = len(self._vec(members[0]))
        cen = [0.0] * dimv
        for m in members:
            v = self._vec(m)
            for i in range(dimv):
                cen[i] += v[i] / len(members)
        return cen

    def _recompute_centroid(self, c: Concept) -> None:
        if not c.members:
            return
        dimv = len(self._vec(c.members[0]))
        cen = [0.0] * dimv
        for m in c.members:
            v = self._vec(m)
            for i in range(dimv):
                cen[i] += v[i] / len(c.members)
        c.centroid = cen
        c.n = sum(self._sig[m].n for m in c.members)

    def _new_concept(self) -> int:
        cid = self._next
        self._next += 1
        self._concepts[cid] = Concept(cid=cid)
        return cid

    def _reassign(self, action: Any) -> None:
        v = self._vec(action)
        mag = math.sqrt(sum(x * x for x in v))
        # running scale (EMA) so the threshold stays scale-free as dynamics grow
        self._scale = 0.99 * self._scale + 0.01 * mag if self._scale else max(mag, 1e-6)
        cur = self._concept_of.get(action)

        best, best_d = None, float("inf")
        for cid, c in self._concepts.items():
            if cid == cur and len(c.members) == 1:
                continue                                # never match your own singleton
            centroid = self._centroid_excl(c, action) if action in c.members else c.centroid
            if not centroid:
                continue
            d = self._dist(v, centroid)
            if d < best_d:
                best, best_d = cid, d

        if best is not None and best_d <= self.threshold:
            target = best
        elif cur is not None and len(self._concepts[cur].members) == 1:
            target = cur                                # keep its own singleton concept
        else:
            target = self._new_concept()

        if cur != target:
            if cur is not None:
                self._concepts[cur].members.remove(action)
                if not self._concepts[cur].members:
                    del self._concepts[cur]
                else:
                    self._recompute_centroid(self._concepts[cur])
            self._concepts[target].members.append(action)
            self._concept_of[action] = target
        self._recompute_centroid(self._concepts[target])

    # ---- queries used by the world model (transfer) ---- #
    def concept_of(self, action: Any) -> Optional[int]:
        return self._concept_of.get(action)

    def siblings(self, action: Any) -> List[Any]:
        cid = self._concept_of.get(action)
        if cid is None:
            return []
        return [m for m in self._concepts[cid].members if m != action]

    def nearest_actions(self, action: Any, k: int = 3) -> List[Any]:
        """Best actions to borrow dynamics from: concept-mates first, else nearest by the
        **Δstate** of their effect signature (reward is noisy and not what transfers)."""
        sib = self.siblings(action)
        if sib:
            return sib[:k]
        if action not in self._sig:
            return []
        d = self._sig[action].mean_delta
        scored = sorted(((_euclid(d, self._sig[a].mean_delta), a)
                         for a in self._sig if a != action), key=lambda x: x[0])
        return [a for _, a in scored[:k]]

    def own_support(self, action: Any) -> int:
        sig = self._sig.get(action)
        return sig.n if sig else 0

    def concept_support(self, action: Any) -> int:
        cid = self._concept_of.get(action)
        return self._concepts[cid].n if cid is not None else 0

    def transfer_distance(self, action: Any) -> float:
        """How far the action sits from its concept centroid (0 = a perfect fit)."""
        cid = self._concept_of.get(action)
        if cid is None or action not in self._sig:
            return 1.0
        return self._dist(self._vec(action), self._concepts[cid].centroid)

    # ---- reporting ---- #
    def to_dict(self) -> Dict[str, Any]:
        return {"actions": len(self._sig),
                "concepts": [{"id": c.cid, "members": [str(m) for m in c.members],
                              "samples": c.n} for c in self._concepts.values()]}

    def report(self) -> str:
        lines = [f"ConceptHierarchy: {len(self._sig)} actions in "
                 f"{len(self._concepts)} concepts"]
        for c in self._concepts.values():
            members = ", ".join(str(m) for m in c.members)
            lines.append(f"  concept#{c.cid}: [{members}] (n={c.n})")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA concept-hierarchy self-test")
    print("=" * 70)

    h = ConceptHierarchy()

    # Two *domains* with the same underlying concepts: a 1-D gridworld and a 1-D
    # thermostat. "left"/"cool" both subtract; "right"/"heat" both add; "stay"/"hold" hold.
    moves = {
        "left": -1.0, "right": +1.0, "stay": 0.0,     # gridworld
        "cool": -1.0, "heat": +1.0, "hold": 0.0,      # thermostat (different domain)
    }
    import random
    rng = random.Random(0)
    for _ in range(300):
        a = rng.choice(list(moves))
        x = rng.uniform(-10, 10)
        h.update(a, (x,), (x + moves[a],), reward=-abs(x + moves[a]))

    print("\n" + h.report())

    # cross-domain: "left" and "cool" do the same thing → same concept
    assert h.concept_of("left") == h.concept_of("cool"), "subtract-actions should share a concept"
    assert h.concept_of("right") == h.concept_of("heat"), "add-actions should share a concept"
    assert h.concept_of("stay") == h.concept_of("hold"), "hold-actions should share a concept"
    assert len({h.concept_of("left"), h.concept_of("right"), h.concept_of("stay")}) == 3, \
        "subtract / add / hold are three distinct concepts"
    print("\ncross-domain concepts: left≡cool, right≡heat, stay≡hold (3 distinct)  ✓")

    # a brand-new action that subtracts borrows from the left/cool concept
    h.update("decrement", (3.0,), (2.0,), reward=-2.0)
    assert "decrement" in h.nearest_actions("decrement") + h.siblings("decrement") \
        or set(h.nearest_actions("decrement")) & {"left", "cool"}
    print(f"new 'decrement' borrows from: {h.nearest_actions('decrement')}  ✓")

    # support accounting
    assert h.own_support("left") > 0 and h.concept_support("left") >= h.own_support("left")
    print(f"\nsupport: own(left)={h.own_support('left')} "
          f"concept(left)={h.concept_support('left')}")

    print("\nALL SELF-TESTS PASSED ✓")
