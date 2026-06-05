"""NYXARA · social/tom.py — Theory of Mind (✦).

To predict, persuade, protect against, or cooperate with another agent, NYXARA must
model *its mind* — not just the world. This is a Theory-of-Mind engine in the
belief–desire–intention (BDI) tradition:

* **Beliefs** — what an agent holds to be true (which may differ from reality: a
  **false belief**). Agents act on what they believe, not on what is true.
* **Desires** — what they want, with strengths.
* **Intentions** — what they plan to do.

Crucially it is **recursive**: a mental model can contain a model of *another* agent's
mind — "Anne thinks that Sally believes the marble is in the basket" — to arbitrary
depth (orders of intentionality). The classic **Sally-Anne false-belief** test runs
end-to-end, including the second-order question "does Anne know Sally is mistaken?".

NYXARA's chief use: anticipate threats (Rule 3) and serve the Master by understanding
what others believe and intend. Pure standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

__all__ = [
    "MentalState",
    "TheoryOfMind",
]

Path = Union[str, Sequence[str]]


# --------------------------------------------------------------------------- #
# Mental state (recursive)
# --------------------------------------------------------------------------- #
@dataclass
class MentalState:
    """One agent's mind: beliefs, desires, intentions, and models of other minds."""

    beliefs: Dict[str, Any] = field(default_factory=dict)
    desires: Dict[str, float] = field(default_factory=dict)
    intentions: List[str] = field(default_factory=list)
    observed: set = field(default_factory=set)
    models_of: Dict[str, "MentalState"] = field(default_factory=dict)

    def model_of(self, agent: str) -> "MentalState":
        """This agent's (nested) model of ``agent``'s mind — created on demand."""
        return self.models_of.setdefault(agent, MentalState())

    def to_dict(self, depth: int = 1) -> Dict[str, Any]:
        d: Dict[str, Any] = {"beliefs": dict(self.beliefs),
                             "desires": dict(self.desires),
                             "intentions": list(self.intentions)}
        if depth > 0 and self.models_of:
            d["models_of"] = {k: v.to_dict(depth - 1) for k, v in self.models_of.items()}
        return d


# --------------------------------------------------------------------------- #
# Theory of Mind engine
# --------------------------------------------------------------------------- #
class TheoryOfMind:
    """NYXARA's model of the world and of other agents' minds."""

    def __init__(self) -> None:
        self.world: Dict[str, Any] = {}
        self._agents: Dict[str, MentalState] = {}

    # ---- world (ground truth) ---- #
    def set_world(self, prop: str, value: Any) -> None:
        self.world[prop] = value

    def get_world(self, prop: str) -> Any:
        return self.world.get(prop)

    # ---- agents ---- #
    def add_agent(self, name: str) -> MentalState:
        return self._agents.setdefault(name, MentalState())

    def agents(self) -> List[str]:
        return list(self._agents)

    def _norm(self, path: Path) -> List[str]:
        return [path] if isinstance(path, str) else list(path)

    def _resolve(self, path: Path, *, create: bool = True) -> Optional[MentalState]:
        names = self._norm(path)
        if not names:
            return None
        head = names[0]
        if head not in self._agents:
            if not create:
                return None
            self.add_agent(head)
        cur = self._agents[head]
        for nxt in names[1:]:
            if not create and nxt not in cur.models_of:
                return None
            cur = cur.model_of(nxt)
        return cur

    # ---- beliefs (possibly nested) ---- #
    def set_belief(self, path: Path, prop: str, value: Any) -> None:
        self._resolve(path).beliefs[prop] = value  # type: ignore[union-attr]

    def belief(self, path: Path, prop: str) -> Any:
        ms = self._resolve(path, create=False)
        return ms.beliefs.get(prop) if ms else None

    def observe(self, agent: str, prop: str) -> Any:
        """The agent perceives the *true* world value → forms/updates a belief."""
        ms = self.add_agent(agent)
        ms.beliefs[prop] = self.world.get(prop)
        ms.observed.add(prop)
        return ms.beliefs[prop]

    def has_false_belief(self, agent: str, prop: str) -> bool:
        ms = self._agents.get(agent)
        return bool(ms and prop in ms.beliefs and ms.beliefs[prop] != self.world.get(prop))

    # ---- nested attribution ---- #
    def attribute_belief(self, observer_path: Path, target: str, prop: str,
                         value: Any) -> None:
        """Encode "<observer> thinks <target> believes prop=value"."""
        path = self._norm(observer_path) + [target]
        self.set_belief(path, prop, value)

    def attributes_false_belief(self, observer: str, target: str, prop: str) -> bool:
        """Does ``observer`` attribute a (correct-by-our-lights) false belief to target?"""
        attributed = self.belief([observer, target], prop)
        return attributed is not None and attributed != self.world.get(prop)

    # ---- desires / intentions ---- #
    def set_desire(self, agent: str, goal: str, strength: float = 1.0) -> None:
        self.add_agent(agent).desires[goal] = strength

    def set_intention(self, agent: str, intention: str) -> None:
        self.add_agent(agent).intentions.append(intention)

    def strongest_desire(self, agent: str) -> Optional[str]:
        ms = self._agents.get(agent)
        if not ms or not ms.desires:
            return None
        return max(ms.desires, key=ms.desires.get)

    # ---- prediction (act on belief, not truth) ---- #
    def predict_belief_based_action(self, agent: str, prop: str) -> Any:
        """Where will the agent act? At the value they *believe*, not the true one."""
        ms = self._agents.get(agent)
        if ms is None or prop not in ms.beliefs:
            return None
        return ms.beliefs[prop]

    def predict_intention(self, agent: str) -> Optional[str]:
        ms = self._agents.get(agent)
        if ms is None:
            return None
        if ms.intentions:
            return ms.intentions[-1]
        return self.strongest_desire(agent)

    def state(self, agent: str, depth: int = 1) -> Optional[Dict[str, Any]]:
        ms = self._agents.get(agent)
        return ms.to_dict(depth) if ms else None


# --------------------------------------------------------------------------- #
# Self-test / demo — the Sally-Anne false-belief test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA theory-of-mind self-test (Sally-Anne)")
    print("=" * 70)

    tom = TheoryOfMind()
    tom.add_agent("Sally")
    tom.add_agent("Anne")

    # the marble starts in the basket; both see it there
    tom.set_world("marble", "basket")
    tom.observe("Sally", "marble")
    tom.observe("Anne", "marble")
    tom.set_desire("Sally", "find_marble", 1.0)
    print(f"\ninitially both believe: Sally={tom.belief('Sally','marble')} "
          f"Anne={tom.belief('Anne','marble')}")
    assert tom.belief("Sally", "marble") == "basket"

    # Sally LEAVES. Anne moves the marble to the box (truth changes); only Anne sees.
    tom.set_world("marble", "box")
    tom.observe("Anne", "marble")            # Anne witnesses
    # Sally does NOT observe -> she keeps a FALSE belief

    print(f"\nafter Anne moves it    : truth={tom.get_world('marble')} "
          f"Sally believes={tom.belief('Sally','marble')} "
          f"Anne believes={tom.belief('Anne','marble')}")
    assert tom.get_world("marble") == "box"
    assert tom.belief("Sally", "marble") == "basket"   # false belief retained
    assert tom.belief("Anne", "marble") == "box"
    assert tom.has_false_belief("Sally", "marble")
    assert not tom.has_false_belief("Anne", "marble")

    # PREDICTION: Sally will look where she BELIEVES it is — the basket (the ToM insight)
    where = tom.predict_belief_based_action("Sally", "marble")
    print(f"\nprediction            : Sally will search the «{where}» (her belief)")
    assert where == "basket"

    # SECOND-ORDER ToM: Anne knows Sally didn't see the move, so Anne models
    # Sally as still believing 'basket'. Does Anne know Sally is mistaken?
    tom.attribute_belief("Anne", "Sally", "marble", "basket")
    nested = tom.belief(["Anne", "Sally"], "marble")
    print(f"\n2nd-order: Anne thinks Sally believes «{nested}»")
    assert nested == "basket"
    assert tom.attributes_false_belief("Anne", "Sally", "marble")
    print("Anne correctly attributes a false belief to Sally ✓")

    # recursion depth is arbitrary: "Sally thinks Anne thinks the marble is..."
    tom.set_belief(["Sally", "Anne"], "marble", "basket")  # Sally (wrongly) thinks Anne agrees
    assert tom.belief(["Sally", "Anne"], "marble") == "basket"
    print("3-level nesting works (Sally→Anne→marble) ✓")

    # intentions & desires
    tom.set_intention("Anne", "watch how Sally reacts")
    assert tom.predict_intention("Anne") == "watch how Sally reacts"
    assert tom.strongest_desire("Sally") == "find_marble"

    print(f"\nSally's mind          : {tom.state('Sally', depth=1)}")
    print("\nALL SELF-TESTS PASSED ✓")
