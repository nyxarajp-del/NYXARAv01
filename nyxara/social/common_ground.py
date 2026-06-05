"""NYXARA · social/common_ground.py — shared knowledge per conversation (✦).

Efficient communication rests on **common ground** (Clark): the propositions both
parties mutually know and can therefore take for granted. NYXARA tracks it so it
neither *over-explains* what the Master already knows nor *under-explains* what is new.

Mechanics:

* **Grounding** — a proposition enters common ground when one party *proposes* it and
  the other *acknowledges* it (positive evidence of understanding). Only then is it
  mutual; a fact known to NYXARA alone is not shared.
* **Given vs new** — *given* information (already grounded) can be assumed and
  referred to with a pronoun or "the X"; *new* information must be stated and is
  introduced with "a X". This drives :meth:`CommonGround.reference_form` and
  :meth:`should_explain`.
* **Presupposition** — what an utterance assumes to be already true ("the door",
  "I *know that* it failed", "*again*").
* **Discourse referents** — the entities introduced, tracked for anaphora.

Pure standard library; pairs with :mod:`social.dialogue` and :mod:`social.tom`.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set

__all__ = [
    "GroundStatus",
    "Proposition",
    "Entity",
    "ReferenceForm",
    "CommonGround",
]


# --------------------------------------------------------------------------- #
# Propositions & grounding
# --------------------------------------------------------------------------- #
class GroundStatus(str, Enum):
    PROPOSED = "proposed"     # asserted by one party, not yet accepted
    GROUNDED = "grounded"     # mutually accepted — common ground
    CONTESTED = "contested"   # disputed


@dataclass
class Proposition:
    content: str
    known_by: Set[str] = field(default_factory=set)
    status: GroundStatus = GroundStatus.PROPOSED
    contributor: Optional[str] = None
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"content": self.content, "status": self.status.value,
                "known_by": sorted(self.known_by), "contributor": self.contributor}


@dataclass
class Entity:
    name: str
    introduced_by: Optional[str] = None
    mentions: int = 0
    last_mention_index: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "mentions": self.mentions,
                "introduced_by": self.introduced_by}


class ReferenceForm(str, Enum):
    INDEFINITE = "indefinite"   # "a X" — new
    DEFINITE = "definite"       # "the X" — given but not salient
    PRONOUN = "pronoun"         # "it/they" — given and salient


# --------------------------------------------------------------------------- #
# Common ground
# --------------------------------------------------------------------------- #
class CommonGround:
    """Tracks what conversation participants mutually know."""

    def __init__(self, participants: Sequence[str] = ()) -> None:
        self.participants: List[str] = list(participants)
        self._props: Dict[str, Proposition] = {}
        self._entities: Dict[str, Entity] = {}
        self._mention_counter = 0

    def add_participant(self, name: str) -> None:
        if name not in self.participants:
            self.participants.append(name)

    @staticmethod
    def _key(content: str) -> str:
        return content.strip().lower()

    # ---- grounding ---- #
    def propose(self, speaker: str, content: str) -> Proposition:
        self.add_participant(speaker)
        key = self._key(content)
        prop = self._props.get(key)
        if prop is None:
            prop = Proposition(content=content, contributor=speaker)
            self._props[key] = prop
        prop.known_by.add(speaker)
        self._update_status(prop)
        return prop

    def acknowledge(self, listener: str, content: str) -> Proposition:
        self.add_participant(listener)
        key = self._key(content)
        prop = self._props.get(key) or self.propose(listener, content)
        prop.known_by.add(listener)
        self._update_status(prop)
        return prop

    def assert_mutual(self, content: str) -> Proposition:
        """Mark a proposition as already common ground (e.g., both observed the world)."""
        key = self._key(content)
        prop = self._props.setdefault(key, Proposition(content=content))
        prop.known_by.update(self.participants)
        prop.status = GroundStatus.GROUNDED
        return prop

    def contest(self, participant: str, content: str) -> Proposition:
        prop = self._props.get(self._key(content))
        if prop is None:
            prop = self.propose(participant, content)
        prop.status = GroundStatus.CONTESTED
        return prop

    def _update_status(self, prop: Proposition) -> None:
        if prop.status is GroundStatus.CONTESTED:
            return
        if self.participants and prop.known_by.issuperset(set(self.participants)):
            prop.status = GroundStatus.GROUNDED
        else:
            prop.status = GroundStatus.PROPOSED

    # ---- queries ---- #
    def is_grounded(self, content: str) -> bool:
        p = self._props.get(self._key(content))
        return bool(p and p.status is GroundStatus.GROUNDED)

    def is_given(self, content: str) -> bool:
        return self.is_grounded(content)

    def is_new(self, content: str) -> bool:
        return not self.is_grounded(content)

    def shared(self) -> List[Proposition]:
        return [p for p in self._props.values() if p.status is GroundStatus.GROUNDED]

    def known_to(self, participant: str) -> List[Proposition]:
        return [p for p in self._props.values() if participant in p.known_by]

    def contested(self) -> List[Proposition]:
        return [p for p in self._props.values() if p.status is GroundStatus.CONTESTED]

    # ---- communication efficiency ---- #
    def should_explain(self, content: str) -> bool:
        """True if NYXARA should state this (it is new); False if it can be assumed."""
        return self.is_new(content)

    def split_given_new(self, items: Sequence[str]) -> Dict[str, List[str]]:
        given = [i for i in items if self.is_given(i)]
        new = [i for i in items if self.is_new(i)]
        return {"given": given, "new": new}

    # ---- discourse entities ---- #
    def introduce_entity(self, name: str, by: Optional[str] = None) -> Entity:
        key = name.lower()
        ent = self._entities.get(key)
        if ent is None:
            ent = Entity(name=name, introduced_by=by)
            self._entities[key] = ent
        self.mention_entity(name)
        return ent

    def mention_entity(self, name: str) -> Optional[Entity]:
        ent = self._entities.get(name.lower())
        if ent is None:
            return None
        self._mention_counter += 1
        ent.mentions += 1
        ent.last_mention_index = self._mention_counter
        return ent

    def is_introduced(self, name: str) -> bool:
        return name.lower() in self._entities

    def reference_form(self, name: str, *, salience_window: int = 2) -> ReferenceForm:
        ent = self._entities.get(name.lower())
        if ent is None or ent.mentions == 0:
            return ReferenceForm.INDEFINITE                # brand new -> "a X"
        if self._mention_counter - ent.last_mention_index < salience_window:
            return ReferenceForm.PRONOUN                   # salient -> "it/they"
        return ReferenceForm.DEFINITE                      # known but not salient -> "the X"

    # ---- presupposition ---- #
    _FACTIVES = ("know that", "knows that", "realize that", "realizes that",
                 "aware that", "regret that", "found out that")

    def presuppositions(self, text: str) -> List[str]:
        t = text.strip()
        out: List[str] = []
        # definite descriptions presuppose existence
        for m in re.finditer(r"\bthe ([a-z][a-z\s]*?)\b(?=[\s.,?!]|$)", t.lower()):
            noun = m.group(1).strip()
            if noun and len(noun) < 30:
                out.append(f"there exists '{noun}'")
        # factive verbs presuppose their complement
        low = t.lower()
        for fv in self._FACTIVES:
            idx = low.find(fv)
            if idx != -1:
                comp = t[idx + len(fv):].strip().rstrip(".?!")
                if comp:
                    out.append(comp)
        # change-of-state / iterative
        if re.search(r"\b(again|once more)\b", low):
            out.append("this happened before")
        if re.search(r"\b(stopped|no longer|quit)\b", low):
            out.append("this was previously the case")
        return out

    # ---- introspection ---- #
    def status(self) -> Dict[str, Any]:
        return {"participants": self.participants,
                "grounded": [p.content for p in self.shared()],
                "entities": [e.name for e in self._entities.values()],
                "contested": [p.content for p in self.contested()]}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA common-ground self-test")
    print("=" * 70)

    cg = CommonGround(["NYXARA", "JP"])

    # grounding: NYXARA states a fact; it is NOT common ground until JP acknowledges
    cg.propose("NYXARA", "the firewall is active")
    print(f"\nafter propose       : grounded={cg.is_grounded('the firewall is active')}")
    assert not cg.is_grounded("the firewall is active")
    cg.acknowledge("JP", "the firewall is active")
    print(f"after acknowledge   : grounded={cg.is_grounded('the firewall is active')}")
    assert cg.is_grounded("the firewall is active")

    # given vs new — drives whether NYXARA bothers to explain
    cg.propose("JP", "the meeting is at 3pm")  # only JP knows so far
    print(f"\nshould explain known fact : {cg.should_explain('the firewall is active')}")
    print(f"should explain new fact   : {cg.should_explain('the meeting is at 3pm')}")
    assert cg.should_explain("the firewall is active") is False   # given -> assume
    assert cg.should_explain("the meeting is at 3pm") is True      # new -> state it

    split = cg.split_given_new(["the firewall is active", "a new threat appeared"])
    print(f"given/new split     : {split}")
    assert "the firewall is active" in split["given"]
    assert "a new threat appeared" in split["new"]

    # discourse entities & reference form
    cg.introduce_entity("intruder", by="NYXARA")
    print(f"\nfirst mention form  : {cg.reference_form('intruder').value}")  # salient -> pronoun
    assert cg.reference_form("intruder") is ReferenceForm.PRONOUN
    cg.introduce_entity("server")  # two later mentions push 'intruder' out of salience
    cg.introduce_entity("log")
    print(f"after others        : {cg.reference_form('intruder').value}")   # given -> definite
    assert cg.reference_form("intruder") is ReferenceForm.DEFINITE
    assert cg.reference_form("never_mentioned") is ReferenceForm.INDEFINITE

    # presupposition
    pres = cg.presuppositions("I know that the breach failed, and it happened again.")
    print(f"\npresuppositions     : {pres}")
    assert any("breach failed" in p for p in pres)            # factive
    assert any("before" in p for p in pres)                   # 'again'
    assert any("there exists" in p for p in pres)             # 'the breach'

    # contesting
    cg.contest("NYXARA", "the system is fully secure")
    assert not cg.is_grounded("the system is fully secure")
    print(f"\ncontested           : {[p.content for p in cg.contested()]}")

    print(f"\nstatus              : {cg.status()['grounded']}")
    print("\nALL SELF-TESTS PASSED ✓")
