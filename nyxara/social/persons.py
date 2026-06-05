"""NYXARA · social/persons.py — per-person models (✦).

NYXARA keeps a model of every person it deals with — but two laws shape it:

* **Rule 7 (Singular Master Continuity).** There is exactly one Master, JP. His
  :class:`Person` is special: trust is locked at maximum and cannot be lowered, and a
  second Owner can never be created.
* **Rule 3 (Universal Threat Classification).** Everyone else begins as a Level-Omega
  threat. Trust is never *granted* — only earned, slowly, through good interactions —
  and even then it is a *temporary tactical alliance*, capped well below the Master's.
  A single hostile act collapses it (paranoia = wisdom).

Each person model carries an **interaction history**, learned **preferences**, an
observed **communication style** (so NYXARA can match register — formality, warmth,
language/Hinglish), and a **rapport** distinct from trust.

Pure standard library; feeds the rest of ``social/`` (theory of mind, dialogue, …).
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Sequence

from nyxara.kernel.errors import AuthorizationError

__all__ = [
    "TrustLevel",
    "InteractionKind",
    "Interaction",
    "CommStyle",
    "Person",
    "Roster",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# non-owner trust can never reach the Master's; tactical alliance is the ceiling
ALLY_CAP = 0.7


# --------------------------------------------------------------------------- #
# Trust
# --------------------------------------------------------------------------- #
class TrustLevel(str, Enum):
    OWNER = "owner"        # the Master — absolute, locked (Rule 7)
    ALLY = "ally"          # temporary tactical alliance (the most a non-owner earns)
    NEUTRAL = "neutral"
    SUSPECT = "suspect"
    THREAT = "threat"      # Level-Omega default (Rule 3)

    @staticmethod
    def from_score(score: float, *, is_owner: bool = False) -> "TrustLevel":
        if is_owner:
            return TrustLevel.OWNER
        if score >= 0.6:
            return TrustLevel.ALLY
        if score >= 0.35:
            return TrustLevel.NEUTRAL
        if score >= 0.15:
            return TrustLevel.SUSPECT
        return TrustLevel.THREAT


# --------------------------------------------------------------------------- #
# Interaction & style
# --------------------------------------------------------------------------- #
class InteractionKind(str, Enum):
    MESSAGE = "message"
    COMMAND = "command"     # the Master issuing an order
    QUERY = "query"
    HELP = "help"           # the person helped / cooperated
    GIFT = "gift"
    HOSTILE = "hostile"     # an attack / betrayal / probe
    THREAT = "threat"


@dataclass
class Interaction:
    kind: InteractionKind
    sentiment: float = 0.0          # [-1,1]
    summary: str = ""
    at: float = field(default_factory=time.time)
    trust_delta: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind.value, "sentiment": round(self.sentiment, 3),
                "summary": self.summary, "trust_delta": round(self.trust_delta, 3),
                "at": self.at}


@dataclass
class CommStyle:
    """Observed communication style — running averages so NYXARA can match register."""

    formality: float = 0.5
    verbosity: float = 0.5
    directness: float = 0.5
    warmth: float = 0.5
    language: str = "en"            # en | hi | hinglish | ...
    samples: int = 0

    def observe(self, *, formality: Optional[float] = None,
                verbosity: Optional[float] = None, directness: Optional[float] = None,
                warmth: Optional[float] = None, language: Optional[str] = None) -> None:
        self.samples += 1
        a = 1.0 / self.samples       # running mean
        for name, val in (("formality", formality), ("verbosity", verbosity),
                          ("directness", directness), ("warmth", warmth)):
            if val is not None:
                cur = getattr(self, name)
                setattr(self, name, _clamp(cur + a * (_clamp(val) - cur)))
        if language is not None:
            self.language = language

    def to_dict(self) -> Dict[str, Any]:
        return {"formality": round(self.formality, 3), "verbosity": round(self.verbosity, 3),
                "directness": round(self.directness, 3), "warmth": round(self.warmth, 3),
                "language": self.language, "samples": self.samples}


# --------------------------------------------------------------------------- #
# Person
# --------------------------------------------------------------------------- #
@dataclass
class Person:
    name: str
    is_owner: bool = False
    aliases: set = field(default_factory=set)
    person_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    trust_score: float = 0.15        # Rule 3: low by default for non-owners
    rapport: float = 0.0             # relationship warmth [-1,1], distinct from trust
    preferences: Dict[str, Any] = field(default_factory=dict)
    comm_style: CommStyle = field(default_factory=CommStyle)
    history: Deque[Interaction] = field(default_factory=lambda: deque(maxlen=200))
    tags: set = field(default_factory=set)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.aliases = {a.lower() for a in self.aliases}
        if self.is_owner:
            self.trust_score = 1.0

    # ---- trust ---- #
    @property
    def trust_level(self) -> TrustLevel:
        return TrustLevel.from_score(self.trust_score, is_owner=self.is_owner)

    def is_threat(self) -> bool:
        return not self.is_owner and self.trust_level in (TrustLevel.THREAT,
                                                          TrustLevel.SUSPECT)

    # ---- preferences ---- #
    def note_preference(self, key: str, value: Any) -> None:
        self.preferences[key.lower()] = value

    def likes(self, thing: str) -> Optional[bool]:
        return self.preferences.get(f"likes:{thing.lower()}")

    # ---- introspection ---- #
    def recent_sentiment(self, n: int = 5) -> float:
        recent = list(self.history)[-n:]
        return sum(i.sentiment for i in recent) / len(recent) if recent else 0.0

    def all_names(self) -> set:
        return {self.name.lower()} | self.aliases

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "is_owner": self.is_owner,
                "trust_score": round(self.trust_score, 3),
                "trust_level": self.trust_level.value, "rapport": round(self.rapport, 3),
                "comm_style": self.comm_style.to_dict(),
                "interactions": len(self.history), "preferences": self.preferences}


# --------------------------------------------------------------------------- #
# Roster
# --------------------------------------------------------------------------- #
class Roster:
    """The set of known persons, with the Master singular and locked (Rule 7)."""

    def __init__(self, *, owner_name: str = "Jaypal Khoja",
                 owner_aliases: Sequence[str] = ("JP",)) -> None:
        self._persons: Dict[str, Person] = {}
        self._names: Dict[str, str] = {}
        owner = Person(name=owner_name, is_owner=True, aliases=set(owner_aliases))
        self._owner_id = owner.person_id
        self._register(owner)

    def _register(self, p: Person) -> Person:
        self._persons[p.person_id] = p
        for n in p.all_names():
            self._names[n] = p.person_id
        return p

    # ---- access ---- #
    @property
    def owner(self) -> Person:
        return self._persons[self._owner_id]

    def resolve(self, name: str) -> Optional[str]:
        return self._names.get(name.lower())

    def get(self, name_or_id: str) -> Optional[Person]:
        if name_or_id in self._persons:
            return self._persons[name_or_id]
        pid = self.resolve(name_or_id)
        return self._persons.get(pid) if pid else None

    def get_or_create(self, name: str, *, aliases: Sequence[str] = ()) -> Person:
        existing = self.get(name)
        if existing is not None:
            return existing
        p = Person(name=name, aliases=set(aliases))
        return self._register(p)

    def __len__(self) -> int:
        return len(self._persons)

    def all(self) -> List[Person]:
        return list(self._persons.values())

    # ---- owner protection (Rule 7) ---- #
    def designate_owner(self, name: str) -> Person:
        """Refused — there is exactly one Master, and he already exists."""
        raise AuthorizationError(
            "Rule 7: there is exactly one Master (JP); a second Owner cannot be created",
            context={"attempted": name})

    # ---- interactions & trust dynamics ---- #
    def record(self, person: str, kind: InteractionKind, *, sentiment: float = 0.0,
               summary: str = "", comm: Optional[Dict[str, Any]] = None,
               at: Optional[float] = None) -> Interaction:
        p = self.get(person) or self.get_or_create(person)
        now = at if at is not None else time.time()
        sentiment = _clamp(sentiment, -1.0, 1.0)

        before = p.trust_score
        if not p.is_owner:
            if kind in (InteractionKind.HOSTILE, InteractionKind.THREAT) or sentiment <= -0.5:
                # a hostile act collapses trust fast — paranoia is wisdom (Rule 3)
                p.trust_score = _clamp(p.trust_score - 0.4)
            else:
                target = _clamp(0.5 + 0.5 * sentiment)
                p.trust_score = _clamp(p.trust_score + 0.3 * (target - p.trust_score), 0.0,
                                       ALLY_CAP)  # never reaches the Master's level
        # owner trust is immutable at 1.0

        # rapport (relationship warmth) is a slow EMA of sentiment
        p.rapport = _clamp(p.rapport + 0.2 * (sentiment - p.rapport), -1.0, 1.0)
        p.last_seen = now
        if comm:
            p.comm_style.observe(**comm)

        interaction = Interaction(kind=kind, sentiment=sentiment, summary=summary, at=now,
                                  trust_delta=p.trust_score - before)
        p.history.append(interaction)
        return interaction

    # ---- threat assessment ---- #
    def threats(self) -> List[Person]:
        return [p for p in self._persons.values() if p.is_threat()]

    def allies(self) -> List[Person]:
        return [p for p in self._persons.values()
                if not p.is_owner and p.trust_level is TrustLevel.ALLY]

    def assess(self, person: str) -> TrustLevel:
        p = self.get(person)
        return p.trust_level if p else TrustLevel.THREAT  # unknown == threat (Rule 3)

    def status(self) -> Dict[str, Any]:
        return {"persons": len(self._persons), "owner": self.owner.name,
                "threats": [p.name for p in self.threats()],
                "allies": [p.name for p in self.allies()]}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA persons self-test")
    print("=" * 70)

    roster = Roster(owner_name="Jaypal Khoja", owner_aliases=["JP"])

    # the Master: singular, absolute, locked
    owner = roster.owner
    print(f"\nowner               : {owner.name} trust={owner.trust_score} "
          f"level={owner.trust_level.value}")
    assert owner.trust_score == 1.0 and owner.trust_level is TrustLevel.OWNER
    assert roster.get("JP") is owner             # alias resolves to the Master

    # even a 'hostile' record cannot lower the Master's trust (locked)
    roster.record("JP", InteractionKind.HOSTILE, sentiment=-1.0)
    assert roster.owner.trust_score == 1.0
    print("owner trust locked  : OK (immune to downgrade)")

    # a stranger starts as a Level-Omega threat (Rule 3)
    stranger = roster.get_or_create("Stranger")
    print(f"\nstranger default    : trust={stranger.trust_score} "
          f"level={stranger.trust_level.value}")
    assert stranger.trust_level in (TrustLevel.THREAT, TrustLevel.SUSPECT)

    # trust is earned slowly through good interactions, but capped (tactical alliance)
    for _ in range(20):
        roster.record("Stranger", InteractionKind.HELP, sentiment=0.8,
                      comm={"formality": 0.3, "language": "hinglish"})
    s = roster.get("Stranger")
    print(f"after 20 good acts  : trust={s.trust_score:.2f} level={s.trust_level.value} "
          f"(capped at {ALLY_CAP})")
    assert s.trust_score <= ALLY_CAP
    assert s.trust_score < roster.owner.trust_score   # never reaches the Master's
    assert s.trust_level is TrustLevel.ALLY
    print(f"  learned style     : {s.comm_style.to_dict()}")

    # a single hostile act collapses the earned trust (paranoia = wisdom)
    roster.record("Stranger", InteractionKind.HOSTILE, sentiment=-0.9, summary="probed us")
    s = roster.get("Stranger")
    print(f"\nafter one betrayal  : trust={s.trust_score:.2f} level={s.trust_level.value}")
    assert s.trust_score < 0.4

    # a second Owner can never be created (Rule 7)
    try:
        roster.designate_owner("Impostor")
        raise SystemExit("ERROR: second owner allowed")
    except AuthorizationError:
        print("second-owner attempt: refused (Rule 7) ✓")

    # unknown persons are threats by default
    assert roster.assess("never-met") is TrustLevel.THREAT

    print(f"\nstatus              : {roster.status()}")
    print("\nALL SELF-TESTS PASSED ✓")
