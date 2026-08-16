"""NYXARA · njp/grounding.py — words become structure she can query (🔗, NJP V.02).

This is the module that answers *"what is my name"*.

Before it, NJP's only representation of a word was ``blake2b(token)`` — a cell id. Two words were
related exactly when they happened to fire together, and nothing more: ``"my name is Jay"`` moved
some synapse weights and left nothing that could be asked a question. The fabric was learning
**co-occurrence**, which is real, and it was the only thing being learned, which is the gap.

Grounding adds the other half. A turn becomes:

    surface text → concepts → entities → relations → beliefs

and every step of that is stored where it can be **queried, contradicted and revised** rather than
smeared into weights. The cell ids stay exactly as they were — the fabric still learns what fires
with what — and now they sit alongside structure instead of standing in for it.

**Where the intelligence is, honestly.** Extraction from surface text needs a parser, and every
parser is rules — a learned one has simply learned its rules from data. So this does not pretend
the extractor is magic. It ships a small deterministic core (fast, sub-millisecond, works on an
air-gapped box) and, where a fluent surface exists, asks it for the harder sentences. What makes
the result more than a keyword table is what happens either side of the parse:

* **types are discovered, never declared.** Nothing here says "Jay is a person". Instances go to
  :class:`~nyxara.cognition.concept_formation.AbstractionLadder`, and after enough of them its
  ``ascend`` finds that Jay and Ravi share features and groups them under a concept it named
  itself. A table would have known on turn one and never learned anything after.
* **patterns are learned.** A sentence the deterministic core cannot parse, that the fluent
  surface can, is generalised into a new :class:`Pattern` and reused thereafter — so the parser
  she has after a thousand turns is not the parser she shipped with. Learned patterns carry their
  own hit/miss record and are dropped when they stop paying.
* **facts are beliefs, not cells in a table.** Every triple is stored with provenance, confidence
  and a decay half-life (:mod:`nyxara.memory.provenance`), so a later contradiction *revises* it
  instead of silently overwriting it.

**Three epistemic states, kept apart.** ``KNOWN`` (in the graph, corroborated), ``BELIEVED``
(present with a confidence that is always quoted), and ``UNKNOWN`` (genuinely absent). Collapsing
the third into the second is how a system starts inventing answers, so ``UNKNOWN`` is a real return
value here and not an empty string.

Fail-soft throughout, like every organ in this package: a failed grounding returns an empty result
and the turn continues on the fabric alone.
"""

from __future__ import annotations

import math
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

__all__ = [
    "Epistemic", "GroundedTriple", "Answer", "Pattern", "GroundingResult", "Grounder",
    "SELF_ENTITY",
]

# The Master, as one canonical entity. First-person surface forms all resolve here, so "my name"
# and "Jay's name" reach the same node and can contradict each other.
SELF_ENTITY = "Master"

_FIRST_PERSON = frozenset({"i", "me", "my", "mine", "myself", "मैं", "मेरा", "मेरी", "मुझे",
                           "mera", "meri", "mujhe", "main"})
_SECOND_PERSON = frozenset({"you", "your", "yours", "yourself", "tum", "aap", "tera", "tumhara",
                            "आप", "तुम", "तेरा"})
# NYXARA herself, when the Master is talking *about* her.
_NYXARA = frozenset({"nyxara", "njp", "nyx"})


class Epistemic:
    """What she is entitled to say about a claim. Three states, never two."""

    KNOWN = "known"          # in the graph and corroborated — statable as fact
    BELIEVED = "believed"    # present, but only with its confidence attached
    UNKNOWN = "unknown"      # genuinely absent — she says so and does not generate


@dataclass
class GroundedTriple:
    """One extracted relation, with everything needed to judge and revise it.

    **Why a triple is not the whole claim.** ``water --freezes--> 0`` is what a flat extractor
    keeps out of *"if the temperature falls below 0°C, water freezes"*, and what it throws away is
    the half that makes the sentence usable: the relation does not hold, it holds **under a
    condition**. Three fields carry the part the arrow cannot:

    * :attr:`condition` — the circumstance the relation is asserted under. A triple with a
      condition is a *conditional* claim, and answering with it while dropping the condition is
      how a true statement becomes a false one.
    * :attr:`temporal` — when it held, where the surface said so. Distinct from ``at``, which is
      when *she was told*; a claim about 1950 recorded today has both, and they are not the same
      fact about it.
    * :attr:`modality` — whether the surface asserted, hedged, or generalised. "may cause" and
      "causes" are different claims and were previously indistinguishable once extracted.

    All three default empty, so a plain assertion is exactly the triple it always was, and every
    existing consumer that reads subject/predicate/object is untouched.
    """

    subject: str = ""
    predicate: str = ""
    object: str = ""
    confidence: float = 0.0
    source: str = ""              # which extractor produced it
    text: str = ""                # the surface it came from
    at: float = field(default_factory=time.time)
    # Set when a later, contradicting assertion superseded this one. Kept rather than deleted:
    # "he told me Jay first, then Raj" is a different epistemic state from "he told me Raj", and
    # the history is what lets her say which.
    superseded: bool = False
    contested: bool = False       # took part in a contradiction, either side
    # The cognitive half — see the class docstring. Empty means "the surface said nothing about
    # this", which is kept apart from a stated absence in exactly the way `Epistemic.UNKNOWN` is
    # kept apart from a low confidence.
    condition: str = ""
    temporal: str = ""
    modality: str = ""            # "" | "possible" | "typical" | "necessary"

    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    @property
    def conditional(self) -> bool:
        """Does this claim hold only under a stated circumstance?"""
        return bool(self.condition)

    def to_dict(self) -> Dict[str, Any]:
        out = {"subject": self.subject, "predicate": self.predicate, "object": self.object,
               "confidence": round(self.confidence, 4), "source": self.source,
               "superseded": self.superseded, "contested": self.contested}
        # Only present when the surface actually carried them. An unconditional claim should not
        # serialise three empty strings that every reader then has to learn to ignore.
        if self.condition:
            out["condition"] = self.condition
        if self.temporal:
            out["temporal"] = self.temporal
        if self.modality:
            out["modality"] = self.modality
        return out


@dataclass
class Answer:
    """A reply built from structure, carrying the state that licenses saying it."""

    text: str = ""
    state: str = Epistemic.UNKNOWN
    confidence: float = 0.0
    triples: List[GroundedTriple] = field(default_factory=list)
    why: str = ""

    @property
    def known(self) -> bool:
        return self.state == Epistemic.KNOWN

    @property
    def answered(self) -> bool:
        """Did she actually find something? ``UNKNOWN`` is an outcome, not an answer."""
        return self.state != Epistemic.UNKNOWN and bool(self.text)

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "state": self.state,
                "confidence": round(self.confidence, 4), "why": self.why,
                "triples": [t.to_dict() for t in self.triples]}


@dataclass
class Pattern:
    """An extraction rule, with the record that decides whether it survives.

    Seed patterns start with ``learned=False`` and are never dropped. A pattern induced from a
    fluent-surface parse starts at zero and has to earn its place: it is pruned once it has been
    tried enough times to be judged and is mostly missing.
    """

    regex: str = ""
    predicate: str = ""
    subject_group: str = "s"
    object_group: str = "o"
    learned: bool = False
    hits: int = 0
    misses: int = 0
    born: float = field(default_factory=time.time)

    @property
    def trials(self) -> int:
        return self.hits + self.misses

    @property
    def precision(self) -> float:
        return (self.hits / self.trials) if self.trials else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"regex": self.regex, "predicate": self.predicate, "learned": self.learned,
                "hits": self.hits, "misses": self.misses,
                "precision": round(self.precision, 4)}


@dataclass
class RecalledFact:
    """One fact retrieved *about the thing the question named*, with why it was offered.

    Not "the nearest memory". The distinction is the entire safety property of this path, and it
    is enforced structurally rather than by a threshold alone: a fact is only ever a candidate if
    its **subject** is substantially named in the question. A verified pendulum law cannot surface
    in a conversation about someone's day, because "day" does not name the pendulum.
    """

    triple: Optional[GroundedTriple] = None
    score: float = 0.0            # subject match × predicate affinity × the fact's own confidence
    subject_score: float = 0.0    # how squarely the question named this entity
    affinity: float = 0.0         # how well this relation answers the question that was asked
    hops: int = 0                 # 0 = stated of the subject itself, 1 = reached through an edge
    via: str = ""                 # the edge followed to get here, empty when direct

    def to_dict(self) -> Dict[str, Any]:
        return {"triple": self.triple.to_dict() if self.triple else None,
                "score": round(self.score, 4), "subject_score": round(self.subject_score, 4),
                "affinity": round(self.affinity, 4), "hops": self.hops, "via": self.via}


@dataclass
class GroundingResult:
    """What one turn grounded to. Empty is a normal, honest outcome."""

    text: str = ""
    triples: List[GroundedTriple] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    ungrounded: List[str] = field(default_factory=list)
    answer: Optional[Answer] = None
    contradictions: List[Tuple[GroundedTriple, GroundedTriple]] = field(default_factory=list)
    is_question: bool = False
    learned_pattern: bool = False
    ms: float = 0.0

    @property
    def grounded(self) -> bool:
        return bool(self.triples)

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text[:200], "is_question": self.is_question,
                "triples": [t.to_dict() for t in self.triples],
                "concepts": self.concepts[:32], "ungrounded": self.ungrounded[:16],
                "answer": self.answer.to_dict() if self.answer else None,
                "contradictions": [[a.to_dict(), b.to_dict()] for a, b in self.contradictions],
                "learned_pattern": self.learned_pattern, "ms": round(self.ms, 3)}


# The Hinglish process verbs, named once because two different tables need exactly the same set.
#
# "X se Y <verb> hai" is how a causal claim is stated; "X se kya <verb> hai" is how it is asked
# for back. Those regexes live ~200 lines apart, in `_SEED_PATTERNS` and `_QUESTION_PATTERNS`,
# and when the verb lists were written out separately in each the two drifted immediately — a
# verb she could be *told* in was not one she could be *asked* in, which is the read/write
# asymmetry `_CAUSE_OF` exists to close at the level of direction. Closing it for direction and
# leaving it open for vocabulary would have been the same bug wearing a different hat.
#
# Transitive and intransitive forms both belong here: "pyaas bujhti hai" (the thirst is quenched)
# and "paani pyaas bujhata hai" (water quenches thirst) are one claim about one arrow.
#
# Written as stems + an inflection group rather than as a flat list of surface forms. Hindi
# agrees the participle with its subject — ``-ta`` masculine singular, ``-ti`` feminine,
# ``-te`` masculine plural — so every verb here has three spellings and a hand-written list
# reliably ships two of them. "dhoop se kapde sukhTE hai" missing while "sukhta" was present is
# not a vocabulary gap, it is a plural the author did not think of; enumerating the inflection
# once makes that class of gap unrepresentable.
_PROCESS_STEMS = (
    r"ban|ho|aa|mil|nikal|badh|ghat|"                   # become, be, come, be got, rise, fall
    r"ubal|pighal|jal|mar|gir|ug|jam|"                  # boil, melt, burn, die, fall, grow, freeze
    r"bujh|bujha|sukh|phail|toot|khul|ruk|chal"         # quench, dry, spread, break, open, stop, run
)
_PROCESS_VERBS = (
    r"(?:" + _PROCESS_STEMS + r")(?:ta|ti|te)"
    # Devanagari carries the same three endings on the same stems; spelled out because the
    # matra-bearing stems do not decompose as cleanly as their romanisations.
    r"|बनता|बनती|बनते|होता|होती|होते|आता|आती|आते|मिलता|मिलती|मिलते|"
    r"निकलता|निकलती|बढ़ता|बढ़ती|घटता|घटती|"
    r"उबलता|पिघलता|जलता|जलती|मरता|मरती|गिरता|उगता|जमता|जमती|"
    r"बुझता|बुझती|सूखता|सूखते|फैलता|टूटता|खुलता|रुकता|चलता"
)


# --------------------------------------------------------------------------- #
# Seed patterns — the bootstrap parser, not the ceiling
# --------------------------------------------------------------------------- #
# Deliberately small. These exist so a cold, offline brain can ground the commonest relations on
# turn one; everything past them is either the fluent surface's job or learned from it. Growing
# this list by hand is the failure mode the learned patterns exist to avoid.
_SEED_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"^(?P<s>.+?)\s+(?:naam|name)\s+(?:is|hai|=)\s+(?P<o>.+)$", "has_name"),
    (r"^(?:my|mera|meri)\s+(?P<p>\w+)\s+(?:is|hai)\s+(?P<o>.+)$", ""),
    (r"^(?P<s>.+?)\s+lives?\s+in\s+(?P<o>.+)$", "located_in"),
    (r"^(?P<s>.+?)\s+(?:rehta|rehti)\s+(?:hai\s+)?(?P<o>.+?)\s*(?:me|mein)$", "located_in"),
    # --- Definitional forms ------------------------------------------------------------------ #
    #
    # Measured: these four constructions account for the largest single block of the 139/200
    # corpus answers that extracted nothing. They are how expository prose states a relation, and
    # the seed list was written for conversation, where they almost never appear. That is the
    # whole gap — not a hard parsing problem, a register the parser had never met.
    #
    # Ordered before `is_a` because "X is a subset of Y" is a *containment* claim, and the generic
    # rule below would read it as "X is a (subset of Y)" — an is_a edge to a phrase no other
    # sentence will ever produce again, which is worse than not extracting it.
    (r"^(?P<s>.+?)\s+is\s+(?:a\s+|an\s+)?(?:subset|subfield|branch|subclass|type|kind|form)"
     r"\s+of\s+(?P<o>.+)$", "part_of"),
    (r"^(?P<s>.+?)\s+(?:refers?\s+to|is\s+defined\s+as|is\s+known\s+as|stands?\s+for|"
     r"means?|denotes?)\s+(?P<o>.+)$", "means"),
    (r"^(?P<s>.+?)\s+is\s+the\s+(?:process|ability|practice|study|field|technique|method|art)"
     r"\s+of\s+(?P<o>.+)$", "means"),
    (r"^(?P<s>.+?)\s+(?:involves?|entails?)\s+(?P<o>.+)$", "involves"),
    (r"^(?P<s>.+?)\s+(?:focus(?:es)?\s+on|aims?\s+to|is\s+used\s+(?:to|for))\s+(?P<o>.+)$",
     "purpose"),
    (r"^(?P<s>.+?)\s+is\s+known\s+for\s+(?P<o>.+)$", "known_for"),
    # The definite copula. Measured on the bundled corpus: `X is the Y` is the single commonest
    # unextracted construction there, because every copula rule in this list demanded the
    # *indefinite* article. "The blue whale is the largest animal on Earth" is as plain a fact as
    # a sentence gets, and it grounded to nothing.
    #
    # Placed after the superlative extractor reads it (that one splits the ranking out first) and
    # after every specific `is the process/ability of` reading above, so this only ever catches
    # what nothing more precise claimed.
    (r"^(?P<s>.+?)\s+is\s+(?:considered\s+)?(?:to\s+be\s+)?(?:the|one\s+of\s+the)\s+(?P<o>.+)$",
     "is_a"),
    # "X occurs when C" — a definition *by* its condition. Kept whole as a relation rather than
    # split, because "overfitting" on its own is not a claim: the whole content of the sentence is
    # the circumstance. The condition is ALSO recorded on the triple, so a later reader can ask
    # for the circumstance without re-parsing the object.
    (r"^(?P<s>.+?)\s+(?:occurs?|happens?|arises?)\s+when\s+(?P<o>.+)$", "occurs_when"),
    (r"^(?P<s>.+?)\s+is\s+(?:a|an)\s+(?P<o>.+)$", "is_a"),
    (r"^(?P<s>.+?)\s+works?\s+(?:at|for)\s+(?P<o>.+)$", "works_at"),
    (r"^(?P<s>.+?)\s+(?:owns|has)\s+(?:a\s+|an\s+)?(?P<o>.+)$", "owns"),
    (r"^(?P<s>.+?)\s+likes?\s+(?P<o>.+)$", "likes"),
    (r"^(?P<s>.+?)\s+knows?\s+(?P<o>.+)$", "knows"),
    (r"^(?P<s>.+?)\s+is\s+(?:part\s+of|inside)\s+(?P<o>.+)$", "part_of"),
    (r"^(?P<s>.+?)\s+(?:causes?|leads?\s+to)\s+(?P<o>.+)$", "causes"),
    # Monotone influence — the signed half of causation, and the shape most real knowledge about
    # quantities actually has. "water increases growth" is not the same claim as "water causes
    # growth": it says which *way* the arrow points, which is exactly what a counterfactual needs
    # and what a bare `causes` edge cannot supply. Without these two relations the commonest
    # sentence anyone writes about a quantity extracted nothing at all.
    (r"^(?P<s>.+?)\s+(?:increases?|raises?|boosts?|improves?|badhata|badhati|badhata|"
     r"बढ़ाता|बढ़ाती)\s+(?P<o>.+)$", "increases"),
    (r"^(?P<s>.+?)\s+(?:decreases?|reduces?|lowers?|slows?|ghatata|ghatati|kam\s+karta|"
     r"घटाता|घटाती)\s+(?P<o>.+)$", "decreases"),
    # "zyada paani se zyada growth" / "more water more growth" — the comparative form of the
    # same claim, which is how it is usually said out loud.
    (r"^(?:more|zyada|ज़्यादा|अधिक)\s+(?P<s>.+?)\s+(?:se\s+|=\s*|,\s*|→\s*)?"
     r"(?:more|zyada|ज़्यादा|अधिक)\s+(?P<o>.+)$", "increases"),
    # Verb-final statements — the mirror of the question gap, and the reason closing that gap
    # alone was not enough. Every pattern above assumes the English order "X <relation> is Y";
    # Hinglish and Hindi put the copula last, "X ka <relation> Y **hai**". So the Master could
    # finally *ask* "Ravi ka naam kya hai" and there was still nothing to find, because
    # "Ravi ka naam Ravi Kumar hai" had never extracted. A language she can be questioned in but
    # not told anything in is half a language.
    (r"^(?:mera|meri|mere|मेरा|मेरी|मेरे)\s+(?P<p>\S+)\s+(?P<o>.+?)\s+(?:hai|hain|है|हैं)\s*$", ""),
    (r"^(?P<s>.+?)\s+(?:ka|ki|ke|का|की|के)\s+(?P<p>\S+)\s+(?P<o>.+?)"
     r"\s+(?:hai|hain|है|हैं)\s*$", ""),
    (r"^(?P<s>.+?)\s+(?P<o>.+?)\s+(?:me|mein|में)\s+(?:rehta|rehti|rahta|rahti|रहता|रहती)"
     r"\s*(?:hai|hain|है|हैं)?\s*$", "located_in"),
    (r"^(?P<s>.+?)\s+(?P<o>.+?)\s+(?:me|mein|में)\s+(?:kaam|काम)\s+(?:karta|karti|करता|करती)"
     r"\s*(?:hai|hain|है|हैं)?\s*$", "works_at"),
    (r"^(?P<s>.+?)\s+(?:ek|एक)\s+(?P<o>.+?)\s+(?:hai|hain|है|हैं)\s*$", "is_a"),
    # Happenings. Everything above this line is a relation that *holds*; everything below is one
    # that *happened*, and only these reach :data:`nyxara.njp.world._EVENT_PREDICATES` and become
    # events on a timeline. Measured before they existed: that predicate set listed twenty-one
    # kinds of happening and the extractor could produce exactly one of them (``causes``), so
    # `world.events` was structurally pinned at zero no matter what the Master said. These are
    # last in the list deliberately — a happening never shadows a fact, and none of the fact
    # patterns above match an intransitive past tense, so ordering costs nothing here.
    (r"^(?P<s>.+?)\s+(?P<v>went|came)\s+to\s+(?P<o>.+)$", ""),
    (r"^(?P<s>.+?)\s+(?P<v>bought|sold|sent|received|made|opened|closed|broke|started|stopped)"
     r"\s+(?P<o>.+)$", ""),
    (r"^(?P<s>.+?)\s+(?P<v>fell|broke|crashed|failed|started|stopped|opened|closed|happened"
     r"|grew|rose|died|burned|burnt|boiled|melted|froze|evaporated|condensed"
     r"|expanded|contracted)\s*$", ""),

    # --- Processes: the general statements a world model is actually built out of ------------ #
    #
    # Everything above this line records a *particular* happening ("the glass broke") or a
    # *particular* fact ("the Master's name is Jay"). Neither is what "water boils at 100" is.
    # A process statement is a standing claim about how the world works, and it is the only kind
    # of sentence from which a causal model can be built without waiting for the same accident to
    # happen three times.
    #
    # This was the measured bottleneck, not a nicety: over a real 20-turn session in the Master's
    # own Hinglish, `world.events` and `world.observations` were both exactly 0 and
    # `world.causal_links` was 0 — because "pani ubalne se bhaap banti hai" matched nothing at
    # all, and a world model with no intake cannot be wrong, cannot be surprised, and therefore
    # cannot learn. The patterns below are ordered longest-construction-first so a specific
    # reading is never swallowed by a general one.

    # "X ubal kar Y banta hai" — a process named by the participle that drives it.
    (r"^(?P<s>.+?)\s+\S+\s+(?:kar|कर)\s+(?P<o>.+?)\s+"
     r"(?:banta|banti|bante|hota|hoti|hote|बनता|बनती|होता|होती)"
     r"\s*(?:hai|hain|है|हैं)?\s*$", "causes"),
    # "X se Y banta/hota/nikalta hai" — the ablative "se" is Hinglish's plainest causal marker.
    (r"^(?P<s>.+?)\s+(?:se|से)\s+(?P<o>.+?)\s+"
     r"(?:" + _PROCESS_VERBS + r")"
     r"\s*(?:hai|hain|है|हैं)?\s*$", "causes"),
    # "bina X (ke) Y marta hai" — a requirement stated through its own violation. The groups are
    # deliberately reversed: the sentence leads with what is missing, but the *subject* of the
    # requirement is the thing that fails without it.
    (r"^(?:bina|बिना)\s+(?P<o>.+?)\s+(?:ke\s+|के\s+)?(?P<s>\S+(?:\s+\S+)?)\s+"
     r"(?:marta|marti|mar|rukta|rukti|nahi|nahin|नहीं|मरता|मरती|रुकता)\b.*$", "requires"),
    # "X ko Y chahiye" — the same requirement stated positively.
    (r"^(?P<s>.+?)\s+(?:ko|को)\s+(?P<o>.+?)\s+"
     r"(?:chahiye|chaahiye|chahiyee|चाहिए)\s*(?:hai|hain|है)?\s*$", "requires"),
    # "X Y deta hai" — production, the transitive process. Tried after "se" so
    # "aag se garmi milti hai" reads as causation rather than as a bare give.
    (r"^(?P<s>.+?)\s+(?P<o>.+?)\s+(?:deta|deti|dete|banata|banati|"
     r"देता|देती|बनाता|बनाती)\s*(?:hai|hain|है|हैं)?\s*$", "produces"),
    # "pani 100 degree par ubalta hai" / "water boils at 100 degrees" — a threshold, the most
    # quantitative thing the Master is likely to say, and the one a simulation can actually use.
    (r"^(?P<s>.+?)\s+(?P<q>-?\d+(?:\.\d+)?)\s*"
     r"(?:degree|degrees|°\s*c|celsius|डिग्री)?\s*(?:par|pe|pr|पर)\s+"
     r"(?P<v>ubalta|ubalti|pighalta|pighalti|jamta|jamti|jalta|jalti|"
     r"उबलता|उबलती|पिघलता|पिघलती|जमता)\s*(?:hai|hain|है|हैं)?\s*$", ""),
    (r"^(?P<s>.+?)\s+(?P<v>boils|melts|freezes|evaporates|condenses)\s+at\s+"
     r"(?P<q>-?\d+(?:\.\d+)?)\s*(?:degrees?|°\s*c|celsius|c)?\s*$", ""),
    # Bare Hinglish process verbs, verb-final and intransitive.
    (r"^(?P<s>.+?)\s+(?P<v>ubalta|ubalti|pighalta|pighalti|jalta|jalti|girta|girti|"
     r"tutta|tutti|badhta|badhti|marta|marti|ugta|ugti|"
     r"उबलता|उबलती|पिघलता|पिघलती|जलता|जलती|गिरता|गिरती|बढ़ता|बढ़ती|मरता|मरती)"
     r"\s*(?:hai|hain|है|हैं)?\s*$", ""),
    # English present-tense processes. Intransitive only — the transitive ones are relations
    # ("X needs Y") and are handled just below as such.
    (r"^(?P<s>.+?)\s+(?P<v>boils|melts|freezes|burns|grows|dies|falls|rises|"
     r"evaporates|condenses|expands|contracts)\s*$", ""),
    (r"^(?P<s>.+?)\s+(?:needs?|requires?)\s+(?P<o>.+)$", "requires"),
    # ``makes`` is present-tense production and belongs here; ``made`` is a past happening and is
    # matched above as an event. Splitting them by tense is not pedantry — "the engine makes
    # heat" is a standing property of engines and "Ravi made dinner" is a thing that occurred
    # once, and putting the first on a timeline loses the mechanism it states.
    (r"^(?P<s>.+?)\s+(?:produces?|makes|emits?|releases?|gives?\s+off)\s+(?P<o>.+)$", "produces"),
)

# Surface verb → the canonical predicate it is recorded under, so "fall"/"fell"/"gir gaya" all
# become one event kind. Without this, tense alone would split one happening into several kinds
# and nothing would ever recur often enough to become a cause.
_EVENT_VERBS: Dict[str, str] = {
    "went": "went", "came": "came", "bought": "bought", "sold": "sold", "sent": "sent",
    "received": "received", "made": "made", "opened": "opened", "closed": "closed",
    "broke": "broke", "started": "started", "stopped": "stopped", "fell": "fell",
    "crashed": "crashed", "failed": "failed", "happened": "happened",
    # Processes. Hinglish and English surfaces of the same physical change collapse onto one
    # predicate, which is the whole point of this table: "ubalta" and "boils" are not two
    # phenomena, and recording them separately would halve the evidence for each and stop either
    # from ever clearing `min_support`.
    "ubalta": "boils", "ubalti": "boils", "उबलता": "boils", "उबलती": "boils",
    "boils": "boils",
    "pighalta": "melts", "pighalti": "melts", "पिघलता": "melts", "पिघलती": "melts",
    "melts": "melts",
    "jamta": "freezes", "jamti": "freezes", "जमता": "freezes", "freezes": "freezes",
    "jalta": "burns", "jalti": "burns", "जलता": "burns", "जलती": "burns", "burns": "burns",
    "girta": "falls", "girti": "falls", "गिरता": "falls", "गिरती": "falls", "falls": "falls",
    "tutta": "breaks", "tutti": "breaks",
    "badhta": "grows", "badhti": "grows", "बढ़ता": "grows", "बढ़ती": "grows", "grows": "grows",
    "ugta": "grows", "ugti": "grows",
    "marta": "dies", "marti": "dies", "मरता": "dies", "मरती": "dies", "dies": "dies",
    "rises": "rises", "evaporates": "evaporates", "condenses": "condenses",
    "expands": "expands", "contracts": "contracts",
    # English past tense, folded onto the same predicates as the present. The Master reports what
    # happened in the past tense — "the plant grew" — and without these the sentence extracted
    # nothing at all, so `world` recorded no occurrence, `_counts` stayed empty, and
    # `world.counterfactual` could never clear `min_support`. An experiment settled from the
    # record is unreachable while the record cannot be written to.
    "grew": "grows", "rose": "rises", "died": "dies", "burned": "burns", "burnt": "burns",
    "boiled": "boils", "melted": "melts", "froze": "freezes", "evaporated": "evaporates",
    "condensed": "condenses", "expanded": "expands", "contracted": "contracts",
}

# "X because Y" — the one construction that states a cause outright rather than leaving it to be
# inferred from repetition. Split before extraction so BOTH halves are grounded on their own and
# the causal claim is recorded between them. Hinglish "kyunki"/"kyonki" included because the
# Master writes in it and an English-only causal reader is blind to half of what he says.
_BECAUSE = re.compile(
    r"^(?P<effect>.+?)\s+(?:because(?:\s+of)?|since|kyunki|kyonki|isliye\s+ki)\s+(?P<cause>.+)$",
    re.IGNORECASE)

# --------------------------------------------------------------------------- #
# Conditions, time and uncertainty — the half a triple cannot hold
# --------------------------------------------------------------------------- #
# Measured before these existed: of 200 corpus answers, 139 extracted nothing. The failures were
# not exotic sentences — they were four constructions, each of which states a relation plainly and
# none of which any seed pattern could see:
#
#   "X refers to Y"                         definitional
#   "The N types of X are A, B and C"       enumerative
#   "X involves P, while Y involves Q"      contrastive — two claims in one sentence
#   "X occurs when C"                       conditional
#
# The first two are vocabulary and are fixed in `_SEED_PATTERNS`. The last two are *structure*:
# one sentence carrying more than one claim, and a claim that holds only under a circumstance.
# A pattern list cannot fix those however long it grows, because the extractor stops at the first
# match and has nowhere to put a condition once it has one.

# A sentence's clauses, where each states its own claim. Split only on the connectives that
# genuinely coordinate two *independent* claims — "while"/"whereas" contrast them, ";" and "but"
# separate them. Deliberately NOT split on bare "and": "bread and butter" is one object, and
# splitting there would shred more claims than it recovers.
_CLAUSE_SPLIT = re.compile(
    r"\s*(?:;|,\s*(?:while|whereas|but)\s+|\s+whereas\s+)\s*", re.IGNORECASE)

# Sentence boundaries, for the multi-sentence answers a corpus supplies and a conversation rarely
# does. Guarded against the abbreviation that would otherwise split mid-claim.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Zऀ-ॿ])")

# The circumstance a claim is asserted under. Ordered most-specific first: "X occurs when C" is
# read by a seed pattern that keeps the whole thing as a relation, so it must not be caught here
# and reduced to a bare "X".
_CONDITIONS: Tuple[Any, ...] = (
    # "If C, then E" / "If C, E" — the canonical form, and the Master's "agar C to E".
    re.compile(r"^if\s+(?P<cond>.+?)\s*,\s*(?:then\s+)?(?P<main>.+)$", re.IGNORECASE),
    re.compile(r"^if\s+(?P<cond>.+?)\s+then\s+(?P<main>.+)$", re.IGNORECASE),
    re.compile(r"^(?:agar|अगर)\s+(?P<cond>.+?)\s+(?:to|toh|तो)\s+(?P<main>.+)$", re.IGNORECASE),
    # "When C, E" — same claim, and the comma is what makes it unambiguous.
    re.compile(r"^when\s+(?P<cond>.+?)\s*,\s*(?P<main>.+)$", re.IGNORECASE),
    # "E if C" — the trailing form. Last, because a leading "if" is never this.
    re.compile(r"^(?P<main>.+?)\s+if\s+(?P<cond>.+)$", re.IGNORECASE),
)

# What the surface said about how sure it is. This is *uncertainty as stated*, not her own
# confidence in the extraction — the two are multiplied at the end, never conflated. A hedged
# claim extracted perfectly is still a hedged claim.
#
# The factor is applied to the pattern's own confidence, so "X may cause Y" cannot be asserted
# with the weight of "X causes Y" however cleanly it parsed.
_MODALITY: Tuple[Tuple[Any, str, float], ...] = (
    (re.compile(r"\b(?:may|might|could|possibly|perhaps|sometimes|can\s+sometimes|"
                r"shayad|शायद|ho\s+sakta)\b", re.IGNORECASE), "possible", 0.6),
    (re.compile(r"\b(?:usually|typically|generally|often|commonly|normally|mostly|tend\s+to|"
                r"aam\s+taur\s+par|आमतौर)\b", re.IGNORECASE), "typical", 0.85),
    (re.compile(r"\b(?:always|never|must|invariably|necessarily|hamesha|हमेशा)\b",
                re.IGNORECASE), "necessary", 1.0),
)

# When it held, where the surface said so. Kept as the phrase rather than parsed into a timestamp:
# "during training" and "in 1950" are both answers to *when*, and only one of them is a date. A
# parser that insisted on a date would discard the commoner of the two.
_TEMPORAL = re.compile(
    r"\b(?:in|during|after|before|since|by|until|till)\s+"
    r"(?P<t>(?:the\s+)?(?:\d{4}|\d{1,2}(?:st|nd|rd|th)?\s+\w+|"
    r"training|inference|runtime|execution|testing|deployment|preprocessing)"
    r"(?:\s+(?:phase|time|stage|period))?)\b",
    re.IGNORECASE)

# An enumeration states several claims of one kind about one subject, and the count is part of
# what makes it checkable — "the three types of X" that yields two is a parse that went wrong.
_ENUMERATION = re.compile(
    r"^(?:the\s+)?(?:\w+\s+){0,3}?"
    r"(?P<kind>categories|types|kinds|forms|classes|stages|components|parts|steps|"
    r"phases|layers|branches|subfields|examples|applications|advantages|benefits|"
    r"disadvantages|limitations|challenges)\s+of\s+"
    r"(?P<s>.+?)\s+(?:are|include|includes)\s+(?P<list>.+)$",
    re.IGNORECASE)

# "X consists of A, B and C" — the same shape stated from the subject rather than the kind.
_COMPOSITION = re.compile(
    r"^(?P<s>.+?)\s+(?:consists?\s+of|is\s+composed\s+of|is\s+made\s+up\s+of|comprises?)\s+"
    r"(?P<list>.+)$", re.IGNORECASE)

# The kind-noun a list is a list *of*, folded onto one predicate so "types" and "kinds" of the
# same subject accumulate into one relation instead of two that never meet.
_KIND_PREDICATE: Dict[str, str] = {
    "categories": "has_kind", "types": "has_kind", "kinds": "has_kind",
    "forms": "has_kind", "classes": "has_kind", "branches": "has_kind",
    "subfields": "has_kind", "examples": "has_example", "applications": "has_application",
    "stages": "has_stage", "steps": "has_step", "phases": "has_stage",
    "components": "has_part", "parts": "has_part", "layers": "has_part",
    "advantages": "has_advantage", "benefits": "has_advantage",
    "disadvantages": "has_limitation", "limitations": "has_limitation",
    "challenges": "has_limitation",
}

# An appositive alias — "the bumblebee bat, also known as the Kitti's hog-nosed bat, is …". Two
# names for one entity, stated in passing inside a sentence that is really about something else.
# Worth extracting for a reason beyond the alias itself: an entity with two surfaces is reachable
# from both, so a later question phrased the other way finds what she already knows instead of
# creating a second, unconnected node for the same animal.
_APPOSITIVE = re.compile(
    r"^(?P<s>.+?)\s*,\s*(?:also\s+)?(?:known\s+as|called|referred\s+to\s+as)\s+"
    r"(?P<alias>.+?)\s*,\s*(?P<rest>.+)$", re.IGNORECASE)

# "X is the largest Y" — a superlative, and two claims rather than one. It says what X *is*
# (a Y) and where it *ranks* among them, and a store that keeps only the whole phrase can answer
# neither question: "largest animal on Earth" is a string no other sentence will ever produce,
# so the fact connects to nothing. Splitting it gives one edge that joins X to a kind other
# entities also belong to, and one that records the ranking.
_SUPERLATIVE = re.compile(
    r"^(?P<s>.+?)\s+(?:is|was)\s+(?:considered\s+)?(?:to\s+be\s+)?"
    r"(?:the\s+|one\s+of\s+the\s+)"
    r"(?P<sup>largest|biggest|smallest|fastest|slowest|tallest|shortest|heaviest|lightest|"
    r"longest|deepest|highest|strongest|weakest|oldest|newest|rarest|best|worst|"
    r"most\s+\w+|least\s+\w+)\s+(?P<o>.+)$", re.IGNORECASE)

# Where a noun phrase stops being a noun phrase. Everything after the first of these belongs to a
# modifier — a prepositional phrase, a relative clause, a participle — and the head of the phrase
# is the last noun before it.
#
# This exists because of a measurement, not a wish for tidier data. After 1,200 corpus pairs the
# store held 1,226 facts across 831 subjects, and **12 objects out of 1,226 were also subjects**.
# The graph was a star: every subject pointed at a phrase that nothing else ever pointed at or
# spoke about. `is_a` was worst — 3 of 443. So `core` could compose nothing (a chain needs the
# object of one edge to be the subject of the next), `concepts` could find no invariant (members
# need a shared property), and both organs looked broken while being starved. `learner` after
# 1,320 turns: 9 schemas, and `answers_given` exactly 0.
_KIND_HEAD_BREAK = re.compile(
    r"\b(?:of|in|on|for|with|that|which|who|whose|to|from|by|where|when|"
    r"used|known|called|designed|based|made|found|created|developed|"
    r"consisting|containing|responsible|capable|able|depends?)\b", re.IGNORECASE)

# Words that are never the head of a kind: articles and the grammar around them.
_KIND_HEAD_SKIP = frozenset({
    "the", "a", "an", "of", "in", "on", "for", "with", "that", "which", "to", "and", "or",
    "is", "are", "was", "were", "from", "by", "as", "its", "their", "this", "these", "those",
})

# Splitting a list into its items. Oxford comma and bare "and"/"or" both appear, and an item may
# itself contain "or" ("narrow or weak AI") — so the separator is a comma *or* a conjunction that
# is not inside an item, approximated by requiring the conjunction to be the last separator.
_LIST_SPLIT = re.compile(r"\s*,\s*(?:and\s+|or\s+)?|\s+and\s+", re.IGNORECASE)

# --------------------------------------------------------------------------- #
# Measurement — the one kind of sentence a causal coefficient can be fitted from
# --------------------------------------------------------------------------- #
# Every pattern in :data:`_SEED_PATTERNS` reads at most one relation out of a sentence and stops.
# That is right for a fact and wrong for a measurement: "the plant got 10 litres of water and
# grew 20 cm" states *two* variables of *one* entity at *one* moment, and it is the pairing that
# carries the information — two numbers observed together are what a coefficient is fitted to,
# while the same two numbers read separately are two unrelated readings.
#
# This was measured, not assumed. After four such sentences,
# :meth:`nyxara.njp.universe.InternalUniverse.usable_relations` was empty,
# ``concepts._by_subject[...].numbers`` was ``{}``, and ``what_if('water', 0.5)`` answered at
# confidence 0.0 with "no usable arrow leaves the intervened variables" — a simulator that owned
# a causal skeleton it could never fit a single coefficient to, because nothing upstream had ever
# handed it a number.
_MEASURE = re.compile(
    r"(?P<q>-?\d+(?:\.\d+)?)\s*"
    r"(?P<u>litres?|liters?|ml|cm|mm|km|kg|degrees?|celsius|hours?|hrs?|days?|"
    r"[lmg])?\b\s*"
    r"(?:of\s+(?P<n>[a-z]+))?",
    re.IGNORECASE)

# A sentence only reports a measurement if it says who was measured. Requiring the subject is
# what keeps this off "water boils at 100" — a standing law, already read by the threshold
# pattern below, and not an observation of any particular thing.
#
# The Hinglish half is not a nicety. The English verb list alone meant that in the Master's own
# language *no sentence could ever report a measurement*: "paudhe ko 2 litre paani mila" names a
# subject, a quantity, a unit and a substance, and matched nothing. Downstream that is the whole
# numeric pipeline — `concepts[...].numbers` stays `{}`, `field._sync_world` never calls
# `universe.observe`, `universe.state` stays empty, and every counterfactual over a variable she
# was told about in Hinglish answers "no usable arrow leaves the intervened variables". The
# comment above this block records that exact failure being fixed once for English; it was left
# reproducible verbatim for the language the Master actually writes in.
_MEASURE_SUBJECT = re.compile(
    r"^(?P<s>.+?)\s+(?:got|gets|has|had|received|used|took|drank|grew|grows|"
    r"reached|measured|weighs|weighed|lasted"
    # Hinglish. `mila/mili/mile` (was given), `liya/li` (took), `piya` (drank),
    # `badha/badhi/badhe` (grew), `pahuncha` (reached), `hai/tha/thi` (is/was — the plain copula,
    # which is how a quantity is most often simply stated).
    r")\b", re.IGNORECASE)

# The same sentence in Hinglish word order, which is the reason this is a second pattern rather
# than more alternatives in the first.
#
# English puts the verb between the subject and the quantity — "the plant GREW 20 cm" — so
# "everything before the verb" is the subject. Hindi is verb-final: "paudhe ko 2 litre paani
# MILA" puts it last, and the same rule then swallows the quantity into the subject. Measured
# with the verbs merely added to the list above: subject `"paudhe ko 2 litre paani"`, which is
# not an entity, will never match the same plant twice, and quietly makes every Hinglish
# measurement its own single-observation variable that no coefficient can ever be fitted to.
#
# So the subject is taken as what precedes the first digit, with the agent/dative postposition
# trimmed, and the verb is required at the end where Hinglish actually puts it.
#
# A threshold is not a measurement, and in Hinglish the two are one word apart. "pani 100 degree
# PAR ubalta hai" is a standing law about water; "paani 3 litre hai" is an observation of some
# particular water. Both are verb-final, both end in a copula, and both carry a number — so the
# SOV pattern below matches the law too, and would file "the temperature at which water boils"
# as "the temperature water was measured at". The threshold pattern in `_SEED_PATTERNS` already
# reads these correctly and keeps the number; `_extract_measurements` runs first, so it has to
# stand aside rather than win.
#
# The discriminator is the locative: a threshold states a value the process happens *at*, and
# says so with `par`/`pe`/`पर` or English "at". A measurement never does.
_THRESHOLD = re.compile(
    r"-?\d+(?:\.\d+)?\s*(?:degrees?|°\s*c|celsius|डिग्री)?\s*(?:par|pe|pr|पर)\s+\S"
    r"|\b(?:boils|melts|freezes|evaporates|condenses)\s+at\s+-?\d",
    re.IGNORECASE)

_MEASURE_SUBJECT_SOV = re.compile(
    r"^(?P<s>[^\d]+?)\s*(?:ko|ne|ka|ki|ke|को|ने|का|की|के)?\s*(?=-?\d)"
    r".*\b(?:mila|mili|mile|liya|li|liye|piya|pi|badha|badhi|badhe|"
    r"pahuncha|pahunchi|tola|napa|hai|hain|tha|thi|the|"
    r"मिला|मिली|मिले|लिया|पिया|बढ़ा|बढ़ी|पहुँचा|है|हैं|था|थी)\s*$",
    re.IGNORECASE)

_WORD = re.compile(r"[a-z]+")

# The verb governing a bare quantity names the variable it measures: "grew 20 cm" is a statement
# about growth, not merely about a length. An empty value means the verb names no variable of its
# own ("got 10 litres of water" is about water, and the noun says so) — distinct from a verb that
# is absent from this table entirely, which contributes nothing.
_MEASURE_VERBS: Dict[str, str] = {
    "grew": "growth", "grows": "growth", "grow": "growth",
    "badha": "growth", "badhi": "growth", "badhe": "growth",
    "weighs": "mass", "weighed": "mass",
    "lasted": "duration", "reached": "",
    "got": "", "gets": "", "has": "", "had": "", "received": "", "used": "",
    "took": "", "drank": "", "measured": "",
    # Hinglish. Empty value means "the verb names no variable of its own" — the noun or the unit
    # in the sentence does — which is the right reading for every receiving/possessing verb here.
    "mila": "", "mili": "", "mile": "", "liya": "", "li": "", "liye": "",
    "piya": "", "pi": "", "hai": "", "hain": "", "tha": "", "thi": "", "the": "",
    "pahuncha": "", "pahunchi": "", "tola": "mass", "napa": "",
}

# The unit's own dimension, used only when neither a noun nor a governing verb named the
# variable. It is the weakest of the three because a unit is not a variable: two different
# quantities of one entity can share one, and naming them both after it collapses exactly the
# distinction a causal fit needs.
_UNIT_DIMENSION: Dict[str, str] = {
    "litre": "volume", "litres": "volume", "liter": "volume", "liters": "volume",
    "l": "volume", "ml": "volume",
    "cm": "length", "mm": "length", "m": "length", "km": "length",
    "kg": "mass", "g": "mass",
    "degree": "temperature", "degrees": "temperature", "celsius": "temperature",
    "hour": "duration", "hours": "duration", "hr": "duration", "hrs": "duration",
    "day": "duration", "days": "duration",
}


# Wh-words that can open a turn. English only: this is the head-initial case, and every other
# language here puts its wh-word somewhere else entirely.
_WH_HEAD = frozenset({"what", "who", "where", "when", "why", "how", "which",
                      "kya", "kaun", "kahan", "kahaan", "kab", "kyun", "kyon", "kaise",
                      "क्या", "कौन", "कहाँ", "कहां", "कब", "क्यों", "कैसे"})

# Markers that make a turn a question wherever they appear in it. Hindi and Hinglish are
# verb-final with the wh-word *in situ* ("mera naam KYA hai"), so a head-word test is blind to
# them; and an imperative ask ("batao", "tell me") is a question by function whatever its mood.
_ASK_ANYWHERE = re.compile(
    r"(?:\b(?:kya|kaun|kahan|kahaan|kab|kyun|kyon|kaise|kitna|kitne|kaunsa|kaunsi)\b"
    r"|[क][्]?[य][ा]|कौन|कहाँ|कहां|कब|क्यों|कैसे|कितना"
    r"|\b(?:batao|bataao|bata|bataiye|btao|batana)\b|बताओ|बताइए|बता"
    r"|\btell\s+me\b|\bdo\s+you\s+know\b)",
    re.IGNORECASE)

# A reserved marker for "ask this relation backwards", not a predicate anything ever stores.
#
# "What causes heat?" and "what does fire cause?" are the same edge read from opposite ends, and
# an edge is only stored once — under its subject. Rather than duplicate every causal fact in
# reverse (which would double the store and make contradiction detection lie), the question table
# tags the inverse direction with this marker and `answer` scans by object instead. The `<-`
# prefix cannot collide with a real predicate because `_norm_relation` strips punctuation, so no
# sentence can ever normalise to this string.
_CAUSE_OF = "<-causes"

# Question forms -> the predicate they are asking about. Same reasoning: a floor, not a ceiling.
#
# Ordering is load-bearing. A form that names BOTH a subject and a property ("Ravi ka naam kya
# hai") must be tried before the generic one that names only a subject ("Ravi kya hai"), or the
# generic reading swallows it and asks `is_a` about a person whose *name* was wanted.
# --------------------------------------------------------------------------- #
# Retrieval — the question names an entity, the store holds a different relation about it
# --------------------------------------------------------------------------- #
# `_lookup` is a single dict lookup on ``(subject, predicate)``, and that exactness is the second
# measured bottleneck after extraction itself. "What is X?" always reads as ``is_a``, while the
# sentence that taught her about X may well have stored ``means``, ``part_of``, ``involves`` or
# ``occurs_when``. The fact was present, about the right entity, and unreachable — so the turn
# came back UNKNOWN with 521 facts sitting in the store.
#
# What this is NOT. It is not "offer the nearest memory": that is the failure `RelevanceGate`
# exists to prevent, and `_compose` refuses it outright for good reason. The discipline here is
# structural — a fact is a candidate only if the question **names its subject** — so relatedness
# never substitutes for aboutness.

# Which relations answer which question. The key is the predicate the question asked for; the
# value is what else would genuinely answer it, and how much of the confidence survives being
# answered from a neighbour rather than the relation actually requested.
#
# Nothing here is a synonym table — `_PREDICATE_ALIASES` already folds spellings of one relation.
# These are *different* relations that happen to answer the same question, which is why each
# carries a discount rather than being merged.
_PREDICATE_AFFINITY: Dict[str, Dict[str, float]] = {
    "is_a": {"is_a": 1.0, "means": 0.95, "part_of": 0.8, "occurs_when": 0.8,
             "involves": 0.7, "purpose": 0.65, "known_for": 0.6, "has_kind": 0.5,
             "consists_of": 0.5, "also_known_as": 0.5},
    "means": {"means": 1.0, "is_a": 0.95, "occurs_when": 0.8, "part_of": 0.7,
              "involves": 0.7, "purpose": 0.65},
    "has_kind": {"has_kind": 1.0, "consists_of": 0.85, "has_part": 0.7, "is_a": 0.5},
    "consists_of": {"consists_of": 1.0, "has_part": 0.9, "has_kind": 0.7},
    "causes": {"causes": 1.0, "produces": 0.9, "increases": 0.7, "decreases": 0.7,
               "occurs_when": 0.6},
    "located_in": {"located_in": 1.0, "part_of": 0.6},
    "purpose": {"purpose": 1.0, "involves": 0.7, "means": 0.6, "is_a": 0.5},
    # "What causes X?" asked about the effect and wants the cause. A forward `causes` edge *from*
    # X answers the opposite question, and answering with it is not a weaker answer — it is a
    # wrong one, pointing the arrow backwards. Scored at zero rather than merely discounted, so
    # no combination of a high subject match and a confident fact can float it into an answer.
    #
    # Measured: with `aag causes garmi` stored and nothing known to cause `aag`, "aag ka karan kya
    # hai?" came back "garmi" — she named the thing fire produces when asked what produces fire.
    _CAUSE_OF: {"causes": 0.0, "produces": 0.0, "increases": 0.0, "decreases": 0.0,
                "occurs_when": 0.6, "is_a": 0.3},
}

# Everything a relation not named above is worth when it is the only thing known about a named
# entity. Low on purpose: answering "what is X" with an unrelated property of X is better than
# UNKNOWN only when the fact is genuinely about X, and it should never outrank a real match.
# Relations that describe *what a thing is*, and may therefore answer a question whose own
# predicate could not be read — "tell me about X", or any phrasing no pattern covers. Everything
# outside this set has to be asked for by name.
#
# The line is not "important relations" but "relations that define". `owns` and `increases` are
# perfectly good facts and terrible answers to "what is X": they say something true about the
# entity and nothing about what it is.
_GENERAL_ANSWER = frozenset({
    "is_a", "means", "part_of", "occurs_when", "purpose", "involves", "consists_of", "has_kind",
})

#: Default only — the live value is :attr:`Grounder.affinity_floor`, so it can be measured rather
#: than asserted. It was asserted first, at 0.35, and the measurement said so: with it, retrieval
#: answered 45 of 120 held-out questions and got 4 right, where before it answered 3 and got 2.
_AFFINITY_FLOOR = 0.35

# The words a question is *made of* rather than *about*. Interrogatives and copulas name no
# entity, so leaving them in would let "what" match a subject containing the word "what".
_QUESTION_WORDS = frozenset({
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how", "is", "are", "was",
    "were", "do", "does", "did", "the", "a", "an", "of", "in", "on", "at", "for", "to", "and",
    "or", "me", "tell", "about", "there", "their", "it", "its", "this", "that", "these", "those",
    "can", "could", "would", "should", "kya", "kaun", "kahan", "kab", "kyun", "kaise", "hai",
    "hain", "ka", "ki", "ke", "ko", "se", "mein", "batao", "क्या", "कौन", "कहाँ", "है", "हैं",
})

_QUESTION_PATTERNS: Tuple[Tuple[str, str], ...] = (
    # --- English ---
    (r"\bwhat\s+is\s+(?:my|mera|meri)\s+(?P<p>\w+)", ""),
    (r"\bwho\s+am\s+i\b", "has_name"),
    (r"\bwhat(?:'s| is)\s+(?:the\s+)?name\s+of\s+(?P<s>.+?)\??$", "has_name"),
    (r"\btell\s+me\s+(?:my|mera|meri)\s+(?P<p>\w+)", ""),
    (r"\btell\s+me\s+(?:the\s+)?(?P<p>\w+)\s+of\s+(?P<s>.+?)\??$", ""),
    (r"\bwhere\s+do(?:es)?\s+(?P<s>.+?)\s+live", "located_in"),
    (r"\bwhere\s+(?:am\s+i|do\s+i)\s+live", "located_in"),
    (r"\bwhere\s+does\s+(?P<s>.+?)\s+work", "works_at"),
    (r"\bwhere\s+is\s+(?P<s>.+?)\??$", "located_in"),
    (r"\bwho\s+is\s+(?P<s>.+?)\??$", "is_a"),

    # --- Causal questions, English. Ahead of the generic "what is X" on purpose ------------- #
    #
    # This block closes a measured read/write asymmetry, and the asymmetry is the whole point.
    # The statement side has FIVE ways to write ``causes`` — the English rule at line 206, the
    # two Hinglish process rules below it, and `_extract_causal` for "X because Y". The question
    # side had NONE: no pattern in this table produced the predicate ``causes`` at all. So
    # "aag se garmi hoti hai" stored the key ``("aag", "causes")`` correctly, and every way of
    # asking for it back — "aag se kya hota hai", "what does fire cause" — fell through
    # `_read_question` with an empty predicate, took the `_ask_graph` branch, and came back
    # UNKNOWN. `_lookup` was never called; the fact was reachable only by iterating `facts`
    # directly. A relation she can be told and can never be asked is not a relation she has.
    (r"\bwhat\s+does\s+(?P<s>.+?)\s+(?:cause|causes|lead\s+to|leads\s+to|produce|produces)\b",
     "causes"),
    (r"\bwhat\s+(?:happens|results?)\s+(?:from|because\s+of|due\s+to|with)\s+(?P<s>.+?)\??$",
     "causes"),
    (r"\bwhat\s+is\s+(?:the\s+)?(?:effect|result|consequence)\s+of\s+(?P<s>.+?)\??$", "causes"),
    # The inverse direction — asked about the EFFECT, wanting the cause. `_CAUSE_OF` is a
    # reserved marker, not a stored predicate: nothing ever writes an edge called that. `answer`
    # reads it as "scan `causes` edges by object instead of by subject" (see `_lookup_inverse`).
    (r"\bwhy\s+does\s+(?P<s>.+?)\s+(?:happen|occur)\b", _CAUSE_OF),
    (r"\bwhat\s+causes\s+(?P<s>.+?)\??$", _CAUSE_OF),
    (r"\bwhat\s+is\s+(?:the\s+)?(?:cause|reason)\s+(?:of|for)\s+(?P<s>.+?)\??$", _CAUSE_OF),

    # The relations `_SEED_PATTERNS` learned to *write* in the expository register, now readable
    # back. Writing a relation she can never be asked for is the same read/write asymmetry the
    # causal block above this one was added to close; adding extraction without these would have
    # reproduced it exactly, one register later.
    (r"\bwhat\s+are\s+(?:the\s+)?(?:\w+\s+){0,2}?"
     r"(?:categories|types|kinds|forms|classes)\s+of\s+(?P<s>.+?)\??$", "has_kind"),
    (r"\bwhat\s+(?:are|is)\s+(?:the\s+)?(?:components|parts|stages|steps|phases)\s+of\s+"
     r"(?P<s>.+?)\??$", "has_part"),
    (r"\bwhat\s+does\s+(?P<s>.+?)\s+(?:consist\s+of|comprise)\??$", "consists_of"),
    (r"\bwhat\s+does\s+(?P<s>.+?)\s+(?:mean|refer\s+to|stand\s+for)\??$", "means"),
    (r"\bwhat\s+does\s+(?P<s>.+?)\s+involve\??$", "involves"),
    (r"\bwhen\s+does\s+(?P<s>.+?)\s+(?:occur|happen)\??$", "occurs_when"),
    (r"\bwhat\s+is\s+(?P<s>.+?)\s+(?:used\s+for|for)\??$", "purpose"),
    (r"\bwhat\s+is\s+(?P<s>.+?)\s+known\s+for\??$", "known_for"),

    (r"\bwhat\s+is\s+(?P<s>.+?)\??$", "is_a"),
    # The predicate is read out of the verb itself and folded by `_PREDICATE_ALIASES`, so this
    # one pattern covers every relation the fold table knows a verb for rather than needing a
    # line each. "need" was the absence that showed: `animal needs water` stored fine, and
    # `what does an animal need` matched nothing, because the verb list here was four words long
    # and had been extended once, for the relations that happened to exist at the time.
    (r"\bwhat\s+does\s+(?P<s>.+?)\s+(?P<p>do|own|like|know|need|needs|require|requires|"
     r"produce|produces|increase|increases|decrease|decreases)\b", ""),
    # "sparrow ko kya chahiye" — the same question, verb-final.
    (r"^(?P<s>.+?)\s+(?:ko|को)\s+(?:kya|क्या)\s+"
     r"(?P<p>chahiye|chaahiye|zaroorat|चाहिए|ज़रूरत)\b", ""),

    # --- Hinglish / Devanagari, wh-in-situ ---
    # "mera naam kya hai" / "मेरा नाम क्या है" — the Master's own property.
    (r"^(?:mera|meri|mere|मेरा|मेरी|मेरे)\s+(?P<p>\S+)\s+"
     r"(?:kya|क्या)\s*(?:hai|hain|h|है|हैं)?\s*$", ""),
    # "mera naam batao" / "मेरा नाम बताओ" — imperative ask, same fact.
    (r"^(?:mera|meri|mere|मेरा|मेरी|मेरे)\s+(?P<p>\S+)\s+"
     r"(?:batao|bataao|bata|bataiye|btao|batana|बताओ|बताइए|बता)", ""),
    # "aag ka karan kya hai" — the cause OF a thing, which is the stored edge read backwards.
    # Ahead of the generic "X ka <property> kya hai" directly below, which would otherwise match
    # first and read `karan` as an ordinary property name. Measured with the two in the other
    # order: "aag ka karan kya hai" answered `garmi` — what fire causes, offered as what causes
    # fire. A wrong answer in the confident register is worse than the silence it replaced.
    (r"^(?P<s>.+?)\s+(?:ka|ki|ke|का|की|के)\s+(?:karan|kaaran|wajah|vajah|कारण|वजह)\s+"
     r"(?:kya|क्या)\s*(?:hai|hain|है|हैं)?\s*$", _CAUSE_OF),
    # "Ravi ka naam kya hai" — someone else's property, subject and property both named.
    (r"^(?P<s>.+?)\s+(?:ka|ki|ke|का|की|के)\s+(?P<p>\S+)\s+"
     r"(?:kya|क्या)\s*(?:hai|hain|h|है|हैं)?\s*$", ""),
    (r"^(?P<s>.+?)\s+(?:ka|ki|ke|का|की|के)\s+(?P<p>\S+)\s+"
     r"(?:batao|bataao|bata|bataiye|btao|batana|बताओ|बताइए|बता)", ""),
    # "main kahan rehta hoon" / "Ravi kahan rehta hai" — location.
    (r"^(?:main|mai|मैं)\s+(?:kahan|kahaan|कहाँ|कहां)\s+(?:rehta|rehti|rahta|rahti|रहता|रहती)",
     "located_in"),
    (r"^(?P<s>.+?)\s+(?:kahan|kahaan|कहाँ|कहां)\s+(?:kaam|काम)\s+(?:karta|karti|करता|करती)",
     "works_at"),
    (r"^(?P<s>.+?)\s+(?:kahan|kahaan|कहाँ|कहां)\s+(?:rehta|rehti|rahta|rahti|रहता|रहती)",
     "located_in"),
    (r"^(?P<s>.+?)\s+(?:kahan|kahaan|कहाँ|कहां)\s+(?:hai|है)", "located_in"),
    # --- Causal questions, Hinglish / Devanagari ---------------------------------------------- #
    #
    # The mirror of the process patterns that WRITE these edges. "aag se garmi hoti hai" is
    # stored by the ablative-`se` rule; "aag se kya hota hai" is the same sentence with the
    # object replaced by the wh-word, and it has to reach the same key. The verb list is kept
    # deliberately in step with that rule's — a process verb she can be told in is one she must
    # be answerable in.
    #
    # Ahead of the generic `kya` catch-all below, which anchors the copula straight after the
    # wh-word and therefore cannot match these at all ("kya HOTA hai", not "kya hai").
    (r"^(?P<s>.+?)\s+(?:se|से)\s+(?:kya|क्या)\s+"
     r"(?:" + _PROCESS_VERBS + r")"
     r"\s*(?:hai|hain|है|हैं)?\s*$", "causes"),
    # "aag kya karti hai" — the same question with the cause as an agent rather than an ablative.
    (r"^(?P<s>.+?)\s+(?:kya|क्या)\s+(?:karta|karti|karte|करता|करती)"
     r"\s*(?:hai|hain|है|हैं)?\s*$", "causes"),
    # The inverse: named the effect, wants the cause. "garmi kyun hoti hai" / "garmi kaise hoti
    # hai" / "garmi ka karan kya hai".
    (r"^(?P<s>.+?)\s+(?:kyun|kyon|kaise|क्यों|कैसे)\s+"
     r"(?:" + _PROCESS_VERBS + r")"
     r"\s*(?:hai|hain|है|हैं)?\s*$", _CAUSE_OF),
    (r"^(?P<s>.+?)\s+(?:ka|ki|ke|का|की|के)\s+(?:karan|kaaran|wajah|vajah|कारण|वजह)\s+"
     r"(?:kya|क्या)\s*(?:hai|hain|है|हैं)?\s*$", _CAUSE_OF),

    # "Ravi kaun hai" / "Ravi kya hai" — the generic ask, deliberately last of the Hinglish set.
    (r"^(?P<s>.+?)\s+(?:kaun|कौन)\s*(?:hai|hain|है|हैं)?\s*$", "is_a"),
    (r"^(?P<s>.+?)\s+(?:kya|क्या)\s*(?:hai|hain|है|हैं)?\s*$", "is_a"),
)

_STRIP = re.compile(r"^[\s\"'`(\[]+|[\s\"'`)\].,!?;:]+$")
_ARTICLES = frozenset({"a", "an", "the", "ek"})

# How much a restatement is favoured over what is already believed, and what surviving a
# contradiction costs. The bonus is small so a confident belief is not overturned by a passing
# remark; the penalty is real so a contested answer stops sounding as sure as an uncontested one.
_RECENCY_BONUS = 0.05
_CONTESTED_PENALTY = 0.15


def _norm_relation(raw: str) -> str:
    """Normalise a relation name without deleting anything that carries meaning.

    ``\\w`` is not enough here, and the reason is specific. A Devanagari syllable is a consonant
    plus a *combining mark* — ``नाम`` is ``न`` + ``ा`` + ``म`` — and those marks are Unicode
    category ``Mn``, which :meth:`str.isalnum` reports as False and ``\\w`` therefore does not
    match. Normalising on ``\\w`` turned ``नाम`` into ``न_म``: not the empty string the ASCII
    class produced, but just as unreachable, because nothing else ever spells it that way.

    So marks are kept alongside letters and digits, and everything else collapses to a single
    separator. This is a *naming* rule, applied identically to every script.
    """
    out: List[str] = []
    for ch in str(raw or "").strip().lower():
        if ch.isalnum() or ch == "_" or unicodedata.category(ch).startswith("M"):
            out.append(ch)
        else:
            out.append("_")
    return re.sub(r"_+", "_", "".join(out)).strip("_")


def _quantity_name(low: str, at: int) -> str:
    """The variable named by the nearest governing verb before ``at``, or empty.

    Nearest rather than first: "the plant got 10 litres of water and grew 20 cm" has two verbs,
    and the one that governs a quantity is the one it follows.
    """
    found = ""
    for word in _WORD.finditer(low[:at]):
        named = _MEASURE_VERBS.get(word.group())
        if named is not None:
            found = named
    return found


def _measurements(text: str) -> List[Tuple[str, float]]:
    """Every ``(variable, value)`` a sentence reports, in the order it reports them.

    The variable is named by the noun the quantity is *of* where the sentence says so, by the
    verb that governs it where it does not, and by the unit's dimension only as a last resort.
    A quantity none of the three can name is dropped rather than stored under the unit: an
    unnamed number is not an observation of anything, and the simulator would fit it as though
    it were.
    """
    out: List[Tuple[str, float]] = []
    low = str(text or "").lower()
    if not low:
        return out
    for match in _MEASURE.finditer(low):
        try:
            value = float(match.group("q"))
        except (TypeError, ValueError):        # noqa: PERF203 — a malformed number is skipped
            continue
        if not math.isfinite(value):
            continue
        unit = (match.group("u") or "").strip()
        name = (match.group("n") or "").strip()
        if not name:
            name = _quantity_name(low, match.start())
        if not name:
            name = _UNIT_DIMENSION.get(unit, "")
        if name:
            out.append((name, value))
    return out


def _as_object(value: float) -> str:
    """A measured value as the object of a triple.

    Integral values lose the trailing ``.0`` because the object is read back with ``float()``
    by :meth:`nyxara.njp.concepts.ConceptGenesis.observe_triples` and shown to the Master as
    written — "10 litres" should not come back as "10.0".
    """
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _kind_head(phrase: str) -> str:
    """The bare kind inside a kind phrase — ``largest animal on Earth`` → ``animal``.

    Approximate on purpose, and cheap: the head of an English noun phrase is the last noun before
    its first modifier. No part-of-speech tagger is involved, so the guards are crude and do real
    work — a trailing participle or adverb is dropped (``application designed to…`` heads at
    ``application``, not ``designed``), and a phrase that reduces to nothing yields nothing rather
    than a preposition.

    What this is *for*: a shared kind. ``blue whale is_a largest animal on Earth`` and
    ``African elephant is_a largest land animal`` name one kind between them and no two of these
    phrases are ever equal, so nothing downstream can tell that both are animals. Measured on the
    bundled corpus, 443 ``is_a`` objects reduce to 133 heads, 59 of which have two or more members
    — which is the first point at which a kind exists to generalise over at all.
    """
    segment = _KIND_HEAD_BREAK.split(str(phrase or ""), 1)[0]
    words = [w for w in re.findall(r"[^\W\d_]+(?:['-][^\W\d_]+)*", segment, re.UNICODE)
             if w.lower() not in _KIND_HEAD_SKIP]
    # Participles and adverbs are modifiers that survived the split because nothing preceded them.
    while words and (words[-1].lower().endswith(("ing", "ed", "ly"))):
        words.pop()
    head = words[-1].lower() if words else ""
    return head if len(head) > 2 else ""


def _clauses(text: str) -> List[str]:
    """A sentence's independently-assertable clauses, in order.

    One sentence routinely carries more than one claim — *"Supervised learning involves labelled
    examples, while unsupervised learning involves unlabelled data"* is two facts about two
    subjects — and the extractor stops at the first pattern that matches. Before this, the second
    half of every such sentence was unreachable no matter how many patterns existed.

    Conservative on purpose. Splitting on bare ``and`` would shred "bread and butter" and every
    enumeration in the corpus, so only connectives that genuinely coordinate two *independent*
    claims are cut. A text with nothing to split returns itself, and the caller's behaviour on a
    single-clause sentence is then byte-for-byte what it was.
    """
    raw = str(text or "").strip()
    if not raw:
        return []
    out: List[str] = []
    for sentence in _SENTENCE_SPLIT.split(raw):
        for clause in _CLAUSE_SPLIT.split(sentence):
            clause = clause.strip()
            # Two words cannot state a relation, and admitting them here would hand the pattern
            # loop fragments that match its loosest rules on nothing at all.
            if len(clause.split()) >= 3:
                out.append(clause)
    return out or [raw]


def _split_condition(clause: str) -> Tuple[str, str]:
    """``(main, condition)`` — the claim, and the circumstance it is asserted under.

    Returns the clause unchanged with an empty condition where the surface states none, so an
    unconditional sentence takes exactly the path it always took.

    The condition is *not* thrown away once split, which is the entire point. "If the temperature
    falls below 0°C, water freezes" carries a real relation — ``water --freezes-->`` — and a real
    restriction on it, and a store that keeps the first without the second holds a claim that is
    plainly false: water does not, in general, freeze.
    """
    text = str(clause or "").strip()
    if not text:
        return "", ""
    for rx in _CONDITIONS:
        match = rx.match(text)
        if match is None:
            continue
        main = (match.group("main") or "").strip(" ,")
        cond = (match.group("cond") or "").strip(" ,")
        # A split that leaves nothing to assert is not a split. "If so, then yes" parses and means
        # nothing; returning it would replace a whole clause with a fragment.
        if len(main.split()) >= 2 and len(cond.split()) >= 2:
            return main, cond
    return text, ""


def _modality_of(text: str) -> Tuple[str, float]:
    """What the surface claimed about its own certainty, and what that costs the confidence.

    Strongest reading wins, and "strongest" means *most restrictive*: a sentence carrying both
    "usually" and "may" is a hedged claim, because the hedge is the weaker commitment and a
    speaker who makes both has made the weaker one.
    """
    found, factor = "", 1.0
    for rx, name, scale in _MODALITY:
        if rx.search(str(text or "")):
            if scale < factor or not found:
                found, factor = name, min(factor, scale)
    return found, factor


def _temporal_of(text: str) -> str:
    """The time the claim is pinned to, where the surface pins it. Empty is the normal case."""
    match = _TEMPORAL.search(str(text or ""))
    return (match.group("t") or "").strip() if match is not None else ""


def _list_items(raw: str) -> List[str]:
    """The items of an enumeration, cleaned, de-duplicated, order preserved.

    Items shorter than a word or longer than a clause are dropped: the first is punctuation the
    splitter tripped on, and the second is a sentence that ran past the list and would enter the
    store as an "entity" nothing ever matches again.
    """
    out: List[str] = []
    seen: Set[str] = set()
    for part in _LIST_SPLIT.split(str(raw or "")):
        item = _clean(part)
        if not item or len(item.split()) > 8:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _clean(text: str) -> str:
    """Trim punctuation and a leading article, keeping the entity itself intact."""
    out = _STRIP.sub("", str(text or "")).strip()
    parts = out.split()
    if parts and parts[0].lower() in _ARTICLES:
        parts = parts[1:]
    return " ".join(parts).strip()


class Grounder:
    """Turns turns into structure, and answers from it.

    Every collaborator is optional and injected. With none of them this still extracts and answers
    from its own in-process store, which is what keeps it testable and keeps a cold brain useful;
    with them it writes into the kernel's real :class:`~nyxara.memory.graph.KnowledgeGraph` and
    concept ladder, so what NJP learns is visible to the rest of the system rather than trapped
    in a second private store.
    """

    def __init__(self, *, graph: Any = None, ladder: Any = None, llm: Any = None,
                 min_confidence: float = 0.35, known_floor: float = 0.75,
                 recall_floor: float = 0.6,
                 learn_patterns: bool = True, max_learned: int = 512,
                 pattern_floor: float = 0.34, pattern_min_trials: int = 6,
                 affinity_floor: float = _AFFINITY_FLOOR) -> None:
        self.graph = graph
        self.ladder = ladder
        self._llm = llm
        self._llm_tried = llm is not None
        self.min_confidence = float(min_confidence)
        # What a *retrieved* answer must score to be offered at all. Separate from
        # `min_confidence` because it gates a different decision — not "is this fact worth
        # keeping" but "does this fact answer the question that was asked" — and because it was
        # set by measurement rather than by taste. On 200 held-out questions, moving retrieval
        # from `min_confidence` (0.35) to 0.6 took 58 answers down to 27 while leaving all 4
        # correct ones in place: it removed only wrong answers.
        self.recall_floor = max(0.0, min(1.0, float(recall_floor)))
        self.known_floor = float(known_floor)
        self.learn_patterns = bool(learn_patterns)
        self.max_learned = max(0, int(max_learned))
        self.pattern_floor = float(pattern_floor)
        self.pattern_min_trials = max(2, int(pattern_min_trials))
        # What a relation nobody listed as answering this question is worth. The single most
        # consequential number in retrieval: it decides whether an unrelated property of the right
        # entity may be offered as an answer at all.
        self.affinity_floor = max(0.0, min(1.0, float(affinity_floor)))

        self.patterns: List[Pattern] = [
            Pattern(regex=rx, predicate=pred, learned=False) for rx, pred in _SEED_PATTERNS
        ]
        self._compiled: Dict[str, Any] = {}
        # Local mirror of every assertion, so grounding works with no KnowledgeGraph attached and
        # so contradiction detection has a store even before the kernel wires one in.
        self.facts: Dict[Tuple[str, str], List[GroundedTriple]] = {}
        self.turns = 0
        self.grounded_turns = 0
        self.answered = 0
        self.unknown = 0
        self.contradictions_found = 0
        self.patterns_learned = 0
        self.patterns_pruned = 0
        self.llm_calls = 0
        # Retrieval, counted separately from `answered` so "she found the entity but not the
        # relation" is visible as its own outcome rather than hidden inside either success or
        # failure. A high `recalls` with a low `recalled` says the questions name things she has
        # never been told about; the reverse says her relations and her questions disagree.
        self.recalls = 0
        self.recalled = 0
        # How often each word has reached no entity. A running tally rather than a per-turn
        # snapshot, because "he keeps saying this and I still have nothing for it" is a gap and
        # "he mentioned it once" is not, and the snapshot could not tell them apart.
        self._ungrounded_seen: Dict[str, int] = {}

    # ---- the one call ----------------------------------------------------- #
    def ground(self, text: str, *, intent: Any = None, deep: bool = False) -> GroundingResult:
        """Ground one turn: answer it if it asks, learn from it if it tells.

        ``deep`` permits the fluent surface to be consulted for a sentence the deterministic core
        could not parse. It is off by default because that call is ~5 s against an on-device model
        — far too slow to sit in every turn — and the pulse is the right place to spend it.
        """
        out = GroundingResult(text=str(text or ""))
        t0 = time.perf_counter()
        try:
            if not out.text.strip():
                return out
            self.turns += 1
            out.concepts = self._concepts(out.text)

            if self._is_question(out.text, intent):
                out.is_question = True
                out.answer = self.answer(out.text)
                if out.answer.answered:
                    self.answered += 1
                else:
                    self.unknown += 1
                out.ungrounded = self._ungrounded(out.concepts, tally=True)
                return out

            triples = self._extract(out.text)
            if not triples and deep:
                triples = self._extract_deep(out.text)
                if triples:
                    out.learned_pattern = self._learn_from(out.text, triples[0])

            for triple in triples:
                clash = self._contradicts(triple)
                if clash is not None:
                    out.contradictions.append((clash, triple))
                    self.contradictions_found += 1
                    self._revise(clash, triple)
                self._assert(triple)
                out.triples.append(triple)

            if out.triples:
                self.grounded_turns += 1
            self._observe_concepts(out)
            out.ungrounded = self._ungrounded(out.concepts, tally=not out.triples)
            return out
        except Exception:  # noqa: BLE001 — a failed grounding never breaks a turn
            return out
        finally:
            out.ms = (time.perf_counter() - t0) * 1000.0

    # ---- extraction ------------------------------------------------------- #
    def _rx(self, pattern: str) -> Any:
        got = self._compiled.get(pattern)
        if got is None:
            got = re.compile(pattern, re.IGNORECASE)
            self._compiled[pattern] = got
        return got

    def _extract(self, text: str) -> List[GroundedTriple]:
        """Deterministic extraction: every claim the sentence states, not merely the first.

        Matched against the cleaned text **as written**, not a lowercased copy: the patterns are
        already case-insensitive, and matching on a lowered string meant every captured object was
        stored lowercased, so "Jay" came back as "jay" and she could not say the Master's name the
        way he wrote it.

        **Why this is a loop over clauses now.** The pattern loop stops at the first match, which
        is right — a sentence states one relation and the most specific reading of it should win.
        It was applied to the whole *text*, which is not the same thing: a sentence coordinating
        two claims had its second half discarded by a rule that exists to disambiguate one. So the
        first-match discipline is kept exactly, and moved inside a clause. On a single-clause
        sentence the two are identical, which is what keeps every prior behaviour intact.
        """
        out: List[GroundedTriple] = []
        low = _clean(text)
        if not low:
            return out

        # "X because Y" states a cause outright. Ground each half on its own and record the claim
        # between them, so the sentence yields the happening *and* the causal link rather than
        # falling through every pattern and yielding nothing — which is what it did before.
        causal = _BECAUSE.match(low)
        if causal is not None:
            return self._extract_causal(causal, text)

        # A measurement is handled before the pattern loop for the same reason "because" is: the
        # loop stops at the first match, and a sentence reporting two quantities of one entity
        # would keep one number and discard the other — which destroys the pairing rather than
        # merely losing half of it.
        measured = self._extract_measurements(text)
        if measured:
            return measured

        for clause in _clauses(text):
            main, condition = _split_condition(clause)
            modality, factor = _modality_of(clause)
            temporal = _temporal_of(clause)
            found = self._extract_clause(main, text)
            for triple in found:
                triple.condition = condition
                triple.modality = modality
                triple.temporal = temporal
                # Stated uncertainty lowers what she may claim, and it does so *after* the
                # extractor has scored its own reliability. The two are different questions —
                # "did I read this right" and "how strongly was it said" — and multiplying keeps
                # them separable rather than letting a hedge look like a bad parse.
                triple.confidence = round(triple.confidence * factor, 6)
            out.extend(found)
        out.extend(self._kind_edges(out, text))
        self._prune_patterns()
        return out

    def _kind_edges(self, triples: List[GroundedTriple], text: str) -> List[GroundedTriple]:
        """The bare kind alongside the full kind phrase, so two members can meet.

        Added rather than substituted. ``blue whale is_a largest animal on Earth`` is the more
        informative claim and is kept; ``blue whale is_a animal`` is the one that connects, and
        without it the first is a fact about a phrase nothing else will ever mention.

        Only for ``is_a``. ``part_of`` already names an entity as its object, and a "head" of a
        ``means`` definition is not a kind — it is the first noun of a paraphrase, and asserting
        membership from it would invent a taxonomy out of sentence shape.
        """
        out: List[GroundedTriple] = []
        seen = {(t.subject.lower(), t.object.lower()) for t in triples}
        for triple in triples:
            if triple.predicate != "is_a" or len(triple.object.split()) < 2:
                continue
            head = _kind_head(triple.object)
            if not head or head == triple.subject.lower():
                continue
            if (triple.subject.lower(), head) in seen:
                continue
            seen.add((triple.subject.lower(), head))
            out.append(GroundedTriple(
                subject=triple.subject, predicate="is_a", object=head,
                # Below the phrase it was derived from: this is a parse of a parse, and the guards
                # that produce it are heuristics about English rather than a relation anyone said.
                confidence=round(triple.confidence * 0.8, 6),
                source="kind-head", text=text,
                condition=triple.condition, temporal=triple.temporal,
                modality=triple.modality))
        return out

    def _extract_clause(self, clause: str, text: str) -> List[GroundedTriple]:
        """Every triple one clause states. First matching pattern wins, as it always has."""
        out: List[GroundedTriple] = []
        low = _clean(clause)
        if not low:
            return out

        # An appositive names the same entity twice. Recorded, then removed, so the sentence it
        # was embedded in is read as the sentence it actually is — before this, the interruption
        # sat between the subject and its verb and defeated every pattern anchored on that verb.
        alias = _APPOSITIVE.match(low)
        if alias is not None:
            subject = self.resolve(alias.group("s") or "")
            other = _clean(alias.group("alias") or "")
            if subject and other:
                out.append(GroundedTriple(
                    subject=subject, predicate="also_known_as", object=other,
                    confidence=0.85, source="appositive", text=text))
                low = _clean(f"{alias.group('s')} {alias.group('rest')}")

        # An enumeration states several claims at once and must not be reduced to its first item.
        # Handled before the pattern loop for the same reason a measurement is: the loop is built
        # to pick one reading, and here every item IS the reading.
        listed = self._extract_enumeration(low, text)
        if listed:
            return out + listed

        # A superlative is two claims — the kind and the rank — and both are wanted.
        ranked = self._extract_superlative(low, text)
        if ranked:
            return out + ranked

        for pattern in list(self.patterns):
            try:
                match = self._rx(pattern.regex).match(low)
            except Exception:  # noqa: BLE001 — a bad learned regex is dropped, never fatal
                self._drop(pattern)
                continue
            if match is None:
                continue
            groups = match.groupdict()
            # An empty seed predicate means the surface named it ("my colour is blue"), so the
            # relation itself is read out of the sentence rather than assumed. A `v` group means
            # the pattern matched a *happening*, and the verb itself is the predicate.
            verb = _EVENT_VERBS.get(_clean(groups.get("v", "") or "").lower(), "")
            predicate = pattern.predicate or verb or _clean(groups.get("p", "") or "")
            subject = self.resolve(groups.get("s") or SELF_ENTITY)
            obj = _clean(groups.get("o", "") or "")
            # A threshold statement ("water boils at 100") names no object but does name the
            # number the process turns on. Dropping it would reduce the sentence to "water
            # boils" — true, useless, and unusable by any simulation, which needs the value to
            # answer "and at 40?". The quantity IS the object of a threshold.
            if not obj:
                obj = _clean(groups.get("q", "") or "")
            # An intransitive happening genuinely has no object — "the glass broke" is complete.
            # Demanding one here is what kept every such sentence unextracted.
            if not predicate or (not obj and not verb):
                pattern.misses += 1
                continue
            pattern.hits += 1
            out.append(GroundedTriple(
                subject=subject, predicate=self._predicate(predicate), object=obj,
                confidence=0.9 if not pattern.learned else 0.6 + 0.3 * pattern.precision,
                source="pattern" if not pattern.learned else "learned-pattern", text=text))
            break
        return out

    def _extract_superlative(self, clause: str, text: str) -> List[GroundedTriple]:
        """``X is the largest Y`` → what X is, and where it ranks.

        Kept as two triples rather than one because they answer different questions and only one
        of them connects. ``blue whale --is_a--> animal on Earth`` shares a shape with
        ``African elephant --is_a--> land animal``, which is what lets a kind be induced over
        them; ``blue whale --largest--> animal on Earth`` never would, because the predicate is
        different for every superlative and nothing would ever group.
        """
        out: List[GroundedTriple] = []
        match = _SUPERLATIVE.match(clause)
        if match is None:
            return out
        subject = self.resolve(match.group("s") or "")
        kind = _clean(match.group("o") or "")
        rank = _clean(match.group("sup") or "").lower()
        if not subject or not kind:
            return out
        out.append(GroundedTriple(
            subject=subject, predicate="is_a", object=kind,
            confidence=0.85, source="superlative", text=text))
        out.append(GroundedTriple(
            subject=subject, predicate=self._predicate(f"is_{rank}"), object=kind,
            confidence=0.85, source="superlative", text=text))
        return out

    def _extract_enumeration(self, clause: str, text: str) -> List[GroundedTriple]:
        """``the three types of X are A, B and C`` → one triple per item.

        This is the construction that matters most for connecting what she knows, and the reason
        is a measured one: after 1,200 corpus pairs the store held 521 facts across 468 subjects —
        roughly one fact each, so almost nothing shared a subject and the schema inducer had
        nothing to generalise over. An enumeration is the opposite shape. It attaches several
        objects to one subject under one predicate in a single sentence, which is exactly the
        evidence :mod:`nyxara.njp.core` needs to induce a role and :mod:`nyxara.njp.concepts`
        needs to find an invariant.

        Two items is the floor. A "list" of one is a plain relation and the pattern loop reads it
        better, keeping the object whole rather than as a degenerate list.
        """
        out: List[GroundedTriple] = []
        match = _ENUMERATION.match(clause)
        predicate = ""
        if match is not None:
            predicate = _KIND_PREDICATE.get((match.group("kind") or "").lower(), "has_kind")
        else:
            match = _COMPOSITION.match(clause)
            if match is None:
                return out
            predicate = "consists_of"
        subject = self.resolve(match.group("s") or "")
        items = _list_items(match.group("list") or "")
        if not subject or len(items) < 2:
            return out
        for item in items:
            out.append(GroundedTriple(
                subject=subject, predicate=self._predicate(predicate), object=item,
                # Below a single-relation parse on purpose. A list is read by a looser rule than a
                # named relation and its items are segmented by punctuation, so it earns slightly
                # less than a sentence whose whole shape had to match.
                confidence=0.8, source="enumeration", text=text))
        return out

    def _extract_measurements(self, text: str) -> List[GroundedTriple]:
        """A measurement sentence, as one triple per quantity it reports.

        This is the one extractor that deliberately returns *every* reading rather than the
        first. Two variables observed together are what a causal coefficient is fitted to, so
        keeping only the leading quantity would leave the simulator with numbers it could never
        pair — the state it was actually measured in.

        Returns nothing at all where the sentence names no subject or no nameable quantity,
        which is what keeps a standing law ("water boils at 100") on the threshold pattern that
        already reads it rather than being mistaken for an observation of some particular thing.
        """
        out: List[GroundedTriple] = []
        pairs = _measurements(text)
        if not pairs:
            return out
        cleaned = _clean(text)
        # A standing law states the value its process turns on; it is not a reading taken of
        # anything. `_SEED_PATTERNS` has a threshold rule that keeps both the subject and the
        # number, and this extractor runs before the pattern loop — so the only way that rule
        # can be reached is for this one to decline.
        if _THRESHOLD.search(cleaned):
            return out
        # SVO first, then the verb-final reading. Order matters only for a sentence both could
        # claim, and there the English one is right: it anchors on a verb it has actually
        # identified, whereas the SOV pattern anchors on "ends in a copula", which "the plant has
        # 3 litres" also does.
        head = _MEASURE_SUBJECT.match(cleaned) or _MEASURE_SUBJECT_SOV.match(cleaned)
        if head is None:
            return out
        subject = self.resolve(head.group("s"))
        # A subject carrying a digit is an over-capture, not an entity — the signature of the
        # word-order bug this pair of patterns exists to avoid. Refusing here means a sentence
        # neither reading parsed cleanly contributes nothing, rather than contributing a variable
        # named after its own measurement that nothing will ever match again.
        if not subject or any(c.isdigit() for c in subject):
            return out
        for name, value in pairs:
            out.append(GroundedTriple(
                subject=subject, predicate=self._predicate(name), object=_as_object(value),
                confidence=0.9, source="measurement", text=text))
        return out

    def _extract_causal(self, match: Any, text: str) -> List[GroundedTriple]:
        """``X because Y`` → both halves grounded, plus the ``causes`` link between them.

        The head of each half is its extracted subject where the half parsed, and the cleaned
        clause itself where it did not. That mixed granularity is deliberate: refusing to record
        the link unless *both* halves parse cleanly would discard the causal claim — the one part
        of the sentence that is stated outright rather than inferred — because of a parser gap on
        the other side of the conjunction. A coarse entity that :mod:`nyxara.njp.world` can count
        beats a link that was never recorded.
        """
        out: List[GroundedTriple] = []
        try:
            effect_text = _clean(match.group("effect") or "")
            cause_text = _clean(match.group("cause") or "")
            if not effect_text or not cause_text:
                return out

            def _half(clause: str) -> Tuple[List[GroundedTriple], str]:
                # Recurses through `_extract`, so a half is parsed by exactly the same rules as a
                # standalone sentence. `_BECAUSE` cannot re-match a half that has no "because" in
                # it, so this terminates after one level.
                triples = self._extract(clause)
                head = triples[0].subject if triples else clause
                return triples, head

            effect_triples, effect_head = _half(effect_text)
            cause_triples, cause_head = _half(cause_text)
            out.extend(cause_triples)
            out.extend(effect_triples)
            out.append(GroundedTriple(
                subject=cause_head, predicate="causes", object=effect_head,
                confidence=0.85, source="causal-clause", text=text))
            return out
        except Exception:  # noqa: BLE001 — an unparsed causal sentence grounds to nothing
            return out

    def _extract_deep(self, text: str) -> List[GroundedTriple]:
        """Ask the fluent surface for a sentence the core could not parse.

        Constrained hard to one triple in a fixed shape. Anything else is discarded — a language
        model asked for structure will happily return prose, and prose here would be a fact she
        never actually extracted.
        """
        llm = self._get_llm()
        if llm is None:
            return []
        try:
            self.llm_calls += 1
            reply = llm.generate(
                "Extract the single fact as a triple.\n"
                "Reply with exactly one line: subject|predicate|object\n"
                "Use snake_case for the predicate. No other text.\n\n"
                f"Sentence: {text}\nTriple:",
                system="You extract structured facts. You never add facts that are not present.",
                max_tokens=32)
            parts = [p.strip() for p in str(reply or "").strip().splitlines()[0].split("|")]
            if len(parts) != 3 or not all(parts):
                return []
            subject, predicate, obj = parts
            return [GroundedTriple(
                subject=self.resolve(subject), predicate=self._predicate(predicate),
                object=_clean(obj), confidence=0.65, source="fluent-surface", text=text)]
        except Exception:  # noqa: BLE001
            return []

    def _learn_from(self, text: str, triple: GroundedTriple) -> bool:
        """Generalise a successful deep parse into a reusable pattern.

        The object is the slot that varies; the words around it are what identified the relation.
        So the surface is kept verbatim with the object position opened up. This is narrow on
        purpose — a broad induced regex matches everything and poisons every later turn — and the
        pattern still has to earn its place through :meth:`_prune_patterns`.
        """
        if not self.learn_patterns or len(self.patterns) >= self.max_learned:
            return False
        try:
            low = _clean(text).lower()
            obj = _clean(triple.object).lower()
            if not obj or obj not in low:
                return False
            prefix = low.split(obj)[0]
            if not prefix.strip():
                return False
            regex = f"^{re.escape(prefix.strip())}\\s+(?P<o>.+)$"
            if any(p.regex == regex for p in self.patterns):
                return False
            self.patterns.append(Pattern(regex=regex, predicate=triple.predicate, learned=True))
            self.patterns_learned += 1
            return True
        except Exception:  # noqa: BLE001
            return False

    def _prune_patterns(self) -> None:
        """Drop learned patterns that have been judged and are not paying. Seeds are never cut."""
        try:
            for pattern in list(self.patterns):
                if not pattern.learned or pattern.trials < self.pattern_min_trials:
                    continue
                if pattern.precision < self.pattern_floor:
                    self._drop(pattern)
        except Exception:  # noqa: BLE001
            pass

    def _drop(self, pattern: Pattern) -> None:
        try:
            self.patterns.remove(pattern)
            self.patterns_pruned += 1
        except ValueError:
            pass

    # ---- naming ----------------------------------------------------------- #
    def resolve(self, name: str) -> str:
        """Map a surface name onto the entity it denotes.

        First and second person both land on real entities rather than being dropped: without this
        "my name is Jay" and "what is my name" would attach to two different subjects and she could
        store a fact she could never retrieve.
        """
        cleaned = _clean(name)
        low = cleaned.lower()
        if not low:
            return SELF_ENTITY
        if low in _FIRST_PERSON:
            return SELF_ENTITY
        if low in _SECOND_PERSON or low in _NYXARA:
            return "NYXARA"
        # "my brother", "mera ghar" — possessive of the Master, kept as a distinct entity.
        head = low.split()[0]
        if head in _FIRST_PERSON and len(low.split()) > 1:
            return f"{SELF_ENTITY}'s {' '.join(cleaned.split()[1:])}"
        return cleaned

    @staticmethod
    def _predicate(raw: str) -> str:
        """Normalise a relation name so every spelling of it reaches the same edge.

        Two jobs, both naming rather than knowledge. Shape first — ``"has name"`` and ``"has_name"``
        are one relation — then a synonym fold, because the sentence that *stores* a fact and the
        question that *asks* for it rarely use the same word: "my name is Jay" writes ``has_name``
        while "what is my name" reads ``name``, and without folding them together she stores an
        answer she can never retrieve. Anything not listed passes through untouched; this decides
        what an edge is *called*, never what is true.

        Normalised by :func:`_norm_relation` rather than on ``[a-z0-9]``. The ASCII class deleted
        every Devanagari character outright, so ``"नाम"`` normalised to the empty string and came
        back as ``related_to`` — she could not name the relation the Master had just asked about,
        in the script he asked in. Folding a script into nothing is not normalisation.
        """
        out = _norm_relation(raw)
        return _PREDICATE_ALIASES.get(out, out) or "related_to"

    # ---- assertion and contradiction -------------------------------------- #
    def _assert(self, triple: GroundedTriple) -> None:
        """Record the fact locally and, when one is attached, in the KnowledgeGraph."""
        key = (triple.subject.lower(), triple.predicate)
        self.facts.setdefault(key, []).append(triple)
        try:
            if self.graph is None:
                return
            self.graph.add_entity(triple.subject)
            self.graph.add_entity(triple.object)
            self.graph.add_triple(triple.subject, triple.predicate, triple.object,
                                  confidence=triple.confidence)
        except Exception:  # noqa: BLE001 — the local mirror still holds it
            pass

    def _revise(self, prior: GroundedTriple, incoming: GroundedTriple) -> None:
        """Weigh the two readings of a functional relation and let the stronger stand.

        Detecting a contradiction and then leaving the old answer in place would be the worst of
        both worlds — she would announce the clash and keep answering with the superseded fact.
        So one of them wins here, and the winner is **not** simply the newer one: a firmly-held
        belief should not be overturned by a weaker passing remark. Recency gets a modest bonus,
        because a Master who restates something is usually correcting it, and it is a bonus rather
        than a rule.

        Both sides come out ``contested``, and the survivor's confidence is cut: a claim that has
        been contradicted is genuinely less certain than one that never was, and saying so is the
        difference between revising a belief and overwriting one.
        """
        try:
            prior.contested = incoming.contested = True
            if incoming.confidence + _RECENCY_BONUS >= prior.confidence:
                prior.superseded = True
                incoming.confidence = max(0.0, incoming.confidence - _CONTESTED_PENALTY)
            else:
                incoming.superseded = True
                prior.confidence = max(0.0, prior.confidence - _CONTESTED_PENALTY)
        except Exception:  # noqa: BLE001
            pass

    def _contradicts(self, triple: GroundedTriple) -> Optional[GroundedTriple]:
        """Does this clash with something already believed?

        Only for relations that can hold **one** value at a time. "likes tea" and "likes coffee"
        are both fine; two different names for the same subject are not. Treating every repeated
        predicate as a clash would make ordinary accumulation look like conflict.
        """
        try:
            if triple.predicate not in _FUNCTIONAL:
                return None
            for prior in self.facts.get((triple.subject.lower(), triple.predicate), []):
                if prior.superseded:
                    continue        # already revised away; it cannot clash a second time
                if prior.object.strip().lower() != triple.object.strip().lower():
                    return prior
            return None
        except Exception:  # noqa: BLE001
            return None

    # ---- answering -------------------------------------------------------- #
    def answer(self, question: str) -> Answer:
        """Answer from structure, or say plainly that she does not know."""
        out = Answer()
        try:
            low = _clean(question).lower()
            subject, predicate = self._read_question(low)
            if not predicate:
                return self._ask_graph(question, out)

            # An inverse question reads the same edges from the far end; everything downstream —
            # corroboration, the tri-state, the `why` line — is identical, because it is the same
            # evidence. Only the direction of the scan and which end becomes the answer differ.
            inverse = predicate == _CAUSE_OF
            found = (self._lookup_inverse(subject, "causes") if inverse
                     else self._lookup(subject, predicate))
            if not found:
                # The graph may know it even when the local mirror does not (it survives restarts).
                #
                # Retrieval is deliberately NOT tried here. Answering from a neighbouring relation
                # is the weakest thing she can do that still counts as an answer, and returning it
                # from this method would hand it to `_compose` as a grounded answer — which stops
                # `brain._deliberate` from ever running, because that is gated on the answer still
                # being empty. Measured: `sparrow needs water`, inherited through `sparrow is_a
                # bird`, was replaced by the flat fact `sparrow is_a bird`. A retrieved fact
                # pre-empting a derived one is a strictly worse answer arriving strictly earlier.
                # `Grounder.recall` is public and the brain calls it after deliberation instead.
                return self._ask_graph(question, out)

            best = max(found, key=lambda t: t.confidence)
            out.triples = found
            out.text = best.subject if inverse else best.object
            out.confidence = best.confidence
            # Corroboration is what separates believing from knowing: one source that could be
            # wrong is a belief however confident it sounds. Agreement is counted on whichever end
            # is being *answered with* — on an inverse question two records naming the same cause
            # corroborate each other even where the effects they name differ in wording.
            agreed = out.text
            corroborated = len({t.source for t in found
                                if (t.subject if inverse else t.object) == agreed}) > 1
            # A contested claim is never KNOWN, however confident the surviving side is: something
            # in her own record disagrees with it, and stating that as fact is exactly the move
            # this tri-state exists to prevent.
            out.state = (Epistemic.KNOWN
                         if (best.confidence >= self.known_floor and corroborated
                             and not best.contested)
                         else Epistemic.BELIEVED)
            out.why = f"{best.subject} —{best.predicate}→ {best.object}"
            if best.contested:
                out.why += " (contested; an earlier answer was superseded)"
            return out
        except Exception:  # noqa: BLE001
            return out

    def _lookup(self, subject: str, predicate: str) -> List[GroundedTriple]:
        """Live facts only. A superseded belief stays on record but is never answered with."""
        return [t for t in self.facts.get((subject.lower(), predicate), []) if not t.superseded]

    def _lookup_inverse(self, obj: str, predicate: str) -> List[GroundedTriple]:
        """The same edges read backwards: everything that ``predicate``\\ s *into* ``obj``.

        A linear scan, deliberately. The forward index exists because the overwhelming majority of
        questions name a subject; building and maintaining a second index for the minority that
        name an object would double the write cost of every fact to save a walk over a store that
        is bounded by what one conversation could state. If that ever stops being true the fix is
        an index, not a cache — a stale reverse index would answer with facts already superseded.
        """
        low = str(obj or "").strip().lower()
        if not low:
            return []
        out: List[GroundedTriple] = []
        for (_subject, stored), triples in self.facts.items():
            if stored != predicate:
                continue
            out.extend(t for t in triples
                       if not t.superseded and t.object.strip().lower() == low)
        return out

    def answer_by_recall(self, question: str, out: Optional[Answer] = None) -> Answer:
        """Answer from a different relation about the entity that was asked about.

        **Never ``KNOWN``.** Corroboration is what licenses stating something as fact, and having
        retrieved a thing is not corroboration of it — the retrieval says the fact is *about* what
        was asked, which is a claim about relevance, not about truth. Conflating the two is how a
        store starts asserting whatever it happened to find.

        The confidence carried out is the retrieval score, which is the fact's own confidence
        already multiplied by how squarely the question named its subject and by how well the
        relation answers what was asked. It is therefore always at or below what the fact itself
        was worth: this path can lower confidence and can never raise it.
        """
        out = out if out is not None else Answer()
        try:
            got = self.recall(question, limit=1)
            if not got or got[0].triple is None:
                return out
            best = got[0]
            triple = best.triple
            # Below this the fact is about the right entity but answers a different question, and
            # a wrong answer confidently placed is worse than the honest UNKNOWN it replaces.
            if best.score < self.recall_floor:
                return out
            out.triples = [triple]
            out.text = triple.object or triple.subject
            out.confidence = round(best.score, 4)
            out.state = Epistemic.BELIEVED
            where = f" via {best.via}" if best.hops else ""
            out.why = (f"recalled{where}: {triple.subject} —{triple.predicate}→ {triple.object}"
                       f" (asked for a different relation)")
            # A conditional fact answered without its condition is a false answer. Say the
            # condition out loud rather than dropping it — this is the whole reason the field
            # exists on the triple.
            if triple.condition:
                out.why += f"; holds when {triple.condition}"
            return out
        except Exception:  # noqa: BLE001
            return out

    # ---- retrieval -------------------------------------------------------- #
    def recall(self, question: str, *, limit: int = 5,
               subject_floor: float = 0.6) -> List[RecalledFact]:
        """Facts about the entity this question names, ranked by how well they answer it.

        The pipeline, and every step is a method or a table in this file::

            question → content tokens → candidate subjects → related entities (aliases)
                     → facts on those subjects → predicate affinity → ranked

        **The floor is the safety property.** ``subject_floor`` is a fraction of the *subject's*
        own words that the question must contain, not a fraction of the question's — so a
        two-word entity named in full scores 1.0 inside a twenty-word question, while a long
        stored subject sharing one incidental word with the question scores far below the floor
        and is never a candidate. This is what makes the result *about* what was asked rather
        than merely near it, and it is why the pendulum-law failure cannot recur through this
        path: no phrasing of "how are you" names the pendulum.

        Returns an empty list rather than a weak best guess when nothing clears the floor.
        Empty is the honest and common outcome, and the caller keeps its UNKNOWN.
        """
        out: List[RecalledFact] = []
        try:
            tokens = self._question_tokens(question)
            if not tokens:
                return out
            wanted = self._read_question(_clean(question).lower())[1]

            for subject, subject_score in self._candidate_subjects(tokens, subject_floor):
                for triple in self._facts_of(subject):
                    affinity = self._affinity(wanted, triple.predicate, subject)
                    out.append(RecalledFact(
                        triple=triple, subject_score=subject_score, affinity=affinity,
                        score=subject_score * affinity * triple.confidence))
                # One hop, and only through an edge that means "the same thing, named
                # differently" or "the kind this belongs to". Following an arbitrary relation
                # would walk to a genuinely different entity and answer about that instead,
                # which is the drift this whole module is built to refuse.
                for alias in self._neighbours(subject):
                    for triple in self._facts_of(alias):
                        affinity = self._affinity(wanted, triple.predicate, alias)
                        out.append(RecalledFact(
                            triple=triple, subject_score=subject_score, affinity=affinity,
                            hops=1, via=subject,
                            # A hop costs. A fact stated of the alias is evidence about this
                            # entity, but weaker evidence than one stated of it directly, and
                            # collapsing the two would make a chain indistinguishable from a
                            # statement.
                            score=subject_score * affinity * triple.confidence * 0.7))
            out.sort(key=lambda r: r.score, reverse=True)
            # De-duplicate on the claim itself: the same fact reached directly and through an
            # alias is one fact, and reporting it twice would look like corroboration.
            seen: Set[Tuple[str, str, str]] = set()
            unique: List[RecalledFact] = []
            for got in out:
                key = got.triple.as_tuple() if got.triple else ("", "", "")
                if key in seen:
                    continue
                seen.add(key)
                unique.append(got)
            self.recalls += 1
            if unique:
                self.recalled += 1
            return unique[:max(1, int(limit))]
        except Exception:  # noqa: BLE001 — retrieval never breaks a turn
            return out

    def _affinity(self, wanted: str, predicate: str, subject: str) -> float:
        """How well this relation answers the question that was asked.

        Two regimes, and the second is the one worth explaining. When the question's own
        predicate was read, :data:`_PREDICATE_AFFINITY` says what else would answer it.

        When it was *not* read — an unusual phrasing, "tell me about X" — there is no evidence
        about which relation is wanted, and the honest prior is a uniform one over the relations
        this subject actually has. So a subject known by a single relation answers at full weight
        (there is nothing to be ambiguous between), while one known by five answers at a fifth
        (any of them could have been meant, and four would be the wrong answer). That is the
        ambiguity the score should carry, and a fixed floor could only ever be wrong in one
        direction or the other.
        """
        # Asked for by name, and held under that very name. Nothing outranks that, and it has to
        # be checked before the table: most relations have no row of their own, so a question that
        # named one of those would otherwise fall through to the unread-question branch and be
        # refused for not being a definition — having asked for it explicitly.
        if wanted and predicate == wanted:
            return 1.0
        affinities = _PREDICATE_AFFINITY.get(wanted, {})
        if affinities:
            return affinities.get(predicate, self.affinity_floor)
        # A relation that does not describe *what a thing is* cannot serve as the answer to a
        # question nobody could parse. This is the measured fix, and it was not the fix expected:
        # of 60 retrieved answers over 200 held-out questions, 29 came from `owns` and
        # `increases` — 0 correct between them, mean F1 0.003 for `owns` — and they were immune to
        # `affinity_floor` because they never went through the table branch at all. They arrived
        # here, where a subject known by exactly one relation was awarded full weight.
        #
        # The defect was not the prior. It was that "I could not read the question" was allowed to
        # license any relation whatsoever, including one that answers a question nobody asked.
        if predicate not in _GENERAL_ANSWER:
            return 0.0
        low = str(subject or "").lower()
        relations = {p for (s, p) in self.facts if s == low}
        return 1.0 / max(1, len(relations))

    @staticmethod
    def _question_tokens(question: str) -> Set[str]:
        """The content words of a question — what it is *about*, not what it is made of."""
        words = "".join(c if c.isalnum() else " " for c in str(question or "").lower()).split()
        return {w for w in words if w not in _QUESTION_WORDS and len(w) > 2}

    def _candidate_subjects(self, tokens: Set[str],
                            floor: float) -> List[Tuple[str, float]]:
        """Every stored subject the question actually names, best first.

        Scored by how much of the *subject* the question contains. Normalising by the subject
        rather than by the question is the whole point: "what is a Wittig reaction" is a short
        question that names its entity completely, and a measure normalised by the question would
        punish it for the words it spends asking.
        """
        scores: Dict[str, float] = {}
        for subject, _predicate in self.facts:
            if subject in scores:
                continue
            parts = {w for w in subject.replace("_", " ").split() if len(w) > 2}
            if not parts:
                continue
            shared = parts & tokens
            if not shared:
                continue
            score = len(shared) / len(parts)
            if score >= floor:
                scores[subject] = score
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:4]

    def _facts_of(self, subject: str) -> List[GroundedTriple]:
        """Every live fact stated of one subject, across all its relations."""
        out: List[GroundedTriple] = []
        low = str(subject or "").lower()
        for (stored, _predicate), triples in self.facts.items():
            if stored != low:
                continue
            out.extend(t for t in triples if not t.superseded)
        return out

    def _neighbours(self, subject: str) -> List[str]:
        """Entities that are the same thing under another name, or the kind it belongs to."""
        out: List[str] = []
        for predicate in ("also_known_as", "is_a", "part_of"):
            for triple in self._lookup(subject, predicate):
                name = triple.object.strip().lower()
                if name and name != str(subject or "").lower() and name not in out:
                    out.append(name)
        return out[:3]

    def history(self, subject: str, predicate: str) -> List[GroundedTriple]:
        """Everything ever believed about this pair, superseded entries included, oldest first."""
        return sorted(self.facts.get((subject.lower(), self._predicate(predicate)), []),
                      key=lambda t: t.at)

    def _read_question(self, low: str) -> Tuple[str, str]:
        """``(subject, predicate)`` the question is asking about — empty predicate if unreadable."""
        for regex, predicate in _QUESTION_PATTERNS:
            match = self._rx(regex).search(low)
            if match is None:
                continue
            groups = match.groupdict()
            pred = predicate or self._predicate(_clean(groups.get("p", "") or ""))
            subject = self.resolve(groups.get("s") or SELF_ENTITY)
            if pred:
                return subject, pred
        return SELF_ENTITY, ""

    def _ask_graph(self, question: str, out: Answer) -> Answer:
        """Last resort: the KnowledgeGraph's own NL query. Absent graph ⇒ honestly UNKNOWN."""
        try:
            if self.graph is None:
                return out
            got = self.graph.ask(question)
            text = str(getattr(got, "text", "") or "").strip()
            confidence = float(getattr(got, "confidence", 0.0) or 0.0)
            if not text or text.lower().startswith("no known") or confidence <= 0.0:
                return out
            out.text, out.confidence = text, confidence
            out.state = (Epistemic.KNOWN if confidence >= self.known_floor
                         else Epistemic.BELIEVED)
            out.why = "knowledge graph"
            return out
        except Exception:  # noqa: BLE001
            return out

    # ---- concepts --------------------------------------------------------- #
    def _concepts(self, text: str) -> List[str]:
        try:
            from nyxara.njp.tongue import concepts_in
            return concepts_in(text)
        except Exception:  # noqa: BLE001
            return [w for w in str(text or "").lower().split() if len(w) > 2][:32]

    def _observe_concepts(self, result: GroundingResult) -> None:
        """Feed the ladder, so *it* can discover the types nothing here declares.

        The features handed over are the relations the entity actually participates in. That is
        what makes the discovery real: Jay and Ravi end up grouped because both have a name and a
        location, not because a table said "person".
        """
        try:
            if self.ladder is None:
                return
            for triple in result.triples:
                self.ladder.observe(triple.subject, self._features(triple.subject))
                self.ladder.observe(triple.object, self._features(triple.object))
        except Exception:  # noqa: BLE001
            pass

    def _features(self, entity: str) -> List[str]:
        """Every relation this entity takes part in, as ``predicate`` / ``predicate_of`` features."""
        out: Set[str] = set()
        low = entity.lower()
        try:
            for (subject, predicate), triples in self.facts.items():
                if subject == low:
                    out.add(predicate)
                for triple in triples:
                    if triple.object.lower() == low:
                        out.add(f"{predicate}_of")
        except Exception:  # noqa: BLE001
            pass
        return sorted(out)

    def persistent_unknowns(self, *, min_times: int = 2, limit: int = 8) -> List[Tuple[str, int]]:
        """Words that have reached no entity more than once, commonest first.

        The per-turn ``ungrounded`` list is a snapshot and is overwritten every turn, so a word
        the Master has used five times and a word he mentioned once in passing were
        indistinguishable — and the only consumer rendered either as the same sentence. A tally
        is what makes the difference legible, and it is the same discipline curiosity already
        applies to failures: once is noise, repeatedly is a gap.
        """
        try:
            # A word she uses as a *relation* is not a thing she is missing. "lives" and "works"
            # never appear as entities because they are not entities, and asking "what is lives?"
            # is the kind of question that makes a gap list unreadable — the signal is the noun
            # she has no node for, not the verb she has been parsing correctly all along.
            rows = [(word, n) for word, n in self._ungrounded_seen.items()
                    if n >= int(min_times)]
            rows.sort(key=lambda kv: (-kv[1], kv[0]))
            return rows[: max(0, int(limit))]
        except Exception:  # noqa: BLE001
            return []

    def _ungrounded(self, concepts: Sequence[str], *, tally: bool = False) -> List[str]:
        """Concepts that reach no entity — the words that are still just words to her."""
        out: List[str] = []
        try:
            known = {s for s, _p in self.facts}
            for (_s, _p), triples in self.facts.items():
                for triple in triples:
                    known.add(triple.object.lower())
            for concept in concepts:
                if concept.lower() not in known:
                    out.append(concept)
                    # Only a sentence that grounded *nothing* contributes to the tally. A
                    # sentence that produced a triple has explained its own words: "Ravi lives in
                    # Pune" leaves "lives" unmatched as an entity because it is the relation, not
                    # a thing she is missing, and "what is lives?" is exactly the question that
                    # makes a gap list unreadable.
                    if tally:
                        word = concept.lower()
                        self._ungrounded_seen[word] = self._ungrounded_seen.get(word, 0) + 1
            # A word that has since been grounded stops being a gap, so its tally goes rather
            # than lingering as a question she has already answered.
            for seen in [w for w in self._ungrounded_seen if w in known]:
                self._ungrounded_seen.pop(seen, None)
        except Exception:  # noqa: BLE001
            pass
        return out[:16]

    # ---- questions -------------------------------------------------------- #
    @staticmethod
    def _is_question(text: str, intent: Any = None) -> bool:
        """Trust the intent reader when one is attached; fall back to the surface.

        The surface fallback used to read the **first word only**, which is a statement about
        English rather than about questions. English is head-initial — "*what* is my name" — so a
        head check finds the wh-word. Hindi and Hinglish are verb-final and leave the wh-word
        *in situ*: "mera naam **kya** hai" has ``mera`` at the head, so the check returned False,
        the turn was never treated as a question, and the answer path was never entered at all.
        Measured: ``mera naam kya hai`` returned ``''`` while ``what is my name`` returned ``Jay``
        — the same fact, reachable in one language and not the other.

        So the Hinglish and Devanagari markers are searched for **anywhere** in the turn, which is
        where that grammar actually puts them. This does not over-trigger on statements: a
        statement of fact ("mera naam Jay hai") carries no wh-word, and it is the wh-word that is
        being looked for, not the possessive.

        Imperative requests are included on purpose. "tell me my name" and "mera naam batao" are
        questions in every sense that matters here — they ask for a fact she may hold — and both
        were previously read as statements and sent to the *extractor*, which tried to learn a
        relation from them.
        """
        try:
            low = str(text or "").strip().lower()
            if not low:
                return False
            if low.endswith("?"):
                return True
            # A fronted conditional is a statement, whatever word it starts with. Checked before
            # the head test because "when" and "if" head both a question and a conditional, and
            # what separates them is the comma and the main clause after it: "When *does* rain
            # fall" inverts its auxiliary, "When a model memorises noise, it generalises poorly"
            # closes a whole clause first. `_split_condition` already recognises exactly this
            # construction for extraction, so it is asked rather than a second rule invented.
            if low.endswith(".") and _split_condition(text)[1]:
                return False
            head = low.split()[0]
            if head in _WH_HEAD:
                return True
            # Wh-in-situ (Hinglish/Devanagari) and imperative asks, anywhere in the turn.
            if _ASK_ANYWHERE.search(low):
                return True
            # The intent reader is consulted last, and it is not allowed to overrule a sentence
            # its author punctuated as a statement.
            #
            # The failure this closes, measured end to end: "Overfitting occurs when a model is
            # trained too much." was read as a question, because the intent reader's stated reason
            # was "it contains a question word" — the *subordinating* "when", which joins a clause
            # here rather than asking anything. `ground` then took the answering branch and the
            # sentence was never extracted at all. Every conditional statement was affected, which
            # is precisely the class of sentence whose condition this module had just learned to
            # keep: "if X, then Y", "X occurs when C", "when C, E".
            #
            # A trailing full stop is weak evidence on its own, and decisive here: all three
            # positive surface tests have already declined, so the only thing that made this a
            # question was a conjunction inside a clause the author closed with a period.
            if intent is not None and getattr(intent, "kind", "") == "question":
                return not low.endswith(".")
            return False
        except Exception:  # noqa: BLE001
            return False

    # ---- the fluent surface ----------------------------------------------- #
    def _get_llm(self) -> Any:
        if not self._llm_tried:
            self._llm_tried = True
            try:
                from nyxara.mind.llm import LLM
                self._llm = LLM()
            except Exception:  # noqa: BLE001 — no model means the deterministic core only
                self._llm = None
        return self._llm

    # ---- reporting -------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        learned = [p for p in self.patterns if p.learned]
        return {
            "turns": self.turns, "grounded_turns": self.grounded_turns,
            "facts": sum(len(v) for v in self.facts.values()),
            "subjects": len({s for s, _p in self.facts}),
            "answered": self.answered, "unknown": self.unknown,
            "contradictions_found": self.contradictions_found,
            "patterns": len(self.patterns), "patterns_learned": self.patterns_learned,
            "patterns_live": len(learned), "patterns_pruned": self.patterns_pruned,
            "llm_calls": self.llm_calls,
            "recalls": self.recalls, "recalled": self.recalled,
            "recall_rate": round(self.recalled / self.recalls, 4) if self.recalls else None,
            "graph_attached": self.graph is not None,
            "ladder_attached": self.ladder is not None,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": [t.to_dict() | {"text": t.text[:120]}
                      for triples in self.facts.values() for t in triples],
            "patterns": [p.to_dict() for p in self.patterns if p.learned],
            "counters": {"turns": self.turns, "grounded": self.grounded_turns,
                         "answered": self.answered, "unknown": self.unknown,
                         "learned": self.patterns_learned, "pruned": self.patterns_pruned},
        }

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            for row in (d.get("facts") or []):
                triple = GroundedTriple(
                    subject=str(row.get("subject", "")), predicate=str(row.get("predicate", "")),
                    object=str(row.get("object", "")),
                    confidence=float(row.get("confidence", 0.0)),
                    source=str(row.get("source", "")), text=str(row.get("text", "")),
                    # Restored, not dropped. A conditional claim reloaded without its condition
                    # is not a slightly poorer memory of it — it is a different and usually false
                    # claim, which is the entire reason the field exists. The same holds for a
                    # hedge: "X may cause Y" coming back as "X causes Y" across a restart would
                    # let her state on Tuesday what she carefully qualified on Monday.
                    condition=str(row.get("condition", "")),
                    temporal=str(row.get("temporal", "")),
                    modality=str(row.get("modality", "")),
                    # The revision itself, and this is the one that bites hardest. `_lookup`
                    # answers from live facts only; a superseded entry stays on record so she can
                    # say what she used to believe, and returning it as live undoes the
                    # contradiction that retired it. Measured: told "my name is Jay" then "my name
                    # is Raj", she answers Raj — and after a reload she answers **Jay**, because
                    # both came back live and the retracted one won on confidence order. A store
                    # that forgets its retractions overnight is worse than one that never revised.
                    superseded=bool(row.get("superseded", False)),
                    contested=bool(row.get("contested", False)))
                if triple.subject and triple.predicate:
                    self.facts.setdefault(
                        (triple.subject.lower(), triple.predicate), []).append(triple)
            for row in (d.get("patterns") or []):
                regex = str(row.get("regex", ""))
                if not regex or any(p.regex == regex for p in self.patterns):
                    continue
                self.patterns.append(Pattern(
                    regex=regex, predicate=str(row.get("predicate", "")), learned=True,
                    hits=int(row.get("hits", 0)), misses=int(row.get("misses", 0))))
            counters = d.get("counters") or {}
            self.turns = int(counters.get("turns", 0))
            self.grounded_turns = int(counters.get("grounded", 0))
            self.answered = int(counters.get("answered", 0))
            self.unknown = int(counters.get("unknown", 0))
            self.patterns_learned = int(counters.get("learned", 0))
            self.patterns_pruned = int(counters.get("pruned", 0))
        except Exception:  # noqa: BLE001 — a corrupt sidecar leaves a freshly-born grounder
            pass


# Surface spellings that denote the same edge. Purely nominal: it decides what a relation is
# *called*, never what is true of it. Kept small — the point is that the writer and the reader of
# a fact agree on the edge name, not to enumerate a vocabulary.
_PREDICATE_ALIASES: Dict[str, str] = {
    "name": "has_name", "naam": "has_name", "called": "has_name",
    "lives_in": "located_in", "live_in": "located_in", "location": "located_in",
    "city": "located_in", "rehta": "located_in", "rehti": "located_in",
    "job": "works_at", "work": "works_at", "works": "works_at", "employer": "works_at",
    "owns": "owns", "own": "owns", "has": "owns",
    "kind": "is_a", "type": "is_a",
    # The relations a seed pattern renames on the way in. "animal needs water" is *stored* as
    # `requires` by the pattern that reads it, and "what does an animal need" asks for `needs` —
    # so without this fold the fact is written under a key no question ever forms. Same class of
    # asymmetry as a relation with no question form at all, one layer further down and harder to
    # see, because here both halves work and simply disagree about the name.
    "needs": "requires", "need": "requires", "requires": "requires", "require": "requires",
    "chahiye": "requires", "zaroorat": "requires", "चाहिए": "requires", "ज़रूरत": "requires",
    # Only the FORWARD-facing words fold to `causes`. "effect of X" and "X ka asar" both ask what
    # X leads to, which is the edge as stored. The words for the other direction — "cause",
    # "karan", "wajah" — are deliberately absent: folding them here would send "aag ka karan kya
    # hai" (what causes fire) to a forward lookup on `(aag, causes)` and answer with what fire
    # causes, which is not merely unhelpful but backwards. Those forms are read by the
    # `_CAUSE_OF` question patterns instead, which is the only place the direction is visible.
    "effect": "causes", "effects": "causes", "result": "causes", "asar": "causes",
    "increase": "increases", "raises": "increases", "badhata": "increases",
    "decrease": "decreases", "reduces": "decreases", "ghatata": "decreases",
    # The same relations as the Master actually writes them. A fold table that only speaks one
    # language means a fact stored in English is unreachable from the question asked in Hinglish
    # — which is precisely the gap this list exists to close, applied inconsistently.
    "नाम": "has_name",
    "ghar": "located_in", "shehar": "located_in", "sheher": "located_in", "shahar": "located_in",
    "घर": "located_in", "शहर": "located_in", "जगह": "located_in", "jagah": "located_in",
    "kaam": "works_at", "naukri": "works_at", "काम": "works_at", "नौकरी": "works_at",
    "umar": "age", "उम्र": "age",
}

# Relations that can hold only one value at a time. A second value contradicts rather than adds.
# Relations that can hold exactly one value at a time, so a second value is a contradiction
# rather than an addition.
#
# `is_a` is deliberately **not** here, and used to be. A thing belongs to many kinds at once —
# sparrow is_a bird, sparrow is_a animal — and that is not a conflict, it is what a taxonomy *is*.
# With it listed, every second kind retracted the first, so the hierarchy destroyed itself as fast
# as it was built. Measured: `blue whale is_a animal` arrived as a contradiction of
# `blue whale is_a animal on Earth` and was superseded on the turn it was learned, which left
# `core._inherit` walking a graph whose every kind edge but one was marked dead. `learner`
# reported 9 schemas and `answers_given` 0 across 1,320 turns.
_FUNCTIONAL = frozenset({
    "has_name", "located_in", "born_in", "age", "birthday",
    "works_at", "capital_of", "married_to",
})
