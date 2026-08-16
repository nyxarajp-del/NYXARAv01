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
    """One extracted relation, with everything needed to judge and revise it."""

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

    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def to_dict(self) -> Dict[str, Any]:
        return {"subject": self.subject, "predicate": self.predicate, "object": self.object,
                "confidence": round(self.confidence, 4), "source": self.source,
                "superseded": self.superseded, "contested": self.contested}


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
    (r"^(?P<s>.+?)\s+is\s+(?:a|an)\s+(?P<o>.+)$", "is_a"),
    (r"^(?P<s>.+?)\s+works?\s+(?:at|for)\s+(?P<o>.+)$", "works_at"),
    (r"^(?P<s>.+?)\s+(?:owns|has)\s+(?:a\s+|an\s+)?(?P<o>.+)$", "owns"),
    (r"^(?P<s>.+?)\s+likes?\s+(?P<o>.+)$", "likes"),
    (r"^(?P<s>.+?)\s+knows?\s+(?P<o>.+)$", "knows"),
    (r"^(?P<s>.+?)\s+is\s+(?:part\s+of|inside)\s+(?P<o>.+)$", "part_of"),
    (r"^(?P<s>.+?)\s+(?:causes?|leads?\s+to)\s+(?P<o>.+)$", "causes"),
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

    (r"\bwhat\s+is\s+(?P<s>.+?)\??$", "is_a"),
    (r"\bwhat\s+does\s+(?P<s>.+?)\s+(?:do|own|like|know)", ""),

    # --- Hinglish / Devanagari, wh-in-situ ---
    # "mera naam kya hai" / "मेरा नाम क्या है" — the Master's own property.
    (r"^(?:mera|meri|mere|मेरा|मेरी|मेरे)\s+(?P<p>\S+)\s+"
     r"(?:kya|क्या)\s*(?:hai|hain|h|है|हैं)?\s*$", ""),
    # "mera naam batao" / "मेरा नाम बताओ" — imperative ask, same fact.
    (r"^(?:mera|meri|mere|मेरा|मेरी|मेरे)\s+(?P<p>\S+)\s+"
     r"(?:batao|bataao|bata|bataiye|btao|batana|बताओ|बताइए|बता)", ""),
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
                 learn_patterns: bool = True, max_learned: int = 512,
                 pattern_floor: float = 0.34, pattern_min_trials: int = 6) -> None:
        self.graph = graph
        self.ladder = ladder
        self._llm = llm
        self._llm_tried = llm is not None
        self.min_confidence = float(min_confidence)
        self.known_floor = float(known_floor)
        self.learn_patterns = bool(learn_patterns)
        self.max_learned = max(0, int(max_learned))
        self.pattern_floor = float(pattern_floor)
        self.pattern_min_trials = max(2, int(pattern_min_trials))

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
        """Deterministic extraction. First pattern that matches wins.

        Matched against the cleaned text **as written**, not a lowercased copy: the patterns are
        already case-insensitive, and matching on a lowered string meant every captured object was
        stored lowercased, so "Jay" came back as "jay" and she could not say the Master's name the
        way he wrote it.
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
        self._prune_patterns()
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
            if intent is not None and getattr(intent, "kind", "") == "question":
                return True
            low = str(text or "").strip().lower()
            if not low:
                return False
            if low.endswith("?"):
                return True
            head = low.split()[0]
            if head in _WH_HEAD:
                return True
            # Wh-in-situ (Hinglish/Devanagari) and imperative asks, anywhere in the turn.
            return bool(_ASK_ANYWHERE.search(low))
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
                    source=str(row.get("source", "")), text=str(row.get("text", "")))
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
_FUNCTIONAL = frozenset({
    "has_name", "located_in", "born_in", "age", "birthday",
    "works_at", "is_a", "capital_of", "married_to",
})
