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

import re
import time
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
)

# Question forms -> the predicate they are asking about. Same reasoning: a floor, not a ceiling.
_QUESTION_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\bwhat\s+is\s+(?:my|mera|meri)\s+(?P<p>\w+)", ""),
    (r"\bwho\s+am\s+i\b", "has_name"),
    (r"\bwhat(?:'s| is)\s+(?:the\s+)?name\s+of\s+(?P<s>.+?)\??$", "has_name"),
    (r"\bwhere\s+do(?:es)?\s+(?P<s>.+?)\s+live", "located_in"),
    (r"\bwhere\s+(?:am\s+i|do\s+i)\s+live", "located_in"),
    (r"\bwhere\s+is\s+(?P<s>.+?)\??$", "located_in"),
    (r"\bwho\s+is\s+(?P<s>.+?)\??$", "is_a"),
    (r"\bwhat\s+is\s+(?P<s>.+?)\??$", "is_a"),
    (r"\bwhat\s+does\s+(?P<s>.+?)\s+(?:do|own|like|know)", ""),
)

_STRIP = re.compile(r"^[\s\"'`(\[]+|[\s\"'`)\].,!?;:]+$")
_ARTICLES = frozenset({"a", "an", "the", "ek"})

# How much a restatement is favoured over what is already believed, and what surviving a
# contradiction costs. The bonus is small so a confident belief is not overturned by a passing
# remark; the penalty is real so a contested answer stops sounding as sure as an uncontested one.
_RECENCY_BONUS = 0.05
_CONTESTED_PENALTY = 0.15


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
                out.ungrounded = self._ungrounded(out.concepts)
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
            out.ungrounded = self._ungrounded(out.concepts)
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
            # relation itself is read out of the sentence rather than assumed.
            predicate = pattern.predicate or _clean(groups.get("p", "") or "")
            subject = self.resolve(groups.get("s") or SELF_ENTITY)
            obj = _clean(groups.get("o", "") or "")
            if not predicate or not obj:
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
        """
        out = re.sub(r"[^a-z0-9]+", "_", str(raw or "").strip().lower()).strip("_")
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

            found = self._lookup(subject, predicate)
            if not found:
                # The graph may know it even when the local mirror does not (it survives restarts).
                return self._ask_graph(question, out)

            best = max(found, key=lambda t: t.confidence)
            out.triples = found
            out.text = best.object
            out.confidence = best.confidence
            # Corroboration is what separates believing from knowing: one source that could be
            # wrong is a belief however confident it sounds.
            corroborated = len({t.source for t in found if t.object == best.object}) > 1
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

    def _ungrounded(self, concepts: Sequence[str]) -> List[str]:
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
        except Exception:  # noqa: BLE001
            pass
        return out[:16]

    # ---- questions -------------------------------------------------------- #
    @staticmethod
    def _is_question(text: str, intent: Any = None) -> bool:
        """Trust the intent reader when one is attached; fall back to the surface."""
        try:
            if intent is not None and getattr(intent, "kind", "") == "question":
                return True
            low = str(text or "").strip().lower()
            if low.endswith("?"):
                return True
            head = low.split()[0] if low.split() else ""
            return head in {"what", "who", "where", "when", "why", "how", "which",
                            "kya", "kaun", "kahan", "kab", "kyun", "kaise"}
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
}

# Relations that can hold only one value at a time. A second value contradicts rather than adds.
_FUNCTIONAL = frozenset({
    "has_name", "located_in", "born_in", "age", "birthday",
    "works_at", "is_a", "capital_of", "married_to",
})
