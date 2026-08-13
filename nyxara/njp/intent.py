"""NYXARA · nyx/intent.py — what was actually asked for (🎯, NYX V.02, gap 4).

NYX V.01 decided what kind of turn had arrived with one regular expression
(``brain.py:56``): does it start with a wh-word, or end with a question mark. Measured, that
gives

    nlp.intent("mera code fix kar do lekin pehle test chala")  ->  kind='statement'

which misses two things at once: it is an **imperative**, and it carries an **ordering
constraint** — run the tests *before* touching the code. Acting on the first half of that
sentence and ignoring the second is exactly the failure mode a tool-using agent must not have.

:class:`IntentReader` pulls real structure out of an instruction:

* ``kind`` — question, command, statement, or greeting, decided from mood rather than from
  punctuation, in English and in Hinglish/Devanagari (which V.01 could not see at all).
* ``goal`` and ``actions`` — what is being asked for.
* ``ordering`` — ``(before, after)`` pairs from "pehle X phir Y", "first X then Y",
  "X ke baad Y", "before/after".
* ``polarity`` — per action: do it, or explicitly *don't* ("mat karo", "don't", "avoid").
* ``constraints`` — the clauses that narrow the request ("only", "sirf", "without", "bina",
  "make sure", "must").
* ``conditions`` — "if/agar … then/to".
* ``scope`` — the paths, files and entities named.
* ``open_questions`` — what is genuinely under-specified, so she can *ask* instead of assume.
* ``contradictions`` — asked to do and not do the same thing.
* ``readings`` — when two readings are live, **both** are returned with confidences. This is
  the point of the module: ambiguity is surfaced, not resolved by coin flip.

Honest framing (the rule this repo keeps, see ``docs/CAPABILITIES.md``):

* This is a **parser, not a mind-reader**. It reads mood, clause structure, negation scope and
  a closed list of ordering/constraint markers. It does not know what you meant and does not
  pretend to: a genuinely half-stated instruction produces an entry in ``open_questions``, and
  :mod:`nyxara.nyx.dialogue` turns that into a question rather than a confident answer.
* There is no parts-of-speech model and no dependency parse aboard, so verb detection is
  marker-driven — an imperative in a shape nobody listed reads as a statement. That is a
  false negative, which is the safe direction: it under-claims rather than inventing a command.
* Confidence is about the *parse*, never about whether the request is a good idea.

Pure standard library, on top of :mod:`nyxara.nyx.lingua` so every script is readable. Every
public method is fail-soft: on any error it degrades to a null result rather than breaking a
turn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.tongue import Lingua, content_tokens, tokenize

__all__ = ["Reading", "Intent", "IntentReader", "KINDS"]

KINDS = ("question", "command", "statement", "greeting")

# --------------------------------------------------------------------------- #
# Markers. Every list here is closed and explicit — this is a marker-driven parser, and
# pretending otherwise would be the same mistake the regex it replaces made.
# --------------------------------------------------------------------------- #

# Interrogative openers and particles. Hinglish and Devanagari are first-class.
_Q_WORDS = frozenset("""
what which who whom whose when where why how whether
kya kaun kaunsa kaunsi kab kahan kaha kaise kese kyu kyun kyon kitna kitni kitne
क्या कौन कौनसा कब कहाँ कहां कैसे क्यों कितना कितनी कितने
""".split())
_Q_AUX = frozenset("""
is are was were am do does did can could shall should will would may might must have has had
""".split())

# Imperative markers. English is verb-initial, which needs a verb list; Hinglish and Devanagari
# mark the imperative morphologically, which is far more reliable.
_IMPERATIVE_VERBS = frozenset("""
add remove delete fix build make write run test check show list find search open close create
update change rename move copy install uninstall commit push pull merge rebase clean refactor
explain summarise summarize compare generate compile deploy start stop restart print give tell
""".split())
_IMPERATIVE_HI = frozenset("""
karo kar karna kijiye kijiiye kariye banao banaao likho likh dikhao dikha chalao chala
bhejo bhej lao la do de dedo dena hatao hata jodo jod badlo badal padho padh dekho dekh
suno sun batao bata rakho rakh nikalo nikal shuru band
करो कर करना कीजिए बनाओ लिखो दिखाओ चलाओ भेजो लाओ दो हटाओ जोड़ो बदलो पढ़ो देखो सुनो बताओ रखो निकालो
""".split())
_POLITE_IMPERATIVE = frozenset("""
please kindly pls plz kripya kripaya zara zara-sa कृपया ज़रा जरा
""".split())

_GREETINGS = frozenset("""
hi hello hey yo namaste namaskar salaam hola sup greetings gm gn
नमस्ते नमस्कार
""".split())

# Ordering. Each pattern names which side runs first.
_ORDER_FIRST_THEN = (
    # "pehle X phir/uske baad Y", "first X then Y"
    re.compile(r"\b(?:pehle|pahle|first|firstly)\b(?P<a>.+?)"
               r"\b(?:phir|fir|then|uske\s+baad|iske\s+baad|after\s+that|baad\s+me[ni]?)\b"
               r"(?P<b>.+)$", re.I | re.S),
    re.compile(r"\bपहले\b(?P<a>.+?)\b(?:फिर|उसके\s+बाद|बाद\s+में)\b(?P<b>.+)$", re.S),
)
_ORDER_BEFORE = (
    re.compile(r"(?P<b>.+?)\bbefore\b(?P<a>.+)$", re.I | re.S),
    re.compile(r"(?P<a>.+?)\b(?:se\s+pehle|ke\s+pehle)\b(?P<b>.+)$", re.I | re.S),
    re.compile(r"(?P<a>.+?)\b(?:से\s+पहले)\b(?P<b>.+)$", re.S),
)
_ORDER_AFTER = (
    # Anaphoric "after that" points *backwards*, so the clause before it is the earlier one.
    # Without this case first, "add a CHANGELOG, and after that commit it" parses exactly
    # inside out — which is the kind of error that only shows up once a tool has already run.
    re.compile(r"(?P<a>.+?),?\s+(?:and\s+)?(?:after|uske|iske|इसके|उसके)\s+"
               r"(?:that|this|baad|बाद)\s*(?:me[ni]?|में)?\s*,?\s*(?P<b>.+)$", re.I | re.S),
    re.compile(r"(?P<b>.+?)\bafter\b(?P<a>.+)$", re.I | re.S),
    re.compile(r"(?P<a>.+?)\bke\s+baad\b(?P<b>.+)$", re.I | re.S),
    re.compile(r"(?P<a>.+?)\bके\s+बाद\b(?P<b>.+)$", re.S),
)

#: "can you fix this", "could you run the tests" — grammatically a question, pragmatically an
#: instruction. Both readings are genuinely live and the module refuses to silently pick one.
_POLITE_REQUEST = re.compile(r"^\s*(?:can|could|would|will)\s+(?:you|u)\b", re.I)
#: A bare "but first" / "lekin pehle" flips the order of the clauses around it.
_BUT_FIRST = re.compile(r"\b(?:lekin|but|magar|par)\s+(?:pehle|pahle|first)\b", re.I)

# Negation. Scope runs from the marker to the end of its clause.
_NEGATORS = frozenset("""
not dont don't do-not never avoid without except
mat nahi nahin naa bina
मत नहीं ना बिना
""".split())

# Constraint markers — clauses that narrow rather than add.
_CONSTRAINT_MARKERS = (
    "only", "just", "sirf", "keval", "सिर्फ", "केवल",
    "without", "bina", "बिना", "except", "unless", "jab tak", "जब तक",
    "make sure", "ensure", "must", "should", "dhyan rakh", "ध्यान रख",
    "at most", "at least", "no more than", "zyada nahi", "ज़्यादा नहीं",
)
_CONDITION_MARKERS = ("if ", "agar ", "अगर ", "when ", "in case", "jab ", "जब ")

# Clause separators. Kept explicit so Devanagari danda and Hinglish conjunctions both split.
_CLAUSE_SPLIT = re.compile(
    r"\s*(?:[;।,]|\b(?:but|lekin|magar|however)\b|\.\s+)\s*", re.I)

# Things that look like a scope: paths, dotted names, quoted strings, flags.
_SCOPE_RE = re.compile(r"(?:(?:[\w.\-]+/)+[\w.\-]*|[\w\-]+\.[a-z]{1,5}\b|--?[a-z][\w\-]*|"
                       r"`[^`]+`|\"[^\"]+\"|'[^']+')", re.I)

#: Pronouns with no antecedent in a single turn — the commonest way an instruction is half-said.
_DANGLING = frozenset("""
it this that those these them they he she him her
wo woh usko uske isko iske unko unke ye yeh
वो वह उसको उसके इसको इसके ये यह
""".split())

_VAGUE = frozenset("""
some few several many any something anything everything stuff things soon later properly
kuch kuchh thoda zyada jaldi theek sahi
कुछ थोड़ा ज़्यादा जल्दी ठीक सही
""".split())
#: Anaphors that an ordering marker has already accounted for — "after *that*" points at the
#: clause beside it, so flagging it as an unresolved reference would be noise, not honesty.
_ORDER_ANAPHORS = frozenset({"that", "this", "uske", "iske", "इसके", "उसके"})


@dataclass
class Reading:
    """One way the turn can be understood, with what it commits her to."""

    kind: str = "statement"
    goal: str = ""
    actions: List[str] = field(default_factory=list)
    confidence: float = 0.0
    why: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "goal": self.goal, "actions": list(self.actions),
                "confidence": round(self.confidence, 4), "why": self.why}


@dataclass
class Intent:
    """The structure of one instruction — everything a tool-using agent needs before acting."""

    text: str = ""
    kind: str = "statement"
    goal: str = ""
    actions: List[str] = field(default_factory=list)
    ordering: List[Tuple[str, str]] = field(default_factory=list)   # (before, after)
    constraints: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    polarity: Dict[str, bool] = field(default_factory=dict)         # action -> do / don't
    scope: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    readings: List[Reading] = field(default_factory=list)
    language: str = "und"
    confidence: float = 0.0

    @property
    def ambiguous(self) -> bool:
        """Two readings are genuinely live. She asks; she does not pick one and hope."""
        return len(self.readings) > 1

    @property
    def is_question(self) -> bool:
        return self.kind == "question"

    @property
    def is_command(self) -> bool:
        return self.kind == "command"

    def clarifying_question(self) -> str:
        """The one thing worth asking before acting — empty when nothing needs asking."""
        if self.contradictions:
            return (f"You have asked me both to and not to {self.contradictions[0]}. "
                    f"Which did you mean?")
        if self.ambiguous:
            a, b = self.readings[0], self.readings[1]
            return (f"I can read that two ways: {a.why} ({a.confidence:.0%}), "
                    f"or {b.why} ({b.confidence:.0%}). Which one?")
        if self.open_questions:
            return self.open_questions[0]
        return ""

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "kind": self.kind, "goal": self.goal,
                "actions": list(self.actions),
                "ordering": [list(p) for p in self.ordering],
                "constraints": list(self.constraints), "conditions": list(self.conditions),
                "polarity": dict(self.polarity), "scope": list(self.scope),
                "open_questions": list(self.open_questions),
                "contradictions": list(self.contradictions),
                "readings": [r.to_dict() for r in self.readings],
                "ambiguous": self.ambiguous, "language": self.language,
                "confidence": round(self.confidence, 4)}


def _clauses(text: str) -> List[str]:
    try:
        parts = [c.strip(" ,.;:!?") for c in _CLAUSE_SPLIT.split(str(text or ""))]
        return [c for c in parts if c]
    except Exception:  # noqa: BLE001
        return [str(text or "").strip()] if str(text or "").strip() else []


def _words(text: str) -> List[str]:
    return [t.lower for t in tokenize(text, keep_punct=False) if t.kind == "word"]


class IntentReader:
    """Reads an instruction into structure, and says out loud what it could not resolve."""

    def __init__(self, *, lingua: Any = None, min_reading_gap: float = 0.15,
                 max_open_questions: int = 3) -> None:
        self.lingua = lingua if lingua is not None else Lingua()
        # Two readings both survive when neither dominates by more than this. Below the gap she
        # asks; above it she acts on the leader and records the other in ``readings[0].why``.
        self.min_reading_gap = max(0.0, min(1.0, float(min_reading_gap)))
        self.max_open_questions = max(0, int(max_open_questions))
        self.reads = 0
        self.ambiguous_reads = 0
        self.commands = 0
        self.questions = 0

    # ---- the one call ----------------------------------------------------- #
    def read(self, text: str) -> Intent:
        """Parse one turn. Never raises; an unparseable turn is an honest ``statement``."""
        out = Intent(text=str(text or "").strip())
        try:
            if not out.text:
                return out
            self.reads += 1
            words = _words(out.text)
            clauses = _clauses(out.text)
            try:
                out.language = self.lingua.read(out.text).primary
            except Exception:  # noqa: BLE001 — the tongue is a convenience here
                out.language = "und"

            out.readings = self._readings(out.text, words)
            if out.readings:
                out.kind = out.readings[0].kind
                out.goal = out.readings[0].goal
                out.actions = list(out.readings[0].actions)
                out.confidence = out.readings[0].confidence

            out.ordering = self._ordering(out.text, clauses)
            out.constraints = self._constraints(clauses)
            out.conditions = self._conditions(clauses)
            out.polarity = self._polarity(clauses, out.actions)
            out.scope = self._scope(out.text)
            out.contradictions = self._contradictions(out.polarity, clauses)
            out.open_questions = self._open_questions(out, words)

            if out.ambiguous:
                self.ambiguous_reads += 1
            if out.kind == "command":
                self.commands += 1
            elif out.kind == "question":
                self.questions += 1
            return out
        except Exception:  # noqa: BLE001 — reading an instruction never breaks a turn
            return out

    # ---- mood ------------------------------------------------------------- #
    def _readings(self, text: str, words: Sequence[str]) -> List[Reading]:
        """Score every mood the turn could be in, and keep the ones that stay live."""
        scores: Dict[str, Tuple[float, str]] = {}
        low = text.lower()
        first = words[0] if words else ""

        # question
        q = 0.0
        why_q = ""
        if low.rstrip().endswith("?"):
            q, why_q = 0.85, "it ends in a question mark"
        if first in _Q_WORDS:
            q, why_q = max(q, 0.8), why_q or "it opens with a question word"
        elif any(w in _Q_WORDS for w in words):
            q, why_q = max(q, 0.55), why_q or "it contains a question word"
        if first in _Q_AUX and not low.rstrip().endswith("?"):
            q, why_q = max(q, 0.6), why_q or "it opens with an auxiliary (subject–verb inversion)"
        if q:
            scores["question"] = (q, why_q)

        # command
        c = 0.0
        why_c = ""
        if first in _IMPERATIVE_VERBS:
            c, why_c = 0.8, "it opens with a bare verb (imperative)"
        if any(w in _IMPERATIVE_HI for w in words):
            c, why_c = max(c, 0.85), why_c or "it carries a Hindi imperative ending"
        if first in _POLITE_IMPERATIVE:
            c, why_c = max(c, 0.7), why_c or "it opens with a politeness marker before a request"
        if _POLITE_REQUEST.match(text) and any(
                w in _IMPERATIVE_VERBS or w in _IMPERATIVE_HI for w in words):
            c, why_c = max(c, 0.7), why_c or "'can you …' is a question in form and a request in use"
        if any(w in _POLITE_IMPERATIVE for w in words) and c:
            c = min(0.95, c + 0.05)
        if c:
            scores["command"] = (c, why_c)

        # greeting
        if first in _GREETINGS and len(words) <= 4:
            scores["greeting"] = (0.75, "it is a greeting and nothing else")

        if not scores:
            return [Reading(kind="statement", goal=self._goal(text), confidence=0.6,
                            why="no interrogative or imperative marker — read as a statement")]

        ranked = sorted(scores.items(), key=lambda kv: -kv[1][0])
        out: List[Reading] = []
        top = ranked[0][1][0]
        for kind, (score, why) in ranked:
            if score < top - self.min_reading_gap:
                break                       # decisively beaten — not a live reading
            actions = self._actions(text) if kind == "command" else []
            out.append(Reading(kind=kind, goal=self._goal(text), actions=actions,
                               confidence=score, why=why))
        return out

    # ---- content ---------------------------------------------------------- #
    @staticmethod
    def _goal(text: str) -> str:
        """A short statement of what is wanted — the content words, in order."""
        try:
            got = content_tokens(text, cap=8)
            return " ".join(got)
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _actions(text: str) -> List[str]:
        """The verbs she is being asked to perform, in the order they were asked."""
        out: List[str] = []
        try:
            for word in _words(text):
                if word in _IMPERATIVE_VERBS or word in _IMPERATIVE_HI:
                    if word not in out:
                        out.append(word)
            return out
        except Exception:  # noqa: BLE001
            return out

    def _ordering(self, text: str, clauses: Sequence[str]) -> List[Tuple[str, str]]:
        """``(before, after)`` pairs — what has to happen first.

        This is the half of ``"...lekin pehle test chala"`` that V.01 dropped entirely, and it
        is the half that decides whether an agent is safe to hand a tool to.
        """
        out: List[Tuple[str, str]] = []
        try:
            for group in (_ORDER_FIRST_THEN, _ORDER_BEFORE, _ORDER_AFTER):
                for pattern in group:
                    match = pattern.search(text)
                    if match is None:
                        continue
                    a, b = match.group("a").strip(" ,.;:"), match.group("b").strip(" ,.;:")
                    if a and b and (a, b) not in out:
                        out.append((a, b))
                    break                       # one pattern per family is enough
            if out:
                return out
            # "fix the code, but first run the tests" — the marker sits between two clauses and
            # reverses them, which no single regex above can see because the order is inverted.
            match = _BUT_FIRST.search(text)
            if match is not None:
                head = text[: match.start()].strip(" ,.;:")
                tail = text[match.end():].strip(" ,.;:")
                if head and tail:
                    out.append((tail, head))
            return out
        except Exception:  # noqa: BLE001
            return out

    @staticmethod
    def _constraints(clauses: Sequence[str]) -> List[str]:
        out: List[str] = []
        for clause in clauses:
            low = clause.lower()
            if any(marker in low for marker in _CONSTRAINT_MARKERS) and clause not in out:
                out.append(clause)
        return out

    @staticmethod
    def _conditions(clauses: Sequence[str]) -> List[str]:
        out: List[str] = []
        for clause in clauses:
            low = clause.lower() + " "
            if any(low.startswith(m) or f" {m}" in low for m in _CONDITION_MARKERS):
                if clause not in out:
                    out.append(clause)
        return out

    @staticmethod
    def _polarity(clauses: Sequence[str], actions: Sequence[str]) -> Dict[str, bool]:
        """Per action: is she being asked to do it, or explicitly not to?

        Negation scope runs from the negator to the end of *its own clause* — which is why the
        clause split matters. "run the tests but don't push" must not negate "run".
        """
        out: Dict[str, bool] = {a: True for a in actions}
        for clause in clauses:
            words = _words(clause)
            negated = False
            for word in words:
                if word in _NEGATORS:
                    negated = True
                    continue
                if word in _IMPERATIVE_VERBS or word in _IMPERATIVE_HI:
                    # A Hindi negator precedes its verb ("mat karo"); an English one does too
                    # ("don't push"). Either way, everything after it in this clause is negated.
                    out[word] = not negated
        return out

    @staticmethod
    def _scope(text: str) -> List[str]:
        try:
            seen: List[str] = []
            for match in _SCOPE_RE.finditer(str(text or "")):
                token = match.group(0).strip("`\"'")
                if token and token not in seen:
                    seen.append(token)
            return seen[:16]
        except Exception:  # noqa: BLE001
            return []

    @staticmethod
    def _contradictions(polarity: Dict[str, bool], clauses: Sequence[str]) -> List[str]:
        """Asked to do and not do the same thing — in different clauses of one turn."""
        out: List[str] = []
        try:
            per_clause: Dict[str, set] = {}
            for clause in clauses:
                words = _words(clause)
                negated = False
                for word in words:
                    if word in _NEGATORS:
                        negated = True
                        continue
                    if word in _IMPERATIVE_VERBS or word in _IMPERATIVE_HI:
                        per_clause.setdefault(word, set()).add(not negated)
            for action, polarities in per_clause.items():
                if len(polarities) > 1 and action not in out:
                    out.append(action)
            return out
        except Exception:  # noqa: BLE001
            return out

    def _open_questions(self, intent: Intent, words: Sequence[str]) -> List[str]:
        """What is genuinely under-specified. She asks about these instead of assuming.

        Deliberately conservative: a false "I need to ask" costs a round trip, and a false
        "I understood" costs whatever the tool then did.
        """
        out: List[str] = []
        try:
            if intent.kind == "command" and not intent.actions:
                out.append("What would you like me to do? I can see it is an instruction, "
                           "but I cannot tell which action you want.")
            dangling = [w for w in words if w in _DANGLING
                        and not (intent.ordering and w in _ORDER_ANAPHORS)]
            if dangling and intent.kind == "command" and not intent.scope:
                out.append(f"What does '{dangling[0]}' refer to? Nothing in this turn names it.")
            vague = [w for w in words if w in _VAGUE]
            if vague and intent.kind == "command":
                out.append(f"'{vague[0]}' is not specific enough for me to act on — "
                           f"how much, or which ones?")
            return out[: self.max_open_questions]
        except Exception:  # noqa: BLE001
            return out

    # ---- introspection ---------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        try:
            return {
                "reads": self.reads, "commands": self.commands, "questions": self.questions,
                "ambiguous": self.ambiguous_reads,
                "markers": {"question_words": len(_Q_WORDS),
                            "imperative_verbs": len(_IMPERATIVE_VERBS),
                            "imperative_hi": len(_IMPERATIVE_HI),
                            "negators": len(_NEGATORS)},
                "note": ("a marker-driven parser with no part-of-speech model aboard: an "
                         "imperative in a shape nobody listed reads as a statement, which "
                         "under-claims rather than inventing a command"),
            }
        except Exception:  # noqa: BLE001
            return {}

    def describe(self, intent: Intent) -> str:
        """What ``/nyx intent`` prints."""
        try:
            lines = [f"kind      : {intent.kind}  ({intent.confidence:.0%})",
                     f"goal      : {intent.goal or '—'}"]
            if intent.actions:
                marks = ", ".join(f"{a}{'' if intent.polarity.get(a, True) else ' (NOT)'}"
                                  for a in intent.actions)
                lines.append(f"actions   : {marks}")
            for before, after in intent.ordering:
                lines.append(f"ordering  : «{before}» before «{after}»")
            for clause in intent.constraints:
                lines.append(f"constraint: {clause}")
            for clause in intent.conditions:
                lines.append(f"condition : {clause}")
            if intent.scope:
                lines.append(f"scope     : {', '.join(intent.scope)}")
            for question in intent.open_questions:
                lines.append(f"unclear   : {question}")
            for action in intent.contradictions:
                lines.append(f"CONFLICT  : asked both to and not to {action}")
            if intent.ambiguous:
                for reading in intent.readings:
                    lines.append(f"reading   : {reading.kind} ({reading.confidence:.0%}) "
                                 f"— {reading.why}")
            return "\n".join(lines)
        except Exception:  # noqa: BLE001
            return ""


#: A module-level reader for the cheap call sites that only want the mood.
_DEFAULT: Optional[IntentReader] = None


def _reader() -> IntentReader:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = IntentReader()
    return _DEFAULT


def is_question(text: str) -> bool:
    """Was this asked, rather than asserted? The replacement for V.01's opener regex.

    Same signature and same fail-soft contract as :func:`nyxara.nyx.brain.is_question`, but
    decided from mood across three registers instead of from a leading wh-word.
    """
    try:
        return _reader().read(text).is_question
    except Exception:  # noqa: BLE001
        return False


def is_command(text: str) -> bool:
    """Is she being told to *do* something? V.01 had no way to ask this at all."""
    try:
        return _reader().read(text).is_command
    except Exception:  # noqa: BLE001
        return False
