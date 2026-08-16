"""NYXARA · njp/relevance.py — truth is not relevance, and reasoning is not always the answer.

The failure this module exists to end, in the Master's own transcript::

    Master:  How are you NYXARA?
    NYXARA:  From my own experiments I discovered the law
             period_squared = 39.67·(L / g) + 0.0003079.
    trace:   [perceive] classified the question as 'chat'
             [recall]   recalled 5 relevant memories
             [reason]   recalled a governing law
             [gate]     confidence 0.80 → 1.00
             [decide]   concluded via law            verified=True

Every stage did its job and the answer was still absurd, which means the defect is not in any
stage — it is in what connects them. Three separate confusions, each fixed by one of the parts
below.

**1. Truth was standing in for relevance.** The pendulum law is *true*. She verified it herself.
Nothing in the pipeline ever asked the only question that mattered: is this true thing anything
to do with what was just asked? :class:`RelevanceGate` asks it, and it scores a memory against
the live context on six dimensions — semantic, conversational, temporal, causal, goal, minus
contradiction and domain distance. A memory below threshold **never reaches the reasoner at
all**; it is not down-weighted, it is not seen. A true fact that answers nothing is noise with a
certificate.

**2. Reasoning ran because reasoning was available.** Perception correctly called the turn
*chat*, and the recall→reason machinery ran anyway, because it always runs. :class:`CognitivePolicy`
holds the rule the Master named — *never reason merely because reasoning is available* — as an
actual table: each speech act permits a specific set of cognitive pathways and forbids the rest.
A greeting may consult her own state; it may not consult physics. Intelligence is not running
maximum cognition on every input, it is selecting the right process for the input in front of you.

**3. Confidence rose because a conclusion was reached.** ``0.80 → 1.00`` on the strength of
having picked an answer, and ``verified=True`` meaning nothing more than "the reasoner finished".
:func:`revise_confidence` makes the ceiling monotonic — confidence may only exceed what it
started at when *independent* evidence, real relevance and consistency all say so, and being more
sure because you thought harder is exactly the move it refuses. :func:`is_verified` insists that
``verified`` means independent verification succeeded, never that a pipeline ran to completion.

**And one smaller thing, which the Master also caught.** "I'm certain that I understand: Hii
NYXARA is a sovereign cognitive architecture" is not an answer to a greeting; it is a machine
narrating its own comprehension and then over-reading a hello as an ontological claim.
:func:`is_meta_commentary` recognises that shape — a restatement of the input wrapped in a claim
to have understood it — so it can be refused before it is ever spoken. Understanding belongs in
the internal state, not in the reply.

Pure standard library, no LLM. Every score here is arithmetic over text and over organs this
brain already has.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "SpeechAct",
    "Act",
    "SpeechActReader",
    "Pathway",
    "CognitivePolicy",
    "RelevanceScore",
    "RelevanceGate",
    "revise_confidence",
    "is_verified",
    "is_meta_commentary",
]


def _words(text: Any) -> List[str]:
    return [w for w in re.split(r"[^0-9a-zऀ-ॿ]+", str(text or "").lower()) if w]


# Function words carry no topic. Leaving them in the *query* is not harmless: relevance is
# measured as a share of the query, so "gravity kya hai" spends two of its three slots on "kya"
# and "hai", and a memory that answers it perfectly scores 0.33 instead of 1.0 and is rejected.
# A gate that rejects the right memory is worse than no gate, because it is confidently wrong.
_STOP = frozenset({
    "a", "an", "the", "is", "are", "am", "was", "were", "be", "been", "of", "to", "in", "on",
    "at", "for", "with", "and", "or", "but", "it", "this", "that", "these", "those", "do",
    "does", "did", "so", "as", "by", "from", "me", "my", "i", "we", "us",
    "hai", "hain", "ho", "hu", "hun", "hoon", "kya", "ka", "ki", "ke", "ko", "se", "me", "mein",
    "aur", "ya", "bhi", "to", "toh", "na", "ne", "par", "pe", "wala", "wali", "raha", "rahi",
    "rha", "rhi", "kar", "karo", "hota", "hoti", "tha", "thi", "ek",
    "है", "हैं", "का", "की", "के", "को", "से", "में", "और", "या", "भी", "एक",
})


def _content(text: Any) -> Set[str]:
    """The topic-bearing words of a turn — what relevance is actually about."""
    out = {w for w in _words(text) if w not in _STOP and len(w) > 1}
    # A turn made entirely of function words still has to be comparable to something, so fall
    # back to the raw set rather than returning nothing and scoring every memory at zero.
    return out or set(_words(text))


# --------------------------------------------------------------------------- #
# What kind of thing was said
# --------------------------------------------------------------------------- #
class SpeechAct:
    """What the Master was *doing* with a turn, which is prior to what it is about."""

    GREETING = "greeting"
    FAREWELL = "farewell"
    THANKS = "thanks"
    SOCIAL_CHECKIN = "social_checkin"       # "how are you"
    STATE_QUERY = "state_query"             # "kya kar rahi ho"
    CAPABILITY_QUERY = "capability_query"   # "can you learn this"
    KNOWLEDGE_QUERY = "knowledge_query"     # "gravity kya hai"
    CAUSAL_QUERY = "causal_query"           # "why did X happen"
    TASK = "task"                           # "ye code fix karo"
    CORRECTION = "correction"               # "nahi, aisa nahi"
    STATEMENT = "statement"                 # a fact stated to her
    UNKNOWN = "unknown"

    #: Acts that are about the two of them rather than about the world. Nothing factual is owed
    #: in reply, and reaching for a fact is the bug.
    SOCIAL = frozenset({GREETING, FAREWELL, THANKS, SOCIAL_CHECKIN})
    #: Acts that are about *her*, answerable only from her own state.
    REFLEXIVE = frozenset({STATE_QUERY, CAPABILITY_QUERY, SOCIAL_CHECKIN})
    ALL = (GREETING, FAREWELL, THANKS, SOCIAL_CHECKIN, STATE_QUERY, CAPABILITY_QUERY,
           KNOWLEDGE_QUERY, CAUSAL_QUERY, TASK, CORRECTION, STATEMENT, UNKNOWN)


@dataclass
class Act:
    """The classification, how sure, and what nearly won instead."""

    kind: str = SpeechAct.UNKNOWN
    confidence: float = 0.0
    margin: float = 0.0
    cues: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)

    @property
    def social(self) -> bool:
        return self.kind in SpeechAct.SOCIAL

    @property
    def reflexive(self) -> bool:
        return self.kind in SpeechAct.REFLEXIVE

    @property
    def ambiguous(self) -> bool:
        """A narrow win. Worth asking rather than guessing — the Master's 'calibrated' case."""
        return self.margin < 0.15

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "confidence": round(self.confidence, 4),
                "margin": round(self.margin, 4), "social": self.social,
                "reflexive": self.reflexive, "ambiguous": self.ambiguous,
                "cues": self.cues[:6],
                "scores": {k: round(v, 3) for k, v in self.scores.items() if v > 0}}


# Surface cues. Deliberately multilingual: the Master writes Hinglish, and an English-only
# greeting detector reads "kya kar rahi ho" as an unknown act and hands it to the reasoner —
# which is the second failure in the transcript.
_GREET = frozenset({"hi", "hii", "hiii", "hello", "helo", "hey", "yo", "namaste", "namaskar",
                    "salaam", "salam", "hola", "नमस्ते", "हाय"})
_BYE = frozenset({"bye", "goodbye", "alvida", "chalta", "chalti", "tata", "अलविदा"})
_THANKS = frozenset({"thanks", "thank", "thx", "shukriya", "dhanyavaad", "dhanyawad",
                     "धन्यवाद", "शुक्रिया"})
_SELF_REF = frozenset({"you", "your", "yourself", "tum", "tu", "tera", "teri", "tumhara",
                       "tumhari", "aap", "aapka", "aapki", "nyxara", "njp", "तुम", "आप"})
_HOWARE = re.compile(
    r"\bhow\s+(?:are|r)\s+(?:you|u)\b|\bkaisi?\s+ho\b|\bkaise\s+ho\b|\bkya\s+haal\b"
    r"|\bsab\s+theek\b|\bkaisa\s+hai\b|कैसी\s+हो|कैसे\s+हो|क्या\s+हाल",
    re.IGNORECASE)
_DOING = re.compile(
    r"\bwhat\s+are\s+(?:you|u)\s+doing\b|\bkya\s+kar\s+rah[ieu]\b|\bkya\s+kar\s+rhi\b"
    r"|\bkya\s+chal\s+raha\b|क्या\s+कर\s+रह",
    re.IGNORECASE)
_CAN_YOU = re.compile(
    r"\b(?:can|could)\s+(?:you|u)\b|\bare\s+you\s+able\b|\bkya\s+tum\s+kar\s+sakti\b"
    r"|\bkar\s+sakti\s+ho\b|\bkar\s+paogi\b|\bcapable\b",
    re.IGNORECASE)
_WHY = re.compile(r"\bwhy\b|\bkyun\b|\bkyon\b|\bkyu\b|क्यों|\bwajah\b|\breason\s+for\b",
                  re.IGNORECASE)
# An intervention, which is a different question from a correlation and needs a different organ.
# "why do plants grow" asks for a mechanism she may already hold; "what if I halve the water"
# asks her to *change* something and report what follows, which only the do-operator answers.
# Scored above `_WHY` because the construction is the more specific of the two: every phrase here
# names an intervention outright, where "why" merely asks after a cause.
_WHATIF = re.compile(
    r"\bwhat\s+if\b|\bwhat\s+happens\s+if\b|\bwhat\s+would\s+happen\b|\bsuppose\b"
    r"|\bif\s+(?:i|we|you)\s+(?:halve|double|remove|increase|reduce|cut|raise|lower|stop)\b"
    r"|\bagar\b|\bmaan\s+lo\b|अगर|\bkya\s+ho\s+(?:agar|to)\b",
    re.IGNORECASE)
_WHAT_IS = re.compile(r"\bwhat\s+is\b|\bwhat's\b|\bkya\s+ha[ie]\b|\bkya\s+hota\b|क्या\s+है"
                      r"|\bdefine\b|\bmatlab\b|\bmeaning\b", re.IGNORECASE)
_TASK = re.compile(
    r"\b(?:fix|write|build|make|run|create|delete|update|deploy|install|refactor|test)\b"
    r"|\b(?:karo|kar\s+do|banao|likho|chalao|theek\s+karo|fix\s+kar)\b"
    r"|करो|बनाओ|लिखो", re.IGNORECASE)
_CORRECTION = re.compile(r"\bno\b|\bnahi\b|\bnahin\b|\bwrong\b|\bgalat\b|नहीं|\bactually\b"
                         r"|\bcorrection\b", re.IGNORECASE)


class SpeechActReader:
    """Scores a turn on independent cues and reports the winner with its margin.

    A scoring rule, not a keyword chain: several cues can fire at once ("how are you doing" is
    both a check-in and a state query), and the *margin* between the top two is what tells the
    brain whether to answer or to ask which was meant.
    """

    def read(self, text: str, *, intent: Any = None) -> Act:
        raw = str(text or "").strip()
        words = _words(raw)
        wordset = set(words)
        scores: Dict[str, float] = {k: 0.0 for k in SpeechAct.ALL}
        cues: List[str] = []

        if not raw:
            return Act(kind=SpeechAct.UNKNOWN)

        # Greetings and their kin are *short* and lead with the cue. "hi" inside a sentence is
        # usually not a greeting, and treating it as one is how "this is a high-level fix" reads
        # as hello.
        head = words[0] if words else ""
        if head in _GREET:
            scores[SpeechAct.GREETING] += 0.8 if len(words) <= 4 else 0.35
            cues.append(f"greeting-head:{head}")
        elif wordset & _GREET and len(words) <= 3:
            scores[SpeechAct.GREETING] += 0.5
            cues.append("greeting-word")
        if head in _BYE or (wordset & _BYE and len(words) <= 3):
            scores[SpeechAct.FAREWELL] += 0.8
            cues.append("farewell")
        if wordset & _THANKS:
            scores[SpeechAct.THANKS] += 0.7
            cues.append("thanks")

        if _HOWARE.search(raw):
            scores[SpeechAct.SOCIAL_CHECKIN] += 0.9
            cues.append("how-are-you")
        if _DOING.search(raw):
            scores[SpeechAct.STATE_QUERY] += 0.9
            cues.append("what-are-you-doing")
        if _CAN_YOU.search(raw):
            scores[SpeechAct.CAPABILITY_QUERY] += 0.75
            cues.append("can-you")
        if _WHY.search(raw):
            scores[SpeechAct.CAUSAL_QUERY] += 0.8
            cues.append("why")
        if _WHATIF.search(raw):
            scores[SpeechAct.CAUSAL_QUERY] += 0.85
            cues.append("what-if")
        if _WHAT_IS.search(raw):
            scores[SpeechAct.KNOWLEDGE_QUERY] += 0.6
            cues.append("what-is")
        if _TASK.search(raw):
            scores[SpeechAct.TASK] += 0.7
            cues.append("imperative")
        if _CORRECTION.search(raw) and len(words) <= 12:
            scores[SpeechAct.CORRECTION] += 0.4
            cues.append("negation")

        # Being about *her* turns a general question into a reflexive one. "what is gravity" and
        # "what are you" are the same syntax and different acts.
        if wordset & _SELF_REF:
            for reflexive in (SpeechAct.SOCIAL_CHECKIN, SpeechAct.STATE_QUERY,
                              SpeechAct.CAPABILITY_QUERY):
                if scores[reflexive] > 0:
                    scores[reflexive] += 0.25
            cues.append("about-her")
            # A bare "NYXARA" with nothing else asked is address, not enquiry.
            if len(words) <= 2 and not any(scores[k] > 0 for k in
                                           (SpeechAct.SOCIAL_CHECKIN, SpeechAct.STATE_QUERY)):
                scores[SpeechAct.GREETING] += 0.3

        # The intent organ already decided statement/question/command. Used as a prior, never as
        # the verdict — it does not know a greeting from a fact.
        kind = str(getattr(intent, "kind", "") or "")
        if kind == "question":
            scores[SpeechAct.KNOWLEDGE_QUERY] += 0.25
        elif kind == "command":
            scores[SpeechAct.TASK] += 0.3
        elif kind == "statement":
            scores[SpeechAct.STATEMENT] += 0.3

        # The residual reading. A turn that is not doing anything else is telling her something.
        scores[SpeechAct.STATEMENT] += 0.2
        if "?" in raw and scores[SpeechAct.KNOWLEDGE_QUERY] <= 0:
            scores[SpeechAct.KNOWLEDGE_QUERY] += 0.3

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best, best_score = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        return Act(kind=best if best_score > 0 else SpeechAct.UNKNOWN,
                   confidence=min(1.0, best_score), margin=best_score - runner,
                   cues=cues, scores=scores)


# --------------------------------------------------------------------------- #
# Which cognition is even allowed
# --------------------------------------------------------------------------- #
class Pathway:
    """The kinds of cognition available. A turn gets some of them, never all of them."""

    SELF_STATE = "self_state"     # her own interoception, goals, capability model
    SOCIAL = "social"             # the relationship, the Master's standing preferences
    GROUNDING = "grounding"       # the fact store
    RECALL = "recall"             # episodic memory
    REASON = "reason"             # the hypothesis ladder
    SIMULATE = "simulate"         # the internal universe
    EXPERIMENT = "experiment"     # design an experiment
    ACT = "act"                   # tools


class CognitivePolicy:
    """*Never reason merely because reasoning is available.*

    The table is the point. Before this, every turn went ``perceive → recall → reason`` because
    that is what the code did next, and a greeting therefore got the full apparatus pointed at it.
    Here a greeting is permitted the relationship and her own state, and nothing else — so there
    is no path by which a pendulum law can reach the reply, however true and however well
    recalled it is.
    """

    PATHWAYS: Dict[str, Tuple[str, ...]] = {
        SpeechAct.GREETING: (Pathway.SOCIAL, Pathway.SELF_STATE),
        SpeechAct.FAREWELL: (Pathway.SOCIAL, Pathway.SELF_STATE),
        SpeechAct.THANKS: (Pathway.SOCIAL,),
        SpeechAct.SOCIAL_CHECKIN: (Pathway.SELF_STATE, Pathway.SOCIAL),
        SpeechAct.STATE_QUERY: (Pathway.SELF_STATE,),
        SpeechAct.CAPABILITY_QUERY: (Pathway.SELF_STATE, Pathway.GROUNDING),
        SpeechAct.KNOWLEDGE_QUERY: (Pathway.GROUNDING, Pathway.RECALL, Pathway.REASON,
                                    Pathway.EXPERIMENT),
        SpeechAct.CAUSAL_QUERY: (Pathway.GROUNDING, Pathway.SIMULATE, Pathway.REASON,
                                 Pathway.RECALL),
        SpeechAct.TASK: (Pathway.ACT, Pathway.SELF_STATE, Pathway.REASON),
        SpeechAct.CORRECTION: (Pathway.SOCIAL, Pathway.GROUNDING),
        SpeechAct.STATEMENT: (Pathway.GROUNDING, Pathway.RECALL),
        SpeechAct.UNKNOWN: (Pathway.GROUNDING, Pathway.RECALL),
    }

    def allowed(self, act: Any) -> Tuple[str, ...]:
        kind = getattr(act, "kind", act)
        return self.PATHWAYS.get(str(kind), self.PATHWAYS[SpeechAct.UNKNOWN])

    def permits(self, act: Any, pathway: str) -> bool:
        return pathway in self.allowed(act)

    def forbids_world_knowledge(self, act: Any) -> bool:
        """True when nothing factual may reach the reply. The direct fix for the transcript."""
        allowed = self.allowed(act)
        return not ({Pathway.GROUNDING, Pathway.RECALL, Pathway.REASON,
                     Pathway.SIMULATE} & set(allowed))


# --------------------------------------------------------------------------- #
# Truth ≠ relevance
# --------------------------------------------------------------------------- #
@dataclass
class RelevanceScore:
    """Why a memory was admitted to the reply, or why it was not."""

    memory: str = ""
    semantic: float = 0.0
    conversational: float = 0.0
    temporal: float = 0.0
    causal: float = 0.0
    goal: float = 0.0
    contradiction: float = 0.0     # subtracted
    distance: float = 0.0          # subtracted
    threshold: float = 0.22

    @property
    def total(self) -> float:
        """The weighted sum, clamped to ``[0, 1]``.

        Semantic relevance carries the most weight because it is the one dimension that cannot be
        faked by a memory simply being recent or well-established — the two properties that let
        the pendulum law win.
        """
        positive = (0.45 * self.semantic + 0.2 * self.conversational
                    + 0.1 * self.temporal + 0.15 * self.causal + 0.1 * self.goal)
        return max(0.0, min(1.0, positive - self.contradiction - self.distance))

    @property
    def admitted(self) -> bool:
        return self.total >= self.threshold

    @property
    def why(self) -> str:
        if self.admitted:
            return "admitted"
        if self.semantic <= 0.02:
            return "nothing in common with what was asked"
        if self.distance > 0.2:
            return "different domain"
        if self.contradiction > 0:
            return "contradicts something held"
        return "below relevance threshold"

    def to_dict(self) -> Dict[str, Any]:
        return {"memory": self.memory[:120], "total": round(self.total, 4),
                "admitted": self.admitted, "why": self.why,
                "semantic": round(self.semantic, 3),
                "conversational": round(self.conversational, 3),
                "temporal": round(self.temporal, 3), "causal": round(self.causal, 3),
                "goal": round(self.goal, 3), "contradiction": round(self.contradiction, 3),
                "distance": round(self.distance, 3)}


class RelevanceGate:
    """``R(memory | context)`` — and below threshold the reasoner never sees it.

    Rejection rather than down-weighting is deliberate and is the whole difference. A
    down-weighted irrelevant memory is still in the candidate set, and a candidate set is a thing
    an optimiser will eventually pick from — especially one that scores candidates by how
    *verified* they are, which is exactly how a true pendulum law wins a conversation about
    someone's day.
    """

    #: Rough domain buckets, for the distance penalty. Not a taxonomy — a check that a physics
    #: memory is not being offered in a social turn.
    DOMAINS: Dict[str, frozenset] = {
        "physics": frozenset({"force", "gravity", "energy", "mass", "velocity", "period",
                              "pendulum", "boils", "melts", "temperature", "degree", "law",
                              "equation", "squared", "acceleration"}),
        "biology": frozenset({"plant", "paudha", "paudhe", "water", "pani", "grow", "badhta",
                              "cell", "organism", "animal", "food"}),
        "code": frozenset({"python", "code", "function", "test", "bug", "commit", "branch",
                           "file", "error", "import"}),
        "social": frozenset({"you", "tum", "aap", "master", "hello", "hi", "feel", "how",
                             "kaisi", "kaise", "haal", "theek", "good", "fine"}),
        "self": frozenset({"nyxara", "njp", "brain", "dimag", "learn", "seekh", "memory",
                           "concept", "goal"}),
    }

    def __init__(self, *, threshold: float = 0.22, half_life_s: float = 86400.0) -> None:
        self.threshold = float(threshold)
        self.half_life_s = max(1.0, float(half_life_s))
        self.scored = 0
        self.admitted = 0
        self.rejected = 0
        self.last: List[RelevanceScore] = []

    # ---- the dimensions ---------------------------------------------------- #
    @staticmethod
    def _semantic(query: Set[str], memory: Set[str]) -> float:
        """Overlap, normalised by the *query* rather than the union.

        Normalising by the union would let a long memory dilute its own irrelevance: a
        two-hundred-word note sharing one word with the question scores near zero either way,
        but a short memory sharing one of the question's three words is genuinely on topic and
        the union measure hides that.
        """
        if not query or not memory:
            return 0.0
        shared = query & memory
        if not shared:
            return 0.0
        return len(shared) / len(query)

    def _domain(self, words: Set[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, vocabulary in self.DOMAINS.items():
            hits = len(words & vocabulary)
            if hits:
                out[name] = hits / max(1, len(words))
        return out

    def _distance(self, query: Set[str], memory: Set[str]) -> float:
        """Penalty for answering from a different domain than the one asked about."""
        q, m = self._domain(query), self._domain(memory)
        if not q or not m:
            return 0.0
        q_top = max(q, key=q.get)
        m_top = max(m, key=m.get)
        if q_top == m_top:
            return 0.0
        # A social or self-directed question answered from a world-knowledge domain is the exact
        # transcript failure, and is penalised hardest.
        if q_top in ("social", "self") and m_top in ("physics", "biology", "code"):
            return 0.6
        return 0.25

    def _temporal(self, at: Optional[float], now: float) -> float:
        if not at:
            return 0.5           # unknown age is neither fresh nor stale
        age = max(0.0, now - float(at))
        return 0.5 ** (age / self.half_life_s)

    @staticmethod
    def _causal(query: Set[str], memory: Set[str], world: Any) -> float:
        """Is anything in the memory causally linked to anything in the question?

        This is the dimension that lets a *seemingly* unrelated memory through when it genuinely
        belongs — "why is the kettle loud" reaching a memory about boiling — so the gate is not
        merely a keyword filter with extra steps.
        """
        if world is None or not query or not memory:
            return 0.0
        try:
            for link in world.links(causal_only=True)[:64]:
                cause = _content(link.cause)
                effect = _content(link.effect)
                if (cause & query and effect & memory) or (cause & memory and effect & query):
                    return min(1.0, float(getattr(link, "strength", 0.5)))
        except Exception:  # noqa: BLE001
            return 0.0
        return 0.0

    @staticmethod
    def _goal(memory: Set[str], goals: Sequence[str]) -> float:
        if not goals or not memory:
            return 0.0
        best = 0.0
        for goal in list(goals)[:8]:
            gw = _content(goal)
            if gw:
                best = max(best, len(gw & memory) / len(gw))
        return best

    # ---- the gate ----------------------------------------------------------- #
    def score(self, memory: Any, *, query: str, act: Any = None, world: Any = None,
              goals: Sequence[str] = (), thread: Sequence[str] = (),
              at: Optional[float] = None, contradicts: bool = False) -> RelevanceScore:
        """Score one candidate memory against the live context."""
        text = str(getattr(memory, "text", memory) or "")
        query_words = _content(query)
        memory_words = _content(text)
        now = time.time()

        out = RelevanceScore(memory=text, threshold=self.threshold)
        out.semantic = self._semantic(query_words, memory_words)
        thread_words: Set[str] = set()
        for turn in list(thread)[-4:]:
            thread_words |= _content(turn)
        out.conversational = self._semantic(thread_words, memory_words) if thread_words else 0.0
        out.temporal = self._temporal(at if at is not None else getattr(memory, "at", None), now)
        out.causal = self._causal(query_words, memory_words, world)
        out.goal = self._goal(memory_words, goals)
        out.distance = self._distance(query_words, memory_words)
        out.contradiction = 0.4 if contradicts else 0.0

        # A social or reflexive turn owes nothing factual. Rather than hoping the arithmetic
        # rejects the pendulum law, the policy says outright that no world memory belongs here —
        # which is a rule about what kind of turn it is, not a guess about this memory.
        if act is not None and CognitivePolicy().forbids_world_knowledge(act):
            if out.semantic < 0.5:
                out.distance = max(out.distance, 0.8)

        self.scored += 1
        if out.admitted:
            self.admitted += 1
        else:
            self.rejected += 1
        return out

    def filter(self, memories: Iterable[Any], *, query: str, act: Any = None,
               world: Any = None, goals: Sequence[str] = (),
               thread: Sequence[str] = ()) -> List[RelevanceScore]:
        """Score every candidate; return only those admitted, best first."""
        scored = [self.score(m, query=query, act=act, world=world, goals=goals, thread=thread)
                  for m in (memories or [])]
        self.last = sorted(scored, key=lambda s: s.total, reverse=True)[:16]
        return [s for s in self.last if s.admitted]

    def stats(self) -> Dict[str, Any]:
        return {"scored": self.scored, "admitted": self.admitted, "rejected": self.rejected,
                "admit_rate": (round(self.admitted / self.scored, 4) if self.scored else None),
                "threshold": self.threshold,
                "last": [s.to_dict() for s in self.last[:4]]}


# --------------------------------------------------------------------------- #
# Confidence may not rise because you thought about it
# --------------------------------------------------------------------------- #
def revise_confidence(prior: float, *, independent_evidence: int = 0,
                      relevance: float = 0.0, consistent: bool = True,
                      reasoning_depth: int = 0) -> float:
    """The monotonic rule: ``final ≤ prior`` unless evidence genuinely earns more.

    ``reasoning_depth`` is accepted and deliberately **ignored** as a source of confidence. That
    is the whole point of the signature: the transcript's ``0.80 → 1.00`` happened because a
    conclusion had been reached, and "I thought about it harder" is not evidence about the world.
    Depth may lower confidence — a long descent means the answer was not obvious — and may never
    raise it.

    A rise requires all three of: at least one *independent* piece of evidence, real relevance,
    and consistency with what she already holds. Each independent source can add at most a
    diminishing slice, and the ceiling stays below 1.0 because certainty is not something this
    function is entitled to hand out.
    """
    prior = max(0.0, min(1.0, float(prior)))
    if not consistent:
        return max(0.0, prior * 0.5)
    if relevance <= 0.0:
        # An answer that is not relevant cannot be more credible for having been produced.
        return min(prior, 0.3)
    if independent_evidence <= 0:
        out = prior
    else:
        room = 0.97 - prior
        gained = room * (1.0 - 0.6 ** independent_evidence) * min(1.0, relevance)
        out = prior + gained
    if reasoning_depth > 2:
        out *= 0.95 ** (reasoning_depth - 2)
    return max(0.0, min(0.97, out))


def is_verified(*, independent_sources: int = 0, reproduced: bool = False,
                pipeline_completed: bool = False) -> bool:
    """``verified`` means independent verification succeeded. Nothing else counts.

    ``pipeline_completed`` is taken and ignored for the same reason ``reasoning_depth`` is above:
    in the transcript ``verified=True`` meant only that the reasoner had picked something. A flag
    that is set by the act of finishing tells the Master nothing about whether the answer is true.
    """
    return bool(independent_sources >= 2 or (independent_sources >= 1 and reproduced))


# --------------------------------------------------------------------------- #
# "I understand: <your words back>"
# --------------------------------------------------------------------------- #
_UNDERSTAND = re.compile(
    r"^\s*(?:i'?m\s+certain\s+that\s+)?i\s+(?:think\s+i\s+)?understand\b"
    r"|^\s*(?:i\s+)?(?:see|got\s+it)\b\s*[:,]"
    r"|^\s*samajh\s+(?:gayi|gaya)\b",
    re.IGNORECASE)


def is_meta_commentary(reply: str, stimulus: str = "", *, echo_threshold: float = 0.6) -> bool:
    """Is this reply narrating her own comprehension instead of answering?

    Two shapes, and the transcript has both. The explicit one opens with a claim to have
    understood. The quieter one simply hands the Master's own words back — "I think I understand:
    Kya kar rahi ho" is a paraphrase presented as a reply, and the bridge between understanding
    and responding is exactly what is missing there.

    Understanding belongs in the internal state (``intent = greeting, confidence = 0.99``), never
    in what is said. Where the act is genuinely ambiguous the honest reply is a question about
    which reading was meant, not an assertion of having understood.
    """
    text = str(reply or "").strip()
    if not text:
        return False
    if _UNDERSTAND.search(text):
        return True
    stim_words = set(_words(stimulus))
    reply_words = set(_words(text))
    if not stim_words or not reply_words:
        return False
    # A reply made almost entirely of the question's own words is an echo, whatever it is called.
    echoed = len(stim_words & reply_words) / len(reply_words)
    return echoed >= echo_threshold and len(reply_words) <= len(stim_words) + 3
