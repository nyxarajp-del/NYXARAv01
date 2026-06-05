"""NYXARA · social/dialogue.py — multi-party tracking, turn-taking, implicature, subtext.

What people *say* and what they *mean* are rarely identical. This module gives NYXARA
the pragmatic competence to follow a conversation properly:

* **Multi-party tracking** — who is present, who said what, and *to whom* (addressee
  resolution), with per-speaker histories.
* **Turn-taking** — whose turn it is, who holds the floor, and detection of
  **interruptions** (someone speaking when another was expected to answer).
* **Implicature** (Grice) — the *implied* meaning when a maxim is flouted: scalar
  ("*some* passed" ⇒ not all), relevance ("it's cold in here" ⇒ please close the
  window), and quality/irony ("oh, great" ⇒ the opposite).
* **Subtext** — indirect requests, hedging, politeness level, and sarcasm cues.

This is the layer that lets NYXARA grasp a hint from the Master, catch a veiled
threat (Rule 3), and read the room. Rule-based and deterministic; an LLM can refine
it later. Pure standard library.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "SpeechAct",
    "Utterance",
    "Implicature",
    "Subtext",
    "Conversation",
    "ImplicatureEngine",
    "SubtextDetector",
    "DialogueManager",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Speech acts & utterances
# --------------------------------------------------------------------------- #
class SpeechAct(str, Enum):
    ASSERTION = "assertion"
    QUESTION = "question"
    REQUEST = "request"
    COMMAND = "command"
    GREETING = "greeting"
    ACKNOWLEDGMENT = "acknowledgment"
    EXPRESSIVE = "expressive"


_IMPERATIVE_VERBS = {
    "close", "open", "stop", "go", "give", "show", "tell", "make", "do", "send",
    "bring", "find", "run", "check", "fix", "delete", "add", "remove", "look",
    "wait", "listen", "help", "defend", "protect", "block", "report", "explain",
}
_GREETINGS = {"hi", "hello", "hey", "namaste", "greetings", "yo", "good morning",
              "good evening"}
_THANKS = {"thanks", "thank you", "thankyou", "shukriya", "appreciated"}


def detect_speech_act(text: str) -> SpeechAct:
    t = text.strip().lower()
    if not t:
        return SpeechAct.ASSERTION
    first = t.split()[0].strip(",.!?")
    if t.endswith("?") or first in ("who", "what", "when", "where", "why", "how",
                                    "is", "are", "do", "does", "can", "could", "would",
                                    "will", "should"):
        # polite "could you ...?" reads as a request, not a bare question
        if re.match(r"(could|would|can|will)\s+you\b", t):
            return SpeechAct.REQUEST
        return SpeechAct.QUESTION
    if "please" in t or re.match(r"(could|would|can|will)\s+you\b", t):
        return SpeechAct.REQUEST
    if first in _GREETINGS or t in _GREETINGS:
        return SpeechAct.GREETING
    if any(t.startswith(g) for g in _THANKS):
        return SpeechAct.ACKNOWLEDGMENT
    if first in _IMPERATIVE_VERBS:
        return SpeechAct.COMMAND
    return SpeechAct.ASSERTION


@dataclass
class Utterance:
    speaker: str
    text: str
    addressee: Optional[str] = None
    speech_act: SpeechAct = SpeechAct.ASSERTION
    turn: int = 0
    at: float = field(default_factory=time.time)
    is_interruption: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"turn": self.turn, "speaker": self.speaker, "text": self.text,
                "addressee": self.addressee, "speech_act": self.speech_act.value,
                "interruption": self.is_interruption}


# --------------------------------------------------------------------------- #
# Conversation (multi-party + turn-taking)
# --------------------------------------------------------------------------- #
class Conversation:
    """Tracks participants, utterances, addressees, and turn-taking."""

    def __init__(self, participants: Sequence[str] = ()) -> None:
        self.participants: List[str] = list(participants)
        self.turns: List[Utterance] = []

    def add_participant(self, name: str) -> None:
        if name not in self.participants:
            self.participants.append(name)

    def _resolve_addressee(self, text: str, speaker: str,
                           explicit: Optional[str]) -> Optional[str]:
        if explicit:
            return explicit
        m = re.match(r"@?(\w+)[,:]", text.strip())
        if m and m.group(1) in self.participants:
            return m.group(1)
        for p in self.participants:
            if re.search(rf"\b{re.escape(p)}\b", text) and p != speaker:
                return p
        # default: the previous different speaker (they're being replied to)
        for u in reversed(self.turns):
            if u.speaker != speaker:
                return u.speaker
        return None

    def expected_speaker(self) -> Optional[str]:
        """Who is expected to speak next (the addressee of the last question/request)."""
        if not self.turns:
            return None
        last = self.turns[-1]
        if last.speech_act in (SpeechAct.QUESTION, SpeechAct.REQUEST, SpeechAct.COMMAND) \
                and last.addressee:
            return last.addressee
        return None

    def add(self, speaker: str, text: str, *, addressee: Optional[str] = None,
            at: Optional[float] = None) -> Utterance:
        self.add_participant(speaker)
        if addressee:
            self.add_participant(addressee)
        act = detect_speech_act(text)
        resolved = self._resolve_addressee(text, speaker, addressee)

        expected = self.expected_speaker()
        interruption = bool(expected and speaker != expected
                            and self.turns and self.turns[-1].speaker != speaker)

        u = Utterance(speaker=speaker, text=text, addressee=resolved, speech_act=act,
                      turn=len(self.turns), at=at if at is not None else time.time(),
                      is_interruption=interruption)
        self.turns.append(u)
        return u

    # ---- introspection ---- #
    def by_speaker(self, speaker: str) -> List[Utterance]:
        return [u for u in self.turns if u.speaker == speaker]

    def last(self) -> Optional[Utterance]:
        return self.turns[-1] if self.turns else None

    def interruptions(self) -> List[Utterance]:
        return [u for u in self.turns if u.is_interruption]

    def floor(self) -> Optional[str]:
        return self.turns[-1].speaker if self.turns else None

    def to_dict(self) -> Dict[str, Any]:
        return {"participants": self.participants,
                "turns": [u.to_dict() for u in self.turns],
                "interruptions": len(self.interruptions())}


# --------------------------------------------------------------------------- #
# Implicature (Grice)
# --------------------------------------------------------------------------- #
class ImplicatureType(str, Enum):
    SCALAR = "scalar"
    RELEVANCE = "relevance"
    QUANTITY = "quantity"
    QUALITY = "quality"        # irony / sarcasm
    MANNER = "manner"


@dataclass
class Implicature:
    type: ImplicatureType
    implied: str
    maxim_flouted: str
    confidence: float
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type.value, "implied": self.implied,
                "maxim": self.maxim_flouted, "confidence": round(self.confidence, 3)}


class ImplicatureEngine:
    """Derives conversational implicatures from an utterance (rule-based)."""

    _SCALARS = {"some": "not all", "sometimes": "not always", "might": "not certainly",
                "many": "not all", "few": "not many", "or": "not both",
                "tried": "did not necessarily succeed", "started": "did not finish"}
    _RELEVANCE = [
        (r"\bit'?s (cold|chilly|freezing)\b", "please warm it / close the window"),
        (r"\bit'?s (hot|stuffy|warm)\b", "please cool it / open a window"),
        (r"\bit'?s (dark|dim)\b", "please turn on a light"),
        (r"\bit'?s (loud|noisy)\b", "please lower the noise"),
        (r"\bi'?m (thirsty|hungry|tired)\b", "please address my need"),
        (r"\bthe (door|window) is open\b", "please close it"),
        (r"\bthe (trash|bin) is full\b", "please empty it"),
    ]
    _IRONY = [r"\boh,? (great|wonderful|fantastic|perfect)\b", r"\byeah,? right\b",
              r"\bjust great\b", r"\bwhat a surprise\b", r"\bnice going\b"]

    def analyze(self, text: str) -> List[Implicature]:
        t = text.lower()
        out: List[Implicature] = []

        for word, implied in self._SCALARS.items():
            if re.search(rf"\b{word}\b", t):
                out.append(Implicature(ImplicatureType.SCALAR, implied, "quantity", 0.8, text))
                break

        for pat, implied in self._RELEVANCE:
            if re.search(pat, t):
                out.append(Implicature(ImplicatureType.RELEVANCE, implied, "relation",
                                       0.75, text))
                break

        for pat in self._IRONY:
            if re.search(pat, t):
                out.append(Implicature(ImplicatureType.QUALITY,
                                       "the literal positive is meant negatively",
                                       "quality", 0.7, text))
                break

        return out


# --------------------------------------------------------------------------- #
# Subtext
# --------------------------------------------------------------------------- #
@dataclass
class Subtext:
    indirect_request: bool = False
    request_hint: Optional[str] = None
    politeness: float = 0.5
    hedged: bool = False
    sarcasm: bool = False
    emotion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"indirect_request": self.indirect_request, "request_hint": self.request_hint,
                "politeness": round(self.politeness, 3), "hedged": self.hedged,
                "sarcasm": self.sarcasm, "emotion": self.emotion}


class SubtextDetector:
    """Reads beneath the surface: hints, hedges, politeness, sarcasm."""

    _INDIRECT = [
        (r"\b(could|would|can|will) you\b", "polite request"),
        (r"\bwould you mind\b", "very polite request"),
        (r"\bdo you think you could\b", "tentative request"),
        (r"\bit would be (nice|great|helpful) if\b", "soft request"),
        (r"\bi (wonder|was wondering) if\b", "indirect ask"),
        (r"\bif you (have|get) (a moment|time)\b", "deferential request"),
    ]
    _HEDGES = ["maybe", "perhaps", "sort of", "kind of", "i guess", "i think",
               "possibly", "probably", "i suppose", "might", "just"]
    _SARCASM = [r"\boh,? (great|wonderful|fantastic)\b", r"\byeah,? right\b",
                r"\bjust great\b", r"\bthanks a lot\b.*\b(broken|again|nothing)\b"]
    _EMOTION = {"angry": ["furious", "angry", "annoyed", "annoying", "frustrated",
                          "frustrating", "irritated", "mad"],
                "sad": ["sad", "down", "lonely", "upset", "hurt", "miserable"],
                "happy": ["happy", "glad", "delighted", "pleased", "excited", "thrilled"],
                "afraid": ["scared", "afraid", "worried", "anxious", "nervous", "terrified"]}

    def detect(self, text: str) -> Subtext:
        t = text.lower()
        s = Subtext()

        for pat, hint in self._INDIRECT:
            if re.search(pat, t):
                s.indirect_request = True
                s.request_hint = hint
                break
        # a bare environmental hint is also an indirect request
        if not s.indirect_request and re.search(
                r"\bit'?s (cold|hot|dark|loud|stuffy)\b|\bi'?m (thirsty|hungry)\b", t):
            s.indirect_request = True
            s.request_hint = "environmental hint"

        polite_markers = sum(1 for w in ("please", "thank", "would", "could", "sorry",
                                         "kindly", "appreciate") if w in t)
        s.politeness = _clamp(0.4 + 0.2 * polite_markers)

        s.hedged = any(re.search(rf"\b{re.escape(h)}\b", t) for h in self._HEDGES)
        s.sarcasm = any(re.search(p, t) for p in self._SARCASM)

        for emo, words in self._EMOTION.items():
            if any(re.search(rf"\b{w}\b", t) for w in words):
                s.emotion = emo
                break
        return s


# --------------------------------------------------------------------------- #
# Dialogue manager (facade)
# --------------------------------------------------------------------------- #
class DialogueManager:
    """Tracks a conversation and analyses pragmatics on each utterance."""

    def __init__(self, participants: Sequence[str] = ()) -> None:
        self.conversation = Conversation(participants)
        self.implicature = ImplicatureEngine()
        self.subtext = SubtextDetector()

    def add(self, speaker: str, text: str, *, addressee: Optional[str] = None
            ) -> Dict[str, Any]:
        utt = self.conversation.add(speaker, text, addressee=addressee)
        return {
            "utterance": utt,
            "implicatures": self.implicature.analyze(text),
            "subtext": self.subtext.detect(text),
        }

    def status(self) -> Dict[str, Any]:
        return {"conversation": self.conversation.to_dict(),
                "expected_next": self.conversation.expected_speaker(),
                "floor": self.conversation.floor()}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA dialogue self-test")
    print("=" * 70)

    # multi-party conversation with turn-taking
    dm = DialogueManager(["JP", "NYXARA", "Guest"])
    dm.add("JP", "NYXARA, what is the network status?")
    print(f"\nspeech act          : {dm.conversation.last().speech_act.value} "
          f"(addressee {dm.conversation.last().addressee})")
    assert dm.conversation.last().speech_act is SpeechAct.QUESTION
    assert dm.conversation.last().addressee == "NYXARA"
    assert dm.conversation.expected_speaker() == "NYXARA"

    # an interruption: Guest speaks when NYXARA was expected
    r = dm.add("Guest", "Actually, let me say something.")
    print(f"interruption        : {r['utterance'].is_interruption}")
    assert r["utterance"].is_interruption

    # Gricean scalar implicature
    imp = dm.implicature.analyze("Some of the defenses held.")
    print(f"\nscalar implicature  : {[i.to_dict() for i in imp]}")
    assert any(i.implied == "not all" for i in imp)

    # relevance implicature — an indirect request
    imp2 = dm.implicature.analyze("It's cold in here.")
    print(f"relevance implicature: {[i.to_dict() for i in imp2]}")
    assert any(i.type is ImplicatureType.RELEVANCE for i in imp2)

    # irony
    imp3 = dm.implicature.analyze("Oh great, another security alert.")
    assert any(i.type is ImplicatureType.QUALITY for i in imp3)
    print(f"irony detected      : meant negatively ✓")

    # subtext: indirect request + politeness + hedging
    st = dm.subtext.detect("Could you maybe, if you have a moment, check the logs please?")
    print(f"\nsubtext             : {st.to_dict()}")
    assert st.indirect_request and st.hedged and st.politeness > 0.5

    # sarcasm
    sarc = dm.subtext.detect("Yeah right, that'll work.")
    assert sarc.sarcasm
    print(f"sarcasm             : detected ✓")

    # emotional subtext
    emo = dm.subtext.detect("I'm really worried about the breach.")
    assert emo.emotion == "afraid"
    print(f"emotion subtext     : {emo.emotion}")

    print(f"\nstatus              : floor={dm.status()['floor']} "
          f"participants={dm.conversation.participants}")
    print("\nALL SELF-TESTS PASSED ✓")
