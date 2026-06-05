"""NYXARA · social/culture.py — context norms & register switching (incl. Hinglish) (✦).

Speaking *correctly* is not enough; NYXARA must speak *appropriately* — in the right
register, the right language, with the right deference. This module gives it cultural
pragmatic competence:

* **Code detection & mirroring** — is the input English, Hindi (Devanagari or
  romanised), or **Hinglish** (the natural Hindi-English code-switch)? When the Master
  writes "kal ka plan ready hai?", NYXARA answers in kind, not in stiff English.
* **Register switching** — FORMAL / NEUTRAL / CASUAL / INTIMATE, chosen from the
  setting and the relationship (warm and easy with the Master, correct and guarded
  with a stranger).
* **Honorifics & address** — the Master is addressed as such; others by their due.
* **Politeness strategy** (Brown & Levinson) — positive politeness with intimates,
  negative (deferential, hedged) politeness across distance, bald directness when the
  face-threat is low.
* **Norms** — what is expected in a given setting.

The output is *guidance* (register + code + politeness + a system fragment); an LLM
renders the actual words. Pure standard library; optional :mod:`social.persons`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from nyxara.social.persons import Roster, TrustLevel

__all__ = [
    "Register",
    "CodeProfile",
    "StyleGuidance",
    "CultureSystem",
]


# --------------------------------------------------------------------------- #
# Register
# --------------------------------------------------------------------------- #
class Register(str, Enum):
    FORMAL = "formal"
    NEUTRAL = "neutral"
    CASUAL = "casual"
    INTIMATE = "intimate"


# --------------------------------------------------------------------------- #
# Language / code detection
# --------------------------------------------------------------------------- #
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# common romanised-Hindi function words (the glue of Hinglish)
_HINDI_MARKERS = frozenset("""
kya hai hain nahi nahin nai kar karo karna kiya raha rahi rahe ho hoga hona
tum tu tum aap mera meri mere tera teri tumhara aapka humara hum main mujhe tujhe
bhai yaar arre acha accha theek thik matlab kyun kyon kaise kaisa kab kahan abhi
chal chalo bas haan haa ji na bhi toh to ka ki ke ko se me mein par pe wala wali
zaroor bilkul shukriya dhanyavaad namaste kal aaj kal sab kuch ek do teen
""".split())

_ENGLISH_COMMON = frozenset("""
the a an is are was were be been do does did will would can could should you i we
they it he she and or but if then this that here there what when where why how please
""".split())

# English content words that commonly code-switch into Hinglish
_ENGLISH_CONTENT = frozenset("""
plan ready meeting status network check done time work call message problem fix
system file code project update ok okay thanks sorry email report data server log
defense security task help test run start stop now today tomorrow office home phone
deploy backup scan alert firewall password login logout dashboard team boss
""".split())


def _is_english(word: str) -> bool:
    return word in _ENGLISH_COMMON or word in _ENGLISH_CONTENT


@dataclass
class CodeProfile:
    language: str           # "en" | "hi" | "hinglish"
    romanized: bool         # Hindi written in Latin script
    hindi_ratio: float
    register: Register

    def to_dict(self) -> Dict[str, Any]:
        return {"language": self.language, "romanized": self.romanized,
                "hindi_ratio": round(self.hindi_ratio, 3), "register": self.register.value}


# --------------------------------------------------------------------------- #
# Style guidance
# --------------------------------------------------------------------------- #
@dataclass
class StyleGuidance:
    register: Register
    language: str
    honorific: str
    politeness: str
    fragment: str

    def to_dict(self) -> Dict[str, Any]:
        return {"register": self.register.value, "language": self.language,
                "honorific": self.honorific, "politeness": self.politeness,
                "fragment": self.fragment}


# --------------------------------------------------------------------------- #
# Culture system
# --------------------------------------------------------------------------- #
class CultureSystem:
    """Detects code/register and produces culturally-appropriate style guidance."""

    _NORMS = {
        "professional": ["use formal register", "avoid slang", "be concise and precise",
                         "use honorifics"],
        "personal": ["warmth is welcome", "casual register is fine",
                     "mirror the other's language"],
        "family": ["intimate register", "affectionate address", "informal"],
        "public": ["neutral register", "inclusive language", "avoid private references"],
    }

    def __init__(self, roster: Optional[Roster] = None) -> None:
        self.roster = roster or Roster()

    # ---- code detection ---- #
    def detect_code(self, text: str) -> CodeProfile:
        has_dev = bool(_DEVANAGARI.search(text))
        words = re.findall(r"[A-Za-zऀ-ॿ']+", text.lower())
        latin = [w for w in words if not _DEVANAGARI.search(w)]
        hindi_markers = sum(1 for w in latin if w in _HINDI_MARKERS)
        english_count = sum(1 for w in latin if _is_english(w))
        n = max(1, len(words))
        hindi_ratio = (hindi_markers + (len(words) - len(latin))) / n

        if has_dev and latin and (english_count or len(latin) > 2):
            language, romanized = "hinglish", False        # script-mixed
        elif has_dev and not latin:
            language, romanized = "hi", False
        elif hindi_markers >= 1 and english_count >= 1:
            language, romanized = "hinglish", True          # romanised mix
        elif hindi_markers >= 2 and hindi_ratio > 0.4:
            language, romanized = "hi", True                # romanised Hindi
        else:
            language, romanized = "en", False

        return CodeProfile(language=language, romanized=romanized,
                           hindi_ratio=hindi_ratio, register=self.infer_register(text))

    def infer_register(self, text: str) -> Register:
        t = text.lower()
        if re.search(r"\b(sir|madam|kindly|please|respected|would you|may i)\b", t) \
                or re.search(r"\b(aap|aapka|aapko|ji)\b", t):
            return Register.FORMAL
        if re.search(r"\b(yaar|bhai|arre|dude|bro|lol|haha|abey|oye)\b", t):
            return Register.CASUAL
        return Register.NEUTRAL

    def match_code(self, text: str) -> CodeProfile:
        """Mirror the other party's code (what language/register to reply in)."""
        return self.detect_code(text)

    # ---- person-aware register & address ---- #
    def register_for(self, person: str) -> Register:
        trust = self.roster.assess(person)
        return {
            TrustLevel.OWNER: Register.CASUAL,       # warm, easy with the Master
            TrustLevel.ALLY: Register.NEUTRAL,
            TrustLevel.NEUTRAL: Register.FORMAL,
            TrustLevel.SUSPECT: Register.FORMAL,
            TrustLevel.THREAT: Register.FORMAL,      # correct and guarded
        }.get(trust, Register.FORMAL)

    def honorific(self, person: str) -> str:
        p = self.roster.get(person)
        if p is not None and p.is_owner:
            return "Master"
        trust = self.roster.assess(person)
        if trust is TrustLevel.ALLY:
            return p.name if p else person
        return "Sir/Madam"

    # ---- politeness (Brown & Levinson) ---- #
    def politeness(self, *, face_threat: float, closeness: float) -> str:
        if closeness > 0.6:
            return "positive"      # friendly, inclusive, in-group
        if face_threat > 0.5:
            return "negative"      # deferential, hedged, apologetic, indirect
        return "bald"              # direct, efficient

    def norms(self, setting: str) -> List[str]:
        return self._NORMS.get(setting, self._NORMS["public"])

    # ---- top-level adaptation ---- #
    def adapt(self, *, person: Optional[str] = None, text: Optional[str] = None,
              setting: str = "personal", face_threat: float = 0.3) -> StyleGuidance:
        # register/language: mirror the input if given, else derive from the person
        if text:
            code = self.detect_code(text)
            language, register = code.language, code.register
        else:
            language = "en"
            register = self.register_for(person) if person else Register.NEUTRAL

        if person:
            # a person's relationship can override a neutral text-derived register
            person_reg = self.register_for(person)
            if register is Register.NEUTRAL:
                register = person_reg

        honorific = self.honorific(person) if person else "you"
        closeness = 0.0
        if person:
            p = self.roster.get(person)
            closeness = 1.0 if (p and p.is_owner) else max(0.0, p.rapport if p else 0.0)
        politeness = self.politeness(face_threat=face_threat, closeness=closeness)

        lang_note = {
            "hinglish": "Reply in natural Hinglish, code-switching as the Master does.",
            "hi": "Reply in Hindi.",
            "en": "Reply in English.",
        }[language]
        fragment = (f"{lang_note} Use a {register.value} register, address them as "
                    f"'{honorific}', with {politeness} politeness.")

        return StyleGuidance(register=register, language=language, honorific=honorific,
                             politeness=politeness, fragment=fragment)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA culture self-test")
    print("=" * 70)

    culture = CultureSystem(Roster(owner_name="Jaypal Khoja", owner_aliases=["JP"]))

    # code detection
    cases = {
        "What is the network status?": "en",
        "नेटवर्क की स्थिति क्या है?": "hi",
        "yaar kal ka plan ready hai kya?": "hinglish",
        "abhi defense ready karo, jaldi": "hinglish",
        "kal meeting hai, please ready rehna": "hinglish",
    }
    print()
    for text, expected in cases.items():
        code = culture.detect_code(text)
        print(f"  {code.language:9} ({'romanized' if code.romanized else 'native':9}) "
              f"<- {text}")
        assert code.language == expected, (text, code.language)

    # register inference
    assert culture.infer_register("Sir, would you kindly review this?") is Register.FORMAL
    assert culture.infer_register("arre yaar, chill kar") is Register.CASUAL
    print("\nregister inference   : formal/casual detected ✓")

    # mirroring: the Master writes Hinglish -> NYXARA replies in Hinglish
    g = culture.adapt(person="JP", text="kal ka kaam ready hai kya?")
    print(f"\nadapt (JP, hinglish):")
    print(f"  {g.fragment}")
    assert g.language == "hinglish"
    assert g.honorific == "Master"
    assert g.register is Register.CASUAL or "Hinglish" in g.fragment

    # a stranger -> formal, guarded, deferential
    culture.roster.get_or_create("Stranger")
    g2 = culture.adapt(person="Stranger", text="Hello, who are you?", face_threat=0.7)
    print(f"\nadapt (stranger)     : register={g2.register.value} "
          f"honorific='{g2.honorific}' politeness={g2.politeness}")
    assert g2.register is Register.FORMAL
    assert g2.honorific == "Sir/Madam"
    assert g2.politeness == "negative"

    # honorific & register for the Master
    assert culture.honorific("JP") == "Master"
    assert culture.register_for("JP") is Register.CASUAL
    print("\nMaster address       : 'Master', casual+warm ✓")

    # norms per setting
    print(f"\nprofessional norms   : {culture.norms('professional')}")
    assert "use formal register" in culture.norms("professional")

    # politeness strategy
    assert culture.politeness(face_threat=0.2, closeness=0.9) == "positive"
    assert culture.politeness(face_threat=0.8, closeness=0.1) == "negative"
    assert culture.politeness(face_threat=0.1, closeness=0.2) == "bald"
    print("politeness strategies: positive/negative/bald ✓")

    print("\nALL SELF-TESTS PASSED ✓")
