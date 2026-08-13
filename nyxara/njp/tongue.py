"""NYXARA · nyx/lingua.py — the script-aware tongue (🗣, NYX V.02, gap 2).

NYX V.01 read the world through one regular expression: ``[a-z0-9][a-z0-9\\-']*``. That is a
statement about which languages exist. Measured, it said so out loud —
``concepts_in("गुरुत्वाकर्षण सेब को खींचता है")`` returned ``[]``. Devanagari was not
mis-handled, it was *invisible*. And in Hinglish the function words — "ka", "ki", "hai",
"kya", "aur" — were not stop-words, so they co-occurred with everything and became the
**hubs** of her concept graph.

:class:`Lingua` replaces that regex with a Unicode scanner that has no favourite alphabet:

* **Script-aware tokenisation** — every token carries the Unicode script it is written in
  (Devanagari, Latin, Arabic, Han, Cyrillic, …), derived from the character database rather
  than from a hard-coded range table, so a script nobody thought about still tokenises.
* **Function words in three registers** — English, romanised Hinglish, and Devanagari. This is
  a stop-list, not an ontology: it exists so that grammar does not become structure.
* **Code-mixing** — a turn may be in more than one language at once. Hinglish is a
  first-class language here, not a broken dialect of either parent.
* **Register and tone** — formality, politeness, humour and emphasis are *recorded* features
  of what arrived. They are not generated: :mod:`nyxara.nyx.dialogue` reads them, and still
  refuses to fake a fluent surface it does not have.
* **A transliteration bridge** — "गुरुत्वाकर्षण" and "gurutvakarshan" fold to the same key, so
  :mod:`nyxara.nyx.semantics` can later learn that both are "gravity" without either spelling
  being privileged.

Honest framing (the rule this repo keeps, see ``docs/CAPABILITIES.md``):

* Language identification is **evidence-based, not lexical**. Nothing here ships a dictionary
  of English or Hindi, so the evidence is exactly two things: the script a token is written
  in, and membership of the small function-word lists in this file. A content word in Latin
  script ("code", "test") carries no evidence of its language and is *reported* as inheriting
  the turn's dominant language rather than being confidently mislabelled.
* Transliteration is **approximate**. It applies ITRANS-style consonant/matra mapping with
  word-final schwa deletion, which is right far more often than not and wrong on words whose
  medial schwa also deletes ("कमरा" → ``kamara``, not ``kamra``). The folded key is designed
  to absorb most of that, and the module never claims the mapping is a reversible one.
* Register is a **surface signal**: capitals, repetition, punctuation and a handful of named
  markers. It is not a model of what the Master feels.

Pure standard library. Every public entry point is fail-soft: on any error it degrades to a
null result rather than breaking a turn.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "Token", "Register", "LinguaRead", "Lingua",
    "tokenize", "content_tokens", "concepts_in", "script_of", "detect_language",
    "transliterate", "roman_key", "register_of", "is_function_word",
]


# --------------------------------------------------------------------------- #
# Function words — a stop-list, deliberately small, in three registers.
#
# These are the words that carry grammar rather than content. Left in, they co-occur with
# everything and become the hubs of the concept graph, which is precisely the bug this file
# exists to fix. Everything here is a *closed class* word: articles, pronouns, copulas,
# postpositions, conjunctions, light verbs. No content nouns, ever.
# --------------------------------------------------------------------------- #

EN_FUNCTION = frozenset("""
a an the and or but if then than that this these those there here of to in on at by for with
from as is am are was were be been being do does did done have has had having i you he she it
we they me him her us them my your his its our their not no yes so such very just only also
what which who whom whose when where why how can could shall should will would may might must
into onto over under about after before while because though although unless until since via
""".split())

# Romanised Hindi/Urdu — the shape Hinglish actually arrives in. Spelling is not standardised,
# so the common variants are listed rather than normalised away.
#
# Split in two on purpose. These are the spellings that are *only* ever Hindi, so meeting one
# is proof the turn is Hinglish.
HI_LATN_FUNCTION = frozenset("""
aur ya ka ki ke ko se mein mai hai hain hu hun hoon hota hoti hote hone ne
thi thay toh tho bhi hee nahi nahin nai kya kaun kab kahan kaha kaise kese
kyu kyun kyon kyunki kyonki agar lekin magar phir fir pehle pahle baad jab tab
yeh woh yah vah un isko usko iska uska inka unka isse usse
mera meri mere tera teri tere tumhara tumhari tumhare aap aapka aapki aapke tum tu
mujhe mujhko hume humein humara hamara sabhi kuch kuchh koi apna apni apne
wahi yahi sirf wala wali wale liye saath sath bina jaise waise vaise abhi
yaha yahan waha wahan karo kare karen karein karna karke kiya kiye
diya diye dena liya lena raha rahi rahe rha gaya gayi gaye chahiye
sakta sakti sakte hoga hogi honge
""".split())

# ...and these are Hindi function words whose romanisation collides with an ordinary English
# word. "mat" is *don't* in Hindi and a thing you wipe your feet on in English. Treating them
# as grammar unconditionally would silently delete English content, so they only count as
# function words in a turn that has already proved itself Hinglish — see ``_resolve_function``.
HI_LATN_AMBIGUOUS = frozenset("""
mat main me in is us par pe hi to do so the ho de le bas ye hum lie na jo tha tak ab sab kar
""".split())

# Devanagari. Same closed classes, written in the script they are usually written in.
DEVA_FUNCTION = frozenset("""
और या का की के को से में पर है हैं हूँ हूं हो होता होती होते होने था थी थे
भी ही तो नहीं ना मत क्या कौन कब कहाँ कहां कैसे क्यों क्योंकि अगर लेकिन मगर
फिर पहले बाद अब जब तब तक ये यह वो वह इस उस इन उन इसका उसका इसे उसे
मेरा मेरी मेरे तेरा तुम्हारा आप आपका आपकी आपके तुम तू मैं मुझे हम हमें हमारा
सब सभी कुछ कोई अपना अपनी अपने जो वही यही सिर्फ बस वाला वाली वाले लिए साथ बिना
जैसे वैसे अभी यहाँ यहां वहाँ वहां जी कर करो करना करके किया दिया देना दे ले लिया लेना
रहा रही रहे गया गई गए चाहिए सकता सकती सकते होगा होगी होंगे एक
""".split())

# The unconditional union — what is grammar in any turn, whatever language it turns out to be.
# Kept as one frozenset because the hot path is a membership test, not a language question.
_ALL_FUNCTION = EN_FUNCTION | HI_LATN_FUNCTION | DEVA_FUNCTION


# --------------------------------------------------------------------------- #
# Script detection — from the Unicode character database, not a range table.
# --------------------------------------------------------------------------- #

# Unicode does not expose Script directly in the stdlib, but every character's *name* begins
# with its script for the alphabets that matter here ("DEVANAGARI LETTER KA", "LATIN SMALL
# LETTER A", "CYRILLIC CAPITAL LETTER A"). Reading the name is slower than a range check and
# is cached per codepoint; the payoff is that a script nobody enumerated still resolves.
_NAME_SCRIPTS = (
    "LATIN", "DEVANAGARI", "ARABIC", "CYRILLIC", "GREEK", "HEBREW", "THAI", "HANGUL",
    "HIRAGANA", "KATAKANA", "CJK", "BENGALI", "GUJARATI", "GURMUKHI", "TAMIL", "TELUGU",
    "KANNADA", "MALAYALAM", "ORIYA", "SINHALA", "TIBETAN", "MYANMAR", "KHMER", "LAO",
    "GEORGIAN", "ARMENIAN", "ETHIOPIC", "CHEROKEE",
)

# Script → the language it is overwhelmingly written in, when a token gives no other evidence.
# Latin and Devanagari are deliberately absent: both are shared, and guessing is the bug.
_SCRIPT_LANG = {
    "arabic": "ar", "cyrillic": "ru", "greek": "el", "hebrew": "he", "thai": "th",
    "hangul": "ko", "hiragana": "ja", "katakana": "ja", "cjk": "zh", "bengali": "bn",
    "gujarati": "gu", "gurmukhi": "pa", "tamil": "ta", "telugu": "te", "kannada": "kn",
    "malayalam": "ml", "oriya": "or", "sinhala": "si", "georgian": "ka", "armenian": "hy",
}

_KIND_WORD, _KIND_NUMBER, _KIND_EMOJI, _KIND_PUNCT = "word", "number", "emoji", "punct"

# Characters that may sit *inside* a word without ending it, as the V.01 regex allowed.
_WORD_JOINERS = frozenset("-'’_‍")
# Characters that may sit inside a number ("3.14", "1,024").
_NUM_JOINERS = frozenset(".,")


@lru_cache(maxsize=8192)
def script_of(ch: str) -> str:
    """The Unicode script of one character, lowercased — ``"latin"``, ``"devanagari"``, ….

    Returns ``"digit"`` for decimal digits, ``"emoji"`` for pictographs, ``"common"`` for
    punctuation and separators, and ``"unknown"`` when the character database has no name for
    it. Never raises: an unnameable character is simply unknown, not an error.
    """
    try:
        if not ch:
            return "unknown"
        cat = unicodedata.category(ch)
        if cat == "Nd":
            return "digit"
        if _is_emoji_char(ch):
            return "emoji"
        if cat[0] not in ("L", "M"):
            return "common"
        try:
            name = unicodedata.name(ch)
        except ValueError:
            return "unknown"
        head = name.split(" ", 1)[0]
        for script in _NAME_SCRIPTS:
            if head.startswith(script):
                return script.lower()
        return head.lower()
    except Exception:  # noqa: BLE001 — an odd codepoint is never allowed to break a turn
        return "unknown"


def _is_emoji_char(ch: str) -> bool:
    """Pictographs, dingbats and their modifiers — anything that is a picture, not a letter."""
    cp = ord(ch)
    return (
        0x1F300 <= cp <= 0x1FAFF      # the emoji planes
        or 0x1F000 <= cp <= 0x1F0FF   # mahjong / dominoes / cards
        or 0x2600 <= cp <= 0x27BF     # misc symbols and dingbats
        or 0x2190 <= cp <= 0x21FF     # arrows, which arrive as emoji far more often than not
        or 0xFE00 <= cp <= 0xFE0F     # variation selectors
        or 0x1F1E6 <= cp <= 0x1F1FF   # regional indicators (flags)
        or cp in (0x00A9, 0x00AE, 0x2122, 0x203C, 0x2049, 0x2B50, 0x2B55, 0x3030)
    )


# --------------------------------------------------------------------------- #
# Transliteration — Devanagari → Latin, and a folded key both sides can meet on.
# --------------------------------------------------------------------------- #

_DEVA_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "ळ": "l",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f",
}

_DEVA_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e",
}

_DEVA_MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu",
    "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    "ॉ": "o", "ॅ": "e",
}

_DEVA_VIRAMA = "्"
# Anusvara, chandrabindu, visarga — nasalisation and aspiration marks.
_DEVA_SIGNS = {"ं": "n", "ँ": "n", "ः": "h"}
_DEVA_DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}


def transliterate(text: str) -> str:
    """Romanise Devanagari with ITRANS-style mapping and word-final schwa deletion.

    Text that is not Devanagari is returned lowercased and otherwise untouched, so this is
    safe to call on anything. Approximate by construction — see the module docstring.
    """
    try:
        text = str(text or "")
        if not text:
            return ""
        out: List[str] = []
        pending_a = False          # a consonant's inherent vowel, not yet committed
        for ch in text:
            if ch in _DEVA_CONSONANTS:
                if pending_a:
                    out.append("a")
                out.append(_DEVA_CONSONANTS[ch])
                pending_a = True
            elif ch in _DEVA_MATRAS:
                out.append(_DEVA_MATRAS[ch])
                pending_a = False
            elif ch == _DEVA_VIRAMA:
                pending_a = False   # the inherent vowel is explicitly killed
            elif ch in _DEVA_VOWELS:
                if pending_a:
                    out.append("a")
                    pending_a = False
                out.append(_DEVA_VOWELS[ch])
            elif ch in _DEVA_SIGNS:
                if pending_a:
                    out.append("a")
                    pending_a = False
                out.append(_DEVA_SIGNS[ch])
            elif ch in _DEVA_DIGITS:
                if pending_a:
                    out.append("a")
                    pending_a = False
                out.append(_DEVA_DIGITS[ch])
            elif ch == "़" or ch == "ॎ" or ch == "ऽ":
                continue            # nukta / prishthamatra / avagraha carry no sound alone
            else:
                # Word boundary: the trailing inherent vowel is dropped, which is what makes
                # "सेब" come out as "seb" and not "seba".
                pending_a = False
                out.append(ch.lower())
        return "".join(out)
    except Exception:  # noqa: BLE001
        return str(text or "").lower()


_FOLD_PAIRS = (("aa", "a"), ("ii", "i"), ("ee", "i"), ("uu", "u"), ("oo", "u"),
               ("ph", "f"), ("w", "v"))
_DOUBLE_RE = re.compile(r"(.)\1+")


def roman_key(text: str) -> str:
    """A spelling-tolerant key: the bridge between "गुरुत्वाकर्षण" and "gurutvakarshan".

    Romanise, then fold the distinctions that romanised Hindi does not spell consistently —
    long vowels, doubled letters, ``w``/``v``, ``ph``/``f``. Two spellings of the same word
    land on the same key; that is all this promises. It is a matching key, not a meaning.
    """
    try:
        key = transliterate(text).lower()
        key = "".join(ch for ch in key if ch.isalnum())
        for src, dst in _FOLD_PAIRS:
            key = key.replace(src, dst)
        key = _DOUBLE_RE.sub(r"\1", key)
        if len(key) > 2 and key.endswith("a"):
            key = key[:-1]          # schwa that survived the romanisation
        return key
    except Exception:  # noqa: BLE001
        return ""


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #

@dataclass
class Token:
    """One unit of what arrived, with the script it was written in and a language tag."""

    text: str                    # exactly as written
    lower: str = ""              # casefolded, for matching
    script: str = "unknown"      # "latin", "devanagari", "digit", "emoji", "common", …
    kind: str = _KIND_WORD       # word | number | emoji | punct
    lang: str = "und"            # "en", "hi", "hi-Latn", … or "und" when there is no evidence
    evidence: bool = False       # True when ``lang`` came from script or a function word
    function: bool = False       # a closed-class word: grammar, not content
    maybe_function: bool = False  # grammar *if* this turn is Hinglish ("mat", "par", "hi")
    start: int = 0
    end: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "script": self.script, "kind": self.kind,
                "lang": self.lang, "function": self.function, "evidence": self.evidence}


@dataclass
class Register:
    """Surface tone of a turn — recorded, never generated."""

    formality: float = 0.5       # 0.0 casual … 1.0 formal
    politeness: float = 0.0      # explicit courtesy markers
    humor: float = 0.0           # laughter, emoji, playful punctuation
    emphasis: float = 0.0        # capitals, repetition, exclamation
    markers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"formality": round(self.formality, 3), "politeness": round(self.politeness, 3),
                "humor": round(self.humor, 3), "emphasis": round(self.emphasis, 3),
                "markers": list(self.markers)}


@dataclass
class LinguaRead:
    """Everything :class:`Lingua` can honestly say about one turn."""

    text: str = ""
    tokens: List[Token] = field(default_factory=list)
    scripts: Dict[str, int] = field(default_factory=dict)
    languages: Dict[str, float] = field(default_factory=dict)   # share of evidence tokens
    primary: str = "und"
    code_mixed: bool = False
    hinglish: bool = False
    register: Register = field(default_factory=Register)
    concepts: List[str] = field(default_factory=list)
    keys: Dict[str, str] = field(default_factory=dict)          # concept → roman_key

    @property
    def words(self) -> List[Token]:
        return [t for t in self.tokens if t.kind == _KIND_WORD]

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "primary": self.primary, "code_mixed": self.code_mixed,
                "hinglish": self.hinglish, "scripts": dict(self.scripts),
                "languages": {k: round(v, 4) for k, v in self.languages.items()},
                "register": self.register.to_dict(), "concepts": list(self.concepts),
                "keys": dict(self.keys),
                "tokens": [t.to_dict() for t in self.tokens]}


# --------------------------------------------------------------------------- #
# The scanner
# --------------------------------------------------------------------------- #

def tokenize(text: str, *, max_tokens: int = 512, keep_punct: bool = False) -> List[Token]:
    """Split ``text`` into script-tagged tokens. No alphabet is privileged.

    A run of letters (plus their combining marks, plus the joiners a word may legally contain)
    is one word token; a run of digits is one number; a run of pictographs is one emoji token.
    Punctuation is dropped unless ``keep_punct`` is set — :mod:`nyxara.nyx.intent` needs it to
    see clause boundaries, the graph does not.
    """
    out: List[Token] = []
    try:
        text = str(text or "")
        n = len(text)
        i = 0
        cap = max(1, int(max_tokens))
        while i < n and len(out) < cap:
            ch = text[i]
            s = script_of(ch)
            if s == "common" and ch.isspace():
                i += 1
                continue
            if s == "digit":
                j = i
                while j < n:
                    cj = text[j]
                    sj = script_of(cj)
                    if sj == "digit":
                        j += 1
                    elif (cj in _NUM_JOINERS and j + 1 < n
                          and script_of(text[j + 1]) == "digit"):
                        j += 1
                    else:
                        break
                out.append(_make_token(text, i, j, _KIND_NUMBER, "digit"))
                i = j
                continue
            if s == "emoji":
                j = i
                while j < n and script_of(text[j]) == "emoji":
                    j += 1
                out.append(_make_token(text, i, j, _KIND_EMOJI, "emoji"))
                i = j
                continue
            if s in ("common", "unknown"):
                if keep_punct and not ch.isspace():
                    out.append(_make_token(text, i, i + 1, _KIND_PUNCT, "common"))
                i += 1
                continue

            # A word: letters and marks of one script, digits and joiners allowed inside.
            j = i
            scripts: Dict[str, int] = {}
            while j < n:
                cj = text[j]
                sj = script_of(cj)
                if sj in ("common", "emoji", "unknown"):
                    if (cj in _WORD_JOINERS and j + 1 < n
                            and script_of(text[j + 1]) not in ("common", "emoji", "unknown")):
                        j += 1
                        continue
                    break
                if sj == "digit":
                    j += 1
                    continue
                if unicodedata.category(cj)[0] == "M":
                    j += 1              # a combining mark belongs to the letter before it
                    continue
                scripts[sj] = scripts.get(sj, 0) + 1
                j += 1
            if j == i:                  # defensive: never fail to advance
                i += 1
                continue
            dominant = max(scripts.items(), key=lambda p: p[1])[0] if scripts else "unknown"
            out.append(_make_token(text, i, j, _KIND_WORD, dominant))
            i = j
        return out
    except Exception:  # noqa: BLE001 — tokenisation never breaks a turn
        return out


def _make_token(text: str, start: int, end: int, kind: str, script: str) -> Token:
    raw = text[start:end]
    low = raw.lower()
    tok = Token(text=raw, lower=low, script=script, kind=kind, start=start, end=end)
    if kind == _KIND_WORD:
        tok.function = low in _ALL_FUNCTION
        tok.maybe_function = (not tok.function) and script == "latin" \
            and low in HI_LATN_AMBIGUOUS
        tok.lang, tok.evidence = _tag_language(low, script)
    elif kind == _KIND_NUMBER:
        tok.lang, tok.evidence = "num", False
    else:
        tok.lang, tok.evidence = "und", False
    return tok


def _tag_language(low: str, script: str) -> Tuple[str, bool]:
    """Best-effort language for one word, and whether that came from actual evidence.

    Evidence is exactly two things: a script that is written in one language, and membership
    of a function-word list. A Latin content word carries neither, and says so.
    """
    if script == "devanagari":
        return "hi", True
    if script == "latin":
        in_hi = low in HI_LATN_FUNCTION
        in_en = low in EN_FUNCTION
        if in_hi and not in_en:
            return "hi-Latn", True
        if in_en and not in_hi:
            return "en", True
        if in_en and in_hi:
            return "en", False      # shared spelling ("me", "in", "to") proves nothing
        return "und", False
    mapped = _SCRIPT_LANG.get(script)
    if mapped:
        return mapped, True
    return "und", False


def _resolve_function(tokens: List[Token]) -> bool:
    """Decide the ambiguous Hinglish function words for this turn. Returns *is it Hinglish*.

    Evidence is one unmistakable Hindi marker — a Devanagari word, or a romanisation that is
    only ever Hindi ("aur", "nahi", "kyunki"). Without one, "mat" stays a mat.
    """
    hinglish = any(t.kind == _KIND_WORD and t.evidence and t.lang in ("hi", "hi-Latn")
                   for t in tokens)
    if hinglish:
        for t in tokens:
            if t.maybe_function:
                t.function = True
                if t.lang == "und":
                    t.lang, t.evidence = "hi-Latn", True
    return hinglish


def is_function_word(word: str) -> bool:
    """Is this a closed-class word — grammar rather than content — in any register we know?"""
    try:
        return str(word or "").lower() in _ALL_FUNCTION
    except Exception:  # noqa: BLE001
        return False


def content_tokens(text: str, *, cap: int = 32, min_len: int = 2) -> List[str]:
    """Concept labels: content words, lowercased, order kept, duplicates dropped.

    This is the replacement for the V.01 ASCII-only extractor, and :func:`concepts_in`
    below is the fail-soft wrapper every caller should use. Numbers, emoji, punctuation and
    function words in any of the three registers are excluded — a measurement is not a
    concept, and grammar is not structure.
    """
    try:
        out: List[str] = []
        seen = set()
        floor = max(1, int(min_len))
        toks = tokenize(text, max_tokens=max(32, int(cap) * 8))
        _resolve_function(toks)
        for tok in toks:
            if tok.kind != _KIND_WORD or tok.function:
                continue
            low = tok.lower
            # Devanagari packs a syllable into one codepoint, so its words are legitimately
            # shorter than the Latin floor allows for.
            limit = floor if tok.script == "latin" else 1
            if len(low) < limit or low in seen or low.isdigit():
                continue
            seen.add(low)
            out.append(low)
            if len(out) >= max(0, int(cap)):
                break
        return out
    except Exception:  # noqa: BLE001 — extraction never breaks a turn
        return []


# --------------------------------------------------------------------------- #
# Register / tone
# --------------------------------------------------------------------------- #

_POLITE = frozenset("""
please kindly sorry thanks thank welcome regards pls plz kripya kripaya dhanyavaad shukriya
maaf मेहरबानी कृपया धन्यवाद शुक्रिया माफ़ माफ
""".split())
_CASUAL = frozenset("""
yaar yar bhai bro dude bruh arre arey abey abe oye lol lmao lmfao wtf yo yeah yep nah nope
भाई यार अरे ओए
""".split())
_HUMOR = frozenset("""
lol lmao lmfao rofl haha hahaha hehe heh hihi xd lel kek joke joking kidding mazak मज़ाक मजाक
""".split())
_FORMAL_HINT = frozenset("""
aap aapka aapki aapke sir madam respected sincerely regarding therefore hereby accordingly
आप आपका आपकी आपके श्रीमान महोदय
""".split())
_ELONG_RE = re.compile(r"([a-z])\1{2,}", re.I)
_BANG_RE = re.compile(r"[!?]{2,}")


def register_of(text: str, tokens: Optional[List[Token]] = None) -> Register:
    """Read formality, politeness, humour and emphasis off the surface of a turn."""
    reg = Register()
    try:
        text = str(text or "")
        if not text.strip():
            return reg
        toks = tokens if tokens is not None else tokenize(text, keep_punct=True)
        words = [t for t in toks if t.kind == _KIND_WORD]
        emojis = [t for t in toks if t.kind == _KIND_EMOJI]
        n = max(1, len(words))

        polite = sum(1 for t in words if t.lower in _POLITE)
        casual = sum(1 for t in words if t.lower in _CASUAL)
        funny = sum(1 for t in words if t.lower in _HUMOR)
        formal = sum(1 for t in words if t.lower in _FORMAL_HINT)
        shouted = sum(1 for t in words if len(t.text) >= 3 and t.text.isupper())
        elongated = len(_ELONG_RE.findall(text))
        bangs = len(_BANG_RE.findall(text))

        reg.politeness = min(1.0, (polite + 0.5 * formal) / n * 3.0)
        reg.humor = min(1.0, (funny + 0.5 * len([e for e in emojis])) / n * 3.0)
        reg.emphasis = min(1.0, (shouted / n) + 0.25 * bangs + 0.15 * elongated)
        # Formality starts neutral and is moved by what is actually present.
        score = 0.5 + 0.20 * polite + 0.20 * formal - 0.25 * casual - 0.15 * funny
        score -= 0.10 * len(emojis) + 0.10 * min(2, bangs)
        if "'" in text and any(t.lower.endswith(("n't", "'s", "'re", "'ll", "'ve"))
                               for t in words):
            score -= 0.05                      # contractions read as informal
        reg.formality = max(0.0, min(1.0, score))

        markers: List[str] = []
        if polite:
            markers.append("polite")
        if formal:
            markers.append("deferential")
        if casual:
            markers.append("casual")
        if funny or emojis:
            markers.append("playful")
        if shouted or bangs:
            markers.append("emphatic")
        if elongated:
            markers.append("elongated")
        reg.markers = markers
        return reg
    except Exception:  # noqa: BLE001
        return reg


def detect_language(text: str) -> str:
    """The dominant language of ``text`` — ``"und"`` when nothing in it is evidence.

    Unlike :func:`nyxara.senses.nlp.detect_language`, which sees only ASCII stop-words and
    answers ``"en"`` by default, this abstains. An abstention is a result.
    """
    try:
        return Lingua().read(text).primary
    except Exception:  # noqa: BLE001
        return "und"


# --------------------------------------------------------------------------- #
# The faculty
# --------------------------------------------------------------------------- #

class Lingua:
    """NYX's tongue: tokenise anything, tag what it is, and remember what she has met.

    Stateless per call except for counters — what scripts and languages she has actually seen,
    which is what ``/nyx lingua`` reports and what the ``nyx.json`` sidecar carries forward.
    """

    def __init__(self, *, max_tokens: int = 512, min_concept_len: int = 2,
                 transliterate_bridge: bool = True, use_nlp: bool = True) -> None:
        self.max_tokens = max(8, int(max_tokens))
        self.min_concept_len = max(1, int(min_concept_len))
        self.transliterate_bridge = bool(transliterate_bridge)
        self.use_nlp = bool(use_nlp)

        self.reads = 0
        self.code_mixed_reads = 0
        self.scripts_seen: Dict[str, int] = {}
        self.languages_seen: Dict[str, int] = {}

    # ---- the one method everything else uses ----------------------------- #
    def read(self, text: str, *, keep_punct: bool = False) -> LinguaRead:
        """Full analysis of one turn: tokens, scripts, languages, register, concepts."""
        out = LinguaRead(text=str(text or ""))
        try:
            toks = tokenize(out.text, max_tokens=self.max_tokens, keep_punct=keep_punct)
            out.tokens = toks
            words = [t for t in toks if t.kind == _KIND_WORD]
            if not words:
                out.register = register_of(out.text, toks)
                self.reads += 1
                return out

            for t in words:
                out.scripts[t.script] = out.scripts.get(t.script, 0) + 1

            # Hinglish is Hindi *plus* Latin content in the same turn. Pure Devanagari is
            # Hindi, and saying otherwise would be a nicer-sounding wrong answer.
            hindi = _resolve_function(toks)
            out.hinglish = hindi and any(t.script == "latin" and not t.function for t in words)
            self._resolve_languages(out, words)
            out.register = register_of(out.text, toks)
            out.concepts = [t.lower for t in self._content(words)][: 32]
            if self.transliterate_bridge:
                out.keys = {c: roman_key(c) for c in out.concepts}

            self.reads += 1
            if out.code_mixed:
                self.code_mixed_reads += 1
            for script, count in out.scripts.items():
                self.scripts_seen[script] = self.scripts_seen.get(script, 0) + count
            for lang in out.languages:
                self.languages_seen[lang] = self.languages_seen.get(lang, 0) + 1
            return out
        except Exception:  # noqa: BLE001 — reading never breaks a turn
            return out

    def _content(self, words: List[Token]) -> List[Token]:
        out: List[Token] = []
        seen = set()
        for t in words:
            if t.function or t.lower in seen or t.lower.isdigit():
                continue
            limit = self.min_concept_len if t.script == "latin" else 1
            if len(t.lower) < limit:
                continue
            seen.add(t.lower)
            out.append(t)
        return out

    def _resolve_languages(self, out: LinguaRead, words: List[Token]) -> None:
        """Fix the language of every token, and say how much of it was actually evidence.

        Tokens with no evidence inherit the dominant *evidenced* language of the turn — and
        keep ``evidence=False``, so a caller can always tell an inference from a reading.
        """
        counts: Dict[str, int] = {}
        for t in words:
            if t.evidence:
                counts[t.lang] = counts.get(t.lang, 0) + 1

        total_ev = sum(counts.values())
        if total_ev:
            dominant = max(counts.items(), key=lambda p: (p[1], p[0]))[0]
        else:
            # Nothing proved anything. Ask the ASCII detector as a last resort, and if that is
            # only guessing too, abstain rather than inventing a language.
            dominant = self._nlp_guess(out.text)

        for t in words:
            if not t.evidence:
                t.lang = dominant

        if total_ev:
            out.languages = {lang: count / float(total_ev) for lang, count in counts.items()}
            # A language is a real participant when it brought two evidence tokens, or a
            # quarter of them. One shared function word is not a second language.
            live = {lang for lang, count in counts.items()
                    if count >= 2 or (count / float(total_ev)) >= 0.25}
            # Two *proved* languages, or two scripts carrying words. Romanised Hinglish is one
            # script and — with no English lexicon aboard — only one provable language, so it
            # is reported through ``hinglish`` rather than smuggled in here as proof.
            content_scripts = {s for s, c in out.scripts.items() if c >= 1 and s != "digit"}
            out.code_mixed = len(live) > 1 or len(content_scripts) > 1
        else:
            out.languages = {dominant: 1.0} if dominant != "und" else {}
        out.primary = dominant

    def _nlp_guess(self, text: str) -> str:
        """Fall back to the repo's ASCII stop-word detector — and treat its default as a guess."""
        if not self.use_nlp:
            return "und"
        try:
            from nyxara.senses.nlp import detect_language as _ascii_detect
            got = str(_ascii_detect(text) or "").strip()
            # It answers "en" both when it is sure and when it has nothing; "unknown" is its
            # empty-input case. Neither is evidence of English, so neither is reported as such.
            return got if got and got not in ("unknown", "en") else "und"
        except Exception:  # noqa: BLE001 — the fallback is optional, never required
            return "und"

    # ---- convenience ------------------------------------------------------ #
    def tokenize(self, text: str, *, keep_punct: bool = False) -> List[Token]:
        return tokenize(text, max_tokens=self.max_tokens, keep_punct=keep_punct)

    def concepts(self, text: str, *, cap: int = 32) -> List[str]:
        """Concept labels for the graph — the same contract as V.01's ``concepts_in``."""
        return content_tokens(text, cap=cap, min_len=self.min_concept_len)

    def key(self, word: str) -> str:
        """The spelling-tolerant key for one word — the transliteration bridge."""
        return roman_key(word)

    def same_word(self, a: str, b: str) -> bool:
        """Are these two spellings — possibly in different scripts — the same word?

        This is orthographic identity only. "gravity" and "गुरुत्वाकर्षण" are *not* the same
        word by this test; that is :mod:`nyxara.nyx.semantics`'s relational rung to learn.
        """
        try:
            ka, kb = roman_key(a), roman_key(b)
            return bool(ka) and ka == kb
        except Exception:  # noqa: BLE001
            return False

    # ---- introspection ---------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        try:
            return {
                "reads": self.reads,
                "code_mixed_reads": self.code_mixed_reads,
                "scripts": dict(sorted(self.scripts_seen.items(), key=lambda p: -p[1])),
                "languages": dict(sorted(self.languages_seen.items(), key=lambda p: -p[1])),
                "function_words": {"en": len(EN_FUNCTION), "hi-Latn": len(HI_LATN_FUNCTION),
                                   "hi": len(DEVA_FUNCTION)},
                "transliteration": self.transliterate_bridge,
            }
        except Exception:  # noqa: BLE001
            return {}

    # ---- persistence ------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {"reads": self.reads, "code_mixed_reads": self.code_mixed_reads,
                "scripts_seen": dict(self.scripts_seen),
                "languages_seen": dict(self.languages_seen)}

    def load_dict(self, data: Any) -> bool:
        try:
            if not isinstance(data, dict):
                return False
            self.reads = int(data.get("reads", 0))
            self.code_mixed_reads = int(data.get("code_mixed_reads", 0))
            self.scripts_seen = {str(k): int(v)
                                 for k, v in (data.get("scripts_seen") or {}).items()}
            self.languages_seen = {str(k): int(v)
                                   for k, v in (data.get("languages_seen") or {}).items()}
            return True
        except Exception:  # noqa: BLE001 — a corrupt sidecar must never block boot
            return False


# --------------------------------------------------------------------------- #
# The concept extractor every caller outside this module should use
# --------------------------------------------------------------------------- #
_ASCII_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-']*")

# Words that carry no conceptual weight. They would otherwise become the hubs of anything built
# on top of this, since they co-occur with everything. Deliberately small: a stop-list, not an
# ontology. Only reached when the full tokenizer above fails.
_ASCII_STOP = frozenset("""
a an the and or but if then than that this these those there here of to in on at by for with
from as is am are was were be been being do does did done have has had having i you he she it
we they me him her us them my your his its our their not no yes so such very just only also
what which who whom whose when where why how can could shall should will would may might must
""".split())


def concepts_in(text: str, *, cap: int = 32) -> List[str]:
    """Candidate concept labels from ``text`` — the script-aware extractor with a floor under it.

    Moved here from the deleted concept-graph module, which is where it always belonged: this is
    tokenization, and this file is the tongue. :func:`content_tokens` above does the real work
    across Devanagari, Cyrillic, Han and the rest; the ASCII path below is the fallback, so a
    failure in the tokenizer degrades to lexical extraction rather than to an empty turn.
    """
    try:
        got = content_tokens(text, cap=cap)
        if got:
            return got
    except Exception:  # noqa: BLE001 — the full tongue is a capability, never a dependency
        pass
    return _ascii_concepts_in(text, cap=cap)


def _ascii_concepts_in(text: str, *, cap: int = 32) -> List[str]:
    """The ASCII-only floor. Kept so extraction never returns nothing because of a tokenizer bug."""
    try:
        out: List[str] = []
        seen = set()
        for tok in _ASCII_TOKEN_RE.findall((text or "").lower()):
            if len(tok) < 2 or tok in _ASCII_STOP or tok in seen:
                continue
            # A bare number is a measurement, not a concept. Without this, ambient telemetry
            # ("host mem rose to 0.0906") mints a fresh concept for every decimal it ever sees.
            if tok.isdigit():
                continue
            seen.add(tok)
            out.append(tok)
            if len(out) >= cap:
                break
        return out
    except Exception:  # noqa: BLE001 — extraction never breaks a turn
        return []
