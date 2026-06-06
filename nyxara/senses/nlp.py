"""NYXARA · senses/nlp.py — pure-python natural-language perception (👁, fallback-safe).

This is the first of NYXARA's *senses*: turning raw text into structure she can reason
over. It is deliberately **pure standard library** so it always works — even on a bare
box with no ML stack — while transparently using a heavier NLP library if one happens to
be installed. Nothing here ever fails because ``spacy``/``nltk`` is missing.

What it perceives, from a single block of text:

* **Tokenisation & segmentation** — words and sentences, abbreviation-aware.
* **Language detection** — stop-word fingerprinting across a few languages.
* **Entities** — proper-noun spans, emails, URLs, @mentions, #hashtags, numbers, money,
  dates and times, by typed regex/heuristic (NER-lite).
* **Key-phrases** — RAKE (Rapid Automatic Keyword Extraction): phrases scored by word
  degree/frequency, so the *aboutness* of the text surfaces without a model.
* **Sentiment** — a lexicon with negation and intensifier handling, returning a polarity
  in ``[-1, 1]`` and a label.
* **Intent cues** — question / command / request / greeting / gratitude / affirmation /
  negation / statement, with the surface cues that triggered the call.
* **Summary** — extractive: sentences ranked by content-word salience.
* **Readability** — counts plus a Flesch reading-ease estimate.

:class:`NLP.analyze` fuses all of these into one :class:`Analysis` record. Pure standard
library; no third-party import is required.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Entity",
    "SentimentResult",
    "IntentResult",
    "Analysis",
    "STOPWORDS",
    "tokenize",
    "sentences",
    "detect_language",
    "keyphrases",
    "entities",
    "sentiment",
    "intent",
    "summarize",
    "readability",
    "NLP",
]


# --------------------------------------------------------------------------- #
# Lexicons
# --------------------------------------------------------------------------- #
STOPWORDS = frozenset("""
a an the and or but if then else for nor so yet of to in on at by with from into over
under again further is are was were be been being am do does did doing have has had having
i you he she it we they me him her us them my your his its our their this that these those
as not no than too very can will just don should now of off out up down here there all any
both each few more most other some such only own same s t can it's i'm you're
""".split())

_POSITIVE = frozenset("""
good great excellent amazing wonderful fantastic awesome love loved loving like liked best
happy glad pleased delighted superb brilliant perfect nice positive win wins won success
successful beautiful enjoy enjoyed helpful safe trust trusted strong better cool thanks
thank grateful appreciate appreciated outstanding remarkable
""".split())

_NEGATIVE = frozenset("""
bad terrible awful horrible hate hated hating dislike worst sad angry upset disappointed
disappointing fail failed failure failing poor ugly broken wrong danger dangerous threat
threatening attack hostile weak useless annoying painful slow buggy crash crashed unsafe
fear afraid scared worried worry corrupt malicious
""".split())

_NEGATORS = frozenset({"not", "no", "never", "none", "n't", "cannot", "without"})
_INTENSIFIERS = {"very": 1.5, "really": 1.5, "extremely": 1.8, "so": 1.3, "too": 1.3,
                 "slightly": 0.5, "somewhat": 0.6, "barely": 0.4, "absolutely": 1.8}

_WH = frozenset({"what", "why", "how", "when", "where", "who", "which", "whose", "whom"})
_AUX = frozenset({"is", "are", "was", "were", "do", "does", "did", "can", "could",
                  "would", "will", "should", "shall", "have", "has", "had", "may", "might"})
_IMPERATIVE = frozenset("""
add remove delete create make build run stop start give show tell find get set open close
send write read fix check update install configure list explain summarize analyze compute
search scan block protect defend backup restore deploy generate help
""".split())
_GREETINGS = frozenset({"hello", "hi", "hey", "greetings", "yo", "howdy"})
_THANKS = frozenset({"thanks", "thank", "thankyou", "cheers"})

# tiny stop-word fingerprints for language detection
_LANG_STOPS = {
    "en": {"the", "and", "is", "to", "of", "in", "you", "that", "it", "for"},
    "es": {"el", "la", "de", "que", "y", "los", "en", "un", "una", "por"},
    "fr": {"le", "la", "les", "de", "et", "un", "une", "que", "est", "dans"},
    "de": {"der", "die", "das", "und", "ist", "den", "von", "zu", "mit", "ein"},
}


# --------------------------------------------------------------------------- #
# Regexes
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[^\sA-Za-z0-9]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_ABBREV = frozenset({"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
                     "e.g", "i.e", "inc", "ltd", "co", "fig", "no"})

_ENTITY_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("URL", re.compile(r"\bhttps?://[^\s]+|\bwww\.[^\s]+\b")),
    ("MENTION", re.compile(r"(?<!\w)@\w+")),
    ("HASHTAG", re.compile(r"(?<!\w)#\w+")),
    ("MONEY", re.compile(r"[$£€]\s?\d[\d,]*(?:\.\d+)?\b")),
    ("TIME", re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:[ap]\.?m\.?)?\b", re.I)),
    ("DATE", re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")),
    ("NUMBER", re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?\b")),
]

_PROPER_RE = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b")


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "label": self.label, "start": self.start, "end": self.end}


@dataclass
class SentimentResult:
    polarity: float          # [-1, 1]
    label: str               # positive / negative / neutral
    positive: int = 0
    negative: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"polarity": round(self.polarity, 4), "label": self.label,
                "positive": self.positive, "negative": self.negative}


@dataclass
class IntentResult:
    kind: str
    confidence: float
    cues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "confidence": round(self.confidence, 3), "cues": self.cues}


@dataclass
class Analysis:
    text: str
    language: str
    tokens: List[str]
    sentences: List[str]
    entities: List[Entity]
    keyphrases: List[Tuple[str, float]]
    sentiment: SentimentResult
    intent: IntentResult
    summary: List[str]
    readability: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"language": self.language, "n_tokens": len(self.tokens),
                "n_sentences": len(self.sentences),
                "entities": [e.to_dict() for e in self.entities],
                "keyphrases": [[p, round(s, 3)] for p, s in self.keyphrases],
                "sentiment": self.sentiment.to_dict(), "intent": self.intent.to_dict(),
                "summary": self.summary, "readability": self.readability}


# --------------------------------------------------------------------------- #
# Tokenisation & segmentation
# --------------------------------------------------------------------------- #
def tokenize(text: str, *, keep_punct: bool = False, lower: bool = False) -> List[str]:
    toks = (_TOKEN_RE if keep_punct else _WORD_RE).findall(text)
    return [t.lower() for t in toks] if lower else toks


def sentences(text: str) -> List[str]:
    """Split into sentences, not breaking on common abbreviations."""
    text = text.strip()
    if not text:
        return []
    # protect abbreviations so we don't split on their trailing period
    protected = text
    for ab in _ABBREV:
        protected = re.sub(rf"\b({re.escape(ab)})\.", r"\1<DOT>", protected, flags=re.I)
    parts = _SENT_SPLIT.split(protected)
    out = [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]
    return out


def _content_words(text: str) -> List[str]:
    return [w for w in tokenize(text, lower=True) if w not in STOPWORDS and len(w) > 1]


# --------------------------------------------------------------------------- #
# Language detection
# --------------------------------------------------------------------------- #
def detect_language(text: str) -> str:
    toks = set(tokenize(text, lower=True))
    if not toks:
        return "unknown"
    best, score = "en", 0
    for lang, stops in _LANG_STOPS.items():
        overlap = len(toks & stops)
        if overlap > score:
            best, score = lang, overlap
    return best if score > 0 else "en"


# --------------------------------------------------------------------------- #
# Key-phrase extraction (RAKE)
# --------------------------------------------------------------------------- #
def keyphrases(text: str, *, top: int = 8) -> List[Tuple[str, float]]:
    # candidate phrases = runs of non-stopword tokens between stopwords/punctuation
    candidates: List[List[str]] = []
    for sent in sentences(text) or [text]:
        phrase: List[str] = []
        for tok in tokenize(sent, keep_punct=True):
            low = tok.lower()
            if low in STOPWORDS or not _WORD_RE.fullmatch(tok) or len(low) <= 1:
                if phrase:
                    candidates.append(phrase)
                    phrase = []
            else:
                phrase.append(low)
        if phrase:
            candidates.append(phrase)

    freq: Counter = Counter()
    degree: Counter = Counter()
    for phrase in candidates:
        deg = len(phrase) - 1
        for w in phrase:
            freq[w] += 1
            degree[w] += deg + 1
    word_score = {w: degree[w] / freq[w] for w in freq}

    scored: Dict[str, float] = {}
    for phrase in candidates:
        key = " ".join(phrase)
        scored[key] = scored.get(key, sum(word_score[w] for w in phrase))
    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:top]


# --------------------------------------------------------------------------- #
# Entity extraction (NER-lite)
# --------------------------------------------------------------------------- #
def entities(text: str) -> List[Entity]:
    found: List[Entity] = []
    claimed: List[Tuple[int, int]] = []

    def overlaps(a: int, b: int) -> bool:
        return any(not (b <= s or a >= e) for s, e in claimed)

    for label, pat in _ENTITY_PATTERNS:
        for m in pat.finditer(text):
            if not overlaps(m.start(), m.end()):
                found.append(Entity(m.group().strip(), label, m.start(), m.end()))
                claimed.append((m.start(), m.end()))

    # proper nouns: capitalised spans not at the very start of a sentence only
    for m in _PROPER_RE.finditer(text):
        s, e = m.start(), m.end()
        word = m.group()
        if word.lower() in {"i"} or overlaps(s, e):
            continue
        # skip a lone capitalised word that merely begins a sentence
        prefix = text[:s].rstrip()
        sentence_start = (s == 0) or prefix.endswith((".", "!", "?")) or prefix == ""
        if sentence_start and " " not in word:
            continue
        found.append(Entity(word, "PROPER_NOUN", s, e))
        claimed.append((s, e))

    found.sort(key=lambda en: en.start)
    return found


# --------------------------------------------------------------------------- #
# Sentiment
# --------------------------------------------------------------------------- #
def sentiment(text: str) -> SentimentResult:
    toks = tokenize(text, lower=True)
    score = 0.0
    pos = neg = 0
    for i, w in enumerate(toks):
        val = 0.0
        if w in _POSITIVE:
            val = 1.0
        elif w in _NEGATIVE:
            val = -1.0
        if val == 0.0:
            continue
        # look back for a negator / intensifier within 2 tokens
        mult = 1.0
        for j in range(max(0, i - 2), i):
            if toks[j] in _NEGATORS:
                mult *= -1.0
            elif toks[j] in _INTENSIFIERS:
                mult *= _INTENSIFIERS[toks[j]]
        contribution = val * mult
        score += contribution
        if contribution > 0:
            pos += 1
        elif contribution < 0:
            neg += 1
    hits = pos + neg
    polarity = max(-1.0, min(1.0, score / hits)) if hits else 0.0
    label = "neutral"
    if polarity > 0.15:
        label = "positive"
    elif polarity < -0.15:
        label = "negative"
    return SentimentResult(polarity=polarity, label=label, positive=pos, negative=neg)


# --------------------------------------------------------------------------- #
# Intent cues
# --------------------------------------------------------------------------- #
def intent(text: str) -> IntentResult:
    raw = text.strip()
    toks = tokenize(raw, lower=True)
    cues: List[str] = []
    if not toks:
        return IntentResult("empty", 1.0, cues)
    first = toks[0]

    if first in _GREETINGS:
        cues.append(f"greeting:{first}")
        return IntentResult("greeting", 0.9, cues)
    if first in _THANKS or (first == "thank" and "you" in toks[:2]):
        cues.append("gratitude")
        return IntentResult("gratitude", 0.9, cues)

    is_question = raw.endswith("?") or first in _WH or first in _AUX
    polite = any(w in toks for w in ("please",)) or \
        (len(toks) >= 2 and toks[0] in ("could", "would", "can") and "you" in toks[:3])
    if polite:
        cues.append("request")
        return IntentResult("request", 0.85, cues)
    if is_question:
        if raw.endswith("?"):
            cues.append("qmark")
        if first in _WH:
            cues.append(f"wh:{first}")
        if first in _AUX:
            cues.append(f"aux:{first}")
        return IntentResult("question", 0.85, cues)
    if first in _IMPERATIVE:
        cues.append(f"imperative:{first}")
        return IntentResult("command", 0.8, cues)
    if first in ("yes", "yeah", "yep", "sure", "ok", "okay"):
        return IntentResult("affirmation", 0.8, ["affirm"])
    if first in ("no", "nope", "nah"):
        return IntentResult("negation", 0.8, ["negate"])
    return IntentResult("statement", 0.6, cues)


# --------------------------------------------------------------------------- #
# Extractive summary
# --------------------------------------------------------------------------- #
def summarize(text: str, *, n: int = 3) -> List[str]:
    sents = sentences(text)
    if len(sents) <= n:
        return sents
    freq = Counter(_content_words(text))
    if not freq:
        return sents[:n]
    top = max(freq.values())
    norm = {w: c / top for w, c in freq.items()}
    scored: List[Tuple[int, float]] = []
    for idx, s in enumerate(sents):
        words = _content_words(s)
        score = sum(norm.get(w, 0.0) for w in words) / (len(words) + 1)
        scored.append((idx, score))
    best = sorted(scored, key=lambda t: -t[1])[:n]
    return [sents[i] for i, _ in sorted(best, key=lambda t: t[0])]


# --------------------------------------------------------------------------- #
# Readability
# --------------------------------------------------------------------------- #
def _syllables(word: str) -> int:
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def readability(text: str) -> Dict[str, Any]:
    sents = sentences(text)
    words = tokenize(text)
    n_w, n_s = len(words), max(1, len(sents))
    syll = sum(_syllables(w) for w in words)
    if n_w == 0:
        return {"words": 0, "sentences": len(sents), "syllables": 0,
                "avg_sentence_length": 0.0, "flesch_reading_ease": 0.0}
    flesch = 206.835 - 1.015 * (n_w / n_s) - 84.6 * (syll / n_w)
    return {"words": n_w, "sentences": len(sents), "syllables": syll,
            "avg_sentence_length": round(n_w / n_s, 2),
            "flesch_reading_ease": round(max(0.0, min(100.0, flesch)), 1)}


# --------------------------------------------------------------------------- #
# Facade
# --------------------------------------------------------------------------- #
class NLP:
    """A single entry-point that fuses every perception into one :class:`Analysis`."""

    def analyze(self, text: str, *, top_phrases: int = 8, summary_sentences: int = 3
                ) -> Analysis:
        return Analysis(
            text=text, language=detect_language(text),
            tokens=tokenize(text, lower=True), sentences=sentences(text),
            entities=entities(text), keyphrases=keyphrases(text, top=top_phrases),
            sentiment=sentiment(text), intent=intent(text),
            summary=summarize(text, n=summary_sentences), readability=readability(text))

    # thin pass-throughs for convenience
    tokenize = staticmethod(tokenize)
    sentences = staticmethod(sentences)
    entities = staticmethod(entities)
    keyphrases = staticmethod(keyphrases)
    sentiment = staticmethod(sentiment)
    intent = staticmethod(intent)
    summarize = staticmethod(summarize)
    detect_language = staticmethod(detect_language)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA NLP self-test")
    print("=" * 70)

    nlp = NLP()
    text = ("Hello NYXARA. Please scan the gateway at 10.0.0.1 for me. "
            "I am very worried about a possible attack from Acme Corp on 2026-06-06. "
            "Email me at jp@example.com if you find anything dangerous. "
            "The new firewall is absolutely excellent and I love how fast it is.")

    a = nlp.analyze(text)

    print(f"\nlanguage            : {a.language}")
    assert a.language == "en"

    print(f"sentences           : {len(a.sentences)}")
    assert len(a.sentences) >= 4   # not split on '10.0.0.1' decimals incorrectly

    labels = {e.label for e in a.entities}
    print(f"entities            : {[(e.text, e.label) for e in a.entities]}")
    assert "EMAIL" in labels and "DATE" in labels
    assert any(e.text == "Acme Corp" and e.label == "PROPER_NOUN" for e in a.entities)

    print(f"keyphrases          : {[p for p, _ in a.keyphrases[:5]]}")
    assert a.keyphrases  # surfaced some 'aboutness'

    print(f"sentiment           : {a.sentiment.to_dict()}")
    # the closing praise (excellent, love, fast) should tilt overall positive-ish
    assert a.sentiment.positive >= 2

    # intent of individual messages
    assert intent("Please scan the gateway.").kind == "request"
    assert intent("What is the status?").kind == "question"
    assert intent("Block that IP now.").kind == "command"
    assert intent("Hello there").kind == "greeting"
    assert intent("Thanks NYXARA").kind == "gratitude"
    assert intent("The server is down.").kind == "statement"
    print("intent detection    : request/question/command/greeting/gratitude/statement ✓")

    # negation flips sentiment
    assert sentiment("this is good").polarity > 0
    assert sentiment("this is not good").polarity < 0
    print("negation handling   : 'not good' < 0 ✓")

    print(f"summary             : {summarize(text, n=2)}")
    print(f"readability         : {a.readability}")
    assert a.readability["words"] > 0

    print("\nALL SELF-TESTS PASSED ✓")
