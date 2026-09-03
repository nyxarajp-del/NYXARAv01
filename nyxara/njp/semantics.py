"""NYXARA · njp/semantics.py — the semantic compiler (⚗, NJP V.09).

**The wall this exists to break.** :mod:`nyxara.njp.grounding` reads a sentence with 101 regular
expressions over *words*: 51 predicate aliases, 71 event verbs, 36 measure verbs. Every one of
them is a statement about which sentences may be said to her, and two measurements say what that
costs:

* ``"Zibbies eat crystals."`` → **no triple at all**. ``eat`` is not in the lexicon, so the fact
  was never stored; the following question then answered from a neighbouring ``is_a`` edge and
  returned a confident wrong answer.
* ``"Zorbins don't need glarn."`` → ``("Zorbins don't", requires, glarn)``. The string ``negat``
  did not occur anywhere in ``grounding.py``, so negation was not mis-handled — it was
  *invisible*, and the denial of a claim was stored as a different entity's **assertion** of it.
  That is worse than a gap: it is the extractor manufacturing the opposite of what was said.

Measured on :mod:`nyxara.eval.adversarial` before this module existed: ``open_verb`` concept
**0.00** over six ordinary transitive verbs, ``surface`` **0.29** over seven ordinary question
forms, ``polar`` **0.33**.

**Why the answer is not more regexes.** Adding ``eat`` fixes ``eat``. The set of English verbs is
open and the set of ways to ask a question is large, so a word-level pattern list is a race that
cannot be won — and each new pattern is a new chance for the wrong one to match first, which is
how ``"Zorbins don't"`` became an entity in the first place.

**What is closed is the other half of the sentence.** Determiners, auxiliaries, negators, modals,
prepositions, wh-words, subordinators and pronouns are a genuinely finite set in any language —
a few hundred words, learned once, changing over centuries. Content words are the open half. So
this module inverts the usual arrangement:

    tag the **closed** class exhaustively · treat everything else as open material · match
    frames over the **tag sequence**, never over the words

Ten tags, and a frame like ``WH AUX NP VERB`` covers *every* transitive English verb that will
ever exist, because ``VERB`` is a position, not a list. That is the whole architectural claim,
and the ``open_verb`` family is the measurement that decides whether it is true.

**The pipeline**, as the Master specified it::

    surface  →  tokenise + tag  →  clause split  →  frame match  →  slot fill
             →  negation · modality · condition · temporal · evidential
             →  typed Meaning

**Locating the verb without a verb list.** The one genuinely hard step. In ``NP VERB NP`` no tag
marks the verb, so instead of guessing the first plausible token, every candidate position is
scored on independent evidence — does an auxiliary precede it, does it carry third-person ``-s``,
does what follows begin a noun phrase, is it in the middle rather than at an edge — and the best
scoring candidate wins. **Every candidate is tried; the winner is chosen on evidence, never by
pattern order.** That is the same selection rule :mod:`nyxara.njp.compile` documents, and for the
same reason: ``or`` short-circuits, and a short-circuit is how a sentence gets read by the pattern
written for a different sentence.

**What this module does not do.** It does not decide truth, relevance, or what to answer; it
produces a :class:`Meaning` and stops. It carries no dictionary of English or Hindi content words,
so it cannot tell a noun from a verb by *knowing* — only by position and morphology, which is
right often and not always. Where the evidence does not separate two readings it reports
:attr:`Meaning.confidence` low rather than picking one, and where it can find no frame at all it
returns a Meaning with :attr:`Meaning.kind` ``unreadable`` — the honest empty answer, so the
existing extractor keeps the turn rather than this one guessing at it.

Pure standard library. No LLM. Every entry point is fail-soft: on any error it returns an
unreadable Meaning rather than breaking a turn.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Tag", "Token", "Meaning", "SemanticCompiler", "compile_meaning", "tag_tokens",
]


class Tag:
    """The closed tag alphabet. Ten members, and ``WORD`` is everything the other nine are not."""

    DET = "DET"        # a, an, the, ek
    WH = "WH"          # what, which, who, kya, kaun
    AUX = "AUX"        # do, does, is, are, hai, hain
    MODAL = "MODAL"    # can, may, must, sakta, shayad
    NEG = "NEG"        # not, n't, never, nahi, mat
    PREP = "PREP"      # of, for, to, ko, se, ka
    CONJ = "CONJ"      # and, or, aur, ya
    SUB = "SUB"        # if, when, because, agar, jab
    PRON = "PRON"      # i, you, it, mera
    ADV = "ADV"        # exactly, usually, always
    WORD = "WORD"      # the open class


#: The closed class, exhaustively. Adding a word here is cheap and rare — this is the half of
#: language that does not grow. English and romanised Hinglish share the table because a turn may
#: be in both at once (see :mod:`nyxara.njp.tongue`), and Devanagari forms sit beside their
#: transliterations so neither spelling is privileged.
_CLOSED: Dict[str, str] = {}


def _load(tag: str, words: Sequence[str]) -> None:
    for word in words:
        _CLOSED[word] = tag


_load(Tag.DET, ("a", "an", "the", "this", "that", "these", "those", "some", "any", "each",
                "every", "all", "both", "ek", "yeh", "woh", "ye", "vo", "इस", "यह", "वह", "एक"))
_load(Tag.WH, ("what", "which", "who", "whom", "whose", "where", "when", "why", "how",
               "kya", "kaun", "kaunsa", "kahan", "kab", "kyun", "kyu", "kaise", "kitna", "kitne",
               "क्या", "कौन", "कहाँ", "कब", "क्यों", "कैसे"))
#: The auxiliaries that carry tense and agreement for a following **bare** verb. The distinction
#: is grammatical, not stylistic: after ``does`` the verb is uninflected by rule ("does a zorbin
#: eat"), while after a copula there is no separate verb at all ("what is deep learning" is a
#: noun phrase, not a subject and a verb). Reading the two the same way is what made
#: ``"what is deep learning"`` compile as ``deep --learn--> ?``.
_DO_SUPPORT: Tuple[str, ...] = ("do", "does", "did", "karta", "karte", "karti")

_load(Tag.AUX, ("do", "does", "did", "is", "are", "was", "were", "be", "been", "being",
                "has", "have", "had", "am",
                # Hinglish past tense: "tha/thi/thay" only. The masculine plural is also spelled
                # "the", and it collides head-on with the English determiner — which is orders of
                # magnitude commoner in the turns this reads. Loaded here it silently retagged
                # every English "the" as an auxiliary, so "the wardens guard the vault" became a
                # clause with two auxiliaries and no noun phrase. The rare reading loses; a
                # collision between a closed-class word in two languages is resolved by frequency,
                # and saying so is better than a table that quietly reads one language wrong.
                "hai", "hain", "tha", "thi", "thay", "ho", "hota", "hote", "hoti",
                "karta", "karte", "karti", "है", "हैं", "था", "थे", "थी", "हो"))
_load(Tag.MODAL, ("can", "could", "may", "might", "must", "should", "would", "will", "shall",
                  "sakta", "sakte", "sakti", "shayad", "zaroor", "chahiye_modal",
                  "सकता", "सकते", "सकती", "शायद"))
_load(Tag.NEG, ("not", "n't", "never", "no", "none", "neither", "nor", "cannot",
                # Written without the apostrophe, which the contraction rule cannot reach because
                # there is nothing there to split on. Common enough in typed turns to matter.
                "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent", "cant", "wont",
                "nahi", "nahin", "na", "mat", "bilkul_nahi",
                "नहीं", "ना", "मत"))
_load(Tag.PREP, ("of", "for", "in", "on", "at", "to", "from", "with", "by", "about", "into",
                 "over", "under", "through", "between", "against", "without",
                 "ko", "ke", "ka", "ki", "se", "me", "mein", "par", "tak", "liye",
                 "को", "के", "का", "की", "से", "में", "पर"))
_load(Tag.CONJ, ("and", "or", "but", "aur", "ya", "lekin", "par_conj", "और", "या", "लेकिन"))
_load(Tag.SUB, ("if", "when", "unless", "because", "since", "while", "although", "though",
                "whenever", "agar", "jab", "kyunki", "kyonki", "toh", "warna",
                "अगर", "जब", "क्योंकि", "तो"))
_load(Tag.PRON, ("i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
                 "my", "your", "his", "its", "our", "their", "mine", "yours",
                 "main", "tum", "aap", "mera", "meri", "tera", "tumhara", "uska", "unka",
                 "मैं", "तुम", "आप", "मेरा", "उसका"))
_load(Tag.ADV, ("exactly", "really", "always", "usually", "sometimes", "often", "generally",
                "typically", "normally", "actually", "just", "only", "please", "also", "too",
                "hi", "bhi", "sirf", "hamesha", "kabhi", "aksar", "आमतौर", "हमेशा", "भी"))

#: Words that mark the surface as hedged, habitual or obligatory. Read off the closed class, so
#: this is a reading of the *modal*, not a lexicon of verbs.
_MODALITY: Dict[str, str] = {
    "can": "possible", "could": "possible", "may": "possible", "might": "possible",
    "shayad": "possible", "sakta": "possible", "sakte": "possible", "sakti": "possible",
    "must": "necessary", "should": "necessary", "shall": "necessary", "zaroor": "necessary",
    "usually": "typical", "typically": "typical", "generally": "typical", "often": "typical",
    "normally": "typical", "always": "typical", "aksar": "typical", "hamesha": "typical",
}

#: Evidential openers — who is vouching for the claim, and how hard. Matched at the *start* of a
#: sentence only, because "I think" mid-sentence is usually a different clause's subject.
_EVIDENTIAL: Tuple[Tuple[str, str, float], ...] = (
    ("i think", "hedged", 0.55),
    ("i believe", "hedged", 0.6),
    ("i guess", "hedged", 0.45),
    ("maybe", "hedged", 0.45),
    ("shayad", "hedged", 0.45),
    ("mujhe lagta hai", "hedged", 0.55),
    ("apparently", "reported", 0.5),
    ("reportedly", "reported", 0.5),
    ("i know", "asserted", 0.9),
    ("obviously", "asserted", 0.85),
)

#: Time markers, read as a closed class like everything else here.
_TEMPORAL: Tuple[str, ...] = (
    "yesterday", "today", "tomorrow", "now", "currently", "recently", "previously",
    "before", "after", "already", "still", "yet", "soon", "later",
    "kal", "aaj", "abhi", "pehle", "baad", "ab",
)

#: Hinglish predicate words that behave as the clause's verb in SOV order. These are *not* an
#: open verb list — they are the small set of light verbs Hinglish uses to carry a relation, and
#: the content sits in the nouns around them.
_HINGLISH_VERBS: Tuple[str, ...] = (
    "chahiye", "hota", "hoti", "hote", "karta", "karte", "karti", "khata", "khati", "khate",
    "banata", "banati", "banate", "rehta", "rehti", "rehte", "milta", "milti", "milte",
    "चाहिए", "होता", "करता", "खाता",
)

#: Speech-act verbs that take a wh-complement. Closed enough to list, and listing them is what
#: lets an imperative be recognised as the question it embeds rather than as a claim about the
#: imperative itself.
_ASK_VERBS: Tuple[str, ...] = (
    "tell", "say", "explain", "describe", "show", "list", "name", "define",
    "batao", "bata", "bolo", "batana", "बताओ",
)

#: The combining marks a word may carry. Python's ``\w`` is alphanumerics, and a Devanagari
#: vowel sign is neither: ``category('\u0941') == 'Mn'`` and ``'\u0941'.isalnum()`` is ``False``.
#: So the pattern below, written as ``[^\W_]+``, **shattered every Indic word at every matra** —
#: ``कुत्ता`` came back as ``['क', 'त', 'त']`` and ``मेरा`` as ``['म', 'र']``.
#:
#: That is not a small thing. This module ships 45 Devanagari closed-class words and this package
#: has claimed Hindi since V.09; almost none of those words can ever have matched, because almost
#: every Hindi word carries a mark. Only the ones built of bare consonants — ``यह``, ``जय`` —
#: survived, which is why the failure never showed up as an exception and never showed up in a
#: test: the reading degraded to ``unreadable`` and this package treats that as an honest answer.
#:
#: Found by measuring :class:`~nyxara.njp.discourse.ClosedClassLearner` on real Hindi sentences in
#: V.35, where the "closed class" it recovered was a list of bare consonants.
#:
#: The ranges cover the scripts this package names anywhere — Devanagari and its neighbours, plus
#: the general combining block and Arabic — rather than every mark in Unicode, because a list that
#: is checked is worth more than one that is exhaustive and untested.
_MARKS = (
    "\u0300-\u036f\u0483-\u0489\u0591-\u05bd\u0610-\u061a\u064b-\u065f\u0670"
    "\u06d6-\u06ed\u0900-\u0903\u093a-\u094f\u0951-\u0957\u0962-\u0963"
    "\u0981-\u0983\u09bc\u09be-\u09cd\u0a01-\u0a03\u0a3c-\u0a4d"
    "\u0a81-\u0a83\u0abc-\u0acd\u0b01-\u0b03\u0b3c-\u0b57\u0bbe-\u0bcd"
    "\u0c00-\u0c03\u0c3e-\u0c56\u0c81-\u0c83\u0cbc-\u0ccd\u0d00-\u0d03"
    "\u0d3b-\u0d4d\u0e31-\u0e3a\u0e47-\u0e4e\u200c\u200d"
)

#: One token is a run of word characters, letters and digits together, **with the combining marks
#: that belong to them**. Splitting the digits apart — which is what excluding ``\d`` did — cuts
#: ``h2o`` into three tokens and ``covid19`` into two, and an identifier cut into pieces is an
#: entity that cannot be looked up. Splitting the marks apart did the same thing to every word of
#: an Indic language. Decimals keep their point; nothing else does, so a full stop still ends a
#: sentence.
_WORD_RX = re.compile(rf"\d+\.\d+|[^\W_](?:[^\W_]|[{_MARKS}])*(?:['’][^\W_]+)*",
                      re.UNICODE)


def _fold(word: str) -> str:
    """Lower-cased, accent-stripped, with the English contraction split off as its own token."""
    return unicodedata.normalize("NFKC", word).strip().lower()


@dataclass
class Token:
    """One surface token and the closed-class tag it carries (``WORD`` if it carries none)."""

    text: str = ""
    tag: str = Tag.WORD
    index: int = 0

    @property
    def open_class(self) -> bool:
        return self.tag == Tag.WORD

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.text}/{self.tag}"


def tag_tokens(text: str) -> List[Token]:
    """Tokenise and tag against the closed class. Anything not in it is ``WORD``.

    Contractions are split before tagging so ``don't`` becomes ``do``/``n't`` — an ``AUX`` and a
    ``NEG``. That single step is what stops the negator from being absorbed into the subject noun
    phrase, which is the exact mechanism that produced ``("Zorbins don't", requires, glarn)``.
    """
    out: List[Token] = []
    try:
        low = _fold(text)
        # Split contractions into their parts. English writes the negator attached to the
        # auxiliary; every other layer here reads them as two tokens because they are two.
        low = re.sub(r"\b(do|does|did|is|are|was|were|has|have|had|ca|wo|sha|could|would|should)"
                     r"n['’]t\b", r"\1 n't", low)
        low = low.replace("cann't", "can n't").replace("ca n't", "can n't")
        # The copula written as a clitic. Left joined, "what's" is a single WORD token and every
        # wh-frame fails on a sentence that is unmistakably a question.
        low = re.sub(r"\b(what|who|where|when|why|how|that|it|he|she|there)['’]s\b",
                     r"\1 is", low)
        low = re.sub(r"\b(i|you|we|they)['’]re\b", r"\1 are", low)
        low = re.sub(r"\b(i)['’]m\b", r"\1 am", low)
        low = low.replace("wo n't", "will n't").replace("sha n't", "shall n't")
        for i, match in enumerate(_WORD_RX.finditer(low)):
            word = match.group(0)
            out.append(Token(text=word, tag=_CLOSED.get(word, Tag.WORD), index=i))
        # "n't" loses its apostrophe to the tokeniser; re-tag it rather than widening the regex,
        # which would let stray apostrophes elsewhere become negators.
        for token in out:
            if token.text in ("nt", "n t", "t") and token.index > 0:
                previous = out[token.index - 1]
                if previous.tag == Tag.AUX or previous.text in ("can", "will", "shall"):
                    token.tag = Tag.NEG
    except Exception:  # noqa: BLE001
        return []
    return out


@dataclass
class Meaning:
    """A sentence compiled into a typed cognitive representation.

    This is the interface between language and thought: everything above it reads these fields
    and never the surface. The three that did not exist before this module are :attr:`negated`,
    :attr:`focus` and :attr:`verb` — and the first of them is the one that was silently changing
    claims into their opposites.
    """

    #: ``assertion`` · ``question`` · ``polar_question`` · ``command`` · ``unreadable``
    kind: str = "unreadable"
    subject: str = ""
    #: The relation, lemmatised but **not** canonicalised — mapping ``need``→``requires`` is
    #: :mod:`nyxara.njp.grounding`'s job and it already has the alias table. Emitting a canonical
    #: name here would mean carrying a second, drifting copy of it.
    relation: str = ""
    object: str = ""
    #: The single field whose absence made this module necessary.
    negated: bool = False
    modality: str = ""            # "" | possible | typical | necessary
    condition: str = ""
    temporal: str = ""
    #: For questions: which slot the question is *about* — ``object``, ``subject``, or ``truth``
    #: for a polar. A question with no focus is one this compiler could not read.
    focus: str = ""
    #: The surface verb before lemmatisation, kept so a consumer can see what was actually said.
    verb: str = ""
    evidential: str = ""          # "" | hedged | reported | asserted
    confidence: float = 0.0
    frame: str = ""               # which frame matched, for diagnosis
    language: str = "en"
    tokens: List[Token] = field(default_factory=list)
    #: Arguments beyond the three this compiler can read, keyed by role name — ``recipient``,
    #: ``location``, ``instrument``, a modifier on a noun, the inside of a relative clause.
    #:
    #: Added for :mod:`nyxara.njp.language`, and empty everywhere else. Three named fields is the
    #: right shape for what the frames in *this* module can extract, and the wrong shape for a
    #: sentence with four arguments in it: *"she gave the book to Ravi"* has an agent, a theme and
    #: a recipient, and nothing here could hold the third without inventing a fourth attribute,
    #: then a fifth. A dict has no such ceiling and asserts no ontology — a role name is whatever
    #: a demonstration called it, exactly as a concept id is whatever the clustering found.
    #:
    #: ``subject``, ``relation`` and ``object`` stay where they are and stay authoritative. Every
    #: consumer that reads them is untouched, and a Meaning this compiler produced has an empty
    #: ``roles`` — so nothing downstream has to learn about this field to keep working.
    roles: Dict[str, str] = field(default_factory=dict)

    @property
    def readable(self) -> bool:
        return self.kind != "unreadable"

    @property
    def complete(self) -> bool:
        """Does this Meaning carry enough to become a triple?

        An **intransitive** assertion is complete without an object, and refusing it here is what
        starved the world model's timeline. "The fire spread", "it rained", "aag lagi" report that
        something happened and there is nothing further to name: demanding an object of them is
        demanding a part of speech the sentence does not have. Measured, the consequence reached
        four organs deep — `world.events` stood at 1 over a session full of happenings, so
        `world.counterfactual` answered "X has only happened 0 times, too few to say", so every
        designed experiment was unresolvable, so `experiments_run` and `bits_gained` were
        permanently zero and no hypothesis was ever eliminated.

        Only for the intransitive *frame*, never as a general relaxation. A transitive reading
        that lost its object is a parse failure and stays incomplete.
        """
        if self.frame == "np-verb":
            return bool(self.subject and self.relation)
        return bool(self.subject and self.relation and self.object)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "kind": self.kind, "subject": self.subject, "relation": self.relation,
            "object": self.object, "negated": self.negated,
            "confidence": round(self.confidence, 4), "frame": self.frame,
        }
        if self.roles:
            out["roles"] = dict(self.roles)
        for name in ("modality", "condition", "temporal", "focus", "verb", "evidential"):
            value = getattr(self, name)
            if value:
                out[name] = value
        return out


# --------------------------------------------------------------------------- #
# lemmatisation — morphological, dictionary-free
# --------------------------------------------------------------------------- #

_IRREGULAR: Dict[str, str] = {
    "is": "be", "are": "be", "was": "be", "were": "be", "am": "be", "been": "be",
    "has": "have", "had": "have", "does": "do", "did": "do",
    "needs": "need", "requires": "require", "eats": "eat", "makes": "make",
}


def _lemma(word: str) -> str:
    """Strip English inflection by shape. Wrong on irregulars this table does not name, and that
    is acceptable: a wrong lemma costs a canonicalisation miss, where a missing verb costs the
    whole fact."""
    word = _fold(word)
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith(("ses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    if len(word) > 4 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    return word


def _plural(word: str) -> bool:
    """Does this token carry the ``-s`` that English puts on both plural nouns and 3sg verbs?"""
    word = _fold(word)
    return len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is"))


# --------------------------------------------------------------------------- #
# the compiler
# --------------------------------------------------------------------------- #

class SemanticCompiler:
    """Surface sentence → :class:`Meaning`, by frames over tags rather than patterns over words.

    Stateless and cheap: constructing one per call is fine, and :func:`compile_meaning` does.
    """

    def compile(self, text: str, *, interrogative: Optional[bool] = None) -> Meaning:
        """Compile one sentence. Never raises — an unreadable surface returns an unreadable
        Meaning, which is the signal for the caller to keep whatever it had.

        ``interrogative`` lets a caller that already knows the act say so, instead of this module
        re-deriving it from punctuation that may no longer be there. That is not a convenience:
        :func:`nyxara.njp.grounding._clean` strips the question mark before
        :meth:`~nyxara.njp.grounding.Grounder._read_question` runs, so ``"zorbins need what"``
        arrived here with nothing left marking it as a question and compiled as an *assertion*
        that zorbins need something called "what". Four of seven surface forms failed on exactly
        that. The caller had the answer the whole time — ``ground()`` only reaches that path after
        ``_is_question`` has returned True — so it passes it rather than this module guessing.
        """
        try:
            return self._compile(text, interrogative=interrogative)
        except Exception:  # noqa: BLE001
            return Meaning()

    # ---- top level -------------------------------------------------------- #
    def _compile(self, text: str, *, interrogative: Optional[bool] = None) -> Meaning:
        raw = str(text or "").strip()
        if not raw:
            return Meaning()
        out = Meaning()
        body, out.evidential, out.confidence = self._strip_evidential(raw)
        body, out.condition = self._split_condition(body)
        tokens = tag_tokens(body)
        if not tokens:
            return Meaning()
        out.language = self._language(tokens)
        tokens, out.temporal = self._strip_temporal(tokens)
        tokens, out.modality = self._strip_modality(tokens)
        # Negation is lifted out before any frame runs, and this ordering is the fix: with the
        # negator still in the stream every frame sees an extra token in the middle of a noun
        # phrase and either fails or absorbs it into a slot.
        tokens, out.negated = self._strip_negation(tokens)
        if interrogative is None:
            interrogative = (raw.rstrip().endswith("?") or self._fronted(tokens)
                             or self._embedded_question(tokens))
        # "tell me …", "batao …" — a matrix speech act, removed after it has been used to decide
        # the act and before any frame reads slots. Left in, it becomes the subject: measured,
        # "batao ek zorbin ko kya chahiye?" compiled with subject "batao zorbin".
        tokens = self._strip_matrix(tokens)
        out.tokens = tokens
        if interrogative:
            self._question(tokens, out)
        else:
            self._assertion(tokens, out)
        if not out.frame:
            out.kind = "unreadable"
        return out

    # ---- pre-passes ------------------------------------------------------- #
    @staticmethod
    def _strip_evidential(text: str) -> Tuple[str, str, float]:
        """Remove a leading evidential and report what it said about the source's own certainty."""
        low = _fold(text)
        for opener, kind, confidence in _EVIDENTIAL:
            if low.startswith(opener + " ") or low.startswith(opener + ","):
                rest = text[len(opener):].lstrip(" ,")
                # "I think that X" — the complementiser is part of the frame, not the claim.
                if _fold(rest).startswith("that "):
                    rest = rest[5:]
                return rest, kind, confidence
        return text, "", 0.0

    @staticmethod
    def _split_condition(text: str) -> Tuple[str, str]:
        """Separate a subordinate clause from the claim it qualifies.

        Both orders occur — ``if A, B`` and ``B if A`` — and the condition is the subordinate
        half in each. A claim answered without its condition is how a true statement becomes a
        false one, which is why :class:`~nyxara.njp.grounding.GroundedTriple` has carried the
        field since before this module existed; what was missing was anything filling it from an
        arbitrary sentence.
        """
        low = _fold(text)
        for marker in ("if ", "agar ", "when ", "jab ", "unless ", "whenever "):
            if low.startswith(marker):
                rest = text[len(marker):]
                # "if A then B" / "agar A to B" / "if A, B"
                split = re.split(r",|\bthen\b|\bto\b(?!\w)|\btoh\b", rest, maxsplit=1)
                if len(split) == 2 and split[1].strip():
                    return split[1].strip(), split[0].strip()
                return rest, ""
            index = low.find(" " + marker)
            if index > 0:
                return text[:index].strip(), text[index + 1 + len(marker):].strip()
        return text, ""

    @staticmethod
    def _strip_temporal(tokens: List[Token]) -> Tuple[List[Token], str]:
        found = [t.text for t in tokens if t.text in _TEMPORAL]
        if not found:
            return tokens, ""
        kept = [t for t in tokens if t.text not in _TEMPORAL]
        return _reindex(kept), " ".join(found)

    @staticmethod
    def _strip_modality(tokens: List[Token]) -> Tuple[List[Token], str]:
        modality = ""
        kept: List[Token] = []
        for token in tokens:
            reading = _MODALITY.get(token.text, "")
            # An ADV that carries modality ("usually") is removed with it; a MODAL always is.
            if reading and token.tag in (Tag.MODAL, Tag.ADV):
                modality = modality or reading
                continue
            kept.append(token)
        return (_reindex(kept), modality) if modality else (tokens, "")

    @staticmethod
    def _strip_negation(tokens: List[Token]) -> Tuple[List[Token], bool]:
        """Lift every negator out of the stream and report that the clause was negated.

        Clause-level rather than scoped, and the docstring says so because the difference is
        real: "X does not need Y" and "X needs no Y" both come back negated here, which is right,
        while "not every X needs Y" also does, which is a quantifier scope this compiler cannot
        represent. Reporting the coarse reading is what the whole module is for — the alternative
        in place before it was reporting the *opposite* one.
        """
        negators = [t for t in tokens if t.tag == Tag.NEG]
        if not negators:
            return tokens, False
        # A bare "no" heading the sentence is a reply, not a negated claim about anything.
        if len(tokens) == 1:
            return tokens, True
        kept = [t for t in tokens if t.tag != Tag.NEG]
        return _reindex(kept), True

    @staticmethod
    def _language(tokens: Sequence[Token]) -> str:
        """Which register this turn is in, from closed-class membership only — see
        :mod:`nyxara.njp.tongue`, whose rule this follows: a content word in Latin script carries
        no evidence of its language and is never used as any."""
        hinglish = sum(1 for t in tokens
                       if t.text in ("ko", "ka", "ki", "ke", "se", "hai", "hain", "kya", "nahi",
                                     "chahiye", "mein", "kaun", "kaise")
                       or unicodedata.name(t.text[0], "").startswith("DEVANAGARI"))
        return "hi" if hinglish >= 2 else "en"

    @staticmethod
    def _strip_matrix(tokens: List[Token]) -> List[Token]:
        """Drop a leading speech-act verb and the pronoun it addresses."""
        if not tokens or tokens[0].text not in _ASK_VERBS:
            return tokens
        cut = 1
        if len(tokens) > 1 and tokens[1].tag == Tag.PRON:
            cut = 2
        remainder = tokens[cut:]
        return _reindex(remainder) if remainder else tokens

    @staticmethod
    def _embedded_question(tokens: Sequence[Token]) -> bool:
        """``tell me what a X requires.`` — a question wearing a full stop.

        Punctuation is not the test and never was: the act is fixed by the speech-act verb taking
        a wh-complement. Reading this as an assertion is how "tell me what a zorbin requires"
        became a *claim* that something called "tell me" requires something.
        """
        if not tokens or tokens[0].text not in _ASK_VERBS:
            return False
        return any(t.tag == Tag.WH for t in tokens[1:])

    @staticmethod
    def _fronted(tokens: Sequence[Token]) -> bool:
        """Is this a question by structure rather than by punctuation? ``what does X eat`` with no
        question mark is still a question, and the Master's turns frequently lack the mark."""
        if not tokens:
            return False
        if tokens[0].tag == Tag.WH:
            return True
        # Subject-auxiliary inversion: "do zorbins need glarn" — an AUX heading the clause with a
        # noun phrase after it is interrogative in a way a declarative never is.
        return tokens[0].tag == Tag.AUX and any(t.open_class for t in tokens[1:])

    # ---- assertions ------------------------------------------------------- #
    def _assertion(self, tokens: List[Token], out: Meaning) -> None:
        """``NP VERB NP`` (English) or ``NP PREP NP VERB`` (Hinglish SOV)."""
        out.kind = "assertion"
        if out.confidence <= 0.0:
            out.confidence = 0.8
        if self._hinglish_sov(tokens, out):
            return
        # Do-support carries tense and agreement, never content: "zorbins do not need glarn"
        # asserts exactly what "zorbins need glarn" denies, and the ``do`` is machinery for the
        # negation that has already been lifted out. Removed here so the copula guard in
        # :meth:`_locate_verb` sees only genuine copulas — left in, it read every negated clause
        # as copular and returned nothing, silently undoing the negation support this module
        # exists for.
        tokens = _reindex([t for t in tokens
                           if not (t.tag == Tag.AUX and t.text in _DO_SUPPORT)])
        pivot = self._locate_verb(tokens)
        if pivot is None:
            self._intransitive(tokens, out)
            return
        subject = _phrase(tokens[:pivot])
        obj = _phrase(tokens[pivot + 1:])
        if not subject or not obj:
            return
        # A noun phrase has a size. Past it, what filled the slot is not a phrase — it is the
        # remainder of a sentence this frame could not parse, swept up because something had to
        # go there. Refusing is the honest outcome: the caller keeps whatever it had, which for
        # an unreadable sentence is nothing.
        if _width(subject) > _MAX_PHRASE or _width(obj) > _MAX_PHRASE:
            return
        out.subject, out.object = subject, obj
        out.verb = tokens[pivot].text
        out.relation = _lemma(out.verb)
        out.frame = "np-verb-np"

    @staticmethod
    def _intransitive(tokens: List[Token], out: Meaning) -> None:
        """``plants weaken`` — a happening with no object.

        Deliberately after :meth:`_locate_verb` has declined, and only for exactly two open-class
        tokens, because every longer sentence that reaches here is one the verb scorer could not
        read and filling slots from it would be inventing structure. The resulting Meaning is
        readable but **not** :attr:`~Meaning.complete`, so a consumer that needs a triple still
        gets nothing — what it gains is the condition and modality this sentence carried, which
        were being discarded along with it.
        """
        # No subject in a clause that opens with a preposition — "about deep learning" names no
        # one who did anything. The same guard `_q_bare` keeps, and it has to be repeated here
        # because making an intransitive assertion `complete` reopened the precision hole this
        # module's refusals exist to close: measured, "tell me about deep learning" and "define
        # overfitting" both started grounding again.
        if tokens and tokens[0].tag == Tag.PREP:
            return
        content = [t for t in tokens if t.open_class]
        if len(content) != 2:
            return
        subject, verb = content
        if subject.index >= verb.index:
            return
        out.subject, out.verb, out.relation = subject.text, verb.text, _lemma(verb.text)
        out.frame = "np-verb"

    def _hinglish_sov(self, tokens: List[Token], out: Meaning) -> bool:
        """``<subject> ko <object> chahiye`` — subject-object-verb, with the case marker doing the
        work English does with word order.

        Handled by its own frame rather than by the English one because the English pattern
        matches it *wrongly* rather than failing: it takes the light verb's position as the object
        and the object's as the verb. That exact substitution is the failure
        :mod:`nyxara.njp.compile` documents on ``"agar paani aadha kar doon"``.
        """
        marker = next((t for t in tokens if t.text in ("ko", "को")), None)
        verb = next((t for t in reversed(tokens) if t.text in _HINGLISH_VERBS), None)
        if marker is None or verb is None or verb.index <= marker.index:
            return False
        subject = _phrase(tokens[:marker.index])
        obj = _phrase(tokens[marker.index + 1:verb.index])
        if not subject or not obj:
            return False
        out.subject, out.object = subject, obj
        out.verb = verb.text
        out.relation = _lemma(verb.text)
        out.frame = "hinglish-sov"
        out.language = "hi"
        return True

    # ---- questions -------------------------------------------------------- #
    def _question(self, tokens: List[Token], out: Meaning) -> None:
        """Every question frame is tried and the one that fills the most slots wins.

        Not first-match. The frames overlap by construction — ``what does a X eat`` and
        ``X needs what`` differ only in where the wh-word sits — and ordering them would mean
        choosing, once and for all, which overlap resolves in which direction. Scoring instead
        lets the sentence decide.
        """
        if out.confidence <= 0.0:
            out.confidence = 0.7
        best: Optional[Meaning] = None
        for frame in (self._q_fronted_wh, self._q_trailing_wh, self._q_polar,
                      self._q_nominal, self._q_wh_np_verb, self._q_hinglish, self._q_bare):
            candidate = Meaning(kind="question")
            try:
                if not frame(list(tokens), candidate):
                    continue
            except Exception:  # noqa: BLE001
                continue
            if best is None or _slots(candidate) > _slots(best):
                best = candidate
        if best is None:
            return
        out.kind = best.kind
        out.subject, out.relation, out.object = best.subject, best.relation, best.object
        out.verb, out.focus, out.frame = best.verb, best.focus, best.frame

    @staticmethod
    def _q_fronted_wh(tokens: List[Token], out: Meaning) -> bool:
        """``what does a X VERB`` / ``what is a X`` — wh, auxiliary, noun phrase, verb."""
        if not tokens or tokens[0].tag != Tag.WH:
            return False
        rest = tokens[1:]
        if not rest:
            return False
        if rest[0].tag == Tag.AUX:
            body = rest[1:]
            # The verb is the LAST open-class token: "what does a big zorbin eat" — everything
            # between the auxiliary and it is the subject noun phrase.
            verbs = [t for t in body if t.open_class]
            # Only do-support licenses a separate bare verb. After a copula the remainder is one
            # noun phrase however many words it has, so "what is deep learning" asks what *deep
            # learning* is — not what "deep" does. Without this the frame split every multi-word
            # nominal into a subject and an invented verb.
            if rest[0].text not in _DO_SUPPORT:
                # A preposition inside the complement means it is not one noun phrase: "what is
                # necessary FOR a zorbin" names a relation and its argument, which
                # :meth:`_q_nominal` reads correctly. Declining rather than scoring lower,
                # because both frames fill two slots and a tie would be settled by frame order.
                if any(t.tag == Tag.PREP for t in body):
                    return False
                subject = _phrase(body)
                if not subject:
                    return False
                out.subject, out.relation, out.verb = subject, "be", rest[0].text
                out.focus, out.frame = "object", "wh-copula-np"
                return True
            if len(verbs) < 2:
                # "what does a zorbin" — the auxiliary IS the relation, and there is no other verb.
                subject = _phrase(body)
                if not subject:
                    return False
                out.subject, out.relation, out.verb = subject, "be", rest[0].text
                out.focus, out.frame = "object", "wh-aux-np"
                return True
            verb = verbs[-1]
            # A preposition standing between the head and the candidate verb means this is not a
            # verb-object structure — it is a nominal with a prepositional argument, and
            # :meth:`_q_nominal` reads it correctly. Bailing here rather than scoring lower,
            # because the two frames fill the same number of slots and a tie would be broken by
            # frame order, which is the thing this module refuses to decide anything by.
            if any(t.tag == Tag.PREP and verbs[0].index < t.index < verb.index for t in body):
                return False
            subject = _phrase([t for t in body if t.index < verb.index])
            if not subject:
                return False
            out.subject, out.verb, out.relation = subject, verb.text, _lemma(verb.text)
            out.focus, out.frame = "object", "wh-aux-np-verb"
            return True
        # "what colour is the sky" — wh, nominal, aux, np. The nominal is the relation.
        nominal = [t for t in rest if t.open_class]
        aux = next((t for t in rest if t.tag == Tag.AUX), None)
        if aux is not None and nominal and nominal[0].index < aux.index:
            out.relation = _lemma(nominal[0].text)
            out.verb = nominal[0].text
            subject = _phrase([t for t in rest if t.index > aux.index])
            if not subject:
                return False
            out.subject, out.focus, out.frame = subject, "object", "wh-nominal-aux-np"
            return True
        return False

    @staticmethod
    def _q_trailing_wh(tokens: List[Token], out: Meaning) -> bool:
        """``Xs need what`` — the wh-word in the object's own position."""
        if not tokens or tokens[-1].tag != Tag.WH:
            return False
        body = tokens[:-1]
        verbs = [t for t in body if t.open_class]
        if len(verbs) < 2:
            return False
        verb = verbs[-1]
        subject = _phrase([t for t in body if t.index < verb.index])
        if not subject:
            return False
        out.subject, out.verb, out.relation = subject, verb.text, _lemma(verb.text)
        out.focus, out.frame = "object", "np-verb-wh"
        return True

    @staticmethod
    def _q_polar(tokens: List[Token], out: Meaning) -> bool:
        """``do Xs need Y`` — auxiliary-initial, and the answer set is yes/no/unknown.

        Its own kind rather than a question with an object already filled, because those are
        different acts: one asks which thing, the other asks whether *this* thing. Answering a
        polar with a name is a category error that reads as fluent.
        """
        if not tokens or tokens[0].tag != Tag.AUX:
            return False
        body = tokens[1:]
        verbs = [t for t in body if t.open_class]
        if len(verbs) < 3:
            return False
        # NP VERB NP after the auxiliary. The verb is the pivot; scoring it here would double the
        # work the assertion path already does, so reuse it.
        pivot = SemanticCompiler._locate_verb(body)
        if pivot is None:
            return False
        subject = _phrase(body[:pivot])
        obj = _phrase(body[pivot + 1:])
        if not subject or not obj:
            return False
        out.kind = "polar_question"
        out.subject, out.object = subject, obj
        out.verb, out.relation = body[pivot].text, _lemma(body[pivot].text)
        out.focus, out.frame = "truth", "aux-np-verb-np"
        return True

    @staticmethod
    def _q_nominal(tokens: List[Token], out: Meaning) -> bool:
        """``what's necessary for a X`` / ``what is the colour of the sky`` — the relation is a
        noun or adjective and a preposition attaches the subject to it."""
        if not tokens or tokens[0].tag != Tag.WH:
            return False
        prep = next((t for t in tokens if t.tag == Tag.PREP), None)
        if prep is None:
            return False
        head = [t for t in tokens[:prep.index] if t.open_class]
        subject = _phrase(tokens[prep.index + 1:])
        if not head or not subject:
            return False
        out.subject = subject
        out.verb = head[-1].text
        out.relation = _lemma(head[-1].text)
        out.focus, out.frame = "object", "wh-nominal-prep-np"
        return True

    @staticmethod
    def _q_wh_np_verb(tokens: List[Token], out: Meaning) -> bool:
        """``what a X requires`` — wh, noun phrase, verb, with no auxiliary inverted.

        This is the shape an embedded question has once its matrix speech act is removed, and
        also the shape of a bare relative. No auxiliary follows the wh-word, which is exactly
        what distinguishes it from :meth:`_q_fronted_wh` — so the two never compete.
        """
        if not tokens or tokens[0].tag != Tag.WH:
            return False
        rest = tokens[1:]
        if not rest or rest[0].tag == Tag.AUX:
            return False
        verbs = [t for t in rest if t.open_class]
        if len(verbs) < 2:
            return False
        verb = verbs[-1]
        subject = _phrase([t for t in rest if t.index < verb.index])
        if not subject:
            return False
        out.subject, out.verb, out.relation = subject, verb.text, _lemma(verb.text)
        out.focus, out.frame = "object", "wh-np-verb"
        return True

    @staticmethod
    def _q_hinglish(tokens: List[Token], out: Meaning) -> bool:
        """``X ko kya chahiye`` — subject, case marker, wh, light verb."""
        marker = next((t for t in tokens if t.text in ("ko", "को")), None)
        wh = next((t for t in tokens if t.tag == Tag.WH), None)
        if marker is None or wh is None or wh.index <= marker.index:
            return False
        subject = _phrase(tokens[:marker.index])
        if not subject:
            return False
        verb = next((t for t in tokens[wh.index + 1:] if t.open_class), None)
        out.subject = subject
        out.verb = verb.text if verb is not None else wh.text
        out.relation = _lemma(out.verb)
        out.focus, out.frame = "object", "hinglish-ko-wh"
        return True

    @staticmethod
    def _q_bare(tokens: List[Token], out: Meaning) -> bool:
        """``X needs?`` — a noun phrase and a verb, the object simply absent.

        Last and weakest on purpose: it fills the fewest slots, so any other frame that matches
        outscores it. It exists because the elliptical question is common in typed conversation
        and previously grounded to nothing at all.
        """
        # No subject is available in a clause that opens with a preposition — what precedes the
        # noun phrase is the preposition, not something that could be doing the verb. Measured:
        # "tell me about deep learning" reduced to "about deep learning" and compiled as
        # ``deep --learn--> ?``, which then flipped `Grounder._affinity` into the wrong regime and
        # cost the recall path three tests.
        if tokens and tokens[0].tag == Tag.PREP:
            return False
        verbs = [t for t in tokens if t.open_class]
        if len(verbs) < 2:
            return False
        verb = verbs[-1]
        # Subject-verb agreement, which is the only evidence available here that these two tokens
        # stand in a subject-verb relation at all rather than being two nouns. English marks it
        # in one of two places: on the verb when the subject is singular ("zorbin needs"), or on
        # the subject when it is plural and the verb is bare ("reeds filter"). Neither present
        # means no evidence — and "define overfitting" is what that looks like: an imperative the
        # frame read as a claim that something called "define" overfits.
        subject_tokens = [t for t in tokens if t.index < verb.index]
        head = next((t for t in reversed(subject_tokens) if t.open_class), None)
        if head is None:
            return False
        if not (_plural(verb.text) or _plural(head.text)):
            return False
        subject = _phrase(subject_tokens)
        if not subject:
            return False
        out.subject, out.verb, out.relation = subject, verb.text, _lemma(verb.text)
        out.focus, out.frame = "object", "np-verb-bare"
        return True

    # ---- the hard step ---------------------------------------------------- #
    @staticmethod
    def _locate_verb(tokens: Sequence[Token]) -> Optional[int]:
        """Which token is the verb, by evidence rather than by a verb list.

        No tag marks a verb — that is precisely what makes the open class open. So every interior
        open-class position is a candidate and each is scored on signals that are independently
        available without a dictionary:

        =========================================  ======  ==================================
        signal                                     weight  why it is evidence
        =========================================  ======  ==================================
        an ``AUX`` immediately before               +3.0   "does **eat**" — auxiliaries take
                                                           verbs, and nothing else
        carries ``-s`` where the subject does not   +1.5   third-person agreement
        a ``DET`` immediately after                 +2.0   "eat **the** meat" — determiners
                                                           open noun phrases, so what precedes
                                                           one ended a predicate
        material on both sides                      +1.0   a transitive verb has two arguments
        immediately after a single-token subject    +0.8   the commonest English shape
        at either edge                              −4.0   a clause-final or initial token in
                                                           this frame is an argument, not a
                                                           predicate
        preceded by a ``PREP``                      −2.5   "for **water**" — that is an object
        =========================================  ======  ==================================

        The weights are ordinal, not fitted: they encode which signal beats which, and the
        benchmark in :mod:`nyxara.eval.adversarial` is what says whether the ordering is right.
        Returning ``None`` where the best score is not positive is deliberate — an unreadable
        sentence must reach the caller as unreadable, not as a guess with a slot filled.
        """
        n = len(tokens)
        if n < 3:
            return None
        # A copular clause has no verb to locate — the auxiliary *is* the relation, and which
        # relation it names ("my name is Jay" vs "a sparrow is a bird") is decided by the 101
        # patterns that already read copulas in detail. Declining here keeps the compiler to the
        # sentences nobody else could read. Measured: without it "my lucky number is seven"
        # compiled as ``my --lucky--> number seven``, a reading of a sentence nobody said.
        if any(t.tag == Tag.AUX for t in tokens):
            return None
        subject_plural = tokens[0].open_class and _plural(tokens[0].text)
        best_index, best_score = None, 0.0
        for i, token in enumerate(tokens):
            if not token.open_class:
                continue
            if i == 0 or i == n - 1:
                continue
            # A preposition anywhere before the candidate means the material in front of it is a
            # prepositional phrase, not a subject in a verb-object relation with this token. Every
            # case-marked language puts one there ("cricket ke baare mein mujhe …"), and reading
            # such a clause as English SVO is the exact substitution :mod:`nyxara.njp.compile`
            # documents: it grounded that sentence — which says "I don't know much about
            # cricket" — as ``cricket baare mujhe --zyada--> pata``.
            if any(t.tag == Tag.PREP for t in tokens[:i]):
                continue
            # A preposition immediately after the candidate means what follows is a
            # prepositional phrase, not a direct object — "rolling WITH the constant flow" has
            # no object at all. Without this the frame swept the rest of the sentence into the
            # object slot: a rap lyric grounded as ``whoa --roll--> constant flow rap game dont
            # own me``. A threshold reading like "water freezes at 100" is not lost by this;
            # `Grounder._extract_measurements` runs before the pattern loop and owns it.
            if tokens[i + 1].tag == Tag.PREP:
                continue
            score = 1.0                                    # material on both sides
            previous, following = tokens[i - 1], tokens[i + 1]
            if previous.tag == Tag.AUX:
                score += 3.0
            if previous.tag == Tag.PREP:
                score -= 2.5
            if previous.tag == Tag.DET:
                score -= 2.0                               # determiners open NPs, not predicates
            if following.tag == Tag.DET:
                score += 2.0
            if _plural(token.text) and not subject_plural:
                score += 1.5
            if _plural(token.text) and subject_plural:
                score -= 0.5                               # more likely a second plural noun
            if i == 1:
                score += 0.8
            if best_index is None or score > best_score:
                best_index, best_score = i, score
        if best_index is None or best_score <= 0.0:
            return None
        return best_index


def _reindex(tokens: List[Token]) -> List[Token]:
    """Renumber after a removal. Every frame reads ``Token.index``, so a stale one silently
    reorders the sentence for whatever runs next."""
    return [Token(text=t.text, tag=t.tag, index=i) for i, t in enumerate(tokens)]


#: How many content words a subject or object may hold before the reading stops being a noun
#: phrase. Four is generous — "the very large red zorbin" is four — and the sentences it excludes
#: are ones where the frame matched nothing and absorbed the rest of the clause.
_MAX_PHRASE = 4


def _width(phrase: str) -> int:
    """Content words in a filled slot."""
    return len(str(phrase or "").split())


def _phrase(tokens: Sequence[Token]) -> str:
    """The content of a noun phrase — its open-class tokens, determiners and case markers gone.

    Pronouns are kept: "my name" and "name" are different subjects, and dropping the pronoun is
    how a claim about the Master becomes a claim about nobody.
    """
    kept = [t.text for t in tokens if t.open_class or t.tag == Tag.PRON]
    return " ".join(kept).strip()


def _slots(meaning: Meaning) -> int:
    """How much of the representation a frame managed to fill — the score frames compete on."""
    return sum(1 for value in (meaning.subject, meaning.relation, meaning.object) if value)


def compile_meaning(text: str, *, interrogative: Optional[bool] = None) -> Meaning:
    """Compile one sentence into a :class:`Meaning`. Fail-soft; never raises."""
    return SemanticCompiler().compile(text, interrogative=interrogative)
