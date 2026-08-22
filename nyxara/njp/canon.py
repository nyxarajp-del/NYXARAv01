"""NYXARA · njp/canon.py — one entity, one key (🔑, NJP V.07).

**The failure this module exists for**, reproduced end to end before a line of it was written::

    "a sparrow is a bird"  +  "bird needs water"   →  "what does a sparrow need?"  →  water   ✅
    "a sparrow is a bird"  +  "birds need water"   →  "what does a sparrow need?"  →  bird    ❌

One character. The inheritance in :meth:`~nyxara.njp.core.CognitiveLearningCore._inherit` is
correct, the relation fold is correct, the question is read correctly as ``(sparrow, requires)`` —
and the chain still breaks, because :attr:`~nyxara.njp.grounding.Grounder.facts` is keyed on
``(subject.lower(), predicate)``, which is the **surface form**. ``birds`` and ``bird`` are two
entities that never meet, so ``sparrow is_a bird`` walks to a kind about which nothing is held.

The store's spelling rule was ``.strip().lower()``, and lowercasing is not canonicalisation. This
module is the missing half.

**Relations were already nearly right, and that is why this is entity-only.**
``_PREDICATE_ALIASES`` in :mod:`nyxara.njp.grounding` already folds ``need`` / ``needs`` /
``require`` / ``requires`` onto one edge, in two languages. Measured before changing anything: the
fact stored as ``birds --requires--> water`` was queried as ``(sparrow, requires)`` — the predicate
matched perfectly and the subject did not. So :func:`canonical_relation` here is a thin, additive
gap-filler over that table, and deliberately **does not** touch the directional words. The alias
table's own comment is explicit that ``cause`` / ``karan`` / ``wajah`` are absent on purpose:
folding them onto ``causes`` would send *"aag ka karan kya hai"* to a forward lookup and answer
with what fire causes, which is backwards rather than merely unhelpful. That reasoning is load
bearing and this module preserves it.

**Script-aware, because English morphology is not a universal rule.** The plural fold runs only on
Latin-script words. Everything else — Devanagari, Cyrillic, Han, and Hinglish written in Devanagari
— returns from :func:`canonical_entity` exactly as ``.strip().lower()`` would have returned it,
which is byte-for-byte what the store did before this module existed. That is the guarantee that
makes this safe to put on the write path: for every non-Latin key, nothing changed.

Note in particular that :func:`~nyxara.njp.tongue.roman_key` is **not** used here. It is the right
tool for *"is this the same word across two scripts"* and the wrong tool for a store key: it folds
doubled letters and drops a trailing schwa, so ``sparrow`` becomes ``sparow`` and ``data`` becomes
``dat``. Those collisions are acceptable in a matching heuristic and not acceptable in the identity
of a stored fact.

**The head is folded, not the phrase.** ``canonical_entity("deep learning")`` is ``deep learning``,
not ``learning``. :func:`~nyxara.njp.grounding._kind_head` reduces a *kind phrase* to its head noun
and is genuinely useful for finding shared kinds — but applying it to an entity key would collide
``deep learning`` with ``machine learning``, which is a far worse failure than the plural break it
was meant to fix. English pluralises the head, so only the final token is folded.

**Nothing here changes what a fact means.** :class:`~nyxara.njp.grounding.GroundedTriple` carries
modality, condition, negation, source and confidence; canonicalisation rewrites the *lookup key*
and leaves every one of those, and the surface spelling, untouched. A canonical key is how two
statements find each other. It is not a claim that they say the same thing.

Pure standard library. No LLM.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, FrozenSet

__all__ = ["canonical_entity", "canonical_relation", "singular", "is_latin_word"]


#: Determiners that carry no identity. Stripped only from the FRONT of a phrase, and never when
#: they are the whole of it — "the" alone is not an entity, but neither is "".
_ARTICLES: FrozenSet[str] = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "some", "any",
    "every", "each", "all", "my", "your", "his", "her", "its", "our", "their",
})

#: Plurals English does not form by suffix. Small on purpose: a long list of hand-entered word
#: pairs is a vocabulary, and a vocabulary is the thing this module is trying not to become. These
#: are here because the suffix rules below get them actively wrong, not merely miss them.
_IRREGULAR: Dict[str, str] = {
    "children": "child", "people": "person", "men": "man", "women": "woman",
    "feet": "foot", "teeth": "tooth", "geese": "goose", "mice": "mouse",
    "oxen": "ox", "leaves": "leaf", "lives": "life", "knives": "knife",
    "wolves": "wolf", "halves": "half", "selves": "self", "shelves": "shelf",
    "loaves": "loaf", "thieves": "thief", "wives": "wife", "calves": "calf",
}

#: Words that end in ``s`` while being singular. Without these, ``-s`` stripping turns "physics"
#: into "physic" and "species" into "specie" — both real words, both the wrong entity.
_SINGULAR_S: FrozenSet[str] = frozenset({
    "physics", "mathematics", "maths", "economics", "politics", "ethics", "statistics",
    "news", "species", "series", "means", "species", "gas", "lens", "bus", "atlas",
    "canvas", "status", "focus", "virus", "campus", "census", "bonus", "corpus", "genus",
    "analysis", "basis", "crisis", "thesis", "diagnosis", "hypothesis", "axis", "oasis",
    "always", "perhaps", "sometimes", "yes", "his", "its", "this", "thus", "less",
})

#: Suffixes after which ``-es`` is the plural marker rather than part of the stem: box→boxes,
#: church→churches, bush→bushes, glass→glasses, buzz→buzzes.
_ES_STEMS = ("ss", "x", "z", "ch", "sh")

_WORD_RE = re.compile(r"\S+")


def is_latin_word(word: str) -> bool:
    """Is every letter here written in the Latin script?

    The gate on all English morphology below. A word with a single Devanagari or Cyrillic letter
    in it is not a word English pluralisation rules have anything true to say about, and guessing
    is worse than declining: the store would end up with two keys for one entity, which is the
    exact defect this module exists to remove.

    Digits, hyphens and apostrophes are permitted — they appear inside real entity names and carry
    no script of their own.
    """
    try:
        seen = False
        for ch in str(word or ""):
            if not ch.isalpha():
                continue
            seen = True
            if "LATIN" not in unicodedata.name(ch, ""):
                return False
        return seen
    except Exception:  # noqa: BLE001 — an unnameable character is not Latin
        return False


def singular(word: str) -> str:
    """The singular of one Latin-script word, or the word unchanged.

    Rules only, in the order that matters. Every branch declines rather than guesses when the
    result would be too short to be a name: "as", "is" and "us" are not the plurals of "a", "i"
    and "u", and a rule that produced those would quietly shred short entity names.
    """
    try:
        low = str(word or "").strip().lower()
        if not low or not is_latin_word(low):
            return low
        if low in _IRREGULAR:
            return _IRREGULAR[low]
        if low in _SINGULAR_S or not low.endswith("s"):
            return low
        # "-ies" → "-y", but only where a stem survives: "ties" must not become "ty".
        if low.endswith("ies") and len(low) > 4:
            return low[:-3] + "y"
        # "-es" after a sibilant stem. Checked before the bare "-s" rule because "glasses" ends in
        # "ss" too, and the bare rule would leave "glasse".
        if low.endswith("es") and len(low) > 4:
            stem = low[:-2]
            if stem.endswith(_ES_STEMS):
                return stem
        # "-ss" is never a plural marker: "grass", "class", "process" keep their tail.
        if low.endswith("ss"):
            return low
        if len(low) > 3:
            return low[:-1]
        return low
    except Exception:  # noqa: BLE001
        return str(word or "").strip().lower()


def canonical_entity(phrase: str) -> str:
    """The store key for an entity: one key per thing, whatever the sentence called it.

    ``birds`` · ``a bird`` · ``the Birds`` · ``these birds`` all reach ``bird``. Multi-word names
    keep every word — only the head is folded — so ``deep learning`` and ``machine learning``
    stay two entities.

    Non-Latin input returns ``.strip().lower()`` and nothing else, which is exactly what the store
    did before this function existed.
    """
    try:
        low = str(phrase or "").strip().lower()
        if not low:
            return ""
        words = _WORD_RE.findall(low)
        if not words:
            return low
        # Leading determiners, but never all of them: "the" on its own stays "the" rather than
        # becoming the empty key, which would silently merge every article-only phrase into one.
        while len(words) > 1 and words[0] in _ARTICLES:
            words.pop(0)
        if len(words) == 1 and words[0] in _ARTICLES:
            return words[0]
        words[-1] = singular(words[-1])
        return " ".join(w for w in words if w)
    except Exception:  # noqa: BLE001 — a key that cannot be built is the surface form
        return str(phrase or "").strip().lower()


#: Verb forms the alias table in :mod:`nyxara.njp.grounding` does not reach. Additive only, and
#: every entry folds onto an edge name that table already produces — this widens the door, it does
#: not name a new relation. The directional words ("cause", "karan", "wajah") stay absent for the
#: reason `_PREDICATE_ALIASES` documents: they are read by the `_CAUSE_OF` question patterns, and
#: folding them forward would answer the reverse question with the forward edge.
_RELATION_TAIL: Dict[str, str] = {
    "needed": "requires", "required": "requires", "requiring": "requires",
    "needing": "requires",
    "owned": "owns", "owning": "owns", "having": "owns",
    "increased": "increases", "increasing": "increases", "raise": "increases",
    "decreased": "decreases", "decreasing": "decreases", "reduce": "decreases",
    "located": "located_in", "lived": "located_in",
    "worked": "works_at", "working": "works_at",
}


def canonical_relation(predicate: str) -> str:
    """Fold a relation name that :mod:`nyxara.njp.grounding`'s alias table already normalised.

    Applied **after** that table, never instead of it. The table is where relation naming lives;
    this closes the tense and participle forms it does not list, and returns anything it does not
    recognise untouched so an unknown relation keeps its own name rather than being rounded onto
    a neighbour.
    """
    try:
        low = str(predicate or "").strip().lower()
        return _RELATION_TAIL.get(low, low)
    except Exception:  # noqa: BLE001
        return str(predicate or "").strip().lower()
