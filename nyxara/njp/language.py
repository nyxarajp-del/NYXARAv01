"""NYXARA · njp/language.py — the grammar she learns, not the one she shipped (🗣, NJP V.19).

:mod:`nyxara.njp.tongue` tokenises any script. :mod:`nyxara.njp.semantics` tags the closed class
and matches **frames over the tag sequence**, which is what made an open verb list unnecessary.
:mod:`nyxara.njp.grounding` turns what came back into beliefs. Three organs, and between them one
thing that does not exist: a way for her to **acquire a construction she was not shipped with**.

The frames in ``semantics.py`` are written by a person — ``_q_fronted_wh``, ``_q_polar``,
``_q_hinglish`` and six more. They are good frames, and they are the whole of the grammar. A
sentence whose shape is not one of them is ``unreadable``, and it stays unreadable however many
times it is said to her, because nothing in the package turns *a sentence she was shown the
meaning of* into *a shape she can read the next sentence with*. Measured, that is not a subtle
gap: over a minted language whose words come in the order object–subject–verb, the shipped
compiler reads **0 of 24** held-out sentences, and would still read 0 after a thousand lessons.

This module is the missing half, and it is deliberately the same discipline
:class:`~nyxara.njp.coding.Coder` applies to a demonstrated program:

    a lesson leaves a **shape**, never an answer — and a shape is only kept once it has read a
    sentence nobody demonstrated.

**Four things are learned here, and they are kept apart because they are four claims.**

* **Affixes are induced.** :class:`Morphology` finds them from a vocabulary in which many stems
  appear both bare and extended. No suffix list ships. A suffix carried by three stems whose bare
  forms are also words is evidence; one seen once is a coincidence and is not kept.
* **An affix *means* something only where a lesson said so.** Induction finds the shape ``-ik``;
  it cannot find that ``-ik`` is a plural. :meth:`Morphology.bind` takes a demonstrated pair and
  binds the feature, and it **refuses to bind an affix that was never independently induced** —
  otherwise the wug test is one memorised pair with a rule's name on it.
* **Word classes are discovered from where words occur**, by :class:`Lexicon`, and are given no
  names. A cluster found by distribution has no name, and inventing one ("noun") would be the
  ontology this package refuses everywhere else, sneaking back in through the grammar.
* **Constructions are generalised from demonstrations**, by :class:`Grammar`. A demonstration is
  a surface and the :class:`~nyxara.njp.semantics.Meaning` it carries. What is retained is that
  sentence with its content words replaced by **slots**, and everything else — word order, case
  markers, particles, the negator, the question word — kept as fixed material.

**A role that is not in the sentence is a feature, not a refusal.** V.18 refused any demonstration
whose meaning was not in its surface, as a guard against a mislabelled lesson — and that guard had
to go, because a subject that is not in the sentence is **pro-drop**. What replaced it needed
nothing written for it: a mislabelled lesson leaves the word that *would* have been the subject
sitting in the sentence, so it becomes fixed material, and fixed material differs from one
demonstration to the next, so no shape ever gets a second demonstration to agree with it. A
genuinely dropped argument leaves no such residue.

**Two demonstrations, not one, and they have to disagree.** A shape supported by a single sentence
is that sentence. A shape is kept only when at least two demonstrations produce it *with different
fillers in at least one slot*, which is the smallest evidence that a slot is a position rather
than a word. Then every kept construction is re-run over the whole demonstration corpus, and one
that reads any demonstration into a **different** meaning than the one it came with is dropped —
a grammar that contradicts its own lessons is not a grammar.

**Every candidate is tried; the winner is chosen on evidence, never on order.** That is the rule
:mod:`nyxara.njp.semantics` and :mod:`nyxara.njp.compile` both state, and for the same reason:
``or`` short-circuits, and a short-circuit is how a sentence gets read by the construction written
for a different sentence. Here it has one more consequence worth having — when the two best
constructions score **equal** and disagree about what was said, the sentence comes back
``unreadable`` with the frame ``ambiguous``. A genuinely ambiguous sentence read confidently one
way is worse than one refused.

**She speaks only what she can read back.** :meth:`Grammar.say` renders a meaning through a
construction and then **parses its own output**; unless what comes back is the meaning it started
from, the sentence is thrown away and the next candidate is tried, and if none survives she
returns the empty string. This is what :mod:`nyxara.njp.voice` has always refused to do without —
a fluent surface she cannot verify is babble with her name on it. Here the verification is
mechanical, so the surface is allowed.

**And a meaning is not a language.** A faculty holds many :class:`Tongue`\\ s. Reading with no
tongue named tries all of them and lets the evidence say which language a sentence was in;
:meth:`LanguageFaculty.translate` reads in one and says in another, which is the only honest test
that meaning is stored apart from surface — the same fact, twice, in two grammars that share no
word.

**Six axes, and what each of them bought.** V.18 was the four bullets above and nothing else, and
:mod:`nyxara.njp.hard` measured it on twenty-one problems chosen against it: **1 / 21**. What
closed that was not twenty-one mechanisms but six general ones, each of which is a refusal
loosened in exactly one place —

#. **any number of named roles** (:data:`LEGACY_ROLES`), because three was a statement about
   which sentences may exist;
#. **detachable markers** (:class:`Marker`), so three demonstrated cells license a fourth;
#. **processes that are not affixes** (:class:`Process`) — circumfix, infix, reduplication,
   template — proposed by a lesson and corroborated by the vocabulary;
#. **an edit at a nameable anchor** (:func:`_edits`), for the shapes that remove, swap or double
   something instead of adding it;
#. **free order where marking is not free** (:meth:`Grammar._licence_order`);
#. **a role remembering what has filled it** (:attr:`Construction.fillers`), which is the only
   evidence there is when two shapes read the same three tokens differently.

Measured after all six: **21 / 21** on that bank, and **14 / 15** on a second bank written
afterwards. The full account, including the one problem that is not solved and the seven flaws the
banks turned out to have, is in ``docs/CAPABILITIES.md`` under NJP V.19.

**What is NOT claimed.** This is not a parser for English. It has no dictionary and no syntax
theory. **One construction is one flat sentence**: an embedded clause is read as more roles on a
flat pattern, not by recursion, so a clause inside a clause inside a clause has no representation
and arbitrary depth is not claimed. A filler is **one token** (see :data:`MAX_SPAN`), so a
two-word noun phrase is *unreadable* rather than mis-cut. A morphological rule needs a *paradigm*
to count — a language met only in running text grows none. The free-order axis enumerates, and
stops at :attr:`Grammar.max_free_slots`. A feature realised on two slots at once is read and said
correctly and is never folded into a paradigm, so it composes with nothing. Nothing here decides
truth or relevance: it produces a :class:`~nyxara.njp.semantics.Meaning` and stops, and where it
cannot, it produces an unreadable one rather than a guess.

Pure standard library. No LLM, no corpus on disk, no network. Every public entry point is
fail-soft: on any internal error it degrades to a null result rather than breaking a turn.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import (Any, Dict, FrozenSet, Iterable, List, Optional, Sequence,
                    Set, Tuple)

from nyxara.njp.semantics import Meaning

__all__ = [
    "MAX_SPAN", "ROLES", "LEGACY_ROLES", "Segment", "Affix", "Process", "Rule",
    "Morphology", "Lexicon",
    "Slot", "Joint", "Marker", "Construction", "Demonstration", "Grammar", "Tongue",
    "LanguageFaculty", "Learned", "Reading", "tokenize_surface",
]

#: The most tokens a filler may be **demonstrated** at. A ceiling on what a lesson can teach, not
#: a licence to guess: what any particular slot accepts is :attr:`Slot.widths`, learned from what
#: it was actually shown holding.
#:
#: This was a flat 1 for a measured reason. At a flat 3, a construction of three bare slots
#: matched a four-word sentence by letting one slot swallow two words — so *a sentence with one
#: word too many came back read instead of refused*, on 8 of 20 controls in two of six minted
#: languages. Abstention is the property this package spends everything else to keep.
#:
#: But a flat 1 makes *"the big dog"* unreadable, which rules out most of English, so neither
#: constant is right and the constant was the mistake. Learned per slot, the minted languages
#: keep refusing (their fillers are all one token, so every slot learns width 1 and nothing
#: widens) and English gets its noun phrases. Measured both ways after the change, in
#: ``tests/njp/test_language.py``.
MAX_SPAN = 3

#: The three roles that live in named fields on :class:`~nyxara.njp.semantics.Meaning`. Every
#: other role a construction binds lives in that class's ``roles`` dict, and a construction may
#: bind **any number** of them.
#:
#: It was this tuple and only this tuple, and that was a statement about which sentences may exist.
#: Measured on twenty-one hard problems: a clause with an agent, a theme *and* a recipient scored
#: 0/8, and so did one with a location, an adjective, a possessor, a coordinate subject, a
#: relative clause and a complement clause — nine of the twenty-one, all for the same reason, none
#: of them mis-read. There was nowhere to put the fourth thing.
LEGACY_ROLES: Tuple[str, ...] = ("subject", "verb", "object")

#: Kept under its old name because it is exported, and no longer the ceiling it used to be.
ROLES: Tuple[str, ...] = LEGACY_ROLES

def tokenize_surface(text: str) -> List[str]:
    r"""Lower-cased tokens, script-agnostic, with ``?`` and ``!`` kept as tokens of their own.

    Delegates to :func:`nyxara.njp.tongue.tokenize`, which is the module in this package whose
    whole job is that no alphabet is privileged — and it is used here because the regex that was
    here instead **shattered Devanagari**. ``[^\W\d_]+`` looks script-neutral and is not:
    Python's ``\w`` excludes combining marks, so every matra and every virama fell out and
    ``लड़का आम खाता है`` tokenised as ``['लड', 'क', 'आम', 'ख', 'त', 'ह']`` — six fragments, none
    of them a word. Every Hindi lesson was then a sentence whose own words were not in it, so the
    whole course was refused and the Devanagari grammar came back with **0 shapes**.

    That is the third time this exact failure has been recorded in this package: NYX V.01's
    ``[a-z0-9]`` regex, V.09's missing negation, and now this. It is the reason ``tongue.py``
    exists, and the reason this function no longer has a regex of its own.

    Terminal punctuation is kept because in several languages it is the *only* mark a question
    carries, and dropping it would make a question and its assertion the same string. The full
    stop is dropped, because nothing distinguishes it from the end of the sentence.
    """
    try:
        from nyxara.njp.tongue import tokenize as _script_tokenize
        folded = unicodedata.normalize("NFKC", str(text or "")).strip().lower()
        out: List[str] = []
        for token in _script_tokenize(folded, keep_punct=True):
            text_of = str(getattr(token, "text", "") or "")
            if not text_of:
                continue
            if text_of in ("?", "!"):
                out.append(text_of)
            elif any(char.isalnum() for char in text_of):
                out.append(text_of)
        return out
    except Exception:  # noqa: BLE001
        return []


# --------------------------------------------------------------------------- #
# morphology — affixes induced from a vocabulary, features bound by demonstration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Affix:
    """One induced affix: its form, which end of the word it sits on, and its evidence."""

    form: str = ""
    side: str = "suffix"          # "suffix" | "prefix"
    stems: int = 0                #: distinct stems appearing both bare and with this affix
    feature: str = ""             #: bound by a lesson, never by induction

    @property
    def bound(self) -> bool:
        return bool(self.feature)

    def apply(self, stem: str) -> str:
        return f"{stem}{self.form}" if self.side == "suffix" else f"{self.form}{stem}"

    def strip(self, word: str) -> str:
        """The stem, or ``""`` where this affix is not on this word."""
        if self.side == "suffix":
            if len(word) > len(self.form) and word.endswith(self.form):
                return word[: len(word) - len(self.form)]
        elif len(word) > len(self.form) and word.startswith(self.form):
            return word[len(self.form):]
        return ""

    def to_dict(self) -> Dict[str, Any]:
        return {"form": self.form, "side": self.side, "stems": self.stems,
                "feature": self.feature}


#: What counts as a vowel when a process has to talk about them. Latin-script only, and the one
#: place in this module that is not script-agnostic — a templatic or harmony rule is a claim about
#: vowels, and there is no way to make that claim without knowing which letters are vowels. A
#: language written in a script this does not cover simply grows no such rule, which is the same
#: failure mode as an affix with no support: absence, not a wrong answer.
_VOWELS = frozenset("aeiouāēīōūáéíóúàèìòùäëïöü")


@dataclass(frozen=True)
class Process:
    """How one form is made from another — and not always by sticking something on the end.

    :class:`Morphology` induced suffixes and prefixes, which is a statement about which languages
    exist. Measured on six minted languages that mark a feature some other way: a circumfix, an
    infix, full reduplication, partial reduplication, a root-and-pattern template, and a suffix
    with two allomorphs — **0, 0, 0, 0, 0 and 4 of 8**. Not mis-analysed. Invisible, which is the
    same word this package used about Devanagari at NYX V.01 and about negation at V.09, and it
    means the same thing here: the mechanism had no way to represent what it was looking at.

    Six kinds, and the list is closed on purpose — each is a shape a process can have, not a
    language it belongs to:

    ``suffix`` / ``prefix``
        ``a`` on the end or the front.
    ``circumfix``
        ``a`` on the front **and** ``b`` on the end, as one process. Two halves that only ever
        occur together are one morpheme, and inducing them separately gets both of them wrong.
    ``infix``
        ``a`` inserted ``at`` characters in. Neither end, which is what makes it invisible to a
        mechanism that only looks at ends.
    ``reduplicate``
        the stem copied — the whole of it when ``at`` is 0, its first ``at`` characters otherwise.
        The affix *is* the stem, so no table of forms can hold it.
    ``template``
        the consonants are the word and the vowels are the grammar: keep the stem's consonants in
        order and write ``a``'s vowels between them. Nothing is concatenated anywhere.
    """

    kind: str = "suffix"
    a: str = ""
    b: str = ""
    at: int = 0
    #: For ``kind="edit"``: what is done — ``ins``, ``del``, ``dup`` or ``swap``.
    op: str = ""
    #: For ``kind="edit"``: what ``at`` is measured from — the left end (``L``), the right end
    #: (``R``), or the stem's last vowel (``V``).
    anchor: str = "R"
    #: For ``kind="edit"``: how many characters the edit touches.
    n: int = 0

    # -- what it does ------------------------------------------------------- #
    @staticmethod
    def _last_vowel_at(stem: str) -> int:
        for index in range(len(stem) - 1, -1, -1):
            if stem[index] in _VOWELS:
                return index
        return -1

    def _position(self, stem: str) -> int:
        if self.anchor == "L":
            return self.at
        if self.anchor == "R":
            return len(stem) - self.at
        return self._last_vowel_at(stem)

    def apply(self, stem: str) -> str:
        """The inflected form, or ``""`` where this process cannot act on this stem."""
        stem = str(stem or "")
        if not stem:
            return ""
        if self.kind == "edit":
            where = self._position(stem)
            if where < 0 or where > len(stem):
                return ""
            if self.op == "ins":
                return stem[:where] + self.a + stem[where:]
            if self.op == "del":
                if where - self.n < 1 or where > len(stem):
                    return ""
                return stem[: where - self.n] + stem[where:]
            if self.op == "swap":
                if where - 2 < 0 or where > len(stem):
                    return ""
                return stem[: where - 2] + stem[where - 1] + stem[where - 2] + stem[where:]
            if self.op == "dup":
                if where >= len(stem):
                    return ""
                return stem[: where + 1] + stem[where] + stem[where + 1:]
            if self.op == "put":
                if where - self.n < 1 or where > len(stem):
                    return ""
                return stem[: where - self.n] + self.a + stem[where:]
            return ""
        if self.kind == "suffix":
            return stem + self.a
        if self.kind == "prefix":
            return self.a + stem
        if self.kind == "circumfix":
            return f"{self.a}{stem}{self.b}"
        if self.kind == "infix":
            if len(stem) <= self.at:
                return ""
            return stem[:self.at] + self.a + stem[self.at:]
        if self.kind == "reduplicate":
            if self.at <= 0:
                return stem + stem
            return stem[:self.at] + stem if len(stem) > self.at else ""
        if self.kind == "template":
            bones = [char for char in stem if char not in _VOWELS]
            if len(bones) != len(self.a) + 1:
                return ""
            out = bones[0]
            for vowel, bone in zip(self.a, bones[1:]):
                out += vowel + bone
            return out
        return ""

    def strip(self, word: str) -> str:
        """The stem this process would have acted on, or ``""``.

        Not defined for every process, and that is honest rather than incomplete: a template
        erases the information that says which vowels the stem had, so ``strip`` on one cannot
        return a stem and does not pretend to.
        """
        word = str(word or "")
        if self.kind == "suffix":
            if self.a and len(word) > len(self.a) and word.endswith(self.a):
                return word[: len(word) - len(self.a)]
        elif self.kind == "prefix":
            if self.a and len(word) > len(self.a) and word.startswith(self.a):
                return word[len(self.a):]
        elif self.kind == "circumfix":
            if (len(word) > len(self.a) + len(self.b)
                    and word.startswith(self.a) and word.endswith(self.b)):
                return word[len(self.a): len(word) - len(self.b)] if self.b \
                    else word[len(self.a):]
        elif self.kind == "infix":
            head, rest = word[:self.at], word[self.at:]
            if self.a and rest.startswith(self.a) and len(head) + len(rest) > len(self.a):
                return head + rest[len(self.a):]
        elif self.kind == "reduplicate":
            if self.at <= 0:
                half = len(word) // 2
                if half and word[:half] == word[half:]:
                    return word[:half]
            elif len(word) > self.at and word[:self.at] == word[self.at:self.at * 2]:
                return word[self.at:]
        return ""

    @property
    def size(self) -> int:
        """How much of the word this process is responsible for — the tie-break on support."""
        if self.kind == "edit":
            # Below a suffix of the same length on purpose. An edit is the most powerful shape
            # here and the most easily fitted to a coincidence, so where a plain affix accounts
            # for the same pair it wins, and the edit is what is left for the pairs no affix can
            # explain at all.
            return 1
        if self.kind == "reduplicate":
            return 8 if self.at <= 0 else 6
        if self.kind == "template":
            return 7
        return len(self.a) + len(self.b) + (2 if self.kind in ("circumfix", "infix") else 0)

    @property
    def key(self) -> Tuple[Any, ...]:
        return (self.kind, self.a, self.b, self.at, self.op, self.anchor, self.n)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "a": self.a, "b": self.b, "at": self.at,
                "op": self.op, "anchor": self.anchor, "n": self.n}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Process":
        return Process(kind=str(data.get("kind") or "suffix"), a=str(data.get("a") or ""),
                       b=str(data.get("b") or ""), at=int(data.get("at") or 0),
                       op=str(data.get("op") or ""), anchor=str(data.get("anchor") or "R"),
                       n=int(data.get("n") or 0))


def _candidates(base: str, inflected: str) -> List[Process]:
    """Every process that could relate these two forms. All of them, not the first that fits.

    Which one is right is a question about the *rest of the vocabulary*, so this proposes and
    :meth:`Morphology.bind` disposes. A generator that stopped at the first fit would call
    ``maran`` → ``mamaran`` a prefix ``ma``, which is true of that pair and false of the language.
    """
    out: List[Process] = []
    if not base or not inflected or base == inflected:
        return out
    if inflected.startswith(base):
        out.append(Process("suffix", a=inflected[len(base):]))
    if inflected.endswith(base):
        out.append(Process("prefix", a=inflected[: len(inflected) - len(base)]))
    at = inflected.find(base)
    if at > 0 and at + len(base) < len(inflected):
        out.append(Process("circumfix", a=inflected[:at], b=inflected[at + len(base):]))
    # An infix: the two forms share a head and a tail, and the difference sits between them.
    head = 0
    while head < len(base) and head < len(inflected) and base[head] == inflected[head]:
        head += 1
    tail = 0
    while (tail < len(base) - head and tail < len(inflected) - head
           and base[len(base) - 1 - tail] == inflected[len(inflected) - 1 - tail]):
        tail += 1
    if head and tail and head + tail == len(base) and len(inflected) > len(base):
        out.append(Process("infix", a=inflected[head: len(inflected) - tail], at=head))
    if head + tail > min(len(base), len(inflected)):
        tail = max(0, min(len(base), len(inflected)) - head)
    if inflected == base + base:
        out.append(Process("reduplicate"))
    for copy in range(1, min(len(base), 4) + 1):
        if inflected == base[:copy] + base:
            out.append(Process("reduplicate", at=copy))
    bones = [char for char in base if char not in _VOWELS]
    other = [char for char in inflected if char not in _VOWELS]
    vowels = [char for char in inflected if char in _VOWELS]
    if bones and bones == other and len(vowels) == len(bones) - 1:
        out.append(Process("template", a="".join(vowels)))
    out.extend(_edits(base, inflected, head, tail))
    return out


def _edits(base: str, inflected: str, head: int, tail: int) -> List[Process]:
    """Processes expressed as one edit, measured from an end or from the stem's last vowel.

    The six named shapes above are a closed list, and a closed list of shapes is a statement about
    which languages exist. Measured on a bank chosen after they were written: a plural made by
    **removing** the last letter, one made by **swapping** the last two, and one made by writing
    the last vowel **twice** — 0/6, 0/6 and 3/6, all abstentions, because none of the three adds
    anything anywhere and every one of those shapes is about something added.

    What they have in common is that each is one edit at a position that can be *named* — the far
    end of the word, or its last vowel. So this proposes the same edit measured from each anchor,
    and the vocabulary decides which anchor the language actually uses: a deletion that is always
    one character from the right end is a rule, and the same deletion measured from the left is a
    different offset in every word and corroborates nothing.
    """
    out: List[Process] = []
    removed = base[head: len(base) - tail]
    added = inflected[head: len(inflected) - tail]
    anchors: List[Tuple[str, int]] = [("L", head + len(removed)),
                                      ("R", len(base) - head - len(removed))]
    vowel_at = Process._last_vowel_at(base)
    if not removed and added:
        for anchor, _at in (("L", head), ("R", len(base) - head)):
            out.append(Process("edit", a=added, op="ins", anchor=anchor,
                               at=head if anchor == "L" else len(base) - head))
        # The last vowel written twice: an insertion whose content is the character before it,
        # sitting exactly where the stem's last vowel is. Nothing about that is a fixed offset.
        if (len(added) == 1 and head > 0 and base[head - 1] == added
                and head - 1 == vowel_at):
            out.append(Process("edit", op="dup", anchor="V"))
    if removed and not added:
        for anchor, at in anchors:
            out.append(Process("edit", op="del", anchor=anchor, at=at, n=len(removed)))
    if removed and added and len(removed) == 2 and added == removed[::-1]:
        for anchor, at in anchors:
            out.append(Process("edit", op="swap", anchor=anchor, at=at))
    if removed and added and len(removed) == len(added) <= 2:
        # One thing written in place of another — Hindi's `लड़का` → `लड़के` is the whole plural,
        # with nothing added and nothing removed. The three ops above all move material about;
        # this one exchanges it, and without it a vowel-change paradigm is a set of pairs.
        for anchor, at in anchors:
            out.append(Process("edit", a=added, op="put", anchor=anchor, at=at,
                               n=len(removed)))
    return out


@dataclass(frozen=True)
class Rule:
    """A process, the feature a lesson bound it to, and when it applies rather than its rival.

    ``condition`` is the set of final vowels a stem may have for this rule to be the one that
    fires. It is empty for a feature with a single rule, and it is **induced** where a feature has
    two — from the stems that corroborated each of them, not from a theory about harmony. Two
    rules whose conditions overlap are two rules this module cannot choose between, and it says so
    rather than picking the more popular one and calling that a grammar.
    """

    feature: str = ""
    process: Process = field(default_factory=Process)
    condition: FrozenSet[str] = frozenset()
    #: What about the stem's ending the condition tests: whether it **is a vowel** (``class``),
    #: which vowel it last had (``vowel``), or which character it ends in (``final``). All three
    #: are induced and the **most general one that separates the family** wins.
    #:
    #: The last vowel alone is the shape a **harmony** rule has, and it was the only shape here.
    #: Hindi's habitual ending is `-ता` after a consonant and `-ाता` after a vowel — a condition
    #: on whether the stem *ends* in one — and `kha`, `padh` and `tod` do not differ in their last
    #: vowel at all. So no condition was defensible, the better-supported allomorph took every
    #: stem, and `tod` came back as `todata`.
    on: str = "vowel"
    support: int = 0

    def fits(self, stem: str) -> bool:
        if not self.condition:
            return True
        stem = str(stem or "")
        if self.on == "class":
            return bool(stem) and ("V" if stem[-1] in _VOWELS else "C") in self.condition
        if self.on == "final":
            return bool(stem) and stem[-1] in self.condition
        for char in reversed(stem):
            if char in _VOWELS:
                return char in self.condition
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {"feature": self.feature, "process": self.process.to_dict(),
                "condition": sorted(self.condition), "on": self.on, "support": self.support}


@dataclass(frozen=True)
class Segment:
    """A word taken apart. ``analysed`` is false when no induced affix explains it."""

    word: str = ""
    stem: str = ""
    affix: str = ""
    side: str = ""
    feature: str = ""

    @property
    def analysed(self) -> bool:
        return bool(self.affix)

    def to_dict(self) -> Dict[str, Any]:
        return {"word": self.word, "stem": self.stem, "affix": self.affix,
                "side": self.side, "feature": self.feature}


class Morphology:
    """Affixes found in a vocabulary, and what — if anything — a lesson said they mean.

    The induction is paradigmatic rather than statistical over characters: a suffix counts once
    for every stem that appears in the vocabulary **both bare and with the suffix attached**. That
    is the one piece of evidence separating a morpheme from a coincidence of spelling. English
    ``-ing`` scores because *walk*, *talk* and *read* are all words; ``-ea`` scores nothing,
    because *r*, *t* and *br* are not.

    Two thresholds, both deliberately blunt: an affix has to be carried by ``min_stems`` distinct
    stems, and it may not be longer than the stems that carry it. Everything else about morphology
    — allomorphy, stem changes, infixes, whether an affix is inflectional or derivational — is
    outside what a concatenative induction can see, and is reported absent rather than
    approximated.
    """

    def __init__(self, *, min_stems: int = 3, max_affix: int = 4) -> None:
        self.min_stems = max(2, int(min_stems))
        self.max_affix = max(1, int(max_affix))
        self.vocabulary: Dict[str, int] = {}
        self.affixes: List[Affix] = []
        #: Pairs a lesson demonstrated that induction could not support, kept so the refusal is
        #: inspectable rather than silent.
        self.refused: List[Tuple[str, str, str]] = []
        self.irregular: Dict[Tuple[str, str], str] = {}
        #: What a lesson bound, as processes rather than as suffixes. See :class:`Rule`.
        self.rules: List[Rule] = []
        #: Which stems corroborated which rule, kept so a condition can be induced from them.
        self._witness: Dict[Tuple[str, Tuple[str, str, str, int]], List[str]] = {}

    # -- observation -------------------------------------------------------- #
    def observe(self, words: Iterable[str]) -> int:
        """Add words to the vocabulary. Returns how many were new."""
        new = 0
        for raw in words:
            word = str(raw or "").strip().lower()
            if not word or not word[0].isalpha():
                continue
            if word not in self.vocabulary:
                new += 1
            self.vocabulary[word] = self.vocabulary.get(word, 0) + 1
        return new

    def observe_text(self, text: str) -> int:
        return self.observe(tokenize_surface(text))

    # -- induction ---------------------------------------------------------- #
    def induce(self) -> List[Affix]:
        """Find the affixes the vocabulary supports. Idempotent, and keeps bound features."""
        bound = {(a.form, a.side): a.feature for a in self.affixes if a.feature}
        words = set(self.vocabulary)
        suffixes: Dict[str, Set[str]] = {}
        prefixes: Dict[str, Set[str]] = {}
        for word in words:
            for cut in range(1, len(word)):
                head, tail = word[:cut], word[cut:]
                # One cut, read twice: `head` is a stem with `tail` suffixed to it, or `head` is a
                # prefix on the stem `tail`. Which of the two the evidence supports is decided by
                # whichever of the halves is itself a word.
                if len(tail) <= self.max_affix and len(tail) <= len(head) and head in words:
                    suffixes.setdefault(tail, set()).add(head)
                if len(head) <= self.max_affix and len(head) <= len(tail) and tail in words:
                    prefixes.setdefault(head, set()).add(tail)
        found: List[Affix] = []
        for form, stems in suffixes.items():
            if len(stems) >= self.min_stems:
                found.append(Affix(form=form, side="suffix", stems=len(stems),
                                   feature=bound.get((form, "suffix"), "")))
        for form, stems in prefixes.items():
            if len(stems) >= self.min_stems:
                found.append(Affix(form=form, side="prefix", stems=len(stems),
                                   feature=bound.get((form, "prefix"), "")))
        # Longest first, then best supported: :meth:`analyse` takes the first that fits, and a
        # longer affix is the more specific claim about the same word.
        found.sort(key=lambda a: (-len(a.form), -a.stems, a.form))
        self.affixes = found
        return found

    # -- binding: what a process means, and only where the vocabulary agrees -- #
    def _corroborate(self, process: Process, exclude: str) -> Tuple[int, List[str]]:
        """How many *other* words in the vocabulary this process also relates to a word.

        The whole of the honesty in :meth:`bind` is here. A demonstrated pair proposes a process;
        this counts how much of the language agrees. ``maran`` → ``mamaran`` proposes both "prefix
        ``ma``" and "copy the first syllable", and they are told apart not by taste but by the
        fact that the language contains ``sotel`` → ``sosotel`` and ``pigan`` → ``pipigan`` and
        no ``masotel`` at all.
        """
        support: List[str] = []
        for word in self.vocabulary:
            if word == exclude:
                continue
            made = process.apply(word)
            if made and made != word and made in self.vocabulary:
                support.append(word)
        return len(support), support

    @staticmethod
    def _last_vowel(word: str) -> str:
        for char in reversed(str(word or "")):
            if char in _VOWELS:
                return char
        return ""

    def _condition_rules(self, feature: str) -> None:
        """Where one feature has two processes, work out which stems each of them is for.

        Induced from the stems that corroborated each rule, never assumed: the condition is the
        set of final vowels those stems actually had. Two rules whose stem sets overlap in that
        vowel are two rules she cannot choose between, and both keep an empty condition so
        :meth:`inflect` falls back to support and the ambiguity stays visible rather than being
        decided by whichever was demonstrated first.
        """
        family = [rule for rule in self.rules if rule.feature == feature]
        if len(family) < 2:
            return

        def sets(pick: Any) -> Dict[Any, Set[str]]:
            return {rule.process.key: {pick(stem)
                                       for stem in self._witness.get(
                                           (feature, rule.process.key), [])} - {""}
                    for rule in family}

        def separates(found: Dict[Any, Set[str]]) -> bool:
            keys = list(found)
            return all(not (found[one] & found[other])
                       for index, one in enumerate(keys) for other in keys[index + 1:])

        # Three properties of the stem's ending are induced, and the **most general one that
        # separates the allomorphs** is the condition — which is why they are tried in this order.
        #
        # The character list was the only one for a while, and it is the narrowest possible
        # description of the evidence: a rule witnessed on stems ending in `h` and `d` fires on
        # those two letters and abstains on a stem ending in `m`, even though every one of them
        # is a consonant and the rule is about consonants. Whether a stem ends in a vowel is the
        # coarser claim, and where it accounts for the same split it is the better one. Where it
        # does not — vowel harmony, where every stem ends in a vowel and the split is *which* —
        # it separates nothing and the finer descriptions are still there.
        for name, pick in (("class", lambda stem: ("V" if stem and stem[-1] in _VOWELS
                                                   else "C") if stem else ""),
                           ("vowel", self._last_vowel),
                           ("final", lambda stem: stem[-1] if stem else "")):
            found = sets(pick)
            if not separates(found) or not all(found.values()):
                continue
            conditioned = [Rule(feature=rule.feature, process=rule.process,
                                condition=frozenset(found[rule.process.key]), on=name,
                                support=rule.support)
                           for rule in family]
            self.rules = ([rule for rule in self.rules if rule.feature != feature]
                          + conditioned)
            return

    def bind(self, base: str, inflected: str, feature: str) -> bool:
        """Bind the process relating ``base`` to ``inflected`` to ``feature``.

        Every process that could relate the two is proposed, each is checked against the rest of
        the vocabulary, and the one the vocabulary actually supports wins — on support first and
        on how much of the word it accounts for second, so a bigger claim never wins on a tie it
        did not earn.

        Refuses — and records the refusal — when nothing reaches ``min_stems``. The alternative is
        a "rule" whose entire evidence is the pair that named it, which is exactly what the wug
        test exists to tell apart from a rule. A refused pair is kept as an **irregular** and
        never generalises to a stem it was not shown on.
        """
        base, inflected = str(base or "").strip().lower(), str(inflected or "").strip().lower()
        feature = str(feature or "").strip()
        if not base or not inflected or not feature or inflected == base:
            return False
        best: Optional[Tuple[int, int, Process, List[str]]] = None
        for process in _candidates(base, inflected):
            support, witnesses = self._corroborate(process, base)
            if support + 1 < self.min_stems:
                continue
            scored = (support, process.size, process, witnesses)
            if best is None or (scored[0], scored[1]) > (best[0], best[1]):
                best = scored
        if best is None:
            self.irregular[(base, feature)] = inflected
            self.refused.append((base, inflected,
                                 f"no process reached {self.min_stems} stems in the vocabulary"))
            return False
        support, _size, process, witnesses = best
        self._witness[(feature, process.key)] = witnesses + [base]
        self.rules = [rule for rule in self.rules
                      if not (rule.feature == feature and rule.process.key == process.key)]
        self.rules.append(Rule(feature=feature, process=process, support=support + 1))
        self.rules.sort(key=lambda rule: (-rule.support, rule.feature))
        self._condition_rules(feature)
        # Keep the old flat view in step for the concatenative cases, so `analyse` and the word
        # classes keep seeing what they always saw.
        if process.kind in ("suffix", "prefix"):
            for index, affix in enumerate(self.affixes):
                if affix.form == process.a and affix.side == process.kind:
                    self.affixes[index] = Affix(form=process.a, side=process.kind,
                                                stems=affix.stems, feature=feature)
                    break
            else:
                self.affixes.append(Affix(form=process.a, side=process.kind,
                                          stems=support + 1, feature=feature))
                self.affixes.sort(key=lambda a: (-len(a.form), -a.stems, a.form))
        return True

    # -- use ---------------------------------------------------------------- #
    def analyse(self, word: str) -> Segment:
        """Split a word into stem and affix, or report it unanalysed."""
        word = str(word or "").strip().lower()
        if not word:
            return Segment()
        # A memorised pair works in both directions. An irregular was bound because no process
        # corroborated it, and a learner who could produce `went` and not recognise it would be
        # holding half a fact.
        for (base, feature), form in self.irregular.items():
            if form == word:
                return Segment(word=word, stem=base, affix=form, side="irregular",
                               feature=feature)

        # Bound rules first: a process a lesson gave a meaning to is a better account of a word
        # than a bare shape that happens to fit, and it is the only one that can name a feature.
        ordered = sorted(self.rules, key=lambda rule: (-rule.process.size, -rule.support))
        found: List[Segment] = []
        for rule in ordered:
            stem = rule.process.strip(word)
            if stem and len(stem) >= 2 and rule.fits(stem):
                found.append(Segment(word=word, stem=stem,
                                     affix=rule.process.a or rule.process.kind,
                                     side=rule.process.kind, feature=rule.feature))
        for affix in self.affixes:
            stem = affix.strip(word)
            if stem and len(stem) >= 2:
                found.append(Segment(word=word, stem=stem, affix=affix.form, side=affix.side,
                                     feature=affix.feature))

        # **An analysis whose stem is a word she has met beats one whose stem is not.** Without
        # that, a rule fires wherever its letters happen to fall: `seed` was read as `se` + past,
        # `teacher` as `teach` + comparative, `letter` as `lett` + comparative. Every one of those
        # is the ending doing what it was taught to do, in a word that does not contain it — and
        # the evidence that says so is simply that `se`, `teach` and `lett` are not words and
        # `seed`, `teacher` and `letter` are.
        for segment in found:
            if segment.stem in self.vocabulary:
                return segment
        # No known stem. If the whole word is one she has met, it is a word rather than a form —
        # and if it is not, an ending is the only account of it there is, which is what makes an
        # unseen inflection of an unseen stem readable at all.
        if word in self.vocabulary:
            return Segment(word=word, stem=word)
        return found[0] if found else Segment(word=word, stem=word)

    def stem_of(self, word: str) -> str:
        """The stem, but only where the affix carries a **bound** feature.

        Stripping an affix she has induced and does not understand would silently merge two
        entities whose difference she cannot name, so an unbound affix leaves the word alone.
        """
        segment = self.analyse(word)
        return segment.stem if segment.feature else segment.word

    def feature_of(self, word: str) -> str:
        return self.analyse(word).feature

    def inflect(self, stem: str, feature: str) -> str:
        """The surface for ``stem`` under ``feature``, or ``""`` where nothing binds it.

        The empty string is an abstention and is meant to be scored as one. A morphology that
        answered every request would have nothing to say about the difference between a rule it
        learned and a rule nobody taught it.
        """
        stem = str(stem or "").strip().lower()
        feature = str(feature or "").strip()
        if not stem or not feature:
            return ""
        memorised = self.irregular.get((stem, feature))
        if memorised:
            return memorised
        family = [rule for rule in self.rules if rule.feature == feature]
        # A rule whose condition this stem meets, best-supported first; then any unconditioned
        # rule. The order matters exactly once — under harmony, where two rules both exist and
        # only one of them is for this stem.
        for rule in sorted(family, key=lambda rule: (-len(rule.condition), -rule.support)):
            if rule.fits(stem):
                made = rule.process.apply(stem)
                if made:
                    return made
        return ""

    def features(self) -> List[str]:
        return sorted({rule.feature for rule in self.rules}
                      | {a.feature for a in self.affixes if a.feature})

    def stats(self) -> Dict[str, Any]:
        return {"vocabulary": len(self.vocabulary), "affixes": len(self.affixes),
                "bound": len(self.rules), "rules": [rule.to_dict() for rule in self.rules[:8]],
                "features": self.features(), "irregular": len(self.irregular),
                "refused": len(self.refused)}

    # -- persistence -------------------------------------------------------- #
    #: The most word types carried across a restart, most-heard first. A vocabulary is evidence
    #: for the affixes and it is not the affixes; past this many, one more hapax changes no
    #: induction and costs a sidecar that has to be read on every wake.
    max_kept = 8192

    def to_dict(self) -> Dict[str, Any]:
        """The vocabulary and the **bound** features, and nothing derived from them.

        The affixes themselves are not stored — :meth:`induce` re-runs on load and finds exactly
        what the vocabulary supports, which is the point: a stored affix could outlive the
        evidence for it, and an affix with no evidence is the one thing this class exists not to
        have. Only the feature *bindings* are written out, because those came from a lesson rather
        than from the corpus and re-counting would not recover them.
        """
        ranked = sorted(self.vocabulary.items(), key=lambda kv: (-kv[1], kv[0]))[:self.max_kept]
        return {"vocabulary": dict(ranked),
                "bound": [[a.form, a.side, a.feature] for a in self.affixes if a.feature],
                "rules": [rule.to_dict() for rule in self.rules],
                "irregular": [[base, feature, form]
                              for (base, feature), form in self.irregular.items()]}

    def load_dict(self, data: Dict[str, Any]) -> None:
        data = data or {}
        self.vocabulary = {str(word): int(count)
                           for word, count in (data.get("vocabulary") or {}).items()}
        self.irregular = {}
        for row in data.get("irregular") or []:
            if isinstance(row, (list, tuple)) and len(row) == 3:
                self.irregular[(str(row[0]), str(row[1]))] = str(row[2])
        self.induce()
        bound = {(str(row[0]), str(row[1])): str(row[2])
                 for row in (data.get("bound") or [])
                 if isinstance(row, (list, tuple)) and len(row) == 3}
        for index, affix in enumerate(self.affixes):
            feature = bound.get((affix.form, affix.side))
            if feature:
                self.affixes[index] = Affix(form=affix.form, side=affix.side,
                                            stems=affix.stems, feature=feature)
        self.rules = []
        for row in data.get("rules") or []:
            if not isinstance(row, dict):
                continue
            self.rules.append(Rule(feature=str(row.get("feature") or ""),
                                   process=Process.from_dict(row.get("process") or {}),
                                   condition=frozenset(row.get("condition") or ()),
                                   on=str(row.get("on") or "vowel"),
                                   support=int(row.get("support") or 0)))


# --------------------------------------------------------------------------- #
# word classes — discovered from where words occur, and named nothing
# --------------------------------------------------------------------------- #

class Lexicon:
    """Distributional word classes: words occurring in the same places are the same kind.

    The context of a word is the multiset of ``(side, neighbour)`` pairs it was seen with, with
    ``#`` standing for the sentence boundary. Two words fall in one class when the cosine between
    those vectors clears ``threshold``. The clustering is greedy and single-pass over words sorted
    by how much evidence each carries, so the classes are deterministic given the same corpus.

    **A neighbour seen once is not a context, it is a word.** Every content word in a fresh corpus
    is unique, so contexts built from raw neighbours share nothing and every word lands in a class
    of its own — measured, one class per word and nothing learned. So a neighbour occurring fewer
    than ``context_min`` times is bucketed as ``*``, and what is left as a feature is the material
    that actually recurs: the boundary, the particles, and whichever words the corpus repeats.
    That is the frequent/open split :mod:`nyxara.njp.semantics` makes by hand, arrived at here by
    counting instead of by a table — which is why it works in a language nobody wrote a table for.

    **The classes have no names, and that is the claim.** ``class:0`` is a set of words that
    behave alike. Calling it "noun" would assert a part-of-speech theory the evidence does not
    contain, and every other discovery layer here (``concepts``, ``core.represent``) refuses the
    same temptation for the same reason.
    """

    def __init__(self, *, threshold: float = 0.35, min_count: int = 2,
                 context_min: int = 2, passes: int = 2) -> None:
        self.threshold = float(threshold)
        self.min_count = max(1, int(min_count))
        self.context_min = max(1, int(context_min))
        self.passes = max(1, int(passes))
        self.counts: Dict[str, int] = {}
        self.contexts: Dict[str, Dict[str, int]] = {}
        self.classes: Dict[str, str] = {}
        self.members: Dict[str, List[str]] = {}

    def observe(self, tokens: Sequence[str]) -> None:
        """One sentence, already tokenised."""
        padded = ["#"] + [str(token).lower() for token in tokens] + ["#"]
        for index in range(1, len(padded) - 1):
            word = padded[index]
            self.counts[word] = self.counts.get(word, 0) + 1
            bag = self.contexts.setdefault(word, {})
            for key in (f"L:{padded[index - 1]}", f"R:{padded[index + 1]}"):
                bag[key] = bag.get(key, 0) + 1

    def observe_text(self, text: str) -> None:
        self.observe(tokenize_surface(text))

    @staticmethod
    def _cosine(left: Dict[str, int], right: Dict[str, int]) -> float:
        if not left or not right:
            return 0.0
        shared = set(left) & set(right)
        if not shared:
            return 0.0
        dot = sum(left[key] * right[key] for key in shared)
        norm = (math.sqrt(sum(value * value for value in left.values()))
                * math.sqrt(sum(value * value for value in right.values())))
        return dot / norm if norm else 0.0

    def _vectors(self, classes: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, int]]:
        """Contexts, with each neighbour named either by its word or by its **kind**.

        The first pass names it by its word, with the rare ones collapsed to ``*``. Every later
        pass names it by the class the previous pass put it in, which is the step that makes the
        induction bootstrap: two words whose neighbours are different words but the *same kind of
        word* have nothing in common in the first description and everything in common in the
        second.

        That is not a refinement, it is the difference between working and not working, and the
        measurement says so. In a clause ordered verb-subject-object the subject is the one
        position touching neither edge of the sentence, so two subjects sharing no verb and no
        object share **no context at all** — measured on one minted language, eight subject nouns
        in eight classes of one, and two words that behave identically judged different kinds. One
        further pass over kinds puts all eight together.
        """
        frequent = {word for word, count in self.counts.items() if count >= self.context_min}
        out: Dict[str, Dict[str, int]] = {}
        for word, bag in self.contexts.items():
            vector: Dict[str, int] = {}
            for key, count in bag.items():
                side, neighbour = key.split(":", 1)
                if neighbour == "#":
                    shown = "#"
                elif classes is not None:
                    shown = classes.get(neighbour, "*")
                else:
                    shown = neighbour if neighbour in frequent else "*"
                folded = f"{side}:{shown}"
                vector[folded] = vector.get(folded, 0) + count
            out[word] = vector
        return out

    def _cluster(self, vectors: Dict[str, Dict[str, int]]) -> Dict[str, str]:
        """Greedy single pass over words ordered by evidence. Deterministic given the corpus."""
        ranked = sorted((word for word, count in self.counts.items() if count >= self.min_count),
                        key=lambda word: (-self.counts[word], word))
        centroids: Dict[str, Dict[str, int]] = {}
        members: Dict[str, List[str]] = {}
        assigned: Dict[str, str] = {}
        for word in ranked:
            vector = vectors.get(word, {})
            best, best_score = "", 0.0
            for name, centroid in centroids.items():
                score = self._cosine(vector, centroid)
                if score > best_score:
                    best, best_score = name, score
            if best and best_score >= self.threshold:
                assigned[word] = best
                members[best].append(word)
                for key, value in vector.items():
                    centroids[best][key] = centroids[best].get(key, 0) + value
            else:
                name = f"class:{len(centroids)}"
                centroids[name] = dict(vector)
                assigned[word] = name
                members[name] = [word]
        self.members = members
        return assigned

    def induce(self) -> int:
        """Cluster, then re-cluster over neighbour kinds. Returns the number of classes found.

        ``passes`` is two and stops there deliberately. Each further pass describes a word by a
        description that was itself inferred, so the evidence gets thinner while the numbers look
        the same, and an induction run to a fixed point is one whose later passes are agreeing
        with themselves rather than with the corpus.
        """
        self.classes, self.members = {}, {}
        assigned = self._cluster(self._vectors())
        for _ in range(max(0, self.passes - 1)):
            assigned = self._cluster(self._vectors(assigned))
        self.classes = assigned
        return len(self.members)

    def class_of(self, word: str) -> str:
        return self.classes.get(str(word or "").strip().lower(), "")

    def same_class(self, left: str, right: str) -> Optional[bool]:
        """``True`` / ``False``, or ``None`` where she has no grounds — an unseen word.

        Three outcomes rather than two, for the reason every other organ here keeps three: a word
        she has never met is not a word that belongs to a different class, and answering "no"
        would be an assertion made out of an absence.
        """
        first, second = self.class_of(left), self.class_of(right)
        if not first or not second:
            return None
        return first == second

    def stats(self) -> Dict[str, Any]:
        return {"words": len(self.counts), "classified": len(self.classes),
                "classes": len(self.members)}

    # -- persistence -------------------------------------------------------- #
    #: The most words whose contexts are carried across a restart, most-heard first. The tail of
    #: a corpus is words met once, and :attr:`min_count` already refuses to classify those.
    max_kept = 4096

    def to_dict(self) -> Dict[str, Any]:
        """The contexts, which are the evidence. The classes are re-derived on load.

        Same discipline as :meth:`Morphology.to_dict` and for the same reason: a stored class
        assignment could disagree with the contexts stored beside it, and nothing downstream would
        be able to tell which of the two was right.
        """
        keep = [word for word, _count in
                sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))[:self.max_kept]]
        return {"counts": {word: self.counts[word] for word in keep},
                "contexts": {word: self.contexts.get(word, {}) for word in keep}}

    def load_dict(self, data: Dict[str, Any]) -> None:
        data = data or {}
        self.counts = {str(word): int(count)
                       for word, count in (data.get("counts") or {}).items()}
        self.contexts = {}
        for word, bag in (data.get("contexts") or {}).items():
            if isinstance(bag, dict):
                self.contexts[str(word)] = {str(key): int(value) for key, value in bag.items()}
        self.induce()


# --------------------------------------------------------------------------- #
# constructions — a sentence shape generalised from a demonstration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Slot:
    """A position in a construction, with whatever fixed material clings to its filler.

    ``prefix`` and ``suffix`` are the case marker, the plural, the tense — anything the
    demonstration attached to the content word. Keeping them **on the slot** rather than
    discarding them is what lets two constructions differing only by a verb ending stay two
    constructions, instead of colliding into one that reads both of them wrong.
    """

    role: str = ""
    prefix: str = ""
    suffix: str = ""
    #: How many tokens this slot has actually been shown holding.
    #:
    #: A **learned** property, not a global constant, and that is the whole of the argument for
    #: it. The constant was one, because at three a construction of bare slots matched a sentence
    #: with a word too many by letting one slot swallow two — a grammar that cannot refuse. But
    #: one makes *"the big dog"* unreadable, which rules out most of English.
    #:
    #: Learned, both hold at once: a slot nobody ever filled with two words stays at one and
    #: refuses the extra token exactly as before, and a slot demonstrated with a two-word phrase
    #: accepts two words *there*. The ceiling on what can be demonstrated is :data:`MAX_SPAN`.
    widths: Tuple[int, ...] = (1,)

    @property
    def bound_chars(self) -> int:
        return len(self.prefix) + len(self.suffix)

    def wrap(self, filler: str) -> str:
        return f"{self.prefix}{filler}{self.suffix}"

    def unwrap(self, token: str) -> str:
        """The filler inside ``token``, or ``""`` if the fixed material is not there."""
        if len(token) <= self.bound_chars:
            return ""
        if not token.startswith(self.prefix) or not token.endswith(self.suffix):
            return ""
        inner = token[len(self.prefix): len(token) - len(self.suffix)] \
            if self.suffix else token[len(self.prefix):]
        return inner if len(inner) >= 2 else ""


@dataclass(frozen=True)
class Marker:
    """A piece of fixed material that carries a feature, and can be left off.

    This is the axis that reaches a cell no lesson contained. Shown singular-present,
    plural-present and singular-past, a grammar made of whole sentence shapes holds three shapes
    and has no fourth — plural-past is a *combination*, and a list of shapes cannot combine.
    Measured before this existed: **0 / 8**, and not as a refusal either, because the
    singular-past shape matched the plural-past sentence and read the plural marker as part of the
    subject's name.

    A marker says: *this affix, on this slot, means this feature has this value.* Markers of
    different **dimensions** — different ``name`` — attach independently, so three demonstrated
    cells license the fourth without anybody demonstrating it. Markers of the same dimension are
    alternatives, because a word cannot be singular and plural at once.

    ``slot`` is an index into the construction's pattern, or ``-1`` for a feature the shape
    carries with no material at all — which is what a *pro-dropped* subject is: the sentence never
    contains it, and the ending on the verb is what says who it was.
    """

    slot: int = -1
    prefix: str = ""
    suffix: str = ""
    name: str = ""
    value: str = ""

    @property
    def material(self) -> int:
        return len(self.prefix) + len(self.suffix)

    def to_dict(self) -> Dict[str, Any]:
        return {"slot": self.slot, "prefix": self.prefix, "suffix": self.suffix,
                "name": self.name, "value": self.value}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Marker":
        return Marker(slot=int(data.get("slot", -1)), prefix=str(data.get("prefix") or ""),
                      suffix=str(data.get("suffix") or ""), name=str(data.get("name") or ""),
                      value=str(data.get("value") or ""))


#: Names a construction may carry that live on the Meaning itself rather than among its roles.
_MEANING_FEATURES = ("temporal", "modality", "focus", "negated")


def _pool_fillers(group: Sequence["Construction"]) -> Dict[str, FrozenSet[str]]:
    """What has stood in each role, across a whole family of shapes."""
    pooled: Dict[str, Set[str]] = {}
    for construction in group:
        for role, values in construction.fillers.items():
            pooled.setdefault(role, set()).update(values)
    return {role: frozenset(values) for role, values in pooled.items()}


def _stamp(item: Any) -> Any:
    """One pattern item, as something hashable that says exactly what it is."""
    if isinstance(item, str):
        return item
    if isinstance(item, Joint):
        return ("joint", item.left.role, item.left.prefix, item.left.suffix,
                item.right.role, item.right.prefix, item.right.suffix,
                item.prefix, item.suffix)
    return ("slot", item.role, item.prefix, item.suffix)


def _render_item(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, Joint):
        return f"{item.prefix}<{item.left.role}+{item.right.role}>{item.suffix}"
    return f"{item.prefix}<{item.role}>{item.suffix}"


def _orderings(markers: Sequence[Marker]) -> List[List[Marker]]:
    """The orders these markers could stack in, where two of them land on one slot.

    Only the ones that could differ: markers on different slots commute, so the ordering question
    is per slot, and with two markers on a slot there are two answers. Bounded hard, because a
    paradigm whose stacking order is a free variable in six places is not a paradigm.
    """
    import itertools
    by_slot: Dict[int, List[Marker]] = {}
    for marker in markers:
        by_slot.setdefault(marker.slot, []).append(marker)
    contested = [slot for slot, group in by_slot.items() if slot >= 0 and len(group) > 1]
    if not contested or len(contested) > 2:
        return [list(markers)]
    out: List[List[Marker]] = []
    slot = contested[0]
    for order in itertools.permutations(by_slot[slot]):
        arranged: List[Marker] = []
        for other, group in by_slot.items():
            arranged.extend(order if other == slot else group)
        out.append(arranged)
        if len(out) >= 6:
            break
    return out


@dataclass(frozen=True)
class Joint:
    """One token holding two roles, written with nothing between them.

    A compound. ``benuapseliao`` is a modifier and a head with no space and no marker, and every
    other pattern item here consumes whole tokens — so before this existed, a language that writes
    two of its arguments as one word scored **0 / 8**, as refusals, because there was no shape in
    which a token could be two things.

    Deliberately two and not many: the split point is what has to be learned, and with three parts
    there are two of them and the evidence for each is halved. What makes this safe is that both
    halves have to be filled by roles the demonstration named — a joint never invents a boundary
    to make a sentence fit, it reproduces one a lesson showed it.
    """

    left: Slot = field(default_factory=Slot)
    right: Slot = field(default_factory=Slot)
    prefix: str = ""
    suffix: str = ""

    @property
    def roles(self) -> Tuple[str, str]:
        return (self.left.role, self.right.role)

    @property
    def bound_chars(self) -> int:
        return len(self.prefix) + len(self.suffix) + self.left.bound_chars \
            + self.right.bound_chars

    def splits(self, token: str) -> List[Tuple[str, str]]:
        """Every way this joint reads one token. Usually one; never zero silently."""
        if not token.startswith(self.prefix) or not token.endswith(self.suffix):
            return []
        inner = token[len(self.prefix): len(token) - len(self.suffix)] if self.suffix \
            else token[len(self.prefix):]
        out: List[Tuple[str, str]] = []
        for cut in range(2, len(inner) - 1):
            head = self.left.unwrap(inner[:cut])
            tail = self.right.unwrap(inner[cut:])
            if head and tail:
                out.append((head, tail))
        return out

    def wrap(self, left: str, right: str) -> str:
        return f"{self.prefix}{self.left.wrap(left)}{self.right.wrap(right)}{self.suffix}"


@dataclass
class Construction:
    """One learned sentence shape, and the record of how it has done.

    ``pattern`` is a tuple whose members are either a literal string or a :class:`Slot`. It
    matches a whole sentence and never part of one: a construction matching a *prefix* of a
    sentence would read "the cat sat on the mat" with the shape for "the cat sat" and lose the
    half that changes what it says.

    ``features`` are the things this shape says that no slot of it holds — a tense, a polarity, a
    number, a person. ``markers`` are the ones that can be *left off*, each with the material that
    carries it, and they are what make a shape a paradigm instead of a list. See :class:`Marker`.
    """

    tongue: str = ""
    pattern: Tuple[Any, ...] = ()
    kind: str = "assertion"
    negated: bool = False
    focus: str = ""
    modality: str = ""
    temporal: str = ""
    #: ``(name, value)`` pairs this shape asserts without a slot holding them, sorted.
    features: Tuple[Tuple[str, str], ...] = ()
    #: Optional material, each piece carrying one feature. See :class:`Marker`.
    markers: Tuple[Marker, ...] = ()
    #: Dimensions every demonstration of this shape marked. A variant missing one of these is
    #: not offered — no lesson showed the shape without it, so a sentence lacking it is not a
    #: sentence of this language and reading one as if it were would be inventing a reading.
    required: Tuple[str, ...] = ()
    #: What has actually stood in each role, as words. Evidence, never a gate: a word she has not
    #: met is not barred from a role, it just brings nothing to a tie.
    #:
    #: This is the axis for the sentences whose roles order alone cannot settle. Where a language
    #: says *"denied"* by moving the verb to the front, ``s v o`` and ``v s o`` are the same three
    #: tokens under two shapes that disagree, and she returned ``ambiguous`` to all of them —
    #: correctly, on the evidence she was using. The evidence she was **not** using is that the
    #: verbs of a corpus recur and its subjects do not, so the reading that puts a word she has
    #: seen as a verb in the verb's place is the better-supported one.
    fillers: Dict[str, FrozenSet[str]] = field(default_factory=dict)
    support: int = 0
    hits: int = 0
    misses: int = 0
    source: str = ""              #: one demonstration, kept for the report card only

    # -- shape -------------------------------------------------------------- #
    @property
    def skeleton(self) -> Tuple[Any, ...]:
        """The shape with every slot's fixed material stripped off.

        Two constructions with one skeleton are candidates for being one paradigm, and this is
        what :meth:`Grammar._factorise` groups on.
        """
        return (self.tongue, self.kind, self.focus,
                tuple(item if isinstance(item, str)
                      else (("joint",) + item.roles if isinstance(item, Joint)
                            else ("slot", item.role))
                      for item in self.pattern))

    @property
    def signature(self) -> Tuple[Any, ...]:
        """What makes two demonstrations the same shape."""
        return (self.tongue, self.kind, self.negated, self.focus, self.modality, self.temporal,
                self.features, tuple(_stamp(item) for item in self.pattern))

    @property
    def widths(self) -> Tuple[Tuple[int, ...], ...]:
        """Each slot's demonstrated widths, in order. Not part of the signature, on purpose:
        *"the dog runs"* and *"the big dog runs"* are one shape whose subject can be two words,
        not two shapes that happen to look alike."""
        return tuple(item.widths for item in self.pattern if isinstance(item, Slot))

    @property
    def shape(self) -> str:
        """The pattern as one readable line: ``<subject>gu <verb> nuza``."""
        return " ".join(_render_item(item) for item in self.pattern)

    @property
    def key(self) -> str:
        """:attr:`signature` as a string — the same identity, in something JSON can hold."""
        return (f"{self.kind}|{int(self.negated)}|{self.focus}|{self.temporal}|"
                f"{self.modality}|{self.shape}")

    @property
    def roles(self) -> Tuple[str, ...]:
        out: List[str] = []
        for item in self.pattern:
            if isinstance(item, Slot):
                out.append(item.role)
            elif isinstance(item, Joint):
                out.extend(item.roles)
        return tuple(out)

    @property
    def literals(self) -> int:
        return sum(1 for item in self.pattern if isinstance(item, str))

    @property
    def bound_chars(self) -> int:
        return sum(item.bound_chars for item in self.pattern
                   if isinstance(item, (Slot, Joint)))

    @property
    def record(self) -> float:
        """Held-out precision, ``0.5`` before there is any. Never a default that flatters."""
        seen = self.hits + self.misses
        return self.hits / seen if seen else 0.5

    # -- the paradigm this shape stands for --------------------------------- #
    def variants(self) -> List[Tuple[Tuple[Any, ...], Dict[str, str], int]]:
        """Every combination of markers this shape licenses, with the features each one asserts.

        One dimension at most once — a word is not singular and plural at the same time — and
        every combination of *different* dimensions, which is the whole point: three cells
        demonstrated, the fourth reachable because its two halves attach independently.

        Where two markers land on one slot their order was never demonstrated (each lesson showed
        one of them alone), so **both orders are offered** and the sentence decides. That is a
        genuine underdetermination rather than a gap, and offering both is what lets reading
        succeed on a stacking nobody showed her — while :meth:`Grammar.say`, which has to pick
        one, is the place where it costs something.
        """
        dimensions: Dict[str, List[Marker]] = {}
        for marker in self.markers:
            dimensions.setdefault(marker.name, []).append(marker)
        chosen: List[List[Marker]] = [[]]
        for name in sorted(dimensions):
            grown: List[List[Marker]] = []
            for existing in chosen:
                if name not in self.required:
                    grown.append(existing)
                for marker in dimensions[name]:
                    grown.append(existing + [marker])
            chosen = grown
            if len(chosen) > 64:                # a paradigm this wide is not one shape
                break
        out: List[Tuple[Tuple[Any, ...], Dict[str, str], int]] = []
        seen: Set[Tuple[Any, ...]] = set()
        for subset in chosen:
            for order in _orderings(subset):
                pattern = self._with(order)
                features = {marker.name: marker.value for marker in order}
                stamp = (tuple(_stamp(item) for item in pattern),
                         tuple(sorted(features.items())))
                if stamp in seen:
                    continue
                seen.add(stamp)
                out.append((pattern, features, sum(m.material for m in order)))
        return out

    def _with(self, markers: Sequence[Marker]) -> Tuple[Any, ...]:
        """The pattern with these markers' material wrapped onto their slots, in order."""
        pattern = list(self.pattern)
        for marker in markers:
            if marker.slot < 0 or marker.slot >= len(pattern):
                continue
            item = pattern[marker.slot]
            if not isinstance(item, Slot):
                continue
            pattern[marker.slot] = Slot(role=item.role,
                                        prefix=marker.prefix + item.prefix,
                                        suffix=item.suffix + marker.suffix,
                                        widths=item.widths)
        return tuple(pattern)

    # -- matching ----------------------------------------------------------- #
    def match(self, tokens: Sequence[str]) -> List[Dict[str, str]]:
        """Every way this construction reads ``tokens``, ignoring the features it also asserts."""
        return [fills for fills, _features, _material in self.readings(tokens)]

    def readings(self, tokens: Sequence[str],
                 resolve: Optional[Any] = None
                 ) -> List[Tuple[Dict[str, str], Dict[str, str], int]]:
        """Every way this construction reads ``tokens``, with the features each reading asserts.

        All readings are returned rather than the first, because *which* one is right is a
        question about evidence and the caller is what holds the evidence. A construction
        returning two readings of one sentence has found an ambiguity, and saying so is the
        honest outcome.
        """
        out: List[Tuple[Dict[str, str], Dict[str, str], int]] = []
        try:
            words = list(tokens)
            for pattern, features, material in self.variants():
                found: List[Dict[str, str]] = []
                self._walk(pattern, words, 0, 0, {}, found, resolve)
                for fills in found:
                    out.append((fills, features, material))
        except Exception:  # noqa: BLE001
            return []
        return out

    def _walk(self, pattern: Tuple[Any, ...], tokens: List[str], at: int, index: int,
              fills: Dict[str, str], out: List[Dict[str, str]],
              resolve: Optional[Any] = None) -> None:
        if len(out) >= 8:                       # an ambiguity this deep is already reported
            return
        if index == len(pattern):
            if at == len(tokens):
                out.append(dict(fills))
            return
        item = pattern[index]
        if isinstance(item, str):
            if at < len(tokens) and tokens[at] == item:
                self._walk(pattern, tokens, at + 1, index + 1, fills, out, resolve)
            return
        if isinstance(item, Joint):
            if at >= len(tokens):
                return
            for head, tail in item.splits(tokens[at]):
                fills[item.left.role], fills[item.right.role] = head, tail
                self._walk(pattern, tokens, at + 1, index + 1, fills, out, resolve)
                fills.pop(item.left.role, None)
                fills.pop(item.right.role, None)
            return
        remaining = len(pattern) - index - 1
        # What this slot has been shown holding, and nothing wider. Two further conditions cut it
        # back to one token, and they are different conditions:
        #
        # Fixed material: case marking a whole noun phrase is a real thing in real languages and
        # it is *not* something this matcher can see, so it is refused rather than guessed at.
        #
        # No literals anywhere in the pattern: an unanchored multi-token slot is unfalsifiable. A
        # construction of three bare slots would otherwise match a sentence of four words by
        # letting one slot swallow two of them. With at least one literal in the pattern the
        # phrase has something to end against, and then a span is a reading rather than a shrug.
        spans = (1,) if (item.bound_chars or not self.literals) else item.widths
        for span in spans:
            if at + span > len(tokens) - remaining:
                break
            if span == 1:
                filler = item.unwrap(tokens[at])
                if filler and resolve is not None:
                    filler = resolve(tokens[at], filler)
            else:
                filler = " ".join(tokens[at:at + span])
            if not filler:
                continue
            fills[item.role] = filler
            self._walk(pattern, tokens, at + span, index + 1, fills, out, resolve)
            fills.pop(item.role, None)

    # -- rendering ---------------------------------------------------------- #
    def render(self, fills: Dict[str, str],
               pattern: Optional[Tuple[Any, ...]] = None,
               repair: Optional[Any] = None) -> str:
        """Instantiate. Returns ``""`` when a slot has nothing to put in it.

        ``repair`` is the production counterpart of :meth:`Grammar._resolve`: a construction can
        only write the ending it was shown, and the ending is a fact about the language.
        """
        words: List[str] = []
        for item in (self.pattern if pattern is None else pattern):
            if isinstance(item, str):
                words.append(item)
                continue
            if isinstance(item, Joint):
                head = str(fills.get(item.left.role, "") or "").strip().lower()
                tail = str(fills.get(item.right.role, "") or "").strip().lower()
                if not head or not tail:
                    return ""
                words.append(item.wrap(head, tail))
                continue
            filler = str(fills.get(item.role, "") or "").strip().lower()
            if not filler:
                return ""
            written = item.wrap(filler)
            if repair is not None:
                written = repair(written, filler, item.prefix, item.suffix) or written
            words.append(written)
        return re.sub(r"\s+([?!])", r"\1", " ".join(words))

    def to_dict(self) -> Dict[str, Any]:
        return {"tongue": self.tongue, "shape": self.shape, "kind": self.kind,
                "negated": self.negated, "focus": self.focus, "temporal": self.temporal,
                "modality": self.modality, "support": self.support,
                "features": [list(pair) for pair in self.features],
                "markers": [marker.to_dict() for marker in self.markers],
                "hits": self.hits, "misses": self.misses, "record": round(self.record, 3)}


@dataclass(frozen=True)
class Demonstration:
    """A surface and what it meant. The only thing a teacher ever hands this module."""

    surface: str = ""
    meaning: Meaning = field(default_factory=Meaning)


@dataclass
class Reading:
    """A candidate parse and the evidence behind it, so the choice between two is inspectable."""

    meaning: Meaning = field(default_factory=Meaning)
    construction: Optional[Construction] = None
    score: float = 0.0
    anchor: float = 0.0
    #: Characters of **fixed material this sentence actually contained** — case markers, tense
    #: endings, the material of whichever markers matched. Distinct from anything the shape merely
    #: has: a paradigm carrying a past-tense marker matched a present-tense sentence with no
    #: material at all, and asking the *construction* whether it has markers said yes.
    material: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"meaning": self.meaning.to_dict(), "score": round(self.score, 4),
                "anchor": round(self.anchor, 4), "material": round(self.material, 4),
                "construction": self.construction.to_dict() if self.construction else None}


@dataclass
class Learned:
    """What one round of learning did, including everything it threw away and why."""

    tongue: str = ""
    kept: int = 0
    rejected: int = 0
    reasons: List[str] = field(default_factory=list)
    affixes: int = 0
    bound: int = 0
    classes: int = 0
    demonstrations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"tongue": self.tongue, "kept": self.kept, "rejected": self.rejected,
                "affixes": self.affixes, "bound": self.bound, "classes": self.classes,
                "demonstrations": self.demonstrations, "reasons": self.reasons[:6]}


# --------------------------------------------------------------------------- #
# the grammar — generalise, verify, read, and say only what reads back
# --------------------------------------------------------------------------- #

#: The fields of a :class:`~nyxara.njp.semantics.Meaning` a construction is answerable for, and
#: therefore the only ones a demonstration needs to carry across a restart. Confidence, frame and
#: tokens are all *outputs* of a reading and are deliberately not stored: a persisted confidence
#: would be re-loaded as evidence for the very grammar that produced it.
_MEANING_FIELDS = ("kind", "subject", "relation", "object", "verb", "focus", "temporal",
                   "modality")


def _meaning_to_dict(meaning: Meaning) -> Dict[str, Any]:
    out: Dict[str, Any] = {"negated": bool(meaning.negated)}
    for name in _MEANING_FIELDS:
        value = getattr(meaning, name, "")
        if value:
            out[name] = str(value)
    return out


def _meaning_from_dict(data: Dict[str, Any]) -> Meaning:
    meaning = Meaning(negated=bool(data.get("negated")))
    for name in _MEANING_FIELDS:
        setattr(meaning, name, str(data.get(name) or ""))
    meaning.kind = meaning.kind or "assertion"
    return meaning


def _fills_of(meaning: Meaning) -> Dict[str, str]:
    """Every role a meaning carries, whatever it is called and however many there are.

    The **verb** role is filled from ``relation`` — the lemma — rather than from ``verb``, the
    surface. That one choice is what makes tense visible to this module: whatever the surface
    carries beyond the lemma becomes fixed material on the slot, so *"glim"* and *"glimta"* end
    up in two constructions rather than in one that cannot tell a past from a present.

    Everything past the legacy three comes out of
    :attr:`~nyxara.njp.semantics.Meaning.roles` under whatever name a demonstration used. There
    is no list of permitted role names here and there is not going to be one: a role is whatever
    a teacher pointed at, exactly as a concept id is whatever the clustering found.
    """
    out: Dict[str, str] = {}
    for key, value in (getattr(meaning, "roles", None) or {}).items():
        text = " ".join(str(value or "").strip().lower().split())
        if text:
            out[str(key)] = text
    for role in LEGACY_ROLES:
        value = getattr(meaning, "relation" if role == "verb" else role, "")
        value = " ".join(str(value or "").strip().lower().split())
        if value:
            out[role] = value
    return out


def _write_fills(meaning: Meaning, fills: Dict[str, str]) -> None:
    """The inverse: put the roles back where each of them lives."""
    for role, value in fills.items():
        if role == "subject":
            meaning.subject = value
        elif role == "verb":
            meaning.relation = meaning.verb = value
        elif role == "object":
            meaning.object = value
        else:
            meaning.roles[role] = value


def _affixes_around(token: str, filler: str) -> Optional[Tuple[str, str]]:
    """Where ``filler`` sits inside ``token``, as ``(prefix, suffix)``. ``None`` if it does not."""
    at = token.find(filler)
    if at < 0:
        return None
    return token[:at], token[at + len(filler):]


class Grammar:
    """The constructions of one language: how they are learned, and how one is chosen.

    Learning is three steps and each of them can reject:

    #. **generalise** — one demonstration becomes one candidate shape, or nothing at all when the
       meaning it came with cannot be located in the surface it came with;
    #. **support** — a shape is kept only where two or more demonstrations produced it with
       different fillers, so a sentence cannot masquerade as a rule;
    #. **verify** — every kept shape is run back over the whole corpus, and one that reads any
       demonstration into a different meaning than it arrived with is dropped.
    """

    #: Demonstrations needed before a shape counts as a shape.
    min_support = 2

    def __init__(self, tongue: str = "en") -> None:
        self.tongue = str(tongue or "en")
        self.constructions: List[Construction] = []
        self.demonstrations: List[Demonstration] = []
        self.rejected: List[str] = []
        #: "Have I met this word?", supplied by the :class:`Tongue` that owns this grammar. Used
        #: only to break a tie between readings, never to reject one — a word she has not met is
        #: not a word, and a grammar that refused new vocabulary would refuse every real sentence.
        self.known: Optional[Any] = None
        #: "What is the stem of this word?", supplied by the :class:`Tongue` that owns this
        #: grammar. See :meth:`_resolve`.
        self.stem: Optional[Any] = None
        #: "Which word that I have met is this stem, written this way?" See :meth:`_repair`.
        self.realise: Optional[Any] = None

    # -- teaching ----------------------------------------------------------- #
    def show(self, surface: str, meaning: Meaning) -> bool:
        """Record one demonstration. Nothing is generalised until :meth:`learn` runs."""
        if not str(surface or "").strip() or meaning is None:
            return False
        self.demonstrations.append(Demonstration(surface=str(surface), meaning=meaning))
        return True

    def generalise(self, surface: str, meaning: Meaning) -> Optional[Construction]:
        """One demonstration → one candidate shape, or ``None``.

        Content words become slots; **everything else stays**. That asymmetry is the whole of the
        generalisation: word order, particles, the negator and the case markers are not noise
        around the meaning, they are how the meaning was carried, and a shape that dropped them
        would match sentences that mean something else.
        """
        tokens = tokenize_surface(surface)
        fills = _fills_of(meaning)
        if not tokens or not fills:
            return None
        # A role the surface does not contain is not a mislabelled lesson — it is a **feature**.
        # "these are plural" is said by an ending, not by a word, and demanding that every role
        # appear as material is what made every plural demonstration unlearnable: measured, all
        # five plural lessons rejected, and the singular-past shape then read the plural-past
        # sentence with the ending inside the subject's name.
        #
        # Which roles are which is decided by looking, not by a list of feature names — a role is
        # a feature here exactly when this sentence does not contain it.
        placed = self._place(tokens, fills)
        joints: Dict[int, Joint] = {}
        features: Dict[str, str] = {}
        if placed is None:
            # Before deciding a role is a feature, ask whether two of them are sharing a token.
            # A compound writes its two halves with nothing between them, and a placer that only
            # ever consumes whole tokens reads that as one word it cannot account for.
            found = self._place_joined(tokens, fills)
            if found is not None:
                placed, joints = found
        if placed is None:
            spoken = {role: value for role, value in fills.items()
                      if self._locatable(tokens, value)}
            if not spoken or spoken == fills:
                return None
            features = {role: value for role, value in fills.items() if role not in spoken}
            placed = self._place(tokens, spoken)
            if placed is None:
                return None
        pattern: List[Any] = []
        index = 0
        while index < len(tokens):
            here = placed.get(index)
            if here is None:
                pattern.append(tokens[index])
                index += 1
                continue
            role, span, prefix, suffix = here
            joint = joints.get(index)
            pattern.append(joint if joint is not None
                           else Slot(role=role, prefix=prefix, suffix=suffix,
                                     widths=(span,)))
            index += span
        return Construction(
            tongue=self.tongue, pattern=tuple(pattern),
            kind=meaning.kind or "assertion", negated=bool(meaning.negated),
            focus=meaning.focus or "", modality=meaning.modality or "",
            temporal=meaning.temporal or "",
            features=tuple(sorted(features.items())),
            support=1, source=" ".join(tokens))

    def _place_joined(self, tokens: Sequence[str], fills: Dict[str, str]
                      ) -> Optional[Tuple[Dict[int, Tuple[str, int, str, str]], Dict[int, Joint]]]:
        """Try reading one token as two roles written together, then place the rest normally.

        A boundary is only ever taken from a demonstration: the two halves must be values the
        lesson named, in the order the token has them, and together with whatever fixed material
        surrounds them they must account for the **whole** token. A joint never invents a split to
        make a sentence fit.
        """
        singles = {role: value for role, value in fills.items() if " " not in value}
        for index, token in enumerate(tokens):
            for left, head in singles.items():
                if head not in token:
                    continue
                for right, tail in singles.items():
                    if right == left or tail not in token:
                        continue
                    at = token.find(head)
                    then = token.find(tail, at + len(head))
                    if at < 0 or then != at + len(head):
                        continue
                    joint = Joint(left=Slot(role=left), right=Slot(role=right),
                                  prefix=token[:at], suffix=token[then + len(tail):])
                    rest = {role: value for role, value in fills.items()
                            if role not in (left, right)}
                    placed = self._place(tokens, rest, blocked={index})
                    if placed is None:
                        continue
                    placed[index] = (left, 1, joint.prefix, joint.suffix)
                    return placed, {index: joint}
        return None

    @staticmethod
    def _locatable(tokens: Sequence[str], value: str) -> bool:
        """Is this value anywhere in this sentence at all?"""
        if " " in value:
            return value in " ".join(tokens)
        return any(value in token for token in tokens)

    def _place(self, tokens: Sequence[str], fills: Dict[str, str], *,
               blocked: Optional[Set[int]] = None
               ) -> Optional[Dict[int, Tuple[str, int, str, str]]]:
        """Assign every role to a non-overlapping span of ``tokens``, or refuse.

        Refusal is the right answer more often than it looks. A demonstration whose subject does
        not appear in its own surface is not a hard sentence, it is a **mislabelled** one, and
        generalising it would mint a shape that reads every sentence of that form into whatever
        the mislabelling said.
        """
        options: Dict[str, List[Tuple[int, int, str, str]]] = {}
        held = set(blocked or ())
        for role, filler in fills.items():
            found: List[Tuple[int, int, str, str]] = []
            for start in range(len(tokens)):
                if start in held:
                    continue
                for span in range(1, MAX_SPAN + 1):
                    if start + span > len(tokens):
                        break
                    if span == 1:
                        around = _affixes_around(tokens[start], filler)
                        if around is not None:
                            found.append((start, 1, around[0], around[1]))
                    elif " ".join(tokens[start:start + span]) == filler:
                        found.append((start, span, "", ""))
            if not found:
                return None
            # The tightest wrapping first: a filler that *is* the token beats one that had to
            # explain away four characters of what it did not cover.
            found.sort(key=lambda item: (len(item[2]) + len(item[3]), item[1], item[0]))
            options[role] = found
        roles = sorted(options, key=lambda role: len(options[role]))
        chosen: Dict[int, Tuple[str, int, str, str]] = {}
        used: Set[int] = set(held)

        def assign(depth: int) -> bool:
            if depth == len(roles):
                return True
            role = roles[depth]
            for start, span, prefix, suffix in options[role]:
                positions = set(range(start, start + span))
                if positions & used:
                    continue
                used.update(positions)
                chosen[start] = (role, span, prefix, suffix)
                if assign(depth + 1):
                    return True
                used.difference_update(positions)
                chosen.pop(start, None)
            return False

        return chosen if assign(0) else None

    def learn(self) -> Learned:
        """Generalise every demonstration, keep what is supported, drop what contradicts."""
        report = Learned(tongue=self.tongue, demonstrations=len(self.demonstrations))
        candidates: Dict[Tuple[Any, ...], List[Tuple[Construction, Dict[str, str]]]] = {}
        for shown in self.demonstrations:
            candidate = self.generalise(shown.surface, shown.meaning)
            if candidate is None:
                report.rejected += 1
                report.reasons.append(f"{shown.surface!r}: its meaning is not in its surface")
                continue
            candidates.setdefault(candidate.signature, []).append(
                (candidate, _fills_of(shown.meaning)))
        # How much evidence there is for each *skeleton*, as distinct from each exact shape. Two
        # shapes that differ only in one slot's material are two **allomorphs** of one shape, and
        # the support for the shape is the support for the family.
        #
        # Without this, romanised Hindi learned one construction out of twelve. `khaata` is
        # `kha` + `ata` because the stem ends in a vowel and `padhta` is `padh` + `ta` because it
        # does not — one ending, two shapes — so every allomorph arrived with a single
        # demonstration and every one of them was rejected as "a sentence, not a shape". The same
        # course in Devanagari scored 12/12, because there the matra rides inside the stem's own
        # syllable and the ending is `-ता` either way. The grammar was never the problem; the
        # orthography was showing something the grammar had no way to count.
        family: Dict[Tuple[Any, ...], int] = {}
        family_fills: Dict[Tuple[Any, ...], List[Dict[str, str]]] = {}
        for signature, group in candidates.items():
            key = (group[0][0].skeleton, group[0][0].features, group[0][0].negated,
                   group[0][0].temporal, group[0][0].modality)
            family[key] = family.get(key, 0) + len(group)
            family_fills.setdefault(key, []).extend(item[1] for item in group)

        kept: List[Construction] = []
        for signature, group in candidates.items():
            constructions = [item[0] for item in group]
            fillings = [item[1] for item in group]
            key = (constructions[0].skeleton, constructions[0].features,
                   constructions[0].negated, constructions[0].temporal,
                   constructions[0].modality)
            related, related_fills = family.get(key, 0), family_fills.get(key, fillings)
            if len(group) < self.min_support and related < self.min_support:
                report.rejected += 1
                report.reasons.append(
                    f"{constructions[0].source!r}: one demonstration is a sentence, not a shape")
                continue
            if not self._varies(fillings) and not self._varies(related_fills):
                report.rejected += 1
                report.reasons.append(
                    f"{constructions[0].source!r}: {len(group)} demonstrations, identical fillers")
                continue
            head = constructions[0]
            head.support = len(group)
            # Every width any member was demonstrated at, on the slot it was demonstrated on.
            # This is where "the dog runs" and "the big dog runs" become one shape rather than
            # two: the signature already ignores width, so they arrive in the same group, and
            # what the group knows is the union of what its members were shown.
            widths: Dict[int, Set[int]] = {}
            for construction in constructions:
                for index, item in enumerate(construction.pattern):
                    if isinstance(item, Slot):
                        widths.setdefault(index, set()).update(item.widths)
            head.pattern = tuple(
                Slot(role=item.role, prefix=item.prefix, suffix=item.suffix,
                     widths=tuple(sorted(widths.get(index, item.widths))))
                if isinstance(item, Slot) else item
                for index, item in enumerate(head.pattern))
            seen: Dict[str, Set[str]] = {}
            for filling in fillings:
                for role, value in filling.items():
                    seen.setdefault(role, set()).add(value)
            head.fillers = {role: frozenset(values) for role, values in seen.items()}
            kept.append(head)
        kept = self._factorise(kept, report)
        kept = self._licence_order(kept, report)
        previous = {c.signature: c for c in self.constructions}
        for construction in kept:
            older = previous.get(construction.signature)
            if older is not None:
                construction.support = max(construction.support, older.support)
                construction.hits, construction.misses = older.hits, older.misses
        self.constructions = kept
        self._verify(report)
        report.kept = len(self.constructions)
        self.rejected = list(report.reasons)
        return report

    # -- the paradigm axis --------------------------------------------------- #
    @staticmethod
    def _feature_map(construction: Construction) -> Dict[str, str]:
        """Everything a shape asserts that no slot of it holds, in one dict."""
        out = {"temporal": construction.temporal or "", "modality": construction.modality or "",
               "negated": "true" if construction.negated else ""}
        out.update(dict(construction.features))
        return {name: value for name, value in out.items() if value}

    @staticmethod
    def _extends(item: Slot, base: Tuple[str, str]) -> bool:
        """Is this slot's material the base's material with more added outside it?"""
        return item.prefix.endswith(base[0]) and item.suffix.startswith(base[1])

    @staticmethod
    def _common(values: Sequence[str], *, end: bool) -> str:
        """The longest string every one of these ends with (or starts with)."""
        if not values:
            return ""
        shared = values[0]
        for value in values[1:]:
            while shared and not (value.endswith(shared) if end else value.startswith(shared)):
                shared = shared[1:] if end else shared[:-1]
            if not shared:
                break
        return shared

    def _factorise(self, kept: List[Construction], report: "Learned") -> List[Construction]:
        """Turn a family of near-identical shapes into one shape with detachable markers.

        This is the axis, and it is the difference between a paradigm and a list. Shapes that
        share a skeleton and differ in *one* piece of material carrying *one* feature are not
        three facts about a language, they are one fact and two markers — and two markers of
        different dimensions attach independently, which is how a cell nobody demonstrated becomes
        reachable.

        It refuses wherever the evidence does not separate the pieces: a member differing in two
        features at once, or in material on two slots, leaves the family unfactorised, because
        which piece carries which feature is then a guess. And a dimension **every** member of the
        family marks is recorded as *required* — no lesson showed the shape without it, so a
        sentence lacking it is not a sentence of this language and must not be read as one.
        """
        groups: Dict[Tuple[Any, ...], List[Construction]] = {}
        for construction in kept:
            groups.setdefault(construction.skeleton, []).append(construction)
        out: List[Construction] = []
        for group in groups.values():
            if len(group) < 2:
                out.extend(group)
                continue
            merged = self._merge(group)
            if merged is None:
                out.extend(group)
                continue
            report.reasons.append(
                f"{merged.shape!r}: {len(group)} shapes folded into one paradigm with "
                f"{len(merged.markers)} marker(s)")
            out.append(merged)
        return out

    def _merge(self, group: Sequence[Construction]) -> Optional[Construction]:
        slots = [index for index, item in enumerate(group[0].pattern) if isinstance(item, Slot)]
        features = [self._feature_map(c) for c in group]

        # The base is the **least marked** member, not the intersection of all of them. With the
        # intersection, a family whose every member says something — three verbs each marking a
        # different combination of person — has no member that is plain, so every one of them
        # differs from the base in two things at once and the family is refused. Measured, that
        # is a verb marking both its arguments: 0/8, and the shape that would have read it was
        # thrown away for being ambiguous about which ending meant what.
        order = sorted(range(len(group)),
                       key=lambda i: (len(features[i]),
                                      sum(item.bound_chars for item in group[i].pattern
                                          if isinstance(item, Slot))))
        base_at = order[0]
        shared: Dict[int, Tuple[str, str]] = {
            index: (group[base_at].pattern[index].prefix, group[base_at].pattern[index].suffix)
            for index in slots}
        base_features = dict(features[base_at])
        if not all(self._extends(group[i].pattern[index], shared[index])
                   for i in range(len(group)) for index in slots):
            # The members do not extend one base — two different endings in one place, neither of
            # them the plain form. Then the shared part is whatever they all have in common, and
            # every member is a marker.
            shared = {index: (self._common([c.pattern[index].prefix for c in group], end=False),
                              self._common([c.pattern[index].suffix for c in group], end=True))
                      for index in slots}
            base_features = {name: value for name, value in features[0].items()
                             if all(other.get(name) == value for other in features[1:])}

        markers: List[Marker] = []
        dimensions_seen: List[Set[str]] = []
        for construction, feature_map in zip(group, features):
            delta = {name: value for name, value in feature_map.items()
                     if base_features.get(name) != value}
            extras: List[Tuple[int, str, str]] = []
            for index in slots:
                item = construction.pattern[index]
                head = item.prefix[: len(item.prefix) - len(shared[index][0])]
                tail = item.suffix[len(shared[index][1]):]
                if head or tail:
                    extras.append((index, head, tail))
            if not delta and not extras:
                dimensions_seen.append(set())
                continue
            if len(delta) != 1 or len(extras) > 1:
                return None                     # which piece carries which feature is a guess
            (name, value), = delta.items()
            index, head, tail = extras[0] if extras else (-1, "", "")
            markers.append(Marker(slot=index, prefix=head, suffix=tail, name=name, value=value))
            dimensions_seen.append({name})
        if not markers:
            return None
        # A dimension every single member marks is one no lesson ever showed the shape without.
        dimensions = {marker.name for marker in markers}
        required = tuple(sorted(name for name in dimensions
                                if all(name in seen for seen in dimensions_seen)))
        pattern = list(group[0].pattern)
        for index in slots:
            item = pattern[index]
            spans = {width for member in group for width in member.pattern[index].widths}
            pattern[index] = Slot(role=item.role, prefix=shared[index][0],
                                  suffix=shared[index][1], widths=tuple(sorted(spans)))
        head = group[0]
        return Construction(
            tongue=head.tongue, pattern=tuple(pattern), kind=head.kind, focus=head.focus,
            negated=base_features.get("negated") == "true",
            temporal=base_features.get("temporal", ""),
            modality=base_features.get("modality", ""),
            features=tuple(sorted((name, value) for name, value in base_features.items()
                                  if name not in _MEANING_FEATURES)),
            markers=tuple(markers),
            required=required,
            fillers=_pool_fillers(group),
            support=sum(c.support for c in group),
            source=head.source)

    # -- the free-order axis -------------------------------------------------- #
    #: The most slots a free-order family may have. Four, because the orders are enumerated and
    #: five slots is a hundred and twenty shapes for one sentence — at which point the search is
    #: paying more than the generalisation is worth.
    max_free_slots = 4

    def _licence_order(self, kept: List[Construction],
                       report: "Learned") -> List[Construction]:
        """Where case says the roles, order does not have to, and every order is licensed.

        Shown a case-marked language in two of its six orders, a grammar made of shapes holds two
        and refuses the other four — measured, **0 / 8** on exactly the orders that were not
        demonstrated, all of them refusals. But the case markers say the roles unambiguously in
        all six, and *that* is a property of the family rather than of any one member of it.

        The precondition is what keeps this from destroying refusal, and it is checked rather than
        assumed: every slot must carry material **distinct** from every other slot's, and at most
        one may carry none. Then a token names its own role and the order genuinely cannot change
        the meaning. A family with two unmarked slots gets nothing, because there the order is the
        only thing telling the two apart and licensing a permutation would be inventing a reading.

        Two demonstrated orders, not one: one order is not evidence that order is free.
        """
        import itertools
        groups: Dict[Tuple[Any, ...], List[Construction]] = {}
        for construction in kept:
            if construction.literals or construction.markers:
                continue
            slots = [item for item in construction.pattern if isinstance(item, Slot)]
            if len(slots) != len(construction.pattern) or len(slots) > self.max_free_slots:
                continue
            key = (construction.tongue, construction.kind, construction.focus,
                   construction.negated, construction.temporal, construction.modality,
                   construction.features,
                   tuple(sorted((item.role, item.prefix, item.suffix) for item in slots)))
            groups.setdefault(key, []).append(construction)
        extra: List[Construction] = []
        for key, group in groups.items():
            if len(group) < 2:
                continue
            slots = [item for item in group[0].pattern if isinstance(item, Slot)]
            material = [(item.prefix, item.suffix) for item in slots]
            if len(set(material)) != len(material):
                continue                        # two slots marked alike: order is the only cue
            if sum(1 for pair in material if not pair[0] and not pair[1]) > 1:
                continue                        # two bare slots: same problem
            seen = {tuple(item.role for item in c.pattern) for c in group}
            support = min(c.support for c in group)
            for order in itertools.permutations(slots):
                roles = tuple(item.role for item in order)
                if roles in seen:
                    continue
                seen.add(roles)
                head = group[0]
                extra.append(Construction(
                    tongue=head.tongue, pattern=tuple(order), kind=head.kind,
                    negated=head.negated, focus=head.focus, modality=head.modality,
                    temporal=head.temporal, features=head.features,
                    fillers=_pool_fillers(group),
                    support=support, source=f"{head.source} (order licensed by case)"))
            report.reasons.append(
                f"{group[0].shape!r}: {len(group)} orders shown, every order licensed — "
                f"each slot carries its own marking")
        return kept + extra

    @staticmethod
    def _varies(fillings: Sequence[Dict[str, str]]) -> bool:
        """Did any slot ever hold two different words? If not, this is one sentence twice."""
        for role in {role for filling in fillings for role in filling}:
            seen = {filling.get(role, "") for filling in fillings}
            if len(seen - {""}) > 1:
                return True
        return False

    def _verify(self, report: Learned) -> None:
        """Re-read every demonstration. Drop any construction that changes what one of them said.

        This is the step that makes the grammar answerable to its own lessons. A construction is
        *not* dropped for failing to be chosen — several shapes can read one sentence and only one
        can win — it is dropped for **winning and being wrong**, which is the only failure that
        puts a false meaning into the rest of the brain.
        """
        for _pass in range(3):
            guilty: Dict[Tuple[Any, ...], str] = {}
            for shown in self.demonstrations:
                got = self.read(shown.surface)
                if not got.readable:
                    continue
                if self._agrees(got, shown.meaning):
                    continue
                winner = self._winner_signature(shown.surface)
                if winner is not None:
                    guilty[winner] = shown.surface
            if not guilty:
                return
            before = len(self.constructions)
            self.constructions = [c for c in self.constructions if c.signature not in guilty]
            report.rejected += before - len(self.constructions)
            for surface in list(guilty.values())[:3]:
                report.reasons.append(f"{surface!r}: a shape read its own lesson differently")

    def _winner_signature(self, surface: str) -> Optional[Tuple[Any, ...]]:
        readings = self.candidates(surface)
        return readings[0].construction.signature if readings and readings[0].construction \
            else None

    @staticmethod
    def _agrees(got: Meaning, want: Meaning) -> bool:
        """Same sentence, same reading — on every field a construction is responsible for.

        Every role, not the legacy three: a reading that recovered the agent and the theme and
        dropped the recipient is a different claim about the sentence, and scoring it as a match
        would let the one thing this module was extended to do go unmeasured.
        """
        def norm(value: Any) -> str:
            return " ".join(str(value or "").strip().lower().split())
        return (got.kind == (want.kind or "assertion")
                and bool(got.negated) == bool(want.negated)
                and norm(got.focus) == norm(want.focus)
                and norm(got.temporal) == norm(want.temporal)
                and _fills_of(got) == _fills_of(want))

    # -- reading ------------------------------------------------------------ #
    def _resolve(self, token: str, filler: str) -> str:
        """The filler a slot took, corrected by the morphology where it is plainly not a word.

        **An ending is a fact about the language; a construction only knows the ending it was
        shown.** The transitive family here was shown both `-s` and `-es`, so it reads `pushes`;
        the possessive family was only ever shown `-s`, so it read the same word as `pushe` —
        which is not a word, in a language where `push` is. The morphology already holds the pair,
        because it holds the whole paradigm rather than one construction's share of it.

        Narrow on purpose: it fires only where the slot's own reading is **not a word she has
        met** and the morphology's is, so it can correct a wrong cut and can never overrule a
        filler that is already a word of the language.
        """
        if self.known is None or self.stem is None or self.known(filler):
            return filler
        try:
            better = self.stem(token)
        except Exception:  # noqa: BLE001
            return filler
        return better if better and better != token and self.known(better) else filler

    def _repair(self, written: str, filler: str, prefix: str, suffix: str) -> str:
        """A word she has met, where the construction's own ending would write one she has not.

        The mirror of :meth:`_resolve`, and the same fact seen from the other side: a family that
        was only ever shown ``-s`` writes ``pushs``, which is not a word, in a language where
        ``pushes`` is. What this may do is **choose a form she has already met** whose stem is the
        filler and which ends the way the construction says. What it may not do is invent one —
        so a word she has never heard comes out however the construction writes it, and the
        round-trip check that follows still has to pass.

        This is why comprehension ran ahead of production for a while, which is the ordinary shape
        of the thing: reading needs only to recognise a form, and speaking has to pick one.
        """
        if self.realise is None or self.known is None or self.known(written):
            return ""
        try:
            return self.realise(filler, prefix, suffix)
        except Exception:  # noqa: BLE001
            return ""

    def candidates(self, surface: str) -> List[Reading]:
        """Every reading every construction offers, best evidence first."""
        tokens = tokenize_surface(surface)
        if not tokens:
            return []
        out: List[Reading] = []
        for construction in self.constructions:
            for fills, features, material in construction.readings(tokens, self._resolve):
                # The anchor is **whole tokens of the sentence accounted for**, and nothing else.
                #
                # Affix characters used to count towards it, and that put a thumb on the scale for
                # whichever reading cut deepest into a word: `chases` is `chase` + `s` or `chas` +
                # `es`, and the second matched two characters instead of one, so it outranked the
                # first before any other evidence was consulted. It then read its own lesson
                # wrongly, `_verify` dropped it, and the whole transitive family went with it.
                #
                # Endings are still evidence — they are in the soft score below, where a known
                # word and a role's own history can outweigh one more matched character.
                anchor = float(construction.literals)
                slack = sum(len(value.split()) - 1 for value in fills.values())
                # A reading whose fillers are words she has actually met is better supported than
                # one whose are not. It decides nothing on its own — it is worth less than a
                # literal — and it is the only evidence there is for where a **compound** divides:
                # two novel words written together have a boundary nothing can see, and one novel
                # word written against a known one has a boundary that is simply where the known
                # word starts.
                familiar = 0
                if self.known is not None:
                    familiar = sum(1 for value in fills.values() if self.known(value))
                # And a word that has stood in *this* role before is stronger evidence still.
                fitted = sum(1 for role, value in fills.items()
                             if value in construction.fillers.get(role, ()))
                score = ((construction.bound_chars + material) / 2.0
                         - 0.5 * slack + 0.25 * min(construction.support, 8)
                         + 0.5 * familiar + 0.4 * fitted + construction.record)
                out.append(Reading(
                    meaning=self._meaning(construction, fills, anchor, tokens, features),
                    construction=construction, score=score, anchor=anchor,
                    material=float(construction.bound_chars + material)))
        # **Material the sentence contains outranks everything else, always.** Anchor first and
        # the rest only as a tie-break, rather than all of it added into one number.
        #
        # Added together, it was not a ranking of evidence, it was a race between kinds of it —
        # and a popularity count won. A content question `<verb> <subject> zumu` matched the
        # question word that is *in the sentence*; the plain assertion matched nothing at all and
        # read that word as an ordinary object. The assertion was the more-demonstrated shape, so
        # its support term overtook the question's matched literal and the two tied, and a tie
        # that disagrees is a refusal: 42 of 480 sentences came back `ambiguous` for it. Support
        # is a fact about the corpus; a literal is a fact about the sentence in front of her.
        out.sort(key=lambda reading: (-reading.anchor, -reading.score))
        return out

    def read(self, surface: str) -> Meaning:
        """The best-supported reading, or an unreadable Meaning.

        Two ways to come back unreadable, and they are different facts about the sentence:
        ``frame=""`` means no construction matched at all, and ``frame="ambiguous"`` means two
        matched equally well and disagreed. The second is the one worth having — it is a sentence
        her grammar genuinely cannot resolve, and reporting it as a confident reading of one of
        the two would be inventing a decision the evidence does not contain.
        """
        try:
            readings = self.candidates(surface)
        except Exception:  # noqa: BLE001
            return Meaning(language=self.tongue)
        if not readings:
            return Meaning(language=self.tongue)
        best = readings[0]
        for other in readings[1:]:
            if (other.anchor < best.anchor - 1e-9
                    or other.score < best.score - 1e-9):
                break
            if not self._agrees(other.meaning, best.meaning):
                return Meaning(kind="unreadable", frame="ambiguous", language=self.tongue)
        return best.meaning

    def anchored(self, surface: str) -> bool:
        """Did the construction that won this sentence match any **fixed material**?

        A literal token, or a case marker, or a tense ending — anything that is a fact about the
        words in front of it rather than about their number and order. It is the one property that
        separates a reading with lexical evidence from a reading that would have fired on any
        three tokens whatsoever, and :mod:`nyxara.njp.grounding` uses exactly that distinction to
        decide whether a learned construction may outrank the shipped positional frame.
        """
        try:
            readings = self.candidates(surface)
        except Exception:  # noqa: BLE001
            return False
        if not readings or readings[0].construction is None:
            return False
        # Fixed material of **any** kind — a literal token, a case marker, a tense ending — but
        # only material *this sentence actually contained*. Not `Reading.anchor`, which counts
        # literal tokens alone because that is what ranking two readings of one sentence needs; and
        # emphatically not "does the construction have any", which is what this asked first.
        #
        # A grammar for a minted dialect is a grammar for a language nobody is speaking, and the
        # question a real turn asks is whether *this* sentence is in it. Asked about the shape
        # rather than the match, a three-slot dialect shape whose paradigm happens to carry a
        # tense marker answered yes to plain English — so after the school taught two dialects,
        # the grounder took **82** English sentences through them, mangled the entity names, and
        # `composition` and `depth` fell from 1.00 to 0.33 in the retention run. Sixteen
        # abstentions, no wrong answers, and a defect that only appears once she is multilingual.
        return bool(readings[0].anchor > 0.0 or readings[0].material > 0.0)

    def _meaning(self, construction: Construction, fills: Dict[str, str],
                 anchor: float, tokens: Sequence[str],
                 features: Optional[Dict[str, str]] = None) -> Meaning:
        meaning = Meaning(
            kind=construction.kind, negated=construction.negated, focus=construction.focus,
            modality=construction.modality, temporal=construction.temporal,
            frame=f"learned:{construction.tongue}", language=construction.tongue,
        )
        carried = dict(construction.features)
        carried.update(features or {})
        for name, value in carried.items():
            if name == "negated":
                meaning.negated = value == "true"
            elif name in _MEANING_FEATURES:
                setattr(meaning, name, value)
            else:
                fills = dict(fills)
                fills[name] = value
        _write_fills(meaning, fills)
        meaning.confidence = round(min(0.95, 0.55 + 0.05 * anchor
                                       + 0.3 * (construction.record - 0.5)), 4)
        return meaning

    # -- saying ------------------------------------------------------------- #
    def say(self, meaning: Meaning) -> str:
        """A sentence for ``meaning``, or ``""``.

        Rendered, then **read back**. A construction whose output does not parse into the meaning
        it was given is not used, however well it scored, and when none survives that check she
        says nothing at all. There is no best-effort surface here for the same reason there is no
        best-effort program in :mod:`nyxara.njp.coding`: a sentence that nearly says what was
        meant says something else.

        With markers, one construction is a paradigm rather than a sentence shape, so what is
        searched is every cell of it — and the cell that matches the meaning may be one nobody
        ever demonstrated.
        """
        try:
            fills = _fills_of(meaning)
            if not fills:
                return ""
            fallback = ""
            wanted = {"temporal": meaning.temporal or "", "modality": meaning.modality or "",
                      "negated": "true" if meaning.negated else ""}
            ranked = sorted(
                (c for c in self.constructions
                 if c.kind == (meaning.kind or "assertion")
                 and (c.focus or "") == (meaning.focus or "")),
                key=lambda c: (-sum(1 for role, value in fills.items()
                                    if value in c.fillers.get(role, ())),
                               -c.record, -c.support, -c.literals))
            for construction in ranked:
                for pattern, extra, _material in construction.variants():
                    carried = dict(construction.features)
                    carried.update(extra)
                    said = {"temporal": construction.temporal or "",
                            "modality": construction.modality or "",
                            "negated": "true" if construction.negated else ""}
                    roles: Dict[str, str] = {}
                    for name, value in carried.items():
                        if name in _MEANING_FEATURES:
                            said[name] = value
                        else:
                            roles[name] = value
                    if said != wanted:
                        continue
                    slots: Set[str] = set()
                    for item in pattern:
                        if isinstance(item, Slot):
                            slots.add(item.role)
                        elif isinstance(item, Joint):
                            slots.update(item.roles)
                    if slots | set(roles) != set(fills):
                        continue
                    if any(fills.get(name) != value for name, value in roles.items()):
                        continue
                    surface = construction.render(fills, pattern, self._repair)
                    if surface and self._agrees(self.read(surface), meaning):
                        # Among renderings that read back correctly, one made of **words she has
                        # met** is the better sentence. With two allomorphs of an ending, both
                        # round-trip — `khata` parses back to the right meaning as surely as
                        # `khaata` does — and only one of them is a word of the language.
                        if self.known is None or all(self.known(word)
                                                     for word in tokenize_surface(surface)):
                            return surface
                        fallback = fallback or surface
        except Exception:  # noqa: BLE001
            return ""
        return fallback

    # -- the record --------------------------------------------------------- #
    def score(self, surface: str, meaning: Meaning) -> bool:
        """Grade one held-out sentence against the construction that won it.

        This is what turns :attr:`Construction.record` into something other than a default. It is
        called by whoever holds the answer — the school, or a later turn in which the Master said
        what he meant — and never by :meth:`read`, which has no way of knowing.
        """
        readings = self.candidates(surface)
        if not readings or readings[0].construction is None:
            return False
        construction = readings[0].construction
        if self._agrees(readings[0].meaning, meaning):
            construction.hits += 1
            return True
        construction.misses += 1
        return False

    def stats(self) -> Dict[str, Any]:
        return {"tongue": self.tongue, "constructions": len(self.constructions),
                "demonstrations": len(self.demonstrations),
                "shapes": [c.to_dict() for c in self.constructions[:8]]}

    # -- persistence -------------------------------------------------------- #
    #: The most demonstrations carried across a restart. A grammar's demonstrations are its
    #: evidence and they do not expire, but a sidecar is not an archive: the newest are kept
    #: because a shape supported by five hundred sentences is not supported any better by a
    #: thousand, and one whose only support is older than that is a shape she has not used.
    max_kept = 512

    def to_dict(self) -> Dict[str, Any]:
        """The **lessons**, not the conclusions.

        Constructions are re-derived on load by re-running :meth:`learn`, rather than written out
        and read back. It is a little slower and it makes one thing impossible: a persisted
        grammar cannot contain a construction that its own demonstrations do not support. A
        sidecar that stored the conclusions could be edited, truncated or written by an older
        version into exactly that state, and nothing downstream would ever find out.

        What *is* stored beside them is each construction's held-out record, keyed by
        :attr:`Construction.key`, because that is evidence from outside the lessons and re-running
        them would not recover it.
        """
        return {
            "tongue": self.tongue,
            "demonstrations": [{"surface": shown.surface,
                                "meaning": _meaning_to_dict(shown.meaning)}
                               for shown in self.demonstrations[-self.max_kept:]],
            "records": {c.key: [c.hits, c.misses] for c in self.constructions},
        }

    def load_dict(self, data: Dict[str, Any]) -> None:
        self.demonstrations = []
        for row in (data or {}).get("demonstrations") or []:
            try:
                self.demonstrations.append(
                    Demonstration(surface=str(row.get("surface") or ""),
                                  meaning=_meaning_from_dict(row.get("meaning") or {})))
            except Exception:  # noqa: BLE001 — one unreadable row is not a lost grammar
                continue
        self.learn()
        records = (data or {}).get("records") or {}
        for construction in self.constructions:
            got = records.get(construction.key)
            if isinstance(got, (list, tuple)) and len(got) == 2:
                construction.hits, construction.misses = int(got[0]), int(got[1])


# --------------------------------------------------------------------------- #
# a tongue, and the faculty that holds several of them
# --------------------------------------------------------------------------- #

class Tongue:
    """One language: its morphology, its word classes, and its constructions.

    Kept as one object because the three are not independent — the grammar reads a noun's stem
    through the morphology, and the morphology's vocabulary is whatever the grammar was shown.
    Keeping them per-language is the other half of it: a suffix is a fact about a language, and
    a faculty that pooled them would let an English plural cut a Hindi word in half.
    """

    def __init__(self, name: str = "en") -> None:
        self.name = str(name or "en")
        self.morphology = Morphology()
        self.lexicon = Lexicon()
        self.grammar = Grammar(self.name)
        self.grammar.known = self._met
        self.grammar.stem = self._stem
        self.grammar.realise = self._realise
        self.heard = 0

    def _met(self, word: str) -> bool:
        """Has this exact word been heard, on its own, in this language?"""
        return str(word or "").strip().lower() in self.morphology.vocabulary

    def _stem(self, word: str) -> str:
        """What the morphology makes of this word, whatever a construction made of it."""
        return self.morphology.analyse(word).stem

    def _realise(self, filler: str, prefix: str, suffix: str) -> str:
        """A form she has met, of this stem, written the way this construction writes it.

        Selects, never invents: the answer is always a word already in the vocabulary. Where two
        of them fit, the shortest wins, because the longer one is carrying material the
        construction did not ask for.
        """
        found = [word for word in self.morphology.vocabulary
                 if word.startswith(prefix) and word.endswith(suffix)
                 and self.morphology.analyse(word).stem == filler]
        return min(found, key=len) if found else ""

    def hear(self, text: str) -> None:
        """Surface only, with no meaning attached — which is most of what anyone ever hears."""
        tokens = tokenize_surface(text)
        if not tokens:
            return
        self.heard += 1
        self.morphology.observe(tokens)
        self.lexicon.observe(tokens)

    def hear_words(self, words: Iterable[str]) -> int:
        """A word list, which is **not** a corpus, and is kept out of the class induction.

        The distinction is not fussiness. A wug test is administered on isolated forms — *this is
        a wug; now there are two of them* — and feeding those forms to :class:`Lexicon` as though
        they were sentences would invent adjacencies nobody uttered and put the two halves of one
        paradigm in one distributional class for a reason that is an artefact of how the lesson
        was delivered.
        """
        return self.morphology.observe(words)

    def show(self, surface: str, meaning: Meaning) -> bool:
        self.hear(surface)
        return self.grammar.show(surface, meaning)

    def learn(self) -> Learned:
        self.morphology.induce()
        classes = self.lexicon.induce()
        report = self.grammar.learn()
        report.affixes = len(self.morphology.affixes)
        report.bound = sum(1 for affix in self.morphology.affixes if affix.feature)
        report.classes = classes
        return report

    def read(self, surface: str) -> Meaning:
        """Parse, then let the morphology have the last word on the content words.

        The stem is substituted **only** where the affix carries a feature a lesson bound, so
        ``zorbik`` becomes ``zorb`` once she knows what ``-ik`` does and stays ``zorbik`` while
        she does not. An organ that stripped what it could not name would be merging two entities
        on the strength of a spelling.
        """
        meaning = self.grammar.read(surface)
        if not meaning.readable:
            return meaning
        for role in ("subject", "object"):
            value = getattr(meaning, role, "")
            if value and " " not in value:
                setattr(meaning, role, self.morphology.stem_of(value))
        return meaning

    def say(self, meaning: Meaning) -> str:
        return self.grammar.say(meaning)

    def stats(self) -> Dict[str, Any]:
        return {"name": self.name, "heard": self.heard,
                "morphology": self.morphology.stats(), "lexicon": self.lexicon.stats(),
                "grammar": self.grammar.stats()}

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "heard": self.heard,
                "morphology": self.morphology.to_dict(), "lexicon": self.lexicon.to_dict(),
                "grammar": self.grammar.to_dict()}

    def load_dict(self, data: Dict[str, Any]) -> None:
        data = data or {}
        self.heard = int(data.get("heard") or 0)
        self.morphology.load_dict(data.get("morphology") or {})
        self.lexicon.load_dict(data.get("lexicon") or {})
        self.grammar.load_dict(data.get("grammar") or {})


class LanguageFaculty:
    """Every language she has been taught, and the one thing that is not in any of them.

    A meaning belongs to no language. That is not a slogan here — it is the only reason
    :meth:`translate` can exist without a phrase table, and the reason a
    :class:`~nyxara.njp.semantics.Meaning` produced by one tongue can be handed to another and
    come back as a sentence sharing not one character with the first.

    On day one she holds nothing: :meth:`read` returns an unreadable Meaning, :meth:`say` returns
    the empty string, and neither is a failure. Everything this faculty can do, somebody showed
    her.
    """

    def __init__(self, *, default: str = "en") -> None:
        self.default = str(default or "en")
        self.tongues: Dict[str, Tongue] = {}
        #: Word correspondences between two tongues, ``(from, into) -> {word: word}``.
        #:
        #: Translation between **minted** languages needed none of this, because those banks give
        #: both dialects the same content words and only the grammar differs — so carrying the
        #: meaning across was carrying the role names and nothing else. Between two real languages
        #: the words differ too, and a meaning made of words cannot cross without them. That is
        #: not a gap in the mechanism, it is what a bilingual lexicon is *for*, and like everything
        #: else here it is taught rather than assumed: :meth:`pair` records a correspondence
        #: somebody showed her, and :meth:`translate` refuses where one is missing.
        self.gloss: Dict[Tuple[str, str], Dict[str, str]] = {}
        self.readings = 0
        self.refusals = 0
        self.said = 0
        self.mute = 0

    # -- the tongues -------------------------------------------------------- #
    def tongue(self, name: Optional[str] = None) -> Tongue:
        key = str(name or self.default)
        if key not in self.tongues:
            self.tongues[key] = Tongue(key)
        return self.tongues[key]

    def known(self) -> List[str]:
        return sorted(self.tongues)

    # -- learning ----------------------------------------------------------- #
    def hear(self, text: str, *, tongue: Optional[str] = None) -> None:
        try:
            self.tongue(tongue).hear(text)
        except Exception:  # noqa: BLE001
            pass

    def hear_words(self, words: Iterable[str], *, tongue: Optional[str] = None) -> int:
        """A list of forms, for the morphology only. See :meth:`Tongue.hear_words`."""
        try:
            return self.tongue(tongue).hear_words(words)
        except Exception:  # noqa: BLE001
            return 0

    def classify(self, *, tongue: Optional[str] = None) -> int:
        """Re-run the class induction alone. Returns how many classes the corpus supports."""
        try:
            return self.tongue(tongue).lexicon.induce()
        except Exception:  # noqa: BLE001
            return 0

    def show(self, surface: str, meaning: Meaning, *, tongue: Optional[str] = None) -> bool:
        try:
            return self.tongue(tongue).show(surface, meaning)
        except Exception:  # noqa: BLE001
            return False

    def bind(self, base: str, inflected: str, feature: str, *,
             tongue: Optional[str] = None) -> bool:
        """Demonstrate one inflection. Returns whether an induced affix could carry it."""
        try:
            spoken = self.tongue(tongue)
            spoken.morphology.observe((base, inflected))
            spoken.morphology.induce()
            return spoken.morphology.bind(base, inflected, feature)
        except Exception:  # noqa: BLE001
            return False

    def learn(self, *, tongue: Optional[str] = None) -> Learned:
        try:
            return self.tongue(tongue).learn()
        except Exception as exc:  # noqa: BLE001
            return Learned(tongue=str(tongue or self.default),
                           reasons=[f"{type(exc).__name__}: {exc}"])

    def learn_all(self) -> List[Learned]:
        return [self.learn(tongue=name) for name in self.known()]

    # -- use ---------------------------------------------------------------- #
    def read(self, surface: str, *, tongue: Optional[str] = None) -> Meaning:
        """Read, in a named language or in whichever of hers reads it best.

        With no tongue named this is language identification done the only honest way available
        here: not by guessing from the alphabet, but by which grammar can actually account for the
        sentence. A tie between two languages that disagree about what was said comes back
        unreadable, exactly as a tie inside one language does.
        """
        try:
            if tongue is not None:
                meaning = self.tongue(tongue).read(surface)
            else:
                best: Optional[Tuple[float, Meaning]] = None
                rivals: List[Tuple[float, Meaning]] = []
                for name in self.known():
                    spoken = self.tongues[name]
                    readings = spoken.grammar.candidates(surface)
                    if not readings:
                        continue
                    scored = (readings[0].score, spoken.read(surface))
                    rivals.append(scored)
                    if best is None or scored[0] > best[0]:
                        best = scored
                if best is None:
                    meaning = Meaning()
                else:
                    contested = [meaning for score, meaning in rivals
                                 if score >= best[0] - 1e-9
                                 and not Grammar._agrees(meaning, best[1])]
                    meaning = Meaning(kind="unreadable", frame="ambiguous") if contested \
                        else best[1]
        except Exception:  # noqa: BLE001
            return Meaning()
        if meaning.readable:
            self.readings += 1
        else:
            self.refusals += 1
        return meaning

    def say(self, meaning: Meaning, *, tongue: Optional[str] = None) -> str:
        try:
            surface = self.tongue(tongue).say(meaning)
        except Exception:  # noqa: BLE001
            surface = ""
        if surface:
            self.said += 1
        else:
            self.mute += 1
        return surface

    def pair(self, here: str, there: str, *, frm: Optional[str] = None,
             into: str = "") -> bool:
        """Record that these two words are each other's, in these two languages.

        Both directions, because a correspondence is symmetric and storing it once would make the
        return journey a different fact. No sense disambiguation and no context: one word, one
        word, which is what a glossary is and is nothing like what a translator needs.
        """
        source, target = str(frm or self.default), str(into or self.default)
        here, there = str(here or "").strip().lower(), str(there or "").strip().lower()
        if not here or not there or source == target:
            return False
        self.gloss.setdefault((source, target), {})[here] = there
        self.gloss.setdefault((target, source), {})[there] = here
        return True

    def translate(self, surface: str, *, into: str, frm: Optional[str] = None) -> str:
        """Read in one language, say in another. ``""`` when either half cannot be done.

        No phrase table and no alignment: the only thing carried across is the
        :class:`~nyxara.njp.semantics.Meaning` — the roles, the polarity, the tense, the mood.
        A translation she produces is one she could also have read back, in the target language,
        into the meaning she started from.

        The **words** are carried by :meth:`pair`, and only where somebody has shown her the
        correspondence. A sentence with one unglossed word in it translates to nothing at all
        rather than to a sentence with a hole in it, which is the same rule the rest of this
        module keeps: what she cannot do, she declines.
        """
        source = str(frm or self.default)
        meaning = self.read(surface, tongue=frm)
        if not meaning.readable:
            return ""
        table = self.gloss.get((source, str(into)))
        if table:
            carried = Meaning(kind=meaning.kind, negated=meaning.negated,
                              temporal=meaning.temporal, modality=meaning.modality,
                              focus=meaning.focus)
            for role, value in _fills_of(meaning).items():
                word = table.get(value)
                if word is None:
                    return ""
                _write_fills(carried, {role: word})
            meaning = carried
        return self.say(meaning, tongue=into)

    # -- morphology, straight through --------------------------------------- #
    def inflect(self, stem: str, feature: str, *, tongue: Optional[str] = None) -> str:
        try:
            return self.tongue(tongue).morphology.inflect(stem, feature)
        except Exception:  # noqa: BLE001
            return ""

    def analyse(self, word: str, *, tongue: Optional[str] = None) -> Segment:
        try:
            return self.tongue(tongue).morphology.analyse(word)
        except Exception:  # noqa: BLE001
            return Segment()

    def anchored(self, surface: str, *, tongue: Optional[str] = None) -> bool:
        """Whether the winning construction matched fixed material. See :meth:`Grammar.anchored`.

        With no tongue named this is true where **any** of her grammars reads the sentence with
        lexical evidence, which is the right question for a caller asking "does she have a real
        account of this sentence" rather than "which language is it".
        """
        try:
            if tongue is not None:
                return self.tongue(tongue).grammar.anchored(surface)
            return any(self.tongues[name].grammar.anchored(surface) for name in self.known())
        except Exception:  # noqa: BLE001
            return False

    def same_class(self, left: str, right: str, *,
                   tongue: Optional[str] = None) -> Optional[bool]:
        try:
            return self.tongue(tongue).lexicon.same_class(left, right)
        except Exception:  # noqa: BLE001
            return None

    def stats(self) -> Dict[str, Any]:
        return {"tongues": self.known(), "readings": self.readings, "refusals": self.refusals,
                "said": self.said, "mute": self.mute,
                "detail": {name: spoken.stats() for name, spoken in self.tongues.items()}}

    # -- persistence -------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        """Every tongue she holds, as the lessons that produced it.

        A grammar that did not survive a restart would be the one organ in this brain that has to
        be taught again every morning — and this package's whole claim about persistence is that
        the brain that wakes is the brain that slept. What comes back is re-derived from the
        demonstrations rather than read out of the file, so a sidecar can make her forget and it
        cannot make her believe a shape nobody showed her.
        """
        out: Dict[str, Any] = {"default": self.default, "tongues": {},
                               "gloss": {f"{a}|{b}": dict(words)
                                         for (a, b), words in self.gloss.items()}}
        for name, spoken in self.tongues.items():
            try:
                out["tongues"][name] = spoken.to_dict()
            except Exception:  # noqa: BLE001 — one unwritable tongue is not a lost faculty
                continue
        return out

    def load_dict(self, data: Dict[str, Any]) -> None:
        data = data or {}
        self.default = str(data.get("default") or self.default)
        for key, words in (data.get("gloss") or {}).items():
            if "|" in str(key) and isinstance(words, dict):
                source, target = str(key).split("|", 1)
                self.gloss[(source, target)] = {str(k): str(v) for k, v in words.items()}
        for name, blob in (data.get("tongues") or {}).items():
            try:
                self.tongue(str(name)).load_dict(blob or {})
            except Exception:  # noqa: BLE001
                continue
