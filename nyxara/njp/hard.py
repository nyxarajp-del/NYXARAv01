"""NYXARA · njp/hard.py — language problems she was never designed for (🜃, NJP V.19).

:mod:`nyxara.njp.school` examines her on languages it mints, and she scores 84/84 on every one of
them. That number is worth exactly as much as the bank behind it is hard, and the bank behind it
is not hard: every minted dialect there is one flat clause with three arguments, a particle or
two, and one suffix per feature. It was built alongside the faculty, so it can only ask for what
the faculty was built to do.

The coding half of this package learned what that is worth, expensively and in public. Twenty-five
of twenty-eight hard problems, on a bank written while the skeletons that solve them were being
written; then a **second bank chosen afterwards**, measured once, and the answer was **0 / 21**.
Not a rounding — nothing. "25/28 was a list of answers."

So this file is written **first**, against a faculty that is finished, by someone looking for what
it cannot do rather than for what it can. Every problem here names a property of human language
that :mod:`nyxara.njp.language` has no representation for, and each is minted per seed so it
cannot be memorised:

* **More than three arguments.** ``ROLES`` is the literal tuple ``("subject", "verb", "object")``.
  A clause with an agent, a theme *and* a recipient has nowhere to put the third, and a clause with
  a location or an instrument has nowhere to put those either.
* **Anything inside anything.** One construction is one flat sentence. A noun with an adjective on
  it, two nouns coordinated, a relative clause, a *"she said that …"* complement — all of them are
  a phrase where the grammar expects a word.
* **Features that combine.** Taught singular-present, plural-present and singular-past, she has
  three constructions and no fourth. Plural-past is a combination nobody demonstrated, and
  composing two features she has seen separately is the difference between a paradigm and a list.
* **Order that is free because case is not.** Shown two orders of a case-marked language, nothing
  licences the third — although the case markers say the roles unambiguously in all six.
* **Morphology that is not a suffix.** :class:`~nyxara.njp.language.Morphology` is concatenative
  by construction: prefix or suffix, nothing else. Circumfixes, infixes, reduplication, root-and-
  pattern templates and allomorphs chosen by vowel harmony are each invisible to it — not
  mis-analysed, *invisible*, which is the same word this package used about Devanagari at NYX
  V.01 and about negation at V.09.

**How a problem is scored.** Each mints a language, shows her a lesson, and examines her on items
built from words no lesson used. Three numbers per problem and they are kept apart: sentences
**read** into the right meaning, meanings **said** back into the right sentence, and — where the
problem is morphological — **wug** items, an inflection of a stem nobody has ever inflected. A
problem counts as solved only when all three are perfect, because a grammar that reads and cannot
speak has learned half a language.

Nothing here is graded leniently and nothing is graded by similarity. A reading is right when
every role, the polarity, the tense and the focus all match; a sentence is right when it is the
string the language's own renderer produces. There is no partial credit, because a partly-right
parse is a wrong meaning.

Pure standard library, deterministic per seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.semantics import Meaning

__all__ = [
    "Trial", "Problem", "Report", "BANK", "SECOND_BANK", "EVERY_PROBLEM",
    "build", "grade", "sweep",
]

_ONSETS = "bdgklmnprstvz"
_VOWELS = "aeiou"
#: Split for vowel harmony: a language whose suffix vowel copies the class of the stem's last one.
_FRONT, _BACK = "ei", "aou"


class Mint:
    """Fresh words that cannot repeat, and a marker pool that cannot collide with them."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.issued = 0
        self.markers: List[str] = []

    @staticmethod
    def _tag(number: int) -> str:
        letters = ""
        number += 1
        while number:
            number, remainder = divmod(number - 1, 26)
            letters = chr(ord("a") + remainder) + letters
        return letters

    def word(self, syllables: int = 2) -> str:
        """A content word. Ends in the counter's letters, so no two are ever the same."""
        self.issued += 1
        body = "".join(self.rng.choice(_ONSETS) + self.rng.choice(_VOWELS)
                       for _ in range(syllables))
        word = f"{body}{self._tag(self.issued)}"
        # A content word must not end in one of this language's own markers, or the sentence
        # built from it is ambiguous by the language's own rules and the exam is scoring itself.
        if any(word.endswith(marker) for marker in self.markers):
            return self.word(syllables)
        return word

    def words(self, count: int, syllables: int = 2) -> List[str]:
        return [self.word(syllables) for _ in range(count)]

    def harmonic(self, front: bool) -> str:
        """A word whose last vowel is of the named class, for the harmony problem."""
        pool = _FRONT if front else _BACK
        self.issued += 1
        body = (self.rng.choice(_ONSETS) + self.rng.choice(_VOWELS)
                + self.rng.choice(_ONSETS) + self.rng.choice(pool))
        word = f"{body}{self._tag(self.issued)}{self.rng.choice(pool)}"
        if any(word.endswith(marker) for marker in self.markers):
            return self.harmonic(front)
        return word

    def marker(self, syllables: int = 1) -> str:
        """A grammatical form, drawn so that no marker is a prefix or suffix of another."""
        while True:
            form = "".join(self.rng.choice(_ONSETS) + self.rng.choice(_VOWELS)
                           for _ in range(syllables))
            if any(form.endswith(other) or other.endswith(form)
                   or form.startswith(other) or other.startswith(form)
                   for other in self.markers):
                continue
            self.markers.append(form)
            return form

    def markers_n(self, count: int, syllables: int = 1) -> List[str]:
        return [self.marker(syllables) for _ in range(count)]


def _meaning(kind: str = "assertion", *, negated: bool = False, temporal: str = "",
             focus: str = "", **roles: str) -> Meaning:
    """A meaning built from named roles, with the legacy three kept where they always were."""
    out = Meaning(kind=kind, negated=negated, temporal=temporal, focus=focus)
    for name, value in roles.items():
        if not value:
            continue
        if name == "subject":
            out.subject = value
        elif name == "verb":
            out.relation = value
        elif name == "object":
            out.object = value
        else:
            out.roles[name] = value
    return out


def role_map(meaning: Meaning) -> Dict[str, str]:
    """Every role a meaning carries, the legacy three folded in with the rest.

    One place, so a grader and a faculty cannot disagree about what a meaning says.
    """
    out: Dict[str, str] = dict(getattr(meaning, "roles", {}) or {})
    for name, value in (("subject", meaning.subject), ("verb", meaning.relation),
                        ("object", meaning.object)):
        if value:
            out[name] = value
    return {key: " ".join(str(value).strip().lower().split())
            for key, value in out.items() if str(value).strip()}


def agrees(got: Optional[Meaning], want: Meaning) -> bool:
    """Right, with no partial credit. Every role, the polarity, the tense and the focus."""
    if got is None or not getattr(got, "readable", False):
        return False
    return (got.kind == (want.kind or "assertion")
            and bool(got.negated) == bool(want.negated)
            and (got.temporal or "") == (want.temporal or "")
            and (got.focus or "") == (want.focus or "")
            and role_map(got) == role_map(want))


@dataclass(frozen=True)
class Item:
    """One exam item: a sentence, what it means, and every surface that would say the same.

    ``accept`` is the fourth flaw this bank found in itself, and the first one that was not about
    minting an ambiguous language. Where a language's order is **free**, six surfaces say one
    meaning and all six are correct — so grading production against the one the renderer happened
    to emit marked a right answer wrong eight times out of eight, on the very problem whose point
    is that order does not matter. She read all eight correctly at the same time, which is what
    made it obvious the fault was on this side.
    """

    surface: str = ""
    meaning: Meaning = field(default_factory=Meaning)
    accept: Tuple[str, ...] = ()

    @property
    def sayable(self) -> Tuple[str, ...]:
        return self.accept or (self.surface,)


@dataclass
class Trial:
    """One minted problem: what she is shown, and what she is then asked.

    ``exam`` is built from words that appear in no lesson. ``wug`` is the morphological half — a
    stem, a feature, and the form the language gives it — and every one of its stems is likewise
    new. Nothing in ``exam`` or ``wug`` was ever uttered during ``lesson``.
    """

    language: str = ""
    lesson: List[Tuple[str, Meaning]] = field(default_factory=list)
    heard: List[str] = field(default_factory=list)
    forms: List[str] = field(default_factory=list)
    bound: List[Tuple[str, str, str]] = field(default_factory=list)
    exam: List[Any] = field(default_factory=list)
    wug: List[Tuple[str, str, str]] = field(default_factory=list)
    note: str = ""


@dataclass(frozen=True)
class Problem:
    """A property of human language, and a way to mint a language that has it."""

    id: str = ""
    title: str = ""
    why_hard: str = ""
    make: Optional[Callable[[Mint], Trial]] = None


@dataclass
class Report:
    """What one problem scored. Three counters kept apart, and no partial credit anywhere."""

    problem: str = ""
    title: str = ""
    read_right: int = 0
    read_total: int = 0
    said_right: int = 0
    said_total: int = 0
    wug_right: int = 0
    wug_total: int = 0
    misses: List[str] = field(default_factory=list)
    note: str = ""

    @property
    def solved(self) -> bool:
        """All three perfect, and at least one of them asked.

        A grammar that reads and cannot speak has learned half a language, so every column has to
        be clean. The "at least one asked" clause is there because a problem that asked nothing
        would otherwise report itself solved — and the first version of this property demanded a
        *reading* count specifically, which quietly scored six morphological problems as unsolved
        on the run where all six of their wug columns went perfect.
        """
        asked = self.read_total + self.said_total + self.wug_total
        return (asked > 0 and self.read_right == self.read_total
                and self.said_right == self.said_total
                and self.wug_right == self.wug_total)

    @property
    def score(self) -> float:
        total = self.read_total + self.said_total + self.wug_total
        right = self.read_right + self.said_right + self.wug_right
        return right / total if total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"problem": self.problem, "title": self.title, "solved": self.solved,
                "score": round(self.score, 3),
                "read": f"{self.read_right}/{self.read_total}",
                "said": f"{self.said_right}/{self.said_total}",
                "wug": f"{self.wug_right}/{self.wug_total}",
                "note": self.note, "misses": self.misses[:4]}


# --------------------------------------------------------------------------- #
# bank one — twenty problems, each naming something the faculty cannot represent
# --------------------------------------------------------------------------- #

def _ditransitive(mint: Mint) -> Trial:
    """Three arguments, each case-marked. ``ROLES`` has room for two of them."""
    agent, theme, goal = mint.markers_n(3)
    order = ("subject", "object", "recipient", "verb")
    case = {"subject": agent, "object": theme, "recipient": goal, "verb": ""}

    def say(s: str, v: str, o: str, r: str) -> Tuple[str, Meaning]:
        fills = {"subject": s, "object": o, "recipient": r, "verb": v}
        surface = " ".join(f"{fills[role]}{case[role]}" for role in order)
        return surface, _meaning(subject=s, verb=v, object=o, recipient=r)

    verbs = mint.words(3)
    trial = Trial(language="ditransitive")
    for index in range(12):
        trial.lesson.append(say(mint.word(), verbs[index % 3], mint.word(), mint.word()))
    for index in range(8):
        trial.exam.append(say(mint.word(), verbs[index % 3], mint.word(), mint.word()))
    return trial


def _oblique(mint: Mint) -> Trial:
    """An adposition and its object, on top of a full transitive clause."""
    at = mint.marker(2)
    verbs = mint.words(3)

    def say(s: str, v: str, o: str, place: str) -> Tuple[str, Meaning]:
        return f"{s} {v} {o} {at} {place}", _meaning(subject=s, verb=v, object=o, location=place)

    trial = Trial(language="oblique")
    for index in range(12):
        trial.lesson.append(say(mint.word(), verbs[index % 3], mint.word(), mint.word()))
    for index in range(8):
        trial.exam.append(say(mint.word(), verbs[index % 3], mint.word(), mint.word()))
    return trial


def _modifier(mint: Mint) -> Trial:
    """An adjective on the subject: a phrase where the grammar expects a word."""
    verbs = mint.words(3)

    def say(adj: str, s: str, v: str, o: str) -> Tuple[str, Meaning]:
        return f"{adj} {s} {v} {o}", _meaning(subject=s, verb=v, object=o, subject_mod=adj)

    trial = Trial(language="modifier")
    for index in range(12):
        trial.lesson.append(say(mint.word(), mint.word(), verbs[index % 3], mint.word()))
    for index in range(8):
        trial.exam.append(say(mint.word(), mint.word(), verbs[index % 3], mint.word()))
    return trial


def _coordination(mint: Mint) -> Trial:
    """Two subjects joined by a particle — one slot holding two things."""
    andp = mint.marker(2)
    verbs = mint.words(3)

    def say(a: str, b: str, v: str, o: str) -> Tuple[str, Meaning]:
        return f"{a} {andp} {b} {v} {o}", _meaning(subject=a, verb=v, object=o, subject2=b)

    trial = Trial(language="coordination")
    for index in range(12):
        trial.lesson.append(say(mint.word(), mint.word(), verbs[index % 3], mint.word()))
    for index in range(8):
        trial.exam.append(say(mint.word(), mint.word(), verbs[index % 3], mint.word()))
    return trial


def _possessive(mint: Mint) -> Trial:
    """A possessor marked on the noun it belongs to."""
    of = mint.marker()
    verbs = mint.words(3)

    def say(owner: str, s: str, v: str, o: str) -> Tuple[str, Meaning]:
        return f"{owner}{of} {s} {v} {o}", _meaning(subject=s, verb=v, object=o, owner=owner)

    trial = Trial(language="possessive")
    for index in range(12):
        trial.lesson.append(say(mint.word(), mint.word(), verbs[index % 3], mint.word()))
    for index in range(8):
        trial.exam.append(say(mint.word(), mint.word(), verbs[index % 3], mint.word()))
    return trial


def _relative(mint: Mint) -> Trial:
    """A whole clause inside a noun phrase — one sentence embedded in another."""
    which = mint.marker(2)
    verbs = mint.words(4)

    def say(s: str, rv: str, ro: str, v: str, o: str) -> Tuple[str, Meaning]:
        return (f"{s} {which} {rv} {ro} {v} {o}",
                _meaning(subject=s, verb=v, object=o, rel_verb=rv, rel_object=ro))

    trial = Trial(language="relative")
    for index in range(14):
        trial.lesson.append(say(mint.word(), verbs[index % 4], mint.word(),
                                verbs[(index + 1) % 4], mint.word()))
    for index in range(8):
        trial.exam.append(say(mint.word(), verbs[index % 4], mint.word(),
                              verbs[(index + 2) % 4], mint.word()))
    return trial


def _complement(mint: Mint) -> Trial:
    """"X says that Y V Z" — a clause as the object of a verb."""
    that = mint.marker(2)
    says = mint.word()
    verbs = mint.words(3)

    def say(s: str, cs: str, cv: str, co: str) -> Tuple[str, Meaning]:
        return (f"{s} {says} {that} {cs} {cv} {co}",
                _meaning(subject=s, verb=says, comp_subject=cs, comp_verb=cv, comp_object=co))

    trial = Trial(language="complement")
    for index in range(12):
        trial.lesson.append(say(mint.word(), mint.word(), verbs[index % 3], mint.word()))
    for index in range(8):
        trial.exam.append(say(mint.word(), mint.word(), verbs[index % 3], mint.word()))
    return trial


def _agreement(mint: Mint) -> Trial:
    """Three of the four cells are demonstrated. The exam is the fourth.

    Number on the noun and tense on the verb are each shown in both values, and each shown
    together with only *one* value of the other. Nothing about the missing cell is new — both its
    parts have been seen — and a grammar that stores whole sentence shapes has no way to reach it.
    """
    plural, past = mint.markers_n(2)
    verbs = mint.words(3)

    def say(s: str, v: str, o: str, *, many: bool, then: bool) -> Tuple[str, Meaning]:
        subject = f"{s}{plural}" if many else s
        surface = f"{subject} {v}{past if then else ''} {o}"
        return surface, _meaning(subject=s, verb=v, object=o,
                                 temporal="past" if then else "",
                                 **({"number": "plural"} if many else {}))

    trial = Trial(language="agreement")
    for index in range(15):
        many, then = (False, False) if index % 3 == 0 else \
            ((True, False) if index % 3 == 1 else (False, True))
        trial.lesson.append(say(mint.word(), verbs[index % 3], mint.word(),
                                many=many, then=then))
    for index in range(8):
        trial.exam.append(say(mint.word(), verbs[index % 3], mint.word(),
                              many=True, then=True))
    trial.note = "plural + past, a cell no lesson contains"
    return trial


def _scrambling(mint: Mint) -> Trial:
    """Case says the roles, so all six orders mean the same thing. Two are demonstrated."""
    nom, acc = mint.markers_n(2)
    verbs = mint.words(3)

    def say(s: str, v: str, o: str, order: Sequence[str]) -> Tuple[str, Meaning]:
        parts = {"S": f"{s}{nom}", "O": f"{o}{acc}", "V": v}
        return " ".join(parts[tag] for tag in order), _meaning(subject=s, verb=v, object=o)

    import itertools

    trial = Trial(language="scrambling")
    for index in range(14):
        order = ("S", "O", "V") if index % 2 else ("O", "S", "V")
        trial.lesson.append(say(mint.word(), verbs[index % 3], mint.word(), order))
    for index in range(8):
        order = ("V", "S", "O") if index % 2 else ("S", "V", "O")
        subject, verb, obj = mint.word(), verbs[index % 3], mint.word()
        surface, meaning = say(subject, verb, obj, order)
        # Reading is examined on the two orders she was not shown. **Production** is examined
        # against all six, because in this language all six are correct and grading her against
        # the one the renderer happened to pick marks a right answer wrong — which it did, 8/8,
        # on the one problem whose whole point is that the order does not matter.
        trial.exam.append(Item(surface, meaning, tuple(
            say(subject, verb, obj, permutation)[0]
            for permutation in itertools.permutations(("S", "O", "V")))))
    trial.note = "two orders shown; the other four are the exam"
    return trial


def _pro_drop(mint: Mint) -> Trial:
    """The subject is left out, and the verb's ending says who it was."""
    first, third = mint.markers_n(2)
    verbs = mint.words(3)
    me, them = mint.words(2)

    def say(v: str, o: str, *, speaker: bool) -> Tuple[str, Meaning]:
        return (f"{v}{first if speaker else third} {o}",
                _meaning(subject=me if speaker else them, verb=v, object=o))

    trial = Trial(language="pro-drop")
    for index in range(12):
        trial.lesson.append(say(verbs[index % 3], mint.word(), speaker=index % 2 == 0))
    for index in range(8):
        trial.exam.append(say(verbs[index % 3], mint.word(), speaker=index % 2 == 0))
    trial.note = "the subject is never in the sentence"
    return trial


def _harmony(mint: Mint) -> Trial:
    """One plural, two shapes, and which one appears is decided by the stem's last vowel."""
    front_form = mint.rng.choice(_ONSETS) + mint.rng.choice(_FRONT)
    back_form = front_form[0] + mint.rng.choice(_BACK)
    mint.markers.extend((front_form, back_form))

    def plural(stem: str) -> str:
        return stem + (front_form if stem[-1] in _FRONT else back_form)

    trial = Trial(language="harmony")
    for _ in range(10):
        for front in (True, False):
            stem = mint.harmonic(front)
            trial.forms.extend((stem, plural(stem)))
    demo_front, demo_back = mint.harmonic(True), mint.harmonic(False)
    trial.forms.extend((demo_front, plural(demo_front), demo_back, plural(demo_back)))
    trial.bound = [(demo_front, plural(demo_front), "plural"),
                   (demo_back, plural(demo_back), "plural")]
    for _ in range(4):
        for front in (True, False):
            stem = mint.harmonic(front)
            trial.wug.append((stem, "plural", plural(stem)))
    trial.note = "two allomorphs, chosen by the stem"
    return trial


def _circumfix(mint: Mint) -> Trial:
    """The past wraps the stem: something before it and something after it."""
    before, after = mint.markers_n(2)

    def past(stem: str) -> str:
        return f"{before}{stem}{after}"

    trial = Trial(language="circumfix")
    stems = mint.words(12)
    for stem in stems:
        trial.forms.extend((stem, past(stem)))
    trial.bound = [(stems[0], past(stems[0]), "past"), (stems[1], past(stems[1]), "past")]
    for _ in range(6):
        stem = mint.word()
        trial.wug.append((stem, "past", past(stem)))
    return trial


def _infix(mint: Mint) -> Trial:
    """The plural goes *inside* the stem, after its first syllable."""
    form = mint.marker()

    def plural(stem: str) -> str:
        return stem[:2] + form + stem[2:]

    trial = Trial(language="infix")
    stems = mint.words(12)
    for stem in stems:
        trial.forms.extend((stem, plural(stem)))
    trial.bound = [(stems[0], plural(stems[0]), "plural"),
                   (stems[1], plural(stems[1]), "plural")]
    for _ in range(6):
        stem = mint.word()
        trial.wug.append((stem, "plural", plural(stem)))
    return trial


def _reduplication(mint: Mint) -> Trial:
    """The plural is the stem said twice."""

    def plural(stem: str) -> str:
        return stem + stem

    trial = Trial(language="reduplication")
    stems = mint.words(12)
    for stem in stems:
        trial.forms.extend((stem, plural(stem)))
    trial.bound = [(stems[0], plural(stems[0]), "plural"),
                   (stems[1], plural(stems[1]), "plural")]
    for _ in range(6):
        stem = mint.word()
        trial.wug.append((stem, "plural", plural(stem)))
    return trial


def _partial_reduplication(mint: Mint) -> Trial:
    """Only the first syllable is copied, and it is copied to the front."""

    def plural(stem: str) -> str:
        return stem[:2] + stem

    trial = Trial(language="partial-reduplication")
    stems = mint.words(12)
    for stem in stems:
        trial.forms.extend((stem, plural(stem)))
    trial.bound = [(stems[0], plural(stems[0]), "plural"),
                   (stems[1], plural(stems[1]), "plural")]
    for _ in range(6):
        stem = mint.word()
        trial.wug.append((stem, "plural", plural(stem)))
    return trial


def _templatic(mint: Mint) -> Trial:
    """Root and pattern: the consonants are the word, the vowels are the grammar."""
    vowel = mint.rng.choice(_VOWELS)
    other = mint.rng.choice([v for v in _VOWELS if v != vowel])

    def root(rng: random.Random) -> str:
        return "".join(rng.choice(_ONSETS) for _ in range(3))

    def shape(consonants: str, first: str, second: str) -> str:
        return f"{consonants[0]}{first}{consonants[1]}{second}{consonants[2]}"

    trial = Trial(language="templatic")
    roots: List[str] = []
    while len(roots) < 14:
        consonants = root(mint.rng)
        if consonants not in roots:
            roots.append(consonants)
    for consonants in roots:
        trial.forms.extend((shape(consonants, vowel, vowel),
                            shape(consonants, other, other)))
    for consonants in roots[:2]:
        trial.bound.append((shape(consonants, vowel, vowel),
                            shape(consonants, other, other), "past"))
    # A three-consonant root has 2197 shapes and this draws twenty of them, so a wug root
    # colliding with a lesson root is unlikely and not rare enough to leave to luck: it happened
    # at one seed in three, and a wug item she has already been shown is not a wug item. The
    # seventh flaw this bank found in itself, and the first that flattered the score.
    fresh: List[str] = []
    while len(fresh) < 6:
        consonants = root(mint.rng)
        if consonants not in roots and consonants not in fresh:
            fresh.append(consonants)
    for consonants in fresh:
        trial.wug.append((shape(consonants, vowel, vowel), "past",
                          shape(consonants, other, other)))
    trial.note = "no affix anywhere — the change is inside the word"
    return trial


def _syncretism(mint: Mint) -> Trial:
    """One ending, two meanings, told apart by something else in the clause."""
    ending, marker = mint.markers_n(2)
    verbs = mint.words(3)

    def say(s: str, v: str, o: str, *, many: bool) -> Tuple[str, Meaning]:
        # The same suffix marks plural on a noun and past on a verb. Which is which is decided by
        # where it sits, and a clause carrying the free marker is unambiguously the plural one.
        if many:
            return f"{marker} {s}{ending} {v} {o}", _meaning(subject=s, verb=v, object=o,
                                                             number="plural")
        return f"{s} {v}{ending} {o}", _meaning(subject=s, verb=v, object=o, temporal="past")

    trial = Trial(language="syncretism")
    for index in range(14):
        trial.lesson.append(say(mint.word(), verbs[index % 3], mint.word(), many=index % 2 == 0))
    for index in range(8):
        trial.exam.append(say(mint.word(), verbs[index % 3], mint.word(), many=index % 2 == 1))
    return trial


def _ergative(mint: Mint) -> Trial:
    """The subject's case depends on whether the clause has an object at all."""
    erg, absol = mint.markers_n(2)
    verbs = mint.words(4)

    def transitive(s: str, v: str, o: str) -> Tuple[str, Meaning]:
        return f"{s}{erg} {o}{absol} {v}", _meaning(subject=s, verb=v, object=o)

    def intransitive(s: str, v: str) -> Tuple[str, Meaning]:
        return f"{s}{absol} {v}", _meaning(subject=s, verb=v)

    trial = Trial(language="ergative")
    for index in range(14):
        if index % 2:
            trial.lesson.append(transitive(mint.word(), verbs[index % 4], mint.word()))
        else:
            trial.lesson.append(intransitive(mint.word(), verbs[index % 4]))
    for index in range(8):
        if index % 2:
            trial.exam.append(transitive(mint.word(), verbs[index % 4], mint.word()))
        else:
            trial.exam.append(intransitive(mint.word(), verbs[index % 4]))
    return trial


def _classifier(mint: Mint) -> Trial:
    """A number cannot touch a noun directly: a classifier has to sit between them."""
    classifier = mint.marker(2)
    numbers = mint.words(3)
    verbs = mint.words(3)

    def say(count: str, s: str, v: str, o: str) -> Tuple[str, Meaning]:
        return (f"{count} {classifier} {s} {v} {o}",
                _meaning(subject=s, verb=v, object=o, count=count))

    trial = Trial(language="classifier")
    for index in range(12):
        trial.lesson.append(say(numbers[index % 3], mint.word(), verbs[index % 3], mint.word()))
    for index in range(8):
        trial.exam.append(say(numbers[index % 3], mint.word(), verbs[index % 3], mint.word()))
    return trial


def _compound(mint: Mint) -> Trial:
    """Two nouns written as one word, and the whole is one argument.

    The heads come from a pool the lesson also uses as bare objects, and that is a fix to this
    problem rather than a kindness. Two **novel** words written together have a boundary nothing
    can see: ``benuapseliao`` divides after six letters, or after five, or after eight, and no
    evidence in the sentence or out of it prefers one. She answered ``ambiguous`` to all eight,
    which was right, and an item whose only correct answer is "I cannot tell" measures the item.
    With one half a word she has met, the boundary is exactly where that word starts — which is
    the ability the problem is supposed to be about.
    """
    verbs = mint.words(3)
    nouns = mint.words(6)

    def say(head: str, modifier: str, v: str, o: str) -> Tuple[str, Meaning]:
        return f"{modifier}{head} {v} {o}", _meaning(subject=head, verb=v, object=o,
                                                     subject_mod=modifier)

    trial = Trial(language="compound")
    for index in range(12):
        trial.lesson.append(say(nouns[index % len(nouns)], mint.word(), verbs[index % 3],
                                nouns[(index + 3) % len(nouns)]))
    for index in range(8):
        trial.exam.append(say(nouns[(index + 1) % len(nouns)], mint.word(), verbs[index % 3],
                              nouns[(index + 4) % len(nouns)]))
    trial.note = "no space between the two halves; one of them is a word she has met"
    return trial


def _double_marking(mint: Mint) -> Trial:
    """Two features on one word, stacked, in a fixed order."""
    plural, past = mint.markers_n(2)
    verbs = mint.words(3)

    def say(s: str, v: str, o: str, *, many: bool, then: bool) -> Tuple[str, Meaning]:
        ending = (plural if many else "") + (past if then else "")
        return (f"{s} {v}{ending} {o}",
                _meaning(subject=s, verb=v, object=o, temporal="past" if then else "",
                         **({"number": "plural"} if many else {})))

    trial = Trial(language="double-marking")
    for index in range(15):
        many, then = (False, False) if index % 3 == 0 else \
            ((True, False) if index % 3 == 1 else (False, True))
        trial.lesson.append(say(mint.word(), verbs[index % 3], mint.word(),
                                many=many, then=then))
    for index in range(8):
        trial.exam.append(say(mint.word(), verbs[index % 3], mint.word(),
                              many=True, then=True))
    trial.note = "the two endings have never been seen on one word"
    return trial


BANK: Tuple[Problem, ...] = (
    Problem("ditransitive", "an agent, a theme and a recipient",
            "ROLES holds three names and a clause here has four", _ditransitive),
    Problem("oblique", "a place, on top of a full clause",
            "a fourth argument with an adposition of its own", _oblique),
    Problem("modifier", "an adjective on the subject",
            "a phrase where the grammar expects a word", _modifier),
    Problem("coordination", "two subjects joined by a particle",
            "one role filled by two things", _coordination),
    Problem("possessive", "a possessor attached to its noun",
            "an argument that belongs to another argument", _possessive),
    Problem("relative", "a clause inside a noun phrase",
            "one sentence embedded in another", _relative),
    Problem("complement", "a clause as the object of a verb",
            "recursion, in the place an object goes", _complement),
    Problem("agreement", "plural and past, in a cell nobody demonstrated",
            "features that have to compose rather than be listed", _agreement),
    Problem("scrambling", "the two orders she was not shown",
            "case says the roles, so order is free — nothing licences that", _scrambling),
    Problem("pro-drop", "the subject recovered from the verb's ending",
            "an argument that is not in the sentence at all", _pro_drop),
    Problem("harmony", "one plural, two shapes, chosen by the stem",
            "allomorphy — Morphology has one form per feature", _harmony),
    Problem("circumfix", "a past tense that wraps the stem",
            "induction is prefix or suffix, never both at once", _circumfix),
    Problem("infix", "a plural inserted after the first syllable",
            "the affix is not at either end", _infix),
    Problem("reduplication", "the plural is the stem twice",
            "the affix is the stem, which no affix table can hold", _reduplication),
    Problem("partial-reduplication", "the first syllable copied to the front",
            "the affix is part of the stem", _partial_reduplication),
    Problem("templatic", "consonants are the word, vowels are the grammar",
            "nothing is concatenated anywhere", _templatic),
    Problem("syncretism", "one ending, two meanings",
            "the same form means different things in different places", _syncretism),
    Problem("ergative", "case on the subject depends on transitivity",
            "a marker chosen by a property of the whole clause", _ergative),
    Problem("classifier", "a number needs a classifier between it and its noun",
            "a word required by another word rather than by the meaning", _classifier),
    Problem("compound", "two nouns written as one word",
            "one token holding two roles", _compound),
    Problem("double-marking", "two endings stacked on one word",
            "a combination of affixes nobody demonstrated together", _double_marking),
)



# --------------------------------------------------------------------------- #
# bank two — chosen after the axes existed, from a different corner of the subject
# --------------------------------------------------------------------------- #

def _apophony(mint: Mint) -> Trial:
    """*sing* / *sang*: the vowel changes and nothing is added."""
    out = mint.rng.choice("aeiou")

    def change(stem: str) -> str:
        return "".join(out if char in _VOWELS else char for char in stem)

    trial = Trial(language="apophony")
    stems = [mint.rng.choice(_ONSETS) + mint.rng.choice("ie")
             + mint.rng.choice(_ONSETS) + mint.rng.choice("ie")
             + mint.rng.choice(_ONSETS) for _ in range(14)]
    stems = [stem for stem in stems if change(stem) != stem]
    for stem in stems:
        trial.forms.extend((stem, change(stem)))
    trial.bound = [(stems[0], change(stems[0]), "past"), (stems[1], change(stems[1]), "past")]
    for _ in range(6):
        stem = (mint.rng.choice(_ONSETS) + mint.rng.choice("ie") + mint.rng.choice(_ONSETS)
                + mint.rng.choice("ie") + mint.rng.choice(_ONSETS))
        trial.wug.append((stem, "past", change(stem)))
    return trial


def _subtractive(mint: Mint) -> Trial:
    """The plural is the singular with its last letter taken off."""

    def plural(stem: str) -> str:
        return stem[:-1]

    trial = Trial(language="subtractive")
    stems = mint.words(14)
    for stem in stems:
        trial.forms.extend((stem, plural(stem)))
    trial.bound = [(stems[0], plural(stems[0]), "plural"),
                   (stems[1], plural(stems[1]), "plural")]
    for _ in range(6):
        stem = mint.word()
        trial.wug.append((stem, "plural", plural(stem)))
    trial.note = "the marker is an absence"
    return trial


def _metathesis(mint: Mint) -> Trial:
    """The last two letters swap places."""

    def swap(stem: str) -> str:
        return stem[:-2] + stem[-1] + stem[-2]

    trial = Trial(language="metathesis")
    stems = [word for word in mint.words(16) if word[-1] != word[-2]]
    for stem in stems:
        trial.forms.extend((stem, swap(stem)))
    trial.bound = [(stems[0], swap(stems[0]), "plural"), (stems[1], swap(stems[1]), "plural")]
    for _ in range(6):
        stem = mint.word()
        while stem[-1] == stem[-2]:
            stem = mint.word()
        trial.wug.append((stem, "plural", swap(stem)))
    trial.note = "nothing is added, removed or changed — two things move"
    return trial


def _lengthening(mint: Mint) -> Trial:
    """The stem's last vowel is written twice."""

    def lengthen(stem: str) -> str:
        for index in range(len(stem) - 1, -1, -1):
            if stem[index] in _VOWELS:
                return stem[: index + 1] + stem[index] + stem[index + 1:]
        return stem

    trial = Trial(language="lengthening")
    stems = mint.words(14)
    for stem in stems:
        trial.forms.extend((stem, lengthen(stem)))
    trial.bound = [(stems[0], lengthen(stems[0]), "plural"),
                   (stems[1], lengthen(stems[1]), "plural")]
    for _ in range(6):
        stem = mint.word()
        trial.wug.append((stem, "plural", lengthen(stem)))
    trial.note = "the position of the change depends on the stem"
    return trial


def _concord(mint: Mint) -> Trial:
    """One number, marked in three places at once."""
    plural = mint.marker()
    verbs = mint.words(3)

    def say(s: str, v: str, o: str, *, many: bool) -> Tuple[str, Meaning]:
        mark = plural if many else ""
        return (f"{s}{mark} {v}{mark} {o}{mark}",
                _meaning(subject=s, verb=v, object=o,
                         **({"number": "plural"} if many else {})))

    trial = Trial(language="concord")
    for index in range(14):
        trial.lesson.append(say(mint.word(), verbs[index % 3], mint.word(),
                                many=index % 2 == 0))
    for index in range(8):
        trial.exam.append(say(mint.word(), verbs[index % 3], mint.word(),
                              many=index % 2 == 1))
    trial.note = "one feature, three pieces of material"
    return trial


def _second_position(mint: Mint) -> Trial:
    """A clitic that always sits after the first word, whichever word that is."""
    clitic = mint.marker(2)
    verbs = mint.words(3)

    def say(s: str, v: str, o: str, *, fronted: bool) -> Tuple[str, Meaning]:
        words = [o, s, v] if fronted else [s, v, o]
        words.insert(1, clitic)
        return " ".join(words), _meaning(subject=s, verb=v, object=o, mood="reported")

    trial = Trial(language="second-position")
    for index in range(14):
        trial.lesson.append(say(mint.word(), verbs[index % 3], mint.word(),
                                fronted=index % 2 == 0))
    for index in range(8):
        subject, verb, obj = mint.word(), verbs[index % 3], mint.word()
        surface, meaning = say(subject, verb, obj, fronted=index % 2 == 1)
        # Both orders say the same thing here — the clitic is the only thing this language marks,
        # and fronting carries no meaning. So both are accepted, for the reason the free-order
        # problem accepts six: grading production against the one the renderer chose marks a right
        # answer wrong. Same flaw, second time, which is why it is worth naming as a class.
        trial.exam.append(Item(surface, meaning, (
            say(subject, verb, obj, fronted=True)[0],
            say(subject, verb, obj, fronted=False)[0])))
    trial.note = "the clitic's place is structural, not a slot"
    return trial


def _polypersonal(mint: Mint) -> Trial:
    """The verb marks who did it and who it was done to; neither is a word."""
    who, whom = mint.markers_n(2)
    speaker, hearer, other = mint.words(3)
    # Four verbs against three cells, so the verb and the cell fall out of step. With three of
    # each they never do: every demonstration of a cell used the same verb, so each shape was five
    # copies of one sentence, and she refused all three — "identical fillers" — which is exactly
    # what a shape supported only by one repeated sentence should get. The sixth flaw this bank
    # found in itself.
    verbs = mint.words(4)

    def say(v: str, *, agent_is_speaker: bool, patient_is_hearer: bool) -> Tuple[str, Meaning]:
        return (f"{v}{who if agent_is_speaker else ''}"
                f"{whom if patient_is_hearer else ''}",
                _meaning(subject=speaker if agent_is_speaker else other, verb=v,
                         object=hearer if patient_is_hearer else other))

    trial = Trial(language="polypersonal")
    for index in range(15):
        agent, patient = (True, False) if index % 3 == 0 else \
            ((False, True) if index % 3 == 1 else (False, False))
        trial.lesson.append(say(verbs[index % 4], agent_is_speaker=agent,
                                patient_is_hearer=patient))
    for index in range(8):
        trial.exam.append(say(verbs[index % 4], agent_is_speaker=True, patient_is_hearer=True))
    trial.note = "both markers at once, a cell no lesson contains"
    return trial


def _incorporation(mint: Mint) -> Trial:
    """The object joins the verb and the two are written as one word."""
    verbs = mint.words(3)
    nouns = mint.words(6)

    def say(s: str, v: str, o: str) -> Tuple[str, Meaning]:
        return f"{s} {o}{v}", _meaning(subject=s, verb=v, object=o)

    trial = Trial(language="incorporation")
    for index in range(12):
        trial.lesson.append(say(mint.word(), verbs[index % 3], nouns[index % len(nouns)]))
        trial.heard.append(nouns[index % len(nouns)])
    for index in range(8):
        trial.exam.append(say(mint.word(), verbs[index % 3], nouns[(index + 2) % len(nouns)]))
    return trial


def _v2_negation(mint: Mint) -> Trial:
    """Denial does not add a word — it moves the verb to the front."""
    verbs = mint.words(3)

    def say(s: str, v: str, o: str, *, denied: bool) -> Tuple[str, Meaning]:
        words = [v, s, o] if denied else [s, v, o]
        return " ".join(words), _meaning(subject=s, verb=v, object=o, negated=denied)

    trial = Trial(language="v2-negation")
    for index in range(14):
        trial.lesson.append(say(mint.word(), verbs[index % 3], mint.word(),
                                denied=index % 2 == 0))
    for index in range(8):
        trial.exam.append(say(mint.word(), verbs[index % 3], mint.word(),
                              denied=index % 2 == 1))
    trial.note = "the negator is a word order"
    return trial


def _inversion_question(mint: Mint) -> Trial:
    """Asking is done by swapping the first two words and nothing else."""
    verbs = mint.words(3)

    def say(s: str, v: str, o: str, *, asked: bool) -> Tuple[str, Meaning]:
        words = [v, s, o] if asked else [s, v, o]
        return (" ".join(words),
                _meaning("polar_question" if asked else "assertion",
                         focus="truth" if asked else "", subject=s, verb=v, object=o))

    trial = Trial(language="inversion-question")
    for index in range(14):
        trial.lesson.append(say(mint.word(), verbs[index % 3], mint.word(),
                                asked=index % 2 == 0))
    for index in range(8):
        trial.exam.append(say(mint.word(), verbs[index % 3], mint.word(),
                              asked=index % 2 == 1))
    return trial


def _case_stacking(mint: Mint) -> Trial:
    """Two case endings on one noun, in a fixed order, each meaning its own thing."""
    of, at = mint.markers_n(2)
    verbs = mint.words(3)

    def say(s: str, v: str, o: str, *, owned: bool, placed: bool) -> Tuple[str, Meaning]:
        ending = (of if owned else "") + (at if placed else "")
        roles: Dict[str, str] = {}
        if owned:
            roles["relation_kind"] = "owned"
        if placed:
            roles["relation_place"] = "at"
        return f"{s} {v} {o}{ending}", _meaning(subject=s, verb=v, object=o, **roles)

    trial = Trial(language="case-stacking")
    for index in range(15):
        owned, placed = (False, False) if index % 3 == 0 else \
            ((True, False) if index % 3 == 1 else (False, True))
        trial.lesson.append(say(mint.word(), verbs[index % 3], mint.word(),
                                owned=owned, placed=placed))
    for index in range(8):
        trial.exam.append(say(mint.word(), verbs[index % 3], mint.word(),
                              owned=True, placed=True))
    return trial


def _long_distance(mint: Mint) -> Trial:
    """The question word is at the front and the hole it belongs to is two clauses in."""
    wh, that = mint.markers_n(2, 2)
    says = mint.word()
    verbs = mint.words(3)

    def say(s: str, cs: str, cv: str) -> Tuple[str, Meaning]:
        return (f"{wh} {s} {says} {that} {cs} {cv}",
                _meaning("question", focus="comp_object", subject=s, verb=says,
                         comp_subject=cs, comp_verb=cv))

    trial = Trial(language="long-distance")
    for index in range(12):
        trial.lesson.append(say(mint.word(), mint.word(), verbs[index % 3]))
    for index in range(8):
        trial.exam.append(say(mint.word(), mint.word(), verbs[index % 3]))
    trial.note = "the gap is inside the embedded clause"
    return trial


def _honorific(mint: Mint) -> Trial:
    """A register, marked on the subject and on the verb together."""
    polite = mint.marker()
    verbs = mint.words(3)

    def say(s: str, v: str, o: str, *, formal: bool) -> Tuple[str, Meaning]:
        mark = polite if formal else ""
        return (f"{s}{mark} {v}{mark} {o}",
                _meaning(subject=s, verb=v, object=o,
                         **({"register": "formal"} if formal else {})))

    trial = Trial(language="honorific")
    for index in range(14):
        trial.lesson.append(say(mint.word(), verbs[index % 3], mint.word(),
                                formal=index % 2 == 0))
    for index in range(8):
        trial.exam.append(say(mint.word(), verbs[index % 3], mint.word(),
                              formal=index % 2 == 1))
    trial.note = "one feature on two words"
    return trial


def _noun_class(mint: Mint) -> Trial:
    """Which classifier a noun takes depends on which class the noun is in."""
    first, second = mint.markers_n(2, 2)
    verbs = mint.words(3)
    ones = mint.words(5)
    twos = mint.words(5)

    def say(s: str, v: str, o: str) -> Tuple[str, Meaning]:
        marker = first if s in ones else second
        return f"{marker} {s} {v} {o}", _meaning(subject=s, verb=v, object=o)

    trial = Trial(language="noun-class")
    for index in range(16):
        pool = ones if index % 2 else twos
        trial.lesson.append(say(pool[index % len(pool)], verbs[index % 3], mint.word()))
    for index in range(8):
        pool = ones if index % 2 else twos
        trial.exam.append(say(pool[(index + 2) % len(pool)], verbs[index % 3], mint.word()))
    trial.note = "the class is a property of the word, not of the sentence"
    return trial


def _mixed_irregulars(mint: Mint) -> Trial:
    """Most stems take the ending; three do something else entirely."""
    ending = mint.marker()
    regular = mint.words(14)
    odd = mint.words(3)
    strange = {stem: mint.word() for stem in odd}

    trial = Trial(language="mixed-irregulars")
    for stem in regular:
        trial.forms.extend((stem, stem + ending))
    for stem, form in strange.items():
        trial.forms.extend((stem, form))
    trial.bound = [(regular[0], regular[0] + ending, "plural"),
                   (regular[1], regular[1] + ending, "plural")]
    trial.bound.extend((stem, form, "plural") for stem, form in strange.items())
    for _ in range(4):
        stem = mint.word()
        trial.wug.append((stem, "plural", stem + ending))
    for stem, form in strange.items():
        trial.wug.append((stem, "plural", form))
    trial.note = "an irregular must not leak onto a regular stem, or the other way round"
    return trial


SECOND_BANK = (
    Problem("apophony", "the vowel changes and nothing is added",
            "a stem change, not an affix", _apophony),
    Problem("subtractive", "the plural is the singular minus its last letter",
            "the marker is an absence", _subtractive),
    Problem("metathesis", "the last two letters swap",
            "nothing is added, removed or changed", _metathesis),
    Problem("lengthening", "the stem's last vowel is written twice",
            "where the change lands depends on the stem", _lengthening),
    Problem("concord", "one number, marked in three places",
            "one feature carried by material on three slots", _concord),
    Problem("second-position", "a clitic after the first word, whichever it is",
            "a position defined by structure rather than by a slot", _second_position),
    Problem("polypersonal", "the verb marks both arguments, neither is a word",
            "two dropped arguments, each with its own marker", _polypersonal),
    Problem("incorporation", "the object is written joined to the verb",
            "a compound across two roles rather than inside one", _incorporation),
    Problem("v2-negation", "denial moves the verb instead of adding a word",
            "a feature carried by word order", _v2_negation),
    Problem("inversion-question", "asking swaps the first two words",
            "a speech act carried by word order", _inversion_question),
    Problem("case-stacking", "two case endings on one noun",
            "two markers on one slot, from two dimensions", _case_stacking),
    Problem("long-distance", "the question word is two clauses from its hole",
            "a focus naming a role of an embedded clause", _long_distance),
    Problem("honorific", "a register marked on two words at once",
            "one feature carried by material on two slots", _honorific),
    Problem("noun-class", "the classifier depends on which class the noun is in",
            "a marker chosen by a lexical property of a filler", _noun_class),
    Problem("mixed-irregulars", "three odd stems among fourteen regular ones",
            "an irregular that must not leak, beside a rule that must", _mixed_irregulars),
)

EVERY_PROBLEM = BANK + SECOND_BANK


def build(problem: Problem, seed: int) -> Trial:
    """Mint this problem's language. Deterministic given the seed."""
    return problem.make(Mint(random.Random(seed)))


def grade(problem: Problem, faculty: Any, seed: int) -> Report:
    """Teach the lesson, then examine on what no lesson contained.

    The faculty is handed the trial exactly as the school hands it a syllabus: a word list to
    overhear, the pairs a lesson binds, the demonstrations with their meanings, and then nothing
    else. Everything after ``learn`` is held out.
    """
    trial = build(problem, seed)
    report = Report(problem=problem.id, title=problem.title, note=trial.note)
    tongue = f"{problem.id}-{seed}"
    try:
        if trial.forms:
            faculty.hear_words(trial.forms, tongue=tongue)
        for text in trial.heard:
            faculty.hear(text, tongue=tongue)
        for surface, meaning in trial.lesson:
            faculty.show(surface, meaning, tongue=tongue)
        for base, inflected, feature in trial.bound:
            faculty.bind(base, inflected, feature, tongue=tongue)
        faculty.learn(tongue=tongue)
    except Exception as exc:  # noqa: BLE001 — a faculty that raises has failed the problem
        report.note = f"{report.note} · teaching raised {type(exc).__name__}: {exc}".strip(" ·")
        return report

    for entry in trial.exam:
        # A builder may hand back a plain (surface, meaning) pair or a full Item. The pair is the
        # common case and the Item is for a language where more than one surface says the thing.
        item = entry if isinstance(entry, Item) else Item(*entry)
        surface, meaning = item.surface, item.meaning
        report.read_total += 1
        try:
            got = faculty.read(surface, tongue=tongue)
        except Exception:  # noqa: BLE001
            got = None
        if agrees(got, meaning):
            report.read_right += 1
        elif len(report.misses) < 6:
            report.misses.append(f"read {surface!r} → {None if got is None else got.to_dict()}")

        report.said_total += 1
        try:
            said = faculty.say(meaning, tongue=tongue)
        except Exception:  # noqa: BLE001
            said = ""
        if said in item.sayable:
            report.said_right += 1
        elif len(report.misses) < 6:
            report.misses.append(f"say {role_map(meaning)} → {said!r}, "
                                 f"wanted one of {item.sayable}")

    for stem, feature, want in trial.wug:
        report.wug_total += 1
        try:
            got = faculty.inflect(stem, feature, tongue=tongue)
        except Exception:  # noqa: BLE001
            got = ""
        if got == want:
            report.wug_right += 1
        elif len(report.misses) < 6:
            report.misses.append(f"wug {stem!r}+{feature} → {got!r}, wanted {want!r}")
    return report


def sweep(bank: Sequence[Problem] = BANK, *, seeds: Sequence[int] = (7,),
          faculty_factory: Optional[Callable[[], Any]] = None) -> List[Report]:
    """Every problem, at every seed, on a **fresh** faculty each time.

    Fresh because these are twenty-one different languages and a faculty that met all of them
    would be answering with a grammar assembled out of twenty-one others. One problem, one
    student, one language — which is also the only arrangement in which a failure names the
    problem it failed on.
    """
    if faculty_factory is None:
        from nyxara.njp.language import LanguageFaculty
        faculty_factory = LanguageFaculty
    out: List[Report] = []
    for problem in bank:
        for seed in seeds:
            out.append(grade(problem, faculty_factory(), seed))
    return out


def summary(reports: Sequence[Report]) -> str:
    """The report card: what was solved, and what each unsolved one is short of."""
    lines = ["", "NYXARA · NJP — hard unseen language problems", ""]
    lines.append("  problem                  read      said      wug    verdict")
    lines.append("  " + "-" * 66)
    for report in reports:
        verdict = "SOLVED" if report.solved else f"{report.score:.2f}"
        lines.append(f"  {report.problem:<22} {report.read_right:>3}/{report.read_total:<4} "
                     f"{report.said_right:>3}/{report.said_total:<4} "
                     f"{report.wug_right:>3}/{report.wug_total:<4}  {verdict}")
    solved = sum(1 for report in reports if report.solved)
    right = sum(r.read_right + r.said_right + r.wug_right for r in reports)
    total = sum(r.read_total + r.said_total + r.wug_total for r in reports)
    lines.append("")
    lines.append(f"  solved        {solved} / {len(reports)} problems")
    lines.append(f"  items         {right} / {total}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m nyxara.njp.hard [--seed N] [--second] [--json]``."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Hard unseen language problems.")
    parser.add_argument("--seed", type=int, nargs="*", default=[7])
    parser.add_argument("--second", action="store_true", help="the bank written afterwards")
    parser.add_argument("--all", action="store_true", help="both banks")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    bank = EVERY_PROBLEM if args.all else (SECOND_BANK if args.second else BANK)
    reports = sweep(bank, seeds=args.seed)
    if args.json:
        print(json.dumps([report.to_dict() for report in reports], indent=2))
    else:
        print(summary(reports))
        for report in reports:
            if not report.solved and report.misses:
                print(f"\n  {report.problem}: {report.misses[0][:160]}")
    return 0 if all(report.solved for report in reports) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
