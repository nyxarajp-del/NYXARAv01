"""NYXARA · njp/dialects.py — languages minted for the exam, so no rule of hers can be shipped
(🧾, NJP V.18).

:mod:`nyxara.njp.school` mints nonsense **vocabulary** so a reasoning exam cannot be answered from
memory. That is the right discipline for reasoning and it is not enough for language, because the
thing under test there is not what a word refers to but what a *shape* means — and a fresh word in
an English sentence is still an English sentence. Asked to read "the zorb chases the plag", any
extractor with an English subject-verb-object frame in it scores full marks having learned
nothing.

So this module mints the **grammar** as well. A :class:`Dialect` is a whole small language drawn
at random: which of the six orders its subject, verb and object come in; what it marks its subject
and object with, if anything; what its plural and its past tense look like; which word means *not*
and where in the clause it goes; how a yes-or-no question is marked and at which end; and which
word stands in the hole of a *what* question. Nothing about it is English, and — this is the part
that matters — nothing about it is in :mod:`nyxara.njp.language` either. Every rule the faculty
has about a dialect is a rule somebody demonstrated to it during the exam.

That gives the school two properties it could not otherwise have:

* **A cold score that means something.** The shipped compiler in :mod:`nyxara.njp.semantics` reads
  a minted dialect at zero, and so does a faculty that has been shown nothing — so a post-test
  number is not measuring a frame that happened to survive from English.
* **A ground truth that is not the thing being tested.** The dialect renders a meaning by its own
  rules, so a generated sentence is graded against the language rather than against her — and the
  grader is thirty lines of concatenation with no parser in it, which is what makes it trustworthy
  as a grader.

**What a Dialect deliberately is not.** It has no morphophonology, no agreement, no articles, no
adjectives, no embedding and one clause per sentence. It is a *sufficient* language for the six
distinctions the syllabus examines — who did what to whom, plural, past, negation, polar question,
content question — and inventing more of it would be inventing exam questions rather than
capabilities. Its affixes are drawn so that no two of them are confusable by suffix, because a
dialect that is ambiguous **to its own renderer** would grade a correct reading as wrong.

Pure standard library. Deterministic given a seed: the same seed mints the same language, so a
score is reproducible rather than anecdotal.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.semantics import Meaning

__all__ = [
    "Dialect", "Utterance", "mint_dialect", "sample", "ORDERS", "KINDS",
]

#: The six orders a clause can come in. All of them occur in real languages, and the two that
#: English is not are the ones that catch a frame written for English.
ORDERS: Tuple[Tuple[str, str, str], ...] = (
    ("S", "V", "O"), ("S", "O", "V"), ("V", "S", "O"),
    ("V", "O", "S"), ("O", "V", "S"), ("O", "S", "V"),
)

#: What an utterance can be. Each is a different claim about her grammar and each is examined
#: separately, because reading an assertion and reading its denial are not one ability — the
#: denial read as the assertion is the failure :mod:`nyxara.njp.semantics` was written for.
#:
#: ``content`` asks about the object and ``content_subject`` about the subject. They are two
#: because the thing being tested is not "is this a question" but *which slot the hole is in*, and
#: a grammar that got the first right and the second wrong would answer every question with the
#: wrong half of the sentence.
KINDS: Tuple[str, ...] = ("assertion", "negated", "past", "polar", "content", "content_subject")

_ONSETS = "bdgklmnprstvz"
_NUCLEI = "aeiou"


def _syllable(rng: random.Random) -> str:
    return rng.choice(_ONSETS) + rng.choice(_NUCLEI)


def _distinct(rng: random.Random, count: int, syllables: int = 1,
              avoid: Sequence[str] = ()) -> List[str]:
    """Forms that no other form in the language ends with, starts with, or equals.

    The condition is stronger than "different". ``-ik`` and ``-nik`` are different strings and a
    word ending in one ends in the other, so a renderer using both mints sentences whose reading
    is genuinely undecidable — and the exam would then be scoring her on an ambiguity it created.

    ``avoid`` is the half that was missing when this was first written, and the faculty is what
    found it. The affixes were drawn distinct from each other and the particles distinct from each
    other, and nothing compared the two sets — so one language minted the object marker ``-za``
    and the wh-word ``nuza``, and ``<subject>gu <verb> nuza`` is *both* "who does she V" and "she
    V's a nu". She returned ``ambiguous`` on all four such sentences rather than picking one,
    which is the behaviour this package is built for and is also how the flaw surfaced: a
    correct refusal scored as a miss for four turns before anybody looked at the language.
    """
    out: List[str] = []
    taken = list(avoid)
    while len(out) < count:
        form = "".join(_syllable(rng) for _ in range(syllables))
        if any(form.endswith(other) or other.endswith(form)
               or form.startswith(other) or other.startswith(form)
               for other in out + taken):
            continue
        out.append(form)
    return out


@dataclass(frozen=True)
class Dialect:
    """One minted language, and the rules for rendering a meaning in it.

    Every field is a decision that a language actually has to make, and every one of them is drawn
    rather than chosen, so a faculty that got the right answer got it from the lesson.
    """

    name: str = "dialect"
    order: Tuple[str, str, str] = ("S", "V", "O")
    subject_case: str = ""        #: suffix marking the subject, or "" if this language has none
    object_case: str = ""
    plural: str = "ik"            #: noun plural suffix
    past: str = "os"              #: verb past suffix
    negator: str = "nul"          #: the free morpheme meaning *not*
    negator_at: str = "before"    #: "before" | "after" | "final" — relative to the verb
    polar: str = "kaz"            #: the yes-or-no question particle
    polar_at: str = "initial"     #: "initial" | "final"
    wh_object: str = "wex"        #: the word standing where the questioned object would be
    wh_subject: str = "wun"

    # -- words -------------------------------------------------------------- #
    def noun(self, stem: str, *, role: str, plural: bool = False) -> str:
        word = f"{stem}{self.plural}" if plural else stem
        case = self.subject_case if role == "subject" else self.object_case
        return f"{word}{case}"

    def verb(self, stem: str, *, past: bool = False) -> str:
        return f"{stem}{self.past}" if past else stem

    def pluralise(self, stem: str) -> str:
        return f"{stem}{self.plural}"

    def pasten(self, stem: str) -> str:
        return f"{stem}{self.past}"

    # -- clauses ------------------------------------------------------------ #
    def express(self, subject: str, verb: str, obj: str, *, kind: str = "assertion") -> str:
        """Render one clause. ``kind`` is one of :data:`KINDS`."""
        past = kind == "past"
        slots: Dict[str, List[str]] = {
            "S": [self.noun(subject, role="subject")] if kind != "content_subject"
            else [self.wh_subject],
            "V": [self.verb(verb, past=past)],
            "O": [self.noun(obj, role="object")] if kind != "content"
            else [self.wh_object],
        }
        if kind == "negated":
            where = self.negator_at
            if where == "before":
                slots["V"] = [self.negator] + slots["V"]
            elif where == "after":
                slots["V"] = slots["V"] + [self.negator]
        words: List[str] = []
        for tag in self.order:
            words.extend(slots[tag])
        if kind == "negated" and self.negator_at == "final":
            words.append(self.negator)
        if kind == "polar":
            if self.polar_at == "initial":
                words.insert(0, self.polar)
            else:
                words.append(self.polar)
        return " ".join(words)

    def meaning(self, subject: str, verb: str, obj: str, *, kind: str = "assertion") -> Meaning:
        """The ground truth for :meth:`express`, in the package's own representation.

        Stems, never surfaces. The case marker, the plural and the tense suffix belong to the
        language rather than to what was said, and a meaning carrying them would be a meaning
        that could not be said in a second language — which is the whole thing translation tests.

        **Exactly one slot is empty in a content question, and it is the one being asked about.**
        Emptying both — which this did, for one run — leaves the object word in the surface with
        nothing accounting for it, so it is kept as *fixed material* and every sentence becomes
        its own shape. The faculty said so immediately and in the right words: eight
        demonstrations, eight signatures, every one of them "one demonstration is a sentence, not
        a shape". The third flaw this generator has had and the third one found from her side of
        the exam.
        """
        return Meaning(
            kind={"polar": "polar_question", "content": "question",
                  "content_subject": "question"}.get(kind, "assertion"),
            subject="" if kind == "content_subject" else subject,
            relation=verb,
            object="" if kind == "content" else obj,
            negated=kind == "negated",
            temporal="past" if kind == "past" else "",
            focus={"polar": "truth", "content": "object",
                   "content_subject": "subject"}.get(kind, ""),
            language=self.name,
        )

    def utter(self, subject: str, verb: str, obj: str, *,
              kind: str = "assertion") -> "Utterance":
        return Utterance(surface=self.express(subject, verb, obj, kind=kind),
                         meaning=self.meaning(subject, verb, obj, kind=kind), kind=kind)

    def forms(self) -> Tuple[str, ...]:
        """Every piece of fixed material this language uses, for the distinctness checks."""
        return tuple(form for form in (self.subject_case, self.object_case, self.plural,
                                       self.past, self.negator, self.polar,
                                       self.wh_object, self.wh_subject) if form)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "order": "".join(self.order),
                "subject_case": self.subject_case, "object_case": self.object_case,
                "plural": self.plural, "past": self.past,
                "negator": f"{self.negator}/{self.negator_at}",
                "polar": f"{self.polar}/{self.polar_at}",
                "wh": f"{self.wh_subject},{self.wh_object}"}


@dataclass(frozen=True)
class Utterance:
    """A sentence of a minted language and exactly what it says."""

    surface: str = ""
    meaning: Meaning = field(default_factory=Meaning)
    kind: str = "assertion"


def mint_dialect(rng: random.Random, name: str = "dialect",
                 avoid: Optional["Dialect"] = None) -> Dialect:
    """Draw a whole language.

    The case markers are drawn from the same pool as the plural and the past so that a language
    which marks its object and a language which marks its plural cannot accidentally use the same
    string for both — the one ambiguity that would make the dialect unreadable on its own terms.
    Whether the language marks case at all is itself a coin toss, because a language without case
    has to be read by **order alone**, and that is the harder half of the test.

    ``avoid`` mints a language that shares no form with another one. Translation is examined
    across two dialects at once, and two dialects that happened to draw the same negator would
    make the *language-identification* item genuinely undecidable — she would answer ``ambiguous``,
    correctly, and be marked wrong for it. Rare rather than impossible, and an exam should not be
    depending on a draw.
    """
    taken = list(avoid.forms()) if avoid is not None else []
    forms = _distinct(rng, 4, avoid=taken)
    particles = _distinct(rng, 4, syllables=2, avoid=forms + taken)
    marks_case = rng.random() < 0.6
    return Dialect(
        name=name,
        order=rng.choice(ORDERS),
        subject_case=forms[0] if marks_case else "",
        object_case=forms[1] if marks_case else "",
        plural=forms[2],
        past=forms[3],
        negator=particles[0],
        negator_at=rng.choice(("before", "after", "final")),
        polar=particles[1],
        polar_at=rng.choice(("initial", "final")),
        wh_object=particles[2],
        wh_subject=particles[3],
    )


def stem(dialect: Dialect, word: Callable[[], str], *, tries: int = 12) -> str:
    """A content word that does not already end in one of ``dialect``'s own markers.

    The second flaw the faculty found in this generator, and the same shape as the first: a verb
    drawn as ``tudubo`` in a language whose past tense is ``-bo`` **is** a past-tense verb by that
    language's own rules, so the present-tense sentence built from it is genuinely ambiguous and
    her past-tense reading of it was not wrong. One item in four hundred and eighty, and it is
    fixed here rather than tolerated, because a benchmark that mints undecidable items reports a
    ceiling that belongs to the benchmark.

    Redrawing is bounded: after ``tries`` the word is returned as drawn. Refusing to return one at
    all would hang an exam over a collision that a different seed will not have.
    """
    markers = tuple(marker for marker in (dialect.subject_case, dialect.object_case,
                                          dialect.plural, dialect.past) if marker)
    for _ in range(max(1, tries)):
        drawn = word()
        if not markers or not drawn.endswith(markers):
            return drawn
    return word()


def sample(dialect: Dialect, word: Callable[[], str], count: int, *,
           kinds: Sequence[str] = ("assertion",),
           verbs: Optional[Sequence[str]] = None) -> List[Utterance]:
    """``count`` utterances of ``dialect``, cycling through ``kinds``.

    ``word`` is the caller's own generator — in the school it is a :class:`~nyxara.njp.school.Mint`
    that cannot repeat itself — so every sentence a lesson uses and every sentence an exam uses is
    built from vocabulary that has never been uttered before. ``verbs`` may be held constant
    across a lesson and its exam, because a shape is a fact about a *language* and re-drawing the
    verb would be examining her on a second language rather than on what she was taught.
    """
    pool = list(verbs) if verbs else [stem(dialect, word)
                                      for _ in range(max(2, min(4, count)))]
    out: List[Utterance] = []
    for index in range(count):
        out.append(dialect.utter(stem(dialect, word), pool[index % len(pool)],
                                 stem(dialect, word), kind=kinds[index % len(kinds)]))
    return out
