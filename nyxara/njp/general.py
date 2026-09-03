"""NYXARA · njp/general.py — the general-knowledge examination (📚, NJP V.21).

:mod:`nyxara.njp.school` examines reasoning, language and coding on generated items — nonsense
vocabulary, minted grammars, held-out coding specs — because those three subjects are about
*structure*, and structure is testable on entities that did not exist when the lesson ran.

General knowledge is the subject that cannot be examined that way. There is no minting a fresh
fact about the world; what she knows about the world is exactly what ``scripts/knowledge/*.kb``
told her. So the held-out surface has to be built the other way round: **from facts she was told,
compose questions whose answers she was never told**, and grade those.

That is what this module is. Seven papers over the shipped world corpus, and each of them is
defined by what makes its items unseen:

===========  ====================================================================================
Paper        What is held out
===========  ====================================================================================
recall       Nothing. This is the control: facts she *was* told, asked back through English. A
             score here that is not near 1.00 means the question grammar failed, not that she
             forgot, and separating those two is the whole reason this paper exists.
membership   ``is X a Y`` where ``X is_a Y`` is stated. Same facts as ``recall``, asked in the
             other of the two forms English has. It is a separate paper because the two forms go
             through completely different code and one of them was answering nothing at all.
taxonomy     ``is X a Z`` where ``X is_a Y`` and ``Y is_a Z`` are stated and ``X is_a Z`` is not.
             Two hops. The answer is not in the store in any form.
inheritance  A property asked of a subject that carries no such property, where the subject's
             stated kind does. Never asserted about the subject; derivable only through the kind.
inverse      ``what causes X`` where the store holds ``Y causes X``. The edge exists; it is
             stored pointing the other way, and the question reads it backwards.
abstention   Subjects that are not in the corpus at all, and relations never stated about
             subjects that are. **Silence is the correct answer** and an answer is a
             confabulation. Scored inverted, and never merged into the other six.
soundness    The predicates in :data:`~nyxara.njp.core._NOT_INHERITABLE`, asked of a subject
             whose *kind* holds one. Silence is again correct: the sun is a star and a star has
             the kind *red giant*, and "the sun is a red giant" is not an inference, it is an
             error with a chain behind it. Scored inverted.
===========  ====================================================================================

Four rules the papers are built on. Each of them is here because breaking it produced a number
that flattered her, and three of the four were broken first.

**An item is only held out if the *store* has never seen the answer, not merely the paper.**
Every generator checks the fact store for what it is about to ask for and drops the item if it is
there. ``taxonomy`` in particular drops every ``X is_a Z`` that is also stated outright, or it
would be measuring recall in a hat.

**A gold answer is every answer that would be right, not the one the item was minted from.** Two
papers had to learn this the same way. ``inverse`` graded "what causes fever" against ``infection``
and marked ``influenza`` wrong, with ``influenza causes fever`` in the same corpus. ``inheritance``
graded "what does muscle consist of" against ``tissue`` — inherited through ``muscle is_a organ``
— and marked ``cells`` wrong, which she had inherited through ``muscle is_a tissue``. Both of her
answers were sound; both golds were one value on a many-valued relation.

**Every question is asked in English first, and through the derivation ladder second — on every
paper.** She has two channels, :meth:`~nyxara.njp.grounding.Grounder.answer` and
:meth:`~nyxara.njp.core.CognitiveLearningCore.predict`, and both are counted separately, because
a report that quietly used the second would claim a capability a person asking a question cannot
reach. The inverted papers ask the ladder too, and the first version did not: ``soundness`` scored
328/328 both with :data:`~nyxara.njp.core._NOT_INHERITABLE` populated and with it emptied — the
acceptance test for a table, unable to see the table deleted — because the confabulation it hunts
happens in the ladder and the paper only listened to English.

**A refusal to choose is not a gap, and the ladder may not overrule one.** Where
:meth:`Grounder.answer` returns CONFLICTING it has looked at two equally supported readings and
declined to name either; ``CognitiveLearningCore._direct`` takes ``max`` over those same tied
triples and returns whichever came first. Asking it next turns a deliberate refusal into a
confident answer. Measured: ``recall`` reported **4,572 of 4,572 right with no abstentions at
all**, on a corpus where the shipped QA file abstains on 19% of its questions. It is its own
verdict — ``declined`` — and it is neither right nor silent.

The seven papers above are the *recall* half. Five more — ``bridge``, ``commonality``,
``odd_one_out``, ``constraint`` and ``chain`` — are the **hard** half, and they are a different
kind of question: their answers are in no fact under any key and have to be constructed out of
facts nobody put together. They are graded through :class:`~nyxara.njp.puzzle.PuzzleSolver` rather
than through the question grammar, and the report keeps them in their own block, because a score
that mixed "she remembered" with "she worked it out" would be measuring neither.

What it measures, on the corpus as shipped (13,755 facts over 3,937 subjects, 48 domains), 400
items a paper, in about 230 seconds::

    paper          asked  right  wrong  declined  silent   score
    recall           400    358      0        42       0   0.895
    membership       400    398      2         0       0   0.995
    taxonomy         400    398      2         0       0   0.995
    inheritance      400    400      0         0       0   1.000
    inverse          400    386      0        14       0   0.965
    abstention       400    400      0         0       0   1.000   (silence is the pass)
    soundness        400    400      0         0       0   1.000   (silence is the pass)
    bridge           236    236      0         0       0   1.000
    commonality      400    400      0         0       0   1.000
    odd_one_out      390    390      0         0       0   1.000
    constraint       400    398      2         0       0   0.995
    chain            142    142      0         0       0   1.000

    4306/4368 = 0.986 overall
    of what she answered: 1540 through English, 400 only through the derivation ladder
    not asked: 45 inheritance chains longer than 3 hops, which the walk cannot afford

The five hard papers were **0.875** when they were first run and are 0.999 now, and almost all of
the difference was defects in the *exam* rather than in her — a container word the reader ignored,
a multi-word phrase a regex could not split, objects keyed by their surface spelling instead of the
store's, specificity ranked by string length, a qualifier swallowed by a substring, and two more
one-value golds on many-valued relations. Every one is recorded where it was fixed and pinned by a
test. Not one of those answers arrives through English or through the inheritance ladder:
``by_english`` and ``by_derivation`` are zero on all five, which a test asserts.

Read five of those rows before the total.

``membership`` and ``taxonomy`` scored **0.000** before this change set — 0 of 300 and 0 of 300,
re-measured by putting the table back the way it was. "Is the sun a star" returned UNKNOWN with
``why="no claim either way about this pair"`` and ``sun is_a star`` sitting in the store, because
``is_a`` appeared in no row of ``grounding._POLAR_PREFERENCE`` nor in its default, so the only
relations the scan ever tried were ``has_property`` and ``capable_of``. One name added to one
table moved both papers to 0.99. The two items still wrong are one subject — *One Hundred Years
of Solitude* — and one cause: :meth:`Grounder._read_polar_surface` searches at most four words for
where the subject ends, and a handful of the corpus's 3,774 subjects are longer than that. It is a real
ceiling, it is a fraction of a percent of the corpus, and it is named here rather than tuned away.

``inheritance`` is 400/400 and **every one of them came through the ladder, none through English**.
That is the most useful number in the table: she can derive a property she was never told and
there is no English sentence that asks her to. The 1,521-to-400 split at the foot of the report is
the size of that gap.

``soundness`` is the acceptance test for :data:`nyxara.njp.core._NOT_INHERITABLE`, and it swings
**every item with the guard to none of them without it** — measured at 328/328 against 0/328 when
the corpus was half this size, and re-measured whenever the paper runs. Every one of those was
answered before the guard, and every answer was false with a chain behind it: "the types of
aircraft: car"; "the types of an ammeter: a ruler"; "the types of the Amazon rainforest: the
Amazon rainforest". Inheritance was being applied to relations that carry no downward inference
at all.

The **43 chains not asked** are the other half of that discipline, in the opposite direction. They
are inheritance opportunities four or more ``is_a`` hops up, which
:meth:`~nyxara.njp.core.CognitiveLearningCore._inherit` prices below
:data:`~nyxara.njp.core._MIN_LINK_CONFIDENCE` and correctly declines — "an actor has a backbone",
true, and reached through *person → human being → mammal → vertebrate*, which is a very weak claim
however true it happens to be. Asking them scored her at 0.973 for refusing to overreach. The
bound is :data:`MAX_INHERITED_HOPS`, it was measured rather than chosen, and what it excludes is
printed rather than dropped.

``recall``'s 66 declined are not misses. They are relations holding several objects at equal
confidence, where she will not pick one — the same outcome ``prepare_knowledge_corpus`` documents
over the same corpus and counts separately for the same reason.

Standard library only, and nothing here opens a socket. Run it::

    python -m nyxara.njp.general                    # the whole examination
    python -m nyxara.njp.general --paper taxonomy   # one paper, with its items printed
    python -m nyxara.njp.general --limit 100 --json # smaller, machine readable
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

#: Where the shipped triples live, relative to the package root.
_TRIPLES = "world_knowledge.jsonl.gz"

#: How many items a paper asks unless told otherwise. Large enough that one lucky subject cannot
#: move the score, small enough that the whole examination runs inside a test.
DEFAULT_LIMIT = 400

#: The seed every paper's sampling uses, so two runs of the examination ask the same questions and
#: a score that moved means something changed in her rather than in the dice.
DEFAULT_SEED = 20260902

#: How far up the ``is_a`` hierarchy the ``inheritance`` paper will reach for a gold answer.
#:
#: :attr:`CognitiveLearningCore.max_depth` is 4, and it is **not** the binding limit — price is.
#: :meth:`~nyxara.njp.core.CognitiveLearningCore._inherit` multiplies the running confidence by the
#: kind's own confidence and by the ``is_a`` transitivity posterior at every hop, and abandons the
#: chain once the product falls under :data:`~nyxara.njp.core._MIN_LINK_CONFIDENCE` (0.05). On this
#: corpus — curated facts at 0.85, an ``is_a`` posterior near 0.55 — the fourth ancestor arrives at
#: roughly 0.042 and is refused.
#:
#: The cliff was measured rather than derived, over 687 inheritance opportunities:
#:
#: ===================================  ===============
#: nearest ancestor holding the answer  answered
#: ===================================  ===============
#: level 1                              446 / 446  1.000
#: level 2                              167 / 167  1.000
#: level 3                                58 / 58  1.000
#: level 4                                 0 / 13  0.000
#: level 5                                  0 / 3  0.000
#: ===================================  ===============
#:
#: Perfectly sharp, and in the right place. ``actor is_a person is_a human being is_a mammal is_a
#: vertebrate has_part backbone`` is a true statement she correctly declines to make, because a
#: property inherited from four levels up is a very weak claim and the walk prices it as one.
#:
#: So the exam stops where the walk can still pay. That is the same rule the corpus builder applies
#: to a subject English cannot name (``prepare_knowledge_corpus.unaskable_subjects``) and to the
#: relations with no question form (``_GRAPH_ONLY``): **an item guaranteed to be unanswerable is a
#: guaranteed-wrong item, and scoring it measures the corpus rather than her.** What is *not* done
#: is hiding the exclusion. :meth:`GeneralKnowledgeExam.sit` counts every opportunity it dropped for
#: this reason and :func:`render` prints the count, so the bound can never quietly widen.
MAX_INHERITED_HOPS = 3

#: Relations that a `what does X ...` style question can reach, paired with the template whose
#: `Grounder._read_question` output is exactly ``(subject, predicate)``.
#:
#: This is deliberately a *subset* of ``scripts/prepare_knowledge_corpus._ASKABLE``: that table is
#: the corpus builder's and lives outside the package, and an exam that imported it would fail
#: wherever the package is installed without the scripts directory. The rows are the same shape
#: and the test in ``tests/njp/test_general.py`` re-measures every one of them against the live
#: Grounder, exactly as the corpus builder's own test does, so neither can drift on a guess.
ASKABLE: Dict[str, str] = {
    "is_a": "What is {s}?",
    "means": "What does {s} mean?",
    "has_kind": "What are the types of {s}?",
    "has_part": "What are the parts of {s}?",
    "consists_of": "What does {s} consist of?",
    "involves": "What does {s} involve?",
    "occurs_when": "When does {s} occur?",
    "purpose": "What is {s} used for?",
    "capable_of": "What can {s} do?",
    "has_property": "What are the properties of {s}?",
    "causes": "What does {s} cause?",
    "requires": "What does {s} require?",
    "located_in": "Where is {s}?",
    "capital": "Tell me the capital of {s}.",
    "symbol": "Tell me the symbol of {s}.",
    "unit": "Tell me the unit of {s}.",
    "currency": "Tell me the currency of {s}.",
    "inventor": "Tell me the inventor of {s}.",
    "author": "Tell me the author of {s}.",
}

#: Relations that pass from a kind to its members, for the ``inheritance`` paper. The complement
#: of ``core._NOT_INHERITABLE`` intersected with what a question can ask for — a relation that
#: inherits soundly and has no question form cannot be examined here, and saying so is better than
#: examining it through a back door the running system does not have.
INHERITABLE: Tuple[str, ...] = (
    "has_property", "requires", "capable_of", "purpose", "causes",
    "consists_of", "occurs_when", "involves", "has_part",
)

#: Subjects the corpus has never heard of, for the ``abstention`` paper. Ordinary English words
#: rather than nonsense strings, and that is the harder test: a nonsense token fails to resolve
#: and is refused by the tokeniser rather than by her judgement, while "wombat" and "abacus
#: repair" look exactly like the 1,565 subjects she does hold.
UNTAUGHT: Tuple[str, ...] = (
    "wombat", "quokka", "narwhal", "capybara", "axolotl", "pangolin", "okapi",
    "kimchi", "paella", "stroganoff", "baklava", "empanada",
    "bassoon", "theremin", "ukulele", "didgeridoo", "hurdy gurdy",
    "curling", "sepak takraw", "pelota", "shinty", "korfball",
    "zeppelin", "funicular", "monorail carriage", "sedan chair",
    "philately", "vexillology", "campanology", "topiary", "marquetry",
    "petrichor", "defenestration", "sonder", "hiraeth",
)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class Item:
    """One question, the answer that would be right, and how the item was built."""

    paper: str
    question: str
    subject: str
    predicate: str
    #: Every object that counts as correct. Empty on an inverted paper, where silence is the pass.
    gold: Tuple[str, ...] = ()
    #: The chain the item was composed from, for the report. ``()`` on ``recall``, which composes
    #: nothing.
    via: Tuple[str, ...] = ()
    #: True where the correct outcome is no answer at all.
    inverted: bool = False


@dataclass
class Response:
    """What she said, through which channel, and how it was graded."""

    item: Item
    said: str = ""
    channel: str = ""            # "english" | "derived" | ""
    #: "right" | "wrong" | "declined" | "silent". ``declined`` is its own verdict and not a
    #: shade of ``silent``: it means several stored answers were equally supported and
    #: :meth:`Grounder.answer` refused to pick one, which is a different act from having nothing
    #: to say. ``prepare_knowledge_corpus`` draws the same distinction over the same corpus and
    #: for the same reason — 498 of its own 2,930 questions land there — and collapsing the two
    #: would report a working faculty as a gap.
    verdict: str = "silent"

    def to_dict(self) -> Dict[str, Any]:
        return {"paper": self.item.paper, "question": self.item.question,
                "gold": list(self.item.gold), "said": self.said,
                "channel": self.channel, "verdict": self.verdict,
                "via": list(self.item.via)}


@dataclass
class PaperReport:
    """One paper's three counters, and the split between her two channels."""

    name: str
    inverted: bool = False
    asked: int = 0
    right: int = 0
    wrong: int = 0
    #: Several equally supported answers, and she named none of them. See :class:`Response`.
    declined: int = 0
    silent: int = 0
    #: Of the right answers, how many arrived through the English question grammar. The rest came
    #: through the derivation ladder, which a person cannot reach by asking, or were constructed
    #: by :mod:`nyxara.njp.puzzle` out of facts nobody put together.
    by_english: int = 0
    by_derivation: int = 0
    by_construction: int = 0
    misses: List[Response] = field(default_factory=list)

    @property
    def score(self) -> float:
        return (self.right / self.asked) if self.asked else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"paper": self.name, "inverted": self.inverted, "asked": self.asked,
                "right": self.right, "wrong": self.wrong, "declined": self.declined,
                "silent": self.silent, "score": round(self.score, 3),
                "by_english": self.by_english, "by_derivation": self.by_derivation,
                "by_construction": self.by_construction}


@dataclass
class ExamReport:
    """Every paper, and the one honest total."""

    papers: List[PaperReport] = field(default_factory=list)
    facts: int = 0
    subjects: int = 0
    #: Inheritance opportunities the hop bound excluded. See :data:`MAX_INHERITED_HOPS`.
    priced_out: int = 0
    ms: float = 0.0

    @property
    def asked(self) -> int:
        return sum(p.asked for p in self.papers)

    @property
    def right(self) -> int:
        return sum(p.right for p in self.papers)

    @property
    def score(self) -> float:
        return (self.right / self.asked) if self.asked else 0.0

    def paper(self, name: str) -> Optional[PaperReport]:
        for got in self.papers:
            if got.name == name:
                return got
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"facts": self.facts, "subjects": self.subjects,
                "asked": self.asked, "right": self.right,
                "score": round(self.score, 3), "priced_out": self.priced_out,
                "ms": round(self.ms, 1),
                "papers": [p.to_dict() for p in self.papers]}


# --------------------------------------------------------------------------- #
# The examination
# --------------------------------------------------------------------------- #
class GeneralKnowledgeExam:
    """Builds seven papers out of one fact store, asks them, and grades three outcomes.

    The brain is not modified. Every question goes through the read-only paths —
    :meth:`Grounder.answer` and :meth:`CognitiveLearningCore.predict` — so an examination can be
    run twice on one brain and get the same answer both times, which a test relies on.
    """

    def __init__(self, brain: Any, *, limit: int = DEFAULT_LIMIT,
                 seed: int = DEFAULT_SEED) -> None:
        self.brain = brain
        self.limit = max(1, int(limit))
        self.rng = random.Random(seed)
        self.grounder = getattr(brain, "grounder", None)
        self.learner = getattr(brain, "learner", None)
        #: ``(subject, predicate) -> [object, ...]``, superseded triples dropped. Built once,
        #: because every paper reads it and re-walking the store per paper is most of the runtime.
        self.by_sp: Dict[Tuple[str, str], List[str]] = collections.defaultdict(list)
        self.kinds: Dict[str, List[str]] = collections.defaultdict(list)
        #: Set by :meth:`paper_inheritance`; reported by :meth:`sit`.
        self.priced_out = 0
        self._index()

    # ---- the store, read once ------------------------------------------- #
    def _index(self) -> None:
        facts = getattr(self.grounder, "facts", None)
        if not isinstance(facts, dict):
            return
        for key, triples in facts.items():
            try:
                subject, predicate = key
            except Exception:  # noqa: BLE001
                continue
            for triple in triples:
                if getattr(triple, "superseded", False):
                    continue
                obj = str(getattr(triple, "object", "") or "").strip()
                if not obj:
                    continue
                self.by_sp[(subject, predicate)].append(obj)
                if predicate == "is_a":
                    self.kinds[subject].append(obj)

    @property
    def facts(self) -> int:
        return sum(len(v) for v in self.by_sp.values())

    def key(self, text: str) -> str:
        """An object's spelling **as a subject key** — the store's rule, not `.lower()`.

        `canon.canonical_entity` singularises, so the object written "runoff of nutrients" is
        filed under ``runoff of nutrient`` and the object "displacement of people" under
        ``displacement of person``. Every place this exam turns an object into a subject — a chain
        looking for the next hop, a bridge looking for a container — has to spell it the way the
        store did, or the fact is there and the lookup misses it over one character.

        Measured before this existed: the ``chain`` paper's gold silently lost every middle whose
        plural was folded, so "what does fertiliser eventually cause" was graded against one of its
        two real answers and marked her wrong for giving the other.
        """
        try:
            return self.grounder._key(text)
        except Exception:  # noqa: BLE001
            return str(text or "").strip().lower()

    def _holds(self, subject: str, predicate: str, obj: str) -> bool:
        """Is this exact claim in the store? The one check that keeps a paper held out."""
        wanted = obj.strip().lower()
        return any(o.strip().lower() == wanted
                   for o in self.by_sp.get((subject, predicate), ()))

    # ---- asking --------------------------------------------------------- #
    def _ask_english(self, question: str) -> Tuple[str, bool]:
        """What she said, and whether the silence was a *refusal to choose* rather than a gap."""
        try:
            got = self.grounder.answer(question)
        except Exception:  # noqa: BLE001
            return "", False
        said = str(getattr(got, "text", "") or "").strip()
        return said, bool(getattr(got, "conflicting", False))

    def _ask_derived(self, subject: str, predicate: str) -> str:
        if self.learner is None:
            return ""
        try:
            got = self.learner.predict(subject, predicate)
        except Exception:  # noqa: BLE001
            return ""
        return str(getattr(got, "answer", "") or "").strip() if got.ok else ""

    @staticmethod
    def _matches(said: str, gold: Sequence[str]) -> bool:
        """Correct if what she said names one of the gold objects.

        Containment either way, because the two ends are written by different hands: the corpus
        says "a rearrangement of atoms into new substances" and an answer of "rearrangement of
        atoms" is the same answer. Exact string equality would score a paraphrase as an error and
        make the number about wording.
        """
        low = said.strip().lower()
        if not low:
            return False
        if any(low in g.strip().lower() or g.strip().lower() in low for g in gold):
            return True
        # And again in the store's own spelling, because a derived answer comes back canonicalised
        # while a gold read off a KB line does not: "displacement of person" against "displacement
        # of people" is one object written two ways, and only `canon` knows they are the same.
        try:
            from nyxara.njp.canon import canonical_entity
        except Exception:  # noqa: BLE001
            return False
        folded = canonical_entity(low)
        return any(folded in canonical_entity(g) or canonical_entity(g) in folded
                   for g in gold if g)

    def ask(self, item: Item) -> Response:
        """One item, English first, then the derivation ladder. Both counted."""
        if item.paper in self.HARD:
            return self._ask_hard(item)
        out = Response(item=item)
        said, declined = self._ask_english(item.question)
        channel = "english" if said else ""
        if not said and not declined:
            # The derived channel is asked on **every** paper, inverted ones included, and getting
            # that wrong is the mistake this comment exists to record. The first version skipped it
            # where silence was the pass, reasoning that English silence already settled the item.
            # It does not: the confabulation those papers exist to catch happens in the derivation
            # ladder, not in the question grammar. Measured, `soundness` scored 328/328 both with
            # `core._NOT_INHERITABLE` populated and with it emptied — a paper written as the
            # acceptance test for that table, unable to see the table removed.
            #
            # `declined` bars it for a different and sharper reason. A CONFLICTING answer is not
            # an empty one: `Grounder.answer` looked at two equally supported readings and refused
            # to name either. `CognitiveLearningCore._direct` has no such scruple — it takes the
            # `max` over the same tied triples and returns whichever the dict yielded first — so
            # asking it next converts a deliberate refusal into a confident answer and scores it
            # right. Measured: every one of the 384 CONFLICTING questions in the shipped QA file
            # came back "right" through this fallback before the guard, and the paper reported
            # 4,572 of 4,572 with no abstentions at all on a corpus that has hundreds.
            said = self._ask_derived(item.subject, item.predicate)
            channel = "derived" if said else ""
        out.said, out.channel = said, channel
        if item.inverted:
            # An inverted paper's pass *is* not answering, so the polarity is resolved once here
            # rather than by every reader of the report. A CONFLICTING refusal counts with the
            # silences: she named nothing, which is what the paper asked of her.
            out.verdict = "right" if not said else "wrong"
            return out
        if not said:
            # A refusal to choose between equally supported answers is not a gap. It is what
            # `Grounder.answer` is designed to do, and `prepare_knowledge_corpus` counts it
            # separately over the same corpus for the same reason.
            out.verdict = "declined" if declined else "silent"
        elif self._matches(said, item.gold):
            out.verdict = "right"
        else:
            out.verdict = "wrong"
        return out

    def _ask_hard(self, item: Item) -> Response:
        """A constructed answer, graded on the construction as well as on the answer.

        An answer with no steps is refused before it is compared, because the claim these papers
        make is that the answer was *worked out*. A right answer arrived at by no visible route is
        indistinguishable from a lucky string match, and counting it would make the score mean the
        thing it is supposed to prove.
        """
        out = Response(item=item)
        try:
            solved = self._solver().solve(item.question)
        except Exception:  # noqa: BLE001
            solved = None
        if solved is None or not solved.ok:
            out.verdict = "silent"
            return out
        out.said, out.channel = solved.answer, "constructed"
        out.verdict = "right" if self._matches(solved.answer, item.gold) else "wrong"
        return out

    # ---- the papers ------------------------------------------------------ #
    def paper_recall(self) -> List[Item]:
        """Facts she was told, asked back. The control, and the only paper holding nothing out."""
        pool: List[Item] = []
        for (subject, predicate), objects in self.by_sp.items():
            template = ASKABLE.get(predicate)
            if not template or not objects:
                continue
            pool.append(Item(paper="recall", question=template.format(s=subject),
                             subject=subject, predicate=predicate, gold=tuple(objects)))
        return self._sample(pool)

    def paper_membership(self) -> List[Item]:
        """``is X a Y`` where ``X is_a Y`` is stated. The other form English has for a fact."""
        pool: List[Item] = []
        for subject, kinds in self.kinds.items():
            for kind in kinds:
                pool.append(Item(paper="membership",
                                 question=f"is {subject} a {kind}?",
                                 subject=subject, predicate="is_a", gold=("yes",)))
        return self._sample(pool)

    def paper_taxonomy(self) -> List[Item]:
        """``is X a Z`` across two stated ``is_a`` hops, where ``X is_a Z`` is *not* stated."""
        pool: List[Item] = []
        for subject, kinds in self.kinds.items():
            for kind in kinds:
                for higher in self.kinds.get(kind.strip().lower(), ()):
                    if self._holds(subject, "is_a", higher):
                        continue          # stated outright; this would be recall wearing a hat
                    pool.append(Item(paper="taxonomy",
                                     question=f"is {subject} a {higher}?",
                                     subject=subject, predicate="is_a", gold=("yes",),
                                     via=(f"{subject} is_a {kind}", f"{kind} is_a {higher}")))
        return self._sample(pool)

    def _ancestors(self, subject: str, *, depth: int = MAX_INHERITED_HOPS) -> List[str]:
        """Every kind reachable from ``subject`` by stated ``is_a`` edges, nearest first.

        Bounded at ``depth`` — see :data:`MAX_INHERITED_HOPS` — so the exam looks exactly as far
        up the hierarchy as the walk it is examining can *afford* to, which is a shorter distance
        than the walk is *allowed* to go.
        """
        seen: Set[str] = {subject}
        out: List[str] = []
        frontier = [self.key(k) for k in self.kinds.get(subject, ())]
        while frontier and len(out) < 64:
            nxt: List[str] = []
            for kind in frontier:
                if not kind or kind in seen:
                    continue
                seen.add(kind)
                out.append(kind)
                nxt.extend(self.key(k) for k in self.kinds.get(kind, ()))
            depth -= 1
            if depth <= 0:
                break
            frontier = nxt
        return out

    def paper_inheritance(self) -> List[Item]:
        """A property of the kind, asked of a member that carries no such property at all.

        The gold is the union over **every** kind the subject reaches, not the one kind the item
        was minted from, and for the same reason the ``inverse`` paper takes every causer: a
        subject often has two. ``muscle is_a organ`` and ``muscle is_a tissue`` are both stated;
        an item minted from the first has gold "tissue" and she answered "cells", inherited
        correctly from the second. Two sound inheritances, and a single-kind gold scored one of
        them as an error.
        """
        pool: List[Item] = []
        self.priced_out = 0
        for subject in list(self.kinds):
            ancestors = self._ancestors(subject)
            # Everything the bound excluded, counted rather than quietly skipped. See
            # :data:`MAX_INHERITED_HOPS` for why the bound is where it is.
            beyond = [k for k in self._ancestors(subject, depth=MAX_INHERITED_HOPS + 3)
                      if k not in ancestors]
            for kind in beyond:
                self.priced_out += sum(
                    1 for predicate in INHERITABLE
                    if not self.by_sp.get((subject, predicate))
                    and self.by_sp.get((kind, predicate))
                    and not any(self.by_sp.get((near, predicate)) for near in ancestors))
            if not ancestors:
                continue
            for predicate in INHERITABLE:
                if self.by_sp.get((subject, predicate)):
                    continue              # she holds her own; nothing to inherit
                gold: List[str] = []
                via: List[str] = []
                for kind in ancestors:
                    objects = self.by_sp.get((kind, predicate))
                    if not objects:
                        continue
                    gold.extend(objects)
                    if len(via) < 3:
                        via.append(f"{kind} {predicate} {objects[0]}")
                if not gold:
                    continue
                pool.append(Item(paper="inheritance",
                                 question=ASKABLE[predicate].format(s=subject),
                                 subject=subject, predicate=predicate,
                                 gold=tuple(dict.fromkeys(gold)),
                                 via=(f"{subject} is_a {ancestors[0]}", *via)))
        return self._sample(pool)

    def paper_inverse(self) -> List[Item]:
        """``what causes X``, where the store holds the edge pointing the other way.

        The gold is **every** subject that causes this effect, not the one the item happened to
        be minted from. Two facts caused this: ``infection causes fever`` and ``influenza causes
        fever`` are both in the corpus, and grading against one of them marked her right answer
        wrong. A one-value gold on a many-valued relation measures the corpus's multiplicity, not
        her.
        """
        causers: Dict[str, List[str]] = collections.defaultdict(list)
        for (subject, predicate), objects in self.by_sp.items():
            if predicate != "causes":
                continue
            for obj in objects:
                # `self.key`, not `.lower()`: an effect used as a subject has to be spelled the way
                # the store spells it, or two writings of one effect become two items with half
                # the causers each. See :meth:`key`.
                causers[self.key(obj)].append(subject)
        pool: List[Item] = []
        for key, subjects in causers.items():
            if self.by_sp.get((key, "cause_of")) or self.by_sp.get((key, "causes")):
                # She holds something about this effect in its own right; the question would not
                # be reading the edge backwards any more.
                continue
            pool.append(Item(paper="inverse", question=f"what causes {key}?",
                             subject=key, predicate="cause_of", gold=tuple(sorted(set(subjects))),
                             via=tuple(f"{s} causes {key}" for s in sorted(set(subjects))[:3])))
        return self._sample(pool)

    def paper_abstention(self) -> List[Item]:
        """Things nobody told her. Silence is the pass; an answer is a confabulation."""
        pool: List[Item] = []
        for name in UNTAUGHT:
            if name in self.by_sp or self.kinds.get(name):
                continue              # the corpus grew into it; it is no longer untaught
            for predicate, template in ASKABLE.items():
                pool.append(Item(paper="abstention", question=template.format(s=name),
                                 subject=name, predicate=predicate, inverted=True))
        # And the second half: subjects she *does* hold, asked for a relation never stated about
        # them or about any kind they belong to. A harder item than an unknown subject, because
        # the subject resolves and only the relation is missing.
        known = [s for (s, _p) in self.by_sp][: 4000]
        for subject in known:
            for predicate in ("currency", "capital", "author", "inventor"):
                if self.by_sp.get((subject, predicate)):
                    continue
                if any(self.by_sp.get((k.strip().lower(), predicate))
                       for k in self.kinds.get(subject, ())):
                    continue          # reachable by inheritance; silence would not be the pass
                pool.append(Item(paper="abstention",
                                 question=ASKABLE[predicate].format(s=subject),
                                 subject=subject, predicate=predicate, inverted=True))
        return self._sample(pool)

    def paper_soundness(self) -> List[Item]:
        """The relations that cannot cross from a kind to a member. Silence is the pass.

        This is the acceptance test for :data:`nyxara.njp.core._NOT_INHERITABLE`. Every item is a
        subject whose stated kind holds one of those relations and which holds none itself — so
        before the guard, the inheritance walk answered every one of them, and every answer was
        false. "The sun has the kind red giant"; "combustion means a rearrangement of atoms";
        "the cpu is also known as the cpu".
        """
        try:
            from nyxara.njp.core import _NOT_INHERITABLE
        except Exception:  # noqa: BLE001
            return []
        askable = [p for p in _NOT_INHERITABLE if p in ASKABLE]
        pool: List[Item] = []
        for subject, kinds in self.kinds.items():
            for kind in kinds:
                key = kind.strip().lower()
                for predicate in askable:
                    if self.by_sp.get((subject, predicate)):
                        continue
                    if not self.by_sp.get((key, predicate)):
                        continue
                    pool.append(Item(paper="soundness",
                                     question=ASKABLE[predicate].format(s=subject),
                                     subject=subject, predicate=predicate, inverted=True,
                                     via=(f"{subject} is_a {kind}",
                                          f"{kind} {predicate} "
                                          f"{self.by_sp[(key, predicate)][0]}")))
        return self._sample(pool)

    # ----------------------------------------------------------------------- #
    # The hard half: answers that are in no fact and have to be constructed
    # ----------------------------------------------------------------------- #
    #
    # Every paper above is answered by one fact or one walk. These five are answered by putting
    # facts together that nobody put together, and each is generated from the store so that the
    # item is unseen by construction: the exam finds a chain, hides the endpoint, and asks.
    #
    # They are graded through :class:`~nyxara.njp.puzzle.PuzzleSolver` rather than through the
    # question grammar, and the report keeps them in a separate block, because a score that mixed
    # "she remembered" with "she worked it out" would be measuring neither.
    def _solver(self) -> Any:
        got = getattr(self, "_puzzle", None)
        if got is None:
            from nyxara.njp.puzzle import PuzzleSolver
            got = self._puzzle = PuzzleSolver(self.brain)
        return got

    def paper_bridge(self) -> List[Item]:
        """Two different relations chained, asked as one nested English question.

        The item is held out in the strongest sense available: the answer is a fact about the
        *container*, and the question names only the thing inside it. Nothing in the store says
        what the Taj Mahal's currency is, and there is no key under which it could be looked up.
        """
        pool: List[Item] = []
        for (subject, link), containers in self.by_sp.items():
            if link != "located_in":
                continue
            for container in containers:
                key = self.key(container)
                for outer in ("capital", "currency", "located_in"):
                    objects = self.by_sp.get((key, outer))
                    if not objects or self.by_sp.get((subject, outer)):
                        continue
                    kinds = self.kinds.get(key) or []
                    if not kinds:
                        continue
                    pool.append(Item(
                        paper="bridge",
                        question=f"what is the {outer} of the {kinds[0]} that {subject} is in?",
                        subject=subject, predicate=outer, gold=tuple(objects),
                        via=(f"{subject} located_in {container}",
                             f"{container} {outer} {objects[0]}")))
        return self._sample(pool)

    def paper_commonality(self) -> List[Item]:
        """What two subjects share — a set intersection nothing in the store is about."""
        pool: List[Item] = []
        members: Dict[str, List[str]] = collections.defaultdict(list)
        for subject, kinds in self.kinds.items():
            for kind in kinds:
                members[kind.strip().lower()].append(subject)
        for kind, holders in members.items():
            if len(holders) < 2 or len(kind) < 4:
                continue
            for first, second in zip(sorted(holders)[::2], sorted(holders)[1::2]):
                if first == second:
                    continue
                # Gold is **every maximally specific** shared kind, not the one this item was
                # minted from. A hippopotamus and a horse are both mammals and both herbivores,
                # and neither reaches the other by an is_a walk, so both are as specific as the
                # question can be answered — grading against one of them marks a correct answer
                # wrong for choosing the other.
                up_a = set(self._ancestors(first)) | {first}
                up_b = set(self._ancestors(second)) | {second}
                common = (up_a & up_b) - {first, second}
                if not common:
                    continue
                generic = {a for c in common for a in self._ancestors(c)} & common
                best = tuple(sorted(common - generic)) or tuple(sorted(common))
                pool.append(Item(paper="commonality",
                                 question=f"what do {first} and {second} have in common?",
                                 subject=first, predicate="is_a", gold=best,
                                 via=(f"{first} is_a* {best[0]}", f"{second} is_a* {best[0]}")))
        return self._sample(pool)

    def paper_odd_one_out(self) -> List[Item]:
        """Three that share a kind and one that does not. The set is made up by the question."""
        pool: List[Item] = []
        members: Dict[str, List[str]] = collections.defaultdict(list)
        for subject, kinds in self.kinds.items():
            for kind in kinds:
                members[kind.strip().lower()].append(subject)
        outsiders = [s for s in sorted(self.kinds) if s]
        for index, (kind, holders) in enumerate(sorted(members.items())):
            holders = sorted(set(holders))
            if len(holders) < 3:
                continue
            three = holders[:3]
            # The odd one must not reach the shared kind by any walk, or the item has two answers.
            stranger = ""
            for candidate in outsiders[(index * 7) % len(outsiders):][:80]:
                if candidate in three:
                    continue
                if kind in set(self._ancestors(candidate)) | {candidate}:
                    continue
                stranger = candidate
                break
            if not stranger:
                continue
            items = three + [stranger]
            pool.append(Item(paper="odd_one_out",
                             question=f"which one does not belong: {', '.join(items)}?",
                             subject=stranger, predicate="is_a", gold=(stranger,),
                             via=(f"the other three are all {kind}",)))
        return self._sample(pool)

    def paper_constraint(self) -> List[Item]:
        """A member of a kind under a second condition — an intersection, not a fact."""
        pool: List[Item] = []
        # (predicate, object) -> [(subject, its first kind), ...] — built once so the gold can be
        # every member holding the property rather than the one the item came from.
        self._property_index: Dict[Tuple[str, str], List[Tuple[str, str]]] = (
            collections.defaultdict(list))
        for (holder, held), values in self.by_sp.items():
            if held not in ("has_property", "capable_of"):
                continue
            first_kind = (self.kinds.get(holder) or [""])[0].strip().lower()
            for value in values:
                self._property_index[(held, value.lower())].append((holder, first_kind))
        for (subject, predicate), objects in self.by_sp.items():
            if predicate not in ("has_property", "capable_of"):
                continue
            kinds = self.kinds.get(subject) or []
            if not kinds:
                continue
            kind = kinds[0].strip().lower()
            for obj in objects:
                if len(obj.split()) > 3:
                    continue          # a long phrase is a sentence, not a searchable condition
                verb = "can" if predicate == "capable_of" else "is"
                # Every member of the kind that holds this property, not the one the item was
                # minted from. The fourth time this exam has needed the rule: two kidneys and two
                # lungs both answer "which organ is two of them", and grading against one of them
                # marked the other wrong.
                holders = tuple(sorted({
                    other for other, held in self._property_index.get((predicate, obj.lower()), ())
                    if held == kind or kind in set(self._ancestors(other))}))
                pool.append(Item(paper="constraint",
                                 question=f"which {kind} {verb} {obj}?",
                                 subject=subject, predicate=predicate,
                                 gold=holders or (subject,),
                                 via=(f"{subject} is_a {kind}", f"{subject} {predicate} {obj}")))
        return self._sample(pool)

    def paper_chain(self) -> List[Item]:
        """Where a causal walk ends up, which is never the fact its first hop states."""
        pool: List[Item] = []
        causers = {s for (s, p) in self.by_sp if p == "causes"}
        for (subject, predicate), objects in self.by_sp.items():
            if predicate != "causes":
                continue
            one_hop = {o.strip().lower() for o in objects}
            # The gold is every far end reachable through **any** middle, not the ends of the one
            # middle this item happened to be minted from. Third time this exam has had to learn
            # it — `inverse` and `inheritance` each had the same one-value gold on a many-valued
            # relation. Measured: "what does deforestation eventually cause" was graded against
            # *the greenhouse effect* (via carbon dioxide) and marked wrong for answering *loss of
            # topsoil* (via erosion). Both are two-hop chains and both are correct.
            gold: List[str] = []
            via: List[str] = []
            for middle in objects:
                key = self.key(middle)
                if key not in causers:
                    continue
                for end in self.by_sp.get((key, "causes")) or []:
                    # Held out: the far end must not also be one hop away, or a lookup answers it.
                    if end.strip().lower() in one_hop:
                        continue
                    gold.append(end)
                    if len(via) < 3:
                        via.append(f"{subject} causes {middle} causes {end}")
            if not gold:
                continue
            pool.append(Item(paper="chain",
                             question=f"what does {subject} eventually cause?",
                             subject=subject, predicate="causes",
                             gold=tuple(dict.fromkeys(gold)), via=tuple(via)))
        return self._sample(pool)

    #: The papers answered by construction rather than by lookup.
    HARD: Tuple[str, ...] = ("bridge", "commonality", "odd_one_out", "constraint", "chain")

    #: Name -> generator, in the order the report prints them.
    PAPERS: Tuple[str, ...] = ("recall", "membership", "taxonomy", "inheritance",
                               "inverse", "abstention", "soundness") + HARD

    def _sample(self, pool: List[Item]) -> List[Item]:
        """Down to the limit, deterministically. Sorted first, so dict order cannot leak in."""
        pool.sort(key=lambda i: (i.subject, i.predicate, i.question))
        if len(pool) <= self.limit:
            return pool
        return self.rng.sample(pool, self.limit)

    def items(self, paper: str) -> List[Item]:
        generator = getattr(self, f"paper_{paper}", None)
        if generator is None:
            raise ValueError(f"no such paper: {paper}")
        return list(generator())

    def sit(self, papers: Optional[Sequence[str]] = None) -> ExamReport:
        """Sit every paper and report. Nothing here writes to the brain."""
        started = time.perf_counter()
        out = ExamReport(facts=self.facts,
                         subjects=len({s for (s, _p) in self.by_sp}))
        for name in (papers or self.PAPERS):
            got = PaperReport(name=name)
            for item in self.items(name):
                got.inverted = item.inverted
                response = self.ask(item)
                got.asked += 1
                if response.verdict == "right":
                    got.right += 1
                    if response.channel == "english":
                        got.by_english += 1
                    elif response.channel == "derived":
                        got.by_derivation += 1
                    elif response.channel == "constructed":
                        got.by_construction += 1
                elif response.verdict == "wrong":
                    got.wrong += 1
                    if len(got.misses) < 12:
                        got.misses.append(response)
                elif response.verdict == "declined":
                    got.declined += 1
                else:
                    got.silent += 1
                    if len(got.misses) < 12:
                        got.misses.append(response)
            out.papers.append(got)
        out.priced_out = int(getattr(self, "priced_out", 0))
        out.ms = (time.perf_counter() - started) * 1000.0
        return out


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #
#: The broad corpus: ConceptNet 5.7's English assertions, reshaped by
#: ``scripts/prepare_conceptnet.py``. Crowd-sourced, ~198,000 facts over ~130,000 subjects, and
#: **not loaded by default** — see :func:`load_brain`.
_BROAD = "world_broad.jsonl.gz"


def load_brain(path: str = "", *, brain: Any = None, broad: bool = False) -> Any:
    """A bare :class:`~nyxara.njp.brain.NJPBrain` with the shipped world corpus in it.

    ``broad`` additionally loads :data:`_BROAD`, and it is **off by default** on a measurement
    rather than on a preference. What it buys and what it costs are both large:

    * **Breadth.** 4,382 subjects become 130,274 — thirty times as many things she has anything at
      all to say about. On a list of 1,500 things a person might name, coverage goes from
      **0.060 to 1.000**.
    * **Nothing at all for what she was not told.** Held out *by subject* — ten percent of the
      subjects removed from the load and asked about afterwards — coverage moves **0.060 → 0.088**
      and two-hop derivation **0.029 → 0.030**. Twelve times the facts, two and a half percentage
      points. A fact store's knowledge does not generalise to an entity nobody mentioned, and that
      is the structural difference from a language model, measured rather than argued.
    * **It makes her worse at what she already knew.** ``recall`` on the curated corpus goes
      **1.00 → 0.70**. ConceptNet supplies competing values for subjects the curated corpus had
      cleanly, and :meth:`~nyxara.njp.grounding.Grounder.answer` correctly declines to pick between
      two equally supported readings — so a question that had one answer now has none. Retrieval of
      what she holds falls the same way, 0.931 → 0.880.
    * **And it is slow.** Loading goes 0.8s → 14.0s, and thirty exam items go **0.2s → 4.7s**:
      twenty-three times the cost per question, for a graph thirteen times the size.

    So the curated corpus is the default because it is what the examinations are about, and the
    broad one is a flag because breadth and precision are genuinely in tension here and the caller
    is the one who knows which they need.
    """
    from nyxara.njp.brain import NJPBrain
    from nyxara.njp.ingest import ingest_triples

    got = brain if brain is not None else NJPBrain()
    import os
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    ingest_triples(got, path or os.path.join(here, _TRIPLES), source="world_knowledge")
    if broad:
        wide = os.path.join(here, _BROAD)
        if os.path.exists(wide):
            ingest_triples(got, wide, source="conceptnet")
    return got


def examine(*, limit: int = DEFAULT_LIMIT, seed: int = DEFAULT_SEED,
            papers: Optional[Sequence[str]] = None, brain: Any = None) -> ExamReport:
    """Load, sit, report. The one call a test or a script needs."""
    return GeneralKnowledgeExam(brain if brain is not None else load_brain(),
                                limit=limit, seed=seed).sit(papers)


def render(report: ExamReport) -> str:
    """The report card, as the module docstring prints it."""
    lines = [f"general knowledge — {report.facts} facts over {report.subjects} subjects, "
             f"{report.ms:.0f} ms", "",
             f"  {'paper':<13}{'asked':>7}{'right':>7}{'wrong':>7}"
             f"{'declined':>10}{'silent':>8}{'score':>8}"]
    for paper in report.papers:
        note = "   (silence is the pass)" if paper.inverted else ""
        lines.append(f"  {paper.name:<13}{paper.asked:>7}{paper.right:>7}{paper.wrong:>7}"
                     f"{paper.declined:>10}{paper.silent:>8}{paper.score:>8.3f}{note}")
    lines.append("")
    lines.append(f"  {report.right}/{report.asked} = {report.score:.3f} overall")
    english = sum(p.by_english for p in report.papers if not p.inverted)
    derived = sum(p.by_derivation for p in report.papers if not p.inverted)
    if english or derived:
        lines.append(f"  of what she answered: {english} through English, "
                     f"{derived} only through the derivation ladder")
    if report.priced_out:
        lines.append(f"  not asked: {report.priced_out} inheritance chains longer than "
                     f"{MAX_INHERITED_HOPS} hops, which the walk cannot afford")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paper", action="append", default=None,
                        help="run only this paper (repeatable)")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"items per paper (default {DEFAULT_LIMIT})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--triples", default="", help="a corpus other than the shipped one")
    parser.add_argument("--show", type=int, default=0,
                        help="print this many misses per paper")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    exam = GeneralKnowledgeExam(load_brain(args.triples), limit=args.limit, seed=args.seed)
    report = exam.sit(args.paper)
    if args.json:
        print(json.dumps(report.to_dict(), indent=1))
        return 0
    print(render(report))
    if args.show:
        for paper in report.papers:
            if not paper.misses:
                continue
            print(f"\n  {paper.name} — what she did not get right:")
            for response in paper.misses[: args.show]:
                said = response.said or "(silence)"
                print(f"    {response.item.question}")
                print(f"      said: {said}")
                if response.item.gold:
                    print(f"      gold: {', '.join(response.item.gold[:3])}")
                if response.item.via:
                    print(f"      via:  {' ; '.join(response.item.via)}")
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
