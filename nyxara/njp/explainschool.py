"""NYXARA · njp/explainschool.py — the examination for *what, how and why* (🧭, NJP V.36).

:mod:`nyxara.njp.general` examines what she was **told**, through questions whose answers she was
not. This examines the two questions that module cannot ask, because no relation in the store is
called *why* and none is called *how*: an explanation is a path, and a path is either derived or
it is not there.

Seven papers, and each is defined by what makes its items unseen.

============  ======================================================================================
Paper         What is held out
============  ======================================================================================
why_chain     ``A causes B``, ``B causes T`` stated; ``A causes T`` **not** stated. Asked *why does
              T happen*, and correct only when the answer reaches **A**. Reaching B is the fact she
              was told and scores nothing — it is the control built into the paper rather than
              beside it.
why_purpose   The same two hops along ``purpose``. *What is X for*, answered with what X's purpose
              is itself for.
mechanism     A subject with parts, where a part has an effect or a purpose. *How does X work*,
              correct when the answer names a part **and** what that part does. The pairing is
              never stated about X; both halves are, about different subjects.
procedure     *What are the steps of P*, graded on the derived **order** rather than on the set. An
              order is correct when it satisfies every prerequisite — and, separately, when the
              **count** of admissible orders is right, because reporting one order for a procedure
              that has six is a different and worse error than reporting a wrong one.
minted        A procedure minted out of nonsense words, with a random prerequisite graph, examined
              five seconds after it is invented. This is the paper that says whether the ordering
              is a derivation: nothing in any corpus contains ``vrenth`` and the ground truth is
              computed by brute force over permutations, which is **not** the algorithm under test.
direction     A *why* asked of a thing with only a purpose, and a *what for* asked of a thing with
              only a cause. The wrong-kind answer is what is being hunted, and it is scored as
              **wrong**, not as missing — "the fuse is caused by a factory" is a false answer to
              *what is a fuse for*, and a system that returns it is worse than one that is silent.
abstention    Topics the corpus does not contain, and procedures nobody was told the steps of.
              **Silence is the correct answer.** Scored inverted, never merged into the other six.
============  ======================================================================================

Two rules the papers are built on, both of them the general-knowledge exam's rules applied to a
walk rather than to a lookup.

**An item is only held out if the store has never seen the answer, not merely the paper.** Every
generator checks the fact store for the whole-chain edge it is about to ask for and drops the item
if it is there. Without that, ``why_chain`` measures recall wearing a chain's clothes.

**Every question is asked in English.** :mod:`nyxara.njp.general` records ``inheritance`` at
400/400 with every one of them arriving through the derivation ladder and none through English —
a real capability that no sentence reached. The papers here go through
:meth:`~nyxara.njp.explain.Explainer.ask`, which starts at the question grammar, so a walk that
worked only from Python would score zero here rather than perfect.

What it measures, on the corpus as shipped, 200 items a paper::

    paper          asked  right  wrong  silent  score
    why_chain        200    ...
    why_purpose      200    ...
    mechanism        200    ...
    procedure         20    ...
    minted           120    ...
    direction        200    ...
    abstention       200    ...    (silence is the pass)

and a per-domain block over the forty-two domains of the corpus, because "she can explain" is a
claim about a subject and not about an average.

**Read the two halves of that block differently.** ``what``/``why``/``how`` are *coverage* — does
a question of that shape, asked about this domain's own subjects, reach an answer the store
supports — and coverage saturates when it is good, so a column of 1.00 there says the corpus
reaches every domain and says nothing about difficulty. ``held`` is the held-out papers above
restricted to this domain's subjects, and that is the number that can fall. A domain with a high
``how`` and a low ``held`` has the facts and cannot compose them.

Pure standard library, deterministic per seed.
"""

from __future__ import annotations

import itertools
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from nyxara.njp.explain import Explainer

__all__ = [
    "DOMAINS", "Item", "Response", "PaperReport", "ExamReport", "DomainReport",
    "ExplanationExam", "examine", "render", "main", "PAPERS", "DEFAULT_LIMIT", "DEFAULT_SEED",
]

DEFAULT_LIMIT = 200
DEFAULT_SEED = 20260903

PAPERS: Tuple[str, ...] = ("why_chain", "why_purpose", "mechanism", "procedure",
                           "minted", "direction", "abstention")

#: Papers where **silence is the pass** and an answer is a confabulation. Kept out of the total
#: for the same reason ``general.INVERTED`` is: averaging a paper that rewards answering with one
#: that rewards refusing produces a number that moves when neither capability does.
INVERTED: Tuple[str, ...] = ("abstention",)

#: The forty-two domains of the knowledge corpus, mapped onto the ``.kb`` files that carry them.
#: The mapping is many-to-many on purpose — biology is spread over five files and ``world.kb``
#: serves both *entities* and *places* — because the domain list is how the corpus is *asked for*
#: and the file list is how it is *written*, and forcing either into the other's shape would mean
#: a domain reported as covered by a file that barely mentions it.
DOMAINS: Dict[str, Tuple[str, ...]] = {
    "entities and relationships": ("taxonomy", "basics"),
    "general knowledge": ("basics", "everyday", "measurement"),
    "physics": ("physics",),
    "chemistry": ("chemistry",),
    "biology": ("biology", "animals", "plants", "microbes", "body"),
    "medicine": ("medicine", "health"),
    "mathematics": ("mathematics",),
    "computer science": ("computing",),
    "programming and software": ("programming",),
    "engineering": ("engineering",),
    "astronomy": ("astronomy", "space"),
    "earth science": ("earth_science", "weather", "ocean"),
    "geography": ("geography",),
    "history": ("history", "ancient", "modern"),
    "economics": ("economics",),
    "finance": ("finance",),
    "law": ("law",),
    "government and civics": ("government",),
    "psychology": ("psychology", "mind"),
    "philosophy": ("philosophy",),
    "sociology": ("sociology", "society"),
    "linguistics": ("language",),
    "literature": ("literature",),
    "arts": ("arts",),
    "music": ("music",),
    # `transport` is here rather than under engineering because what the file holds is vehicles
    # and networks — what exists and what it is for — rather than how a vehicle is designed. It
    # was unclaimed by any of the forty-two until `test_every_kb_file_is_claimed_by_a_domain`
    # said so, which is the only reason that test exists.
    "technology": ("technology", "inventions", "transport"),
    "business": ("business", "work"),
    "agriculture": ("agriculture",),
    "environment": ("environment",),
    "materials": ("materials",),
    "energy": ("energy",),
    "architecture": ("architecture", "home"),
    "education": ("education",),
    "food": ("food",),
    "sports": ("sport",),
    "culture": ("mythology", "media"),
    "everyday knowledge": ("everyday", "home"),
    "organizations": ("organizations",),
    "people": ("people",),
    "places": ("world", "india"),
    "events": ("events", "causal"),
    "procedures": ("procedures",),
}


# --------------------------------------------------------------------------- #
# Items and reports
# --------------------------------------------------------------------------- #
@dataclass
class Item:
    paper: str
    question: str
    topic: str = ""
    gold: Tuple[str, ...] = ()
    forbidden: Tuple[str, ...] = ()
    domain: str = ""
    note: str = ""
    payload: Any = None


@dataclass
class Response:
    item: Item
    said: str = ""
    right: bool = False
    wrong: bool = False
    silent: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"paper": self.item.paper, "question": self.item.question, "said": self.said,
                "right": self.right, "wrong": self.wrong, "silent": self.silent,
                "gold": list(self.item.gold), "domain": self.item.domain}


@dataclass
class PaperReport:
    name: str
    responses: List[Response] = field(default_factory=list)

    @property
    def asked(self) -> int:
        return len(self.responses)

    @property
    def right(self) -> int:
        return sum(1 for r in self.responses if r.right)

    @property
    def wrong(self) -> int:
        return sum(1 for r in self.responses if r.wrong)

    @property
    def silent(self) -> int:
        return sum(1 for r in self.responses if r.silent)

    @property
    def score(self) -> float:
        return round(self.right / self.asked, 4) if self.asked else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"paper": self.name, "asked": self.asked, "right": self.right,
                "wrong": self.wrong, "silent": self.silent, "score": self.score,
                "inverted": self.name in INVERTED}


@dataclass
class DomainReport:
    """One of the forty-two, asked *what*, *how* and *why* about its own subjects.

    Three counts rather than one, because they are three capabilities and the corpus supplies them
    unevenly. A domain that scores well on *what* and nothing on *why* is a domain whose ``.kb``
    file is a glossary, and averaging that into one number is how the gap stays invisible.
    """

    name: str
    what_asked: int = 0
    what_right: int = 0
    why_asked: int = 0
    why_right: int = 0
    how_asked: int = 0
    how_right: int = 0
    subjects: int = 0
    #: The held-out papers restricted to this domain's own subjects. The sweep above is coverage
    #: — *does a question of this shape reach a supported answer here* — and it saturates, which
    #: is what a coverage measure does when coverage is good. These two are the hard numbers: a
    #: chain nobody stated and a mechanism nobody assembled, counted per domain, so "she can
    #: explain" is a claim about physics or about law rather than about an average over both.
    held_asked: int = 0
    held_right: int = 0

    def _rate(self, right: int, asked: int) -> Optional[float]:
        """``None`` when the paper was never asked here, which is not the same as failing it.

        The first run of this report printed ``0.00`` for both, and every ``how`` column in it
        was one of the two without saying which. ``geography`` read 0.00 on *why* because no
        geography subject has a causal or purposive edge; ``people`` read 0.00 on the same column
        because four of them did and she answered none. Those are opposite findings and the table
        showed them identically.
        """
        return round(right / asked, 4) if asked else None

    @property
    def what(self) -> Optional[float]:
        return self._rate(self.what_right, self.what_asked)

    @property
    def why(self) -> Optional[float]:
        return self._rate(self.why_right, self.why_asked)

    @property
    def how(self) -> Optional[float]:
        return self._rate(self.how_right, self.how_asked)

    @property
    def held(self) -> Optional[float]:
        """The held-out papers on this domain's subjects, or ``None`` where none landed here."""
        return self._rate(self.held_right, self.held_asked)

    @property
    def score(self) -> Optional[float]:
        asked = self.what_asked + self.why_asked + self.how_asked
        right = self.what_right + self.why_right + self.how_right
        return self._rate(right, asked)

    def to_dict(self) -> Dict[str, Any]:
        return {"domain": self.name, "subjects": self.subjects,
                "what": self.what, "why": self.why, "how": self.how, "score": self.score,
                "held": self.held, "held_asked": self.held_asked,
                "asked": self.what_asked + self.why_asked + self.how_asked}


@dataclass
class ExamReport:
    papers: List[PaperReport] = field(default_factory=list)
    domains: List[DomainReport] = field(default_factory=list)
    facts: int = 0
    seed: int = DEFAULT_SEED

    def paper(self, name: str) -> Optional[PaperReport]:
        for got in self.papers:
            if got.name == name:
                return got
        return None

    @property
    def asked(self) -> int:
        return sum(p.asked for p in self.papers if p.name not in INVERTED)

    @property
    def right(self) -> int:
        return sum(p.right for p in self.papers if p.name not in INVERTED)

    @property
    def score(self) -> float:
        return round(self.right / self.asked, 4) if self.asked else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"facts": self.facts, "seed": self.seed, "asked": self.asked,
                "right": self.right, "score": self.score,
                "papers": [p.to_dict() for p in self.papers],
                "domains": [d.to_dict() for d in self.domains]}


# --------------------------------------------------------------------------- #
# Reading the domain map off the sources
# --------------------------------------------------------------------------- #
_KB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "scripts", "knowledge")

_LINE = re.compile(r"^(?P<subject>[^|#]+?)\s*\|\s*(?P<rest>.+)$")


def subjects_by_domain(kb_dir: str = "") -> Dict[str, List[str]]:
    """``{domain: [subject]}`` read off the ``.kb`` sources.

    The shipped triple file carries no domain — by design, since the graph does not have domains,
    only subjects — so the per-domain block needs the sources. Returns ``{}`` when they are not
    on disk (an installed package rather than a checkout), and the exam then reports papers only
    rather than inventing a grouping.
    """
    root = kb_dir or _KB_DIR
    if not os.path.isdir(root):
        return {}
    per_file: Dict[str, List[str]] = defaultdict(list)
    for name in sorted(os.listdir(root)):
        if not name.endswith(".kb"):
            continue
        with open(os.path.join(root, name), encoding="utf-8") as handle:
            for line in handle:
                body = line.split("#")[0].strip()
                match = _LINE.match(body)
                if match:
                    per_file[name[:-3]].append(" ".join(match.group("subject").split()).lower())
    out: Dict[str, List[str]] = {}
    for domain, files in DOMAINS.items():
        seen: List[str] = []
        for stem in files:
            for subject in per_file.get(stem, ()):
                if subject not in seen:
                    seen.append(subject)
        out[domain] = seen
    return out


# --------------------------------------------------------------------------- #
# The exam
# --------------------------------------------------------------------------- #
class ExplanationExam:
    """Seven papers and a per-domain sweep, over a brain with the world corpus in it."""

    def __init__(self, brain: Any, *, limit: int = DEFAULT_LIMIT,
                 seed: int = DEFAULT_SEED, kb_dir: str = "") -> None:
        self.brain = brain
        self.limit = max(1, int(limit))
        self.seed = int(seed)
        self.grounder = getattr(brain, "grounder", brain)
        self.explainer = Explainer(self.grounder)
        self.by_sp: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self._index()
        self.domain_of: Dict[str, str] = {}
        self.by_domain = subjects_by_domain(kb_dir)
        for domain, subjects in self.by_domain.items():
            for subject in subjects:
                self.domain_of.setdefault(subject, domain)

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
                if obj:
                    self.by_sp[(subject, predicate)].append(obj)

    @property
    def facts(self) -> int:
        return sum(len(v) for v in self.by_sp.values())

    def key(self, text: str) -> str:
        try:
            return self.grounder._key(text)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            return " ".join(str(text or "").split()).lower()

    def holds(self, subject: str, predicate: str, obj: str) -> bool:
        wanted = " ".join(str(obj).split()).lower()
        return any(" ".join(o.split()).lower() == wanted
                   for o in self.by_sp.get((self.key(subject), predicate), ()))

    # ---- grading -------------------------------------------------------- #
    @staticmethod
    def _mentions(said: str, wanted: str) -> bool:
        """Does the answer name this thing?

        Containment, in the store's own spelling, for the reason ``general._matches`` records:
        the two ends are written by different hands and exact equality would grade a paraphrase
        as an error. Here it also has to survive a **gloss** — the answer is a sentence built out
        of several objects, so what is being asked is whether the object appears in it, not
        whether the whole string matches.
        """
        low = " ".join(str(said or "").split()).lower()
        want = " ".join(str(wanted or "").split()).lower()
        if not low or not want:
            return False
        if want in low:
            return True
        try:
            from nyxara.njp.canon import canonical_entity
        except Exception:  # noqa: BLE001
            return False
        return canonical_entity(want) in canonical_entity(low)

    def ask(self, item: Item) -> Response:
        out = Response(item=item)
        if item.paper == "minted":
            return self._ask_minted(item)
        if item.paper == "procedure":
            return self._ask_procedure(item)
        try:
            got = self.explainer.ask(item.question)
        except Exception:  # noqa: BLE001
            got = None
        said = ""
        if got is not None:
            said = got.text() if hasattr(got, "text") else ""
        out.said = said
        if not said.strip():
            out.silent = True
            out.right = item.paper in INVERTED
            return out
        if item.paper in INVERTED:
            out.wrong = True
            return out
        # A forbidden mention is checked **before** a gold one. `direction` items have a gold and
        # a forbidden set and an answer can contain both — a walk that returned the cause *and*
        # the purpose would otherwise score as right for having also said the right thing.
        if any(self._mentions(said, bad) for bad in item.forbidden):
            out.wrong = True
            return out
        # `mechanism` needs **every** gold; the others need one. A paper whose gold is a
        # disjunction of the ways an answer could be right and one whose gold is a conjunction of
        # what a single answer must contain are different graders, and collapsing them was how
        # `mechanism` scored 60/60 on a walk that only had to name one of the two.
        if item.paper == "mechanism":
            # Any one pair, both halves of it. See `paper_mechanism`.
            out.right = any(self._mentions(said, part) and self._mentions(said, effect)
                            for part, effect in (item.payload or ()))
            out.wrong = not out.right
            return out
        hits = [self._mentions(said, good) for good in item.gold]
        if hits and any(hits):
            out.right = True
        else:
            out.wrong = True
        return out

    def _ask_procedure(self, item: Item) -> Response:
        out = Response(item=item)
        try:
            plan = self.explainer.ask(item.question)
        except Exception:  # noqa: BLE001
            plan = None
        if plan is None or not getattr(plan, "orders", None):
            out.silent = True
            out.said = ""
            return out
        wanted_count, needs = item.payload
        out.said = " → ".join(plan.first)
        valid = all(self._valid(order, needs) for order in plan.orders)
        counted = plan.order_count == wanted_count
        complete = set(plan.first) == set(needs)
        out.right = bool(valid and counted and complete)
        out.wrong = not out.right
        return out

    def _ask_minted(self, item: Item) -> Response:
        out = Response(item=item)
        steps, needs, truth = item.payload
        stub = _Store(item.note)
        # Through `ask`, not through `procedure`. The grammar has to reach the walk here for the
        # same reason it has to on every other paper: a derivation only Python can call is the
        # exact gap `general.py` records `inheritance` sitting in at 400/400.
        plan = Explainer(stub).ask(item.question)
        if plan is None:
            out.said = "the question did not parse as a procedure"
            out.wrong = True
            return out
        out.said = f"{plan.order_count} orders"
        if not plan.orders:
            out.silent = True
            return out
        got = {tuple(order) for order in plan.orders}
        # The whole set when it is small enough to have been enumerated, otherwise a subset check
        # plus the count. Both halves matter: the count is the claim about how determined the
        # telling was, and membership is the claim that each order is admissible.
        right = plan.order_count == len(truth) and got <= truth and set(plan.first) == set(steps)
        out.right = bool(right)
        out.wrong = not right
        return out

    @staticmethod
    def _valid(order: Sequence[str], needs: Dict[str, List[str]]) -> bool:
        seen: Set[str] = set()
        for step in order:
            for need in needs.get(step, ()):
                if need not in seen:
                    return False
            seen.add(step)
        return True

    # ---- papers --------------------------------------------------------- #
    def paper_why_chain(self) -> List[Item]:
        """Two causal hops, where the one-hop shortcut is not in the store."""
        out: List[Item] = []
        for (subject, predicate), objects in list(self.by_sp.items()):
            if predicate != "causes":
                continue
            for middle in objects:
                mid = self.key(middle)
                for far in self.by_sp.get((mid, "causes"), ()):
                    if self.key(far) == self.key(subject):
                        continue
                    if self.holds(subject, "causes", far):
                        continue        # stated outright: this would be recall in a hat
                    out.append(Item(paper="why_chain",
                                    question=f"why does {far} happen?",
                                    topic=far, gold=(subject,),
                                    domain=self.domain_of.get(self.key(far), ""),
                                    note=f"{subject} -> {middle} -> {far}"))
        return out

    def paper_why_purpose(self) -> List[Item]:
        out: List[Item] = []
        for (subject, predicate), objects in list(self.by_sp.items()):
            if predicate != "purpose":
                continue
            for middle in objects:
                for far in self.by_sp.get((self.key(middle), "purpose"), ()):
                    if self.key(far) == self.key(subject):
                        continue
                    if self.holds(subject, "purpose", far):
                        continue
                    out.append(Item(paper="why_purpose",
                                    question=f"what is {subject} for?",
                                    topic=subject, gold=(far,),
                                    domain=self.domain_of.get(self.key(subject), ""),
                                    note=f"{subject} -> {middle} -> {far}"))
        return out

    def paper_mechanism(self) -> List[Item]:
        """A part, and what that part does. The pairing is never stated about the whole.

        **One item per subject, and its gold is every pair that would be right.** The first
        version emitted one item per *part* and graded each against that one part, which measured
        something else entirely: the digestive system has eight parts, the walk returns six chains,
        and an item asking for the small intestine scored her wrong for having answered with the
        large intestine, the liver and the stomach. That is the "a gold answer is every answer that
        would be right, not the one the item was minted from" rule from
        :mod:`nyxara.njp.general`, broken in this file three paragraphs after its docstring quotes
        it. Measured, it cost the paper 0.14.

        The grade is still a conjunction *within* a pair — an answer must name a part **and** what
        that part does — and a disjunction *across* them.
        """
        out: List[Item] = []
        pairs: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for (subject, predicate), objects in list(self.by_sp.items()):
            if predicate not in ("has_part", "consists_of"):
                continue
            for part in objects:
                pkey = self.key(part)
                does: List[str] = []
                for relation in ("purpose", "causes"):
                    does.extend(self.by_sp.get((pkey, relation), ()))
                for effect in does:
                    if self.holds(subject, "purpose", effect) or self.holds(subject, "causes",
                                                                            effect):
                        continue
                    pairs[subject].append((part, effect))
                    break
        for subject, got in pairs.items():
            out.append(Item(paper="mechanism",
                            question=f"how does {subject} work?",
                            topic=subject, payload=got,
                            domain=self.domain_of.get(self.key(subject), ""),
                            note=f"{len(got)} part/effect pairs, none of them stated about "
                                 f"{subject}"))
        return out

    def paper_direction(self) -> List[Item]:
        """A purpose asked of a thing that also has a cause, and a cause asked of one that also
        has a purpose. **Both ways round**, and that is the fix rather than the design.

        The first version asked only *what is X for*. It scored 60/60 and could not have scored
        anything else: the question grammar reads *for* as ``sense="for"``,
        :meth:`~nyxara.njp.explain.Explainer.why` then walks only ``purpose``, and the forbidden
        causes are unreachable by construction. That is an acceptance test for one line of
        dispatch presented as a paper about a confusion.

        Asking both directions makes each the other's control. A walk that ignored ``sense``
        entirely would now fail every item instead of passing every item, and a walk that had the
        two senses **swapped** — the failure most worth catching, since it is invisible when only
        one direction is asked — fails all of them too.

        Only subjects that have exactly one of the two in each role are used, so a forbidden
        object is genuinely wrong rather than merely a second right answer.
        """
        out: List[Item] = []
        for (subject, predicate), objects in list(self.by_sp.items()):
            if predicate != "purpose" or not objects:
                continue
            causes = list(self.by_sp.get((subject, "causes"), ()))
            if not causes:
                continue
            out.append(Item(paper="direction",
                            question=f"what is {subject} for?",
                            topic=subject, gold=tuple(objects), forbidden=tuple(causes),
                            domain=self.domain_of.get(subject, ""),
                            note="a purpose question must not be answered with an effect"))
            # The other direction. `why does X happen` walks backwards, so the gold is what leads
            # *into* X — and where nothing does, the item is dropped rather than made unanswerable.
            incoming = [c for c, _ in self.explainer._in(subject, "causes")]
            incoming += list(self.by_sp.get((subject, "occurs_when"), ()))
            if incoming:
                out.append(Item(paper="direction",
                                question=f"why does {subject} happen?",
                                topic=subject, gold=tuple(incoming), forbidden=tuple(objects),
                                domain=self.domain_of.get(subject, ""),
                                note="a cause question must not be answered with a purpose"))
        return out

    def paper_procedure(self) -> List[Item]:
        out: List[Item] = []
        for (subject, predicate), objects in list(self.by_sp.items()):
            if predicate != "has_step" or len(objects) < 3:
                continue
            steps = sorted(set(objects), key=self.key)
            needs: Dict[str, List[str]] = {}
            index = {self.key(s): s for s in steps}
            for step in steps:
                got = [index[self.key(n)] for n in self.by_sp.get((self.key(step), "requires"), ())
                       if self.key(n) in index]
                needs[step] = sorted(set(got), key=self.key)
            count = sum(1 for _ in _orders(steps, needs))
            out.append(Item(paper="procedure",
                            question=f"what are the steps of {subject}?",
                            topic=subject, domain=self.domain_of.get(subject, ""),
                            payload=(count, needs),
                            note=f"{len(steps)} steps, {count} admissible orders"))
        return out

    def paper_minted(self, count: int) -> List[Item]:
        """Procedures invented per seed. The one paper no corpus can have leaked into."""
        rng = random.Random(self.seed ^ 0x5EED)
        out: List[Item] = []
        for n in range(count):
            size = 3 + (n % 4)
            name = _mint(rng) + " " + _mint(rng)
            steps = [_mint(rng) for _ in range(size)]
            while len(set(steps)) < size:
                steps = [_mint(rng) for _ in range(size)]
            # A random DAG over a random permutation, so the constraints are always satisfiable
            # and the number of admissible orders varies from one to all of them.
            order = list(steps)
            rng.shuffle(order)
            needs: Dict[str, List[str]] = {s: [] for s in steps}
            for i, step in enumerate(order):
                for earlier in order[:i]:
                    if rng.random() < 0.35:
                        needs[step].append(earlier)
            truth = {tuple(o) for o in _orders(sorted(steps), needs)}
            lines: List[Tuple[str, str, str]] = [(name, "has_step", s) for s in steps]
            for step, wants in needs.items():
                lines.extend((step, "requires", w) for w in wants)
            out.append(Item(paper="minted",
                            question=f"what are the steps of {name}?",
                            topic=name, note=lines, payload=(steps, needs, truth),
                            domain="procedures"))
        return out

    def paper_abstention(self, count: int) -> List[Item]:
        """Silence is the pass, and half the items are about things she **does** know.

        Nonsense topics alone are too easy and measure the wrong thing: a walk finds nothing for
        ``vrenth`` because ``vrenth`` is not a key, which no amount of confabulation could change.
        The confabulation risk is on a subject that *is* in the store and has none of the relation
        being asked for — *how does the Pacific Ocean work*, *what are the steps of a mountain* —
        where every neighbouring edge is a candidate for a walk that wants an answer. So half the
        items name a real subject and ask it for a relation it does not carry, and the item is
        only used when the store has none of the edges that walk reads.
        """
        rng = random.Random(self.seed ^ 0xAB57)
        out: List[Item] = []
        forms = ("why does {} happen?", "how does {} work?", "what are the steps of {}?",
                 "what is {} for?")
        while len(out) < count // 2:
            topic = _mint(rng) + (" " + _mint(rng) if rng.random() < 0.4 else "")
            if (self.key(topic), "is_a") in self.by_sp:
                continue
            out.append(Item(paper="abstention",
                            question=forms[len(out) % len(forms)].format(topic),
                            topic=topic, domain="", note="nothing in the store is about this"))
        real = sorted({s for (s, p) in self.by_sp if p == "is_a"})
        rng.shuffle(real)
        for subject in real:
            if len(out) >= count:
                break
            silent_for = []
            if not self.by_sp.get((subject, "has_step")):
                silent_for.append(("what are the steps of {}?", ()))
            if not self.by_sp.get((subject, "purpose")) and not self.by_sp.get(
                    (self.key(subject), "is_a")) is None:
                kinds = self.by_sp.get((subject, "is_a"), ())
                if not any(self.by_sp.get((self.key(k), "purpose")) for k in kinds):
                    silent_for.append(("what is {} for?", ()))
            if not silent_for:
                continue
            form, _ = silent_for[rng.randrange(len(silent_for))]
            out.append(Item(paper="abstention", question=form.format(subject),
                            topic=subject, domain=self.domain_of.get(subject, ""),
                            note="a real subject, asked for a relation it does not carry"))
        return out

    # ---- the per-domain sweep -------------------------------------------- #
    def sweep(self) -> List[DomainReport]:
        """What, how and why over every one of the forty-two, on that domain's own subjects.

        *What* goes through :meth:`~nyxara.njp.grounding.Grounder.answer` because that is the
        channel a what-question has; *why* and *how* go through the walk. Grading is *did she say
        anything the store supports* — an item counts right when the answer names a stated object
        of the relation being asked for. It is a coverage measure and it is labelled one: it says
        the question reaches an answer in this domain, not that the answer is the best one.
        """
        rng = random.Random(self.seed ^ 0xD0A1)
        out: List[DomainReport] = []
        per = max(4, self.limit // 12)
        for domain in DOMAINS:
            subjects = [s for s in self.by_domain.get(domain, ()) if (s, "is_a") in self.by_sp]
            report = DomainReport(name=domain, subjects=len(subjects))
            if subjects:
                sample = subjects if len(subjects) <= per else rng.sample(subjects, per)
                for subject in sample:
                    report.what_asked += 1
                    if self._answers_what(subject):
                        report.what_right += 1
                    # What a *why* about this subject could be built from, and forward `causes`
                    # is deliberately not in it. `max planck causes quantum theory` is material
                    # for *why did quantum theory happen*; it says nothing about why Planck
                    # happened, and counting it here asked the walk for something no walk should
                    # produce and then scored her wrong for not producing it. Measured: the
                    # `people` domain reported why at 0.00 with four subjects, every one of them
                    # a scientist whose only causal edge points away from them.
                    gold = list(self.by_sp.get((subject, "purpose"), ())) + \
                        list(self.by_sp.get((subject, "occurs_when"), ())) + \
                        list(self.by_sp.get((subject, "requires"), ())) + \
                        [c for c, _ in self.explainer._in(subject, "causes")]
                    if gold:
                        report.why_asked += 1
                        got = self.explainer.why(subject)
                        # **What she said**, not merely that she said something. The first version
                        # asked `got.answered` and reported 1.00 on all forty-two domains, which
                        # is what a coverage measure does when it grades the presence of an answer
                        # rather than its content: a walk that returned the wrong chain, or the
                        # right chain about the wrong sense of the word, scored identically to one
                        # that was right. Naming a stated object of the relation asked for is a
                        # low bar and it is a bar.
                        if got.answered and any(self._mentions(got.text(), g) for g in gold):
                            report.why_right += 1
                    # Forward `causes` lands here instead: what a thing brings about is part of
                    # how it works, and `Explainer.mechanism` walks exactly that.
                    parts = list(self.by_sp.get((subject, "has_part"), ())) + \
                        list(self.by_sp.get((subject, "consists_of"), ())) + \
                        list(self.by_sp.get((subject, "has_step"), ())) + \
                        list(self.by_sp.get((subject, "causes"), ()))
                    if parts:
                        report.how_asked += 1
                        said = self.explainer.mechanism(subject).text()
                        plan = self.explainer.procedure(subject)
                        if plan.orders:
                            said += " " + " ".join(plan.first)
                        if any(self._mentions(said, p) for p in parts):
                            report.how_right += 1
            out.append(report)
        return out

    def _answers_what(self, subject: str) -> bool:
        gold = self.by_sp.get((subject, "is_a"), ())
        if not gold:
            return False
        try:
            got = self.grounder.answer(f"what is {subject}?")
        except Exception:  # noqa: BLE001
            return False
        said = str(getattr(got, "text", "") or "")
        return any(self._mentions(said, g) for g in gold) or bool(
            getattr(got, "conflicting", False))

    # ---- sitting it ------------------------------------------------------ #
    def items(self, paper: str) -> List[Item]:
        rng = random.Random(self.seed ^ (hash(paper) & 0xFFFF))
        if paper == "minted":
            return self.paper_minted(min(self.limit, 120))
        if paper == "abstention":
            return self.paper_abstention(self.limit)
        pool = getattr(self, f"paper_{paper}")()
        if len(pool) > self.limit:
            pool = rng.sample(pool, self.limit)
        return pool

    def sit(self, papers: Optional[Sequence[str]] = None, *,
            sweep: bool = True) -> ExamReport:
        report = ExamReport(facts=self.facts, seed=self.seed)
        for name in (papers or PAPERS):
            got = PaperReport(name=name)
            for item in self.items(name):
                got.responses.append(self.ask(item))
            report.papers.append(got)
        if sweep and self.by_domain:
            report.domains = self.sweep()
            by_name = {d.name: d for d in report.domains}
            for paper in report.papers:
                if paper.name in INVERTED:
                    continue
                for response in paper.responses:
                    got = by_name.get(response.item.domain)
                    if got is None:
                        continue
                    got.held_asked += 1
                    got.held_right += 1 if response.right else 0
        return report


# --------------------------------------------------------------------------- #
# Minting, and the ground truth the minted paper is graded against
# --------------------------------------------------------------------------- #
_ONSET = "bdfgklmnprstvz"
_VOWEL = "aeiou"


def _mint(rng: random.Random) -> str:
    """A word no corpus contains. Two or three syllables of legal-looking nonsense."""
    return "".join(rng.choice(_ONSET) + rng.choice(_VOWEL)
                   for _ in range(rng.randint(2, 3))) + rng.choice("thnkr")


def _orders(steps: Sequence[str], needs: Dict[str, List[str]]) -> Iterable[Tuple[str, ...]]:
    """Every admissible order, **by brute force over permutations**.

    This is the ground truth for the minted paper and it is deliberately not the algorithm under
    test. :meth:`~nyxara.njp.explain.Explainer._orders` builds orders by extending a prefix with
    whatever is currently free; this generates all ``n!`` permutations and filters them. Two
    implementations of the same specification, one of them obviously correct and far too slow to
    ship, is the only way a topological sort can be examined rather than asserted — grading it
    against itself would pass whatever it did.
    """
    steps = list(steps)
    if len(steps) > 8:                  # 8! is 40,320; beyond that the truth is not affordable
        return
    for candidate in itertools.permutations(steps):
        seen: Set[str] = set()
        ok = True
        for step in candidate:
            if any(need not in seen for need in needs.get(step, ())):
                ok = False
                break
            seen.add(step)
        if ok:
            yield candidate


class _Store:
    """A fact store of exactly the triples a minted procedure was told, and nothing else.

    Two lines of it, and it is the acceptance test for the claim this module makes: an
    :class:`~nyxara.njp.explain.Explainer` built around this answers the minted questions as well
    as it answers the corpus ones, so the ordering cannot be coming from anything the corpus put
    there.
    """

    class _Triple:
        __slots__ = ("object", "confidence", "superseded")

        def __init__(self, obj: str) -> None:
            self.object = obj
            self.confidence = 1.0
            self.superseded = False

    def __init__(self, triples: Sequence[Tuple[str, str, str]]) -> None:
        self.facts: Dict[Tuple[str, str], List[Any]] = defaultdict(list)
        for subject, predicate, obj in triples:
            self.facts[(subject.lower(), predicate)].append(self._Triple(obj))

    def _key(self, text: str) -> str:
        return " ".join(str(text or "").split()).lower()


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def examine(*, limit: int = DEFAULT_LIMIT, seed: int = DEFAULT_SEED,
            papers: Optional[Sequence[str]] = None, brain: Any = None,
            sweep: bool = True) -> ExamReport:
    from nyxara.njp.general import load_brain

    got = brain if brain is not None else load_brain()
    return ExplanationExam(got, limit=limit, seed=seed).sit(papers, sweep=sweep)


def render(report: ExamReport) -> str:
    lines = [f"explanation exam — {report.facts} facts, seed {report.seed}", "",
             f"{'paper':14} {'asked':>6} {'right':>6} {'wrong':>6} {'silent':>7} {'score':>7}"]
    for paper in report.papers:
        tail = "   (silence is the pass)" if paper.name in INVERTED else ""
        lines.append(f"{paper.name:14} {paper.asked:6} {paper.right:6} {paper.wrong:6} "
                     f"{paper.silent:7} {paper.score:7.3f}{tail}")
    lines.append("")
    lines.append(f"{report.right}/{report.asked} = {report.score:.3f} over the papers that "
                 f"reward answering")
    if report.domains:
        lines += ["", f"{'domain':30} {'subj':>5} {'what':>6} {'why':>6} {'how':>6} {'all':>6}"
                       f" {'held':>6} {'n':>5}"]
        def cell(value: Optional[float]) -> str:
            return "     -" if value is None else f"{value:6.2f}"

        for domain in report.domains:
            lines.append(f"{domain.name:30} {domain.subjects:5} {cell(domain.what)} "
                         f"{cell(domain.why)} {cell(domain.how)} {cell(domain.score)} "
                         f"{cell(domain.held)} {domain.held_asked:5}")
        weakest = sorted((d for d in report.domains if d.held is not None and d.held_asked >= 3),
                         key=lambda d: d.held or 0.0)[:5]
        lines.append("")
        lines.append("weakest on the held-out papers: "
                     + ", ".join(f"{d.name} {d.held or 0.0:.2f} (n={d.held_asked})"
                                 for d in weakest))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="examine what, how and why")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--paper", action="append", default=None, choices=list(PAPERS))
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    report = examine(limit=args.limit, seed=args.seed, papers=args.paper,
                     sweep=not args.no_sweep)
    print(json.dumps(report.to_dict(), indent=1) if args.json else render(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
