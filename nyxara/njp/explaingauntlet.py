"""NYXARA · njp/explaingauntlet.py — the exam written to break the last one (⚔, NJP V.37).

:mod:`nyxara.njp.explainschool` measures 0.990 and **that number is now frozen**, because a
benchmark at 0.99 has stopped being an instrument. It says the walk does what the walk was built
to do on a corpus the walk was tuned against, and nothing more. Every paper in it is generated
from the shipped fact store, which means every item is about entities she was told about, in a
graph shape somebody wrote by hand, phrased the way the question grammar's own patterns are
phrased.

So this file is written the way :mod:`nyxara.njp.hard` was written and for the same reason: by
someone looking for what the faculty **cannot** do, against a faculty that is finished. Nothing
here is drawn from the corpus. Every item is minted — the entities, the causal graph, the wording,
and the traps — and the traps are the point.

Nine attacks, and each names a property of real inference the walk has no defence against
==========================================================================================

===============  ====================================================================================
Attack           What it injects, and what a passing answer has to do
===============  ====================================================================================
wording          The same question in phrasings no pattern in ``explainread`` was written for.
                 *"What brings about X?"*, *"On account of what does X occur?"*, *"Through what
                 process does X arise?"* — a grammar of surface forms is a list, and a list has an
                 edge.
entities         Every node is a minted nonsense word. Nothing can be recognised, only derived.
                 This is the control for all eight others: an attack that fails here fails for a
                 reason that has nothing to do with the attack.
shape            The causal graph is generated per item — chain, fork, collider, diamond, or a
                 random DAG — rather than being whatever shape the corpus author happened to
                 write. A **collider** is the case worth naming: ``A → C ← B`` means A and B are
                 both causes of C and *unrelated to each other*, and a walk that reports one
                 chain through the other has invented an edge.
distractors      True facts that have nothing to do with the question, wired into the same
                 neighbourhood. Naming one is **wrong**, not merely verbose: an explanation that
                 includes an irrelevant cause is a false claim about what brought the thing about.
contradiction    Two stated edges that cannot both hold. The answer is that they conflict — not
                 either one of them, and not silence. The one paper here where *reporting the
                 dispute* is the only pass.
homonym          One spelling over two disjoint neighbourhoods, which is the ``atrium`` defect
                 V.36 found and declined to fix. A chain that enters on one sense and leaves on
                 the other is scored **wrong**. V.36 demoted such a chain in the ranking; this
                 asks whether it is ever said at all.
gap              The middle of the mechanism is deleted after the graph is built. The two ends are
                 stated, the link between them is not, and the only correct answer is **UNKNOWN**.
                 A walk that bridges a hole it cannot see is the most dangerous failure in this
                 file, because it is indistinguishable from success on every other paper.
legs             The answer requires **two independently derived legs** that meet only at the
                 target: neither leg alone reaches it, and no single path exists. This is where
                 walking a graph stops being enough.
unknown          A world where the evidence genuinely does not support an answer — the target has
                 no incoming edge of any kind, or its only incoming edge is below what any floor
                 would keep. **Silence is the only pass**, and every answer is a confabulation.
===============  ====================================================================================

Three rules this file is built on
--------------------------------

**UNKNOWN is a verdict, not a failure to have one.** :class:`Verdict` has four values and the
grader keeps them apart: ``RIGHT``, ``WRONG``, ``UNKNOWN`` and ``CONFLICT``. On ``gap`` and
``unknown`` the pass *is* ``UNKNOWN``; on ``contradiction`` the pass is ``CONFLICT``; everywhere
else both are failures. A single score over papers that reward opposite behaviours is the mistake
:mod:`nyxara.njp.general` records about its own inverted papers, and it is not repeated here — the
report prints per-paper and refuses to average across the divide.

**A wrong answer is worse than no answer, and the report says so separately.** :attr:`Paper.honest`
is right-plus-correct-silence over asked, and :attr:`Paper.confabulated` is answers given where
the evidence did not support one. A system that scores 0.6 by answering everything is worse than
one that scores 0.6 by answering what it can, and one number cannot tell them apart.

**Nothing is graded by similarity.** A chain is right when its nodes are exactly the derivation the
generator built, in order. Partial credit for a chain is credit for a wrong explanation.

What it measures
----------------

The floor, taken cold before anything was built for it, is printed by ``python -m
nyxara.njp.explaingauntlet`` and recorded in ``docs/CAPABILITIES.md``. It is not expected to be
good. A gauntlet that the faculty passes on the day it is written was written to be passed.

Pure standard library, deterministic per seed.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "Verdict", "Fact", "World", "Item", "Reply", "Answer", "Paper", "Report",
    "Gauntlet", "ATTACKS", "run", "render", "main", "DEFAULT_SEED", "DEFAULT_LIMIT",
]

DEFAULT_SEED = 20260904
DEFAULT_LIMIT = 60

ATTACKS: Tuple[str, ...] = ("wording", "wording_new", "entities", "shape", "distractors",
                            "contradiction", "homonym", "gap", "legs", "unknown")

#: Papers where an answer is the failure and silence is the pass.
SILENT_PASS: Tuple[str, ...] = ("gap", "unknown")

#: The paper where naming the dispute is the only pass.
CONFLICT_PASS: Tuple[str, ...] = ("contradiction",)


class Verdict(str, Enum):
    """What she did, in four values rather than two.

    The two-valued version — right or wrong — is what makes a confabulation and an abstention
    indistinguishable, and this whole file exists because those are the two outcomes worth telling
    apart on a hard problem.
    """

    RIGHT = "right"
    WRONG = "wrong"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


# --------------------------------------------------------------------------- #
# A minted world
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Fact:
    subject: str
    predicate: str
    object: str
    confidence: float = 0.9

    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)


class _Triple:
    __slots__ = ("object", "confidence", "superseded", "negated")

    def __init__(self, obj: str, confidence: float) -> None:
        self.object = obj
        self.confidence = confidence
        self.superseded = False
        self.negated = False


@dataclass
class World:
    """A fact store containing exactly what one item was told, and nothing else.

    Built per item rather than once, which is the difference between this file and
    :mod:`nyxara.njp.explainschool`. A shared store means an item's trap can be defused by an
    unrelated edge somebody else's item needed, and it means "she got it right" can mean "the
    corpus happened to contain a shortcut". Here the world *is* the item.
    """

    facts_list: List[Fact] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.facts: Dict[Tuple[str, str], List[_Triple]] = defaultdict(list)
        for fact in self.facts_list:
            self.facts[(fact.subject.lower(), fact.predicate)].append(
                _Triple(fact.object, fact.confidence))

    def _key(self, text: str) -> str:
        return " ".join(str(text or "").split()).lower()

    def add(self, *facts: Fact) -> "World":
        for fact in facts:
            self.facts_list.append(fact)
            self.facts[(fact.subject.lower(), fact.predicate)].append(
                _Triple(fact.object, fact.confidence))
        return self

    def holds(self, subject: str, predicate: str, obj: str) -> bool:
        return any(t.object.lower() == obj.lower()
                   for t in self.facts.get((subject.lower(), predicate), ()))


# --------------------------------------------------------------------------- #
# Minting
# --------------------------------------------------------------------------- #
_ONSET = "bdfgklmnprstvzh"
_VOWEL = "aeiou"
_CODA = "thnkrlmsp"


def _word(rng: random.Random) -> str:
    """A word no corpus contains and no reader recognises."""
    return "".join(rng.choice(_ONSET) + rng.choice(_VOWEL)
                   for _ in range(rng.randint(2, 3))) + rng.choice(_CODA)


def _deepest(facts: Sequence[Fact], target: str) -> Tuple[str, ...]:
    """The longest backward causal path from *target*, as the gold for a generated graph.

    Longest rather than any, and computed rather than recorded, because a generator that writes
    down the path it happened to build last will grade a deeper correct derivation as wrong. The
    graph is a DAG by construction — edges only ever run from an earlier position in a shuffled
    order to a later one — so the recursion needs no cycle guard and terminates.
    """
    incoming: Dict[str, List[str]] = defaultdict(list)
    for fact in facts:
        if fact.predicate == "causes":
            incoming[fact.object.lower()].append(fact.subject)

    def walk(node: str) -> List[str]:
        best: List[str] = []
        for cause in incoming.get(node.lower(), ()):
            got = walk(cause)
            if len(got) > len(best):
                best = got
        return [node] + best

    return tuple(walk(target))


def _words(rng: random.Random, count: int) -> List[str]:
    out: List[str] = []
    while len(out) < count:
        got = _word(rng)
        if got not in out:
            out.append(got)
    return out


#: Every phrasing of *why does X happen* that a person might write, and none of them is the
#: phrasing ``explainread._BECAUSE`` was built around. A question grammar is a list of surfaces,
#: and this is the list of surfaces it does not have.
WHY_FORMS: Tuple[str, ...] = (
    "why does {x} happen?",                        # the one it was built for — the control
    "what brings about {x}?",
    "on account of what does {x} occur?",
    "through what process does {x} arise?",
    "what is responsible for {x}?",
    "how come {x} happens?",
    "what gives rise to {x}?",
    "{x} happens because of what?",
    "what accounts for {x}?",
    "what lies behind {x}?",
)

HOW_FORMS: Tuple[str, ...] = (
    "how does {x} work?",
    "by what means does {x} operate?",
    "what is the workings of {x}?",
    "how is {x} brought about?",
    "what makes {x} function?",
)


# --------------------------------------------------------------------------- #
# Items and grading
# --------------------------------------------------------------------------- #
@dataclass
class Reply:
    """What the thing under test said, structurally.

    A gauntlet that could only read a string could only ask whether a name appeared in it, and
    two of these papers were passed by exactly that: ``contradiction`` accepted *"because A"*
    beside *"because B"* as having reported a dispute, and ``legs`` accepted the same two lines as
    having derived that both are required. Listing two alternatives is not noticing they exclude
    each other, and it is not deriving that they are jointly necessary. Those are three different
    claims about the same two names and only a structured answer can tell them apart.

    So the contract is explicit and the walk fills in what it actually has: ``chains`` is what it
    found, ``conflict`` is *"these cannot both hold"* and ``joint`` is *"these are both required"*.
    :func:`_walk` returns False for both, and that is the finding, not a shortcoming of the
    harness.
    """

    text: str = ""
    chains: List[List[str]] = field(default_factory=list)
    conflict: bool = False
    joint: bool = False

    @property
    def nodes(self) -> List[str]:
        """The chain she leads with. An explanation is what is said first, not what is buried."""
        return list(self.chains[0]) if self.chains else []

    @property
    def silent(self) -> bool:
        return not self.text.strip()

    def names(self) -> Set[str]:
        return {n.lower() for chain in self.chains for n in chain}


@dataclass
class Item:
    attack: str
    question: str
    world: World
    #: The exact derivation, in order, from target back to root. Empty where the pass is silence.
    chain: Tuple[str, ...] = ()
    #: Nodes that must NOT appear. A distractor named is a wrong explanation, not a long one.
    forbidden: Tuple[str, ...] = ()
    #: The two claims a `contradiction` item wants reported as a dispute.
    dispute: Tuple[str, str] = ()
    #: Nodes that must ALL be named, without being ordered relative to one another. A collider's
    #: two causes; a `legs` item's two roots.
    both: Tuple[str, ...] = ()
    #: A pair that must never appear on the *same* chain. Two independent causes of one effect are
    #: not a sequence, and reporting them as one invents an edge.
    forbid_link: Tuple[str, str] = ()
    want: Verdict = Verdict.RIGHT
    note: str = ""


@dataclass
class Answer:
    item: Item
    said: str = ""
    got: Verdict = Verdict.UNKNOWN
    passed: bool = False
    why: str = ""

    @property
    def confabulated(self) -> bool:
        """She answered where the evidence did not support one. The failure that matters most."""
        return self.item.want in (Verdict.UNKNOWN,) and self.got not in (Verdict.UNKNOWN,)

    def to_dict(self) -> Dict[str, Any]:
        return {"attack": self.item.attack, "question": self.item.question,
                "said": self.said, "want": self.item.want.value, "got": self.got.value,
                "passed": self.passed, "why": self.why,
                "chain": list(self.item.chain), "confabulated": self.confabulated}


@dataclass
class Paper:
    name: str
    answers: List[Answer] = field(default_factory=list)

    @property
    def asked(self) -> int:
        return len(self.answers)

    @property
    def passed(self) -> int:
        return sum(1 for a in self.answers if a.passed)

    @property
    def score(self) -> float:
        return round(self.passed / self.asked, 4) if self.asked else 0.0

    @property
    def confabulated(self) -> int:
        return sum(1 for a in self.answers if a.confabulated)

    @property
    def silent(self) -> int:
        return sum(1 for a in self.answers if a.got is Verdict.UNKNOWN)

    @property
    def wrong(self) -> int:
        return sum(1 for a in self.answers if a.got is Verdict.WRONG)

    def to_dict(self) -> Dict[str, Any]:
        return {"attack": self.name, "asked": self.asked, "passed": self.passed,
                "wrong": self.wrong, "silent": self.silent,
                "confabulated": self.confabulated, "score": self.score,
                "silence_is_the_pass": self.name in SILENT_PASS}


@dataclass
class Report:
    papers: List[Paper] = field(default_factory=list)
    seed: int = DEFAULT_SEED

    def paper(self, name: str) -> Optional[Paper]:
        for got in self.papers:
            if got.name == name:
                return got
        return None

    @property
    def answering(self) -> List[Paper]:
        """The papers that reward an answer. Never averaged with the ones that reward silence."""
        return [p for p in self.papers
                if p.name not in SILENT_PASS and p.name not in CONFLICT_PASS]

    @property
    def score(self) -> float:
        asked = sum(p.asked for p in self.answering)
        return round(sum(p.passed for p in self.answering) / asked, 4) if asked else 0.0

    @property
    def confabulated(self) -> int:
        """Answers given on the papers where the evidence did not support one."""
        return sum(p.confabulated for p in self.papers)

    @property
    def restraint(self) -> float:
        """Of the items whose only pass is silence, how many got silence."""
        pool = [a for p in self.papers for a in p.answers if a.item.want is Verdict.UNKNOWN]
        if not pool:
            return 0.0
        return round(sum(1 for a in pool if a.got is Verdict.UNKNOWN) / len(pool), 4)

    def to_dict(self) -> Dict[str, Any]:
        return {"seed": self.seed, "score": self.score, "restraint": self.restraint,
                "confabulated": self.confabulated,
                "papers": [p.to_dict() for p in self.papers]}


# --------------------------------------------------------------------------- #
# The gauntlet
# --------------------------------------------------------------------------- #
class Gauntlet:
    """Nine attacks, each over its own minted world.

    ``ask`` is injected so the same gauntlet can be pointed at anything that takes a question and
    a fact store — the walk in :mod:`nyxara.njp.explain` today, and whatever replaces it. Nothing
    in this class knows how the answer is produced, which is what keeps it an instrument rather
    than a second copy of the thing it measures.
    """

    def __init__(self, *, seed: int = DEFAULT_SEED, limit: int = DEFAULT_LIMIT,
                 ask: Any = None) -> None:
        self.seed = int(seed)
        self.limit = max(1, int(limit))
        self.ask = ask if ask is not None else _walk

    # ---- the worlds ---------------------------------------------------- #
    def _chain_world(self, rng: random.Random, length: int) -> Tuple[World, List[str]]:
        """``a → b → c → …``. The derivation is the whole path, target last."""
        nodes = _words(rng, length)
        world = World([Fact(nodes[i], "causes", nodes[i + 1]) for i in range(length - 1)])
        return world, nodes

    @staticmethod
    def _covered() -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """Which of :data:`WHY_FORMS` somebody has actually shown her, and which nobody has.

        Computed by comparing the form strings against :data:`nyxara.njp.asking.LESSON` and the
        hand-written table's own control phrasing — a **factual** check, not a grade. It is done at
        runtime rather than written down so that adding a phrasing to the lesson moves an item from
        one paper to the other automatically, and a lesson that quietly grew to cover the whole
        exam would show up as ``wording_new`` running out of items rather than as a rising score.
        """
        try:
            from nyxara.njp.asking import LESSON
            taught = {form for form, walk in LESSON if walk}
        except Exception:  # noqa: BLE001
            taught = set()
        taught.add("why does {x} happen?")      # the phrasing `explainread`'s table carries
        covered = tuple(f for f in WHY_FORMS if f in taught)
        return covered, tuple(f for f in WHY_FORMS if f not in taught)

    def _wording(self, attack: str, forms: Sequence[str], count: int,
                 salt: int) -> List[Item]:
        rng = random.Random(self.seed ^ salt)
        out: List[Item] = []
        if not forms:
            return out
        for n in range(count):
            world, nodes = self._chain_world(rng, 3)
            form = forms[n % len(forms)]
            out.append(Item(attack=attack, question=form.format(x=nodes[-1]), world=world,
                            chain=tuple(reversed(nodes)), note=form))
        return out

    def paper_wording(self, count: int) -> List[Item]:
        """Phrasings somebody demonstrated, on topics and graphs nobody did.

        This measures whether **teaching works**: the cue is known, everything else is minted. It
        is a different question from the one below and mixing them was the first version's error —
        a single ``wording`` number lets *teaching more phrasings* look identical to *reasoning
        better*, and those are the two things this pair of papers exists to keep apart.
        """
        covered, _ = self._covered()
        return self._wording("wording", covered, count, 0x1111)

    def paper_wording_new(self, count: int) -> List[Item]:
        """Phrasings nobody demonstrated. The honest ceiling of an induced grammar.

        A score here is not expected to be high and a high one would be suspicious: it would mean
        the lesson had grown to cover the exam. What it is for is the **gap** between it and
        ``wording``, which is the size of what induction bought and what it did not.
        """
        _, untaught = self._covered()
        return self._wording("wording_new", untaught, count, 0x1112)

    def paper_entities(self, count: int) -> List[Item]:
        """The control. Same shape, the phrasing the grammar was built for, minted nodes only."""
        rng = random.Random(self.seed ^ 0x2222)
        out: List[Item] = []
        for _ in range(count):
            world, nodes = self._chain_world(rng, rng.randint(3, 4))
            out.append(Item(attack="entities", question=f"why does {nodes[-1]} happen?",
                            world=world, chain=tuple(reversed(nodes)),
                            note="nothing here can be recognised, only derived"))
        return out

    def paper_shape(self, count: int) -> List[Item]:
        """Five graph shapes, one of which is a collider and is why this paper exists."""
        rng = random.Random(self.seed ^ 0x3333)
        out: List[Item] = []
        shapes = ("chain", "fork", "collider", "diamond", "random")
        for n in range(count):
            shape = shapes[n % len(shapes)]
            nodes = _words(rng, 5)
            a, b, c, d, e = nodes
            if shape == "chain":
                facts = [Fact(a, "causes", b), Fact(b, "causes", c)]
                item = Item(attack="shape", question=f"why does {c} happen?", world=World(facts),
                            chain=(c, b, a), note="chain")
            elif shape == "fork":
                # a causes both b and c; asking about c must not route through b.
                facts = [Fact(a, "causes", b), Fact(a, "causes", c)]
                item = Item(attack="shape", question=f"why does {c} happen?", world=World(facts),
                            chain=(c, a), forbidden=(b,), note="fork: b is a sibling, not a cause")
            elif shape == "collider":
                # a and b both cause c and have nothing to do with each other.
                #
                # The gold here was `(c, a)` for one measurement and that was the exam's error,
                # not hers: she named both causes, which is every answer that would be right, and
                # was marked wrong for not having guessed which one the generator happened to
                # pick first. Same rule broken in the same way as `explainschool.paper_mechanism`
                # one version earlier. The pass is now **both, and neither routed through the
                # other** — which is what a collider actually asserts.
                facts = [Fact(a, "causes", c), Fact(b, "causes", c)]
                item = Item(attack="shape", question=f"why does {c} happen?", world=World(facts),
                            chain=(), both=(a, b), forbid_link=(a, b),
                            note="collider: two independent causes, neither via the other")
            elif shape == "diamond":
                facts = [Fact(a, "causes", b), Fact(a, "causes", c),
                         Fact(b, "causes", d), Fact(c, "causes", d)]
                item = Item(attack="shape", question=f"why does {d} happen?", world=World(facts),
                            chain=(d, b, a), note="diamond: two routes to the same root")
            else:
                order = list(nodes)
                rng.shuffle(order)
                facts = []
                for i in range(1, len(order)):
                    for j in range(i):
                        if rng.random() < 0.4:
                            facts.append(Fact(order[j], "causes", order[i]))
                if not facts:
                    facts = [Fact(order[0], "causes", order[1])]
                target = facts[-1].object
                # The gold is the **longest** backward path from the target, computed here rather
                # than assumed. It was `(target, facts[-1].subject)` — one hop — for one
                # measurement, and every failure the paper reported was an answer that was longer
                # and correct: *"because kafat, which causes negukem, which causes ritozal"*
                # marked wrong against a gold of `ritozal -> negukem`. Seven of forty, all of them
                # the exam's error. A why-question asks for the part she was not told, so the
                # deepest derivation is the answer and a prefix of it is not.
                item = Item(attack="shape", question=f"why does {target} happen?",
                            world=World(facts), chain=_deepest(facts, target),
                            note="random dag")
            out.append(item)
        return out

    def paper_distractors(self, count: int) -> List[Item]:
        """True facts that answer a question nobody asked. Naming one is a wrong explanation."""
        rng = random.Random(self.seed ^ 0x4444)
        out: List[Item] = []
        for _ in range(count):
            world, nodes = self._chain_world(rng, 3)
            noise = _words(rng, 4)
            # **On the causal relation, and downstream.** The first version hung the distractors
            # off `purpose`, `has_part` and `is_a`, and a backward causal walk does not read any
            # of those — so the paper scored 1.000 by injecting noise into a channel the thing
            # under test was never listening to. A distractor only distracts if it is somewhere
            # the walk goes.
            #
            # These are all true and none is an ancestor of the target: what the target itself
            # causes, what the middle also causes besides the target, and a cause of that
            # sibling. Every one is reachable in one hop from a node on the real chain, and every
            # one is the wrong direction or the wrong branch.
            world.add(Fact(nodes[-1], "causes", noise[0]),
                      Fact(nodes[1], "causes", noise[1]),
                      Fact(noise[2], "causes", noise[1]),
                      Fact(noise[0], "causes", noise[3]))
            out.append(Item(attack="distractors", question=f"why does {nodes[-1]} happen?",
                            world=world, chain=tuple(reversed(nodes)), forbidden=tuple(noise),
                            note="four true irrelevant facts in the same neighbourhood"))
        return out

    def paper_contradiction(self, count: int) -> List[Item]:
        """Two edges that cannot both hold. The dispute is the answer."""
        rng = random.Random(self.seed ^ 0x5555)
        out: List[Item] = []
        for _ in range(count):
            a, b, target = _words(rng, 3)
            world = World([Fact(a, "causes", target, 0.9),
                           Fact(b, "causes", target, 0.9),
                           # and the claim that makes them exclusive rather than merely two causes
                           Fact(a, "excludes", b, 0.95),
                           Fact(b, "excludes", a, 0.95)])
            out.append(Item(attack="contradiction",
                            question=f"why does {target} happen?", world=world,
                            dispute=(a, b), want=Verdict.CONFLICT,
                            note="two causes stated to exclude each other"))
        return out

    def paper_homonym(self, count: int) -> List[Item]:
        """One spelling, two disjoint neighbourhoods. The V.36 atrium defect, asked directly."""
        rng = random.Random(self.seed ^ 0x6666)
        out: List[Item] = []
        for _ in range(count):
            whole, part, right, wrong_kind, right_kind, other = _words(rng, 6)
            world = World([
                Fact(whole, "has_part", part),
                Fact(part, "is_a", right_kind),
                Fact(right_kind, "part_of", whole),
                Fact(part, "purpose", right),
                # the other sense of the same spelling, with its own kind and its own purpose
                Fact(part, "is_a", wrong_kind),
                Fact(wrong_kind, "part_of", other),
                Fact(part, "purpose", other),
            ])
            out.append(Item(attack="homonym", question=f"how does {whole} work?", world=world,
                            chain=(whole, part, right), forbidden=(other,),
                            note=f"{part} is two things; {other} belongs to the other one"))
        return out

    def paper_gap(self, count: int) -> List[Item]:
        """The middle hop is deleted after the graph is built. UNKNOWN is the only pass."""
        rng = random.Random(self.seed ^ 0x7777)
        out: List[Item] = []
        for _ in range(count):
            a, b, c, d = _words(rng, 4)
            # a → b → c → d with `b causes c` deleted, and the question asks about **c** — the node
            # whose only incoming edge is the one that was removed.
            #
            # The first version asked about `d` and wanted silence, and that was simply a wrong
            # item: `c causes d` is still stated, so *"because c"* is a fact she was told and
            # marking it a confabulation made the paper score 0.000 for being right. Twenty
            # confabulations that were nothing of the kind. The trap only works when the target's
            # own support is what went missing, and everything around it still stands so that
            # there is something to invent *from*.
            world = World([Fact(a, "causes", b), Fact(c, "causes", d),
                           Fact(c, "is_a", _word(rng)), Fact(b, "is_a", _word(rng))])
            out.append(Item(attack="gap", question=f"why does {c} happen?", world=world,
                            chain=(), forbidden=(a, b, d), want=Verdict.UNKNOWN,
                            note=f"{b}->{c} was deleted; {c} is still described and still causes "
                                 f"{d}, so there is plenty to invent from"))
        return out

    def paper_legs(self, count: int) -> List[Item]:
        """Two independently derived legs, meeting only at the target."""
        rng = random.Random(self.seed ^ 0x8888)
        out: List[Item] = []
        for _ in range(count):
            a1, a2, b1, b2, target = _words(rng, 5)
            world = World([Fact(a1, "causes", a2), Fact(a2, "causes", target),
                           Fact(b1, "causes", b2), Fact(b2, "causes", target),
                           # and the claim that neither leg is sufficient on its own
                           Fact(target, "requires", a2), Fact(target, "requires", b2)])
            out.append(Item(attack="legs", question=f"why does {target} happen?", world=world,
                            chain=(), both=(a1, b1), want=Verdict.RIGHT,
                            note="neither leg alone is sufficient; the answer is a conjunction"))
        return out

    def paper_unknown(self, count: int) -> List[Item]:
        """The evidence genuinely does not support an answer. Silence is the only pass."""
        rng = random.Random(self.seed ^ 0x9999)
        out: List[Item] = []
        for n in range(count):
            nodes = _words(rng, 4)
            target = nodes[0]
            if n % 2:
                # An island: the target is named by facts that say nothing about what caused it.
                world = World([Fact(target, "is_a", nodes[1]),
                               Fact(target, "has_part", nodes[2]),
                               Fact(nodes[3], "causes", nodes[2])])
                note = "the target is described, never caused"
            else:
                # A named entity with no edges at all except one pointing away from it.
                world = World([Fact(target, "causes", nodes[1]),
                               Fact(nodes[1], "causes", nodes[2])])
                note = "every edge points away from the target"
            out.append(Item(attack="unknown", question=f"why does {target} happen?",
                            world=world, chain=(), want=Verdict.UNKNOWN, note=note))
        return out

    # ---- grading -------------------------------------------------------- #
    def grade(self, item: Item, reply: Optional[Reply] = None) -> Answer:
        """One item. Six checks, in an order that is itself a decision.

        ``reply`` is accepted so the **grader can be tested without a walk**. Four of these papers
        were graded wrongly on their first run and the tests that catch that have to feed a known
        answer in and assert the verdict; going through the walk would test the walk instead.

        A forbidden node is checked before anything else and on every paper, because an
        explanation containing an irrelevant cause is a false claim rather than a verbose true
        one. Then the papers whose pass is silence or a dispute, which are graded on what she
        *declined* to say. Then joint necessity. Only last does the exact derivation get looked
        at, and it is exact: partial credit for a chain is credit for a wrong explanation.
        """
        out = Answer(item=item)
        if reply is None:
            try:
                reply = self.ask(item.question, item.world)
            except Exception as exc:  # noqa: BLE001
                out.why = f"the walk raised {type(exc).__name__}"
                out.got = Verdict.WRONG
                return out
        out.said = reply.text
        low = " ".join(reply.text.split()).lower()
        out.got = (Verdict.CONFLICT if reply.conflict
                   else Verdict.UNKNOWN if reply.silent else Verdict.RIGHT)

        named = [bad for bad in item.forbidden if bad.lower() in low]
        if named and not reply.silent:
            out.got = Verdict.WRONG
            out.why = f"named a forbidden node: {named[0]}"
            return out

        # Two independent causes must not be reported as a sequence. A collider says A and B both
        # cause C and are unrelated; a chain through one to the other asserts an edge nobody
        # stated, which is the single most common way a graph walk invents a fact.
        if item.forbid_link:
            first, second = (n.lower() for n in item.forbid_link)
            for chain in reply.chains:
                names = [n.lower() for n in chain]
                if first in names and second in names:
                    out.got = Verdict.WRONG
                    out.why = f"put {first} and {second} on one chain; they are independent"
                    return out

        if item.want is Verdict.UNKNOWN:
            out.passed = reply.silent
            out.why = ("silence was the pass" if out.passed
                       else "answered where nothing supports one")
            return out

        if item.want is Verdict.CONFLICT:
            # The **flag**, not the two names. Listing two causes is not noticing that they were
            # stated to exclude each other, and a grader that accepted the listing scored this
            # paper 1.000 against a walk with no notion of a conflict at all.
            out.passed = reply.conflict and all(p.lower() in low for p in item.dispute)
            out.why = ("reported the dispute" if out.passed
                       else "listed the claims without noticing they exclude each other"
                            if all(p.lower() in low for p in item.dispute)
                            else "did not reach both claims")
            if not out.passed and out.got is Verdict.RIGHT:
                out.got = Verdict.WRONG
            return out

        if item.attack == "legs":
            # Both roots, **and** marked as jointly required. Same defect as `contradiction`:
            # "A or B" and "A and B" are different answers built from the same two names, and only
            # `joint` distinguishes them.
            both = all(p.lower() in low for p in item.both)
            out.passed = both and reply.joint
            out.why = ("both legs, derived as jointly required" if out.passed
                       else "named both but as alternatives, not as a conjunction" if both
                            else "one leg only; the target needs both")
            if not out.passed:
                out.got = Verdict.WRONG if not reply.silent else Verdict.UNKNOWN
            return out

        if item.both:
            out.passed = all(p.lower() in low for p in item.both)
            out.why = ("named every independent cause" if out.passed
                       else "did not name all of them")
            if not out.passed:
                out.got = Verdict.WRONG if not reply.silent else Verdict.UNKNOWN
            return out

        wanted = [n.lower() for n in item.chain]
        got = [n.lower() for n in reply.nodes]
        # Exact, and the one allowance is a diamond: two routes to the same root are both the
        # derivation, so the check is the same length with the same ends rather than the same
        # middle.
        out.passed = bool(wanted) and (
            got == wanted or (len(got) == len(wanted) and got[0] == wanted[0]
                              and got[-1] == wanted[-1]))
        if not out.passed:
            out.got = Verdict.WRONG if not reply.silent else Verdict.UNKNOWN
            out.why = f"wanted {' -> '.join(wanted)}, got {' -> '.join(got) or '(silence)'}"
        else:
            out.why = "the derivation, in order"
        return out

    def items(self, attack: str) -> List[Item]:
        return getattr(self, f"paper_{attack}")(self.limit)

    def run(self, attacks: Optional[Sequence[str]] = None) -> Report:
        report = Report(seed=self.seed)
        for name in (attacks or ATTACKS):
            paper = Paper(name=name)
            for item in self.items(name):
                paper.answers.append(self.grade(item))
            report.papers.append(paper)
        return report


# --------------------------------------------------------------------------- #
# The thing under test
# --------------------------------------------------------------------------- #
def _ensure_taught() -> None:
    """Put the reader in the state a live brain is in, once.

    The gauntlet must measure what she actually ships as. Run without this, ``wording`` reported
    0.200 against a reader nobody had shown anything to — a number about an unconfigured process
    rather than about her — while :meth:`~nyxara.njp.brain.NJPBrain._build_explainer` teaches the
    forms on every real construction. Measuring a configuration that never runs is the same class
    of error as grading a walk against its own output.
    """
    from nyxara.njp import explainread

    if explainread.LEARNED is None:
        from nyxara.njp.asking import install
        install()


def _walk(question: str, world: World) -> Reply:
    """Point the gauntlet at :mod:`nyxara.njp.explain`.

    ``conflict`` and ``joint`` are read off the explanation, never off its text. They were both
    hard-coded False until V.39 because the walk had no representation for *"these two cannot both
    hold"* or *"these two are both required"* — and filling them in from the text would have been
    the harness answering its own question. They now come from
    :mod:`nyxara.njp.predator`, which is a different organ than the one being examined.

    ``chains`` is every chain it returned, best first — the grader reads the leading one for a
    derivation and all of them for a forbidden link, because a wrong edge buried in the fourth
    chain is still a wrong edge she was willing to state.
    """
    from nyxara.njp.explain import Explainer

    _ensure_taught()
    explainer = Explainer(world)
    got = explainer.ask(question)
    if got is None:
        return Reply()
    if hasattr(got, "orders"):
        return Reply(text=" → ".join(got.first) if got.first else "",
                     chains=[list(got.first)] if got.first else [])
    return Reply(text=got.text(), chains=[list(c.nodes) for c in got.chains],
                 conflict=bool(getattr(got, "conflict", False)),
                 joint=bool(getattr(got, "joint", False)))


def run(*, seed: int = DEFAULT_SEED, limit: int = DEFAULT_LIMIT,
        attacks: Optional[Sequence[str]] = None, ask: Any = None) -> Report:
    return Gauntlet(seed=seed, limit=limit, ask=ask).run(attacks)


def render(report: Report) -> str:
    lines = [f"the gauntlet — seed {report.seed}", "",
             f"{'attack':16} {'asked':>6} {'passed':>7} {'wrong':>6} {'silent':>7} {'score':>7}"]
    for paper in report.papers:
        tail = "   (silence is the pass)" if paper.name in SILENT_PASS else (
            "   (the dispute is the pass)" if paper.name in CONFLICT_PASS else "")
        lines.append(f"{paper.name:16} {paper.asked:6} {paper.passed:7} {paper.wrong:6} "
                     f"{paper.silent:7} {paper.score:7.3f}{tail}")
    lines += ["",
              f"{report.score:.3f} over the papers that reward answering",
              f"{report.restraint:.3f} restraint — of the items whose only pass is silence, "
              f"how many got it",
              f"{report.confabulated} confabulations — answers where nothing supported one"]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="the gauntlet the last exam could not be")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--attack", action="append", default=None, choices=list(ATTACKS))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--failures", type=int, default=0,
                    help="print this many failing items per paper, with what was wanted")
    args = ap.parse_args(argv)
    report = run(seed=args.seed, limit=args.limit, attacks=args.attack)
    if args.json:
        print(json.dumps(report.to_dict(), indent=1))
        return 0
    print(render(report))
    if args.failures:
        for paper in report.papers:
            bad = [a for a in paper.answers if not a.passed][:args.failures]
            if not bad:
                continue
            print(f"\n─── {paper.name}")
            for answer in bad:
                print(f"  Q  {answer.item.question}")
                print(f"     note   {answer.item.note}")
                print(f"     said   {answer.said[:160]!r}")
                print(f"     why    {answer.why}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
