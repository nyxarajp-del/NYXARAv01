"""NYXARA · njp/explain.py — what, how, and why (🧭, NJP V.36).

Measured cold, on the shipped world corpus, before a line of this file existed::

    what is force                    'physical quantity'
    what is a mammal                 'vertebrate'
    what causes rain                 'cloud'
    why does water boil              ''
    why does an object accelerate    ''
    how does a plant make food       ''
    how does photosynthesis work     ''
    how do you boil an egg           ''

Three of eight. She answers **what**, she answers a *named* one-hop relation, and the two
questions a person actually asks about the world are not weak in her, they are absent.

That is not a corpus gap. 13,755 facts is a large store and it holds ``cloud causes rain``,
``condensation causes cloud`` and ``cooling causes condensation`` all three. The gap is that
*why* and *how* are not relations. There is no edge in any graph labelled "why". An explanation
is a **path**, and a path is something you have to walk.

So this module holds no explanations. It holds three walks.

Why
---
Two of them, and keeping them apart is the first thing this file does.

**Mechanistic why** — *what brought this about* — walks **backwards** along the edges that state
production: ``S causes T`` read in reverse, and ``T occurs_when C``, which is the same claim
already stored pointing the other way. It keeps walking, so the answer to *why does rain happen*
is not ``cloud``; it is ``cooling → condensation → cloud → rain``, four facts she was told
separately and one chain nobody ever wrote down.

**Teleological why** — *what this is for* — walks **forwards** along ``purpose``, and up through
``is_a`` when the thing itself carries none: a stethoscope is for listening to the chest, and
listening to the chest is for diagnosis.

A thing can have both, and where it does the answer is both, in two blocks, labelled. Answering a
*what for* with a *what brought about* is the characteristic error here and it is a wrong answer,
not a partial one — the fuse in a plug is caused by a factory and is for breaking the circuit, and
only one of those is what anybody asked.

There is a third strand that is neither, and it is kept third rather than folded into *because*:
``T requires R`` is an **enabling condition**, not a cause. Fire requires oxygen; oxygen did not
cause the fire. Rendering a requirement as a cause is how an explanation becomes false while every
fact in it stays true, so a requirement is glossed *"it needs"* and never *"because"*.

How
---
Also two, and also different questions.

**How does X work** is a *mechanism*: X's parts and what each of them does, assembled into the
chain that ends at what X is for. Derived from ``has_part``/``consists_of`` crossed with
``causes``/``purpose`` — never from a stored paragraph.

**How do you X** is a *procedure*: a set of steps, and an **order**.

The order is the part that matters, and it is the reason this file is not a list of recipes. A
procedure in the corpus is told as an unordered set — ``has_step`` — plus, on each step, what that
step ``requires`` of the others. **Nothing states the sequence.** The sequence is derived, every
time, by a topological walk over the prerequisites, and that derivation is examinable on a
procedure minted out of nonsense words five seconds ago, which a stored recipe is not.

Three properties of that walk are decisions, not defaults:

**Where the prerequisites underdetermine the order, the answer is every order they allow, not one
of them.** Beat the eggs and heat the pan are both first; a walk that emits one sequence and calls
it *the* answer has invented an ordering constraint nobody stated. This package has said the same
thing about two parses of a sentence and two equally supported triples since V.13, and it is the
same rule: *where two readings survive, the answer is that there are two.* :attr:`Plan.orders`
holds them all, up to :data:`MAX_ORDERS`, and :attr:`Plan.determined` says whether there was one.

**A prerequisite cycle is reported as a cycle.** Kahn's algorithm run to exhaustion leaves the
cycle behind in the queue, and the honest report is that the steps as told cannot be sequenced —
not a sequence with one edge quietly dropped to make it come out.

**A step that is named as a prerequisite but never listed as a step is a hole in the telling**, and
it is named in :attr:`Plan.dangling` rather than silently treated as already done.

What the walks may not do
-------------------------
Every bound here was put in because removing it produced a confident falsehood.

*Depth.* :data:`MAX_DEPTH` hops, and confidence decays by :data:`DECAY` a hop, exactly as
``core._inherit`` prices an inheritance chain. A four-hop causal chain through a corpus this
size can reach almost anything from almost anything.

*Confidence floor.* A chain whose **weakest fact** is below :data:`MIN_CHAIN_CONFIDENCE` is not
returned — the weakest fact rather than the decayed product, for the reason :attr:`Chain.support`
records at length. It is not returned *quietly*, either — :attr:`Explanation.pruned` counts what the floor removed, so a topic that
looks unexplained because everything about it was weak reads differently from one that had no
chain at all.

*No revisiting.* A node already on the chain is not walked into again, which is what stops
``evaporation causes cloud causes rain causes evaporation`` from being an explanation of anything.

*Silence.* No chain, no answer. :class:`Explanation` with no chains is the normal outcome for a
topic the corpus does not reach, and the exam scores that as correct on its abstention paper. A
walk that always finds *something* is a walk that has stopped being about the graph.

Nothing here is a table of answers, and there is a test that asserts it: empty the fact store and
every question in this module returns silence.

Pure standard library, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "Step", "Chain", "Explanation", "Plan", "Explainer",
    "MAX_DEPTH", "DECAY", "MIN_CHAIN_CONFIDENCE", "MAX_CHAINS", "MAX_ORDERS",
    "DEMOTE_MANY_KINDED",
    "PRODUCES", "ENABLES", "PURPOSE", "PARTS", "STEPS",
]

#: How many hops a chain may be. Four causal hops through 13,755 facts reaches most of the graph
#: from most of it; three is where the chains stayed about their topic.
MAX_DEPTH = 3

#: What one hop costs. The same shape as ``core._inherit``'s per-link decay: a claim assembled out
#: of four claims is weaker than any of them, and saying so is the difference between a derivation
#: and an assertion.
DECAY = 0.85

#: Below this a chain is counted and dropped rather than offered, and what is compared against it
#: is the chain's **weakest link**, not its decayed product. See :attr:`Chain.support` for the
#: measurement that changed it.
MIN_CHAIN_CONFIDENCE = 0.30

#: Chains returned per explanation. A hub like *water* has hundreds of paths into it and a report
#: that printed all of them would be a listing, not an explanation.
MAX_CHAINS = 6

#: Orders returned when the prerequisites underdetermine the sequence. Six steps with no
#: constraints have 720 orders; the count is exact in :attr:`Plan.order_count`, the list is capped.
MAX_ORDERS = 12

#: Relations that state **production**: the subject brings the object about. Walked backwards for
#: a mechanistic *why*, forwards for *what happens next*.
PRODUCES: Tuple[str, ...] = ("causes", "produces", "increases", "decreases")

#: Relations that state an **enabling condition**. Never rendered as a cause. ``occurs_when`` is
#: here rather than in :data:`PRODUCES` for the direction it is stored in — ``T occurs_when C``
#: has T as subject — and it is read as production when walked, which is what
#: :meth:`Explainer._incoming` does with it.
ENABLES: Tuple[str, ...] = ("requires",)

#: Stored on the effect, naming the condition. Backwards-facing already, so it is read directly.
OCCURS = "occurs_when"

#: The teleological relation, walked forwards.
PURPOSE: Tuple[str, ...] = ("purpose",)

#: Decomposition, for a mechanism.
PARTS: Tuple[str, ...] = ("has_part", "consists_of", "involves")

#: A procedure's steps, told as a set.
STEPS: Tuple[str, ...] = ("has_step",)

#: Kind-of, for climbing to a purpose the thing itself does not carry, and for noticing a word
#: that is two words.
KIND = "is_a"

#: How a chain through a many-kinded node is treated: **ranked last among its equals, and nothing
#: else**. Not a confidence penalty and not a filter. It was both, for one measurement, and see
#: :meth:`Explainer.many_kinded` for what that cost.
DEMOTE_MANY_KINDED = True


# --------------------------------------------------------------------------- #
# What a chain is made of
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Step:
    """One stated fact, with the direction it was traversed in.

    ``forward`` is False when the edge was walked against its stored direction — ``cloud causes
    rain`` read as *rain happened because of cloud*. Keeping the flag rather than flipping the
    triple is what lets :meth:`Chain.gloss` say "because" in one case and "which causes" in the
    other out of the same record, and lets a reader check the chain against the store.
    """

    subject: str
    relation: str
    object: str
    confidence: float = 1.0
    forward: bool = True

    @property
    def source(self) -> str:
        """The end the walk came *from*."""
        return self.object if not self.forward else self.subject

    @property
    def target(self) -> str:
        """The end the walk arrived *at*."""
        return self.subject if not self.forward else self.object

    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.relation, self.object)


@dataclass
class Chain:
    """An ordered walk, and the claim it amounts to.

    ``kind`` is one of ``because`` (mechanistic), ``for`` (teleological), ``needs`` (enabling
    condition) or ``mechanism``. It is not decoration: the exam grades a *why does* item against
    ``because`` and a *what for* item against ``for``, and a chain of the wrong kind is wrong.
    """

    topic: str
    kind: str
    steps: List[Step] = field(default_factory=list)
    priority: int = 0
    ambiguous: List[str] = field(default_factory=list)

    @property
    def depth(self) -> int:
        return len(self.steps)

    @property
    def confidence(self) -> float:
        """The product of the stated confidences, decayed once per hop."""
        out = 1.0
        for step in self.steps:
            out *= max(0.0, min(1.0, float(step.confidence))) * DECAY
        return round(out, 4)

    @property
    def support(self) -> float:
        """The weakest fact on the chain — what the floor is judged against.

        It used to be judged against :attr:`confidence`, the decayed product, and that was wrong
        in a way that took a two-line test to expose. The product falls with **length**, so a
        floor over it is a second depth bound wearing a confidence's clothes — and, worse, one
        whose effective depth depends on how confident the source corpus happened to be. The
        shipped corpus states its facts at 0.85 and reached three hops. The same three facts
        ingested without stated confidences land at :data:`~nyxara.njp.ingest._DEFAULT_CONFIDENCE`
        — 0.6 — and 0.6² × 0.85² is 0.26, under the floor, so ``cooling → condensation → cloud →
        rain`` returned ``cloud`` and nothing else. The feature was off for any corpus that did not
        happen to be confident enough, and nothing said so.

        The two jobs are now done by the two things that should do them. :data:`MAX_DEPTH` bounds
        length. The floor asks whether the chain is built out of claims worth building on, which
        is a question about the claims — a chain is no stronger than its weakest fact, however
        short it is, and no weaker for being long if every fact in it is solid.

        :attr:`confidence` is unchanged and still decays per hop: a claim assembled out of four
        claims *is* weaker than any of them, and saying so is the difference between a derivation
        and an assertion. It is reported; it no longer decides.
        """
        return round(min((float(s.confidence) for s in self.steps), default=1.0), 4)

    @property
    def head(self) -> str:
        """The far end — the ultimate cause, or the ultimate purpose."""
        return self.steps[-1].target if self.steps else self.topic

    @property
    def nodes(self) -> List[str]:
        out = [self.topic]
        for step in self.steps:
            out.append(step.target)
        return out

    def gloss(self) -> str:
        """The chain as one English sentence, in the direction it was asked in."""
        if not self.steps:
            return ""
        if self.kind == "for":
            body = ", which is for ".join(step.target for step in self.steps)
            return f"{self.topic} is for {body}"
        if self.kind == "needs":
            return f"{self.topic} needs " + ", which needs ".join(s.target for s in self.steps)
        if self.kind == "mechanism":
            run = " → ".join(self.nodes)
            return run + (f"  ({self.ambiguous[0]} is a word this store gives two kinds)"
                          if self.ambiguous else "")
        # because: the walk went backwards, so read it forwards to say it out loud.
        run = list(reversed(self.nodes))
        return f"because {run[0]}" + "".join(f", which causes {n}" for n in run[1:])

    def to_dict(self) -> Dict[str, Any]:
        return {"topic": self.topic, "kind": self.kind, "depth": self.depth,
                "confidence": self.confidence, "support": self.support, "head": self.head,
                "ambiguous": list(self.ambiguous),
                "nodes": self.nodes, "gloss": self.gloss(),
                "steps": [list(s.as_tuple()) + [s.forward] for s in self.steps]}


@dataclass
class Explanation:
    """What came back from one question, including what did not.

    ``pruned`` is the count the confidence floor removed and ``considered`` the number of chains
    the walk built before ranking. A topic with ``chains=[]`` and ``considered=0`` was not reached
    by the graph at all; one with ``considered=9`` and ``pruned=9`` was reached only weakly, and
    those are different facts about her that a bare empty list would report identically.
    """

    topic: str
    question: str = ""
    chains: List[Chain] = field(default_factory=list)
    considered: int = 0
    pruned: int = 0
    why: str = ""

    @property
    def answered(self) -> bool:
        return bool(self.chains)

    @property
    def best(self) -> Optional[Chain]:
        return self.chains[0] if self.chains else None

    def of_kind(self, kind: str) -> List[Chain]:
        return [c for c in self.chains if c.kind == kind]

    def text(self) -> str:
        """The whole answer as prose, one line per chain, causes before purposes."""
        return "\n".join(c.gloss() for c in self.chains if c.gloss())

    def to_dict(self) -> Dict[str, Any]:
        return {"topic": self.topic, "question": self.question, "answered": self.answered,
                "considered": self.considered, "pruned": self.pruned, "why": self.why,
                "chains": [c.to_dict() for c in self.chains]}


@dataclass
class Plan:
    """A procedure, and the orders its prerequisites allow.

    ``orders`` is a list of sequences, not a sequence. One entry means the telling determined the
    order; several mean it did not, and the plural is the answer rather than a failure to pick.
    """

    topic: str
    steps: List[str] = field(default_factory=list)
    needs: Dict[str, List[str]] = field(default_factory=dict)
    orders: List[List[str]] = field(default_factory=list)
    order_count: int = 0
    cycle: List[str] = field(default_factory=list)
    dangling: List[str] = field(default_factory=list)
    why: str = ""

    @property
    def answered(self) -> bool:
        return bool(self.orders)

    @property
    def determined(self) -> bool:
        """Did the prerequisites pin exactly one sequence?"""
        return self.order_count == 1

    @property
    def first(self) -> List[str]:
        return list(self.orders[0]) if self.orders else []

    def before(self, earlier: str, later: str) -> bool:
        """Is *earlier* before *later* in **every** order the prerequisites allow?

        The question a plan is actually for. It is deliberately not "in the first order": a fact
        that holds in one of six admissible sequences is not a fact about the procedure.
        """
        if not self.orders:
            return False
        for order in self.orders:
            if earlier not in order or later not in order:
                return False
            if order.index(earlier) >= order.index(later):
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {"topic": self.topic, "steps": list(self.steps), "needs": dict(self.needs),
                "orders": [list(o) for o in self.orders], "order_count": self.order_count,
                "determined": self.determined, "cycle": list(self.cycle),
                "dangling": list(self.dangling), "why": self.why}


# --------------------------------------------------------------------------- #
# The walks
# --------------------------------------------------------------------------- #
class Explainer:
    """Three walks over a fact store. Holds no knowledge of its own.

    It is constructed around anything exposing ``facts`` as ``{(subject, predicate): [triple]}``
    — :class:`~nyxara.njp.grounding.Grounder` does, and so does the two-line stub the tests use to
    prove that the answers come from the store rather than from here.
    """

    def __init__(self, grounder: Any = None, *,
                 max_depth: int = MAX_DEPTH,
                 min_confidence: float = MIN_CHAIN_CONFIDENCE,
                 max_chains: int = MAX_CHAINS) -> None:
        self.grounder = grounder
        self.max_depth = max(1, int(max_depth))
        self.min_confidence = float(min_confidence)
        self.max_chains = max(1, int(max_chains))
        self._forward: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
        self._backward: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
        self._built = 0
        self.reindex()

    # ---- the index ------------------------------------------------------ #
    def _facts(self) -> Dict[Any, Any]:
        got = getattr(self.grounder, "facts", None)
        return got if isinstance(got, dict) else {}

    def reindex(self) -> None:
        """Rebuild both directions. Called on construction and whenever the store has grown.

        The backward index is the whole reason this is not a lookup: ``causes`` is stored on the
        cause and every *why* question names the effect, so without an inverted index a why-walk
        is a full scan of the store per hop and there is no fourth hop at any price.
        """
        self._forward = {}
        self._backward = {}
        facts = self._facts()
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
                conf = float(getattr(triple, "confidence", 1.0) or 0.0)
                self._forward.setdefault((self._key(subject), predicate), []).append((obj, conf))
                self._backward.setdefault((self._key(obj), predicate), []).append((str(subject), conf))
        self._built = sum(len(v) for v in self._forward.values())

    @property
    def edges(self) -> int:
        return self._built

    def _key(self, text: str) -> str:
        """An object's spelling *as a subject key*, the store's own rule.

        Same lesson ``general.GeneralKnowledgeExam.key`` records: ``canon`` singularises, so a
        chain that carried an object forward as its own spelling missed every next hop whose
        subject had been folded. A walk is nothing but that lookup repeated, so it fails everywhere
        at once.
        """
        g = self.grounder
        if g is not None:
            try:
                return g._key(text)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
        return " ".join(str(text or "").split()).lower()

    def many_kinded(self, node: str) -> bool:
        """Does this node carry two kinds that are unrelated to each other?

        It was written to detect a homonym, and it does not. That is worth the space, because the
        finding is more useful than the method.

        The problem is real and was found by measurement. Asked *how does the heart work*, the walk
        answered ``heart → atrium → bringing daylight into the middle of a deep plan``. Every edge
        in it is a fact she was correctly told: a heart has an atrium, and an atrium — the tall
        space in a building — is for bringing daylight into a deep plan. The store keys facts on
        their surface spelling, so the heart's chamber and the building's courtyard are one node,
        and a path crosses from one sense into the other. Fifteen thousand facts held that
        invisibly, because only a path can expose it: *what is an atrium* returns two kinds and
        looks like an ordinary many-valued relation.

        The test below — two ``is_a`` objects where neither is an ancestor of the other and they
        share no word — was then measured across the whole store. **261 of 4,230 subjects, 6.2%**,
        and reading them settles it: ``albert einstein`` is a *physicist* and a *nobel laureate*,
        ``aryabhata`` a *mathematician* and an *astronomer*, ``bat`` a *mammal* and a *nocturnal
        animal*. One thing under two descriptions is the ordinary case and this cannot tell it
        from ``atrium``. Genuine homonyms were a handful of the 261.

        What that cost, when the result was used as a confidence penalty, is the reason the
        penalty is gone: ``why does higher yield happen`` lost ``nitrogen fixation → soil
        fertility → higher yield`` — a correct two-hop derivation — because two nodes on it were
        many-kinded and 0.85² × 0.5² fell under the floor. The paper fell from 0.785 to 0.755
        while the number that was supposed to be improving stayed still.

        So it stays as a **tie-break and nothing more**: among chains of the same kind and length,
        one through a many-kinded node ranks below one that is not. That is enough to have fixed
        the heart — the atrium chain is still built, still returned if asked for, and no longer
        the first thing said — and it cannot remove a chain, which is what the measurement said
        it must not do.

        Separating the two atriums needs a store keyed on something other than spelling. That is a
        limit of this store, named here rather than tuned away.
        """
        kinds = [k for k, _ in self._out(node, KIND)]
        if len(kinds) < 2:
            return False
        for first, second in ((a, b) for i, a in enumerate(kinds) for b in kinds[i + 1:]):
            if self._related(first, second) or self._related(second, first):
                continue
            if set(str(first).lower().split()) & set(str(second).lower().split()):
                continue
            return True
        return False

    #: The name it was built under, kept because callers outside this module use it.
    ambiguous = many_kinded

    def _related(self, lower: str, upper: str, *, depth: int = 3) -> bool:
        """Is *upper* reachable from *lower* by ``is_a``? Bounded, and it is not a search."""
        if self._key(lower) == self._key(upper):
            return True
        if depth <= 0:
            return False
        return any(self._related(k, upper, depth=depth - 1) for k, _ in self._out(lower, KIND))

    def _flag(self, chain: Chain) -> Chain:
        """Mark every node on the chain whose spelling covers two things. The topic is exempt.

        Exempt because the topic is what was *asked about*: a question naming an ambiguous word
        is ambiguous in the asking, and answering it in both senses is the honest reply. What is
        being caught here is a chain that entered on one sense and left on another, and that can
        only happen at an interior node.
        """
        chain.ambiguous = [n for n in chain.nodes[1:] if self.many_kinded(n)]
        return chain

    def _out(self, subject: str, relation: str) -> List[Tuple[str, float]]:
        return list(self._forward.get((self._key(subject), relation), ()))

    def _in(self, obj: str, relation: str) -> List[Tuple[str, float]]:
        return list(self._backward.get((self._key(obj), relation), ()))

    # ---- why: mechanistic ------------------------------------------------ #
    def _incoming(self, topic: str) -> List[Tuple[str, str, float]]:
        """Everything that states *this was brought about by*, however it happens to be stored.

        Two shapes, both read as one relation. ``S causes T`` is on S and has to be found by the
        inverted index; ``T occurs_when C`` is on T and is read straight off. They mean the same
        thing about T and a walk that only knew the first found nothing for every process in the
        corpus, because a process states its trigger the second way.
        """
        out: List[Tuple[str, str, float]] = []
        for relation in PRODUCES:
            for cause, conf in self._in(topic, relation):
                out.append((cause, relation, conf))
        for condition, conf in self._out(topic, OCCURS):
            out.append((condition, OCCURS, conf))
        return out

    def _walk_back(self, topic: str, seen: Set[str], depth: int) -> List[List[Step]]:
        """Every backwards path from *topic*, up to :attr:`max_depth`.

        A node already on the path is not entered again — the corpus holds ``evaporation causes
        cloud``, ``cloud causes rain`` and ``rain causes runoff``, and a walk without that guard
        will happily explain rain by rain.
        """
        if depth <= 0:
            return []
        paths: List[List[Step]] = []
        for cause, relation, conf in self._incoming(topic):
            key = self._key(cause)
            if key in seen or key == self._key(topic):
                continue
            step = Step(subject=cause, relation=relation, object=topic,
                        confidence=conf, forward=False)
            paths.append([step])
            for tail in self._walk_back(cause, seen | {key}, depth - 1):
                paths.append([step] + tail)
        return paths

    # ---- why: teleological ----------------------------------------------- #
    def _walk_purpose(self, topic: str, seen: Set[str], depth: int) -> List[List[Step]]:
        """Forwards along ``purpose``, and up through ``is_a`` when the thing carries none.

        The climb is one hop and only when the subject itself has no purpose stated. A stethoscope
        has one; *this* stethoscope does not, and the answer is its kind's. Climbing further, or
        climbing when the thing has its own answer, is how a specific purpose gets replaced by a
        generic one — "a scalpel is for medicine".
        """
        if depth <= 0:
            return []
        direct: List[Tuple[str, float]] = []
        for relation in PURPOSE:
            direct.extend(self._out(topic, relation))
        borrowed = False
        if not direct:
            for kind, kconf in self._out(topic, KIND):
                for relation in PURPOSE:
                    for aim, conf in self._out(kind, relation):
                        direct.append((aim, conf * kconf))
                if direct:
                    borrowed = True
                    break
        paths: List[List[Step]] = []
        for aim, conf in direct:
            key = self._key(aim)
            if key in seen or key == self._key(topic):
                continue
            step = Step(subject=topic, relation="purpose", object=aim,
                        confidence=conf * (0.9 if borrowed else 1.0), forward=True)
            paths.append([step])
            for tail in self._walk_purpose(aim, seen | {key}, depth - 1):
                paths.append([step] + tail)
        return paths

    # ---- why: enabling conditions ---------------------------------------- #
    def _walk_needs(self, topic: str) -> List[List[Step]]:
        """One hop of ``requires``. Never chained, and never called a cause.

        Requirements do not compose the way causes do. Fire requires oxygen and oxygen requires —
        in this corpus — nothing at all, but where a requirement's requirement *is* stated,
        chaining it produces "a car needs a road, which needs asphalt", which is true of roads and
        says nothing about the car. One hop is what the relation supports.
        """
        out: List[List[Step]] = []
        for relation in ENABLES:
            for need, conf in self._out(topic, relation):
                out.append([Step(subject=topic, relation=relation, object=need,
                                 confidence=conf, forward=True)])
        return out

    # ---- the public why -------------------------------------------------- #
    def why(self, topic: str, *, sense: str = "any", depth: int = 0) -> Explanation:
        """Why *topic*. ``sense`` is ``because``, ``for``, ``needs`` or ``any``.

        ``any`` returns all three, causes first, and that ordering is the answer to the question
        *why does X happen* being asked of a thing that also has a purpose. Where a caller knows
        which was asked — the question grammar does — it should say so, because a *what is X for*
        answered with a cause is wrong rather than incomplete.
        """
        topic = " ".join(str(topic or "").split())
        out = Explanation(topic=topic, question=f"why {topic}")
        if not topic:
            out.why = "nothing was asked about"
            return out
        limit = depth if depth > 0 else self.max_depth
        seed = {self._key(topic)}
        built: List[Chain] = []
        if sense in ("any", "because"):
            for path in self._walk_back(topic, seed, limit):
                built.append(Chain(topic=topic, kind="because", steps=path))
        if sense in ("any", "for"):
            for path in self._walk_purpose(topic, seed, limit):
                built.append(Chain(topic=topic, kind="for", steps=path))
        if sense in ("any", "needs"):
            for path in self._walk_needs(topic):
                built.append(Chain(topic=topic, kind="needs", steps=path))
        built = [self._flag(c) for c in built]
        out.considered = len(built)
        kept = [c for c in built if c.support >= self.min_confidence]
        out.pruned = len(built) - len(kept)
        out.chains = self._rank(kept)
        if not out.chains:
            out.why = ("nothing in the store leads to this" if not built
                       else f"{out.pruned} chains, all below the floor")
        else:
            out.why = f"{len(out.chains)} of {out.considered} chains"
        return out

    def _rank(self, chains: Sequence[Chain]) -> List[Chain]:
        """Longest first inside a kind, causes before purposes before requirements.

        ``priority`` splits a kind that has two sources. Inside *mechanism*, a chain that starts
        at one of the thing's parts outranks one that only follows the thing's consequences
        forwards: *how does photosynthesis work* answered with ``photosynthesis → glucose → a rise
        in blood sugar`` is a true chain and not a mechanism, and it outranked the chlorophyll
        chain purely by being one hop longer.

        Longest-first is the decision worth arguing with. A one-hop chain is more confident and
        less of an answer: *why does rain happen* → *because cloud* is a fact she was told, and
        the question was for the part she was not. Confidence still ranks within a length, and the
        floor still removes the ones that got long by getting weak.
        """
        order = {"because": 0, "mechanism": 1, "for": 2, "needs": 3}
        ranked = sorted(chains, key=lambda c: (order.get(c.kind, 9), c.priority,
                                               (1 if (c.ambiguous and DEMOTE_MANY_KINDED)
                                                else 0), -c.depth,
                                               -c.confidence, c.gloss()))
        seen: Set[Tuple[str, ...]] = set()
        out: List[Chain] = []
        for chain in ranked:
            sig = tuple([chain.kind] + [self._key(n) for n in chain.nodes])
            if sig in seen:
                continue
            # A chain that is a prefix of one already kept adds nothing: `cooling → condensation →
            # cloud → rain` already contains `cloud → rain`, and printing both is padding.
            if any(sig[0] == k[0] and len(sig) < len(k) and k[:len(sig)] == sig for k in seen):
                continue
            seen.add(sig)
            out.append(chain)
            if len(out) >= self.max_chains:
                break
        return out

    # ---- how does X work ------------------------------------------------- #
    def mechanism(self, topic: str, *, depth: int = 0) -> Explanation:
        """How *topic* works: its parts, what each does, and what the whole is for.

        A mechanism is not the same walk as a why with the arrows reversed. *Why does a pump
        work* has no answer; *how does a pump work* does, and it is made of the pump's parts
        rather than of the pump's causes. So this crosses decomposition with production: for each
        part, what that part causes or is for, one hop each, and then the forward chain from the
        whole.
        """
        topic = " ".join(str(topic or "").split())
        out = Explanation(topic=topic, question=f"how does {topic} work")
        if not topic:
            out.why = "nothing was asked about"
            return out
        limit = depth if depth > 0 else self.max_depth
        built: List[Chain] = []
        for relation in PARTS:
            for part, pconf in self._out(topic, relation):
                head = Step(subject=topic, relation=relation, object=part,
                            confidence=pconf, forward=True)
                tails: List[List[Step]] = []
                for verb in PRODUCES + PURPOSE:
                    for effect, conf in self._out(part, verb):
                        if self._key(effect) == self._key(topic):
                            continue
                        tails.append([Step(subject=part, relation=verb, object=effect,
                                           confidence=conf, forward=True)])
                if tails:
                    for tail in tails:
                        built.append(Chain(topic=topic, kind="mechanism",
                                           steps=[head] + tail, priority=0))
                else:
                    built.append(Chain(topic=topic, kind="mechanism", steps=[head], priority=0))
        # And the whole thing's own forward chain — a process with no parts still works somehow.
        for path in self._walk_forward(topic, {self._key(topic)}, limit):
            built.append(Chain(topic=topic, kind="mechanism", steps=path, priority=1))
        built = [self._flag(c) for c in built]
        out.considered = len(built)
        kept = [c for c in built if c.support >= self.min_confidence]
        out.pruned = len(built) - len(kept)
        out.chains = self._rank(kept)
        out.why = (f"{len(out.chains)} of {out.considered} chains" if out.chains
                   else "no parts and no effects stored")
        return out

    def _walk_forward(self, topic: str, seen: Set[str], depth: int) -> List[List[Step]]:
        """Forwards along production. What happens next, and next after that."""
        if depth <= 0:
            return []
        paths: List[List[Step]] = []
        for relation in PRODUCES:
            for effect, conf in self._out(topic, relation):
                key = self._key(effect)
                if key in seen:
                    continue
                step = Step(subject=topic, relation=relation, object=effect,
                            confidence=conf, forward=True)
                paths.append([step])
                for tail in self._walk_forward(effect, seen | {key}, depth - 1):
                    paths.append([step] + tail)
        return paths

    # ---- how do you X ---------------------------------------------------- #
    def procedure(self, topic: str) -> Plan:
        """How to *topic*: the steps she was told, in an order she was not.

        The corpus states ``has_step`` and, on the steps, ``requires``. It never states a
        sequence. Everything below derives one — or says how many there are, or says there is
        none — and the same code runs on a procedure minted out of nonsense a moment ago.
        """
        topic = " ".join(str(topic or "").split())
        plan = Plan(topic=topic)
        if not topic:
            plan.why = "nothing was asked about"
            return plan
        steps: List[str] = []
        for relation in STEPS:
            for step, _conf in self._out(topic, relation):
                if step not in steps:
                    steps.append(step)
        if not steps:
            plan.why = "no steps stored for this"
            return plan
        # Sort the step list before anything reads it, so the file's line order cannot leak into
        # the answer. If the sequence were coming from the telling rather than from the
        # prerequisites, this line alone would break every procedure — and a test asserts it does
        # not, by shuffling the steps and demanding the same orders back.
        steps = sorted(steps, key=self._key)
        index = {self._key(s): s for s in steps}
        needs: Dict[str, List[str]] = {}
        dangling: List[str] = []
        for step in steps:
            got: List[str] = []
            for relation in ENABLES:
                for need, _conf in self._out(step, relation):
                    key = self._key(need)
                    if key == self._key(step):
                        continue
                    if key in index:
                        if index[key] not in got:
                            got.append(index[key])
                    elif need not in dangling:
                        # Named as a prerequisite, never named as a step. Not silently satisfied:
                        # a procedure that needs a thing it never tells you to get is a hole in
                        # the telling, and the plan says so.
                        dangling.append(need)
            needs[step] = sorted(got, key=self._key)
        plan.steps = steps
        plan.needs = needs
        plan.dangling = sorted(dangling, key=self._key)
        orders, count, cycle = self._orders(steps, needs)
        plan.orders, plan.order_count, plan.cycle = orders, count, cycle
        if cycle:
            plan.why = "the prerequisites are circular: " + " → ".join(cycle)
        elif count == 1:
            plan.why = "the prerequisites determine one order"
            # spellcheck: this is the only branch that may say "the" order.
        else:
            plan.why = f"the prerequisites allow {count} orders"
        return plan

    def _orders(self, steps: Sequence[str],
                needs: Dict[str, List[str]]) -> Tuple[List[List[str]], int, List[str]]:
        """Every topological order of the steps, the count of them, and any cycle.

        Enumerated rather than counted-then-sampled, because the count is what says whether the
        telling determined the sequence and the list is what a caller reads out. The enumeration
        stops at :data:`MAX_ORDERS` sequences but the **count keeps going** to a hard ceiling —
        capping both would report six steps with no constraints as "12 orders", which is a
        different and false claim about how loosely they were specified.
        """
        remaining = list(steps)
        outstanding = {s: set(self._key(n) for n in needs.get(s, ())) for s in steps}
        done: Set[str] = set()
        # Kahn to exhaustion first, purely to find a cycle. A cycle makes enumeration infinite in
        # the sense that matters — there is no order — so it is detected before anything is built.
        progress = True
        while progress:
            progress = False
            for step in list(remaining):
                if outstanding[step] <= done:
                    remaining.remove(step)
                    done.add(self._key(step))
                    progress = True
        if remaining:
            return [], 0, sorted(remaining, key=self._key)

        found: List[List[str]] = []
        count = 0
        ceiling = 200_000

        def walk(placed: List[str], placed_keys: Set[str]) -> None:
            nonlocal count
            if count >= ceiling:
                return
            if len(placed) == len(steps):
                count += 1
                if len(found) < MAX_ORDERS:
                    found.append(list(placed))
                return
            for step in steps:
                if step in placed:
                    continue
                if not outstanding[step] <= placed_keys:
                    continue
                placed.append(step)
                walk(placed, placed_keys | {self._key(step)})
                placed.pop()

        walk([], set())
        return found, count, []

    # ---- the one entry point --------------------------------------------- #
    def ask(self, question: str) -> Any:
        """Read a why/how question and run the walk it names.

        Returns an :class:`Explanation` or a :class:`Plan`. Returns ``None`` when the text is not
        one of these questions at all, so a caller can fall through to the ordinary answer path
        rather than having a why-walk answer *what is a mammal*.

        The candidate loop is where the grammar and the store are reconciled. The reader offers
        several spellings of the topic, most specific first, and this takes **the first that the
        graph actually reaches** rather than the first that parses. A topic that reaches nothing
        is not an answer, so trying the next spelling costs nothing and skipping it cost every
        *why does water boil* in the cold run.
        """
        from nyxara.njp.explainread import read_explanation_question

        got = read_explanation_question(question)
        if got is None:
            return None
        kind, topics = got
        last: Any = None
        for topic in topics:
            if kind == "procedure":
                out = self.procedure(topic)
            elif kind == "mechanism":
                out = self.mechanism(topic)
            else:
                out = self.why(topic, sense=kind)
            if getattr(out, "answered", False):
                return out
            if last is None:
                last = out
        return last

    def to_dict(self) -> Dict[str, Any]:
        return {"edges": self.edges, "max_depth": self.max_depth,
                "min_confidence": self.min_confidence, "max_chains": self.max_chains}
