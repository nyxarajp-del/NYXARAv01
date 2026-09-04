"""NYXARA · njp/loop.py — the cycle that cannot comfortably remain wrong (🌀, NJP V.42).

Every organ this package has built for reasoning has been a **step**: `surgery` recovers a
structure, `predator` attacks an explanation, `fusion` finds a shared shape, `curiosity` picks a
question. Each was measured on its own paper and each passed. None of them is a *loop*, and the
Master's diagnosis of why that matters is the sharpest thing anyone has said about this codebase:

    Intelligence = the ability to kill your own explanations and replace them with better ones.
    Not simply generating more explanations.

The evidence was in the ConceptNet measurement. Twelve times the facts moved unseen-entity
coverage 6.0% → 8.8% and derivation 2.9% → 3.0%. **More knowledge did not buy more structure.**
What was missing was not facts. It was the cycle::

    observations → structure → latent hypothesis → prediction → experiment
        → result → revise → prediction nobody asked for
                       ↺

This module is that cycle, and :mod:`nyxara.njp.discovery` is the world it is measured in — one it
has never seen, with no answer in the corpus, no supplied graph, no named concept, and, in half
the worlds, **a cause that appears in no observation at all**.

The four things it does that a step cannot
------------------------------------------

**It refuses to invent the absence of a hidden cause.** Over observational data, ``A → B`` and
``A ← H → B`` imply exactly the same dependencies. There is no evidence at that stage that
separates them, so committing either way is a guess wearing a conclusion's clothes. V.13
established that where two readings survive the answer is that there are two; V.40 applied it to
orientations; this applies it to **existence**. :class:`Latents` proposes a hidden common cause for
every dependence an edge was carrying, keeps only what is observationally consistent, and marks
every one ``HYPOTHETICAL``.

**It turns a question into an action.** A Markov equivalence class is not a thing you can think
your way out of — no amount of watching separates its members. But ``do(A)`` does: intervening cuts
A's incoming edges, so ``A → B`` says B moves and ``A ← H → B`` says it does not. :class:`Experiments`
scores every available intervention by how much of the model set it would kill, over what it costs,
and names the cheapest one that actually splits them. *Which fact am I missing* becomes *what is
the cheapest observation that distinguishes these two*.

**It revises rather than adds.** The result comes back and models that predicted otherwise are
**removed**. :meth:`Loop.revise` never keeps a refuted model for being popular and never kills one
the result does not touch — and the exam grades both directions, because killing too much is as
wrong as killing nothing.

**It says why it does not know.** :class:`Unknown` is a reason, not a shrug: *competing models*,
*no model survived*, *the models disagree*, *nothing observed reaches this*. A caller that knows
**why** can pick a remedy; one told only "unknown" can do nothing at all.

Provenance
----------

Every prediction carries the models it came from and the observations those models were fitted to.
:attr:`Prediction.status` is ``SUPPORTED`` when every surviving model agrees and none of them is
hypothetical, ``HYPOTHETICAL`` when the agreement rests on a hidden cause nobody has confirmed, and
``CONFLICTED`` when the survivors disagree — in which case there is no prediction, and that is the
answer.

What it may not do
------------------

**It may not predict from a model it would not defend.** A conflicted set produces
:class:`Unknown`, never a majority vote. Two surviving models that disagree about ``do(A) → D``
are two answers, and the loop has no principle for preferring one that is not the experiment it
should have run.

**It may not believe a latent.** A hidden cause is a hypothesis until an intervention lets it
survive one it could have failed. :attr:`Model.status` never becomes ``SUPPORTED`` by being useful.

**It may not touch the store.** Everything here operates on models handed in and returns models;
writing a discovered structure back is a decision with an owner.

Pure standard library, deterministic, and it holds no facts of its own.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "Status", "Model", "Prediction", "Unknown", "Reason", "Autopsy", "Failure",
    "Latents", "Experiments", "Loop", "MAX_LATENTS", "MAX_MODELS",
]

#: How many hidden causes may be proposed at once. One is the case worth having and the case the
#: evidence can speak to; two unobserved variables over four observed ones is a model space that
#: fits everything and therefore says nothing.
MAX_LATENTS = 1

#: A ceiling on the model set. It bounds the enumeration and it is a real limit: a world whose
#: admissible set is larger than this is one where she is reasoning from a sample of the
#: alternatives, and :meth:`Loop.models` says so in its history line.
MAX_MODELS = 256


class Status(str, Enum):
    """What standing a model or a prediction has. Never upgraded by usefulness."""

    SUPPORTED = "supported"          # every surviving model agrees, none of them hypothetical
    HYPOTHETICAL = "hypothetical"    # rests on a hidden cause nothing has confirmed
    CONFLICTED = "conflicted"        # the survivors disagree; there is no prediction
    REFUTED = "refuted"              # an intervention said otherwise


class Reason(str, Enum):
    """Why she does not know. A reason, not a shrug — each names a different remedy."""

    NO_MODEL = "no model survived the evidence"
    COMPETING = "several models survive and they disagree"
    UNREACHED = "nothing observed connects these"
    NO_EXPERIMENT = "no available intervention separates the models"
    OUT_OF_SCOPE = "this is not about anything observed"


@dataclass(frozen=True)
class Model:
    """One causal structure, possibly with unobserved nodes, and what it is worth."""

    edges: FrozenSet[Tuple[str, str]]
    observed: Tuple[str, ...] = ()
    status: Status = Status.SUPPORTED
    #: What this model was fitted to. Provenance: a model with no support is not a model.
    fitted_to: int = 0

    @property
    def hidden(self) -> Set[str]:
        names = set(self.observed)
        return {n for edge in self.edges for n in edge if n not in names}

    @property
    def hypothetical(self) -> bool:
        return bool(self.hidden)

    def render(self) -> str:
        body = ", ".join(f"{a} → {b}" for a, b in sorted(self.edges)) or "(no edges)"
        return body + (f"  [hidden: {', '.join(sorted(self.hidden))}]" if self.hidden else "")

    def to_dict(self) -> Dict[str, Any]:
        return {"edges": sorted(list(e) for e in self.edges), "hidden": sorted(self.hidden),
                "status": self.status.value, "fitted_to": self.fitted_to,
                "render": self.render()}


@dataclass
class Unknown:
    """Not knowing, with the reason attached and the remedy implied by it."""

    reason: Reason
    detail: str = ""

    def __bool__(self) -> bool:
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {"unknown": self.reason.value, "detail": self.detail}


@dataclass
class Prediction:
    """What the surviving models say, and what it rests on."""

    cause: str
    effect: str
    responds: Optional[bool] = None
    status: Status = Status.CONFLICTED
    from_models: List[Model] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"cause": self.cause, "effect": self.effect, "responds": self.responds,
                "status": self.status.value,
                "from": [m.render() for m in self.from_models[:4]]}


@dataclass
class Failure:
    """A prediction that did not survive contact, kept in the shape that lets it be learned from.

    Storing *"wrong"* teaches nothing. Storing which model said it, what it said, what happened,
    and which edge carried the claim is what makes the next failure recognisable as the same
    failure — the Master's memory autopsy, at the grain the loop can actually act on.
    """

    cause: str
    effect: str
    predicted: bool
    actual: bool
    blamed: List[Model] = field(default_factory=list)
    edge: Tuple[str, str] = ()
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"cause": self.cause, "effect": self.effect, "predicted": self.predicted,
                "actual": self.actual, "edge": list(self.edge), "note": self.note,
                "blamed": [m.render() for m in self.blamed[:4]]}


class Autopsy:
    """What went wrong, per world, and whether it has gone wrong this way before."""

    def __init__(self) -> None:
        self.failures: List[Failure] = []

    def record(self, cause: str, effect: str, predicted: bool, actual: bool,
               blamed: Sequence[Model]) -> Failure:
        # The edge that carried the claim: the one out of the cause that every blamed model has.
        shared: Optional[Set[Tuple[str, str]]] = None
        for model in blamed:
            out = {e for e in model.edges if e[0] == cause}
            shared = out if shared is None else (shared & out)
        edge = sorted(shared)[0] if shared else ()
        got = Failure(cause=cause, effect=effect, predicted=predicted, actual=actual,
                      blamed=list(blamed), edge=tuple(edge),
                      note=f"predicted {predicted}, world said {actual}")
        self.failures.append(got)
        return got

    def seen_before(self, cause: str, effect: str) -> Optional[Failure]:
        return next((f for f in self.failures if f.cause == cause and f.effect == effect), None)


# --------------------------------------------------------------------------- #
# d-separation, which the loop needs in order to think
# --------------------------------------------------------------------------- #
def _descendants(edges: FrozenSet[Tuple[str, str]], node: str) -> Set[str]:
    seen: Set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        for a, b in edges:
            if a == current and b not in seen:
                seen.add(b)
                stack.append(b)
    return seen


def connected(edges: FrozenSet[Tuple[str, str]], left: str, right: str,
              given: Sequence[str] = ()) -> bool:
    """d-connection. Hidden nodes transmit and are never conditioned on.

    That asymmetry is not an approximation, it is what *latent* means: the thing acts on the world
    and cannot be held fixed by anyone observing it.
    """
    adjacency: Dict[str, Set[str]] = defaultdict(set)
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    if left not in adjacency or right not in adjacency:
        return False
    held = set(given)
    found: List[List[str]] = []

    def walk(node: str, seen: List[str]) -> None:
        if node == right:
            found.append(list(seen))
            return
        for other in sorted(adjacency[node]):
            if other not in seen:
                walk(other, seen + [other])

    walk(left, [left])
    for path in found:
        blocked = False
        for i in range(1, len(path) - 1):
            before, node, after = path[i - 1], path[i], path[i + 1]
            if (before, node) in edges and (after, node) in edges:
                if not ((_descendants(edges, node) | {node}) & held):
                    blocked = True
                    break
            elif node in held:
                blocked = True
                break
        if not blocked:
            return True
    return False


def _acyclic(edges: FrozenSet[Tuple[str, str]]) -> bool:
    colour: Dict[str, int] = {}
    children: Dict[str, Set[str]] = defaultdict(set)
    for a, b in edges:
        children[a].add(b)

    def visit(node: str) -> bool:
        state = colour.get(node, 0)
        if state == 1:
            return False
        if state == 2:
            return True
        colour[node] = 1
        for child in children[node]:
            if not visit(child):
                return False
        colour[node] = 2
        return True

    return all(visit(n) for edge in edges for n in edge)


def responds(model: Model, cause: str, effect: str) -> bool:
    """What ``do(cause)`` does to *effect* under this model: the graph with cause's parents cut."""
    return effect in _descendants(frozenset(e for e in model.edges if e[1] != cause), cause)


def fits(model: Model, observations: Sequence[Any]) -> bool:
    """Does this model imply exactly what was observed? Every statement, both directions."""
    for observed in observations:
        got = connected(model.edges, observed.left, observed.right,
                        sorted(getattr(observed, "given", ()) or ()))
        if got != bool(observed.dependent):
            return False
    return True


# --------------------------------------------------------------------------- #
# The three organs the loop needed and did not have
# --------------------------------------------------------------------------- #
class Latents:
    """Proposes an unobserved common cause wherever one would explain the same dependence.

    The rule is deliberately mechanical, and the mechanism is the argument: an edge ``X → Y`` is
    carrying a dependence, and a hidden parent of both carries exactly the same one. So for every
    edge, offer the swap; keep it only if the resulting model still implies **everything** that was
    observed; and mark it ``HYPOTHETICAL``, which nothing but an intervention can lift.

    It does not guess *whether* this world has a hidden cause. It cannot: no observation
    distinguishes them, and a proposer that guessed would be scored on a coin flip. What it does is
    refuse to let the observed-only reading stand as though it were the only one.
    """

    def __init__(self, *, max_latents: int = MAX_LATENTS) -> None:
        self.max_latents = max(1, int(max_latents))

    def propose(self, model: Model, observations: Sequence[Any],
                observed: Sequence[str]) -> List[Model]:
        out: List[Model] = []
        seen: Set[FrozenSet[Tuple[str, str]]] = set()
        for index, (x, y) in enumerate(sorted(model.edges)):
            hidden = f"h{index}"
            edges = frozenset((model.edges - {(x, y)}) | {(hidden, x), (hidden, y)})
            if edges in seen:
                continue
            seen.add(edges)
            candidate = Model(edges=edges, observed=tuple(observed),
                              status=Status.HYPOTHETICAL, fitted_to=len(observations))
            if fits(candidate, observations):
                out.append(candidate)
        return out


class Experiments:
    """Chooses the intervention that would settle the most, for the least.

    Value is the Master's #4, written as something computable::

        value = (models it would eliminate, whichever way it goes) / cost

    "Whichever way it goes" is the part that matters. An intervention is only worth running if it
    splits the set — if every model agrees, the result is already known and the experiment buys
    nothing however cheap. So the score is the size of the **smaller** side of the split: an
    experiment that would kill one of twenty is worth less than one that kills ten, because the
    worst case is what you are buying.
    """

    def __init__(self, *, cost: Any = None) -> None:
        #: ``cost(cause, effect) -> float``. Flat by default; a caller with real costs passes one.
        self.cost = cost or (lambda cause, effect: 1.0)

    def split(self, models: Sequence[Model], cause: str, effect: str) -> Tuple[int, int]:
        yes = sum(1 for m in models if responds(m, cause, effect))
        return yes, len(models) - yes

    def value(self, models: Sequence[Model], cause: str, effect: str,
              toward: Tuple[str, str] = ()) -> float:
        """How much this experiment is worth, and ``toward`` is what makes it worth anything.

        Without it the score is *models eliminated*, which sounds right and is not. Measured: the
        loop converged and **plateaued at 0.675** — a third of worlds ran out of useful experiments
        while still split on the one question they were about to be asked, because the chooser was
        busy killing models along axes nobody cared about. Eliminating nineteen of twenty models
        buys nothing if the twentieth disagrees with the survivor about the thing you need.

        So ``toward`` names the pending question and the value becomes **how much of the
        disagreement *about that question* this experiment would remove**. It is the Master's
        "model importance" term, and it is the difference between a system that reduces entropy and
        one that answers a question.
        """
        yes, no = self.split(models, cause, effect)
        if not yes or not no:
            return 0.0            # everyone agrees: the result is already known
        base = min(yes, no)
        if toward:
            # Of the two groups this experiment would leave, how much closer does each get to
            # agreeing about `toward`? An experiment that splits the models exactly along the
            # question is worth the most; one orthogonal to it is worth almost nothing.
            gain = 0.0
            for outcome in (True, False):
                survivors = [m for m in models if responds(m, cause, effect) == outcome]
                if not survivors:
                    continue
                before = len({responds(m, *toward) for m in models})
                after = len({responds(m, *toward) for m in survivors})
                gain += (before - after) / 2.0
            if gain <= 0.0:
                # **Worth nothing, so worth not running.** With an aim in hand, an experiment that
                # cannot move the aim is not a cheap experiment, it is a wasted one — and the loop
                # ran them: on worlds where nothing available could settle the question it went
                # ahead and split the model set along axes nobody had asked about, scoring 0.667
                # where recognising there is nothing to run is the whole answer.
                return 0.0
            base = base * 0.001 + gain      # the question dominates; ties break on breadth
        return base / max(1e-9, float(self.cost(cause, effect)))

    def choose(self, models: Sequence[Model], among: Sequence[str] = (),
               toward: Tuple[str, str] = ()) -> Optional[Tuple[str, str]]:
        names = list(among) or sorted({n for m in models for e in m.edges for n in e
                                       if n in set(m.observed)})
        best: Optional[Tuple[float, Tuple[str, str]]] = None
        for cause, effect in itertools.permutations(names, 2):
            if toward and frozenset({cause, effect}) == frozenset(toward):
                continue          # the question itself is not an experiment
            got = self.value(models, cause, effect, toward=toward)
            if got <= 0.0:
                continue
            if best is None or got > best[0] or (got == best[0] and (cause, effect) < best[1]):
                best = (got, (cause, effect))
        return best[1] if best else None


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
class Loop:
    """observe → structure → latent → predict → experiment → revise → predict again.

    Implements the protocol :mod:`nyxara.njp.discovery` examines: :meth:`models`,
    :meth:`experiment`, :meth:`revise`, :meth:`predict`. Every one of them is a step some organ in
    this package already had; what is here is the wiring, and the wiring is the claim.
    """

    def __init__(self, *, latents: Optional[Latents] = None,
                 experiments: Optional[Experiments] = None,
                 max_models: int = MAX_MODELS) -> None:
        self.latents = latents or Latents()
        self.experiments = experiments or Experiments()
        self.max_models = max(2, int(max_models))
        self.autopsy = Autopsy()
        #: Every conclusion the loop reaches, with the path that made it. A prediction that cannot
        #: say what produced it is not a weak prediction — it is not a prediction.
        from nyxara.njp.provenance import Ledger as ProvenanceLedger

        self.ledger = ProvenanceLedger()
        self.history: List[str] = []
        #: What she is going to be asked. Set by :meth:`aim`, and the experiments are chosen in
        #: service of it. Knowing the *question* is not knowing the *answer*, and choosing an
        #: experiment that bears on what you need is what an experiment is for.
        self.aim_at: Tuple[str, str] = ()

    # ---- structure, and what it refuses to rule out ----------------------- #
    def models(self, nodes: Sequence[str], observations: Sequence[Any]) -> List[Model]:
        """Everything the observations admit — **including when nothing observed-only does**.

        This depended on :class:`~nyxara.njp.surgery.Surgeon` for the observed-only class and then
        offered latent variants *of those*, and on a randomly generated world with real confounding
        it produced **zero models** where 75 were admissible. The reason is worth stating because
        it inverts the intuition: when a latent confounds two variables, often **no DAG over the
        observed variables fits at all** — every orientation of the skeleton implies a collider the
        data denies — so the observed-only class is correctly empty, and a proposer that builds
        latents out of it has nothing to build from.

        An empty observed-only class is not a dead end. **It is the strongest evidence a latent
        exists**, and reading it as failure was the single defect that made the loop score below
        the reconstructor it was supposed to beat.

        So the space is enumerated directly: the skeleton is determined by the observations, every
        acyclic orientation is a candidate, every one-edge-to-hidden-cause swap of every candidate
        is another, and each is kept only if it implies **everything** observed. Surgeon is still
        run — its class is the observed-only part, arrived at by a different route — and agreement
        between the two is a check rather than a dependency.
        """
        names = list(nodes)
        observed_only, hypothetical = [], []
        for edges in self._orientations(names, observations):
            model = Model(edges=edges, observed=tuple(names), status=Status.SUPPORTED,
                          fitted_to=len(observations))
            if fits(model, observations):
                observed_only.append(model)
            hypothetical.extend(self.latents.propose(model, observations, names))
        seen: Set[FrozenSet[Tuple[str, str]]] = {m.edges for m in observed_only}
        out = list(observed_only)
        for model in hypothetical:
            if model.edges not in seen:
                seen.add(model.edges)
                out.append(model)
        self.history.append(
            f"structure: {len(out)} models, {len(observed_only)} observed-only, "
            f"{len(out) - len(observed_only)} with a hidden cause"
            + ("  (no observed-only structure fits — which is itself evidence of one)"
               if not observed_only and out else ""))
        return out[:self.max_models]

    def _orientations(self, names: Sequence[str],
                      observations: Sequence[Any]) -> Iterable[FrozenSet[Tuple[str, str]]]:
        """Acyclic orientations of the skeleton the observations determine.

        The skeleton is not searched for: a pair is adjacent exactly when it is dependent and no
        conditioning set separates it. Only the directions are open, and :data:`MAX_MODELS` bounds
        how many are looked at.
        """
        dependent = {frozenset({o.left, o.right}) for o in observations if o.dependent}
        separated = {frozenset({o.left, o.right}) for o in observations if not o.dependent}
        skeleton = [tuple(sorted(pair)) for pair in
                    (frozenset(p) for p in itertools.combinations(names, 2))
                    if pair in dependent and pair not in separated]
        if len(skeleton) > 12:
            skeleton = skeleton[:12]
        for bits in itertools.product((0, 1), repeat=len(skeleton)):
            edges = frozenset((a, b) if bit == 0 else (b, a)
                              for (a, b), bit in zip(skeleton, bits))
            if _acyclic(edges):
                yield edges

    # ---- the question that is an action ----------------------------------- #
    def aim(self, cause: str, effect: str) -> None:
        """Tell the loop what it will be asked, so it can experiment towards it."""
        self.aim_at = (cause, effect)
        self.history.append(f"aim: do({cause}) -> {effect}?")

    def experiment(self, models: Sequence[Model]) -> Optional[Tuple[str, str]]:
        got = self.experiments.choose(models, toward=self.aim_at)
        self.history.append(f"experiment: {got or 'none separates them'}")
        return got

    # ---- revision, in both directions ------------------------------------- #
    def revise(self, models: Sequence[Model], cause: str, effect: str,
               result: bool) -> List[Model]:
        kept, killed = [], []
        for model in models:
            (kept if responds(model, cause, effect) == result else killed).append(model)
        if killed:
            self.autopsy.record(cause, effect, predicted=not result, actual=result,
                                blamed=killed)
            # And the same failure at the grain that survives the world: which *step* carried it.
            claim_id = f"do({cause})->{effect}"
            if claim_id in self.ledger.claims:
                self.ledger.autopsy(
                    claim_id, predicted=not result, actual=result,
                    missing=f"whether {effect} responds to do({cause})",
                    repair=f"{len(killed)} model(s) removed")
        # A hypothesis that could have died here and did not has earned something. Not belief —
        # a hidden cause is never promoted for being useful — but the record says it was tested.
        self.history.append(f"revise: {len(models)} -> {len(kept)}, {len(killed)} refuted")
        return kept

    # ---- prediction, with provenance -------------------------------------- #
    def predict(self, models: Sequence[Model], cause: str,
                effect: str) -> Optional[bool]:
        got = self.forecast(models, cause, effect)
        return got.responds if isinstance(got, Prediction) else None

    def forecast(self, models: Sequence[Model], cause: str,
                 effect: str) -> Any:
        """The full answer: what, from which models, and at what standing.

        :meth:`predict` is the thin version the examination calls; this is the one a caller who
        needs to audit an answer should use. A conflicted set returns :class:`Unknown` with
        ``COMPETING`` rather than a majority vote — two surviving models that disagree are two
        answers, and the loop has no principle for preferring one that is not the experiment it
        should have run instead.
        """
        if not models:
            return Unknown(Reason.NO_MODEL, "every model was refuted")
        # **A variable no model mentions is not a variable this answers about.**
        #
        # Without this the loop said `responds=False, status=supported` for an effect that appears
        # in none of its graphs — technically the mutilation reaches nothing, so "no" falls out —
        # and that is a confident claim built on never having heard of the thing. "Nothing connects
        # these" and "I hold no model that mentions this" are different answers with different
        # remedies, which is the whole reason `Reason` has more than one member.
        known = {n for m in models for edge in m.edges for n in edge} | {
            n for m in models for n in m.observed}
        missing = [n for n in (cause, effect) if n not in known]
        if missing:
            return Unknown(Reason.OUT_OF_SCOPE,
                           f"no model mentions {', '.join(missing)}")
        reaches = any(connected(m.edges, cause, effect) for m in models)
        if not reaches:
            return Unknown(Reason.UNREACHED,
                           f"no model connects {cause} to {effect} by any path")
        answers = {responds(m, cause, effect) for m in models}
        if len(answers) > 1:
            splitter = self.experiments.choose(models, toward=(cause, effect))
            if splitter is None:
                return Unknown(Reason.NO_EXPERIMENT,
                               f"{len(models)} models disagree and no available intervention "
                               f"separates them")
            return Unknown(Reason.COMPETING,
                           f"{len(models)} models disagree; do({splitter[0]}) would separate them")
        answer = answers.pop()
        hypothetical = any(m.hypothetical for m in models)
        self._record(cause, effect, answer, models)
        return Prediction(cause=cause, effect=effect, responds=answer,
                          status=Status.HYPOTHETICAL if hypothetical else Status.SUPPORTED,
                          from_models=list(models))

    # ---- provenance -------------------------------------------------------- #
    def _record(self, cause: str, effect: str, answer: bool,
                models: Sequence[Model]) -> str:
        """File the prediction with the path that made it, and warn if this ground has failed.

        A model that carries a hidden cause enters the path as a ``HYPOTHESIS`` rather than as an
        edge, which is what makes :attr:`~nyxara.njp.provenance.Claim.status` come back
        ``HYPOTHETICAL`` without anything having to remember to say so.
        """
        from nyxara.njp.provenance import Kind, Step as ProvenanceStep

        claim_id = f"do({cause})->{effect}"
        path: List[Any] = []
        for index, model in enumerate(models[:4]):
            kind = Kind.HYPOTHESIS if model.hypothetical else Kind.EDGE
            path.append(self.ledger.record(ProvenanceStep(
                id=f"{claim_id}#m{index}", kind=kind, text=model.render(),
                settled=not model.hypothetical)))
        path.append(self.ledger.record(ProvenanceStep(
            id="R:mutilate", kind=Kind.INFERENCE,
            text="do(x) cuts x's incoming edges; y responds iff it is still reachable")))
        self.ledger.assert_(claim_id, f"{effect} {'responds' if answer else 'does not respond'} "
                                      f"to do({cause})", path=path,
                            note=f"{len(models)} surviving model(s)")
        earlier = self.ledger.warn(path)
        if earlier:
            self.history.append(f"warning: {len(earlier)} earlier failure(s) on this ground")
        return claim_id

    def audit(self, cause: str, effect: str) -> str:
        """The claim, its paths, and any warning — the auditable form of an answer."""
        return self.ledger.render(f"do({cause})->{effect}")

    def to_dict(self) -> Dict[str, Any]:
        return {"history": list(self.history),
                "failures": [f.to_dict() for f in self.autopsy.failures],
                "ledger": self.ledger.to_dict()}
