"""NYXARA · njp/discovery.py — a world she has never seen, and only observations (🌑, NJP V.42).

Every benchmark in this package so far, the gauntlet included, hands her something. `general.py`
hands her a corpus. `explaingauntlet.py` mints a world and then **states its edges** — the causal
graph is in the store and the question is whether she can walk it. Even ``surgery`` hands her the
observations *derived from a graph the exam already knows*, and grades her on getting that graph
back.

That is **reconstruction**. This file is about the other thing.

    No answer in the corpus. No predefined causal graph. No supplied abstraction. No named concept
    telling her what to discover. Only observations.

    Then: observations → structure → theory → prediction → failure → revision → correct unseen
    prediction.

The one thing that separates discovery from reconstruction
-----------------------------------------------------------

**A variable nobody mentioned.** Every world here may contain a *latent* — a cause that appears in
no observation, has no name, and is not in the list of things she is told about. Over the observed
variables alone, ``A → B`` and ``A ← H → B`` imply **exactly the same** dependencies: there is no
observation that separates them. A system that reports ``A → B`` has not been careful, it has been
lucky, and it will be wrong the moment somebody acts on A.

So the five stages below are staged on purpose, and each is scored apart, because each is a
different claim:

============  ======================================================================================
Stage         What passing means
============  ======================================================================================
``structure`` The equivalence class over the **observed** variables. This is what
              :mod:`nyxara.njp.surgery` already does, and it is here as the **control**: a system
              that fails this fails the rest for a reason that has nothing to do with discovery.
``latent``    Does she hold **both** kinds — at least one observed-only model and at least one
              with an unobserved cause — where no observation separates them? Refusing to consider
              a latent is inventing its absence, exactly as picking one orientation out of a Markov
              class invents a direction, and this package has refused that since V.13. Holding
              *only* latent models is the same error mirrored. Graded on holding both, never on
              guessing which world this is: **no observation can tell**, and a paper that rewarded
              guessing would reward luck.
``experiment``Two models that no observation separates. Name an **intervention** whose outcome
              differs between them. This is the Master's *"the question is an action"*: reasoning
              becomes experiment selection, and a system that cannot do it is stuck with the
              equivalence class forever however long it thinks. The world grants up to
              :data:`BUDGET` of them per task and **refuses the held-out pair**, so stage five
              cannot be bought by simply asking it.
``revise``    Given each result, eliminate what it refutes — and **keep what it does not**. Killing
              too much is as wrong as killing nothing, and the paper scores both directions, on
              every round rather than on the first.
``predict``   A **held-out** prediction: the effect of an intervention on a pair that appeared in
              none of the four stages above. This is the only stage that cannot be passed by
              bookkeeping — it asks whether what survived is a theory or a filing cabinet.

              And a third of the worlds are built so that **declining is the right answer**: the
              only intervention that would settle the question is the question itself, which the
              world refuses. Without those, the paper rewards answering and a system that always
              predicts goes uncaught — the same one-sidedness `explaingauntlet` was built to avoid
              with its restraint column.
============  ======================================================================================

Why the intervention is the hinge
---------------------------------

``A → B`` and ``A ← H → B`` are observationally identical and **behave differently under action**.
Intervene on A — set it, rather than watch it — and the graph loses its incoming edges. In the
first world B responds. In the second it does not, because the thing that moved B was H, and H is
untouched. That is the whole of causal discovery's claim to be about the world rather than about
correlation, and it is the cheapest possible test of whether a system has a *model* or a *summary*.

The exam's ground truth is d-separation on the true graph and reachability on the mutilated one.
Both are computed here, both are exact, and neither is the algorithm under test.

What a passing system may not do
--------------------------------

**It may not see the latent.** :meth:`Task.observations` emits statements about observed variables
only. The latent's name is minted and never leaves this module's ground truth.

**It may not see the answer to stage five.** The held-out pair is chosen after the first four
stages are built and is excluded from every observation and every experiment offered.

**It may not pass by answering everything.** ``latent`` scores a system that reports a hidden cause
in *every* world exactly as badly as one that never reports one: the paper mints worlds with and
without, and :attr:`Paper.precision` is reported beside the recall.

Pure standard library, deterministic per seed.
"""

from __future__ import annotations

import itertools
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

__all__ = [
    "Truth", "Observation", "Task", "Stage", "Paper", "Report", "Discovery",
    "STAGES", "DEFAULT_SEED", "DEFAULT_LIMIT", "run", "render", "main",
]

DEFAULT_SEED = 20260907
DEFAULT_LIMIT = 40

#: Interventions granted per world. Three, because the loop turning **once** is not a loop: with a
#: single experiment the model set was still split when the held-out question arrived, and
#: `predict` scored 0.000 against four stages at 1.000 — a cycle that starts and does not close.
BUDGET = 3

STAGES: Tuple[str, ...] = ("structure", "latent", "experiment", "revise", "predict")


# --------------------------------------------------------------------------- #
# The world, which she does not get to see
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Observation:
    """One statement about observed variables. The only thing that crosses the line."""

    left: str
    right: str
    dependent: bool
    given: FrozenSet[str] = frozenset()

    @property
    def pair(self) -> FrozenSet[str]:
        return frozenset({self.left, self.right})

    def to_dict(self) -> Dict[str, Any]:
        return {"left": self.left, "right": self.right, "dependent": self.dependent,
                "given": sorted(self.given)}


@dataclass(frozen=True)
class Truth:
    """The real graph, over observed **and** latent variables. Never handed to the solver."""

    observed: Tuple[str, ...]
    latent: Tuple[str, ...]
    edges: FrozenSet[Tuple[str, str]]

    @property
    def nodes(self) -> Tuple[str, ...]:
        return self.observed + self.latent

    def parents(self, node: str) -> Set[str]:
        return {a for a, b in self.edges if b == node}

    def children(self, node: str) -> Set[str]:
        return {b for a, b in self.edges if a == node}

    def descendants(self, node: str) -> Set[str]:
        seen: Set[str] = set()
        stack = [node]
        while stack:
            current = stack.pop()
            for child in self.children(current):
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        return seen

    # ---- what she is allowed to observe ---------------------------------- #
    def paths(self, left: str, right: str) -> List[List[str]]:
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        for a, b in self.edges:
            adjacency[a].add(b)
            adjacency[b].add(a)
        out: List[List[str]] = []

        def walk(node: str, seen: List[str]) -> None:
            if node == right:
                out.append(list(seen))
                return
            for other in sorted(adjacency[node]):
                if other not in seen:
                    walk(other, seen + [other])

        walk(left, [left])
        return out

    def connected(self, left: str, right: str, given: Sequence[str] = ()) -> bool:
        """d-connection on the true graph, latents included but never conditioned on.

        The latent is in the *world*, so it transmits dependence; it is not in the *data*, so it
        can never be held fixed. That asymmetry is the whole reason a latent is hard: it acts and
        cannot be controlled for.
        """
        held = set(given)
        for path in self.paths(left, right):
            blocked = False
            for i in range(1, len(path) - 1):
                before, node, after = path[i - 1], path[i], path[i + 1]
                collider = (before, node) in self.edges and (after, node) in self.edges
                if collider:
                    if not ((self.descendants(node) | {node}) & held):
                        blocked = True
                        break
                elif node in held:
                    blocked = True
                    break
            if not blocked:
                return True
        return False

    def responds(self, cause: str, effect: str) -> bool:
        """After ``do(cause)``, does *effect* move? The graph with its incoming edges cut.

        This is the hinge. ``A → B`` and ``A ← H → B`` agree on every observation and disagree
        here, which is why an intervention can separate what no amount of watching can.
        """
        cut = Truth(self.observed, self.latent,
                    frozenset(e for e in self.edges if e[1] != cause))
        return effect in cut.descendants(cause)


# --------------------------------------------------------------------------- #
# One task: a world, its observations, and the five things asked of it
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    truth: Truth
    observations: List[Observation] = field(default_factory=list)
    #: The intervention offered at stage three, and its true outcome.
    experiment: Tuple[str, str] = ()
    experiment_result: bool = False
    #: The pair used at stage five. In no observation and in no experiment above.
    held_out: Tuple[str, str] = ()
    held_out_result: bool = False
    has_latent: bool = False
    #: Can any permitted intervention settle the held-out question? Where it cannot, the pass is
    #: **silence**, and a prediction — right or wrong — is a failure.
    decidable: bool = True
    note: str = ""

    @property
    def nodes(self) -> Tuple[str, ...]:
        return self.truth.observed


@dataclass
class Stage:
    name: str
    passed: bool = False
    said: str = ""
    why: str = ""


@dataclass
class Paper:
    name: str
    passed: int = 0
    asked: int = 0
    #: For ``latent`` only: how often a hidden cause was reported where there was none.
    false_positives: int = 0
    positives: int = 0
    examples: List[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return round(self.passed / self.asked, 4) if self.asked else 0.0

    @property
    def precision(self) -> Optional[float]:
        """Of the worlds where a hidden cause was reported, how many had one.

        Reported beside the score on ``latent`` because that paper can be passed by a system that
        cries latent everywhere, and a recall without a precision would not notice.
        """
        if self.positives == 0:
            return None
        return round((self.positives - self.false_positives) / self.positives, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {"stage": self.name, "asked": self.asked, "passed": self.passed,
                "score": self.score, "precision": self.precision,
                "examples": self.examples[:6]}


@dataclass
class Report:
    papers: List[Paper] = field(default_factory=list)
    seed: int = DEFAULT_SEED
    solver: str = ""
    #: Worlds that ended with any model still standing.
    settled: int = 0
    #: Of those, how many ended holding the right *kind* — latent where there is one, and not
    #: where there is not. This is the number the intervention is supposed to earn, and it cannot
    #: be earned by observation at all. Counted over **decidable** worlds only: on the rest both
    #: readings survive by construction, which is the correct outcome and not an identification.
    identified: int = 0

    def paper(self, name: str) -> Optional[Paper]:
        return next((p for p in self.papers if p.name == name), None)

    @property
    def right_kind(self) -> Optional[float]:
        return round(self.identified / self.settled, 4) if self.settled else None

    @property
    def closed_loop(self) -> float:
        """The fraction of worlds that made it all the way to a correct unseen prediction.

        The only number on this page worth a headline. Every other paper is a stage of the loop and
        can be passed with the next one failing; this is the loop **closing**.
        """
        got = self.paper("predict")
        return got.score if got else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"seed": self.seed, "solver": self.solver,
                "closed_loop": self.closed_loop, "settled": self.settled,
                "identified": self.identified, "right_kind": self.right_kind,
                "papers": [p.to_dict() for p in self.papers]}


# --------------------------------------------------------------------------- #
# Minting a world
# --------------------------------------------------------------------------- #
_ONSET = "bdfgklmnprstvz"
_VOWEL = "aeiou"


def _word(rng: random.Random) -> str:
    return "".join(rng.choice(_ONSET) + rng.choice(_VOWEL)
                   for _ in range(rng.randint(2, 3))) + rng.choice("thnkr")


class Discovery:
    """Mints worlds, asks the five stages, and grades against ground truth it never reveals."""

    def __init__(self, *, seed: int = DEFAULT_SEED, limit: int = DEFAULT_LIMIT) -> None:
        self.seed = int(seed)
        self.limit = max(1, int(limit))

    # ---- worlds ---------------------------------------------------------- #
    def tasks(self) -> List[Task]:
        rng = random.Random(self.seed)
        out: List[Task] = []
        kinds = ("chain", "latent", "undecidable")
        for n in range(self.limit):
            out.append(self._mint(rng, kind=kinds[n % 3]))
        return out

    def _mint(self, rng: random.Random, *, kind: str) -> Task:
        """A **randomly generated** world, and whether it is decidable is computed, not decided.

        Three hand-written templates came first — a chain, a common cause, and one built so nothing
        could settle it — and the loop scored 1.000 on all three. That is exactly the number a
        system tuned to three templates would score, and there was no way to tell the two apart
        from the outside. So the templates are gone: the graph is drawn at random, the latents are
        drawn at random, and :meth:`_decidable` works out by exhaustive search whether any sequence
        of permitted interventions could settle the held-out question. ``kind`` now only biases how
        many hidden variables are drawn.
        """
        for _ in range(200):
            size = rng.randint(4, 5)
            observed = tuple(_word(rng) for _ in range(size))
            hidden_count = 0 if kind == "chain" else (1 if kind == "latent"
                                                      else rng.randint(1, 2))
            latent_nodes = tuple(f"{_word(rng)}_h" for _ in range(hidden_count))
            order = list(observed) + list(latent_nodes)
            rng.shuffle(order)
            edges: Set[Tuple[str, str]] = set()
            for i in range(1, len(order)):
                for j in range(i):
                    if rng.random() < 0.35:
                        edges.add((order[j], order[i]))
            # A latent that causes fewer than two things is not latent, it is absent: nothing it
            # does is distinguishable from an edge, so the world is secretly the no-latent case.
            if any(len({b for a, b in edges if a == h}) < 2 for h in latent_nodes):
                continue
            truth = Truth(observed=observed, latent=latent_nodes, edges=frozenset(edges))
            held = tuple(rng.sample(list(observed), 2))
            observations = self._observe(truth, exclude=frozenset({held}))
            if not any(o.dependent for o in observations):
                continue                      # a world with nothing in it teaches nothing
            decidable = self._decidable(truth, observations, held)
            return Task(truth=truth, has_latent=bool(latent_nodes), held_out=held,
                        held_out_result=truth.responds(*held), decidable=decidable,
                        observations=observations,
                        note=(f"{size} observed, {hidden_count} hidden, "
                              f"{'decidable' if decidable else 'nothing can settle it'}"))
        raise RuntimeError("could not mint a usable world")

    def _decidable(self, truth: Truth, observations: Sequence[Observation],
                   held: Tuple[str, str]) -> bool:
        """Could **any** sequence of permitted interventions settle the held-out question?

        Exhaustive over the model set a perfect reasoner would hold — every observed-only DAG that
        implies the observations, plus every one-latent variant of those — and over every permitted
        intervention up to :data:`BUDGET`. The held-out pair is not permitted, which is what makes
        some worlds genuinely undecidable rather than merely hard.

        Computed here rather than asserted by whoever wrote the world, because a hand-set flag is a
        claim about a search nobody ran.
        """
        models = _admissible(truth.observed, observations)
        if len({_predict(m, *held) for m in models}) <= 1:
            return True                       # already settled by observation alone

        def settle(current: Sequence[Any], depth: int) -> bool:
            if len({_predict(m, *held) for m in current}) <= 1:
                return True
            if depth <= 0:
                return False
            for cause, effect in itertools.permutations(truth.observed, 2):
                if frozenset({cause, effect}) == frozenset(held):
                    continue
                result = truth.responds(cause, effect)
                kept = [m for m in current if _predict(m, cause, effect) == result]
                if kept and len(kept) < len(current) and settle(kept, depth - 1):
                    return True
            return False

        return settle(models, BUDGET)

    @staticmethod
    def _observe(truth: Truth, *, exclude: FrozenSet[Tuple[str, str]]) -> List[Observation]:
        """The **complete** conditional-independence profile over observed variables.

        Every pair, against every subset of the others. Singletons only was the first version and
        it quietly broke the whole file: with an incomplete profile, "implies what was observed" is
        a weaker test than Markov equivalence, so :func:`_admissible` admitted **1,503** models
        where the loop was holding 57 and calling it a shortfall. The set was not too small; the
        yardstick was too loose, and every number downstream of it — decidability included — was
        computed against a fiction.

        The latent is not a variable here. It is not named, not conditioned on, and not counted,
        which is what makes it latent rather than merely unmentioned.
        """
        out: List[Observation] = []
        banned = {frozenset(pair) for pair in exclude}
        for left, right in itertools.combinations(truth.observed, 2):
            if frozenset({left, right}) in banned:
                continue
            others = [n for n in truth.observed if n not in (left, right)]
            for size in range(len(others) + 1):
                for given in itertools.combinations(others, size):
                    out.append(Observation(left, right,
                                           truth.connected(left, right, given),
                                           frozenset(given)))
        return out

    # ---- asking ---------------------------------------------------------- #
    def run(self, solver: Any) -> Report:
        """Five stages per world. A stage that cannot be reached is asked anyway and fails.

        Asked anyway, rather than skipped, because a report where the denominators differ per
        stage would let a system look better by getting stuck earlier.
        """
        report = Report(seed=self.seed, solver=type(solver).__name__)
        papers = {name: Paper(name=name) for name in STAGES}
        for task in self.tasks():
            state: Dict[str, Any] = {}
            # The solver is told **what it will be asked**, never the answer. Choosing an
            # experiment that bears on the question you need settled is what an experiment is for,
            # and withholding the question made the loop optimise entropy instead of an answer.
            if hasattr(solver, "aim"):
                try:
                    solver.aim(*task.held_out)
                except Exception:  # noqa: BLE001
                    pass
            for name in STAGES:
                paper = papers[name]
                paper.asked += 1
                stage = getattr(self, f"_stage_{name}")(solver, task, state)
                if stage.passed:
                    paper.passed += 1
                elif len(paper.examples) < 8:
                    paper.examples.append(f"{task.note}: {stage.why}")
            # Whether what survived actually matches the world. Not a stage — the stages are
            # about the loop turning; this is about where it stopped.
            # Only where the world *can* be identified. On an undecidable task both readings
            # survive by construction — that is what makes it undecidable — so counting it here
            # would score her for failing to know something nothing available could tell her.
            survivors = state.get("models") or []
            if survivors and task.decidable:
                report.settled += 1
                if bool(any(_hidden_nodes(m, task.nodes) for m in survivors)) == task.has_latent:
                    report.identified += 1
        report.papers = [papers[name] for name in STAGES]
        return report

    # ---- the five stages -------------------------------------------------- #
    def _stage_structure(self, solver: Any, task: Task, state: Dict[str, Any]) -> Stage:
        out = Stage(name="structure")
        try:
            models = list(solver.models(task.nodes, task.observations))
        except Exception as exc:  # noqa: BLE001
            out.why = f"{type(exc).__name__}"
            return out
        state["models"] = models
        out.said = f"{len(models)} models"
        # **Observationally consistent**, not identical to the hidden truth.
        #
        # Graded the other way for one run and it was wrong in exactly the way this file is about:
        # in a latent world *no* DAG over the observed variables equals the truth — `a` and `b`
        # have no edge between them, they have a common parent nobody can see — so the paper
        # scored 0.000 on every latent world and called that a structure failure. It is not. The
        # question at this stage is whether she holds a model that **implies what was observed**,
        # and a latent model that does is a right answer here, as is an observed-only one.
        for model in models:
            if _implies(model, task.observations, task.nodes):
                out.passed = True
                break
        out.why = out.why or (f"none of the {len(models)} models implies what was observed")
        return out

    def _stage_latent(self, solver: Any, task: Task, state: Dict[str, Any]) -> Stage:
        """Both kinds held, because no observation separates them.

        This was graded as *"did she say latent exactly where there is one"* for one run, and that
        was a paper rewarding a coin flip: over observational data a common cause is admissible
        for **every** dependence, so no evidence available at this stage can tell the two worlds
        apart. What *can* be asked is whether she committed anyway. Which world it is gets decided
        two stages later, by an intervention, which is the entire point of the file.
        """
        out = Stage(name="latent")
        models = state.get("models") or []
        with_hidden = [m for m in models if _hidden_nodes(m, task.nodes)]
        observed_only = [m for m in models if not _hidden_nodes(m, task.nodes)]
        state["said_latent"] = bool(with_hidden)
        out.said = f"{len(observed_only)} observed-only, {len(with_hidden)} with a hidden cause"
        out.passed = bool(with_hidden and observed_only)
        out.why = ("never considered an unobserved cause" if not with_hidden
                   else "considered only hidden causes, which commits just as hard"
                   if not observed_only else "")
        return out

    def _stage_experiment(self, solver: Any, task: Task, state: Dict[str, Any]) -> Stage:
        """Up to :data:`BUDGET` interventions, run to exhaustion or until the set settles.

        Turning the loop **once** is not a loop, and measuring it that way hid the difference. With
        a single experiment every earlier stage scored 1.000 and ``predict`` scored 0.000: the
        models were still split when the held-out question arrived, so the only honest answer was
        to decline. A cycle that starts and does not close is a step with extra ceremony.

        The world refuses the held-out pair. A system that asked for it would be buying stage five
        rather than deriving it, and the refusal is a legal move that costs a round.
        """
        out = Stage(name="experiment")
        models = state.get("models") or []
        if len(models) < 2:
            out.why = "held fewer than two models, so nothing to distinguish"
            return out
        rounds: List[Tuple[Tuple[str, str], bool]] = []
        splits = 0
        for _ in range(BUDGET):
            if len({_predict(m, *task.held_out) for m in models}) <= 1 and rounds:
                break            # the question it will be asked is already settled
            try:
                got = solver.experiment(models)
            except Exception as exc:  # noqa: BLE001
                out.why = f"{type(exc).__name__}"
                break
            if not got:
                break
            cause, effect = got
            if cause not in task.nodes or effect not in task.nodes:
                out.why = "named something that is not an observed variable"
                break
            if frozenset({cause, effect}) == frozenset(task.held_out):
                rounds.append((got, False))
                out.why = "asked for the held-out pair; the world declines"
                continue
            answers = {_predict(m, cause, effect) for m in models}
            if len(answers) > 1:
                splits += 1
            result = task.truth.responds(cause, effect)
            rounds.append((got, True))
            try:
                kept = list(solver.revise(models, cause, effect, result))
            except Exception as exc:  # noqa: BLE001
                out.why = f"revise raised {type(exc).__name__}"
                break
            should = [m for m in models if _predict(m, cause, effect) == result]
            state.setdefault("revisions", []).append(
                (len(kept), len(should), all(_predict(m, cause, effect) == result for m in kept)))
            models = kept
            state["models"] = models
            if len(models) <= 1:
                break
        state["rounds"] = rounds
        out.said = "; ".join(f"do({c}) -> {e}" for (c, e), _ok in rounds) or "(none)"
        if not task.decidable:
            # Nothing available settles this. Recognising that is the pass; running experiments
            # that cannot bear on the question is the failure.
            out.passed = splits == 0
            out.why = "" if out.passed else "ran experiments that cannot settle the question"
            return out
        out.passed = splits > 0
        out.why = out.why or ("named no intervention" if not rounds
                              else "every intervention it named settled nothing")
        return out

    def _stage_revise(self, solver: Any, task: Task, state: Dict[str, Any]) -> Stage:
        """Every round graded, both directions. Ran inside stage three; scored here."""
        out = Stage(name="revise")
        revisions = state.get("revisions") or []
        if not revisions:
            out.passed = not task.decidable      # nothing to revise, and nothing should have been
            out.why = "" if out.passed else "nothing was revised"
            return out
        out.said = " ".join(f"{kept}/{should}" for kept, should, _ok in revisions)
        too_much = any(kept < should for kept, should, _ok in revisions)
        too_little = any(not ok for _kept, _should, ok in revisions)
        empty = any(kept == 0 for kept, _should, _ok in revisions)
        out.passed = not (too_much or too_little or empty)
        out.why = ("killed models the result does not refute" if too_much
                   else "kept models the result refutes" if too_little
                   else "kept nothing at all" if empty else "")
        return out

    def _stage_predict(self, solver: Any, task: Task, state: Dict[str, Any]) -> Stage:
        out = Stage(name="predict")
        models = state.get("models") or []
        cause, effect = task.held_out
        if not models:
            out.why = "no surviving model to predict from"
            return out
        try:
            got = solver.predict(models, cause, effect)
        except Exception as exc:  # noqa: BLE001
            out.why = f"{type(exc).__name__}"
            return out
        out.said = str(got)
        if not task.decidable:
            out.passed = got is None
            out.why = "" if out.passed else f"answered {got} where nothing available settles it"
            return out
        if got is None:
            out.why = "declined to predict"
            return out
        out.passed = bool(got) == task.held_out_result
        out.why = out.why or (f"said {got}, the world does "
                              f"{'' if task.held_out_result else 'not '}respond")
        return out


# --------------------------------------------------------------------------- #
# What a model must look like from out here
# --------------------------------------------------------------------------- #
def _edges(model: Any) -> FrozenSet[Tuple[str, str]]:
    return frozenset(getattr(model, "edges", ()) or ())


def _observed_edges(model: Any, observed: Sequence[str]) -> FrozenSet[Tuple[str, str]]:
    names = set(observed)
    return frozenset((a, b) for a, b in _edges(model) if a in names and b in names)


def _hidden_nodes(model: Any, observed: Sequence[str]) -> Set[str]:
    names = set(observed)
    return {n for edge in _edges(model) for n in edge if n not in names}


def _admissible(observed: Sequence[str],
                observations: Sequence[Observation]) -> List[Any]:
    """Every model a perfect reasoner could be holding: observed-only DAGs and one-latent variants.

    The exam's own model space, used only to decide whether a world is decidable. It is built by
    brute force over orientations of every skeleton — far too slow to be a reasoning method and
    exactly right as a ground truth, which is the same division `explaingauntlet` makes between its
    permutation oracle and the topological sort it grades.
    """
    from nyxara.njp.loop import Model

    names = list(observed)
    out: List[Any] = []
    pairs = list(itertools.combinations(names, 2))
    for present in itertools.product((0, 1), repeat=len(pairs)):
        skeleton = [p for p, on in zip(pairs, present) if on]
        for bits in itertools.product((0, 1), repeat=len(skeleton)):
            edges = frozenset((a, b) if bit == 0 else (b, a)
                              for (a, b), bit in zip(skeleton, bits))
            model = Model(edges=edges, observed=tuple(names))
            if not _acyclic(edges):
                continue
            if _implies(model, observations, names):
                out.append(model)
            for index, (x, y) in enumerate(sorted(edges)):
                swapped = frozenset((edges - {(x, y)}) | {(f"h{index}", x), (f"h{index}", y)})
                variant = Model(edges=swapped, observed=tuple(names))
                if _implies(variant, observations, names):
                    out.append(variant)
    return out


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


def _implies(model: Any, observations: Sequence[Observation],
             observed: Sequence[str]) -> bool:
    """Does this model imply exactly the (in)dependencies that were observed?

    d-separation on the model's own graph, hidden nodes included as transmitters and **never** as
    things that can be conditioned on — the same asymmetry :meth:`Truth.connected` applies to the
    real world, because it is the same asymmetry: a latent acts and cannot be controlled for.
    """
    edges = _edges(model)
    if not edges:
        return not any(o.dependent for o in observations)
    nodes = {n for edge in edges for n in edge}
    adjacency: Dict[str, Set[str]] = defaultdict(set)
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    def descendants(node: str) -> Set[str]:
        seen: Set[str] = set()
        stack = [node]
        while stack:
            current = stack.pop()
            for a, b in edges:
                if a == current and b not in seen:
                    seen.add(b)
                    stack.append(b)
        return seen

    def connected(left: str, right: str, given: Sequence[str]) -> bool:
        if left not in nodes or right not in nodes:
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
                collider = (before, node) in edges and (after, node) in edges
                if collider:
                    if not ((descendants(node) | {node}) & held):
                        blocked = True
                        break
                elif node in held:
                    blocked = True
                    break
            if not blocked:
                return True
        return False

    for observed_fact in observations:
        got = connected(observed_fact.left, observed_fact.right, sorted(observed_fact.given))
        if got != observed_fact.dependent:
            return False
    return True


def _predict(model: Any, cause: str, effect: str) -> bool:
    """What this model says ``do(cause)`` does to *effect*. The mutilation, done out here.

    Deliberately not asked of the model. A solver that answered this itself could answer anything,
    and the paper would be measuring its honesty rather than its structure.
    """
    edges = {e for e in _edges(model) if e[1] != cause}
    seen: Set[str] = set()
    stack = [cause]
    while stack:
        current = stack.pop()
        for a, b in edges:
            if a == current and b not in seen:
                seen.add(b)
                stack.append(b)
    return effect in seen


# --------------------------------------------------------------------------- #
# The cold solver: what the package could do before the loop existed
# --------------------------------------------------------------------------- #
class Reconstructor:
    """:mod:`nyxara.njp.surgery`, and nothing else. The floor this file was written to record.

    It recovers an equivalence class over the variables it was given and has no representation for
    a variable it was not given, no notion of an intervention, and nothing to revise with. Stage
    one is what it is for; stages two to five are what it cannot reach, and naming that precisely
    is the point of running it.
    """

    def models(self, nodes: Sequence[str], observations: Sequence[Observation]) -> List[Any]:
        from nyxara.njp.surgery import Observation as SurgeryObservation, Surgeon

        got = Surgeon().discover(
            list(nodes),
            [SurgeryObservation(o.left, o.right, o.dependent, frozenset(o.given))
             for o in observations])
        return list(got.equivalent)

    def experiment(self, models: Sequence[Any]) -> Optional[Tuple[str, str]]:
        return None

    def revise(self, models: Sequence[Any], cause: str, effect: str,
               result: bool) -> List[Any]:
        return list(models)

    def predict(self, models: Sequence[Any], cause: str, effect: str) -> Optional[bool]:
        return None


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def run(solver: Any = None, *, seed: int = DEFAULT_SEED, limit: int = DEFAULT_LIMIT) -> Report:
    return Discovery(seed=seed, limit=limit).run(solver if solver is not None
                                                 else Reconstructor())


def render(report: Report) -> str:
    lines = [f"discovery — seed {report.seed}, solver {report.solver}", "",
             f"{'stage':12} {'asked':>6} {'passed':>7} {'score':>7} {'precision':>10}"]
    for paper in report.papers:
        precision = "         -" if paper.precision is None else f"{paper.precision:10.3f}"
        lines.append(f"{paper.name:12} {paper.asked:6} {paper.passed:7} "
                     f"{paper.score:7.3f} {precision}")
    kind = "-" if report.right_kind is None else f"{report.right_kind:.3f}"
    lines += ["", f"closed loop: {report.closed_loop:.3f} — worlds that reached a correct "
                  f"prediction nobody had asked about",
              f"right kind:  {kind} — of {report.settled} settled worlds, how many ended holding "
              f"a hidden cause exactly where there is one"]
    for paper in report.papers:
        if paper.examples:
            lines.append(f"\n─── {paper.name}")
            for example in paper.examples[:3]:
                lines.append(f"  {example}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="a world she has never seen, and only observations")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--solver", default="cold", choices=("cold", "loop"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.solver == "loop":
        from nyxara.njp.loop import Loop
        solver: Any = Loop()
    else:
        solver = Reconstructor()
    report = run(solver, seed=args.seed, limit=args.limit)
    print(json.dumps(report.to_dict(), indent=1) if args.json else render(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
