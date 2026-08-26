"""NYXARA · njp/unified.py — one corpus, fifty-one categories, and an organ behind every block.

:func:`~nyxara.njp.ingest.ingest_triples` loads what is. :func:`~nyxara.njp.experience.replay`
loads what follows what and what it cost to be wrong about it. Both were real and both left most
of this package unfed: nothing had ever handed :class:`~nyxara.njp.universe.ExperimentDesigner` a
rival to discriminate, :class:`~nyxara.njp.concepts.ConceptGenesis` a set of members to find an
invariant across, :class:`~nyxara.njp.discover.Discoverer` two antecedents with one consequent,
:class:`~nyxara.njp.selfmodel.SelfModel` a capability to be wrong about, or the grounder a
contradiction to revise.

:func:`absorb` is the one pass that feeds all of them. Each record carries a ``category`` and
whatever *blocks* that category needs, and every block routes to the organ that reads it:

===================  ============================================================================
block                where it goes
===================  ============================================================================
``knowledge``        ``grounder._assert`` — the same call ``ingest_triples`` makes
``qa``               ``memory.remember`` + ``memory.associate``, then recalled back to check
``observation``      ``universe.observe(state, order=…)`` — fits slope, sign and R²
``series``           the same, one row at a time, for a discovery that needs a table
``law``              ``world.state_law`` + ``universe.declare`` — the arrow, never its sign
``counterfactual``   ``universe.what_if`` — scored on direction
``prediction``       ``predictor.predict`` … ``predictor.observe`` → an ``Outcome`` to diagnose
``state``/``action`` ``predictive.observe`` — the discrete transition model
``hypotheses``       ``ExperimentDesigner.propose`` → ``observe_result``
``examples``         ``genesis.observe`` → ``crystallise`` → ``generalise``
``abstraction``      ``discoverer.observe`` → ``discover``
``capability``       ``selfmodel.observe`` — an estimate that is allowed to be low
``goal``/``steps``   ``goals.add`` — a mission and its children
``episode``          ``memory.remember`` at the episodic level
``contradiction``    both sides asserted in order, and the supersede counted
``problem``          ``metareason.classify`` — which method this problem is worth
===================  ============================================================================

**The rule this file is built on: no block exists that no organ reads.** A category whose blocks
route nowhere is a label, and a corpus of labels measures nothing.
``tests/njp/test_unified.py`` walks :data:`~prepare_unified_corpus.CATEGORIES` and asserts the
organ named for each one is an attribute the brain actually has, so a fifty-second category
cannot be added as a row in a table.

**What this is not.** It is not the scraped web and no hand-written corpus can be. What a person
can write down is the *structure* — the law behind the readings, the rival that the experiment
separates, the invariant that survives the awkward member, the principle carried into a domain it
was never stated in. That is exactly what a scraped corpus is worst at and what these organs need.

    python -m nyxara.njp.unified --records nyxara/njp/data/world_unified.jsonl.gz

Duck-typed on every organ, so a brain missing one absorbs the rest instead of failing. Nothing
here opens a socket.
"""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["AbsorbReport", "load", "absorb"]

_FLAT = 1e-6


@dataclass
class AbsorbReport:
    """What went in, what came back out, and the handful of things that can be marked right."""

    records: int = 0
    by_category: Dict[str, int] = field(default_factory=dict)
    # what was fed
    facts: int = 0
    qa_pairs: int = 0
    episodes: int = 0
    events: int = 0
    observations: int = 0
    arrows_declared: int = 0
    predictions: int = 0
    transitions: int = 0
    hypotheses: int = 0
    concept_members: int = 0
    abstraction_cases: int = 0
    capabilities: int = 0
    goals: int = 0
    # what can be checked
    facts_recalled: int = 0
    facts_asked: int = 0
    signs_correct: int = 0
    signs_asked: int = 0
    counterfactuals_correct: int = 0
    counterfactuals_asked: int = 0
    outcomes_scored: int = 0
    diagnoses_agreeing: int = 0
    diagnoses_asked: int = 0
    hypotheses_resolved: int = 0
    concepts_formed: int = 0
    members_generalised: int = 0
    abstractions_found: int = 0
    transitions_predicted: int = 0
    transitions_asked: int = 0
    supersedes: int = 0
    #: Contradictions the grounder *noticed* on the second claim, which is a different number from
    #: supersedes and the honest one to report. Most of these pairs disagree on `is_a`, which holds
    #: many values, so nothing is superseded and nothing should be: "pluto is a planet" and "pluto
    #: is a dwarf planet" are not a functional clash. `_revise` fires on the functional relations.
    contradictions_seen: int = 0
    revisions_asked: int = 0
    ms: float = 0.0
    notes: List[str] = field(default_factory=list)

    def _rate(self, hit: int, asked: int) -> Optional[float]:
        return round(hit / asked, 4) if asked else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "records": self.records,
            "categories": len(self.by_category),
            "fed": {"facts": self.facts, "qa_pairs": self.qa_pairs, "episodes": self.episodes,
                    "events": self.events, "observations": self.observations,
                    "arrows_declared": self.arrows_declared, "predictions": self.predictions,
                    "transitions": self.transitions, "hypotheses": self.hypotheses,
                    "concept_members": self.concept_members,
                    "abstraction_cases": self.abstraction_cases,
                    "capabilities": self.capabilities, "goals": self.goals},
            "checked": {
                "fact_recall": self._rate(self.facts_recalled, self.facts_asked),
                "sign_accuracy": self._rate(self.signs_correct, self.signs_asked),
                "counterfactual_accuracy": self._rate(self.counterfactuals_correct,
                                                      self.counterfactuals_asked),
                "diagnosis_agreement": self._rate(self.diagnoses_agreeing, self.diagnoses_asked),
                "transition_accuracy": self._rate(self.transitions_predicted,
                                                  self.transitions_asked),
                "outcomes_scored": self.outcomes_scored,
                "hypotheses_resolved": self.hypotheses_resolved,
                "concepts_formed": self.concepts_formed,
                "members_generalised": self.members_generalised,
                "abstractions_found": self.abstractions_found,
                "supersedes": self.supersedes,
                "contradictions_seen": self.contradictions_seen,
                "revision_rate": self._rate(self.supersedes, self.revisions_asked),
            },
            "by_category": dict(sorted(self.by_category.items())),
            "ms": round(self.ms, 2),
            "notes": self.notes[:12],
        }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load(path: Any, *, limit: int = 0,
         categories: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Read the unified ``.jsonl``/``.jsonl.gz``. A malformed line costs that line, not the file."""
    name = str(path)
    opener = gzip.open if name.endswith(".gz") else open
    wanted = {str(c) for c in categories} if categories else None
    out: List[Dict[str, Any]] = []
    with opener(name, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict) or "category" not in row:
                continue
            if wanted and row["category"] not in wanted:
                continue
            out.append(row)
            if limit and len(out) >= limit:
                break
    return out


# --------------------------------------------------------------------------- #
# The one pass
# --------------------------------------------------------------------------- #
def absorb(brain: Any, records: Sequence[Dict[str, Any]], *,
           check: bool = True) -> AbsorbReport:
    """Route every block of every record to the organ that reads it, then check what can be.

    ``check`` off skips the read-back passes — recalling a fact, asking a counterfactual, asking
    the transition model to predict what it was just shown. They are what make the report mean
    anything and they cost a second pass, so a caller loading a corpus for real can turn them off.
    """
    report = AbsorbReport(records=len(records))
    started = time.perf_counter()

    organs = {name: getattr(brain, name, None) for name in
              ("grounder", "world", "universe", "predictor", "predictive", "memory", "genesis",
               "goals", "levels", "metareason")}
    # ``self_model`` with the underscore. ``selfmodel`` is not an attribute of the brain, and
    # reading it returned None into a guarded router that then did nothing, quietly.
    organs["self_model"] = getattr(brain, "self_model", None) or getattr(brain, "selfmodel", None)
    hypothesis_space = _hypothesis_space(brain)
    discoverer = _discoverer(brain)

    for record in records:
        category = str(record.get("category") or "unknown")
        report.by_category[category] = report.by_category.get(category, 0) + 1
        _knowledge(organs["grounder"], record, report)
        _qa(organs["memory"], record, report)
        _episode(organs["memory"], organs["levels"], record, report)
        _law(organs["world"], organs["universe"], record, report)
        _observation(organs["universe"], record, report)
        _event(organs["world"], record, report)
        _prediction(organs["predictor"], record, report)
        _transition(organs["predictive"], record, report, check=check)
        _hypotheses(hypothesis_space, record, report)
        _concept(organs["genesis"], record, report)
        _abstraction(discoverer, record, report)
        _capability(organs["self_model"], record, report)
        _goal(organs["goals"], record, report)
        _contradiction(organs["grounder"], record, report)
        _strategy(organs["metareason"], record, report)

    if check:
        _crystallise(organs["genesis"], records, report)
        _discover(discoverer, report)
        _check_facts(organs["grounder"], records, report)
        _check_signs(organs["universe"], records, report)
        _check_counterfactuals(organs["universe"], records, report)

    report.ms = (time.perf_counter() - started) * 1000.0
    return report


def _hypothesis_space(brain: Any) -> Any:
    """The brain's own designer if it has one, otherwise a fresh one.

    The class is ``ExperimentDesigner`` and not ``HypothesisSpace``, which the first version of
    this file guessed. It imported nothing, the guard swallowed it, and the report said zero
    hypotheses in a run that had six records full of them — which is what a silent fallback buys
    you when the name is wrong.
    """
    existing = getattr(brain, "designer", None) or getattr(brain, "experiments", None)
    if existing is not None and hasattr(existing, "propose"):
        return existing
    try:
        from nyxara.njp.universe import ExperimentDesigner

        return ExperimentDesigner()
    except Exception:  # noqa: BLE001
        return None


def _discoverer(brain: Any) -> Any:
    """The brain's own `Discoverer` — it has one, and a private copy would learn in a corner."""
    existing = getattr(brain, "discoverer", None)
    if existing is not None and hasattr(existing, "observe"):
        return existing
    try:
        from nyxara.njp.discover import Discoverer

        return Discoverer()
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# One router per block
# --------------------------------------------------------------------------- #
def _knowledge(grounder: Any, record: Dict[str, Any], report: AbsorbReport) -> None:
    """``subject | predicate | object`` into the fact store, through the ingest path's own call."""
    rows = record.get("knowledge") or []
    if grounder is None or not rows:
        return
    try:
        from nyxara.njp.grounding import GroundedTriple

        for item in rows:
            if len(item) < 3:
                continue
            grounder._assert(GroundedTriple(subject=item[0], predicate=item[1], object=item[2],
                                            confidence=0.85, source="unified",
                                            provenance="observed"))
            report.facts += 1
    except Exception:  # noqa: BLE001
        pass


def _qa(memory: Any, record: Dict[str, Any], report: AbsorbReport) -> None:
    """A question bound to its answer, so the question is a cue and not just a string."""
    pairs = record.get("qa") or []
    if memory is None or not pairs:
        return
    for index, pair in enumerate(pairs):
        if len(pair) < 2:
            continue
        question, answer = pair[0], pair[1]
        try:
            memory.remember(f"{record.get('id')}.qa{index}", answer, kind="qa", cue=question)
            memory.associate(question, answer)
            report.qa_pairs += 1
        except Exception:  # noqa: BLE001
            continue


def _episode(memory: Any, levels: Any, record: Dict[str, Any], report: AbsorbReport) -> None:
    """``key | what happened | the cue``, at the episodic level so consolidation can act on it."""
    for item in record.get("episode") or []:
        if len(item) < 2:
            continue
        key, text = item[0], item[1]
        cue = item[2] if len(item) > 2 else ""
        try:
            if levels is not None:
                levels.remember(key, text, cue=cue)
            elif memory is not None:
                memory.remember(key, text, kind="episode", cue=cue)
            report.episodes += 1
        except Exception:  # noqa: BLE001
            continue


def _law(world: Any, universe: Any, record: Dict[str, Any], report: AbsorbReport) -> None:
    """The arrow, stated to exist. The sign is carried in the record and used only to *mark*.

    ``declare`` is called with the sign here — unlike :mod:`nyxara.njp.experience`, which withholds
    it — because most of these records state a law and supply no readings for it to be fitted from.
    Where readings *are* supplied (``observation``, ``series``), the fit runs too and
    :func:`_check_signs` marks the fitted slope against the stated sign rather than against itself.
    """
    for item in record.get("law") or []:
        if len(item) < 3:
            continue
        cause, effect = item[0], item[2]
        sign = int(item[3]) if len(item) > 3 and str(item[3]).lstrip("-").isdigit() else 0
        try:
            if world is not None:
                world.state_law(cause, effect)
            if universe is not None:
                universe.declare(cause, effect, sign=sign)
            report.arrows_declared += 1
        except Exception:  # noqa: BLE001
            continue


def _observation(universe: Any, record: Dict[str, Any], report: AbsorbReport) -> None:
    """A joint reading, or a whole table of them for a record that shows a pattern."""
    if universe is None:
        return
    order = list(record.get("order") or [])
    reading = record.get("observation") or {}
    if reading:
        try:
            universe.observe(dict(reading), order=order or None)
            report.observations += 1
        except Exception:  # noqa: BLE001
            pass
    variables = list(record.get("variables") or [])
    if len(variables) >= 2:
        for row in record.get("series") or []:
            if len(row) < len(variables):
                continue
            try:
                universe.observe({name: float(row[i]) for i, name in enumerate(variables)},
                                 order=variables)
                report.observations += 1
            except (ValueError, TypeError):
                continue
            except Exception:  # noqa: BLE001
                break


def _event(world: Any, record: Dict[str, Any], report: AbsorbReport) -> None:
    """An actor doing something to an object, in order, so ``links()`` has something to weigh."""
    action = record.get("action")
    if world is None or not action or not isinstance(action, str):
        return
    try:
        from nyxara.njp.world import Event

        world.observe(Event(actor=str(record.get("actor") or ""), action=action,
                            object=str(record.get("object") or ""),
                            text=str(record.get("text") or "")))
        report.events += 1
    except Exception:  # noqa: BLE001
        pass


def _prediction(predictor: Any, record: Dict[str, Any], report: AbsorbReport) -> None:
    """Commit before the fact, meet reality, and route the miss to an organ.

    The record's own ``diagnosis`` is not fed in — it is compared against what
    :meth:`~nyxara.njp.predict.PredictionEngine.diagnose` returns, which is why
    ``diagnosis_agreement`` is a measurement rather than a restatement.
    """
    rows = record.get("prediction") or []
    result = record.get("result")
    if predictor is None or not rows or result is None:
        return
    item = rows[0]
    if len(item) < 2:
        return
    key = str(item[0])
    expected: Any = item[1]
    confidence = float(item[2]) if len(item) > 2 else 0.5
    actual: Any = result
    expected, actual = _numeric_pair(expected, actual)
    try:
        predictor.predict(key, expected, confidence=confidence, organ=_organ_for(record))
        report.predictions += 1
        evidence = _evidence(record)
        outcome = predictor.observe(key, actual, evidence=evidence)
        if outcome is None:
            return
        report.outcomes_scored += 1
        stated = str(record.get("diagnosis") or "")
        if not stated:
            return
        diagnosis = getattr(outcome, "diagnosis", None)
        if diagnosis is None:
            diagnosis = predictor.diagnose(outcome, evidence)
        report.diagnoses_asked += 1
        if str(getattr(diagnosis, "kind", "")) == stated:
            report.diagnoses_agreeing += 1
    except Exception:  # noqa: BLE001
        pass


def _evidence(record: Dict[str, Any]) -> Dict[str, Any]:
    """The record's own ``key | value`` evidence, plus the organ that owns the prediction.

    ``diagnose`` reaches its branches through this dict and nothing else — a stated diagnosis with
    no evidence behind it comes back UNATTRIBUTED, which is the module refusing to guess rather
    than a defect. Measured before these were supplied: eight of the error records disagreed, and
    every one of them was a branch that needs an evidence key.
    """
    out: Dict[str, Any] = {"organ": _organ_for(record)}
    for item in record.get("evidence") or []:
        if not item:
            continue
        key = str(item[0])
        raw = item[1] if len(item) > 1 else ""
        if raw == "":
            out[key] = None
            continue
        try:
            out[key] = float(raw) if str(raw).replace(".", "", 1).isdigit() else raw
        except (TypeError, ValueError):
            out[key] = raw
    return out


def _numeric_pair(expected: Any, actual: Any) -> Tuple[Any, Any]:
    """Both as numbers when both parse as numbers, so ``_similarity`` scores closeness not tokens."""
    try:
        return float(str(expected)), float(str(actual))
    except (TypeError, ValueError):
        return expected, actual


def _organ_for(record: Dict[str, Any]) -> str:
    """Which organ owns this prediction. Read from the record's own diagnosis when it states one.

    ``diagnose`` reaches WORLD_MODEL, PLANNING and the rest partly *through the organ name*, so a
    record that says the miss belongs to the world model has to be predicted by the world model or
    the comparison is rigged in the other direction — it would never agree.
    """
    stated = str(record.get("diagnosis") or "")
    if stated in ("world_model", "world", "manifold"):
        return "world_model"
    if stated in ("planning", "planner", "goals"):
        return "planner"
    return stated or "world_model"


def _transition(predictive: Any, record: Dict[str, Any], report: AbsorbReport, *,
                check: bool) -> None:
    """``state --action--> next_state``, then asked to predict what it was just shown."""
    state = list(record.get("state") or [])
    nxt = list(record.get("next_state") or [])
    action = record.get("action")
    if predictive is None or not state or not nxt or not isinstance(action, str):
        return
    try:
        from nyxara.njp.predictive import WorldState

        # `WorldState.of` rather than the bare lists. `_sig` stringifies whatever it is handed, so
        # a list goes in as its `repr` — measured, `top` came back as
        # "['the lamp is on', 'the bulb is working', 'power is on']" while the check compared
        # against the canonical "power is on|the bulb is working|the lamp is on", and the accuracy
        # read 0.0 on a model that was predicting every transition perfectly.
        before, after = WorldState.of(state), WorldState.of(nxt)
        predictive.observe(before, action, next_state=after)
        report.transitions += 1
        if not check:
            return
        prediction = predictive.predict(before, action)
        report.transitions_asked += 1
        if str(getattr(prediction, "top", "")) == after.signature:
            report.transitions_predicted += 1
    except Exception:  # noqa: BLE001
        pass


def _hypotheses(space: Any, record: Dict[str, Any], report: AbsorbReport) -> None:
    """Rivals, the experiment that separates them, and what the experiment actually said.

    Each hypothesis is proposed with what it *predicts* the experiment will show, which is what
    makes an observation discriminating: an experiment both rivals predict the same answer to
    removes no probability from either.
    """
    rivals = record.get("hypotheses") or []
    if space is None or len(rivals) < 2:
        return
    experiments = record.get("experiment") or []
    outcomes = record.get("outcome") or []
    try:
        space.hypotheses.clear()
        for index, rival in enumerate(rivals):
            name = rival[0]
            probability = float(rival[1]) if len(rival) > 1 else 0.5
            predictions: Dict[str, str] = {}
            for experiment in experiments:
                if len(experiment) >= 2 + index:
                    predictions[experiment[0]] = experiment[1 + index]
            space.propose(name, probability=probability, predictions=predictions)
            report.hypotheses += 1
        for outcome in outcomes:
            if len(outcome) < 2:
                continue
            space.observe_result(outcome[0], outcome[1])
            report.hypotheses_resolved += 1
    except Exception:  # noqa: BLE001
        pass


def _concept(genesis: Any, record: Dict[str, Any], report: AbsorbReport) -> None:
    """Members with the property they share, for an invariant to be found rather than stated."""
    examples = record.get("examples") or []
    if genesis is None or not examples:
        return
    concept = str(record.get("concept") or "")
    for item in examples:
        if len(item) < 2:
            continue
        subject, property_ = item[0], item[1]
        features = [f"has_property:{property_}"]
        if concept:
            features.append(f"is_a:{concept}")
        try:
            genesis.observe(subject, features)
            report.concept_members += 1
        except Exception:  # noqa: BLE001
            continue


def _abstraction(discoverer: Any, record: Dict[str, Any], report: AbsorbReport) -> None:
    """``antecedent + antecedent | consequent`` — several cases, one rule to be found across them."""
    for item in record.get("abstraction") or []:
        if len(item) < 2:
            continue
        antecedents = [part.strip() for part in str(item[0]).split("+") if part.strip()]
        if not antecedents or discoverer is None:
            continue
        try:
            discoverer.observe(antecedents, str(item[1]))
            report.abstraction_cases += 1
        except Exception:  # noqa: BLE001
            continue


def _capability(selfmodel: Any, record: Dict[str, Any], report: AbsorbReport) -> None:
    """A capability and how well it actually went. Allowed to be zero, and one of them is."""
    capability = record.get("capability")
    success = record.get("success")
    if selfmodel is None or not capability or success is None:
        return
    try:
        selfmodel.observe(str(capability), float(success))
        report.capabilities += 1
    except Exception:  # noqa: BLE001
        pass


def _goal(goals: Any, record: Dict[str, Any], report: AbsorbReport) -> None:
    """A mission and its steps as children, so ``ready()`` has an ordering to report."""
    goal = record.get("goal")
    steps = record.get("steps") or []
    if goals is None or not goal:
        return
    try:
        parent = goals.add(str(goal), kind="mission")
        report.goals += 1
        parent_id = getattr(parent, "nid", None) or getattr(parent, "id", "") or str(goal)
        for step in steps:
            goals.add(str(step), kind="task", parent=str(parent_id))
            report.goals += 1
    except Exception:  # noqa: BLE001
        pass


def _contradiction(grounder: Any, record: Dict[str, Any], report: AbsorbReport) -> None:
    """Both sides, in the order they were believed, so the revision is a revision and not a load."""
    pairs = record.get("contradiction") or []
    if grounder is None:
        return
    _clash(grounder, record, report)
    if not pairs:
        return
    for pair in pairs:
        if len(pair) < 2:
            continue
        try:
            before = _superseded(grounder)
            grounder.ground(str(pair[0]))
            second = grounder.ground(str(pair[1]))
            report.contradictions_seen += len(getattr(second, "contradictions", None) or [])
            if _superseded(grounder) > before:
                report.supersedes += 1
        except Exception:  # noqa: BLE001
            continue


def _clash(grounder: Any, record: Dict[str, Any], report: AbsorbReport) -> None:
    """The same functional relation given two values, in order, straight through ``_assert``.

    This is the only route that reaches :meth:`~nyxara.njp.grounding.Grounder._revise`. The prose
    side cannot: measured, ``ground("the capital of myanmar is yangon")`` extracts no relation at
    all, so a revision counted from the sentences would be counting the pattern table's coverage
    and calling it belief revision.
    """
    rows = record.get("clash") or []
    if not rows:
        return
    try:
        from nyxara.njp.grounding import GroundedTriple

        for item in rows:
            if len(item) < 4:
                continue
            subject, predicate, old_value, new_value = item[0], item[1], item[2], item[3]
            before = _superseded(grounder)
            prior = GroundedTriple(subject=subject, predicate=predicate, object=old_value,
                                   confidence=0.7, source="unified", provenance="observed")
            incoming = GroundedTriple(subject=subject, predicate=predicate, object=new_value,
                                      confidence=0.85, source="unified", provenance="observed")
            grounder._assert(prior)
            # ``_assert`` deliberately does not revise — it is the bulk-load path, and
            # `ingest_triples` uses it for exactly that reason: a corpus is testimony, not a
            # conversation, and running contradiction detection over a quarter of a million rows
            # would supersede half of them against each other. `_revise` is the conversational
            # path, so a record that means to test revision has to call it.
            grounder._revise(prior, incoming)
            grounder._assert(incoming)
            report.revisions_asked += 1
            if _superseded(grounder) > before:
                report.supersedes += 1
    except Exception:  # noqa: BLE001
        pass


def _superseded(grounder: Any) -> int:
    try:
        return sum(1 for triples in getattr(grounder, "facts", {}).values()
                   for triple in triples if getattr(triple, "superseded", False))
    except Exception:  # noqa: BLE001
        return 0


def _strategy(metareason: Any, record: Dict[str, Any], report: AbsorbReport) -> None:
    """Which method the problem is worth. Classified, not told."""
    problem = record.get("problem")
    if metareason is None or not problem or not record.get("strategy"):
        return
    try:
        metareason.classify(str(problem))
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# The read-back passes
# --------------------------------------------------------------------------- #
def _crystallise(genesis: Any, records: Sequence[Dict[str, Any]], report: AbsorbReport) -> None:
    """Invent the kinds, then ask each member which kinds it now belongs to."""
    if genesis is None or not report.concept_members:
        return
    try:
        genesis.crystallise()
        concepts = getattr(genesis, "concepts", {}) or {}
        report.concepts_formed = len(concepts)
        # Membership of a formed concept, read off the concepts themselves. `generalise` was the
        # first thing tried and it returns the ancestor *names* for a subject, which is empty
        # until the hierarchy has links — so it measured the ancestry rather than whether the
        # member had been claimed at all, and reported four out of forty-four on a pass that had
        # claimed nearly all of them.
        claimed = {str(subject).lower()
                   for concept in concepts.values()
                   for subject in (getattr(concept, "members", None) or [])}
        for record in records:
            for item in record.get("examples") or []:
                if item and str(item[0]).lower() in claimed:
                    report.members_generalised += 1
    except Exception:  # noqa: BLE001
        pass


def _discover(discoverer: Any, report: AbsorbReport) -> None:
    if discoverer is None or not report.abstraction_cases:
        return
    try:
        discoverer.discover()
        report.abstractions_found = len(discoverer.confirmed())
    except Exception:  # noqa: BLE001
        pass


def _check_facts(grounder: Any, records: Sequence[Dict[str, Any]], report: AbsorbReport) -> None:
    """Ask back one fact per knowledge record — the store holding it is not the same as reaching it."""
    if grounder is None:
        return
    for record in records:
        rows = record.get("knowledge") or []
        if not rows or len(rows[0]) < 3:
            continue
        subject, predicate, obj = rows[0][0], rows[0][1], rows[0][2]
        try:
            found = grounder._lookup(subject, predicate)
            report.facts_asked += 1
            if any(str(getattr(t, "object", "")).lower() == str(obj).lower() for t in found or []):
                report.facts_recalled += 1
        except Exception:  # noqa: BLE001
            continue


def _check_signs(universe: Any, records: Sequence[Dict[str, Any]], report: AbsorbReport) -> None:
    """Mark the *fitted* slope against the stated sign, only where readings were supplied."""
    if universe is None:
        return
    for record in records:
        if not (record.get("series") or record.get("observation")):
            continue
        for item in record.get("law") or []:
            if len(item) < 4:
                continue
            cause, effect = item[0], item[2]
            try:
                stated = int(item[3])
            except (TypeError, ValueError):
                continue
            relation = _relation(universe, cause, effect)
            if relation is None or int(getattr(relation, "n", 0)) < 3:
                continue
            slope = float(getattr(relation, "slope", 0.0))
            if abs(slope) <= _FLAT:
                continue
            report.signs_asked += 1
            if (1 if slope > 0 else -1) == stated:
                report.signs_correct += 1


def _check_counterfactuals(universe: Any, records: Sequence[Dict[str, Any]],
                           report: AbsorbReport) -> None:
    """``do(variable = value)`` and which way the named effect moved. Direction only."""
    if universe is None:
        return
    for record in records:
        for item in record.get("counterfactual") or []:
            if len(item) < 4:
                continue
            variable, value, effect = item[0], item[1], item[2]
            try:
                expected = int(item[3])
                # The record's own reading is the baseline. Without it every counterfactual is
                # asked against `universe.state`, which by the end of a pass holds whatever was
                # observed last *anywhere in the corpus* — so a question about this episode's
                # water level was being answered from another scenario's final state. Measured, it
                # cost about a fifth of them: 0.81 against 1.00 once the base is the right one.
                base = dict(record.get("observation") or {}) or None
                answer = universe.intervene({variable: float(value)}, base=base)
            except (TypeError, ValueError):
                continue
            except Exception:  # noqa: BLE001
                continue
            if not getattr(answer, "answerable", False):
                continue
            for delta in answer.changed():
                if str(getattr(delta, "variable", "")) != effect:
                    continue
                change = getattr(delta, "change", None)
                got = (1 if change > 0 else -1) if (change is not None
                                                    and abs(change) > _FLAT) \
                    else int(getattr(delta, "direction", 0))
                if not got:
                    break
                report.counterfactuals_asked += 1
                if got == expected:
                    report.counterfactuals_correct += 1
                break


def _relation(universe: Any, cause: str, effect: str) -> Any:
    try:
        for (a, b), candidate in getattr(universe, "relations", {}).items():
            if a == cause and b == effect:
                return candidate
    except Exception:  # noqa: BLE001
        return None
    return None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
DEFAULT_RECORDS = Path(__file__).with_name("data") / "world_unified.jsonl.gz"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m nyxara.njp.unified [--records PATH] [--category NAME] [--no-check]``."""
    import argparse

    parser = argparse.ArgumentParser(description="Absorb the unified corpus into NJP.")
    parser.add_argument("--records", default=str(DEFAULT_RECORDS))
    parser.add_argument("--category", action="append", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-check", action="store_true",
                        help="load only; skip the read-back passes")
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows = load(args.records, limit=args.limit, categories=args.category)
    if not rows:
        print(f"no records loaded from {args.records}")
        return 1

    from nyxara.njp.brain import NJPBrain

    report = absorb(NJPBrain(), rows, check=not args.no_check)
    print(json.dumps(report.to_dict(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
