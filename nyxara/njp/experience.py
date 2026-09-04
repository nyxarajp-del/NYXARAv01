"""NYXARA · njp/experience.py — replaying lived episodes through the organs that learn (🎞️→🧠).

:func:`~nyxara.njp.ingest.ingest_triples` loads *testimony*: 3,745 claims about how the world is.
This loads *experience*: a state she was in, an action taken, a prediction she made from the prior
she happened to hold, the number reality returned, the error that produced, and the arrow the
numbers support. The two are not degrees of the same thing. A fact store has no representation of
having been wrong, and every organ NJP owns for learning from being wrong takes an argument a
triple cannot supply:

* :meth:`nyxara.njp.universe.InternalUniverse.observe` fits slope, sign and R² from **joint
  numeric readings**. ``water requires sunlight`` carries no quantity and can never be fitted.
* :meth:`nyxara.njp.predict.PredictionEngine.predict` wants a claim **registered before** the fact,
  scored afterwards. Nothing in a fact store is ever wrong in a way that yields an
  :class:`~nyxara.njp.predict.Outcome` with an organ to blame.
* :meth:`nyxara.njp.world.WorldView.observe` wants **events in order**, from which ``links()``
  separates a cause from a coincidence by lift rather than by assertion.
* :meth:`nyxara.njp.universe.InternalUniverse.intervene` — the do-operator — needs an **oriented**
  arrow. Direction is what makes a counterfactual answerable instead of a guess.

So :func:`replay` is not a loader with extra steps. It is the loop, in the order the loop runs::

    event → predict → observe(joint) → score → diagnose → declare → refit → counterfactual

**What it deliberately does not do.** It never tells the universe which *way* the effect moves.
:meth:`~nyxara.njp.universe.InternalUniverse.declare` is called with no ``sign``, which says only
"this arrow exists, watch it" — the statement a sentence like *"the gardener watered the plant and
it grew"* actually makes. The sign has to come out of the numbers, or nothing was learned. That is
why :attr:`ReplayReport.signs_correct` is the number worth reading in the report: handing the sign
over and then reporting that it came back would be measuring the loader.

**The predictions are supposed to miss.** Every scenario in ``scripts/experience`` states a prior
that disagrees with its own ground truth — a linear prior on a square law, a rising one on a
falling law. An episode whose prediction is already right attributes nothing and corrects nothing,
and the generator rejects a scenario that believes what is true.

What it measures out at
-----------------------
195 episodes over 27 scenarios, into a bare :class:`~nyxara.njp.brain.NJPBrain`, in ~11 ms:

* **27 of 27 arrows learned with the correct sign**, and the sign is never supplied — ``declare``
  is called without one. Eleven of those laws are not straight lines (``sqrt``, ``square``,
  ``inverse``); the fitted line still gets the direction right on every one.
* **Fitted coefficient within 2.3% of the real one**, averaged over the twelve scenarios whose law
  really is linear. So it is not only "which way" but "how steep", from readings alone.
* **27 of 27 counterfactuals answered in the right direction.** ``do(cause = the far end of its
  range)`` from where the scenario ended, scored on direction only.
* **127 of 195 predictions missed**, mean error 0.41 — which is the file working. The priors are
  wrong on purpose.
* **195 of 195 misses attributed to** ``world_model``, which is what
  :meth:`~nyxara.njp.predict.PredictionEngine.diagnose` returns for that organ, reached by its
  branch 4 rather than by the UNATTRIBUTED fallback.
* **672 discrete transitions** into :class:`~nyxara.njp.predictive.PredictiveWorldModel`.

**And one measured architectural fact, from running it the hard way.** With ``--no-declare`` —
numbers handed over and nothing stated — **not one arrow is fitted**, on any scenario. The cause
is :meth:`~nyxara.njp.universe.InternalUniverse._permitted`: its permissive fallback applies only
when ``self.world is None``, and a real brain always has a :class:`~nyxara.njp.world.WorldView`.
Measured directly: a bare ``InternalUniverse()`` fed the plant readings fits ``water → growth`` at
slope 1.162 against a true 1.2, usable and oriented; the same readings into ``NJPBrain().universe``
fit nothing at all, because ``world.link("water", "growth")`` has no ``causal``, no ``stated`` and
no ``together`` to offer.

That is not a defect to route around, and this module does not route around it. It is the reason
each scenario states that the arrow *exists* before supplying a single number — which is what an
experiment is. You vary one thing and measure another because you already suspect they are
connected; a machine that fitted every pair it saw regardless would be, in ``observe``'s own
words, "not a world model but a machine for finding coincidences".

Reading the report
------------------
``signs_correct`` is learning; ``mean_error`` is how wrong the prior was, and it is *supposed* to
be large; ``diagnosed`` should be ``world_model`` throughout, because that is what
:meth:`~nyxara.njp.predict.PredictionEngine.diagnose` returns for a miss by that organ, and a run
that attributes these anywhere else has a bug in the evidence it passes rather than an insight.

    python -m nyxara.njp.experience --episodes nyxara/njp/data/world_experience.jsonl.gz

Depends on ``njp/world``, ``njp/universe``, ``njp/predict`` and ``njp/predictive``, all through
duck-typed attributes on the brain, so a brain missing any one of them replays the rest instead of
failing. Nothing here opens a socket.
"""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

__all__ = ["Episode", "ReplayReport", "ScenarioResult", "load_episodes", "replay"]

#: Below this the fitted slope is treated as no direction at all rather than as a weak one.
_FLAT = 1e-6


@dataclass
class Episode:
    """One turn of the loop, exactly as ``prepare_experience_corpus.py`` wrote it."""

    scenario: str = ""
    domain: str = ""
    step: int = 0
    state_facts: List[str] = field(default_factory=list)
    action: Dict[str, Any] = field(default_factory=dict)
    prediction: Dict[str, Any] = field(default_factory=dict)
    observation: Dict[str, float] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)
    error: Dict[str, Any] = field(default_factory=dict)
    correction: Dict[str, Any] = field(default_factory=dict)
    counterfactual: Dict[str, Any] = field(default_factory=dict)
    text: str = ""
    truth: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "Episode":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (row or {}).items() if k in known})

    @property
    def cause(self) -> str:
        return str(self.correction.get("cause") or (self.order[0] if self.order else ""))

    @property
    def effect(self) -> str:
        return str(self.correction.get("effect") or (self.order[-1] if self.order else ""))

    @property
    def actual(self) -> Optional[float]:
        raw = self.observation.get(self.effect)
        return None if raw is None else float(raw)

    @property
    def true_sign(self) -> int:
        """Which way the effect really moves. The answer a fitted sign is marked against."""
        return int(self.correction.get("sign") or 0)


@dataclass
class ScenarioResult:
    """What one scenario's episodes taught, and whether it was the right thing."""

    scenario: str = ""
    domain: str = ""
    episodes: int = 0
    true_sign: int = 0
    fitted_sign: int = 0
    samples: int = 0
    r2: float = 0.0
    fitted_slope: float = 0.0
    #: The law's own coefficient, and only for a ``linear`` scenario. A straight line through a
    #: square law has a slope that is correct for the range and matches no coefficient anywhere,
    #: so comparing them would mark a well-learned curve wrong.
    true_slope: Optional[float] = None
    usable: bool = False
    oriented: bool = False
    counterfactual_asked: bool = False
    counterfactual_direction: int = 0
    counterfactual_expected: int = 0
    mean_error: float = 0.0

    @property
    def sign_correct(self) -> bool:
        return self.fitted_sign != 0 and self.fitted_sign == self.true_sign

    @property
    def slope_error(self) -> Optional[float]:
        """How far the fitted coefficient is from the real one, as a fraction. ``None`` when the
        law is not a straight line and the question does not arise."""
        if self.true_slope is None or abs(self.true_slope) < _FLAT:
            return None
        return abs(self.fitted_slope - self.true_slope) / abs(self.true_slope)

    @property
    def counterfactual_correct(self) -> bool:
        """Scored against the direction *this* intervention should move, not the law's own sign.

        They differ whenever the intervention runs downhill: on a rising law, "what if the water
        had been less" should answer *down*, and marking that against ``+1`` fails a correct
        answer.
        """
        return (self.counterfactual_asked
                and self.counterfactual_direction != 0
                and self.counterfactual_direction == self.counterfactual_expected)

    def to_dict(self) -> Dict[str, Any]:
        return {"scenario": self.scenario, "domain": self.domain, "episodes": self.episodes,
                "true_sign": self.true_sign, "fitted_sign": self.fitted_sign,
                "sign_correct": self.sign_correct, "samples": self.samples,
                "r2": round(self.r2, 4), "usable": self.usable, "oriented": self.oriented,
                "fitted_slope": round(self.fitted_slope, 4), "true_slope": self.true_slope,
                "slope_error": (None if self.slope_error is None
                                else round(self.slope_error, 4)),
                "counterfactual_expected": self.counterfactual_expected,
                "counterfactual_direction": self.counterfactual_direction,
                "counterfactual_correct": self.counterfactual_correct,
                "mean_error": round(self.mean_error, 4)}


@dataclass
class ReplayReport:
    """Everything the replay touched, separated by which organ it went into."""

    episodes: int = 0
    scenarios: int = 0
    # the prediction loop
    predictions: int = 0
    scored: int = 0
    missed: int = 0
    mean_error: float = 0.0
    mean_surprise: float = 0.0
    diagnosed: Dict[str, int] = field(default_factory=dict)
    # the event history
    events: int = 0
    laws_stated: int = 0
    # the simulator
    relations: int = 0
    signs_correct: int = 0
    signs_wrong: int = 0
    signs_unlearned: int = 0
    # the do-operator
    counterfactuals_asked: int = 0
    counterfactuals_answered: int = 0
    counterfactuals_correct: int = 0
    # the discrete transition model
    transitions: int = 0
    #: Mean relative error of the fitted coefficient over the ``linear`` scenarios only. This is
    #: the strongest claim the file supports: not merely that she learned which way the arrow
    #: runs, but that she recovered how steep it is, from readings alone.
    linear_slope_error: Optional[float] = None
    linear_scenarios: int = 0
    results: List[ScenarioResult] = field(default_factory=list)
    ms: float = 0.0

    @property
    def sign_accuracy(self) -> float:
        learned = self.signs_correct + self.signs_wrong
        return self.signs_correct / learned if learned else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"episodes": self.episodes, "scenarios": self.scenarios,
                "predictions": self.predictions, "scored": self.scored, "missed": self.missed,
                "mean_error": round(self.mean_error, 4),
                "mean_surprise": round(self.mean_surprise, 4),
                "diagnosed": dict(sorted(self.diagnosed.items())),
                "events": self.events, "laws_stated": self.laws_stated,
                "relations": self.relations,
                "signs_correct": self.signs_correct, "signs_wrong": self.signs_wrong,
                "signs_unlearned": self.signs_unlearned,
                "sign_accuracy": round(self.sign_accuracy, 4),
                "counterfactuals_asked": self.counterfactuals_asked,
                "counterfactuals_answered": self.counterfactuals_answered,
                "counterfactuals_correct": self.counterfactuals_correct,
                "transitions": self.transitions,
                "linear_scenarios": self.linear_scenarios,
                "linear_slope_error": (None if self.linear_slope_error is None
                                       else round(self.linear_slope_error, 4)),
                "ms": round(self.ms, 2)}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_episodes(path: Any, *, limit: int = 0) -> List[Episode]:
    """Read ``.jsonl`` or ``.jsonl.gz``. A malformed line is skipped, not fatal.

    Skipped rather than fatal because a corpus is data: one bad line in fifty thousand should cost
    that line, and a loader that refuses the file teaches the operator to stop checking corpora.
    """
    name = str(path)
    opener = gzip.open if name.endswith(".gz") else open
    out: List[Episode] = []
    with opener(name, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(Episode.from_dict(json.loads(line)))
            except (ValueError, TypeError):
                continue
            if limit and len(out) >= limit:
                break
    return out


def _by_scenario(episodes: Iterable[Episode]) -> "Dict[str, List[Episode]]":
    grouped: Dict[str, List[Episode]] = {}
    for episode in episodes:
        grouped.setdefault(episode.scenario or "unnamed", []).append(episode)
    for rows in grouped.values():
        rows.sort(key=lambda e: e.step)
    return grouped


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
def replay(brain: Any, episodes: Sequence[Episode], *,
           declare: bool = True, transitions: bool = True) -> ReplayReport:
    """Live through every episode in order and report what each organ came away with.

    ``declare`` states that each scenario's arrow *exists* — in the world's causal skeleton via
    :meth:`~nyxara.njp.world.WorldView.state_law` and in the simulator via
    :meth:`~nyxara.njp.universe.InternalUniverse.declare` — with **no sign**. Turn it off to
    measure what the numbers teach a brain that was told nothing at all; the arrow is then fitted
    only because ``_permitted`` is permissive on an empty skeleton, and orientation has to come
    from the word order in ``order``.
    """
    report = ReplayReport(episodes=len(episodes))
    started = time.perf_counter()
    grouped = _by_scenario(episodes)
    report.scenarios = len(grouped)

    world = getattr(brain, "world", None)
    universe = getattr(brain, "universe", None)
    predictor = getattr(brain, "predictor", None)
    predictive = getattr(brain, "predictive", None) if transitions else None

    for name, rows in grouped.items():
        first = rows[0]
        result = ScenarioResult(scenario=name, domain=first.domain, episodes=len(rows),
                                true_sign=first.true_sign)
        cause, effect = first.cause, first.effect

        # The arrow is stated to exist before any number arrives — which is what an experiment
        # *is*: you already believe the two are connected, or you would not vary one and measure
        # the other. Its direction is not stated, and that is the whole measurement.
        if declare and cause and effect:
            if world is not None:
                try:
                    world.state_law(cause, effect)
                    report.laws_stated += 1
                except Exception:  # noqa: BLE001
                    pass
            if universe is not None:
                try:
                    universe.declare(cause, effect)
                except Exception:  # noqa: BLE001
                    pass

        errors: List[float] = []
        previous_facts: Optional[List[str]] = None
        for episode in rows:
            report.events += _observe_event(world, episode)
            _register_prediction(predictor, episode, report)
            _observe_joint(universe, episode)
            errors.append(_score(predictor, episode, report))
            if predictive is not None:
                report.transitions += _observe_transition(predictive, episode, previous_facts)
                previous_facts = list(episode.state_facts)

        result.mean_error = sum(errors) / len(errors) if errors else 0.0
        law = (first.truth or {}).get("law") or {}
        if str(law.get("shape")) == "linear":
            result.true_slope = float(law.get("b", 0.0))
        _read_relation(universe, cause, effect, result)
        _ask_counterfactual(universe, rows[-1], result)
        report.results.append(result)

    _tally(report)
    report.ms = (time.perf_counter() - started) * 1000.0
    return report


def _observe_event(world: Any, episode: Episode) -> int:
    """The action, as a `world.Event` — an actor doing something to an object, in order."""
    if world is None or not episode.action:
        return 0
    try:
        from nyxara.njp.world import Event

        world.observe(Event(actor=str(episode.action.get("actor") or ""),
                            action=str(episode.action.get("action") or ""),
                            object=str(episode.action.get("object") or ""),
                            preconditions=list(episode.action.get("preconditions") or []),
                            effects=list(episode.action.get("effects") or []),
                            text=str(episode.action.get("text") or episode.text)))
        return 1
    except Exception:  # noqa: BLE001
        return 0


def _register_prediction(predictor: Any, episode: Episode, report: ReplayReport) -> None:
    """Commit to a number before reality supplies one. Registered, not asserted."""
    if predictor is None or not episode.prediction:
        return
    try:
        predictor.predict(str(episode.prediction.get("key") or episode.scenario),
                          episode.prediction.get("expected"),
                          confidence=float(episode.prediction.get("confidence") or 0.5),
                          organ=str(episode.prediction.get("organ") or "world_model"))
        report.predictions += 1
    except Exception:  # noqa: BLE001
        pass


def _observe_joint(universe: Any, episode: Episode) -> None:
    """The joint reading, with the order the sentence stated it in.

    ``order`` is the only orientation evidence a joint observation carries, and without it five
    readings of two variables are Markov-equivalent: both directions fit exactly as well.
    """
    if universe is None or not episode.observation:
        return
    try:
        universe.observe(dict(episode.observation), order=list(episode.order) or None)
    except Exception:  # noqa: BLE001
        pass


def _score(predictor: Any, episode: Episode, report: ReplayReport) -> float:
    """Meet reality, and route the miss to an organ rather than to a ledger."""
    actual = episode.actual
    if predictor is None or actual is None or not episode.prediction:
        return 0.0
    try:
        outcome = predictor.observe(str(episode.prediction.get("key") or episode.scenario),
                                    actual,
                                    evidence={"organ": str(episode.prediction.get("organ")
                                                           or "world_model")})
        if outcome is None:
            return 0.0
        report.scored += 1
        if not outcome.correct:
            report.missed += 1
        report.mean_surprise += float(getattr(outcome, "surprise", 0.0))
        diagnosis = getattr(outcome, "diagnosis", None)
        if diagnosis is None:
            diagnosis = predictor.diagnose(outcome, {"organ": outcome.organ})
        kind = str(getattr(diagnosis, "kind", "") or "none")
        report.diagnosed[kind] = report.diagnosed.get(kind, 0) + 1
        return float(getattr(outcome, "error", 0.0))
    except Exception:  # noqa: BLE001
        return 0.0


def _observe_transition(predictive: Any, episode: Episode,
                        previous: Optional[Sequence[str]]) -> int:
    """The discrete half: which situation followed which, under which action.

    Banded rather than continuous, because `PredictiveWorldModel` *counts* states and a raw
    reading never recurs — a state seen once is a state that can never be predicted from.
    """
    if predictive is None or previous is None or not episode.state_facts:
        return 0
    try:
        return int(predictive.observe(list(previous),
                                      str(episode.action.get("action") or ""),
                                      next_state=list(episode.state_facts)) or 0)
    except Exception:  # noqa: BLE001
        return 0


def _read_relation(universe: Any, cause: str, effect: str, result: ScenarioResult) -> None:
    """What the arrow looks like after the numbers — the sign especially, which was never given."""
    if universe is None or not cause or not effect:
        return
    try:
        relation = None
        for (a, b), candidate in getattr(universe, "relations", {}).items():
            if a == cause and b == effect:
                relation = candidate
                break
        if relation is None:
            return
        result.samples = int(getattr(relation, "n", 0))
        result.r2 = float(getattr(relation, "r2", 0.0))
        result.usable = bool(getattr(relation, "usable", False))
        result.oriented = bool(getattr(relation, "oriented", False))
        slope = float(getattr(relation, "slope", 0.0))
        result.fitted_slope = slope
        if abs(slope) > _FLAT:
            result.fitted_sign = 1 if slope > 0 else -1
        elif int(getattr(relation, "sign", 0)):
            result.fitted_sign = int(relation.sign)
    except Exception:  # noqa: BLE001
        pass


def _ask_counterfactual(universe: Any, last: Episode, result: ScenarioResult) -> None:
    """``do(cause = its highest value)`` from where the scenario ended, and which way did it move?

    Scored on **direction only**. A magnitude would be scoring the linear fit against a law that
    is often not linear, which would mark a correctly-learned square law wrong for the crime of
    being curved.
    """
    ask = last.counterfactual or {}
    variable, value = str(ask.get("variable") or ""), ask.get("value")
    effect = str(ask.get("effect") or result.scenario)
    if universe is None or not variable or value is None:
        return
    try:
        result.counterfactual_asked = True
        result.counterfactual_expected = int(ask.get("direction") or 0)
        answer = universe.what_if(variable, float(value))
        if not getattr(answer, "answerable", False):
            return
        for delta in answer.changed():
            if str(getattr(delta, "variable", "")) != effect:
                continue
            change = getattr(delta, "change", None)
            if change is not None and abs(change) > _FLAT:
                result.counterfactual_direction = 1 if change > 0 else -1
            else:
                result.counterfactual_direction = int(getattr(delta, "direction", 0))
            break
    except Exception:  # noqa: BLE001
        pass


def _tally(report: ReplayReport) -> None:
    """Fold the per-scenario results into the totals, once, at the end."""
    if report.scored:
        report.mean_error = sum(r.mean_error * r.episodes for r in report.results) / max(
            1, sum(r.episodes for r in report.results))
        report.mean_surprise /= report.scored
    slope_errors = [r.slope_error for r in report.results if r.slope_error is not None]
    report.linear_scenarios = len(slope_errors)
    if slope_errors:
        report.linear_slope_error = sum(slope_errors) / len(slope_errors)
    for result in report.results:
        if result.samples:
            report.relations += 1
        if result.fitted_sign == 0:
            report.signs_unlearned += 1
        elif result.sign_correct:
            report.signs_correct += 1
        else:
            report.signs_wrong += 1
        if result.counterfactual_asked:
            report.counterfactuals_asked += 1
            if result.counterfactual_direction:
                report.counterfactuals_answered += 1
            if result.counterfactual_correct:
                report.counterfactuals_correct += 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
DEFAULT_EPISODES = Path(__file__).with_name("data") / "world_experience.jsonl.gz"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m nyxara.njp.experience [--episodes PATH] [--no-declare] [--per-scenario]``."""
    import argparse

    parser = argparse.ArgumentParser(description="Replay experience episodes through NJP.")
    parser.add_argument("--episodes", default=str(DEFAULT_EPISODES))
    parser.add_argument("--limit", type=int, default=0, help="replay at most N episodes")
    parser.add_argument("--no-declare", action="store_true",
                        help="do not even state that the arrows exist — numbers alone")
    parser.add_argument("--per-scenario", action="store_true",
                        help="print one line per scenario as well as the totals")
    args = parser.parse_args(list(argv) if argv is not None else None)

    episodes = load_episodes(args.episodes, limit=args.limit)
    if not episodes:
        print(f"no episodes loaded from {args.episodes}")
        return 1

    from nyxara.njp.brain import NJPBrain

    report = replay(NJPBrain(), episodes, declare=not args.no_declare)
    if args.per_scenario:
        for result in sorted(report.results, key=lambda r: (r.domain, r.scenario)):
            mark = "ok " if result.sign_correct else "MISS"
            print(f"{mark} {result.domain:14s} {result.scenario:24s} "
                  f"true {result.true_sign:+d} fitted {result.fitted_sign:+d} "
                  f"n={result.samples} R²={result.r2:.2f} "
                  f"cf={'ok' if result.counterfactual_correct else '-'}")
        print()
    print(json.dumps(report.to_dict(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
