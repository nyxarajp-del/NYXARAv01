"""NYXARA · eval/unseen.py — the loop, not the answer (🌑→🔁).

Every other eval in this package asks whether she *knows* something.
:mod:`~nyxara.eval.capability` asks it of a held-out claim, which is a real question and a narrow
one: a system that answers well and cannot recover from being wrong passes it.

This one asks the only question that separates a learner from a lookup table::

    builds a model → predicts → fails → diagnoses the failure
        → revises the model → re-solves → and is better on the NEXT unseen world

**The world is invented, not held out.** Its variables are nonsense terms from
:func:`nyxara.eval.intelligence._vocabulary` — the same generator the intelligence benchmark uses,
for the same reason — and its law is drawn at run time from a seed. Nothing about it can be in any
corpus, so no amount of data changes the score. That is the point: the capability surface moves
when the corpus grows, and this deliberately does not.

**Failing is a scored stage, and that is not a joke.** Stage 3 is *achieved* when the first
prediction misses. A run that happens to be right first time scores **lower** than one that is
wrong, notices, attributes the miss and recovers — because recovery is the capability under test
and a lucky guess is evidence of nothing. The prior she is given disagrees with the world by
construction, exactly as the experience corpus does it, so the miss is arranged rather than hoped
for.

The six stages
--------------
=====  ==============  =========================================================================
stage  organ           achieved when
=====  ==============  =========================================================================
1      ``universe``    an arrow between the two invented variables exists after the first readings
2      ``predictor``   a prediction was registered under this world's key
3      ``predictor``   that prediction **missed** — the loop has something to learn from
4      ``predictor``   ``diagnose`` names an organ rather than returning UNATTRIBUTED
5      ``universe``    after more readings the fitted sign matches the world's real one
6      ``universe``    a **second** invented world reaches the correct sign in fewer readings
=====  ==============  =========================================================================

Stage 6 is the one that matters. Stages 1–5 can be passed by a system that re-learns from scratch
every time; only 6 says the first world left something behind.

**And on this brain it is not measurable, which is the finding.** Measured over eight seeds:
stages 1–5 pass 8/8 — the recovery loop genuinely works, on worlds nothing has ever been told
about — and stage 6 reports ``not measurable`` every time, because the first world is already at
the floor. The floor is *architectural*, not a tuning problem: ``Relation`` fits a line through
whatever it has been given, and a line through two points always has a sign, so
readings-to-correct-sign cannot fall below two and almost never rises above it. Noise was raised
to six times the slope to check, and eight of ten worlds still resolved on the second reading.

The deeper reason is worth stating plainly rather than tuned around: **there is no channel in NJP
today by which one invented world could help another.** Two disjoint variable pairs share no
arrow, no concept and no prior, so any apparent speed-up would be luck. That is precisely what
mechanisms ③–⑩ of the plan exist to build — a shared surprise signal, a compression reward, a
learned prior over laws — and this stage is the number they have to make move. An acceptance test
that already passed would have nothing left to say to them.

    python -m nyxara.eval.unseen --seeds 5
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["Stage", "UnseenReport", "run_once", "run"]

#: How many joint readings a world offers before the model is asked to have learned it.
_READINGS = 8

#: Below this the fitted slope is no direction at all.
_FLAT = 1e-6


@dataclass
class Stage:
    """One stage of the loop: what it asks, whether it happened, and what was seen."""

    name: str = ""
    achieved: bool = False
    detail: str = ""
    #: False when this run could not put the question. Kept out of the denominator entirely, the
    #: way `capability.ProbeResult.score` returns None rather than 0.0: "she failed" and "this run
    #: could not ask" must never share one.
    measurable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"stage": self.name, "achieved": self.achieved,
                "measurable": self.measurable, "detail": self.detail[:160]}


@dataclass
class UnseenReport:
    """One run over one pair of invented worlds."""

    seed: int = 0
    stages: List[Stage] = field(default_factory=list)
    first_world_readings: Optional[int] = None
    second_world_readings: Optional[int] = None
    ms: float = 0.0

    @property
    def score(self) -> float:
        asked = [s for s in self.stages if s.measurable]
        return sum(1 for s in asked if s.achieved) / len(asked) if asked else 0.0

    @property
    def recovered(self) -> bool:
        """Missed, attributed the miss, and ended with the right model. The core of the test."""
        by_name = {s.name: s.achieved for s in self.stages}
        return all(by_name.get(n, False) for n in ("fail", "diagnose", "revise"))

    def to_dict(self) -> Dict[str, Any]:
        return {"seed": self.seed, "score": round(self.score, 4), "recovered": self.recovered,
                "first_world_readings": self.first_world_readings,
                "second_world_readings": self.second_world_readings,
                "stages": [s.to_dict() for s in self.stages], "ms": round(self.ms, 1)}


# --------------------------------------------------------------------------- #
# An invented world
# --------------------------------------------------------------------------- #
@dataclass
class World:
    """Two nonsense variables, a linear law between them, and a prior that disagrees with it."""

    cause: str = ""
    effect: str = ""
    slope: float = 1.0
    intercept: float = 0.0
    belief_slope: float = -1.0
    noise: float = 0.0
    values: Tuple[float, ...] = ()
    _rng: random.Random = field(default_factory=lambda: random.Random(0), repr=False)

    @property
    def sign(self) -> int:
        return 1 if self.slope > 0 else -1

    def truth(self, x: float) -> float:
        """The reading, with noise. Without it two clean points fix the sign every time and
        `readings-to-correct-sign` is the floor in every world — which makes the transfer stage
        compare 2 against 2 and pass for a reason that has nothing to do with transfer."""
        clean = self.intercept + self.slope * x
        return clean + self._rng.uniform(-self.noise, self.noise) * abs(self.slope)

    def belief(self, x: float) -> float:
        return self.intercept + self.belief_slope * x


def _world(rng: random.Random, tag: str) -> World:
    """A world nothing has ever been told about, whose prior is wrong on purpose."""
    from nyxara.eval.intelligence import _vocabulary

    cause, effect = _vocabulary(rng, 2, tag=tag)
    slope = rng.choice([-1, 1]) * rng.uniform(1.5, 4.0)
    return World(cause=cause, effect=effect, slope=slope,
                 intercept=rng.uniform(1.0, 5.0),
                 noise=1.6,
                 _rng=random.Random(rng.randrange(1 << 30)),
                 # The prior runs the other way. A prediction that is already right attributes
                 # nothing and corrects nothing — the generator of the experience corpus refuses
                 # such a scenario outright, and this refuses it the same way.
                 belief_slope=-slope,
                 values=tuple(float(v) for v in range(1, _READINGS + 1)))


def _sign_of(universe: Any, world: World) -> int:
    try:
        for (a, b), relation in getattr(universe, "relations", {}).items():
            if a != world.cause or b != world.effect:
                continue
            slope = float(getattr(relation, "slope", 0.0))
            if abs(slope) > _FLAT:
                return 1 if slope > 0 else -1
            return int(getattr(relation, "sign", 0))
    except Exception:  # noqa: BLE001
        return 0
    return 0


def _teach(brain: Any, world: World, upto: int) -> None:
    """Show the world's first ``upto`` readings, in the order a sentence would state them."""
    universe = getattr(brain, "universe", None)
    if universe is None:
        return
    try:
        universe.declare(world.cause, world.effect)          # the arrow exists; not which way
    except Exception:  # noqa: BLE001
        pass
    for x in world.values[:upto]:
        try:
            universe.observe({world.cause: x, world.effect: world.truth(x)},
                             order=[world.cause, world.effect])
        except Exception:  # noqa: BLE001
            return


def _readings_to_correct_sign(brain: Any, world: World, *, upto: int = 0) -> Optional[int]:
    """How many readings this world needed before the arrow pointed the right way.

    Exposure is **progressive and first**: reading 1, then 2, then 3, checking after each. The
    first version taught the whole world and then re-taught it a reading at a time to count, which
    meant the relation was already fitted from all eight before counting began — so every world
    returned the floor and the transfer stage compared 2 against 2 and passed for nothing.

    A *count*, not a score, and that is deliberate: a second world that needs fewer readings has
    carried something over, and answering faster or more confidently cannot fake it.
    """
    limit = upto or len(world.values)
    universe = getattr(brain, "universe", None)
    if universe is None:
        return None
    try:
        universe.declare(world.cause, world.effect)          # the arrow exists; not which way
    except Exception:  # noqa: BLE001
        pass
    for index, x in enumerate(world.values[:limit], start=1):
        try:
            universe.observe({world.cause: x, world.effect: world.truth(x)},
                             order=[world.cause, world.effect])
        except Exception:  # noqa: BLE001
            return None
        if index >= 2 and _sign_of(universe, world) == world.sign:
            return index
    return None


def run_once(*, seed: int = 0, brain: Any = None) -> UnseenReport:
    """One brain, two invented worlds, six stages."""
    from nyxara.njp.brain import NJPBrain

    started = time.perf_counter()
    rng = random.Random(seed)
    report = UnseenReport(seed=seed)
    brain = brain if brain is not None else NJPBrain()
    first = _world(rng, tag="un")
    universe = getattr(brain, "universe", None)
    predictor = getattr(brain, "predictor", None)

    # 1 · model — did anything get fitted at all? Progressive exposure, and the count that comes
    # out of it is the transfer measurement — nothing is taught twice.
    report.first_world_readings = _readings_to_correct_sign(brain, first)
    fitted = _sign_of(universe, first) != 0
    report.stages.append(Stage("model", fitted,
                               f"{first.cause} → {first.effect} fitted"
                               if fitted else "no arrow after every reading"))

    # 2 · predict — commit to a number before reality supplies one.
    key = f"unseen:{first.effect}"
    nxt = first.values[3]
    expected = first.belief(nxt)
    registered = False
    if predictor is not None:
        try:
            predictor.predict(key, expected, confidence=0.6, organ="world_model")
            registered = True
        except Exception:  # noqa: BLE001
            registered = False
    report.stages.append(Stage("predict", registered,
                               f"expected {expected:.2f} for {first.cause}={nxt}"))

    # 3 · fail — the prior disagrees with the world, so this should miss.
    outcome = None
    if predictor is not None and registered:
        try:
            outcome = predictor.observe(key, first.truth(nxt),
                                        evidence={"organ": "world_model"})
        except Exception:  # noqa: BLE001
            outcome = None
    missed = bool(outcome is not None and not getattr(outcome, "correct", True))
    report.stages.append(Stage("fail", missed,
                               f"error {getattr(outcome, 'error', 0.0):.2f}" if outcome
                               else "nothing was scored"))

    # 4 · diagnose — an organ named, not the UNATTRIBUTED fallback.
    kind = ""
    if outcome is not None:
        try:
            diagnosis = getattr(outcome, "diagnosis", None) or predictor.diagnose(
                outcome, {"organ": getattr(outcome, "organ", "world_model")})
            kind = str(getattr(diagnosis, "kind", "") or "")
        except Exception:  # noqa: BLE001
            kind = ""
    attributed = bool(kind) and kind != "unattributed"
    report.stages.append(Stage("diagnose", attributed, kind or "no diagnosis"))

    # 5 · revise — after every reading, does the arrow point the right way?
    _teach(brain, first, len(first.values))
    revised = _sign_of(universe, first) == first.sign
    report.stages.append(Stage("revise", revised,
                               f"fitted {_sign_of(universe, first):+d}, true {first.sign:+d}"))

    # 6 · transfer — a *second* invented world, and did it come faster?
    second = _world(rng, tag="tr")
    report.second_world_readings = _readings_to_correct_sign(brain, second)
    first_n, second_n = report.first_world_readings, report.second_world_readings
    floor = 2
    if first_n is None or second_n is None:
        report.stages.append(Stage("transfer", False, "a world never reached the right sign",
                                   measurable=True))
    elif first_n <= floor:
        # The first world was learned in the fewest readings the fit can use, so there is no
        # headroom for the second to improve on and the question cannot be put. Scoring this as a
        # pass is how the first version of this file read 5/5 on transfer while comparing 2 to 2.
        report.stages.append(Stage("transfer", False,
                                   f"not measurable: first world already at the floor ({first_n})",
                                   measurable=False))
    else:
        report.stages.append(Stage(
            "transfer", second_n < first_n,
            f"first {first_n} readings, second {second_n}"))

    report.ms = (time.perf_counter() - started) * 1000.0
    return report


def run(*, seeds: int = 5, start: int = 0) -> Dict[str, Any]:
    """Several seeds, because one invented world is an anecdote."""
    runs = [run_once(seed=start + i) for i in range(max(1, int(seeds)))]
    by_stage: Dict[str, int] = {}
    for report in runs:
        for stage in report.stages:
            by_stage[stage.name] = by_stage.get(stage.name, 0) + int(stage.achieved)
    return {
        "runs": len(runs),
        "mean_score": round(sum(r.score for r in runs) / len(runs), 4),
        "recovered": sum(1 for r in runs if r.recovered),
        "by_stage": {name: f"{hit}/{len(runs)}" for name, hit in sorted(by_stage.items())},
        "detail": [r.to_dict() for r in runs],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m nyxara.eval.unseen [--seeds N] [--start N]``."""
    import argparse

    parser = argparse.ArgumentParser(description="The unseen-world recovery loop.")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(run(seeds=args.seeds, start=args.start), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
