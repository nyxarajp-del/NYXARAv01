#!/usr/bin/env python3
"""Turn the scenarios in ``scripts/experience/*.exp`` into episodes NJP can actually live through.

    scripts/experience/*.exp  ->  world_experience.jsonl.gz

Why this is a different artefact from ``world_knowledge.jsonl.gz``
-----------------------------------------------------------------
That file is testimony: 3,745 claims about how the world is. It makes her *knowledgeable* and it
cannot make her *predictive*, because a fact store has no notion of having been wrong. Every
organ NJP has for learning from being wrong takes something a triple cannot express:

* :meth:`nyxara.njp.universe.InternalUniverse.observe` wants **joint numeric readings**, and fits
  an arrow's slope, sign and R² from them. A triple carries no quantity, so it can never be fitted.
* :meth:`nyxara.njp.predict.PredictionEngine.predict` wants a **claim registered before the fact**,
  scored later by :meth:`~nyxara.njp.predict.PredictionEngine.observe`. Nothing in a fact store is
  ever wrong in a way that produces an :class:`~nyxara.njp.predict.Outcome`.
* :meth:`nyxara.njp.world.WorldView.observe` wants **events in order** with preconditions and
  effects, from which ``links()`` separates a cause from a coincidence by lift.
* :meth:`nyxara.njp.universe.InternalUniverse.intervene` — the do-operator — wants an arrow with a
  *direction*, which is what makes a counterfactual answerable rather than a guess.

So each episode here carries one full turn of the loop, and every field is the argument to one of
those calls:

    state → action → prediction → result → error → correction → refit → counterfactual

**The numbers are correct by construction, not by hand.** Each scenario states a ground-truth law
(``growth = 2.0 + 1.2 × water``) and the generator *runs* it. This is the same argument
:mod:`nyxara.growth.synth_data` makes for verified synthesis: data whose correctness is a property
of how it was produced can be generated at scale, and data typed in by hand cannot. What is stated
by hand is the law, the units and the sentence — the parts a person actually knows.

**The belief is deliberately wrong.** Every scenario carries a second law, the prior she predicts
from, and it disagrees with the truth — wrong slope, wrong sign, or wrong intercept. An episode
where the prediction is already right teaches nothing: there is no error to attribute and no
correction to make. The error these produce is the point of the file.

**The sentence carries the orientation.** ``InternalUniverse.observe`` takes an ``order`` argument
and its docstring says why: five readings of ``water`` beside ``growth`` cannot tell which drives
which — that is Markov equivalence, and no amount of the same data fixes it. *"The plant got 2
litres of water and grew 4 cm"* does say which came first. Each scenario writes that sentence, and
the order of the two variables in it is emitted with every episode.

The scenario format
-------------------
One block per scenario, ``@scenario <name>``, ``key = value`` lines under it, ``#`` comments::

    @scenario plant_growth
    domain      = biology
    actor       = gardener
    action      = water
    object      = plant
    cause       = water | litres | 0 1 2 3 4 5 6
    effect      = growth | cm
    law         = linear a=2.0 b=1.2
    belief      = linear a=8.0 b=-0.5
    precondition= the plant is alive
    consequence = the soil is wet
    sentence    = the gardener gave the plant {cause} litres of water and it grew {effect} cm
    noise       = 0.12

``law`` and ``belief`` take a shape and two coefficients, and the four shapes are the ones the
shipped scenarios need: ``linear`` (a + bx), ``sqrt`` (a + b√x), ``square`` (a + bx²) and
``inverse`` (a + b/x). No expression is ever evaluated — an ``eval`` in a corpus generator is a
way to run whatever a data file says.

Standard library only. Nothing here opens a socket, and the noise is seeded per scenario so a
rebuild of an unchanged file produces an unchanged artefact.

Examples
--------
    python scripts/prepare_experience_corpus.py --out nyxara/njp/data/world_experience.jsonl.gz
    python scripts/prepare_experience_corpus.py --check
    python scripts/prepare_experience_corpus.py --scenario plant_growth --out -
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import random
import re
import sys
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

_EXP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experience")

#: The shapes a law may take. Each is ``f(x) = a + b * g(x)`` for a fixed ``g``, so a scenario
#: states two numbers and never an expression — nothing in this file evaluates a data file's text.
_SHAPES: Dict[str, Any] = {
    "linear":  lambda x: x,
    "sqrt":    lambda x: math.sqrt(x) if x >= 0 else float("nan"),
    "square":  lambda x: x * x,
    "inverse": lambda x: (1.0 / x) if abs(x) > 1e-9 else float("nan"),
}

#: Required keys. A scenario missing any of them is rejected rather than silently half-built.
_REQUIRED = ("domain", "actor", "action", "object", "cause", "effect", "law", "belief", "sentence")

_LAW = re.compile(r"^(?P<shape>[a-z]+)\s+a=(?P<a>-?[\d.]+)\s+b=(?P<b>-?[\d.]+)$")
_CAUSE = re.compile(r"^(?P<name>[a-z_ ]+?)\s*\|\s*(?P<unit>[^|]*?)\s*\|\s*(?P<values>[-\d.\s]+)$")
_EFFECT = re.compile(r"^(?P<name>[a-z_ ]+?)\s*\|\s*(?P<unit>[^|]*)$")


class ScenarioError(ValueError):
    """A scenario that would produce an episode no organ can use, with the file and line."""


class Law:
    """``a + b·g(x)`` — the ground truth of a scenario, or the wrong belief she starts from."""

    def __init__(self, shape: str, a: float, b: float) -> None:
        if shape not in _SHAPES:
            raise ScenarioError(f"unknown law shape {shape!r}. "
                                f"Have: {', '.join(sorted(_SHAPES))}")
        self.shape, self.a, self.b = shape, float(a), float(b)

    def at(self, x: float) -> float:
        return self.a + self.b * _SHAPES[self.shape](float(x))

    def sign_over(self, values: Sequence[float]) -> int:
        """Which way the effect moves across this scenario's range — measured, not read off ``b``.

        ``inverse`` with a positive ``b`` falls as x rises, and reading the coefficient's sign
        would label it ``+1`` and then mark every counterfactual it answers as wrong.
        """
        return self.direction(min(values), max(values))

    def direction(self, current: float, target: float) -> int:
        """Which way the effect moves going from ``current`` to ``target``. Ordered, not sorted.

        :meth:`sign_over` cannot answer this: it reports the law's direction over a *range*, and a
        counterfactual runs from one particular value to another particular value, which may go
        the other way round. Asking "what if the water had been less" about a rising law is a
        prediction that growth goes **down**, and scoring it against the law's ``+1`` marks a
        correct answer wrong.
        """
        move = self.at(target) - self.at(current)
        if abs(move) < 1e-9:
            return 0
        return 1 if move > 0 else -1

    def to_dict(self) -> Dict[str, Any]:
        return {"shape": self.shape, "a": self.a, "b": self.b}


class Scenario:
    """One world, its truth, the prior she is given, and the readings that will be taken."""

    def __init__(self, name: str, fields: Dict[str, str], where: str) -> None:
        self.name, self.where = name, where
        missing = [key for key in _REQUIRED if key not in fields]
        if missing:
            raise ScenarioError(f"{where}: scenario {name!r} is missing: {', '.join(missing)}")
        self.domain = fields["domain"]
        self.actor, self.action, self.object = fields["actor"], fields["action"], fields["object"]

        cause = _CAUSE.match(fields["cause"])
        if cause is None:
            raise ScenarioError(f"{where}: cause must be 'name | unit | v1 v2 …', "
                                f"got {fields['cause']!r}")
        self.cause = cause.group("name").strip()
        self.cause_unit = cause.group("unit").strip()
        self.values = [float(v) for v in cause.group("values").split()]
        if len(self.values) < 3:
            raise ScenarioError(f"{where}: {name!r} needs at least three readings to fit an arrow")

        effect = _EFFECT.match(fields["effect"])
        if effect is None:
            raise ScenarioError(f"{where}: effect must be 'name | unit', got {fields['effect']!r}")
        self.effect = effect.group("name").strip()
        self.effect_unit = effect.group("unit").strip()
        if self.cause == self.effect:
            raise ScenarioError(f"{where}: {name!r} has the same name for cause and effect")

        self.law = self._law(fields["law"], where)
        self.belief = self._law(fields["belief"], where)
        self.sentence = fields["sentence"]
        for slot in ("{cause}", "{effect}"):
            if slot not in self.sentence:
                raise ScenarioError(f"{where}: {name!r} sentence must contain {slot}")
        self.precondition = fields.get("precondition", "")
        self.consequence = fields.get("consequence", "")
        self.noise = float(fields.get("noise", "0.1"))
        self.seed = int(fields.get("seed", "0")) or (abs(hash(name)) % 10_000)

        self.sign = self.law.sign_over(self.values)
        if self.sign == 0:
            raise ScenarioError(f"{where}: {name!r} states a law with no direction — "
                                f"nothing could be learned from it and no counterfactual "
                                f"could be checked against it")
        if self.belief.sign_over(self.values) == self.sign and self._belief_is_close():
            raise ScenarioError(f"{where}: {name!r} believes what is true. An episode with no "
                                f"error attributes nothing and corrects nothing.")

    def _law(self, text: str, where: str) -> Law:
        match = _LAW.match(text.strip())
        if match is None:
            raise ScenarioError(f"{where}: law must be '<shape> a=<float> b=<float>', "
                                f"got {text!r}")
        return Law(match.group("shape"), float(match.group("a")), float(match.group("b")))

    def _belief_is_close(self) -> bool:
        """True when the prior is right enough that no episode would produce a useful error."""
        spread = max(abs(self.law.at(v)) for v in self.values) or 1.0
        worst = max(abs(self.law.at(v) - self.belief.at(v)) for v in self.values)
        return worst < 0.1 * spread

    # ---- the episodes ------------------------------------------------------- #
    def episodes(self) -> Iterator[Dict[str, Any]]:
        rng = random.Random(self.seed)
        scale = max(abs(self.law.at(v)) for v in self.values) or 1.0
        for step, x in enumerate(self.values):
            truth = self.law.at(x)
            actual = round(truth + rng.uniform(-self.noise, self.noise) * scale, 3)
            expected = round(self.belief.at(x), 3)
            key = f"{self.name}:{self.effect}"
            sentence = self.sentence.format(cause=_number(x), effect=_number(actual))
            yield {
                "scenario": self.name,
                "domain": self.domain,
                "step": step,
                # STATE — what held before, discretely, for `predictive.PredictiveWorldModel`
                "state_facts": [f"{self.cause} {_band(x, self.values)}"]
                               + ([self.precondition] if self.precondition else []),
                # ACTION — a `world.Event`, verbatim
                "action": {"actor": self.actor, "action": self.action, "object": self.object,
                           "preconditions": [self.precondition] if self.precondition else [],
                           "effects": [self.consequence] if self.consequence else [],
                           "text": sentence},
                # PREDICTION — a `predict.PredictionEngine.predict` call
                "prediction": {"key": key, "expected": expected, "confidence": 0.6,
                               "organ": "world_model"},
                # RESULT — the joint reading, for `universe.InternalUniverse.observe`
                "observation": {self.cause: x, self.effect: actual},
                "order": [self.cause, self.effect],
                # ERROR — what the prediction was worth, computed rather than asserted
                "error": {"expected": expected, "actual": actual,
                          "absolute": round(abs(expected - actual), 3),
                          "relative": round(abs(expected - actual) / scale, 4),
                          "expected_organ": "world_model"},
                # CORRECTION — the arrow the evidence supports, for `universe.declare`
                "correction": {"cause": self.cause, "effect": self.effect, "sign": self.sign},
                # COUNTERFACTUAL — checked against the ground truth by `intervene`.
                #
                # The target is the *other end* of the range from wherever this episode sat, so the
                # intervention is always a real move. Targeting the maximum, as the first version
                # did, made the last episode's counterfactual `do(x = x)` — a no-op that the
                # simulator still answered, from the gap between the fitted line and one noisy
                # reading, and eight of twenty-seven of those answers came back the wrong way for
                # no better reason than which side of the line the last point fell.
                "counterfactual": {"variable": self.cause, "value": _far_end(x, self.values),
                                   "effect": self.effect, "from": x,
                                   "direction": self.law.direction(x, _far_end(x, self.values))},
                "text": sentence,
                "truth": {"law": self.law.to_dict(), "belief": self.belief.to_dict(),
                          "sign": self.sign, "noiseless": round(truth, 3)},
            }


def _far_end(value: float, values: Sequence[float]) -> float:
    """The end of the range furthest from ``value`` — so ``do(x = target)`` always moves."""
    low, high = min(values), max(values)
    return low if abs(value - low) > abs(value - high) else high


def _number(value: float) -> str:
    """Render a reading the way a person would write it: no trailing ``.0``."""
    return str(int(value)) if float(value).is_integer() else str(round(float(value), 2))


def _band(value: float, values: Sequence[float]) -> str:
    """A discrete band for the symbolic state. `PredictiveWorldModel` counts states, so it needs
    a signature that recurs; a continuous reading never recurs and never becomes learnable."""
    low, high = min(values), max(values)
    if high - low < 1e-9:
        return "steady"
    share = (value - low) / (high - low)
    return "low" if share < 0.34 else ("medium" if share < 0.67 else "high")


# --------------------------------------------------------------------------- #
# Reading the scenarios
# --------------------------------------------------------------------------- #
def scenarios(directory: str = _EXP_DIR, *,
              wanted: Optional[Sequence[str]] = None) -> List[Scenario]:
    """Every scenario in every ``.exp`` file, in file then declaration order."""
    if not os.path.isdir(directory):
        raise ScenarioError(f"no such directory: {directory}")
    out: List[Scenario] = []
    seen: Dict[str, str] = {}
    for name in sorted(n for n in os.listdir(directory) if n.endswith(".exp")):
        path = os.path.join(directory, name)
        for scenario in _parse_file(path):
            if scenario.name in seen:
                raise ScenarioError(f"{scenario.where}: scenario {scenario.name!r} "
                                    f"already defined at {seen[scenario.name]}")
            seen[scenario.name] = scenario.where
            out.append(scenario)
    if wanted:
        chosen = {str(n) for n in wanted}
        unknown = chosen - {s.name for s in out}
        if unknown:
            raise ScenarioError(f"no such scenario: {', '.join(sorted(unknown))}")
        out = [s for s in out if s.name in chosen]
    if not out:
        raise ScenarioError(f"no scenarios in {directory}")
    return out


def _parse_file(path: str) -> Iterator[Scenario]:
    name, fields, started = "", {}, 0
    base = os.path.basename(path)
    with open(path, "rt", encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("@scenario"):
                if name:
                    yield Scenario(name, fields, f"{base}:{started}")
                name, fields, started = line.split(None, 1)[1].strip(), {}, number
                continue
            if "=" not in line:
                raise ScenarioError(f"{base}:{number}: not 'key = value': {line!r}")
            if not name:
                raise ScenarioError(f"{base}:{number}: a field before any @scenario")
            key, _, value = line.partition("=")
            fields[key.strip()] = value.strip()
    if name:
        yield Scenario(name, fields, f"{base}:{started}")


def episodes(chosen: Iterable[Scenario]) -> Iterator[Dict[str, Any]]:
    for scenario in chosen:
        yield from scenario.episodes()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _write(rows: Iterable[Dict[str, Any]], path: str) -> int:
    written = 0
    if path in ("-", "/dev/stdout"):
        for row in rows:
            sys.stdout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
        return written
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    if path.endswith(".gz"):
        # mtime=0 for the same reason as the knowledge corpus: an unchanged source must not
        # produce a changed file, or every rebuild is a diff.
        handle: Any = io.TextIOWrapper(
            gzip.GzipFile(path, "wb", compresslevel=9, mtime=0), encoding="utf-8")
    else:
        handle = open(path, "wt", encoding="utf-8")
    with handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=_EXP_DIR, help=f"scenario directory (default: {_EXP_DIR})")
    ap.add_argument("--scenario", action="append", default=None, metavar="NAME",
                    help="restrict to this scenario (repeatable)")
    ap.add_argument("--out", metavar="PATH", help="write episodes here (.jsonl/.jsonl.gz/-)")
    ap.add_argument("--check", action="store_true", help="parse and report, write nothing")
    args = ap.parse_args(argv)

    try:
        chosen = scenarios(args.dir, wanted=args.scenario)
    except ScenarioError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    rows = list(episodes(chosen))
    by_domain: Dict[str, int] = {}
    for scenario in chosen:
        by_domain[scenario.domain] = by_domain.get(scenario.domain, 0) + 1
    report: Dict[str, Any] = {
        "scenarios": len(chosen),
        "episodes": len(rows),
        "variables": len({s.cause for s in chosen} | {s.effect for s in chosen}),
        "rising": sum(1 for s in chosen if s.sign > 0),
        "falling": sum(1 for s in chosen if s.sign < 0),
        "shapes": {shape: sum(1 for s in chosen if s.law.shape == shape)
                   for shape in sorted({s.law.shape for s in chosen})},
        "domains": dict(sorted(by_domain.items())),
    }
    if not args.check:
        if not args.out:
            sys.stderr.write("nothing to do: pass --out PATH, or --check\n")
            return 2
        report["written"] = _write(rows, args.out)
    print(json.dumps(report, indent=1, ensure_ascii=False))
    if args.out and args.out not in ("-", "/dev/stdout"):
        print(f"\nuse it:  python -m nyxara.njp.experience --episodes {args.out}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
