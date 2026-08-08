"""NYXARA · nyx/episteme.py — finding things out for herself (🔬, L-EPISTEME).

Everything else in NYX V.01 works on what she was told or what she already had. This module is
where she goes and *finds out*: she picks a sandbox she does not understand, varies its inputs,
measures what comes back, and tries to derive the law behind it. Nobody hands her the answer,
and nobody tells her whether she got it right — she checks that herself, on data she withheld.

The loop is the one science actually uses:

1. **probe** — vary the inputs of a simulator and record what it does;
2. **derive** — search for a law that accounts for the measurements
   (:mod:`nyxara.growth.law_discovery`, real symbolic regression);
3. **withhold and check** — the law must predict trials it never saw. This is the whole gate;
   a curve that fits the points it was fitted to has established nothing;
4. **promote or hold** — only a law that survives the held-out check becomes a *fact* she will
   answer from. Anything else is kept as a **conjecture**, labelled, and never asserted.

What is genuinely real here: the measurements are real (the gas box is a particle simulation;
the electrostatic force is inferred kinematically, not looked up), the regression is real, and
the held-out validation is real. In testing she recovered ``force = 4·(n·T)`` from her own
measurements — the ideal gas law — having been told nothing about it.

Honest framing, and this one matters:

* She is discovering **the laws of her own simulators**, not new physics. The simulator already
  embodies known physics; what is genuine is that *she* did not know it, probed, and derived it.
  Novel-to-her is a real and useful thing, and it is not the same as novel-to-anyone.
* **Abstention is a result.** The discovery engine refuses to publish a law that does not
  generalise, and this module keeps that refusal visible rather than reaching for something to
  say. In testing the electrostatic run found nothing and reported why — that is the system
  working, not failing.
* A conjecture is **never** promoted on plausibility. Only the held-out check promotes.

Pure standard library plus what it composes. Fail-soft: a failed investigation finds nothing.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["Knob", "Experiment", "Finding", "AutonomousDiscovery", "default_experiments"]

ProbeFn = Callable[..., float]


@dataclass
class Knob:
    """One input she can vary, and the range she may vary it over."""

    name: str
    low: float
    high: float
    integer: bool = False

    def draw(self, rng: random.Random) -> float:
        value = rng.uniform(float(self.low), float(self.high))
        return float(int(round(value))) if self.integer else value


@dataclass
class Experiment:
    """Something she can poke at and measure — a sandbox whose law she does not know."""

    name: str
    probe: ProbeFn
    knobs: List[Knob]
    measures: str = "y"
    domain: str = "sandbox"

    def run(self, rng: random.Random) -> Optional[Tuple[List[float], float]]:
        """One trial: pick inputs, measure the output. ``None`` when the probe failed."""
        try:
            values = [k.draw(rng) for k in self.knobs]
            result = float(self.probe(**{k.name: v for k, v in zip(self.knobs, values)}))
            return values, result
        except Exception:  # noqa: BLE001 — a failed trial is a missing row, not a crash
            return None


@dataclass
class Finding:
    """What an investigation established — or honestly did not."""

    experiment: str = ""
    status: str = "abstained"        # promoted | conjecture | abstained | failed
    expression: str = ""
    reason: str = ""
    trials: int = 0
    holdout: int = 0
    holdout_error: Optional[float] = None   # mean relative error on data it never saw
    elapsed_ms: float = 0.0

    @property
    def promoted(self) -> bool:
        return self.status == "promoted"

    def to_dict(self) -> Dict[str, Any]:
        return {"experiment": self.experiment, "status": self.status,
                "expression": self.expression, "reason": self.reason,
                "trials": self.trials, "holdout": self.holdout,
                "holdout_error": (None if self.holdout_error is None
                                  else round(self.holdout_error, 6)),
                "elapsed_ms": round(self.elapsed_ms, 2)}


def default_experiments() -> List[Experiment]:
    """Sandboxes that need no network and no configuration — she can always run these.

    Both *measure* rather than look up: the gas box simulates particles and counts momentum at
    the wall; the electrostatic world releases a test charge and infers force from the motion.
    """
    out: List[Experiment] = []
    try:
        from nyxara.sim.thermo_world import GasBox
        box = GasBox(seed=3)

        def _gas(n: float, temperature: float) -> float:
            return box.measure_wall_force(n_particles=int(n), temperature=float(temperature),
                                          steps=600)

        out.append(Experiment(name="gas-box", probe=_gas, measures="force", domain="thermo",
                              knobs=[Knob("n", 20, 120, integer=True),
                                     Knob("temperature", 150.0, 600.0)]))
    except Exception:  # noqa: BLE001 — a missing sandbox is simply one fewer experiment
        pass
    try:
        from nyxara.sim.em_world import ElectrostaticWorld
        world = ElectrostaticWorld()

        def _em(q: float, r: float) -> float:
            return world.measure_force(q_source=float(q), q_test=1e-6, r=float(r))

        out.append(Experiment(name="electrostatics", probe=_em, measures="force", domain="em",
                              knobs=[Knob("q", 1e-6, 9e-6), Knob("r", 0.05, 0.4)]))
    except Exception:  # noqa: BLE001
        pass
    return out


class AutonomousDiscovery:
    """Probes a sandbox she does not understand and keeps only what survives a held-out check."""

    def __init__(self, brain: Any = None, *, experiments: Optional[Sequence[Experiment]] = None,
                 trials: int = 18, holdout_fraction: float = 0.3,
                 max_holdout_error: float = 0.05, seed: int = 42,
                 budget_ms: float = 4000.0) -> None:
        self.brain = brain
        self.experiments: List[Experiment] = list(
            experiments if experiments is not None else default_experiments())
        self.trials = max(6, int(trials))
        self.holdout_fraction = min(0.6, max(0.1, float(holdout_fraction)))
        # A law must predict unseen trials this closely to be called a fact rather than a fit.
        self.max_holdout_error = max(0.0, float(max_holdout_error))
        self.budget_ms = max(1.0, float(budget_ms))
        self.rng = random.Random(seed)
        self.seed = int(seed)

        self.findings: List[Finding] = []
        self.promoted: Dict[str, str] = {}       # experiment → the law she now holds as fact
        self.conjectures: Dict[str, str] = {}
        self._next = 0

    def available(self) -> bool:
        """Honest capability report — False when there is nothing she can experiment on."""
        return bool(self.experiments)

    # ---- one investigation ------------------------------------------------- #
    def investigate(self, experiment: Optional[Experiment] = None) -> Finding:
        """Probe, derive, and check against data she withheld. The check is the whole point."""
        started = time.monotonic()
        out = Finding()
        try:
            experiment = experiment or self._pick()
            if experiment is None:
                out.reason = "there is nothing she can experiment on"
                return out
            out.experiment = experiment.name

            rows = [r for r in (experiment.run(self.rng) for _ in range(self.trials))
                    if r is not None]
            if len(rows) < 6:
                out.status, out.reason = "failed", "the sandbox did not yield enough measurements"
                return out

            cut = max(2, int(len(rows) * self.holdout_fraction))
            train, held = rows[:-cut], rows[-cut:]
            out.trials, out.holdout = len(train), len(held)

            law = self._derive(experiment, train)
            if law is None:
                out.reason = ("no law generalised beyond the measurements — she abstained "
                              "rather than fitting the noise")
                return out
            out.expression = str(getattr(law, "expression", "") or "")

            out.holdout_error = self._check(experiment, law, held)
            if out.holdout_error is None:
                out.status = "conjecture"
                out.reason = "the law could not be checked against the withheld trials"
            elif out.holdout_error <= self.max_holdout_error:
                out.status = "promoted"
                out.reason = (f"predicted {out.holdout} withheld trials to within "
                              f"{out.holdout_error:.2%}")
            else:
                out.status = "conjecture"
                out.reason = (f"fitted the measurements but missed the withheld trials by "
                              f"{out.holdout_error:.2%}")
            self._record(experiment, out)
            return out
        except Exception:  # noqa: BLE001 — a failed investigation finds nothing
            out.status, out.reason = "failed", "the investigation was abandoned"
            return out
        finally:
            out.elapsed_ms = (time.monotonic() - started) * 1000.0
            self.findings.append(out)
            del self.findings[:-32]

    def _pick(self) -> Optional[Experiment]:
        """Round-robin, so she works through what she has rather than fixating on one."""
        if not self.experiments:
            return None
        experiment = self.experiments[self._next % len(self.experiments)]
        self._next += 1
        return experiment

    def _derive(self, experiment: Experiment, rows: Sequence[Tuple[List[float], float]]) -> Any:
        try:
            from nyxara.growth.law_discovery import LawDiscoveryEngine
            engine = LawDiscoveryEngine(seed=self.seed)
            report = engine.discover_from_data(
                [r[0] for r in rows], [r[1] for r in rows],
                var_names=[k.name for k in experiment.knobs],
                target_name=experiment.measures, source=experiment.name)
            discoveries = list(getattr(report, "discoveries", []) or [])
            if not discoveries:
                return None
            return getattr(discoveries[0], "law", discoveries[0])
        except Exception:  # noqa: BLE001
            return None

    def _check(self, experiment: Experiment, law: Any,
               held: Sequence[Tuple[List[float], float]]) -> Optional[float]:
        """Mean relative error on trials the law never saw. This is what promotion turns on."""
        try:
            if not held:
                return None
            # Law.predict takes *columns*, not rows, and returns one value per row.
            columns = [[row[0][i] for row in held] for i in range(len(experiment.knobs))]
            predictions = law.predict(columns)
            if not predictions or len(predictions) != len(held):
                return None
            errors: List[float] = []
            for (_values, actual), predicted in zip(held, predictions):
                if predicted is None:
                    continue
                scale = max(abs(float(actual)), 1e-9)
                errors.append(abs(float(predicted) - float(actual)) / scale)
            return sum(errors) / len(errors) if errors else None
        except Exception:  # noqa: BLE001
            return None

    def _record(self, experiment: Experiment, finding: Finding) -> None:
        """A validated law becomes something she will answer from; a conjecture stays labelled."""
        brain = self.brain
        try:
            if finding.status == "promoted":
                self.promoted[experiment.name] = finding.expression
                self.conjectures.pop(experiment.name, None)
                if brain is not None:
                    brain.remember(f"law-{experiment.name}",
                                   f"in {experiment.domain}, {finding.expression}",
                                   kind="fact")
            elif finding.status == "conjecture":
                self.conjectures[experiment.name] = finding.expression
                if brain is not None:
                    brain.remember(f"conjecture-{experiment.name}",
                                   f"possibly, in {experiment.domain}, {finding.expression}",
                                   kind="conjecture")
        except Exception:  # noqa: BLE001 — recording is best-effort
            pass

    # ---- the clock ---------------------------------------------------------- #
    def beat(self, *, oversight: Any = None) -> Optional[Finding]:
        """One investigation, when permitted. Stops dead when oversight does."""
        try:
            if oversight is not None:
                gate = getattr(oversight, "gate", None)
                permitted = bool(gate()) if callable(gate) else bool(gate)
                if not permitted:
                    return None
            if not self.experiments:
                return None
            return self.investigate()
        except Exception:  # noqa: BLE001
            return None

    # ---- introspection ------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        try:
            return {"available": self.available(),
                    "experiments": [e.name for e in self.experiments],
                    "investigations": len(self.findings),
                    "promoted": dict(self.promoted),
                    "conjectures": dict(self.conjectures),
                    "abstained": sum(1 for f in self.findings if f.status == "abstained"),
                    "max_holdout_error": self.max_holdout_error,
                    "last": self.findings[-1].to_dict() if self.findings else None}
        except Exception:  # noqa: BLE001
            return {"available": False}

    def to_dict(self) -> Dict[str, Any]:
        return {"promoted": dict(self.promoted), "conjectures": dict(self.conjectures),
                "next": self._next}

    def load_dict(self, data: Any) -> bool:
        try:
            if not isinstance(data, dict):
                return False
            self.promoted = {str(k): str(v) for k, v in (data.get("promoted") or {}).items()}
            self.conjectures = {str(k): str(v)
                                for k, v in (data.get("conjectures") or {}).items()}
            self._next = int(data.get("next", 0))
            return True
        except Exception:  # noqa: BLE001
            return False
