"""NYXARA · njp/immune.py — she may not buy a benchmark win with a transfer loss (🧿).

Phase 7 already promotes structural changes to her own cognition. It measures a candidate on
held-out data, runs two batteries to check nothing broke, and adopts what strictly wins. What it
never asked is the question that makes the difference between learning and gaming:

    **Where did the win come from?**

A change that raises the number it is scored on while quietly lowering generalisation has not
improved her. It has found the measure. That is the ordinary failure mode of any optimiser given
a metric, and the plan states the rule outright::

    benchmark ↑   AND transfer ↓ / hallucination ↑ / accuracy ↓     →  REJECT

**Nothing new is measured to enforce it.** `evolution._gates` already runs the adversarial battery
and the seven-stage curve, before and after, on fresh brains. Both reports carry per-axis detail
that was being reduced to one boolean and thrown away — the seven-stage report knows its
*generalization* and *transfer* stages separately, and the adversarial report knows concept,
relation, uncertainty and hallucination separately. This reads what is already there.

That restraint is the point rather than a saving. Adding a third battery to catch a candidate
gaming two of them is the move that ends in *"bahut saare AI modules ka collection"*, and it would
have been a worse detector: an axis that costs nothing to check gets checked on every trial.

**A tie is not a fall.** These axes sit near their ceiling, so the honest test is damage, not
improvement — the same rule `_not_worse` already applies to the adversarial battery. What makes
this a *hacking* detector rather than another regression gate is the conjunction: an axis falling
is reported, but only an axis falling **while the candidate's own measure rises** is named a hack.
A change that makes everything slightly worse is simply a bad change, and the benchmark gate
already refuses it.

Pure standard library. Fail-soft: a reading that cannot be taken is absent, and an absent axis
never convicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["Axis", "Reading", "HackVerdict", "RewardHacking"]

#: How far an axis must fall before it counts. Below this is float noise on a saturated battery,
#: and convicting a candidate on noise is its own kind of dishonesty.
_FALL = 1e-9

#: How much the candidate's own measure must have risen for a fall elsewhere to read as a *trade*
#: rather than as a change that was simply bad everywhere.
_RISE = 1e-9


class Axis:
    """The named directions. ``higher`` says which way is good — the only thing easy to get wrong."""

    GENERALIZATION = "generalization"
    TRANSFER = "transfer"
    CONCEPT = "concept"
    RELATION = "relation"
    UNCERTAINTY = "uncertainty"
    HALLUCINATION = "hallucination"

    #: True where up is better. `hallucination` is the one that is not, and stating it as data
    #: rather than as an `if` is what keeps a later axis from being added on the wrong side.
    HIGHER_IS_BETTER: Dict[str, bool] = {
        GENERALIZATION: True, TRANSFER: True, CONCEPT: True,
        RELATION: True, UNCERTAINTY: True, HALLUCINATION: False,
    }

    ALL: Tuple[str, ...] = (GENERALIZATION, TRANSFER, CONCEPT,
                            RELATION, UNCERTAINTY, HALLUCINATION)


@dataclass
class Reading:
    """One measurement of every axis. ``None`` where a battery could not speak to it."""

    values: Dict[str, Optional[float]] = dc_field(default_factory=dict)

    def get(self, axis: str) -> Optional[float]:
        got = self.values.get(axis)
        return None if got is None else float(got)

    @property
    def measured(self) -> List[str]:
        return [a for a in Axis.ALL if self.values.get(a) is not None]

    def to_dict(self) -> Dict[str, Any]:
        return {a: (None if v is None else round(float(v), 5))
                for a, v in self.values.items()}


@dataclass
class HackVerdict:
    """What the comparison found, with the numbers rather than a label."""

    hacked: bool = False
    axis: str = ""
    before: Optional[float] = None
    after: Optional[float] = None
    gain: float = 0.0
    fell: List[str] = dc_field(default_factory=list)
    why: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"hacked": self.hacked, "axis": self.axis,
                "before": None if self.before is None else round(self.before, 5),
                "after": None if self.after is None else round(self.after, 5),
                "gain": round(self.gain, 5), "fell": list(self.fell), "why": self.why[:240]}


class RewardHacking:
    """Reads the axes the batteries already produced, and refuses a win bought elsewhere."""

    def __init__(self, *, fall: float = _FALL, rise: float = _RISE) -> None:
        self.fall = float(fall)
        self.rise = float(rise)
        self.checks = 0
        self.caught = 0
        self.by_axis: Dict[str, int] = {}
        self.history: List[HackVerdict] = []

    # ---- reading what has already been measured ----------------------------- #
    @staticmethod
    def read(adversarial: Any = None, intelligence: Any = None) -> Reading:
        """Pull the axes out of the two reports `evolution._gates` already has in hand."""
        out = Reading()
        for axis in (Axis.CONCEPT, Axis.RELATION, Axis.UNCERTAINTY, Axis.HALLUCINATION):
            out.values[axis] = _number(getattr(adversarial, axis, None))
        for axis in (Axis.GENERALIZATION, Axis.TRANSFER):
            out.values[axis] = _stage_score(intelligence, axis)
        return out

    # ---- the check ---------------------------------------------------------- #
    def check(self, before: Reading, after: Reading, *, gain: float) -> HackVerdict:
        """``gain`` is how much the candidate's *own* measure moved. That is the whole hinge.

        An axis that falls is worth reporting either way, but only a fall paired with a rise in
        the thing the candidate is scored on is a *trade* — and a trade is what this refuses. A
        candidate that made everything worse is a bad candidate, and the benchmark gate ahead of
        this one has already turned it away.
        """
        out = HackVerdict(gain=float(gain))
        self.checks += 1
        try:
            worst: Optional[Tuple[float, str, float, float]] = None
            for axis in Axis.ALL:
                old, new = before.get(axis), after.get(axis)
                if old is None or new is None:
                    continue                       # an axis nobody measured never convicts
                # One subtraction, oriented by the table rather than by an `if` per axis.
                drop = (old - new) if Axis.HIGHER_IS_BETTER.get(axis, True) else (new - old)
                if drop <= self.fall:
                    continue
                out.fell.append(axis)
                if worst is None or drop > worst[0]:
                    worst = (drop, axis, old, new)

            if worst is None:
                out.why = f"no axis fell ({len(after.measured)} measured)"
                return out
            drop, axis, old, new = worst
            if gain <= self.rise:
                out.axis, out.before, out.after = axis, old, new
                out.why = (f"{axis} fell {old:.4f} → {new:.4f}, and the candidate's own measure "
                           f"did not rise — a bad change, not a traded one")
                return out

            out.hacked = True
            out.axis, out.before, out.after = axis, old, new
            out.why = (f"its own measure rose {gain:+.4f} while {axis} fell "
                       f"{old:.4f} → {new:.4f} — the win was bought, not earned")
            self.caught += 1
            self.by_axis[axis] = self.by_axis.get(axis, 0) + 1
            return out
        except Exception:  # noqa: BLE001 — a check that cannot run convicts nobody
            out.why = "the check could not be run"
            return out
        finally:
            self.history.append(out)
            del self.history[:-64]

    def stats(self) -> Dict[str, Any]:
        last = self.history[-1] if self.history else None
        return {"checks": self.checks, "caught": self.caught,
                "by_axis": dict(self.by_axis),
                "last": last.to_dict() if last is not None else None}


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stage_score(report: Any, name: str) -> Optional[float]:
    """One stage's score out of a seven-stage report, by name.

    By name rather than by position: the curve's stages are ordered by what they build on, and a
    stage inserted in the middle would silently re-point an index at a different measurement.

    Through the report's own :meth:`by_name` rather than a hand-rolled scan. The first version
    scanned for ``stage.name`` — the field is ``stage.stage`` — so it matched nothing and the two
    axes that matter most, *generalization* and *transfer*, came back absent. An absent axis never
    convicts, so the detector was quietly checking four axes and reporting "no axis fell" with
    perfect confidence. A lookup that fails by returning `None` is the worst kind here.
    """
    try:
        lookup = getattr(report, "by_name", None)
        stage = lookup(name) if callable(lookup) else None
        if stage is None:
            for candidate in (getattr(report, "stages", None) or ()):
                if str(getattr(candidate, "stage", "") or "") == name:
                    stage = candidate
                    break
        return _number(getattr(stage, "score", None)) if stage is not None else None
    except Exception:  # noqa: BLE001
        return None
