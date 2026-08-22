"""NYXARA · njp/universe.py — the Internal Universe and the Counterfactual Engine (🌌, NJP V.04).

A world model that can only answer *"what happened?"* is a log. This is the organ that answers
*"what would happen?"* — including for interventions nobody has ever performed.

**Three things live here, and they are one mechanism seen from three sides.**

1. **A structural model.** Variables (``water``, ``light``, ``growth``) related by *learned
   functions*, not by co-occurrence counts. Each relation is a real incremental least-squares fit
   carrying its own ``R²`` and sample count, so "light drives growth" is a slope she measured and
   can be wrong about, rather than a link that exists because two words appeared near each other.
   :mod:`nyxara.njp.world` supplies the causal *skeleton* — which arrows may exist at all — and
   this supplies the *strength and shape* of each arrow. Neither can do the other's job:
   correlation counts cannot extrapolate, and a regression with no causal skeleton will happily
   fit an arrow pointing the wrong way.

2. **The do-operator.** :meth:`InternalUniverse.intervene` does not condition on a value, it
   *sets* one — severing that variable's own incoming arrows and propagating downstream in
   topological order. That distinction is the entire difference between "plants that get little
   water are small" (a fact about which plants she happened to see) and "if I halve this plant's
   water it will be smaller" (a claim about what she can *do*), and it is why this could not be
   built out of the correlation table.

3. **Imagination, and the surprise that ends it.** :meth:`InternalUniverse.imagine` rolls the
   model forward under an action; :meth:`InternalUniverse.reconcile` scores that rollout against
   what reality actually did and feeds the residual back into the relations that produced it. A
   simulator that is never scored against the world is a daydream, so nothing here reports a
   prediction without also recording that it is now *outstanding*.

**Counterfactuals are graded, not binary.** ``Plant with no water`` and ``plant with half water``
are different questions with different answers, and :class:`CounterfactualResult` carries the
predicted value, the change against the factual baseline, and an honest confidence that decays
with how far outside the observed range the intervention reaches. Extrapolating four times past
anything she has seen is allowed and is reported as the guess it is.

**Experiments are chosen by information gain, not by novelty.**
:class:`ExperimentDesigner` holds a set of rival hypotheses with probabilities and computes, for
each candidate experiment, the *mutual information* between the hypothesis variable and that
experiment's outcome — exactly ``H(prior) − E[H(posterior)]``. The experiment it picks is the one
that most separates the rivals, which is what makes it worth running: an experiment every
hypothesis predicts identically teaches nothing however interesting it looks. Belief update on
the result is plain Bayes.

Pure standard library. No LLM: a simulated universe whose physics is a language model's guess is
that model's universe, not hers.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "Relation",
    "StateDelta",
    "CounterfactualResult",
    "Rollout",
    "Hypothesis",
    "Experiment",
    "InternalUniverse",
    "ExperimentDesigner",
]

_MIN_FIT = 3          # samples before a relation is allowed to claim anything
_MIN_R2 = 0.25        # below this the fit explains too little to propagate through
# What a direction-only answer is worth. Capped well under what any fit earns: she was *told* the
# arrow exists, which is real knowledge and is not a measurement, and a number that reads like one
# would let testimony borrow the credibility of data.
_DIRECTIONAL_CONFIDENCE = 0.35


def _norm(name: Any) -> str:
    return " ".join(str(name or "").strip().lower().split())


def _finite(x: Any) -> Optional[float]:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


# --------------------------------------------------------------------------- #
# One arrow
# --------------------------------------------------------------------------- #
@dataclass
class Relation:
    """``effect = slope · cause + intercept``, fitted online, with its own error bars.

    Sufficient statistics only — no sample list — so a relation costs constant memory however
    long she lives, which is the difference between a model that can run for months and one that
    has to be periodically forgotten.
    """

    cause: str = ""
    effect: str = ""
    n: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_xx: float = 0.0
    sum_yy: float = 0.0
    sum_xy: float = 0.0
    lo: float = math.inf      # observed range of the cause, for honesty about extrapolation
    hi: float = -math.inf
    stated: bool = False      # the skeleton said this arrow may exist, before any data arrived
    # Which way the stated law runs: +1 "more cause, more effect", -1 the reverse, 0 unstated.
    # Only meaningful for an arrow that has never been fitted — once there are numbers, the
    # slope's own sign is the answer and this is not consulted.
    sign: int = 0

    def observe(self, x: float, y: float) -> None:
        self.n += 1
        self.sum_x += x
        self.sum_y += y
        self.sum_xx += x * x
        self.sum_yy += y * y
        self.sum_xy += x * y
        self.lo = min(self.lo, x)
        self.hi = max(self.hi, x)

    @property
    def slope(self) -> float:
        denom = self.n * self.sum_xx - self.sum_x ** 2
        if self.n < 2 or abs(denom) < 1e-12:
            return 0.0
        return (self.n * self.sum_xy - self.sum_x * self.sum_y) / denom

    @property
    def intercept(self) -> float:
        if self.n < 1:
            return 0.0
        return (self.sum_y - self.slope * self.sum_x) / self.n

    @property
    def r2(self) -> float:
        """How much of the effect's variation this arrow accounts for."""
        if self.n < 2:
            return 0.0
        mean_y = self.sum_y / self.n
        ss_tot = self.sum_yy - self.n * mean_y ** 2
        if ss_tot <= 1e-12:
            return 0.0
        slope, intercept = self.slope, self.intercept
        # Residual sum of squares expanded from the sufficient statistics.
        ss_res = (self.sum_yy
                  - 2 * slope * self.sum_xy
                  - 2 * intercept * self.sum_y
                  + slope ** 2 * self.sum_xx
                  + 2 * slope * intercept * self.sum_x
                  + self.n * intercept ** 2)
        return max(0.0, min(1.0, 1.0 - ss_res / ss_tot))

    @property
    def usable(self) -> bool:
        """Fitted well enough to predict a *number*."""
        return self.n >= _MIN_FIT and self.r2 >= _MIN_R2

    @property
    def directional(self) -> bool:
        """Told about, never measured — so it carries a direction and no magnitude.

        This is the state 40 of 40 arrows were in after 1,200 corpus pairs: every one of them
        stated, none of them fitted, `usable_relations` empty, and every counterfactual answered
        "no usable arrow leaves the intervened variables". A causal skeleton the simulator could
        not use for anything.

        A text corpus states laws and almost never states quantities, so waiting for numbers is
        waiting forever. "Fire causes heat" cannot say *how much* heat and does say that removing
        the fire removes the heat, and that is a real answer to a real counterfactual.
        """
        return self.stated and not self.usable and self.sign != 0

    def predict(self, x: float) -> float:
        return self.slope * x + self.intercept

    def confidence_at(self, x: float) -> float:
        """Trust in a prediction at ``x`` — the fit's quality, decayed by how far out it reaches.

        Inside the observed range this is essentially ``R²``. Outside it, confidence falls off
        with the distance in units of the observed span, so an answer about a plant given ten
        times any water she has seen arrives labelled as the extrapolation it is instead of
        borrowing the credibility of the fit.
        """
        if not self.usable:
            return 0.0
        base = self.r2 * min(1.0, self.n / (self.n + 4.0) * 1.5)
        span = self.hi - self.lo
        if span <= 0:
            return base * 0.5
        if self.lo <= x <= self.hi:
            return base
        outside = (x - self.hi) if x > self.hi else (self.lo - x)
        return base / (1.0 + (outside / span))

    def to_dict(self) -> Dict[str, Any]:
        return {"cause": self.cause, "effect": self.effect, "n": self.n,
                "slope": round(self.slope, 5), "intercept": round(self.intercept, 5),
                "r2": round(self.r2, 4), "usable": self.usable, "stated": self.stated,
                "range": [None if self.lo == math.inf else round(self.lo, 4),
                          None if self.hi == -math.inf else round(self.hi, 4)]}


# --------------------------------------------------------------------------- #
# What comes back
# --------------------------------------------------------------------------- #
@dataclass
class StateDelta:
    """One variable's before/after under an intervention."""

    variable: str = ""
    before: Optional[float] = None
    after: Optional[float] = None
    confidence: float = 0.0
    via: List[str] = field(default_factory=list)      # the arrows the change travelled
    # Set when the answer came from a stated law rather than a fitted one. `after` is then None
    # on purpose: the direction is known and the magnitude genuinely is not, and putting a number
    # there would be inventing one.
    direction: int = 0                                # +1 up, -1 down, 0 unknown
    qualitative: bool = False

    @property
    def change(self) -> Optional[float]:
        if self.before is None or self.after is None:
            return None
        return self.after - self.before

    def to_dict(self) -> Dict[str, Any]:
        out = {"variable": self.variable, "before": self.before, "after": self.after,
               "change": None if self.change is None else round(self.change, 5),
               "confidence": round(self.confidence, 4), "via": self.via[:6]}
        if self.qualitative:
            out["direction"] = self.direction
            out["qualitative"] = True
        return out


@dataclass
class CounterfactualResult:
    """"What if X had been different?" — answered, or honestly refused."""

    intervention: Dict[str, float] = field(default_factory=dict)
    factual: Dict[str, float] = field(default_factory=dict)
    predicted: Dict[str, float] = field(default_factory=dict)
    deltas: List[StateDelta] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""

    @property
    def answerable(self) -> bool:
        return bool(self.deltas) and self.confidence > 0.0

    def changed(self, threshold: float = 1e-6) -> List[StateDelta]:
        return [d for d in self.deltas
                if d.change is not None and abs(d.change) > threshold]

    def to_dict(self) -> Dict[str, Any]:
        return {"intervention": {k: round(v, 4) for k, v in self.intervention.items()},
                "predicted": {k: round(v, 4) for k, v in self.predicted.items()},
                "deltas": [d.to_dict() for d in self.deltas[:12]],
                "answerable": self.answerable,
                "confidence": round(self.confidence, 4), "reason": self.reason}


@dataclass
class Rollout:
    """A simulated future, and whether reality later agreed with it."""

    action: str = ""
    states: List[Dict[str, float]] = field(default_factory=list)
    confidence: float = 0.0
    at: float = field(default_factory=time.time)
    scored: bool = False
    error: Optional[float] = None

    @property
    def final(self) -> Dict[str, float]:
        return dict(self.states[-1]) if self.states else {}

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "steps": len(self.states),
                "final": {k: round(v, 4) for k, v in self.final.items()},
                "confidence": round(self.confidence, 4), "scored": self.scored,
                "error": None if self.error is None else round(self.error, 5)}


# --------------------------------------------------------------------------- #
# The universe
# --------------------------------------------------------------------------- #
class InternalUniverse:
    """Her own simulated world: variables, learned arrows between them, and the do-operator."""

    def __init__(self, *, world: Any = None, capacity: int = 512,
                 max_depth: int = 6) -> None:
        self.world = world                  # optional WorldView, for the causal skeleton
        self.capacity = max(16, int(capacity))
        self.max_depth = max(1, int(max_depth))

        self.relations: Dict[Tuple[str, str], Relation] = {}
        self.state: Dict[str, float] = {}   # the last joint observation
        self.observations = 0
        self.rollouts: List[Rollout] = []
        self.reconciled = 0
        self.surprises = 0
        self.total_error = 0.0

    # ---- learning the arrows ---------------------------------------------- #
    def observe(self, state: Dict[str, Any]) -> int:
        """Record one joint observation of the world's variables and refit every live arrow.

        Only pairs that the causal skeleton permits are fitted. Fitting every pair would give a
        model that regresses tomorrow's sunrise on yesterday's mood at ``R² = 0.9`` given enough
        variables, which is not a world model but a machine for finding coincidences.
        """
        clean: Dict[str, float] = {}
        for key, raw in (state or {}).items():
            name, value = _norm(key), _finite(raw)
            if name and value is not None:
                clean[name] = value
        if len(clean) < 1:
            return 0
        self.observations += 1
        fitted = 0
        for cause in clean:
            for effect in clean:
                if cause == effect or not self._permitted(cause, effect):
                    continue
                rel = self.relations.get((cause, effect))
                if rel is None:
                    if len(self.relations) >= self.capacity:
                        continue
                    rel = Relation(cause=cause, effect=effect,
                                   stated=self._stated(cause, effect))
                    self.relations[(cause, effect)] = rel
                rel.observe(clean[cause], clean[effect])
                fitted += 1
        self.state.update(clean)
        return fitted

    def declare(self, cause: str, effect: str, *, sign: int = 0) -> Relation:
        """Assert that an arrow may exist, before any numbers for it have been seen.

        This is how a stated law reaches the simulator. "paudhe ko pani chahiye" gives no
        quantities at all, and so can never be fitted — but it does say which arrow to *watch*,
        and an arrow she is watching gets fitted the moment two numbers for it arrive.
        """
        cause, effect = _norm(cause), _norm(effect)
        key = (cause, effect)
        rel = self.relations.get(key)
        if rel is None:
            rel = Relation(cause=cause, effect=effect, stated=True)
            self.relations[key] = rel
        rel.stated = True
        # A later, more specific statement may say which way an arrow runs; an earlier one that
        # said nothing must not overwrite it back to unknown.
        if sign:
            rel.sign = 1 if sign > 0 else -1
        return rel

    def sync_from_world(self) -> int:
        """Import the causal skeleton from :mod:`nyxara.njp.world`. Idempotent."""
        if self.world is None:
            return 0
        added = 0
        try:
            for link in self.world.links(causal_only=True)[:self.capacity]:
                before = len(self.relations)
                # "X causes Y" is monotone-positive in ordinary use: it is what licenses "no
                # fire, no heat", which is the counterfactual anyone actually asks. A skeleton
                # link carries no sign of its own, so this is the reading applied — and a caller
                # that knows better (grounding saw `decreases`) passes its own.
                self.declare(link.cause, link.effect, sign=1)
                added += int(len(self.relations) > before)
        except Exception:  # noqa: BLE001 — no skeleton just means fewer permitted arrows
            return added
        return added

    def _stated(self, cause: str, effect: str) -> bool:
        rel = self.relations.get((cause, effect))
        return bool(rel is not None and rel.stated)

    @staticmethod
    def _entity(variable: str) -> str:
        """The thing a variable is *about*: ``pani.boils`` → ``pani``.

        The causal skeleton names entities ("aag → garmi") and the data names attributes of
        entities ("pani.boils = 100"). Those are genuinely different levels and neither is wrong,
        but consulting the skeleton with an attribute name means every lookup misses — which is
        how a universe ends up holding nineteen stated arrows and fitting exactly none of them.
        """
        return variable.split(".", 1)[0]

    def _permitted(self, cause: str, effect: str) -> bool:
        """May this arrow exist? The skeleton decides; with no skeleton, everything is allowed.

        The permissive fallback is deliberate and is the honest default for a brain that has just
        woken up with an empty world model: refusing to fit anything until a causal skeleton
        exists would mean she can never acquire one, because the skeleton is partly built from
        what the fits turn up.
        """
        if (cause, effect) in self.relations:
            return True
        if self.world is None:
            return True
        try:
            # Two attributes of the *same* entity always may relate — that is not a causal claim
            # about the world, it is the entity's own state being internally consistent.
            head_cause, head_effect = self._entity(cause), self._entity(effect)
            if head_cause == head_effect:
                return True
            link = self.world.link(head_cause, head_effect)
            if link.refuted:
                return False
            return bool(link.causal or link.stated or link.together)
        except Exception:  # noqa: BLE001
            return True

    # ---- the do-operator --------------------------------------------------- #
    def _parents(self, variable: str) -> List[Relation]:
        return [r for (c, e), r in self.relations.items() if e == variable and r.usable]

    def _children(self, variable: str) -> List[Relation]:
        return [r for (c, e), r in self.relations.items() if c == variable and r.usable]

    def _directional_children(self, variable: str) -> List[Relation]:
        return [r for (c, e), r in self.relations.items() if c == variable and r.directional]

    def directional_relations(self) -> List[Relation]:
        """Arrows she was told about and has never measured. Direction only, no magnitude."""
        return [r for r in self.relations.values() if r.directional]

    def intervene(self, interventions: Dict[str, Any],
                  base: Optional[Dict[str, Any]] = None,
                  stated_direction: Optional[Dict[str, int]] = None) -> CounterfactualResult:
        """``do(X = x)``: set the variables, sever their incoming arrows, propagate downstream.

        Severing is what makes this an intervention rather than an observation. If she merely
        *conditioned* on low water she would also be selecting for the reasons water was low,
        and every one of those reasons would leak into the answer. Setting it cuts those arrows,
        which is precisely the counterfactual the Master asked for.

        ``stated_direction`` carries which way an intervention goes when there is no factual value
        to work it out from. :meth:`_propagate_direction` reads direction by comparing the new
        value against the old one, which is right whenever something has been measured and is
        silent whenever nothing has — and *"halve the water"* says *down* either way. The sentence
        knows something the arithmetic cannot supply, and this is how it gets in. Values here are
        ``-1``/``0``/``+1`` and are used **only** where the factual value is missing; a measured
        baseline always wins, because a claim about a number should come from the number.
        """
        result = CounterfactualResult()
        try:
            factual = {**self.state, **{_norm(k): v for k, v in (base or {}).items()}}
            factual = {k: v for k, v in ((k, _finite(v)) for k, v in factual.items())
                       if v is not None}
            result.factual = dict(factual)

            fixed: Dict[str, float] = {}
            for key, raw in (interventions or {}).items():
                name, value = _norm(key), _finite(raw)
                if name and value is not None:
                    fixed[name] = value
            result.intervention = dict(fixed)
            if not fixed:
                result.reason = "no intervention given"
                return result

            predicted = dict(factual)
            predicted.update(fixed)
            confidences: Dict[str, float] = {name: 1.0 for name in fixed}
            via: Dict[str, List[str]] = {name: ["do"] for name in fixed}

            # Breadth-first along the arrows, depth-capped. A cycle in a learned graph is real
            # (feedback exists) and unbounded propagation around it is not, so the cap is what
            # keeps an honest cyclic model from becoming an infinite regress.
            frontier: List[str] = list(fixed)
            settled: Set[str] = set(fixed)
            for _ in range(self.max_depth):
                nxt: List[str] = []
                for parent in frontier:
                    for rel in self._children(parent):
                        if rel.effect in fixed:
                            continue        # its own arrows are severed
                        x = predicted.get(parent)
                        if x is None:
                            continue
                        estimate = rel.predict(x)
                        conf = rel.confidence_at(x) * confidences.get(parent, 1.0)
                        if conf <= 0.0:
                            continue
                        known = confidences.get(rel.effect, 0.0)
                        if conf <= known:
                            continue        # a better-supported route already explained it
                        predicted[rel.effect] = estimate
                        confidences[rel.effect] = conf
                        via[rel.effect] = via.get(parent, []) + [f"{parent}→{rel.effect}"]
                        if rel.effect not in settled:
                            settled.add(rel.effect)
                            nxt.append(rel.effect)
                frontier = nxt
                if not frontier:
                    break

            result.predicted = predicted
            for name in sorted(settled):
                result.deltas.append(StateDelta(
                    variable=name, before=factual.get(name), after=predicted.get(name),
                    confidence=confidences.get(name, 0.0), via=via.get(name, [])))

            # The qualitative pass, over arrows she was told about and has never measured.
            #
            # Run second and only where the numeric pass reached nothing, so a fitted answer is
            # never displaced by a direction. A stated law cannot say how much and does say which
            # way, and refusing to answer at all — which is what happened for every counterfactual
            # over 1,200 corpus pairs — throws away the only causal knowledge a text corpus
            # actually supplies.
            self._propagate_direction(fixed, factual, settled, result,
                                      stated=stated_direction or {})

            downstream = [d.confidence for d in result.deltas if d.variable not in fixed]
            result.confidence = (sum(downstream) / len(downstream)) if downstream else 0.0
            if not downstream:
                result.reason = "no arrow of any kind leaves the intervened variables"
            elif all(d.qualitative for d in result.deltas if d.variable not in fixed):
                result.reason = "direction only — no arrow on this path has ever been measured"
            return result
        except Exception:  # noqa: BLE001 — a failed simulation answers nothing, never raises
            result.reason = "simulation failed"
            return result

    def _propagate_direction(self, fixed: Dict[str, float], factual: Dict[str, float],
                             settled: Set[str], result: CounterfactualResult,
                             stated: Optional[Dict[str, int]] = None) -> None:
        """Walk the stated skeleton, carrying a sign where there is no coefficient to carry.

        The sign of the intervention matters and is read from the factual state: setting a
        variable *below* where it was pushes a positively-signed effect down, not up. With no
        factual value to compare against, the intervention's own direction is unknown and the
        effect is reported as touched with direction 0 rather than guessed at.

        Confidence is deliberately capped well below what a fit earns. This is structural
        knowledge — she was told the arrow exists — and it should never read like a measurement.
        """
        try:
            told = stated or {}
            direction: Dict[str, int] = {}
            for name, value in fixed.items():
                before = factual.get(name)
                if before is not None:
                    # A measured baseline always decides. A claim about a number comes from the
                    # number, never from what the sentence expected the number to do.
                    direction[name] = 1 if value > before else (-1 if value < before else 0)
                else:
                    # Nothing measured. The sentence said which way, or nobody did.
                    said = int(told.get(name, 0) or 0)
                    direction[name] = 1 if said > 0 else (-1 if said < 0 else 0)

            frontier = list(fixed)
            seen: Set[str] = set(fixed)
            for _ in range(self.max_depth):
                nxt: List[str] = []
                for parent in frontier:
                    for rel in self._directional_children(parent):
                        if rel.effect in fixed or rel.effect in settled or rel.effect in seen:
                            continue        # severed, or the numeric pass already explained it
                        seen.add(rel.effect)
                        moved = direction.get(parent, 0) * rel.sign
                        direction[rel.effect] = moved
                        result.deltas.append(StateDelta(
                            variable=rel.effect, before=factual.get(rel.effect), after=None,
                            confidence=_DIRECTIONAL_CONFIDENCE,
                            via=[f"{parent}→{rel.effect} (stated)"],
                            direction=moved, qualitative=True))
                        nxt.append(rel.effect)
                frontier = nxt
                if not frontier:
                    break
        except Exception:  # noqa: BLE001 — a failed qualitative pass leaves the numeric answer
            return

    def what_if(self, variable: str, value: Any,
                base: Optional[Dict[str, Any]] = None) -> CounterfactualResult:
        """The single-variable convenience form: ``what_if("water", 0.5)``."""
        return self.intervene({variable: value}, base=base)

    def without(self, variable: str,
                base: Optional[Dict[str, Any]] = None) -> CounterfactualResult:
        """"What if there had been no X at all?" — the zero-intervention."""
        return self.intervene({variable: 0.0}, base=base)

    # ---- imagination -------------------------------------------------------- #
    def imagine(self, action: str, interventions: Dict[str, Any], *,
                steps: int = 3) -> Rollout:
        """Roll the model forward under an action, and *record* that a claim is now outstanding.

        Registering the rollout is the part that matters. A simulation nobody ever scores is not
        a prediction, and this module's whole claim to be a world model rather than a daydream
        rests on :meth:`reconcile` being able to find this rollout later and grade it.
        """
        roll = Rollout(action=_norm(action) or "act")
        try:
            state = dict(self.state)
            confidence = 1.0
            for _ in range(max(1, int(steps))):
                out = self.intervene(interventions, base=state)
                if not out.predicted:
                    break
                state = dict(out.predicted)
                roll.states.append(dict(state))
                confidence *= max(0.0, out.confidence)
                # After the first step the intervention is held, not re-applied as news; the
                # dynamics carry it from here.
                interventions = {}
                if out.confidence <= 0.0:
                    break
            roll.confidence = confidence ** (1.0 / max(1, len(roll.states))) if roll.states else 0.0
            self.rollouts.append(roll)
            if len(self.rollouts) > self.capacity:
                self.rollouts = self.rollouts[-self.capacity:]
            return roll
        except Exception:  # noqa: BLE001
            return roll

    def reconcile(self, actual: Dict[str, Any], *, rollout: Optional[Rollout] = None) -> float:
        """Score the newest unscored rollout against reality and learn from the residual.

        Returns the mean absolute error over the variables both the prediction and reality name.
        A surprise — error past the model's own confidence — is counted, because the count of
        times the world contradicted her is the single most useful number about a world model,
        and it is the one a model that never checks itself cannot produce.
        """
        try:
            observed = {}
            for key, raw in (actual or {}).items():
                name, value = _norm(key), _finite(raw)
                if name and value is not None:
                    observed[name] = value
            if not observed:
                return 0.0

            target = rollout
            if target is None:
                target = next((r for r in reversed(self.rollouts) if not r.scored), None)
            # Reality is always worth learning from, whether or not a prediction was outstanding.
            self.observe(observed)
            if target is None:
                return 0.0

            predicted = target.final
            shared = [k for k in predicted if k in observed]
            if not shared:
                target.scored = True
                return 0.0
            errors = [abs(predicted[k] - observed[k]) for k in shared]
            error = sum(errors) / len(errors)
            target.scored = True
            target.error = error
            self.reconciled += 1
            self.total_error += error

            # Surprise: the model was confident and still wrong. Scale-free, so it means the same
            # thing for a temperature and for a probability.
            scale = max(1e-6, sum(abs(observed[k]) for k in shared) / len(shared))
            if error / scale > (1.0 - target.confidence) + 0.25:
                self.surprises += 1
            return error
        except Exception:  # noqa: BLE001
            return 0.0

    # ---- reporting ---------------------------------------------------------- #
    def usable_relations(self) -> List[Relation]:
        return sorted((r for r in self.relations.values() if r.usable),
                      key=lambda r: r.r2, reverse=True)

    def ambiguous(self) -> List[Tuple[str, str]]:
        """Pairs where both directions fit and no skeleton says which way the arrow runs.

        This is Markov equivalence and it is not a bug to be hidden: observational data alone
        genuinely cannot distinguish "water drives growth" from "growth drives water" when the
        two variables move together. The honest consequences are worth stating plainly —
        ``do(growth)`` will propagate into ``water`` for such a pair, because as far as the model
        has been told that arrow is real.

        There are exactly two ways out, and both are elsewhere: a causal skeleton from
        :mod:`nyxara.njp.world` (a stated law fixes the direction), or an actual intervention.
        Reporting the pair is what lets :mod:`nyxara.njp.field` aim an experiment at it.
        """
        out: List[Tuple[str, str]] = []
        for (cause, effect), relation in self.relations.items():
            if not relation.usable or cause > effect:
                continue
            back = self.relations.get((effect, cause))
            if back is not None and back.usable and not (relation.stated ^ back.stated):
                out.append((cause, effect))
        return out

    def stats(self) -> Dict[str, Any]:
        usable = self.usable_relations()
        return {
            "variables": len(self.state),
            "relations": len(self.relations),
            "usable_relations": len(usable),
            # Told about and never measured. Kept apart from `usable_relations` because they
            # answer different questions — how much, versus which way — and a single number
            # covering both would hide that every arrow she has is of the second kind.
            "directional_relations": len(self.directional_relations()),
            "stated_arrows": sum(1 for r in self.relations.values() if r.stated),
            "observations": self.observations,
            "rollouts": len(self.rollouts),
            "scored": sum(1 for r in self.rollouts if r.scored),
            "reconciled": self.reconciled,
            "surprises": self.surprises,
            "ambiguous_pairs": len(self.ambiguous()),
            "mean_error": round(self.total_error / self.reconciled, 5) if self.reconciled else None,
            "strongest": usable[0].to_dict() if usable else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relations": [r.to_dict() | {
                "n": r.n, "sum_x": r.sum_x, "sum_y": r.sum_y, "sum_xx": r.sum_xx,
                "sum_yy": r.sum_yy, "sum_xy": r.sum_xy,
                "lo": None if r.lo == math.inf else r.lo,
                "hi": None if r.hi == -math.inf else r.hi,
            } for r in self.relations.values()],
            "state": dict(self.state), "observations": self.observations,
            "reconciled": self.reconciled, "surprises": self.surprises,
            "total_error": self.total_error,
        }

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.relations = {}
            for row in (d.get("relations") or []):
                cause, effect = _norm(row.get("cause")), _norm(row.get("effect"))
                if not cause or not effect:
                    continue
                rel = Relation(cause=cause, effect=effect, n=int(row.get("n", 0)),
                               sum_x=float(row.get("sum_x", 0.0)),
                               sum_y=float(row.get("sum_y", 0.0)),
                               sum_xx=float(row.get("sum_xx", 0.0)),
                               sum_yy=float(row.get("sum_yy", 0.0)),
                               sum_xy=float(row.get("sum_xy", 0.0)),
                               stated=bool(row.get("stated", False)))
                lo, hi = row.get("lo"), row.get("hi")
                rel.lo = math.inf if lo is None else float(lo)
                rel.hi = -math.inf if hi is None else float(hi)
                self.relations[(cause, effect)] = rel
            self.state = {str(k): float(v) for k, v in (d.get("state") or {}).items()}
            self.observations = int(d.get("observations", 0))
            self.reconciled = int(d.get("reconciled", 0))
            self.surprises = int(d.get("surprises", 0))
            self.total_error = float(d.get("total_error", 0.0))
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Curiosity as information gain
# --------------------------------------------------------------------------- #
@dataclass
class Hypothesis:
    """A rival explanation, its probability, and what it predicts."""

    name: str = ""
    probability: float = 0.5
    # experiment name -> the outcome label this hypothesis predicts for it.
    predictions: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "probability": round(self.probability, 4),
                "predictions": dict(list(self.predictions.items())[:8])}


@dataclass
class Experiment:
    """A candidate action, scored by how much it would actually tell her."""

    name: str = ""
    gain: float = 0.0            # expected information gain, in bits
    outcomes: Dict[str, float] = field(default_factory=dict)   # outcome -> P(outcome)
    separates: List[str] = field(default_factory=list)

    @property
    def decisive(self) -> bool:
        """Does any outcome of this experiment rule a hypothesis out entirely?"""
        return self.gain > 0.0 and len(self.outcomes) > 1

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "gain": round(self.gain, 5),
                "outcomes": {k: round(v, 4) for k, v in self.outcomes.items()},
                "decisive": self.decisive, "separates": self.separates[:8]}


class ExperimentDesigner:
    """Picks the experiment that most reduces uncertainty, then updates belief on the result.

    The whole point is the thing curiosity usually gets wrong. "Interesting" is not a property of
    an experiment; *informative* is, and it is computable. An experiment every live hypothesis
    predicts identically has an expected information gain of exactly zero no matter how novel it
    feels, and this refuses to run it.
    """

    def __init__(self) -> None:
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.designed = 0
        self.run = 0
        self.bits_gained = 0.0
        self.resolved: List[str] = []

    def propose(self, name: str, *, probability: float = 0.5,
                predictions: Optional[Dict[str, str]] = None) -> Optional[Hypothesis]:
        name = _norm(name)
        if not name:
            return None
        hyp = Hypothesis(name=name, probability=max(1e-6, float(probability)),
                         predictions={_norm(k): _norm(v)
                                      for k, v in (predictions or {}).items()})
        self.hypotheses[name] = hyp
        self._renormalise()
        return hyp

    def _renormalise(self) -> None:
        total = sum(h.probability for h in self.hypotheses.values())
        if total <= 0:
            return
        for hyp in self.hypotheses.values():
            hyp.probability /= total

    @staticmethod
    def _entropy(probabilities: Iterable[float]) -> float:
        """Shannon entropy in bits. The uncertainty she is trying to spend an experiment on."""
        total = 0.0
        for p in probabilities:
            if p > 0:
                total -= p * math.log2(p)
        return total

    def prior_entropy(self) -> float:
        return self._entropy(h.probability for h in self.hypotheses.values())

    def evaluate(self, experiment: str) -> Experiment:
        """Expected information gain of one experiment: ``H(prior) − Σ P(o) · H(posterior | o)``.

        Exact, not sampled: the hypothesis set is small and discrete, so the expectation over
        outcomes is a sum rather than an estimate. This is mutual information between the
        hypothesis variable and this experiment's outcome, which is the quantity "how much would
        this teach me" actually names.
        """
        name = _norm(experiment)
        out = Experiment(name=name)
        live = [h for h in self.hypotheses.values() if h.probability > 0]
        if len(live) < 2:
            return out

        # P(outcome) = Σ over hypotheses predicting it. A hypothesis that predicts nothing for
        # this experiment is *silent*, not wrong: it survives every outcome and so contributes to
        # each, which correctly makes the experiment less informative rather than more.
        outcomes: Dict[str, float] = {}
        silent = [h for h in live if name not in h.predictions]
        for hyp in live:
            predicted = hyp.predictions.get(name)
            if predicted is None:
                continue
            outcomes[predicted] = outcomes.get(predicted, 0.0) + hyp.probability
        if not outcomes:
            return out
        for label in list(outcomes):
            outcomes[label] += sum(h.probability for h in silent) / max(1, len(outcomes))

        total = sum(outcomes.values())
        if total <= 0:
            return out
        outcomes = {k: v / total for k, v in outcomes.items()}
        out.outcomes = outcomes

        prior = self.prior_entropy()
        expected = 0.0
        for label, p_outcome in outcomes.items():
            posterior: List[float] = []
            for hyp in live:
                predicted = hyp.predictions.get(name)
                if predicted is None:
                    posterior.append(hyp.probability)          # silent: unrefuted
                elif predicted == label:
                    posterior.append(hyp.probability)
                # a hypothesis that predicted otherwise contributes nothing to this branch
            mass = sum(posterior)
            if mass <= 0:
                continue
            expected += p_outcome * self._entropy(p / mass for p in posterior)
        out.gain = max(0.0, prior - expected)
        out.separates = sorted({h.name for h in live if name in h.predictions})
        return out

    def design(self, experiments: Optional[Sequence[str]] = None) -> Optional[Experiment]:
        """The experiment worth running: highest expected information gain, ties to the simplest.

        With no candidate list, every experiment any hypothesis makes a prediction about is a
        candidate — she can only design an experiment she can already say something about.
        """
        candidates: Set[str] = set()
        for name in (experiments or []):
            if _norm(name):
                candidates.add(_norm(name))
        if not candidates:
            for hyp in self.hypotheses.values():
                candidates.update(hyp.predictions)
        if not candidates:
            return None
        scored = [self.evaluate(name) for name in sorted(candidates)]
        scored = [e for e in scored if e.gain > 1e-9]
        if not scored:
            return None
        self.designed += 1
        return max(scored, key=lambda e: (e.gain, -len(e.name)))

    def observe_result(self, experiment: str, outcome: str) -> Dict[str, float]:
        """Plain Bayes on the result. A hypothesis that predicted otherwise is ruled out.

        Ruling out is the point of having designed for separation: a hypothesis that made a
        commitment and was contradicted is *dead*, not merely less likely, and softening that
        into a small probability decrement is how a belief system becomes unfalsifiable.
        """
        name, label = _norm(experiment), _norm(outcome)
        before = self.prior_entropy()
        for hyp in self.hypotheses.values():
            predicted = hyp.predictions.get(name)
            if predicted is None:
                continue            # silent, so unaffected
            if predicted != label:
                hyp.probability = 0.0
        if not any(h.probability > 0 for h in self.hypotheses.values()):
            # Everything she believed is refuted. That is a real and informative outcome, and
            # flattening back to uniform is the honest response: she now knows only that all of
            # these were wrong, which is what the field's `unknown` gap will act on.
            for hyp in self.hypotheses.values():
                hyp.probability = 1.0
        self._renormalise()
        self.run += 1
        gained = max(0.0, before - self.prior_entropy())
        self.bits_gained += gained
        survivors = {h.name: h.probability for h in self.hypotheses.values() if h.probability > 0}
        if len(survivors) == 1:
            self.resolved.append(next(iter(survivors)))
        return survivors

    def stats(self) -> Dict[str, Any]:
        return {
            "hypotheses": len(self.hypotheses),
            "live": sum(1 for h in self.hypotheses.values() if h.probability > 1e-9),
            "entropy_bits": round(self.prior_entropy(), 4),
            "designed": self.designed, "run": self.run,
            "bits_gained": round(self.bits_gained, 4),
            "resolved": self.resolved[-4:],
            "leader": max((h.to_dict() for h in self.hypotheses.values()),
                          key=lambda h: h["probability"], default=None),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"hypotheses": [h.to_dict() for h in self.hypotheses.values()],
                "designed": self.designed, "run": self.run,
                "bits_gained": self.bits_gained, "resolved": self.resolved[-16:]}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.hypotheses = {}
            for row in (d.get("hypotheses") or []):
                self.propose(row.get("name", ""),
                             probability=float(row.get("probability", 0.5)),
                             predictions=row.get("predictions") or {})
            self.designed = int(d.get("designed", 0))
            self.run = int(d.get("run", 0))
            self.bits_gained = float(d.get("bits_gained", 0.0))
            self.resolved = [str(r) for r in (d.get("resolved") or [])]
        except Exception:  # noqa: BLE001
            pass
