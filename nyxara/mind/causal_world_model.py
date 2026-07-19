"""NYXARA · mind/causal_world_model.py — causation, not just correlation (★).

A pattern-matcher learns *"A and B are often seen together."* That is **correlation**,
and it is treacherous: it confuses confounding (a hidden common cause makes A and B move
together), reverse causation (B actually drives A), and plain coincidence for a real
mechanism. To *act* well — to plan, to explain, to imagine "what if I had not done that?"
— NYXARA must learn the stronger thing:

    A hua  ──↓ isliye──▶  B hua          (A happened, *therefore* B happened)

This module learns a **causal world model** from a *time-ordered stream* of events and,
crucially, **separates causation from mere correlation** using several convergent criteria
— no single one is trusted alone:

1. **Temporal precedence** — a cause precedes its effect. Within a causal **window** W,
   how often is A followed by B? (reuses the timing machinery of
   :class:`~nyxara.mind.temporal.TemporalReasoner`'s idea of precedence)
2. **Contingency (ΔP)** — P(B soon after A) − P(B soon after *not*-A): does B actually
   *depend* on A, or does it happen just as much anyway? (kills coincidence)
3. **Direction** — A→B is only kept if the forward association meaningfully beats the
   reverse one; if B reliably comes first, the arrow is **reverse**.
4. **Confounder screening (conditional independence)** — the heart of correlation-vs-
   causation. If A and B are associated but the association *vanishes once we remove the
   A-events that were themselves triggered by some earlier C* (and C precedes both, so it
   is a common cause, **not** a mediator A→C→B), then C is a **confounder** and A→B is not
   causal. This is "ice-cream sales correlate with drownings — because: summer".
5. **Intervention (the do-operator, gold standard)** — when NYXARA *acted* to bring A
   about (an intervention) versus merely *observed* A, did B follow? Interventional
   evidence is the strongest signal there is, and NYXARA generates it naturally: every
   turn she *does* something, and the kernel records it.

The learned structure is a directed, weighted **causal graph** with honest confidence
(low evidence → low confidence; confounded links demoted — never hallucinated certainty).
On top of it sit the questions that matter:

* :meth:`why` — the genuine causes of an effect ("why did B happen?"), ranked, with evidence.
* :meth:`is_causal` — a verdict for one pair: *causal / correlational / confounded /
  coincidental / reverse*, with the reason.
* :meth:`effects_of` / :meth:`predict_effects` — forward ``do``-propagation.
* :meth:`counterfactual` — "agar A na hota, to B hota?" (had A not happened, would B?).

Forward ``do``-propagation and counterfactual contrasts **reuse**
:class:`~nyxara.mind.strategies.CausalModel`. This module's own job is the one neither
``CausalModel`` (which is *handed* its graph) nor :mod:`mind.world_model` (which learns
state *dynamics*) does: **discovering** the graph — telling causation apart from correlation.

Pure standard library; pairs with :mod:`mind.world_model` and feeds :mod:`mind.explain`
and :mod:`planning`.
"""

from __future__ import annotations

import bisect
import itertools
import json
import os
import random
import time
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "Event",
    "CausalLink",
    "CausalVerdict",
    "Counterfactual",
    "FunctionalCausalMechanism",
    "CausalWorldModel",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _any_in(sorted_times: List[float], lo: float, hi: float) -> bool:
    """Is there any timestamp in the half-open interval ``(lo, hi]``? (binary search)."""
    i = bisect.bisect_right(sorted_times, lo)
    return i < len(sorted_times) and sorted_times[i] <= hi


# Verdicts a single (cause, effect) pair can receive.
CAUSAL = "causal"                 # genuine: A → B
CORRELATIONAL = "correlational"   # they move together, but direction/cause not established
CONFOUNDED = "confounded"         # a hidden common cause explains the link
COINCIDENTAL = "coincidental"     # co-occur with no real dependence (ΔP ≈ 0 / too little support)
REVERSE = "reverse"               # the arrow points the other way (B → A)


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #
@dataclass
class Event:
    """One thing that happened, when it happened, and whether NYXARA *did* it herself."""

    label: str
    at: float = field(default_factory=time.time)
    intervention: bool = False     # True ⇒ she brought it about (a ``do``), not merely observed
    value: Optional[float] = None  # optional magnitude — feeds functional (how-much) mechanisms

    def to_dict(self) -> Dict[str, Any]:
        d = {"label": self.label, "at": round(self.at, 4), "do": self.intervention}
        if self.value is not None:
            d["value"] = round(self.value, 6)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Event":
        raw = d.get("value")
        return cls(label=d["label"], at=float(d.get("at", 0.0)),
                   intervention=bool(d.get("do", False)),
                   value=float(raw) if raw is not None else None)


@dataclass
class CausalLink:
    """A directed link the model believes in, with the evidence that earned it."""

    cause: str
    effect: str
    strength: float                 # signed effect size (interventional if available, else ΔP)
    confidence: float               # [0,1] — honest trust given the convergent evidence
    verdict: str                    # CAUSAL / CORRELATIONAL / CONFOUNDED / COINCIDENTAL / REVERSE
    lag: float = 0.0                # typical delay cause→effect (seconds)
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_causal(self) -> bool:
        return self.verdict == CAUSAL

    def describe(self) -> str:
        if self.verdict == CAUSAL:
            head = f"{self.cause!r} causes {self.effect!r}"
        elif self.verdict == CONFOUNDED:
            c = self.evidence.get("confounder")
            head = f"{self.cause!r} & {self.effect!r} only correlate (confounded by {c!r})"
        elif self.verdict == REVERSE:
            head = f"it is {self.effect!r} that causes {self.cause!r}, not the reverse"
        elif self.verdict == COINCIDENTAL:
            head = f"{self.cause!r} & {self.effect!r} co-occur by coincidence"
        else:
            head = f"{self.cause!r} & {self.effect!r} correlate (cause unproven)"
        return f"{head} [conf {self.confidence:.0%}, strength {self.strength:+.2f}]"

    def to_dict(self) -> Dict[str, Any]:
        return {"cause": self.cause, "effect": self.effect,
                "strength": round(self.strength, 4), "confidence": round(self.confidence, 4),
                "verdict": self.verdict, "lag": round(self.lag, 4), "evidence": self.evidence}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CausalLink":
        return cls(cause=d["cause"], effect=d["effect"], strength=float(d["strength"]),
                   confidence=float(d["confidence"]), verdict=d["verdict"],
                   lag=float(d.get("lag", 0.0)), evidence=dict(d.get("evidence", {})))


@dataclass
class CausalVerdict:
    """The answer to "does A cause B, or do they just co-occur?"."""

    cause: str
    effect: str
    verdict: str
    confidence: float
    reason: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_causal(self) -> bool:
        return self.verdict == CAUSAL

    def to_dict(self) -> Dict[str, Any]:
        return {"cause": self.cause, "effect": self.effect, "verdict": self.verdict,
                "confidence": round(self.confidence, 4), "reason": self.reason,
                "evidence": self.evidence}


class FunctionalCausalMechanism:
    """A *learned functional* causal relation ``value_B ≈ f(value_A)`` for one edge.

    The graph above answers "does A cause B?"; this answers "**by how much**": an online
    ridge regression (closed form, pure stdlib) fitted on the (value_A, value_B) samples
    observed when A was followed by B within the causal window. It reports an honest
    ``r2`` and residual spread, so counterfactuals can carry effect *sizes* with
    calibrated trust instead of bare probability lifts."""

    def __init__(self, ridge: float = 1e-3) -> None:
        self.ridge = float(ridge)
        self.n = 0
        self._sx = self._sy = self._sxx = self._sxy = self._syy = 0.0
        self.slope = 0.0
        self.intercept = 0.0
        self.r2 = 0.0
        self.residual_std = 0.0

    def add(self, x: float, y: float) -> None:
        self.n += 1
        self._sx += x
        self._sy += y
        self._sxx += x * x
        self._sxy += x * y
        self._syy += y * y
        self._refit()

    def add_many(self, pairs: Iterable[Tuple[float, float]]) -> None:
        for x, y in pairs:
            self.add(float(x), float(y))

    def _refit(self) -> None:
        if self.n < 2:
            return
        n = float(self.n)
        var_x = self._sxx - self._sx * self._sx / n
        cov = self._sxy - self._sx * self._sy / n
        self.slope = cov / (var_x + self.ridge)
        self.intercept = (self._sy - self.slope * self._sx) / n
        var_y = self._syy - self._sy * self._sy / n
        if var_y > 1e-12:
            # residual sum of squares of the fitted line, in closed form
            rss = (self._syy - 2 * self.slope * self._sxy - 2 * self.intercept * self._sy
                   + self.slope * self.slope * self._sxx
                   + 2 * self.slope * self.intercept * self._sx
                   + n * self.intercept * self.intercept)
            rss = max(0.0, rss)
            self.r2 = _clamp(1.0 - rss / var_y)
            self.residual_std = (rss / n) ** 0.5
        else:
            self.r2 = 1.0 if abs(self.slope) < 1e-9 else 0.0
            self.residual_std = 0.0

    def predict(self, x: float) -> float:
        return self.slope * x + self.intercept

    def effect_of_change(self, from_value: float, to_value: float) -> float:
        """The learned change in the effect when the cause moves from→to."""
        return self.slope * (to_value - from_value)

    def confidence(self) -> float:
        """Trust in the fitted function: fit quality × evidence volume."""
        return _clamp(self.r2) * _clamp(self.n / 16.0)

    def to_dict(self) -> Dict[str, Any]:
        return {"n": self.n, "slope": round(self.slope, 6),
                "intercept": round(self.intercept, 6), "r2": round(self.r2, 4),
                "residual_std": round(self.residual_std, 6),
                "sx": self._sx, "sy": self._sy, "sxx": self._sxx,
                "sxy": self._sxy, "syy": self._syy}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FunctionalCausalMechanism":
        m = cls()
        m.n = int(d.get("n", 0))
        m._sx = float(d.get("sx", 0.0))
        m._sy = float(d.get("sy", 0.0))
        m._sxx = float(d.get("sxx", 0.0))
        m._sxy = float(d.get("sxy", 0.0))
        m._syy = float(d.get("syy", 0.0))
        m._refit()
        return m


@dataclass
class Counterfactual:
    """A counterfactual contrast: the target's value under the factual world vs. under an
    intervention that sets ``cause`` to a different value.

    ``abducted`` is the honesty flag: ``True`` means this was a genuine per-instance
    structural counterfactual — Pearl's three-step abduction (recover this episode's
    realized exogenous noise from what actually happened) → action (``do``) → prediction
    (recompute forward holding that same noise fixed). ``False`` means no episode evidence
    was available and this fell back to the population-level *interventional* contrast
    (``effect_of(counter) − effect_of(factual)``, averaged over the whole graph) — a real
    answer, but Rung 2 of the ladder of causation, not Rung 3. Never silently upgraded."""

    cause: str
    target: str
    factual: float
    counter: float
    delta: float
    abducted: bool = False

    def describe(self) -> str:
        kind = "" if self.abducted else " (population-average; no episode evidence to abduct)"
        return (f"had {self.cause!r} been {self.counter:g} instead of {self.factual:g}, "
                f"{self.target!r} would shift by {self.delta:+.3f}{kind}")

    def to_dict(self) -> Dict[str, Any]:
        return {"cause": self.cause, "target": self.target,
                "factual": round(self.factual, 4), "counter": round(self.counter, 4),
                "delta": round(self.delta, 4), "abducted": self.abducted}


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
class CausalWorldModel:
    """Learns genuine causal structure from a time-ordered stream of events & interventions.

    Feed it the world as it unfolds (each event with a timestamp; mark the ones NYXARA
    *did* herself)::

        cwm = CausalWorldModel(window=10.0)
        cwm.observe("rain", at=0.0)
        cwm.observe("wet_ground", at=1.0)
        cwm.observe("slippery", at=2.0)
        cwm.observe("sprinkler", at=50.0, intervention=True)   # she acted
        cwm.observe("wet_ground", at=51.0)

    then ask the questions correlation cannot answer::

        cwm.discover()
        cwm.is_causal("rain", "slippery")     # -> CausalVerdict(verdict="causal", ...)
        cwm.why("slippery")                   # -> [CausalLink(rain→slippery, ...), ...]
        cwm.counterfactual("rain", "slippery", factual=1.0, counter=0.0)

    Association is measured over a causal **window** ``W``: B is "an effect of" A if B
    tends to follow A within W. Direction is intrinsic (only forward windows count);
    confounding is screened by conditioning on earlier common causes; intervention
    (``do``) settles what observation alone cannot.
    """

    def __init__(
        self,
        *,
        window: float = 10.0,
        min_support: int = 4,
        min_observations: int = 8,
        min_confidence: float = 0.25,
        min_contingency: float = 0.10,
        confounder_screening: bool = True,
        use_interventions: bool = True,
        max_vars: int = 512,
        max_events: int = 20000,
        persist_path: Optional[str] = None,
        functional_mechanisms: bool = True,
        min_pairs_fit: int = 8,
        confounder_set_size: int = 2,
        confounder_significance: float = 0.05,
        confounder_permutations: int = 300,
        front_door_adjustment: bool = True,
        enforce_acyclicity: bool = True,
        structural_counterfactuals: bool = True,
        necessity_sufficiency_enabled: bool = True,
        neural_mechanisms: bool = True,
        structure_learning: bool = True,
        structure_min_samples: int = 30,
    ) -> None:
        self.window = float(window)
        self.min_support = max(1, int(min_support))
        self.min_observations = max(1, int(min_observations))
        self.min_confidence = _clamp(min_confidence)
        self.min_contingency = max(0.0, float(min_contingency))
        self.confounder_screening = bool(confounder_screening)
        self.use_interventions = bool(use_interventions)
        self.max_vars = max(8, int(max_vars))
        self.max_events = max(64, int(max_events))
        self.persist_path = persist_path
        self.functional_mechanisms = bool(functional_mechanisms)
        self.min_pairs_fit = max(2, int(min_pairs_fit))
        # how many candidate confounders may be jointly conditioned on at once (a fork
        # A<-{C1,C2}->B that neither C1 nor C2 alone screens off) and the permutation
        # test's rigor — replaces yesterday's fixed magic-number collapse thresholds.
        self.confounder_set_size = max(1, int(confounder_set_size))
        self.confounder_significance = _clamp(confounder_significance)
        self.confounder_permutations = max(20, int(confounder_permutations))
        self.front_door_adjustment = bool(front_door_adjustment)
        self.enforce_acyclicity = bool(enforce_acyclicity)
        # per-instance (Rung 3) counterfactuals and PN/PS/PNS are both real capabilities
        # a caller opts INTO (by passing evidence / by calling necessity_sufficiency at
        # all) — these flags let them be disabled outright (reproduce legacy population-
        # level-only behavior) without touching call sites.
        self.structural_counterfactuals = bool(structural_counterfactuals)
        self.necessity_sufficiency_enabled = bool(necessity_sufficiency_enabled)
        # gradient-descent learners (numpy-guarded; degrade to the linear/counting floor):
        #   · neural_mechanisms  — each edge's f(cause)→effect learned by back-prop, not closed-form
        #   · structure_learning — the DAG itself learned by NOTEARS gradient descent, as a
        #     second opinion that cross-checks (never overrides) the symbolic discovery.
        self.neural_mechanisms = bool(neural_mechanisms)
        self.structure_learning = bool(structure_learning)
        self.structure_min_samples = max(4, int(structure_min_samples))
        self._structure: Any = None                       # DifferentiableCausalStructure, lazily fit

        self._n = 0                                       # total events seen
        self._by_label: Dict[str, List[float]] = {}       # label -> sorted occurrence times
        self._int_by_label: Dict[str, List[float]] = {}   # label -> sorted *intervention* times
        self._values: Dict[str, List[Tuple[float, float]]] = {}   # label -> [(at, value)]
        self._stream: List[Event] = []                    # full (bounded) ordered stream
        self._links: Dict[Tuple[str, str], CausalLink] = {}
        self._mechanisms: Dict[Tuple[str, str], FunctionalCausalMechanism] = {}

    # ------------------------------------------------------------------ #
    # ingest
    # ------------------------------------------------------------------ #
    def observe(self, label: str, *, at: Optional[float] = None,
                intervention: bool = False, value: Optional[float] = None) -> Event:
        """Record one timestamped event. Set ``intervention=True`` when NYXARA brought it
        about herself (a ``do`` — the strongest causal evidence she can generate). An
        optional ``value`` records *how much* (a magnitude), feeding the learned
        functional mechanisms that answer "by how much does A move B?"."""
        lab = str(label)
        if not lab:
            return Event(label="", at=at or time.time())
        ts = time.time() if at is None else float(at)
        val = float(value) if value is not None else None
        ev = Event(label=lab, at=ts, intervention=bool(intervention), value=val)
        self._n += 1
        self._by_label.setdefault(lab, []).append(ts)
        if intervention:
            self._int_by_label.setdefault(lab, []).append(ts)
        if val is not None:
            self._values.setdefault(lab, []).append((ts, val))
        self._stream.append(ev)
        if len(self._stream) > self.max_events:
            self._compact()
        self._prune_vars()
        return ev

    def observe_many(self, labels: Iterable[str], *, at: Optional[float] = None,
                     interventions: Optional[Iterable[str]] = None) -> None:
        """Record several events that all happened at the same instant ``at``."""
        ts = time.time() if at is None else float(at)
        ints = frozenset(str(i) for i in (interventions or ()))
        for lab in labels:
            self.observe(lab, at=ts, intervention=str(lab) in ints)

    def observe_snapshot(self, events: Iterable[str], *,
                         interventions: Optional[Iterable[str]] = None,
                         at: Optional[float] = None) -> None:
        """Alias for :meth:`observe_many` — a situation of co-occurring events."""
        self.observe_many(events, at=at, interventions=interventions)

    # ------------------------------------------------------------------ #
    # the association layer (windowed precedence)
    # ------------------------------------------------------------------ #
    def _times(self, label: str) -> List[float]:
        return self._by_label.get(label, [])

    def _followed_ratio(self, cue_times: List[float], effect: str,
                        ) -> Tuple[float, int, float]:
        """Of the cue timestamps, the fraction followed by an ``effect`` within ``window``;
        also the support (#cues) and the median lag of the ones that were followed."""
        b_times = self._times(effect)
        if not cue_times or not b_times:
            return 0.0, 0, 0.0
        hits = 0
        lags: List[float] = []
        for t in cue_times:
            i = bisect.bisect_right(b_times, t)
            if i < len(b_times) and b_times[i] <= t + self.window:
                hits += 1
                lags.append(b_times[i] - t)
        lags.sort()
        med = lags[len(lags) // 2] if lags else 0.0
        return _safe_div(hits, len(cue_times)), len(cue_times), med

    def _span(self) -> float:
        """The timespan covered by the event stream (seconds)."""
        mn = mx = None
        for ts in self._by_label.values():
            if ts:
                mn = ts[0] if mn is None else min(mn, ts[0])
                mx = ts[-1] if mx is None else max(mx, ts[-1])
        return (mx - mn) if (mn is not None and mx is not None and mx > mn) else 0.0

    def _baseline(self, effect: str) -> float:
        """Background rate of ``effect``: the chance a random instant has an ``effect`` within
        the next window — its "anyway" rate, the no-cause reference for the contingency ΔP.

        Crucially this is a *neutral* baseline (occupancy over the whole timeline), so it is
        **not** inflated by sibling effects of a common cause — the trap that would otherwise
        make a real cause look coincidental."""
        span = self._span()
        occ = len(self._times(effect))
        if span <= 0 or occ == 0:
            return 0.0
        return _clamp(occ * self.window / span)

    def association(self, a: str, b: str) -> float:
        """P(B within window after A) — the forward, time-respecting association."""
        ratio, _, _ = self._followed_ratio(self._times(a), b)
        return ratio

    def contingency(self, a: str, b: str) -> float:
        """ΔP = P(B soon after A) − P(B anyway). Zero ⇒ coincidence."""
        return self.association(a, b) - self._baseline(b)

    # ------------------------------------------------------------------ #
    # confounder screening (conditional independence) — correlation ≠ causation
    #
    # Two upgrades over a single fixed-threshold single-variable check:
    #   1. joint conditioning — a SET of confounders {C1, C2, ...} can jointly screen off
    #      A from B even when no single Ci alone does (a genuine fork on more than one
    #      hidden common cause).
    #   2. a real significance test (stratified permutation) in place of magic-number
    #      collapse thresholds — "is the observed conditional dependence small enough to
    #      be explained by chance under true independence", not just "smaller than some
    #      fixed constant".
    # ------------------------------------------------------------------ #
    def _joint_times(self, condition: Sequence[str]) -> List[float]:
        """Instants where EVERY variable in ``condition`` co-occurs within one window of
        each other — the anchor points a joint-conditioning test is stratified over."""
        if not condition:
            return []
        anchor = self._times(condition[0])
        if len(condition) == 1:
            return list(anchor)
        rest = condition[1:]
        return [t for t in anchor
                if all(_any_in(self._times(c), t - self.window, t + self.window) for c in rest)]

    def _stratified_pairs(self, a: str, b: str, condition: Sequence[str]
                          ) -> List[Tuple[bool, bool]]:
        """For each instant the whole ``condition`` set co-occurs: was A present in the
        following window, and was B? The raw strata a conditional-independence test (and
        its permutation null) are built from."""
        a_times, b_times = self._times(a), self._times(b)
        out: List[Tuple[bool, bool]] = []
        for tc in self._joint_times(condition):
            a_present = _any_in(a_times, tc, tc + self.window)
            b_present = _any_in(b_times, tc, tc + 2 * self.window)   # B may trail A
            out.append((a_present, b_present))
        return out

    def _dep_from_pairs(self, pairs: List[Tuple[bool, bool]]) -> Optional[float]:
        """``|P(B | condition, A) − P(B | condition, ¬A)|`` from raw strata, or ``None``
        if a stratum is too thin to judge."""
        n_ca = sum(1 for a_p, _ in pairs if a_p)
        n_cna = len(pairs) - n_ca
        floor = max(2, self.min_support // 2)
        if n_ca < self.min_support or n_cna < floor:
            return None
        n_ca_b = sum(1 for a_p, b_p in pairs if a_p and b_p)
        n_cna_b = sum(1 for a_p, b_p in pairs if not a_p and b_p)
        return abs(_safe_div(n_ca_b, n_ca) - _safe_div(n_cna_b, n_cna))

    def _dependence_given(self, a: str, b: str, c: str) -> Optional[float]:
        """Single-variable convenience wrapper over :meth:`_dep_from_pairs` — kept as the
        documented "conditional dependence given one variable" primitive other code /
        tests may call directly; :meth:`_find_confounder` itself uses the general
        set-conditioned form so it can screen joint confounders too."""
        return self._dep_from_pairs(self._stratified_pairs(a, b, (c,)))

    def _permutation_pvalue(self, pairs: List[Tuple[bool, bool]], observed_dep: float,
                            *, rng: Optional["random.Random"] = None) -> float:
        """Stratified permutation test: reshuffle which strata are labeled A-present
        (preserving the true count of each), recompute the conditional-dependence
        statistic under this null of "A's presence here is unrelated to B", and report
        how often chance alone matches or beats the observed dependence.

        LOW p ⇒ reject independence — the observed dependence given the condition set is
        real, so A→B survives (not confounded). HIGH p ⇒ the observed (small) dependence
        is unremarkable under the null — consistent with true conditional independence,
        i.e. genuinely confounded. Replaces the old fixed magic-number thresholds with an
        actual significance test."""
        rng = rng or random.Random()
        n = len(pairs)
        n_ca = sum(1 for a_p, _ in pairs if a_p)
        if n == 0 or n_ca == 0 or n_ca == n:
            return 1.0
        b_labels = [b_p for _, b_p in pairs]
        trials = self.confounder_permutations
        idx = list(range(n))
        hits = 0
        for _ in range(trials):
            rng.shuffle(idx)
            present = set(idx[:n_ca])
            n_ca_b = sum(1 for i in present if b_labels[i])
            n_cna_b = sum(1 for i in range(n) if i not in present and b_labels[i])
            n_cna = n - n_ca
            dep = abs(_safe_div(n_ca_b, n_ca) - _safe_div(n_cna_b, n_cna))
            if dep >= observed_dep - 1e-9:
                hits += 1
        return (hits + 1) / (trials + 1)

    def _find_confounder(self, a: str, b: str, marginal_dp: float
                         ) -> Optional[Dict[str, Any]]:
        """A set of earlier common cause(s) C that explains the A–B link. Candidate
        variables must each lead to *both* A and B (a common cause, **not** a mediator
        A→C→B) — checked first (cheap). For each SET (size 1 up to
        ``confounder_set_size``) of candidates: conditioning on it must collapse most of
        the marginal A→B dependence (the effect-size read), AND :meth:`_permutation_pvalue`
        must NOT find strong statistical evidence the leftover dependence is real
        (p ≥ ``confounder_significance``) — a real significance test as a safety net
        against a small-sample effect-size fluke, not a wholesale replacement for it
        (a strict "prove independence" p-value test alone is underpowered at the sample
        sizes this streaming module typically sees)."""
        if not self.confounder_screening or marginal_dp <= 0:
            return None
        if len(self._times(a)) < self.min_support:
            return None
        candidates: List[str] = []
        for c in self._by_label:
            if c in (a, b) or len(self._times(c)) < self.min_support:
                continue
            # C must lead to both A and B...
            if self.association(c, a) < 0.5 or self.association(c, b) < 0.5:
                continue
            # ...and must NOT be a mediator of A (A→C→B is still a genuine causal chain)
            if self.association(a, c) >= self.association(c, a):
                continue
            candidates.append(c)
        if not candidates:
            return None

        # a content-stable seed (NOT Python's built-in hash(), which is randomized per
        # process for strings unless PYTHONHASHSEED is pinned) — the permutation test
        # must be reproducible run-to-run for the same event stream.
        seed = zlib.crc32(f"{a}\x00{b}".encode("utf-8"))
        rng = random.Random(seed)
        best: Optional[Dict[str, Any]] = None
        sizes = range(1, min(self.confounder_set_size, len(candidates)) + 1)
        for size in sizes:
            # bounded combinatorics: the coarse pre-filter above already shrinks
            # `candidates` a great deal, and joint sets beyond pairs are rare/expensive —
            # cap how many combinations of a given size are actually tested.
            tested = 0
            for combo in itertools.combinations(candidates, size):
                if tested >= 40:
                    break
                tested += 1
                pairs = self._stratified_pairs(a, b, combo)
                dep = self._dep_from_pairs(pairs)
                if dep is None:
                    continue
                # effect-size criterion: conditioning on this set collapses most of the
                # marginal dependence (the original, proven-reliable collapse read)...
                collapsed = dep < 0.5 * marginal_dp and dep < self.min_contingency * 3
                if not collapsed:
                    continue
                # ...confirmed by an actual significance test rather than trusted blind:
                # only let a STRONG statistical case against independence (leftover
                # dependence not explainable by sampling noise, p < significance level)
                # veto the collapse read. A strict "prove independence" p-value test
                # alone is underpowered at the small sample sizes this streaming module
                # typically sees, so the effect-size collapse stays the primary signal
                # and the permutation test is the safety net against being fooled by it,
                # not a wholesale replacement for it.
                p_value = self._permutation_pvalue(pairs, dep, rng=rng)
                if p_value < self.confounder_significance:
                    continue   # statistically real leftover dependence — not confounded
                candidate = {"set": combo, "dep": dep, "p_value": p_value}
                if best is None or p_value > best["p_value"]:
                    best = candidate
            if best is not None:
                break   # prefer the smallest confounder set that already screens A off
        return best

    # ------------------------------------------------------------------ #
    # front-door adjustment — identification when the backdoor is blocked by an
    # unobserved/uncontrollable confounder but a clean mediator carries the effect
    # ------------------------------------------------------------------ #
    def _front_door(self, a: str, b: str, confounders: Sequence[str]
                    ) -> Optional[Dict[str, Any]]:
        """When A→B is CONFOUNDED (by ``confounders``), do-calculus isn't necessarily
        stuck: if some mediator M carries the WHOLE effect (A→M→B) and that path is
        itself unconfounded — M is independently established as truly caused by A and
        as truly causing B (not merely associated), and M isn't itself reached by the
        same confounder(s) — Pearl's front-door criterion identifies the effect anyway.

        In this module's linear setting the front-door effect reduces to the product of
        the two mediating edge coefficients (the same arithmetic :meth:`as_causal_graph`
        already uses for mediation) — what's different here is the justification: we
        trust it not because A→B's own regression agrees (it's confounded, we don't
        trust that), but because the A→M→B path is independently screened clean.
        Returns ``None`` — never fabricates an identification — when no such mediator
        exists, matching this module's abstain-honestly contract."""
        if not self.front_door_adjustment:
            return None
        for m in self._by_label:
            if m in (a, b) or m in confounders:
                continue
            if len(self._times(m)) < self.min_support:
                continue
            # is_causal(a, m) and is_causal(m, b) each already run their OWN confound
            # screening over every candidate variable — including these same
            # `confounders` — so a link surviving that screening already establishes
            # front-door's exclusion restriction (M isn't reached by the confounder
            # except via A); no need to re-check raw association here (which would
            # wrongly flag the ordinary, allowed C→A→M chain as a violation).
            v_am = self.is_causal(a, m)
            if not v_am.is_causal:
                continue
            v_mb = self.is_causal(m, b)
            if not v_mb.is_causal:
                continue
            w_am = v_am.evidence.get("interventional_lift", v_am.evidence.get("delta_p", 0.0))
            w_mb = v_mb.evidence.get("interventional_lift", v_mb.evidence.get("delta_p", 0.0))
            mech_am = self._mechanisms.get((a, m))
            mech_mb = self._mechanisms.get((m, b))
            if mech_am is not None and mech_am.confidence() >= 0.3:
                w_am = mech_am.slope
            if mech_mb is not None and mech_mb.confidence() >= 0.3:
                w_mb = mech_mb.slope
            return {"mediator": m, "effect": round(w_am * w_mb, 4),
                    "confidence": round(_clamp(min(v_am.confidence, v_mb.confidence) * 0.9), 4),
                    "a_to_m": round(float(w_am), 4), "m_to_b": round(float(w_mb), 4)}
        return None

    # ------------------------------------------------------------------ #
    # intervention layer (the do-operator — gold standard)
    # ------------------------------------------------------------------ #
    def _interventional_lift(self, a: str, b: str) -> Optional[Tuple[float, int]]:
        """P(B | do A) − P(B | not-A): the effect of *acting* to bring A about. The single
        strongest causal signal — returns (lift, #interventions) or None if too few."""
        if not self.use_interventions:
            return None
        do_times = self._int_by_label.get(a, [])
        if len(do_times) < self.min_support:
            return None
        do_ratio, _, _ = self._followed_ratio(do_times, b)
        return (do_ratio - self._baseline(b)), len(do_times)

    # ------------------------------------------------------------------ #
    # the verdict: causation vs correlation for ONE pair
    # ------------------------------------------------------------------ #
    def is_causal(self, a: str, b: str) -> CausalVerdict:
        """Decide whether A *causes* B or merely *correlates* with it, fusing all criteria."""
        a, b = str(a), str(b)
        fwd, support, lag = self._followed_ratio(self._times(a), b)
        ev: Dict[str, Any] = {"events": self._n, "support": support,
                              "assoc_forward": round(fwd, 3), "lag": round(lag, 3)}

        if self._n < self.min_observations or support < self.min_support:
            return CausalVerdict(a, b, COINCIDENTAL, 0.1,
                                 "too little evidence to tell causation from coincidence", ev)

        dp = fwd - self._baseline(b)
        ev["delta_p"] = round(dp, 3)

        # (1) no contingency → B follows A no more than it follows anything → coincidence
        if dp < self.min_contingency:
            return CausalVerdict(a, b, COINCIDENTAL, _clamp(0.25 + 0.1 * support / max(1, self._n)),
                                 f"B follows A no more than it follows other events "
                                 f"(ΔP={dp:+.2f}) — coincidental", ev)

        # (2) direction — the forward link must beat the reverse one
        rev, _, _ = self._followed_ratio(self._times(b), a)
        ev["assoc_reverse"] = round(rev, 3)
        if rev > fwd + 0.15:
            return CausalVerdict(a, b, REVERSE, _clamp(0.4 + 0.3 * (rev - fwd)),
                                 f"{b!r} precedes {a!r} more reliably than the reverse — "
                                 f"the arrow points the other way", ev)

        # (3) confounder screening — the correlation-vs-causation crux
        conf = self._find_confounder(a, b, dp)
        if conf is not None:
            names = list(conf["set"])
            conf_repr = names[0] if len(names) == 1 else names
            ev["confounder"] = conf_repr
            ev["confounder_p_value"] = round(conf["p_value"], 4)
            fd = self._front_door(a, b, names)
            if fd is not None:
                ev["front_door"] = fd
                return CausalVerdict(
                    a, b, CAUSAL, _clamp(fd["confidence"]),
                    f"confounded by {conf_repr!r}, but identified via the front-door "
                    f"path through mediator {fd['mediator']!r} (p={conf['p_value']:.3f})",
                    ev)
            return CausalVerdict(a, b, CONFOUNDED, _clamp(0.45 + 0.2 * support / max(1, self._n)),
                                 f"A and B share an earlier common cause {conf_repr!r}; their "
                                 f"link dissolves once {conf_repr!r} is accounted for "
                                 f"(permutation p={conf['p_value']:.3f})", ev)

        # (4) intervention — did *acting* on A move B?
        inter = self._interventional_lift(a, b)
        intervened = inter is not None
        if intervened:
            lift, n_do = inter
            ev["interventional_lift"] = round(lift, 3)
            ev["interventions"] = n_do
            if lift < self.min_contingency:
                return CausalVerdict(a, b, CORRELATIONAL,
                                     _clamp(0.4 + 0.2 * n_do / max(1, self._n)),
                                     f"intervening on A did not move B (do-lift={lift:+.2f}) — "
                                     f"the association is not causal", ev)

        # (5) convergent evidence → causal. Confidence honestly reflects how much we have.
        net_dir = fwd - rev
        score = 0.25 + 0.45 * _clamp(dp)
        score += 0.10 * _clamp(support / max(self.min_support * 4, 1))
        score += 0.10 * _clamp(net_dir)
        if intervened:
            score += 0.20 * _clamp(inter[0])      # interventional confirmation is worth the most
        reason = ("acting on A reliably brings B about (do-operator confirms)" if intervened
                  else "A precedes B, B depends on A, and no earlier common cause screens it")
        return CausalVerdict(a, b, CAUSAL, _clamp(score), reason, ev)

    # ------------------------------------------------------------------ #
    # discovery: build the whole graph
    # ------------------------------------------------------------------ #
    def _evaluate_pair(self, a: str, b: str) -> Optional[CausalLink]:
        """Run the full convergent criteria on one ordered pair and return the link the
        model should now hold for it (``None`` ⇒ no link: unsupported or coincidental).
        Shared by :meth:`discover` (full rebuild) and :meth:`update_links_for`
        (incremental per-turn refresh) so both always agree."""
        fwd, support, _ = self._followed_ratio(self._times(a), b)
        if support < self.min_support or fwd <= 0.0:
            return None
        v = self.is_causal(a, b)
        if v.verdict == COINCIDENTAL:
            return None
        strength = v.evidence.get("interventional_lift")
        if strength is None:
            strength = v.evidence.get("delta_p", 0.0)
        link = CausalLink(cause=a, effect=b, strength=float(strength),
                          confidence=v.confidence, verdict=v.verdict,
                          lag=float(v.evidence.get("lag", 0.0)), evidence=v.evidence)
        if link.is_causal and self.functional_mechanisms:
            mech = self._fit_mechanism(a, b)
            if mech is not None:
                self._mechanisms[(a, b)] = mech
                link.evidence["mechanism"] = mech.to_dict()
        return link

    def _set_pair(self, a: str, b: str) -> Optional[CausalLink]:
        """Evaluate one pair and write the result into the live graph (or clear it)."""
        link = self._evaluate_pair(a, b)
        if link is None:
            self._links.pop((a, b), None)
            self._mechanisms.pop((a, b), None)
        else:
            self._links[(a, b)] = link
            if not link.is_causal:
                self._mechanisms.pop((a, b), None)
        return link

    def discover(self) -> List[CausalLink]:
        """Scan every well-supported ordered pair, run the full criteria, and (re)build the
        graph. Returns the links it now believes are *causal*, strongest first."""
        self._links.clear()
        self._mechanisms.clear()
        labels = [l for l, ts in self._by_label.items() if len(ts) >= self.min_support]
        for a in labels:
            for b in labels:
                if a == b:
                    continue
                link = self._evaluate_pair(a, b)
                if link is not None:
                    self._links[(a, b)] = link
        self._enforce_acyclicity()
        if self.structure_learning:
            self.learn_structure()          # a gradient-descent second opinion (numpy; best-effort)
        if self.persist_path:
            try:
                self.save()
            except Exception:  # noqa: BLE001 — persistence is best-effort
                pass
        causal = [l for l in self._links.values() if l.is_causal]
        causal.sort(key=lambda l: l.confidence * abs(l.strength), reverse=True)
        return causal

    # ------------------------------------------------------------------ #
    # acyclicity — a valid SCM/DAG never has feedback cycles at a single time-slice
    # ------------------------------------------------------------------ #
    def _find_cycle(self) -> Optional[List[Tuple[str, str]]]:
        """DFS (white/gray/black coloring) over CAUSAL links only; returns one cycle's
        edges as ``[(cause, effect), ...]``, or ``None`` if already acyclic."""
        adj: Dict[str, List[str]] = {}
        for (c, e), link in self._links.items():
            if link.is_causal:
                adj.setdefault(c, []).append(e)
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {n: WHITE for n in self._by_label}
        parent_edge: Dict[str, Tuple[str, str]] = {}
        found: List[Tuple[str, str]] = []

        def dfs(start: str) -> bool:
            stack = [(start, iter(adj.get(start, [])))]
            color[start] = GRAY
            while stack:
                u, it = stack[-1]
                advanced = False
                for v in it:
                    advanced = True
                    if color.get(v, WHITE) == WHITE:
                        color[v] = GRAY
                        parent_edge[v] = (u, v)
                        stack.append((v, iter(adj.get(v, []))))
                    elif color.get(v) == GRAY:
                        cyc = [(u, v)]
                        cur = u
                        while cur != v:
                            pe = parent_edge.get(cur)
                            if pe is None:
                                break
                            cyc.append(pe)
                            cur = pe[0]
                        found.extend(cyc)
                        return True
                    break
                if not advanced:
                    color[u] = BLACK
                    stack.pop()
            return False

        for node in list(self._by_label):
            if color.get(node, WHITE) == WHITE:
                if dfs(node):
                    return found
        return None

    def _enforce_acyclicity(self) -> None:
        """Greedily find and break cycles among CAUSAL links — a real SCM is a DAG, and
        :meth:`_topo_order`, :meth:`as_causal_graph`, and every propagation-based query
        (``effect_of``, ``counterfactual``, ``necessity_sufficiency``) depend on that. The
        WEAKEST edge in each detected cycle (lowest ``confidence * |strength|``) is
        demoted to CORRELATIONAL — never dropped outright, "A and B associate" stays an
        honest, separate claim from "A causes B"."""
        if not self.enforce_acyclicity:
            return
        guard = 0
        limit = len(self._links) + 8
        while guard < limit:
            guard += 1
            cycle = self._find_cycle()
            if not cycle:
                return
            weakest = min(cycle, key=lambda pair: (
                self._links[pair].confidence * abs(self._links[pair].strength)
                if pair in self._links else -1.0))
            link = self._links.get(weakest)
            if link is None:
                return
            link.verdict = CORRELATIONAL
            link.evidence["cycle_resolved"] = list(cycle)
            self._mechanisms.pop(weakest, None)

    # ------------------------------------------------------------------ #
    # functional mechanisms — HOW MUCH does A move B? (learned, not assumed)
    # ------------------------------------------------------------------ #
    def _fit_mechanism(self, a: str, b: str) -> Optional[FunctionalCausalMechanism]:
        """Fit ``value_B ≈ f(value_A)`` from the (A-value, first-following-B-value) samples
        inside the causal window. When numpy is present and ``neural_mechanisms`` is on, ``f`` is a
        **nonlinear net trained by back-prop** (interventional samples up-weighted); otherwise the
        closed-form linear ridge. None when either side lacks enough valued events."""
        va = self._values.get(a)
        vb = self._values.get(b)
        if not va or not vb:
            return None
        b_times = [t for t, _ in vb]
        int_a = set(self._int_by_label.get(a, ()))         # A-times NYXARA brought about herself
        pairs: List[Tuple[float, float, float]] = []       # (x, y, weight)
        for t, x in va:
            i = bisect.bisect_right(b_times, t)
            if i < len(b_times) and b_times[i] <= t + self.window:
                w = 3.0 if t in int_a else 1.0             # do-samples are the gold standard
                pairs.append((x, vb[i][1], w))
        if len(pairs) < self.min_pairs_fit:
            return None
        mech = self._new_mechanism()
        if isinstance(mech, FunctionalCausalMechanism):    # linear floor ignores weights
            mech.add_many([(x, y) for x, y, _ in pairs])
        else:
            for x, y, w in pairs:
                mech.add(x, y, weight=w)
        return mech

    def _new_mechanism(self) -> Any:
        """A learned functional mechanism: the nonlinear neural one when available, else the ridge."""
        if self.neural_mechanisms:
            try:
                from nyxara.mind.neural_causal import NeuralCausalMechanism, has_numpy
                if has_numpy():
                    return NeuralCausalMechanism()
            except Exception:  # noqa: BLE001 — a missing/broken learner never blocks the floor
                pass
        return FunctionalCausalMechanism()

    def mechanism(self, cause: str, effect: str) -> Optional[FunctionalCausalMechanism]:
        """The learned functional relation for a discovered edge, if one was fitted."""
        return self._mechanisms.get((str(cause), str(effect)))

    # ------------------------------------------------------------------ #
    # structure learning — the DAG itself, learned by gradient descent (NOTEARS)
    # ------------------------------------------------------------------ #
    def _evidence_matrix(self, labels: Sequence[str]) -> Any:
        """Build a (time-bin × variable) matrix from the event stream: each variable's value in a
        bin is its magnitude when it fired there (else 0), so downstream effects accumulate variance
        — the natural scale NOTEARS reads to orient arrows. Falls back to presence indicators."""
        import numpy as _np
        span = self._span() or 1.0
        n_bins = max(8, min(400, int(span / max(self.window, 1e-6)) + 1))
        t0 = min((ts[0] for ts in self._by_label.values() if ts), default=0.0)
        width = span / n_bins if n_bins else self.window
        X = _np.zeros((n_bins, len(labels)))
        for j, lab in enumerate(labels):
            vals = dict(self._values.get(lab, ()))         # at -> value
            for ts in self._by_label.get(lab, ()):
                k = int((ts - t0) / width) if width > 0 else 0
                k = max(0, min(n_bins - 1, k))
                X[k, j] += vals.get(ts, 1.0)
        return X

    def learn_structure(self) -> Optional[Any]:
        """Learn the weighted causal adjacency over the tracked variables **by gradient descent**
        (NOTEARS). A *second, gradient-derived opinion* — surfaced in :meth:`stats`/:meth:`to_dict`
        and used to cross-check (never override) the symbolic verdicts. None without numpy / too
        little data. Best-effort: never raises."""
        if not self.structure_learning:
            return None
        try:
            from nyxara.mind.neural_causal import DifferentiableCausalStructure, has_numpy
            if not has_numpy():
                return None
            labels = [l for l, ts in self._by_label.items() if len(ts) >= self.min_support]
            if len(labels) < 2:
                return None
            X = self._evidence_matrix(labels)
            if X.shape[0] < self.structure_min_samples:
                return None
            self._structure = DifferentiableCausalStructure(labels).fit(X)
            return self._structure
        except Exception:  # noqa: BLE001 — structure learning is a capability, never required
            return None

    def learned_structure(self) -> Optional[Any]:
        """The last :class:`DifferentiableCausalStructure` learned by gradient descent, if any."""
        return self._structure

    # ------------------------------------------------------------------ #
    # incremental (per-turn) structure learning — DYNAMIC causal discovery
    # ------------------------------------------------------------------ #
    def update_links_for(self, labels: Iterable[str]) -> int:
        """Refresh the causal links for every well-supported pair that touches any of the
        given ``labels`` — the cheap, per-turn slice of :meth:`discover`, so the graph
        updates DURING the conversation instead of only every N-th turn. Returns the number
        of ordered pairs re-evaluated. Unknown / thin labels are skipped silently."""
        touched = {str(l) for l in labels
                   if len(self._by_label.get(str(l), ())) >= self.min_support}
        if not touched:
            return 0
        others = [l for l, ts in self._by_label.items() if len(ts) >= self.min_support]
        seen: set = set()
        for a in touched:
            for b in others:
                if a == b:
                    continue
                for pair in ((a, b), (b, a)):
                    if pair not in seen:
                        seen.add(pair)
                        self._set_pair(*pair)
        self._enforce_acyclicity()
        return len(seen)

    # ------------------------------------------------------------------ #
    # label lookup — map a natural-language mention onto a tracked variable
    # ------------------------------------------------------------------ #
    def labels(self) -> List[str]:
        """Every variable label the model currently tracks, most-observed first."""
        return [l for l, _ in sorted(self._by_label.items(),
                                     key=lambda kv: len(kv[1]), reverse=True)]

    @staticmethod
    def _label_tokens(text: str) -> List[str]:
        out: List[str] = []
        cur: List[str] = []
        for ch in str(text).lower():
            if ch.isalnum():
                cur.append(ch)
            elif cur:
                out.append("".join(cur))
                cur = []
        if cur:
            out.append("".join(cur))
        return out

    def match_label(self, text: str, *, min_overlap: float = 0.6) -> Optional[str]:
        """Resolve a natural-language mention onto the tracked label it names, if any.

        Pure-stdlib fuzzy match: token overlap between the mention and each label
        (label prefixes like ``topic:`` are tokenized too, so "the ground being wet"
        finds ``wet_ground``). Returns ``None`` below the overlap floor — an honest miss,
        never a guess."""
        mention = set(self._label_tokens(text))
        if not mention:
            return None
        best: Optional[str] = None
        best_score = 0.0
        for lab in self._by_label:
            toks = self._label_tokens(lab)
            core = [t for t in toks if t not in ("topic", "act", "tool", "kind", "wm",
                                                 "do", "outcome", "recall", "mood")]
            core = core or toks
            hit = sum(1 for t in core if t in mention)
            score = hit / len(core)
            if score >= min_overlap and (score > best_score or
                                         (score == best_score and best is not None
                                          and len(core) > len(self._label_tokens(best)))):
                best, best_score = lab, score
        return best

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #
    def why(self, effect: str, *, min_confidence: Optional[float] = None) -> List[CausalLink]:
        """The genuine causes of ``effect`` — "why did this happen?" — ranked by trust."""
        thr = self.min_confidence if min_confidence is None else min_confidence
        out = [l for (c, e), l in self._links.items()
               if e == effect and l.is_causal and l.confidence >= thr]
        out.sort(key=lambda l: l.confidence * abs(l.strength), reverse=True)
        return out

    def effects_of(self, cause: str, *, min_confidence: Optional[float] = None) -> List[CausalLink]:
        """What ``cause`` genuinely brings about, ranked."""
        thr = self.min_confidence if min_confidence is None else min_confidence
        out = [l for (c, e), l in self._links.items()
               if c == cause and l.is_causal and l.confidence >= thr]
        out.sort(key=lambda l: l.confidence * abs(l.strength), reverse=True)
        return out

    def links(self, *, include_noncausal: bool = False) -> List[CausalLink]:
        out = [l for l in self._links.values() if include_noncausal or l.is_causal]
        out.sort(key=lambda l: l.confidence * abs(l.strength), reverse=True)
        return out

    def explain(self, effect: str) -> str:
        """A one-line natural-language "why did X happen?" answer (honest when unknown)."""
        causes = self.why(effect)
        if not causes:
            return f"I have no established cause for {effect!r} yet."
        top = causes[0]
        tail = "" if len(causes) == 1 else \
            " (also: " + ", ".join(repr(l.cause) for l in causes[1:3]) + ")"
        return (f"{effect!r} happened because {top.cause!r} did "
                f"(confidence {top.confidence:.0%}){tail}.")

    def as_causal_graph(self, *, min_confidence: Optional[float] = None,
                        prune_mediated: bool = True) -> List[Tuple[str, str, float]]:
        """Export the *causal* edges as ``[(cause, effect, weight)]`` for the structural
        propagation engine in :mod:`mind.strategies`.

        When a **learned functional mechanism** exists for an edge and has earned trust
        (fit quality × evidence), its fitted slope is the structural coefficient — a real
        measured effect size — instead of the probability-lift proxy.

        ``prune_mediated`` (on by default) does a **transitive reduction**: :meth:`discover`
        evaluates every ordered pair independently, so a genuine mediation chain
        ``a → b → c`` also earns its OWN ``a → c`` causal verdict (a *is* a cause of c —
        just not a direct one). Exporting that redundant edge alongside the two-hop path
        would double-count the same influence when path-summing (:class:`CausalModel`'s
        ``effect_of``/``counterfactual``). When ``a → c``'s weight is (within tolerance)
        fully explained by ``weight(a→b) * weight(b→c)`` for some mediator ``b``, the
        direct edge is dropped from the *propagation* graph — it is fully accounted for
        by the mediated path. A genuine *partial* direct effect beyond the mediated one
        (``a → c``'s weight exceeds the mediated estimate) is never dropped: only edges
        that add no new information to propagation are pruned. ``why``/``is_causal``/
        ``effects_of`` are untouched by this — "does a cause c" stays a real yes."""
        thr = self.min_confidence if min_confidence is None else min_confidence
        raw: Dict[Tuple[str, str], float] = {}
        for l in self._links.values():
            if not (l.is_causal and l.confidence >= thr):
                continue
            weight = l.strength * l.confidence
            mech = self._mechanisms.get((l.cause, l.effect))
            if mech is not None and mech.confidence() >= 0.3:
                weight = mech.slope
            raw[(l.cause, l.effect)] = weight

        if prune_mediated:
            for (a, c), w_ac in list(raw.items()):
                if (a, c) not in raw or abs(w_ac) < 1e-9:
                    continue
                for (a2, b), w_ab in list(raw.items()):
                    if a2 != a or b == c or b == a:
                        continue
                    w_bc = raw.get((b, c))
                    if w_bc is None:
                        continue
                    mediated = w_ab * w_bc
                    if abs(w_ac - mediated) <= max(0.15 * abs(w_ac), 0.05):
                        del raw[(a, c)]
                        break

        return [(c, e, w) for (c, e), w in raw.items()]

    def _structural_model(self) -> Any:
        from nyxara.mind.strategies import CausalModel
        return CausalModel(self.as_causal_graph())

    def predict_effects(self, cause: str, value: float = 1.0,
                        target: Optional[str] = None) -> Any:
        """Forward ``do``-propagation along discovered edges (reuses ``CausalModel``).

        With ``target`` → the total effect of ``do(cause=value)`` on that target (float).
        Without ``target`` → a ``{reachable_effect: effect_size}`` map."""
        model = self._structural_model()
        if target is not None:
            return model.effect_of(cause, value, target)
        out: Dict[str, float] = {}
        for (c, e) in list(self._links.keys()):
            if e != cause and e not in out:
                val = model.effect_of(cause, value, e)
                if abs(val) > 1e-9:
                    out[e] = val
        return out

    def do(self, interventions: Dict[str, float],
           target: Optional[str] = None) -> Any:
        """The multi-variable do-operator: forward propagation of a **simultaneous**
        intervention ``do({var1: value1, var2: value2, ...})`` — the genuine SCM
        operation of setting a SET of variables at once, each cut off from its own
        causal parents (reuses :meth:`CausalModel.effect_of_many`'s linear superposition).

        With ``target`` → the total joint effect (float). Without ``target`` → a
        ``{reachable_effect: effect_size}`` map over every effect reachable from any
        intervened variable."""
        model = self._structural_model()
        do = {str(k): float(v) for k, v in interventions.items()}
        if target is not None:
            return model.effect_of_many(do, target)
        out: Dict[str, float] = {}
        for (c, e) in list(self._links.keys()):
            if e not in do and e not in out:
                val = model.effect_of_many(do, e)
                if abs(val) > 1e-9:
                    out[e] = val
        return out

    # ------------------------------------------------------------------ #
    # counterfactual PROBABILITIES — Pearl's PN / PS / PNS
    # ------------------------------------------------------------------ #
    def necessity_sufficiency(self, cause: str, target: str, *, n_samples: int = 500,
                              threshold: float = 0.5, seed: Optional[int] = None
                              ) -> Optional[Dict[str, float]]:
        """Probability of Necessity (PN), Sufficiency (PS), and Necessity-and-Sufficiency
        (PNS) for ``cause`` on ``target`` — the mathematically rigorous answer to
        "did A really matter for B?", not a bare confidence heuristic:

            PN  = P(Y_{do(cause=0)} < threshold  |  Y_{do(cause=1)} ≥ threshold)
                  "if cause hadn't happened, would target still not have?" (necessity)
            PS  = P(Y_{do(cause=1)} ≥ threshold  |  Y_{do(cause=0)} < threshold)
                  "if cause DID happen, would target now occur?" (sufficiency)
            PNS = P(Y_{do(cause=1)} ≥ threshold  AND  Y_{do(cause=0)} < threshold)
                  cause is simultaneously necessary and sufficient this draw

        Ordinarily PN/PS/PNS are only *boundable* (Tian–Pearl bounds) without a fully
        specified structural model — the identification problem those bounds exist for.
        Here we sidestep it: when a :class:`FunctionalCausalMechanism` is fitted along the
        path, we already HAVE a structural equation + a real fitted residual distribution
        (``residual_std``) at each node, so Monte Carlo directly over that noise gives the
        model-implied exact quantities (under the linear-Gaussian-residual assumption this
        module already makes for ``effect_of``/``counterfactual``) — not just bounds.

        Each Monte Carlo draw samples ONE noise realization shared by both the
        ``do(cause=1)`` and ``do(cause=0)`` worlds (the paired-counterfactual construction
        Pearl's ``Y_x`` and ``Y_{x'}`` require), propagates both through the SAME
        structural equations used by :meth:`counterfactual`, and binarizes each outcome
        against ``threshold`` (indicator semantics, matching this module's
        ``factual=1.0``/``counter=0.0`` convention).

        Returns ``None`` (abstains) when no mechanism is fitted anywhere on the path —
        there is then no real residual distribution to sample from, and fabricating one
        would violate this module's honesty contract."""
        if not self.necessity_sufficiency_enabled:
            return None
        weights: Dict[Tuple[str, str], float] = {
            (c, e): w for c, e, w in self.as_causal_graph()}
        parents_of: Dict[str, List[str]] = {}
        for (c, e) in weights:
            parents_of.setdefault(e, []).append(c)
        order = self._topo_order()
        if cause not in order or target not in order:
            return None

        resid: Dict[str, float] = {}
        has_mechanism = False
        for node in order:
            best = 0.0
            for p in parents_of.get(node, []):
                mech = self._mechanisms.get((p, node))
                if mech is not None and mech.n >= 2:
                    best = max(best, mech.residual_std)
                    has_mechanism = True
            resid[node] = best
        if not has_mechanism:
            return None

        base = self.latest_evidence(order)
        rng = random.Random(seed)

        def _predict(node: str, values: Dict[str, float]) -> Tuple[float, bool]:
            total, known = 0.0, False
            for p in parents_of.get(node, []):
                if p in values:
                    total += weights.get((p, node), 0.0) * values[p]
                    known = True
            return total, known

        def _simulate(cause_value: float, draw_noise: Dict[str, float]) -> Dict[str, float]:
            values = dict(base)
            values[cause] = cause_value
            for node in order:
                if node == cause:
                    continue
                pred, known = _predict(node, values)
                if known:
                    values[node] = pred + draw_noise.get(node, 0.0)
                elif node not in values:
                    values[node] = pred
            return values

        n = max(1, int(n_samples))
        n_t1 = n_t1_and_not_t0 = 0
        n_not_t0 = n_not_t0_and_t1 = 0
        n_pns = 0
        for _ in range(n):
            noise = {node: rng.gauss(0.0, resid[node]) for node in order if resid[node] > 0}
            t1 = _simulate(1.0, noise).get(target, 0.0) >= threshold
            t0 = _simulate(0.0, noise).get(target, 0.0) >= threshold
            if t1:
                n_t1 += 1
                if not t0:
                    n_t1_and_not_t0 += 1
            if not t0:
                n_not_t0 += 1
                if t1:
                    n_not_t0_and_t1 += 1
            if t1 and not t0:
                n_pns += 1

        return {"PN": round(_safe_div(n_t1_and_not_t0, n_t1), 4),
                "PS": round(_safe_div(n_not_t0_and_t1, n_not_t0), 4),
                "PNS": round(n_pns / n, 4),
                "samples": n}

    def counterfactual(self, cause: str, target: str,
                       factual: float = 1.0, counter: float = 0.0,
                       evidence: Optional[Dict[str, float]] = None) -> Counterfactual:
        """"Had ``cause`` been ``counter`` instead of ``factual``, what of ``target``?"

        Without ``evidence`` this is the population-level *interventional* contrast
        (``effect_of(cause=counter) − effect_of(cause=factual)`` over the whole graph) —
        Rung 2 of the ladder of causation, exactly today's behavior (no regressions).

        With ``evidence`` — the actual observed/latest state of the world this episode
        (see :meth:`latest_evidence`) — this runs the real thing: Pearl's three-step
        counterfactual. **Abduction**: recover each node's realized exogenous noise
        (``observed − structural_prediction_from_its_parents``) from the evidence, using
        the learned edge weights (:meth:`as_causal_graph`, which prefers a fitted
        :class:`FunctionalCausalMechanism`'s slope when trusted). **Action**: set
        ``cause = counter``. **Prediction**: recompute every node in topological order
        from its (possibly now-changed) parents plus that *same* abducted noise — nodes
        not downstream of ``cause`` reproduce their factual value exactly (self-consistent:
        their parents didn't change), nodes downstream genuinely propagate the intervention.
        This is Rung 3: a per-*instance* answer, not a population average."""
        if not evidence or not self._mechanisms or not self.structural_counterfactuals:
            model = self._structural_model()
            delta = model.counterfactual(cause, target, factual, counter)
            return Counterfactual(cause=cause, target=target, factual=factual,
                                  counter=counter, delta=delta, abducted=False)

        weights: Dict[Tuple[str, str], float] = {
            (c, e): w for c, e, w in self.as_causal_graph()}
        parents_of: Dict[str, List[str]] = {}
        for (c, e) in weights:
            parents_of.setdefault(e, []).append(c)
        order = self._topo_order()

        fact_values: Dict[str, float] = dict(evidence)
        fact_values.setdefault(cause, factual)

        def _predict(node: str, values: Dict[str, float]) -> Tuple[float, bool]:
            total, known = 0.0, False
            for p in parents_of.get(node, []):
                if p in values:
                    total += weights.get((p, node), 0.0) * values[p]
                    known = True
            return total, known

        # ---- abduction: the noise realized THIS episode, from what actually happened ----
        noise: Dict[str, float] = {}
        for node in order:
            if node == cause or node not in fact_values:
                continue
            pred, known = _predict(node, fact_values)
            noise[node] = (fact_values[node] - pred) if known else fact_values[node]

        def _propagate(values: Dict[str, float]) -> Dict[str, float]:
            out = dict(values)
            for node in order:
                if node == cause:
                    continue
                pred, known = _predict(node, out)
                if known:
                    out[node] = pred + noise.get(node, 0.0)
                elif node not in out:
                    out[node] = pred
            return out

        # ---- action + prediction: do(cause=counter), same noise, forward through the DAG ----
        counter_world = _propagate({**fact_values, cause: counter})
        factual_world = _propagate({**fact_values, cause: fact_values[cause]})

        counter_t = counter_world.get(target, 0.0)
        factual_t = factual_world.get(target, 0.0)
        return Counterfactual(cause=cause, target=target, factual=factual,
                              counter=counter, delta=counter_t - factual_t, abducted=True)

    def latest_evidence(self, labels: Optional[Iterable[str]] = None) -> Dict[str, float]:
        """The most recent known state of each label — "what actually happened" this
        episode, the grounding a per-instance counterfactual abducts from. A valued label
        reports its last recorded ``value``; a bare (indicator) label that has occurred
        reports ``1.0`` (matching the ``factual=1.0`` convention used elsewhere). A label
        never observed is simply omitted — an honest "unknown", never a fabricated zero."""
        labs = list(labels) if labels is not None else list(self._by_label)
        out: Dict[str, float] = {}
        for lab in labs:
            vals = self._values.get(lab)
            if vals:
                out[lab] = vals[-1][1]
            elif self._by_label.get(lab):
                out[lab] = 1.0
        return out

    def _topo_order(self) -> List[str]:
        """A topological order over the discovered CAUSAL links (Kahn's algorithm).

        Cycle-safe: :meth:`discover` enforces acyclicity on the causal graph going
        forward, but this stays safe (leftover cyclic nodes appended in stable order)
        even if called before a `discover()` or on links from an older save."""
        nodes = set(self._by_label)
        indeg: Dict[str, int] = {n: 0 for n in nodes}
        adj: Dict[str, List[str]] = {n: [] for n in nodes}
        for (c, e), link in self._links.items():
            if link.is_causal and c in nodes and e in nodes:
                adj[c].append(e)
                indeg[e] += 1
        from collections import deque
        q = deque(sorted(n for n in nodes if indeg[n] == 0))
        order: List[str] = []
        seen: set = set()
        while q:
            n = q.popleft()
            if n in seen:
                continue
            seen.add(n)
            order.append(n)
            for v in sorted(adj[n]):
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        for n in sorted(nodes):
            if n not in seen:
                order.append(n)
                seen.add(n)
        return order

    # ------------------------------------------------------------------ #
    # housekeeping
    # ------------------------------------------------------------------ #
    def _compact(self) -> None:
        """Drop the oldest events when the stream overflows, keeping counts consistent."""
        keep = self._stream[-self.max_events:]
        self._stream = keep
        self._by_label = {}
        self._int_by_label = {}
        self._values = {}
        for ev in keep:
            self._by_label.setdefault(ev.label, []).append(ev.at)
            if ev.intervention:
                self._int_by_label.setdefault(ev.label, []).append(ev.at)
            if ev.value is not None:
                self._values.setdefault(ev.label, []).append((ev.at, ev.value))
        self._n = len(keep)

    def _prune_vars(self) -> None:
        """Bound the variable set: forget the rarest labels if it overflows."""
        if len(self._by_label) <= self.max_vars:
            return
        keep = {v for v, _ in sorted(self._by_label.items(), key=lambda kv: len(kv[1]),
                                     reverse=True)[:self.max_vars]}
        self._by_label = {v: t for v, t in self._by_label.items() if v in keep}
        self._int_by_label = {v: t for v, t in self._int_by_label.items() if v in keep}
        self._values = {v: t for v, t in self._values.items() if v in keep}

    def stats(self) -> Dict[str, Any]:
        s = {"events": self._n, "variables": len(self._by_label),
             "interventions": sum(len(t) for t in self._int_by_label.values()),
             "valued_labels": len(self._values),
             "links": len(self._links),
             "causal_links": sum(1 for l in self._links.values() if l.is_causal),
             "mechanisms": len(self._mechanisms),
             "neural_mechanisms": sum(
                 1 for m in self._mechanisms.values() if getattr(m, "KIND", "") == "neural")}
        if self._structure is not None and getattr(self._structure, "fitted", False):
            s["learned_structure"] = {
                "edges": len(self._structure.edges()),
                "acyclicity": round(float(getattr(self._structure, "h_final", 0.0)), 6)}
        return s

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        d = {"window": self.window,
             "stream": [e.to_dict() for e in self._stream],
             "links": [l.to_dict() for l in self._links.values()]}
        if self._structure is not None and getattr(self._structure, "fitted", False):
            d["structure"] = self._structure.to_dict()
        return d

    def load_dict(self, d: Dict[str, Any]) -> None:
        self.window = float(d.get("window", self.window))
        self._n = 0
        self._by_label = {}
        self._int_by_label = {}
        self._values = {}
        self._stream = []
        for ed in d.get("stream", []):
            ev = Event.from_dict(ed)
            self._stream.append(ev)
            self._by_label.setdefault(ev.label, []).append(ev.at)
            if ev.intervention:
                self._int_by_label.setdefault(ev.label, []).append(ev.at)
            if ev.value is not None:
                self._values.setdefault(ev.label, []).append((ev.at, ev.value))
        self._n = len(self._stream)
        for lst in self._by_label.values():
            lst.sort()
        for lst in self._int_by_label.values():
            lst.sort()
        for lst in self._values.values():
            lst.sort()
        self._links = {}
        self._mechanisms = {}
        for ld in d.get("links", []):
            link = CausalLink.from_dict(ld)
            self._links[(link.cause, link.effect)] = link
            md = link.evidence.get("mechanism")
            if md:                       # the learned functional relation rides the link
                try:
                    if md.get("kind") == "neural":
                        from nyxara.mind.neural_causal import NeuralCausalMechanism
                        self._mechanisms[(link.cause, link.effect)] = \
                            NeuralCausalMechanism.from_dict(md)
                    else:
                        self._mechanisms[(link.cause, link.effect)] = \
                            FunctionalCausalMechanism.from_dict(md)
                except Exception:  # noqa: BLE001 — a bad mechanism never blocks a restore
                    pass
        sd = d.get("structure")
        if sd:
            try:
                from nyxara.mind.neural_causal import DifferentiableCausalStructure
                self._structure = DifferentiableCausalStructure.from_dict(sd)
            except Exception:  # noqa: BLE001 — a bad structure never blocks a restore
                self._structure = None

    def save(self, path: Optional[str] = None) -> None:
        target = path or self.persist_path
        if not target:
            return
        target = os.path.expanduser(target)
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        tmp = target + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)
        os.replace(tmp, target)

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> "CausalWorldModel":
        path = os.path.expanduser(path)
        model = cls(persist_path=path, **kwargs)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    model.load_dict(json.load(fh))
            except Exception:  # noqa: BLE001 — a corrupt file must never brick the mind
                pass
        return model


# --------------------------------------------------------------------------- #
# self-test / demo — correlation vs causation, the textbook traps
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import random

    def episodes(n: int, gap: float = 100.0):
        """Yield well-separated episode base-times so within-episode order is unambiguous."""
        for k in range(n):
            yield k * gap

    rng = random.Random(7)

    # ---- Case 1: a genuine chain  rain → wet_ground → slippery ----------------
    cwm = CausalWorldModel(window=10.0)
    for base in episodes(300):
        if rng.random() < 0.5:
            cwm.observe("rain", at=base)
            if rng.random() < 0.9:                          # rain wets the ground
                cwm.observe("wet_ground", at=base + 1)
                if rng.random() < 0.85:                     # wet ground is slippery
                    cwm.observe("slippery", at=base + 2)
        if rng.random() < 0.3:                              # independent background noise,
            cwm.observe("birds_singing", at=base + rng.uniform(0, 90))   # spread across time
    cwm.discover()
    print("CASE 1 — real chain (rain → wet_ground → slippery)")
    print("  rain → slippery :", cwm.is_causal("rain", "slippery").verdict)
    print("  why slippery    :", [l.describe() for l in cwm.why("slippery")])
    print("  birds → slippery:", cwm.is_causal("birds_singing", "slippery").verdict)
    print("  counterfactual  :", cwm.counterfactual("rain", "slippery", 1.0, 0.0).describe())
    print("  explain         :", cwm.explain("slippery"))

    # ---- Case 2: a CONFOUNDER  summer → {ice_cream, drowning} -----------------
    conf = CausalWorldModel(window=10.0)
    for base in episodes(400):
        if rng.random() < 0.5:                              # summer
            conf.observe("summer", at=base)
            if rng.random() < 0.85:
                conf.observe("ice_cream", at=base + 1)
            if rng.random() < 0.8:
                conf.observe("drowning", at=base + 2)
        else:
            if rng.random() < 0.1:
                conf.observe("ice_cream", at=base + 1)
            if rng.random() < 0.05:
                conf.observe("drowning", at=base + 2)
    conf.discover()
    print("\nCASE 2 — confounded (common cause: summer)")
    v = conf.is_causal("ice_cream", "drowning")
    print("  ice_cream → drowning :", v.verdict, "—", v.reason)
    print("  summer → drowning    :", conf.is_causal("summer", "drowning").verdict)
    print("  why drowning         :", [l.describe() for l in conf.why("drowning")])

    # ---- Case 3: intervention settles what correlation cannot -----------------
    # A and B are driven by a *hidden* common cause H (never observed). Observationally A
    # and B march together. Only by *acting* — do(A) without H — does NYXARA learn that
    # forcing A does not bring B, so A→B is not causal. This is intervention's whole point.
    inter = CausalWorldModel(window=10.0)
    for base in episodes(400):
        h = rng.random() < 0.5                              # hidden: not recorded
        if h:
            if rng.random() < 0.9:
                inter.observe("A", at=base + 1)
            if rng.random() < 0.9:
                inter.observe("B", at=base + 2)
        if rng.random() < 0.4:                              # she *forces* A — B should not follow
            inter.observe("A", at=base + 5, intervention=True)
            if rng.random() < 0.1:
                inter.observe("B", at=base + 6)
    inter.discover()
    print("\nCASE 3 — intervention overrides correlation")
    v = inter.is_causal("A", "B")
    print("  A → B :", v.verdict, "—", v.reason)
    print("  stats :", inter.stats())

    # ---- Case 4: Rung 2 (intervention, population-average) vs Rung 3 (structural
    # counterfactual, THIS episode) — the exact distinction "she talks counterfactual"
    # vs "she reasons counterfactually" turns on. Same chain, same query; the population
    # answer is a single number for everyone, the structural one is grounded in what
    # *actually happened* this specific time (abducted noise), and differs per episode.
    chain = CausalWorldModel(window=10.0)
    for k in range(300):
        base = k * 100.0
        a_val = rng.gauss(5.0, 2.0)
        chain.observe("rung_a", at=base, value=a_val)
        b_val = 2.0 * a_val + 1.0 + rng.gauss(0.0, 0.3)
        chain.observe("rung_b", at=base + 1, value=b_val)
        c_val = 3.0 * b_val - 2.0 + rng.gauss(0.0, 0.3)
        chain.observe("rung_c", at=base + 2, value=c_val)
    chain.discover()
    print("\nCASE 4 — Rung 2 (intervention) vs Rung 3 (structural counterfactual)")
    rung2 = chain.counterfactual("rung_a", "rung_c", factual=5.0, counter=0.0)
    print("  Rung 2 (population avg, do(rung_a)) :", rung2.describe())
    # this specific episode had unusually large positive/negative noise at each hop —
    # a real "what actually happened" the population answer above knows nothing about
    a_actual = 8.0
    base = 99000.0
    chain.observe("rung_a", at=base, value=a_actual)
    b_actual = 2.0 * a_actual + 1.0 + 4.0
    chain.observe("rung_b", at=base + 1, value=b_actual)
    c_actual = 3.0 * b_actual - 2.0 - 5.0
    chain.observe("rung_c", at=base + 2, value=c_actual)
    evidence = chain.latest_evidence(["rung_a", "rung_b", "rung_c"])
    rung3 = chain.counterfactual("rung_a", "rung_c", factual=a_actual, counter=0.0,
                                 evidence=evidence)
    print("  this episode's evidence              :", evidence)
    print("  Rung 3 (abduction → action → prediction, THIS episode) :", rung3.describe())
    print("  (abducted =", rung3.abducted, "— honest flag distinguishing Rung 3 from Rung 2)")

    # ---- Case 5: Probability of Necessity / Sufficiency on a synthetic SCM ----------
    ns = CausalWorldModel(window=10.0)
    for k in range(400):
        base = k * 100.0
        did = rng.random() < 0.5
        ns.observe("match_struck", at=base, value=1.0 if did else 0.0,
                  intervention=(k % 3 == 0))
        fire = did and rng.random() < 0.92
        ns.observe("fire", at=base + 1, value=1.0 if fire else 0.0)
    ns.discover()
    pns = ns.necessity_sufficiency("match_struck", "fire")
    print("\nCASE 5 — Probability of Necessity / Sufficiency (match → fire)")
    print("  PN / PS / PNS :", pns)
