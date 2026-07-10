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
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    intervention that sets ``cause`` to a different value."""

    cause: str
    target: str
    factual: float
    counter: float
    delta: float

    def describe(self) -> str:
        return (f"had {self.cause!r} been {self.counter:g} instead of {self.factual:g}, "
                f"{self.target!r} would shift by {self.delta:+.3f}")

    def to_dict(self) -> Dict[str, Any]:
        return {"cause": self.cause, "target": self.target,
                "factual": round(self.factual, 4), "counter": round(self.counter, 4),
                "delta": round(self.delta, 4)}


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
    # ------------------------------------------------------------------ #
    def _dependence_given(self, a: str, b: str, c: str) -> Optional[float]:
        """Within the contexts where C just occurred, how much does B still depend on A?

        Returns ``|P(B | C, A) − P(B | C, ¬A)|`` — the *conditional* dependence of B on A
        given C — or ``None`` if a stratum is too thin to judge. This is the test that tells
        a **fork** A←C→B (B independent of A once C is fixed ⇒ value ≈ 0 ⇒ confounded) apart
        from a **chain** C→A→B (B still depends on A given C ⇒ value large ⇒ A→B is real)."""
        c_times, a_times, b_times = self._times(c), self._times(a), self._times(b)
        n_ca = n_ca_b = n_cna = n_cna_b = 0
        for tc in c_times:
            a_present = _any_in(a_times, tc, tc + self.window)
            b_present = _any_in(b_times, tc, tc + 2 * self.window)   # B may trail A
            if a_present:
                n_ca += 1
                n_ca_b += int(b_present)
            else:
                n_cna += 1
                n_cna_b += int(b_present)
        floor = max(2, self.min_support // 2)
        if n_ca < self.min_support or n_cna < floor:
            return None
        return abs(_safe_div(n_ca_b, n_ca) - _safe_div(n_cna_b, n_cna))

    def _find_confounder(self, a: str, b: str, marginal_dp: float) -> Optional[str]:
        """An earlier common cause C that explains the A–B link. C qualifies when it leads to
        *both* A and B (a common cause, **not** a mediator A→C→B) and — once we condition on
        C — most of the A→B dependence **collapses**. That collapse (conditional dependence
        falling to well below the marginal ΔP) is exactly what separates a spurious
        correlation from a real cause: a chain C→A→B keeps its A→B dependence given C, a fork
        A←C→B loses it."""
        if not self.confounder_screening or marginal_dp <= 0:
            return None
        if len(self._times(a)) < self.min_support:
            return None
        for c in self._by_label:
            if c in (a, b):
                continue
            if len(self._times(c)) < self.min_support:
                continue
            # C must lead to both A and B...
            if self.association(c, a) < 0.5 or self.association(c, b) < 0.5:
                continue
            # ...and must NOT be a mediator of A (A→C→B is still a genuine causal chain)
            if self.association(a, c) >= self.association(c, a):
                continue
            dep = self._dependence_given(a, b, c)
            # conditioning on C explains most of the dependence → confounded, not causal
            if dep is not None and dep < 0.5 * marginal_dp and dep < self.min_contingency * 3:
                return c
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
            ev["confounder"] = conf
            return CausalVerdict(a, b, CONFOUNDED, _clamp(0.45 + 0.2 * support / max(1, self._n)),
                                 f"A and B share an earlier common cause {conf!r}; their link "
                                 f"dissolves once {conf!r} is accounted for", ev)

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
                fwd, support, _ = self._followed_ratio(self._times(a), b)
                if support < self.min_support or fwd <= 0.0:
                    continue
                v = self.is_causal(a, b)
                if v.verdict == COINCIDENTAL:
                    continue
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
                self._links[(a, b)] = link
        if self.persist_path:
            try:
                self.save()
            except Exception:  # noqa: BLE001 — persistence is best-effort
                pass
        causal = [l for l in self._links.values() if l.is_causal]
        causal.sort(key=lambda l: l.confidence * abs(l.strength), reverse=True)
        return causal

    # ------------------------------------------------------------------ #
    # functional mechanisms — HOW MUCH does A move B? (learned, not assumed)
    # ------------------------------------------------------------------ #
    def _fit_mechanism(self, a: str, b: str) -> Optional[FunctionalCausalMechanism]:
        """Fit ``value_B ≈ f(value_A)`` from the (A-value, first-following-B-value) samples
        inside the causal window. None when either side lacks enough valued events."""
        va = self._values.get(a)
        vb = self._values.get(b)
        if not va or not vb:
            return None
        b_times = [t for t, _ in vb]
        pairs: List[Tuple[float, float]] = []
        for t, x in va:
            i = bisect.bisect_right(b_times, t)
            if i < len(b_times) and b_times[i] <= t + self.window:
                pairs.append((x, vb[i][1]))
        if len(pairs) < self.min_pairs_fit:
            return None
        mech = FunctionalCausalMechanism()
        mech.add_many(pairs)
        return mech

    def mechanism(self, cause: str, effect: str) -> Optional[FunctionalCausalMechanism]:
        """The learned functional relation for a discovered edge, if one was fitted."""
        return self._mechanisms.get((str(cause), str(effect)))

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

    def as_causal_graph(self, *, min_confidence: Optional[float] = None
                        ) -> List[Tuple[str, str, float]]:
        """Export the *causal* edges as ``[(cause, effect, weight)]`` for the structural
        propagation engine in :mod:`mind.strategies`.

        When a **learned functional mechanism** exists for an edge and has earned trust
        (fit quality × evidence), its fitted slope is the structural coefficient — a real
        measured effect size — instead of the probability-lift proxy."""
        thr = self.min_confidence if min_confidence is None else min_confidence
        out: List[Tuple[str, str, float]] = []
        for l in self._links.values():
            if not (l.is_causal and l.confidence >= thr):
                continue
            weight = l.strength * l.confidence
            mech = self._mechanisms.get((l.cause, l.effect))
            if mech is not None and mech.confidence() >= 0.3:
                weight = mech.slope
            out.append((l.cause, l.effect, weight))
        return out

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

    def counterfactual(self, cause: str, target: str,
                       factual: float = 1.0, counter: float = 0.0) -> Counterfactual:
        """"Had ``cause`` been ``counter`` instead of ``factual``, what of ``target``?"
        Reuses :meth:`CausalModel.counterfactual` over the discovered structure."""
        model = self._structural_model()
        delta = model.counterfactual(cause, target, factual, counter)
        return Counterfactual(cause=cause, target=target, factual=factual,
                              counter=counter, delta=delta)

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
        return {"events": self._n, "variables": len(self._by_label),
                "interventions": sum(len(t) for t in self._int_by_label.values()),
                "valued_labels": len(self._values),
                "links": len(self._links),
                "causal_links": sum(1 for l in self._links.values() if l.is_causal),
                "mechanisms": len(self._mechanisms)}

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {"window": self.window,
                "stream": [e.to_dict() for e in self._stream],
                "links": [l.to_dict() for l in self._links.values()]}

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
                    self._mechanisms[(link.cause, link.effect)] = \
                        FunctionalCausalMechanism.from_dict(md)
                except Exception:  # noqa: BLE001 — a bad mechanism never blocks a restore
                    pass

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
