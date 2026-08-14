"""NYXARA · njp/world.py — event → state → cause → consequence (🌍, NJP V.02).

Grounding (:mod:`nyxara.njp.grounding`) gives her facts that hold: *the Master's name is Jay*.
This module gives her facts that **change**: things happen, they have causes, and they leave the
world different afterwards. A mind with only the first kind knows a snapshot; it cannot answer
*"what happens next"*, *"why did that happen"*, or *"would it still have happened if X hadn't"*.

Three layers, and they do different jobs on purpose:

**Events.** An :class:`Event` is ``actor → action → object`` at a time, with the preconditions it
needed and the effects it left. Events are ordered, so *before* and *after* are real relations she
can traverse rather than an accident of storage order.

**Causes.** A cause is *not* inferred from adjacency. Two things happening in sequence is the
weakest possible evidence, and a model that treats it as causation will confidently tell you the
rooster causes the sunrise. So a causal link needs the candidate to have **repeatedly** preceded
the effect *and* to beat the base rate at which the effect happens anyway — the difference between
``P(effect | cause)`` and ``P(effect)``. Below that margin the link is reported as correlation and
labelled as such.

**Dynamics.** State transitions go to :class:`~nyxara.mind.world_model.WorldModel`, the repo's
existing kNN dynamics model, which already carries ``imagine``, ``rollout``, ``counterfactual``
and ``intervene`` with calibrated confidence. This module does not reimplement any of that; it
supplies the state encoding and reads the answers back.

**Counterfactuals** — *"agar X nahi hota to Y phir bhi hota?"* — are answered by replaying the
history with the named event removed and asking whether the effect still follows. That is a real
re-derivation over her recorded causal structure, and when the evidence is too thin to support one
it says so instead of guessing, which is the only honest thing to do with a question about a world
that never happened.

Honest about the limits, as everywhere here: this is **observational** causal inference over what
she happened to see. She does not run experiments, so confounding is possible and a
:class:`CausalLink` reports the support it actually has rather than a certainty it has not earned.
Fail-soft throughout.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = ["Event", "CausalLink", "Explanation", "Expectation", "Counterfactual", "WorldView"]


@dataclass
class Event:
    """Something that happened, in order, with what it needed and what it changed."""

    actor: str = ""
    action: str = ""
    object: str = ""
    at: float = field(default_factory=time.time)
    seq: int = 0                                    # monotonic — the real ordering
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    text: str = ""
    # Effect kinds this particular occurrence has already been credited with, so one occurrence
    # of a cause cannot be counted twice for the same effect. See `WorldView.observe`.
    credited: Set[str] = field(default_factory=set, repr=False, compare=False)

    @property
    def key(self) -> str:
        """What makes two events the *same kind* of event. Instances differ; kinds recur."""
        return f"{self.action}:{self.object}" if self.object else str(self.action)

    def to_dict(self) -> Dict[str, Any]:
        return {"actor": self.actor, "action": self.action, "object": self.object,
                "seq": self.seq, "key": self.key,
                "preconditions": self.preconditions[:8], "effects": self.effects[:8]}


def _key_of(name: str) -> str:
    """Normalise a thing's name to the same shape :attr:`Event.key` produces for a bare action.

    A stated law names *things* ("aag", "garmi"), an event names a *happening* ("boils:100").
    Both end up as keys in one table, so the only discipline needed is that the same thing
    written two ways lands on one key — otherwise "Aag" and "aag " are two causes with half the
    evidence each and neither ever clears a threshold.
    """
    return " ".join(str(name or "").strip().lower().split())


@dataclass
class CausalLink:
    """``cause → effect``, with the evidence that distinguishes it from coincidence."""

    cause: str = ""
    effect: str = ""
    together: int = 0          # times the effect followed the cause
    cause_total: int = 0       # times the cause happened at all
    effect_total: int = 0      # times the effect happened at all
    observations: int = 0      # total events observed, for the base rate
    # Times the link was *stated* as a standing law ("aag se garmi milti hai") rather than
    # inferred from two things happening near each other. Kept as its own number, never folded
    # into `together`, because the two are different evidence: coincidence needs repetition to
    # mean anything, testimony does not — and testimony can be wrong in ways coincidence cannot.
    stated: int = 0
    refuted: int = 0           # times an observation contradicted the stated law

    @property
    def conditional(self) -> float:
        """``P(effect | cause)``."""
        return (self.together / self.cause_total) if self.cause_total else 0.0

    @property
    def base_rate(self) -> float:
        """``P(effect)`` — how often it happens anyway."""
        return (self.effect_total / self.observations) if self.observations else 0.0

    @property
    def lift(self) -> float:
        """How much the cause raises the effect's odds. Zero or less ⇒ it explains nothing."""
        return self.conditional - self.base_rate

    @property
    def causal(self) -> bool:
        """Enough repetition and a real margin over the base rate — or a law she was told.

        The second clause is not a loosening of the first. Waiting for three coincidences before
        "fire causes heat" may be believed is the right rule for two events that merely co-occur
        and the wrong one for a general law the Master stated outright: the statement is direct
        evidence about the mechanism, not a sample from it. So a stated law counts from the first
        telling — and is dropped the moment anything refutes it, which is a standard coincidence
        is never held to. Testimony is admitted earlier and falsified harder.
        """
        if self.refuted:
            return False
        if self.stated:
            return True
        return self.cause_total >= _MIN_SUPPORT and self.lift >= _MIN_LIFT

    @property
    def strength(self) -> float:
        """How much to lean on this link. A stated law starts high but never at certainty."""
        if self.refuted:
            return 0.0
        observed = max(0.0, min(1.0, self.lift))
        if not self.stated:
            return observed
        # Repetition of a telling raises confidence with diminishing returns and stops short of
        # 1.0: being told a thing many times is not the same as it being true.
        told = 1.0 - (0.5 ** self.stated)
        return max(observed, min(0.95, 0.5 + 0.45 * told))

    def to_dict(self) -> Dict[str, Any]:
        return {"cause": self.cause, "effect": self.effect, "together": self.together,
                "conditional": round(self.conditional, 4),
                "base_rate": round(self.base_rate, 4), "lift": round(self.lift, 4),
                "causal": self.causal, "support": self.cause_total,
                "stated": self.stated, "refuted": self.refuted,
                "strength": round(self.strength, 4)}


@dataclass
class Explanation:
    """Why something happened — the causes that actually earned the word."""

    effect: str = ""
    causes: List[CausalLink] = field(default_factory=list)
    correlates: List[CausalLink] = field(default_factory=list)   # seen together, not established
    text: str = ""

    @property
    def explained(self) -> bool:
        return bool(self.causes)

    def to_dict(self) -> Dict[str, Any]:
        return {"effect": self.effect, "text": self.text, "explained": self.explained,
                "causes": [c.to_dict() for c in self.causes],
                "correlates": [c.to_dict() for c in self.correlates]}


@dataclass
class Expectation:
    """What she expects next, and how much of a reason she has to expect it."""

    events: List[Tuple[str, float]] = field(default_factory=list)
    confidence: float = 0.0
    trusted: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"events": [[k, round(p, 4)] for k, p in self.events[:8]],
                "confidence": round(self.confidence, 4),
                "trusted": self.trusted, "reason": self.reason}


@dataclass
class Counterfactual:
    """*"Would Y still have happened without X?"* — with the honest answer when she cannot tell."""

    removed: str = ""
    effect: str = ""
    still_happens: Optional[bool] = None      # None ⇒ not enough evidence to say
    support: int = 0
    confidence: float = 0.0
    reason: str = ""

    @property
    def answerable(self) -> bool:
        return self.still_happens is not None

    def to_dict(self) -> Dict[str, Any]:
        return {"removed": self.removed, "effect": self.effect,
                "still_happens": self.still_happens, "answerable": self.answerable,
                "support": self.support, "confidence": round(self.confidence, 4),
                "reason": self.reason}


# A cause has to recur before it is a cause, and it has to beat the rate at which the effect
# happens anyway. Both floors are deliberately unglamorous: without them, anything that ever
# preceded anything becomes an explanation.
_MIN_SUPPORT = 3
_MIN_LIFT = 0.15


class WorldView:
    """Her model of what happens, why, and what would have happened otherwise."""

    def __init__(self, *, dynamics: Any = None, window: int = 4,
                 capacity: int = 20000, min_support: int = _MIN_SUPPORT,
                 min_lift: float = _MIN_LIFT) -> None:
        self.window = max(1, int(window))       # how far back a cause may reach
        self.capacity = max(64, int(capacity))
        self.min_support = max(1, int(min_support))
        self.min_lift = float(min_lift)

        self.events: List[Event] = []
        self._seq = 0
        # cause-key -> effect-key -> cause OCCURRENCES that were followed by that effect.
        # Denominator is `_counts[cause]`, so this answers "how often does the cause work".
        self._pairs: Dict[str, Dict[str, int]] = {}
        # effect-key -> cause-key -> effect OCCURRENCES that had that cause in their window.
        # Denominator is `_counts[effect]`, so this answers "how often did the effect arrive by
        # this route" — which is the question a counterfactual asks, and it is genuinely a
        # different count from `_pairs`. Deriving one from the other gives negative occurrences.
        self._preceded: Dict[str, Dict[str, int]] = {}
        self._counts: Dict[str, int] = {}       # event-key -> times observed
        # cause-key -> effect-key -> times the link was STATED as a standing law, and times an
        # observation refuted it. Separate from `_pairs` on purpose: see `CausalLink.stated`.
        self._stated: Dict[str, Dict[str, int]] = {}
        self._refuted: Dict[str, Dict[str, int]] = {}
        self._observations = 0
        self.dynamics = dynamics if dynamics is not None else self._build_dynamics()

    @staticmethod
    def _build_dynamics() -> Any:
        """The repo's existing kNN dynamics model. Absent ⇒ symbolic reasoning only."""
        try:
            from nyxara.mind.world_model import WorldModel
            return WorldModel(k=3)
        except Exception:  # noqa: BLE001
            return None

    # ---- observing -------------------------------------------------------- #
    def observe(self, event: Event) -> Event:
        """Record one event and update what follows what. The only way history grows."""
        try:
            self._seq += 1
            event.seq = self._seq
            self.events.append(event)
            self._observations += 1
            key = event.key
            self._counts[key] = self._counts.get(key, 0) + 1

            # Everything inside the window that preceded this is a *candidate* cause. Candidacy
            # is cheap; becoming a cause is not — that needs repetition and lift, checked in
            # `links()`. Recording the pair here is what makes the later test possible at all.
            #
            # Each prior **occurrence** is credited at most once per effect kind. Counting the
            # pair every time the two fall inside a window instead let `together` exceed
            # `cause_total`: with window=4 a single "rains" was credited for the "wet" that
            # followed it, and again for the next cycle's "wet" still inside its window, giving
            # P(wet | rains) = 200% and a counterfactual that reported minus eight occurrences.
            # A conditional probability above 1.0 is the arithmetic telling you the events were
            # being double-counted.
            window = self.events[-(self.window + 1):-1]
            for prior in window:
                if key in prior.credited:
                    continue
                prior.credited.add(key)
                bucket = self._pairs.setdefault(prior.key, {})
                bucket[key] = bucket.get(key, 0) + 1

            # The effect side: this one occurrence of `key` arrived with these causes available.
            # Counted per distinct cause KIND, once for this occurrence.
            arrived = self._preceded.setdefault(key, {})
            for cause_key in {p.key for p in window}:
                arrived[cause_key] = arrived.get(cause_key, 0) + 1

            if len(self.events) > self.capacity:
                del self.events[: len(self.events) - self.capacity]
            return event
        except Exception:  # noqa: BLE001
            return event

    def observe_transition(self, before: Sequence[float], action: str,
                           after: Sequence[float], *, reward: float = 0.0) -> None:
        """Hand a numeric state transition to the dynamics model, for rollout and imagination."""
        try:
            if self.dynamics is None:
                return
            self.dynamics.observe(tuple(before), action, tuple(after), reward=reward)
        except Exception:  # noqa: BLE001
            pass

    def from_grounding(self, result: Any) -> List[Event]:
        """Turn a grounded turn into events, when it described something happening.

        A relation that *holds* ("the Master's name is Jay") is a fact and belongs in the graph.
        A relation that *happened* ("the apple fell") is an event and belongs here. The two are
        genuinely different and conflating them would put unchanging facts on a timeline.
        """
        out: List[Event] = []
        try:
            for triple in (getattr(result, "triples", None) or []):
                # A *law* is neither of the two. "aag se garmi milti hai" did not happen at a
                # time and is not a property of a particular fire — it is a claim about how fire
                # and heat are related, and it is the only kind of sentence from which a causal
                # model can be built without waiting for the same accident three times. Routed
                # first, because some of these predicates would otherwise fall through as facts
                # and the mechanism they describe would be lost.
                if triple.predicate in _LAW_PREDICATES:
                    self.state_law(triple.subject, triple.object, kind=triple.predicate)
                    continue
                if triple.predicate not in _EVENT_PREDICATES:
                    continue
                out.append(self.observe(Event(
                    actor=triple.subject, action=triple.predicate, object=triple.object,
                    text=getattr(triple, "text", ""))))
            return out
        except Exception:  # noqa: BLE001
            return out

    # ---- causes ----------------------------------------------------------- #
    def state_law(self, cause: str, effect: str, *, kind: str = "causes") -> CausalLink:
        """Record a standing law the Master (or a source) stated outright.

        This is the intake the world model did not have, and its absence is why
        ``world.causal_links`` sat at 0 through sessions in which the Master explained how things
        work in plain sentences. Every other route in here waits for two things to *happen* near
        each other; a general claim about mechanism never happens at all, so it was invisible.

        A law is not an event: nothing goes on the timeline and `observations` does not move.
        Only the belief that this cause produces this effect gets stronger.
        """
        cause = _key_of(cause)
        effect = _key_of(effect)
        if not cause or not effect or cause == effect:
            return CausalLink(cause=cause, effect=effect)
        bucket = self._stated.setdefault(cause, {})
        bucket[effect] = bucket.get(effect, 0) + 1
        # A requirement is the same claim read from the other end: "the plant needs water" says
        # that *absence* of water produces death, so the link that carries predictive weight runs
        # from the requirement to the thing requiring it. Recorded under its own direction so
        # `why()` can answer both "why did it die" and "what does it need".
        if kind == "requires":
            back = self._stated.setdefault(effect, {})
            back[cause] = back.get(cause, 0)  # registers the pair without double-counting it
        return self.link(cause, effect)

    def refute_law(self, cause: str, effect: str) -> CausalLink:
        """Record an observation that contradicted a stated law. One is enough to kill it."""
        cause, effect = _key_of(cause), _key_of(effect)
        bucket = self._refuted.setdefault(cause, {})
        bucket[effect] = bucket.get(effect, 0) + 1
        return self.link(cause, effect)

    def link(self, cause: str, effect: str) -> CausalLink:
        """The evidence for one candidate ``cause → effect``, computed fresh."""
        return CausalLink(
            cause=cause, effect=effect,
            together=self._pairs.get(cause, {}).get(effect, 0),
            cause_total=self._counts.get(cause, 0),
            effect_total=self._counts.get(effect, 0),
            observations=self._observations,
            stated=self._stated.get(cause, {}).get(effect, 0),
            refuted=self._refuted.get(cause, {}).get(effect, 0))

    def links(self, *, causal_only: bool = True) -> List[CausalLink]:
        """Every candidate link, strongest first. ``causal_only`` applies support and lift."""
        out: List[CausalLink] = []
        try:
            # Both intakes, deduplicated: a link can be *observed* (it kept happening), *stated*
            # (she was told the law), or both — and a link she was told about but has never seen
            # is exactly the one worth surfacing, because it is the one she can go and test.
            seen: set = set()
            for source in (self._pairs, self._stated):
                for cause, effects in source.items():
                    for effect in effects:
                        if (cause, effect) in seen:
                            continue
                        seen.add((cause, effect))
                        got = self.link(cause, effect)
                        if causal_only and not self._is_causal(got):
                            continue
                        out.append(got)
            out.sort(key=lambda c: c.strength, reverse=True)
            return out
        except Exception:  # noqa: BLE001
            return out

    def _is_causal(self, link: CausalLink) -> bool:
        if link.refuted:
            return False
        if link.stated:
            return True
        return link.cause_total >= self.min_support and link.lift >= self.min_lift

    def why(self, effect: str) -> Explanation:
        """Why did this happen? Causes that earned the word; correlates named as correlates."""
        out = Explanation(effect=effect)
        try:
            for cause in self._pairs:
                got = self.link(cause, effect)
                if got.together <= 0:
                    continue
                (out.causes if self._is_causal(got) else out.correlates).append(got)
            out.causes.sort(key=lambda c: c.lift, reverse=True)
            out.correlates.sort(key=lambda c: c.together, reverse=True)
            if out.causes:
                best = out.causes[0]
                out.text = (f"{best.cause} — it preceded {effect} {best.together} of the "
                            f"{best.cause_total} times it happened "
                            f"({best.conditional:.0%} vs a {best.base_rate:.0%} base rate)")
            elif out.correlates:
                best = out.correlates[0]
                out.text = (f"I have seen {best.cause} before {effect} {best.together} time(s), "
                            f"but not often enough to call it a cause")
            else:
                out.text = f"I have nothing that precedes {effect}"
            return out
        except Exception:  # noqa: BLE001
            return out

    # ---- forward ---------------------------------------------------------- #
    def expect(self, after: str) -> Expectation:
        """What usually follows this, with the confidence that says whether to believe it."""
        out = Expectation()
        try:
            bucket = self._pairs.get(after, {})
            total = self._counts.get(after, 0)
            if not bucket or total <= 0:
                out.reason = f"I have not seen {after} lead anywhere yet"
                return out
            ranked = sorted(((k, n / total) for k, n in bucket.items()),
                            key=lambda kv: kv[1], reverse=True)
            out.events = ranked[:8]
            out.confidence = ranked[0][1] if ranked else 0.0
            # Trust needs both repetition and a *separated* winner. One observation of anything
            # gives a confident-looking 1.0, and a field of equal candidates gives a top pick that
            # means nothing — both are ways of being sure about nothing.
            separated = len(ranked) == 1 or (ranked[0][1] - ranked[1][1]) >= 0.15
            out.trusted = total >= self.min_support and separated
            out.reason = ("" if out.trusted else
                          f"seen {total} time(s); "
                          f"{'not enough' if total < self.min_support else 'no clear winner'}")
            return out
        except Exception:  # noqa: BLE001
            return out

    def imagine(self, start: Sequence[float], action: str, *, steps: int = 5) -> Any:
        """Roll the dynamics forward without acting. Delegated, never reimplemented."""
        try:
            if self.dynamics is None:
                return None
            return self.dynamics.imagine(tuple(start), action, steps=steps)
        except Exception:  # noqa: BLE001
            return None

    # ---- backward --------------------------------------------------------- #
    def counterfactual(self, removed: str, effect: str) -> Counterfactual:
        """*Would ``effect`` still have happened without ``removed``?*

        Answered from her own record: of the times the effect happened, how often did the removed
        event **not** precede it? If the effect regularly arrives by other routes, removing this
        one changes little; if it never arrives without it, this event looks necessary.

        Returns ``still_happens=None`` when the record cannot support either answer. That is the
        common case early on and it is reported rather than resolved by guessing — a confident
        answer about a world that never happened is the easiest kind of lie to tell.
        """
        out = Counterfactual(removed=removed, effect=effect)
        try:
            effect_total = self._counts.get(effect, 0)
            if effect_total < self.min_support:
                out.reason = (f"{effect} has only happened {effect_total} time(s) — too few to "
                              f"say what it depends on")
                return out

            # Effect-side count: of the times `effect` happened, how many had `removed` available
            # to cause it. `_pairs` answers the mirror-image question (how often the cause worked)
            # and using it here produced negative "without" counts.
            with_cause = self._preceded.get(effect, {}).get(removed, 0)
            without_cause = max(0, effect_total - with_cause)
            out.support = effect_total
            if with_cause <= 0:
                out.still_happens = True
                out.confidence = 0.9
                out.reason = f"{removed} never preceded {effect}, so removing it changes nothing"
                return out

            share_without = without_cause / effect_total
            out.still_happens = share_without >= 0.5
            # Confidence peaks when the record is lopsided either way and collapses at a coin
            # flip, which is exactly where she should not be committing to an answer.
            out.confidence = abs(share_without - 0.5) * 2.0
            out.reason = (f"{effect} happened {effect_total} time(s); {without_cause} of those "
                          f"without {removed}")
            if out.confidence < 0.2:
                out.still_happens = None
                out.reason += " — too even to call"
            return out
        except Exception:  # noqa: BLE001
            return out

    # ---- reading ---------------------------------------------------------- #
    def thin_regions(self, *, limit: int = 8) -> List[Tuple[str, int]]:
        """Events she has seen, but not often enough to predict anything after them.

        The honest definition of a gap in a causal model: something that *occurs* — so it is real
        and she has met it — but whose consequences she has too few observations of to say
        anything about. Events she has never seen at all are not gaps, they are simply outside her
        experience, and listing those would make curiosity infinite.
        """
        out: List[Tuple[str, int]] = []
        try:
            for key, count in self._counts.items():
                if count >= self.min_support * 3:
                    continue          # seen often enough to have learned something
                followers = sum((self._pairs.get(key) or {}).values())
                if followers >= self.min_support:
                    continue          # already predicts something from here
                out.append((key, count))
            out.sort(key=lambda kv: -kv[1])
            return out[: max(0, int(limit))]
        except Exception:  # noqa: BLE001
            return out

    def history(self, *, limit: int = 20) -> List[Event]:
        """The most recent events, oldest first — the timeline she actually reasons over."""
        return self.events[-max(1, int(limit)):]

    def timeline(self, key: str) -> List[Event]:
        """Every occurrence of one kind of event, in order."""
        return [e for e in self.events if e.key == key]

    def stats(self) -> Dict[str, Any]:
        causal = self.links(causal_only=True)
        return {
            "events": len(self.events), "observations": self._observations,
            "event_kinds": len(self._counts),
            "candidate_links": sum(len(v) for v in self._pairs.values()),
            "stated_laws": sum(len(v) for v in self._stated.values()),
            "refuted_laws": sum(len(v) for v in self._refuted.values()),
            "causal_links": len(causal),
            "strongest": causal[0].to_dict() if causal else None,
            "dynamics": (self.dynamics.stats() if self.dynamics is not None
                         and hasattr(self.dynamics, "stats") else None),
            "window": self.window, "min_support": self.min_support,
            "min_lift": round(self.min_lift, 4),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "events": [e.to_dict() | {"at": e.at, "text": e.text[:120]}
                       for e in self.events[-self.capacity:]],
            "pairs": {c: dict(e) for c, e in self._pairs.items()},
            "preceded": {e: dict(c) for e, c in self._preceded.items()},
            "stated": {c: dict(e) for c, e in self._stated.items()},
            "refuted": {c: dict(e) for c, e in self._refuted.items()},
            "counts": dict(self._counts), "observations": self._observations,
            "seq": self._seq,
        }

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.events = []
            for row in (d.get("events") or []):
                self.events.append(Event(
                    actor=str(row.get("actor", "")), action=str(row.get("action", "")),
                    object=str(row.get("object", "")), at=float(row.get("at", 0.0)),
                    seq=int(row.get("seq", 0)),
                    preconditions=list(row.get("preconditions") or []),
                    effects=list(row.get("effects") or []), text=str(row.get("text", ""))))
            self._pairs = {str(c): {str(k): int(v) for k, v in (e or {}).items()}
                           for c, e in (d.get("pairs") or {}).items()}
            self._preceded = {str(e): {str(k): int(v) for k, v in (c or {}).items()}
                              for e, c in (d.get("preceded") or {}).items()}
            self._stated = {str(c): {str(k): int(v) for k, v in (e or {}).items()}
                            for c, e in (d.get("stated") or {}).items()}
            self._refuted = {str(c): {str(k): int(v) for k, v in (e or {}).items()}
                             for c, e in (d.get("refuted") or {}).items()}
            self._counts = {str(k): int(v) for k, v in (d.get("counts") or {}).items()}
            self._observations = int(d.get("observations", 0))
            self._seq = int(d.get("seq", len(self.events)))
        except Exception:  # noqa: BLE001 — a corrupt sidecar leaves an empty history
            pass


# Relations that describe something *happening* rather than something *being*. Only these become
# events; the rest are facts and belong in the graph, where they are not on a timeline.
_EVENT_PREDICATES = frozenset({
    "broke", "fell", "opened", "closed",
    "started", "stopped", "failed", "crashed", "sent", "received", "bought", "sold",
    "did", "made", "went", "came", "happened",
    # Processes. "the water boiled" is a happening in exactly the sense the timeline means, and
    # it is what a stated law predicts — so the two intakes meet here: the law says water boils,
    # the timeline records that it did, and the difference between them is learning.
    "boils", "melts", "freezes", "burns", "grows", "dies", "rises", "breaks",
    "evaporates", "condenses", "expands", "contracts",
})

# Predicates that state a standing law rather than a happening. These go to `state_law`, never on
# the timeline. `causes` moved here from the event set: "X causes Y" was being recorded as a
# single event named "causes:Y" whose actor was thrown away, so the one predicate that literally
# names a causal relation contributed nothing to the causal model.
_LAW_PREDICATES = frozenset({
    "causes", "caused", "led_to", "leads_to", "produces", "requires", "needs", "emits",
})
