"""NYXARA · njp/space.py — when the hypotheses run out, which corner was never looked in (🗺, NJP V.84).

V.83 ran the diagnostic stack on a real organ and it came back *nothing tested here explains it*.
That was honest and it was a **terminal state**, which is the defect this module exists to remove.
*Every hypothesis I hold failed* and *therefore the cause is unknowable* are different claims, and
between them sits the one that is actually true:

    my model of what could be causing this is incomplete.

Two mechanisms, and they are useless apart.

**A map, so that emptiness is visible.** :class:`Knob` names one thing about a system that could be
varied, in a **region** — what the system is *given* (data), what it *does* (method), what it is
*shown* (representation), what it is *scored against* (measurement), how the world it runs on was
*built* (construction). The four repairs the attributor holds — more data, another algorithm,
richer readings, more budget — all land in two of those regions. That was invisible while the
hypotheses were a flat list, and is the first thing you see on a map.

**An intervention that says what it holds still.** V.83 caught a repair that shrank a candidate pool
and, unnoticed, threw the right answer out of it — by re-reading the ceiling afterwards. That check
was hardcoded because the ceiling was the only invariant anyone had thought of. Here an
:class:`Intervention` **declares** its invariants (``holds``) and they are measured after it runs.
``REMOVE(distractors)`` alone is the experiment that failed. ``REMOVE(distractors) HOLD_FIXED(the
right answer stays reachable)`` is the experiment that was owed, and the difference is a thing the
system can now write down rather than a thing I noticed by hand.

**What this does not do, said plainly.** It does not invent the axes. :class:`Knob` is supplied, the
way ``Benchmark.key`` and ``Benchmark.reachable`` are supplied, because *what could be varied* is
domain knowledge. The claim here is narrower and testable: **given the axes, does it notice that
every experiment landed in one corner, and does it stay quiet when they did not?** A detector that
answers *blind spot* to everything is worth nothing, so the exam scores silence as hard as it
scores catches — see :mod:`nyxara.njp.spaceschool`.

Pure standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["REGIONS", "VERBS", "Knob", "Held", "Intervention", "Outcome", "Map", "Gap",
           "survey", "gaps", "propose", "carry_out", "MOVED", "SLACK"]

#: Where a cause can live. Not a taxonomy of the world — a taxonomy of **what a run of a system is
#: made of**, so that "I varied four things" can be checked against "and they were all the same
#: kind of thing". The attributor's four repairs occupy `given` and `method` and nothing else.
REGIONS: Tuple[str, ...] = ("given", "method", "shown", "scored", "built")

#: The intervention language. A verb is not a repair — it is a **way of changing something**, and an
#: experiment is a verb applied to a knob while named quantities are held still. The menu is short
#: on purpose: these are the changes whose effect on a measurement is interpretable.
#:
#: ``hold`` is the one that earns its place here rather than in a list of nice ideas. V.83's whole
#: correction was that an experiment must change exactly one thing, enforced for a single hardcoded
#: quantity. ``hold`` makes that a property of the experiment instead of a property of the checker.
VERBS: Tuple[str, ...] = ("add", "remove", "reorder", "shuffle", "mask", "replace",
                          "perturb", "hold", "split", "merge", "invert", "isolate", "cross")

#: How far a score must move before the intervention is said to have done something. Shared with
#: :mod:`nyxara.njp.attribution` by value rather than by import, because they are the same question
#: asked of the same units and a second opinion on it would be a bug, not a feature.
MOVED = 0.03

#: How far a quantity an intervention promised to hold still may drift before the promise is broken.
#: Same role as `attribution.SPOILED`, generalised from one hardcoded ceiling to any declared
#: invariant.
SLACK = 0.05


@dataclass(frozen=True)
class Knob:
    """One thing about a system that could be varied, and where it lives.

    ``name`` is what it is; ``region`` is which part of a run it belongs to. Two knobs in the same
    region are near each other in the sense that matters here: an experiment that moves one of them
    tells you something about the other, and four experiments inside one region tell you nothing at
    all about the regions nobody entered.
    """

    name: str = ""
    region: str = ""
    says: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "region": self.region, "says": self.says}


@dataclass(frozen=True)
class Held:
    """A quantity an intervention promises not to disturb, and the means to check the promise.

    The means are required. A promise with no way to read it is not an invariant, it is an
    intention, and V.83 is the version that happened because one of those was trusted.
    """

    name: str = ""
    #: Read the quantity off whatever the experiment produced.
    read: Optional[Callable[[Any], float]] = None
    slack: float = SLACK

    def moved(self, before: Any, after: Any) -> Optional[float]:
        """How far it drifted, or ``None`` when it could not be read on both sides."""
        if self.read is None:
            return None
        try:
            return round(self.read(after) - self.read(before), 4)
        except Exception:  # noqa: BLE001 — an unreadable invariant is unknown, never unbroken
            return None

    def broken(self, before: Any, after: Any) -> bool:
        drift = self.moved(before, after)
        return drift is not None and abs(drift) >= self.slack


@dataclass
class Intervention:
    """A verb, a knob, and what must not move while it happens.

    This is the unit an experiment is *composed from* rather than selected from a menu. The four
    experiments :class:`~nyxara.njp.attribution.Failure` accepts are four fixed slots; these are
    ``(verb, knob, holds)`` triples, so ``remove`` applied to *distractors* while holding *the right
    answer is still reachable* is expressible without anybody having written that experiment down
    first. That triple is the one V.83 ended owing.
    """

    verb: str = ""
    on: str = ""                                   # a Knob name
    holds: Tuple[Held, ...] = ()
    #: Run it, and return whatever the invariants will be read off — typically a benchmark.
    run: Optional[Callable[[], Any]] = None
    #: The same change written as a **transformation of a setup** rather than as a thunk.
    #:
    #: ``run`` is opaque and therefore un-composable: two thunks cannot be applied one after the
    #: other, so ``A then B`` and ``B then A`` are not even expressible, and neither is the
    #: interaction between them. V.85 needs both, so an intervention that wants to take part in an
    #: interaction search supplies this instead — and gets ``run`` for free through
    #: :meth:`against`. An intervention with only ``run`` still works everywhere it did before and
    #: is simply not composable, which is reported rather than worked around.
    change: Optional[Callable[[Any], Any]] = None
    #: Could this change be **shipped**, or only measured? An intervention that consults the right
    #: answer — keep twelve candidates, but make sure the gold one is among them — answers a real
    #: diagnostic question and cannot become a repair, because at run time nobody knows the gold.
    #:
    #: The distinction is here because losing it is how a measurement gets announced as a fix. An
    #: oracle intervention that moves the score has established *where the cause is*; it has
    #: established nothing whatever about how to remove it, and V.83 is the version that showed
    #: those two are not the same problem.
    deployable: bool = True
    says: str = ""

    @property
    def name(self) -> str:
        held = " ".join(f"hold({h.name})" for h in self.holds)
        return f"{self.verb}({self.on})" + (f" {held}" if held else "")

    @property
    def composable(self) -> bool:
        return self.change is not None

    def against(self, setup: Any) -> "Intervention":
        """The same intervention with ``run`` filled in from ``change`` and a starting setup."""
        if self.change is None:
            return self
        return Intervention(verb=self.verb, on=self.on, holds=self.holds,
                            deployable=self.deployable, says=self.says, change=self.change,
                            run=lambda: self.change(setup))

    def to_dict(self) -> Dict[str, Any]:
        return {"verb": self.verb, "on": self.on, "name": self.name,
                "holds": [h.name for h in self.holds], "deployable": self.deployable,
                "says": self.says}


@dataclass
class Outcome:
    """What one intervention did, including to the things it promised not to touch."""

    intervention: Optional[Intervention] = None
    moved: float = 0.0
    #: Invariants that drifted past their slack. Non-empty means the number below is about nothing.
    broke: List[str] = field(default_factory=list)
    #: Invariants that could not be read at all — reported, never counted as kept.
    unread: List[str] = field(default_factory=list)
    ran: bool = False
    says: str = ""

    @property
    def verdict(self) -> str:
        """``spoiled`` | ``unvouched`` | ``moved`` | ``flat`` | ``not run``.

        ``unvouched`` is the one worth spelling out. An intervention promised to hold something
        still and that something could not be read afterwards, so nobody knows whether the promise
        was kept. Folding it in with the clean results would be treating an unrun check as a passed
        one, which is the mistake :mod:`nyxara.njp.measurement` was built around; calling it
        ``spoiled`` would be treating an unrun check as a failed one, which is no better. It is its
        own state, and it does not cover the region it was aimed at.
        """
        if not self.ran:
            return "not run"
        if self.broke:
            return "spoiled"
        if self.unread:
            return "unvouched"
        return "moved" if abs(self.moved) >= MOVED else "flat"

    @property
    def informative(self) -> bool:
        """Did this tell us anything we can stand on?

        A spoiled intervention did not, however large its number. Neither did one whose promises
        could not be checked — not because it is likely to have broken them, but because *this
        region has been examined* is a claim, and an unverifiable experiment does not support it.
        """
        return self.ran and not self.broke and not self.unread

    def to_dict(self) -> Dict[str, Any]:
        return {"intervention": self.intervention.name if self.intervention else "",
                "verdict": self.verdict, "moved": self.moved, "broke": self.broke,
                "unread": self.unread, "says": self.says}

    def render(self) -> str:
        mark = {"moved": " ! ", "flat": " x ", "spoiled": " ~ ",
                "unvouched": " ~ ", "not run": " ? "}
        name = self.intervention.name if self.intervention else "?"
        oracle = "" if self.intervention is None or self.intervention.deployable else " [oracle]"
        return f"  {mark.get(self.verdict, '   ')} {name}{oracle}\n      {self.says}"


def carry_out(what: Intervention, before: Any, *, score: Callable[[Any], float]) -> Outcome:
    """Run one intervention and check it against its own promises.

    The asymmetry is V.83's, generalised. An intervention that ran and kept everything it said it
    would keep produces a number that is evidence. One that broke a promise produces a number that
    is evidence about **nothing** — not weaker evidence, not evidence with a caveat — and reporting
    it as either a result or a refutation is the mistake that module was written to stop.
    """
    if what.run is None:
        return Outcome(intervention=what, says="no way to run it was supplied")
    try:
        after = what.run()
    except Exception as error:  # noqa: BLE001 — an experiment that cannot run says so
        return Outcome(intervention=what,
                       says=f"it raised {type(error).__name__}: {error}")
    moved = round(score(after) - score(before), 4)
    broke: List[str] = []
    unread: List[str] = []
    drifts: List[str] = []
    for held in what.holds:
        drift = held.moved(before, after)
        if drift is None:
            unread.append(held.name)
            continue
        if abs(drift) >= held.slack:
            broke.append(held.name)
            drifts.append(f"{held.name} moved {drift:+.4f}")
    out = Outcome(intervention=what, moved=moved, broke=broke, unread=unread, ran=True)
    if broke:
        out.says = (f"{score(before):.4f} → {score(after):.4f} ({moved:+.4f}), but "
                    + "; ".join(drifts) + " — it changed more than it meant to, so this tests nothing")
    elif unread:
        out.says = (f"{score(before):.4f} → {score(after):.4f} ({moved:+.4f}); "
                    + f"{', '.join(unread)} could not be read, so {'they are' if len(unread) > 1 else 'it is'} "
                    "not known to have held")
    else:
        # Direction is part of the finding. The first version said "this knob reaches it" to a
        # change that took a real organ's score to 0.0000, which is true and reads as progress.
        if moved >= MOVED:
            why = "this knob reaches it"
        elif moved <= -MOVED:
            why = "this knob reaches it, downward — it has leverage and this is the wrong way"
        else:
            why = "this knob does not reach it"
        out.says = f"{score(before):.4f} → {score(after):.4f} ({moved:+.4f}) — {why}"
    return out


@dataclass
class Gap:
    """A region of the map that no informative intervention ever entered."""

    region: str = ""
    knobs: List[str] = field(default_factory=list)
    #: Interventions that *tried* to enter it and told us nothing — a region is not covered by an
    #: experiment that spoiled itself, and calling it covered is how a blind spot hides.
    spoiled: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"region": self.region, "knobs": self.knobs, "spoiled": self.spoiled}


@dataclass
class Map:
    """The knobs, the interventions actually carried out, and what that leaves unexamined."""

    name: str = ""
    knobs: List[Knob] = field(default_factory=list)
    outcomes: List[Outcome] = field(default_factory=list)

    @property
    def touched(self) -> List[str]:
        """Knobs an **informative** intervention moved. A spoiled one touched nothing."""
        return sorted({o.intervention.on for o in self.outcomes
                       if o.informative and o.intervention is not None})

    @property
    def reached(self) -> List[str]:
        """Knobs whose intervention moved the score past :data:`MOVED`."""
        return sorted({o.intervention.on for o in self.outcomes
                       if o.informative and o.verdict == "moved" and o.intervention is not None})

    @property
    def regions(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for knob in self.knobs:
            out.setdefault(knob.region, []).append(knob.name)
        return out

    @property
    def covered(self) -> List[str]:
        seen = set(self.touched)
        return sorted(r for r, names in self.regions.items() if seen & set(names))

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "regions": self.regions, "touched": self.touched,
                "reached": self.reached, "covered": self.covered,
                "gaps": [g.to_dict() for g in gaps(self)],
                "outcomes": [o.to_dict() for o in self.outcomes]}

    def render(self) -> str:
        lines = [f"{self.name or 'the space'}:"]
        lines += [o.render() for o in self.outcomes]
        for region in sorted(self.regions):
            names = self.regions[region]
            done = [n for n in names if n in set(self.touched)]
            mark = "ok " if done else " ! "
            lines.append(f"  {mark} {region:<10} {len(done)}/{len(names)} knobs actually examined"
                         + ("" if done else "  ← nothing here was ever varied"))
        found = gaps(self)
        if found:
            lines.append("  → the hypotheses are exhausted and the map is not: "
                         + ", ".join(g.region for g in found))
        else:
            lines.append("  → every region was entered, so the space itself is not what is missing")
        return "\n".join(lines)


def survey(name: str, knobs: Sequence[Knob], interventions: Sequence[Intervention],
           before: Any, *, score: Callable[[Any], float]) -> Map:
    """Carry out every intervention and record what the map looks like afterwards."""
    out = Map(name=name, knobs=list(knobs))
    for what in interventions:
        out.outcomes.append(carry_out(what, before, score=score))
    return out


def gaps(the_map: Map) -> List[Gap]:
    """Regions nothing informative ever entered, emptiest first.

    Deliberately **not** "every knob that was not varied". A region with three knobs of which one
    was varied has been entered, and calling it a blind spot would make the detector fire on every
    map ever drawn — which is the failure mode that matters, because it is the one that looks like
    diligence. Only a region where *nothing* was informatively varied is reported.
    """
    seen = set(the_map.touched)
    out: List[Gap] = []
    for region, names in sorted(the_map.regions.items()):
        if seen & set(names):
            continue
        spoiled = sorted({o.intervention.name for o in the_map.outcomes
                          if o.intervention is not None and o.intervention.on in set(names)
                          and not o.informative})
        out.append(Gap(region=region, knobs=sorted(names), spoiled=spoiled))
    # A region something *attempted* and spoiled ranks first: somebody already judged it worth
    # entering and failed to, which is evidence. Nothing else here is.
    #
    # The first version also ranked by how many knobs a region held, and that was a fake signal —
    # a region's size says nothing about whether the cause is in it, and the exam caught it at once
    # by putting a three-knob innocent region above the two-knob guilty one. There is no honest
    # ordering between two regions that are simply both empty, so they come back alphabetically and
    # **both** are reported. A blind spot is a place to look, not a diagnosis; inventing a
    # preference between them would be exactly the confident wrong answer this package keeps
    # refusing to produce elsewhere.
    return sorted(out, key=lambda g: (-len(g.spoiled), g.region))


def propose(the_map: Map, *, verbs: Sequence[str] = VERBS) -> List[Intervention]:
    """Interventions the map says are worth building, with the invariants already attached.

    These are **not runnable** — ``run`` is empty, because what it would take to remove a
    distractor or reorder a candidate list is the domain's business and inventing that is a
    different and much harder claim than the one this module makes. What is produced is the
    *specification*: a verb, a knob in an unentered region, and the promises any honest version of
    that experiment has to keep — every invariant that a previously spoiled attempt already broke.

    That last clause is the useful part. The pool experiment failed in V.83 by dropping the gold
    answer; a proposal to enter that region again now arrives carrying ``hold(gold reachable)``,
    because the map remembers what the last attempt broke.
    """
    keep: Dict[str, Held] = {}
    for outcome in the_map.outcomes:
        if outcome.intervention is None:
            continue
        for held in outcome.intervention.holds:
            keep.setdefault(held.name, held)
        # Anything a spoiled attempt broke is a promise the next attempt must make explicitly.
        for name in outcome.broke:
            for held in outcome.intervention.holds:
                if held.name == name:
                    keep[name] = held
    out: List[Intervention] = []
    for gap in gaps(the_map):
        for knob in gap.knobs:
            for verb in verbs:
                if verb == "hold":            # `hold` is a promise, never the change itself
                    continue
                out.append(Intervention(
                    verb=verb, on=knob, holds=tuple(keep.values()),
                    says=(f"nothing has entered `{gap.region}`; {verb} on `{knob}` would, "
                          + (f"holding {', '.join(keep)}" if keep else "and holds nothing yet"))))
    return out
