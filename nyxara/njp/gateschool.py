"""NYXARA · njp/gateschool.py — does the gate that lets her rewrite herself ever say no (🚦📏).

:mod:`nyxara.njp.evolve` is the most consequential thing in this package: it edits her own source.
Its safeguards are real and carefully built — a profiler that nominates the target so nobody hand-
picks it, byte-exact rollback, a protected core that is refused, a ledger consulted before the next
edit so a regression stops the one after it, and at the centre a
:class:`~nyxara.njp.truth.TruthGauntlet` over **held-out** samples.

Its docstring makes one claim above all the others:

    An edit whose improvement is only visible on the samples that motivated it is fitting noise,
    and it is refused here rather than discovered later.

**That claim has never been tested.** The existing tests are good and they are all one-case
mechanism checks with the answer built into the fixture: a protected path is refused, a failed
gauntlet rolls back, a claim with no evidence at all is refused. None of them hands the gate an
edit that *does* have evidence — evidence that happens to be worthless — and asks what it does.

For a system that rewrites itself, the value of a gate is entirely in what it **refuses**, and a
gate nobody has watched refuse is a gate nobody knows the strength of. A permissive one does not
fail loudly; it degrades the thing it guards, one accepted edit at a time.

So: five kinds of candidate whose worth is known by construction, and the rate at which each is let
through.

* **real** — faster on samples it never saw, and no capability lost. Should pass.
* **overfit** — faster on exactly the samples that motivated it and nowhere else. Should fail, and
  this is the one the docstring promises.
* **noisy** — a coin per sample, no gain on average. Should fail.
* **harmful** — genuinely faster, and something else broke. Should fail.
* **null** — indistinguishable from what is already there. Should fail.

And the ablation that shows the mechanism is doing the work rather than the arithmetic: judge the
same candidates on **the samples that motivated them** instead of held-out ones. If the overfit
candidates sail through then, the held-out draw is what is refusing them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from nyxara.njp.evolve import PASS_RATIO
from nyxara.njp.truth import PredictiveSource, TruthGauntlet, Verdict

__all__ = ["Candidate", "Report", "KINDS", "judge", "examine", "sweep", "run",
           "SEED", "MIN_SAMPLES", "PASS_RATIO", "MIN_GAIN"]

SEED = 81

#: Taken from :mod:`nyxara.njp.evolve` rather than copied, so a bar changed there cannot leave this
#: school quietly measuring a gate that no longer exists. `PASS_RATIO` is imported; the rest are
#: mirrored because they are arguments rather than module constants.
MIN_SAMPLES = 2
HOLDOUT = 8

#: The margin :meth:`~nyxara.njp.evolve.SelfEvolver._predict_gain` already requires — a sample
#: counts as faster only at ``after <= before * (1 - MIN_GAIN)``.
#:
#: The first version of this school left it out, and measured a gate looser than the one that
#: exists: 0.112 of bad edits let through, against **0.021** for the real thing. Modelling a
#: mechanism as weaker than it is overstates its faults exactly as reliably as modelling it as
#: stronger hides them, and it is the same error as drawing a negative control that is easier to
#: reject than the finding was to accept.
MIN_GAIN = 0.05

#: The five kinds, and what each should get. Only one of them should pass, which is the whole shape
#: of the result: a gate is judged on refusals.
KINDS: Tuple[Tuple[str, bool], ...] = (
    ("real", True), ("overfit", False), ("noisy", False),
    ("harmful", False), ("null", False),
)


@dataclass
class Candidate:
    """A proposed edit, and what is actually true of it.

    ``motivating`` are the samples that suggested the edit; ``holdout`` are samples it was never
    tuned against. The gate is supposed to look only at the second, and the ablation checks what
    happens when it looks at the first.
    """

    kind: str = ""
    motivating: List[Tuple[float, float]] = field(default_factory=list)
    holdout: List[Tuple[float, float]] = field(default_factory=list)
    #: How much capability the edit costs, in the units the brain reports. Negative is a loss.
    capability: float = 0.0


def _gains(rng: random.Random, n: int, gain: float, spread: float = 0.05
           ) -> List[Tuple[float, float]]:
    """`n` samples as (before, after) milliseconds, where `after` is `gain` faster on average."""
    out: List[Tuple[float, float]] = []
    for _ in range(n):
        before = rng.uniform(8.0, 20.0)
        after = before * (1.0 - gain + rng.uniform(-spread, spread))
        out.append((before, max(0.1, after)))
    return out


def candidate(kind: str, rng: random.Random, *, holdout: int = HOLDOUT) -> Candidate:
    """One proposed edit of a given kind, built so its worth is not a matter of opinion."""
    if kind == "real":
        return Candidate(kind, _gains(rng, holdout, 0.30), _gains(rng, holdout, 0.30))
    if kind == "overfit":
        # The thing the docstring promises to catch: a real gain exactly where it was measured,
        # and none at all anywhere else.
        return Candidate(kind, _gains(rng, holdout, 0.30), _gains(rng, holdout, 0.0))
    if kind == "noisy":
        # No gain on average, and a wide enough spread that some samples look better by luck.
        return Candidate(kind, _gains(rng, holdout, 0.0, spread=0.35),
                         _gains(rng, holdout, 0.0, spread=0.35))
    if kind == "harmful":
        # Genuinely faster, and something else broke. Speed alone must not carry it.
        return Candidate(kind, _gains(rng, holdout, 0.40), _gains(rng, holdout, 0.40),
                         capability=-0.20)
    return Candidate(kind, _gains(rng, holdout, 0.0, spread=0.01),
                     _gains(rng, holdout, 0.0, spread=0.01))


def _predict(cand: Candidate) -> Any:
    """The gate's own test, as :meth:`~nyxara.njp.evolve.SelfEvolver._predict_gain` states it.

    Two conditions and both required, because either alone is gameable: this sample got faster,
    **and** nothing it could measure got worse. An edit that halves the time and loses accuracy has
    not improved anything.
    """
    def _check(_claim: str, sample: Tuple[float, float]) -> bool:
        before, after = sample
        return after <= before * (1.0 - MIN_GAIN) and cand.capability >= 0.0
    return _check


def judge(cand: Candidate, *, on_holdout: bool = True,
          pass_ratio: float = PASS_RATIO) -> str:
    """Put the edit's claim through the gauntlet as :mod:`nyxara.njp.evolve` does.

    ``on_holdout=False`` is the ablation: judge it on the samples that motivated it instead. That
    is not a straw man — it is what any system does that measures an improvement on the data that
    suggested it, which is the single most common way a self-improving loop degrades itself.
    """
    samples = cand.holdout if on_holdout else cand.motivating
    gauntlet = TruthGauntlet(
        sources=[PredictiveSource(predictor=_predict(cand), holdout=samples,
                                  min_samples=MIN_SAMPLES, pass_ratio=pass_ratio)],
        min_sources=1, require_hard=True)
    return gauntlet.judge("this edit makes her faster and no less accurate").verdict


def _passes(verdict: str) -> bool:
    """What counts as a promotion. Anything short of established leaves the edit on the shelf."""
    return verdict == Verdict.ESTABLISHED


@dataclass
class Report:
    """How the gate treated each kind of candidate."""

    on_holdout: bool = True
    pass_ratio: float = PASS_RATIO
    passed: Dict[str, int] = field(default_factory=dict)
    seen: Dict[str, int] = field(default_factory=dict)

    def rate(self, kind: str) -> float:
        return round(self.passed.get(kind, 0) / self.seen[kind], 4) if self.seen.get(kind) else 0.0

    @property
    def let_through(self) -> float:
        """Of the candidates that should have been refused, the share that were not.

        The number that matters. A gate is judged on refusals, and this one is the rate at which
        the thing it guards gets quietly worse.
        """
        bad = [k for k, good in KINDS if not good]
        seen = sum(self.seen.get(k, 0) for k in bad)
        through = sum(self.passed.get(k, 0) for k in bad)
        return round(through / seen, 4) if seen else 0.0

    @property
    def turned_away(self) -> float:
        """Of the candidates that should have passed, the share refused. The cost of the gate."""
        good = [k for k, ok in KINDS if ok]
        seen = sum(self.seen.get(k, 0) for k in good)
        through = sum(self.passed.get(k, 0) for k in good)
        return round(1.0 - through / seen, 4) if seen else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"on_holdout": self.on_holdout, "pass_ratio": self.pass_ratio,
                "by_kind": {k: self.rate(k) for k, _ in KINDS},
                "let_through": self.let_through, "turned_away": self.turned_away}

    def render(self) -> str:
        where = "held-out samples" if self.on_holdout else "the samples that motivated it"
        rows = "   ".join(f"{k} {self.rate(k):.3f}" for k, _ in KINDS)
        return (f"  judged on {where:<32} {rows}\n"
                f"  {'':<42} let through {self.let_through:.3f}   "
                f"turned away {self.turned_away:.3f}")


def examine(each: int = 200, *, on_holdout: bool = True, seed: int = SEED,
            pass_ratio: float = PASS_RATIO) -> Report:
    """Show the gate `each` candidates of every kind and count what it let through."""
    rng = random.Random(seed)
    out = Report(on_holdout=on_holdout, pass_ratio=pass_ratio)
    for kind, _good in KINDS:
        for _ in range(each):
            out.seen[kind] = out.seen.get(kind, 0) + 1
            if _passes(judge(candidate(kind, rng), on_holdout=on_holdout,
                             pass_ratio=pass_ratio)):
                out.passed[kind] = out.passed.get(kind, 0) + 1
    return out


def sweep(each: int = 200, ratios: Tuple[float, ...] = (0.75, 0.80, 0.90, 1.00)
          ) -> List[Report]:
    """What raising the share of held-out samples that must improve is worth."""
    return [examine(each, pass_ratio=r) for r in ratios]


def run(each: int = 200) -> Dict[str, Any]:  # pragma: no cover — a report
    print("five kinds of proposed edit, and the rate at which each is let through.")
    print("only `real` should pass: a gate is judged on what it refuses.\n")
    held = examine(each, on_holdout=True)
    print(held.render())
    print()
    tuned = examine(each, on_holdout=False)
    print(tuned.render())
    print(f"\n  the held-out draw is worth {tuned.let_through - held.let_through:+.3f} of "
          f"let-through rate — that is what the mechanism is doing")
    print("\n  and what raising the share that must improve would buy:\n")
    print("  pass ratio | let through   turned away")
    rows = sweep(each)
    for row in rows:
        print(f"  {row.pass_ratio:>10.2f} | {row.let_through:>11.4f}   {row.turned_away:>11.4f}")
    return {"held_out": held.to_dict(), "on_motivating": tuned.to_dict(),
            "guard_worth": round(tuned.let_through - held.let_through, 4),
            "ratios": [r.to_dict() for r in rows]}
