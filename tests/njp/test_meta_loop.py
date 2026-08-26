"""Loop 2 — evaluate herself, find the bottleneck, change one thing, let held-out data decide.

``meta_accepted`` and ``meta_gain`` were zero over every session, and the pipeline's mechanics
were sound: it identified a bottleneck, proposed a bounded reversible change, sandboxed it,
benchmarked it on the held-out fold and ran an adversarial battery. Two things made it unable to
ever finish:

**The detector and the adjudicator measured different things.** ``find_bottleneck`` reads
organ-specific symptoms — compression, answerable share, unresolved experiments — and never the
held-out benchmark every trial is judged by. So a configuration the benchmark scores as badly
degraded was invisible to it. Worse, the two can point opposite ways: refusing to form concepts
at all *raises* compression (fewer, larger kinds) while collapsing coverage, so the organ metric
reports health exactly when the benchmark reports collapse.

**Two of three knobs cannot move the benchmark at all**, and they are offered first. Sweeping
``similarity`` and ``invariant_share`` across their legal ranges leaves the score identical to six
decimal places, because ``concepts._thresholds`` already searches a band around ``similarity`` and
keeps the best — so the base value cannot matter. Those trials were being recorded as "no gain",
which reports a failed experiment where no experiment happened.

Nothing here loosens the accept rule. A trial is still accepted only if it strictly wins on data
it was not tuned on and survives the adversarial battery; ties still revert.
"""

from __future__ import annotations

from nyxara.njp import NJPBrain


def _taught() -> NJPBrain:
    brain = NJPBrain()
    session = ["birds need water", "a sparrow is a bird", "a crow is a bird", "a robin is a bird",
               "plants need water", "plants need light", "a rose is a plant", "a tulip is a plant",
               "sparrows need water", "crows need water", "roses need light", "tulips need light"]
    for _ in range(4):
        for line in session:
            brain.think(line)
    return brain


def test_a_degraded_configuration_is_visible_as_a_bottleneck():
    """The measure she is judged by is a bottleneck in its own right.

    Measured before this: min_members forced to 4 takes the benchmark from 0.779 to 0.458 — a 41%
    fall — and find_bottleneck answered "no bottleneck worth acting on" on every cycle, forever.
    """
    brain = _taught()
    field = brain.field
    healthy = field.benchmark()
    field.concepts.min_members = 4
    degraded = field.benchmark()
    assert degraded < healthy, (healthy, degraded)
    bottleneck = field.find_bottleneck()
    assert bottleneck is not None, field.stats()


def test_a_healthy_configuration_finds_nothing_to_fix():
    """A loop that always finds something wrong is a loop whose findings carry no information."""
    brain = _taught()
    field = brain.field
    assert field.benchmark() > 0.7
    trial = field.meta_cycle()
    assert not trial.accepted
    assert trial.why == "no bottleneck worth acting on", trial.why


def test_a_knob_the_benchmark_cannot_see_is_reported_as_unmeasurable():
    """Exactly equal, not merely close: no measurement happened, so no trial was decided."""
    brain = _taught()
    field = brain.field
    field.concepts.min_members = 4
    seen = []
    for _ in range(4):
        trial = field.meta_cycle()
        seen.append(trial.why)
        if trial.accepted:
            break
    assert field.stats()["meta_unmeasurable"] > 0, seen
    assert any("cannot see this knob" in why for why in seen), seen


def test_the_loop_reaches_an_improvement_and_keeps_it():
    """The whole of loop 2, end to end, on a configuration that genuinely can be improved."""
    brain = _taught()
    field = brain.field
    field.concepts.min_members = 4
    before = field.benchmark()
    for _ in range(10):
        if field.meta_cycle().accepted:
            break
    stats = field.stats()
    assert stats["meta_accepted"] >= 1, stats
    assert stats["meta_gain"] > 0.0, stats
    # The change was kept, not merely scored.
    assert field.benchmark() > before, (before, field.benchmark())
    assert field.concepts.min_members != 4


def test_a_change_that_does_not_strictly_win_is_reverted():
    """Ties revert — accepting them is how a system drifts a long way on no evidence."""
    brain = _taught()
    field = brain.field
    field.concepts.min_members = 4
    trials = [field.meta_cycle() for _ in range(4)]
    rejected = [t for t in trials if not t.accepted and t.modification is not None]
    assert rejected, [t.why for t in trials]
    for trial in rejected:
        holder = {"concepts": field.concepts, "universe": field.universe}.get(
            trial.modification.organ)
        if holder is None:
            continue
        current = getattr(holder, trial.modification.knob)
        assert float(current) == float(trial.modification.before), trial.why
