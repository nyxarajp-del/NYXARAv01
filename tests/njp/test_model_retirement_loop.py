"""Phase 4 ⑥ — giving up a model, and the deadlock that made it unreachable.

The plan listed ⑥ as *"fully built, never called in a loop"*. That was already out of date:
``brain.py`` calls ``field.retire_cycle`` on every turn and ``tests/njp/test_retire_cycle.py``
covers the trial's four brakes. What the measurement found instead is worse and quieter — the
loop was called on every turn of every session and could never do anything, because
:meth:`~nyxara.njp.universe.InternalUniverse.should_retire`'s third condition was circular:

* it required at least one pair reported by :meth:`~nyxara.njp.universe.InternalUniverse.ambiguous`
  — a pair with **neither** direction oriented;
* the only thing in the package that produces such a pair is :meth:`retire`, which un-orients the
  directions she inferred.

So retirement was gated on a state only retirement could create. These tests hold that fix in
place: :func:`test_the_retirement_deadlock_is_broken` is the falsifier, and it fails if the gate
goes back to asking `ambiguous()`.
"""

from __future__ import annotations


from nyxara.njp.universe import InternalUniverse, Orientation, _MIN_DEATHS


def _inferred(noisy: bool = False) -> InternalUniverse:
    """A universe that has inferred a direction for itself from consistent word order.

    ``noisy`` gives the arrow a *poor fit*, which is a separate condition from being surprised and
    is checked separately by ``should_retire``. It takes a small slope against the same wobble
    rather than a big one: at slope 1.5 the signal still dominates and R² comes back at 0.92, over
    the refit threshold, so the model is correctly judged a coefficient problem and never reaches
    the third condition at all.
    """
    universe = InternalUniverse()
    slope = 0.4 if noisy else 1.5
    for i in range(10):
        wobble = ((7 * i) % 5 - 2) if noisy else 0
        universe.observe({"p.x": float(i), "p.y": 3.0 + slope * i + wobble},
                         order=["p.x", "p.y"])
    return universe


def _surprise(universe: InternalUniverse, times: int) -> None:
    """Make the model repeatedly, confidently wrong — the only thing that earns a retirement."""
    for i in range(times):
        roll = universe.imagine("probe", {"p.x": float(i)}, steps=1)
        universe.grade(roll, {"p.y": 500.0 + i})


# --------------------------------------------------------------------------- #
# The falsifier
# --------------------------------------------------------------------------- #
def test_the_retirement_deadlock_is_broken():
    """A model that has failed enough must be retirable *before* anything is ambiguous.

    This is the whole finding in one assertion. If the third condition goes back to reading
    `ambiguous()`, `should_retire` returns None here and this fails.
    """
    universe = _inferred(noisy=True)
    assert universe.ambiguous() == [], "nothing is ambiguous yet — that is the point"
    assert universe.retirable(), "an inferred arrow with a fitted reverse is retirable"
    _surprise(universe, _MIN_DEATHS + 2)
    assert universe.should_retire() is not None


def test_retiring_creates_the_ambiguity_it_used_to_require():
    """Withdrawing an inference is what opens the question — measured in both directions."""
    universe = _inferred()
    assert universe.ambiguous() == []
    version = universe.compile().version
    assert universe.retire(version, why="probe") == 1
    assert universe.ambiguous() == [("p.x", "p.y")]
    assert universe.restore(version)
    assert universe.ambiguous() == []


# --------------------------------------------------------------------------- #
# The three conditions, each rejecting on its own
# --------------------------------------------------------------------------- #
def test_a_model_that_has_not_failed_is_not_retired():
    universe = _inferred(noisy=True)
    assert universe.should_retire() is None


def test_a_model_whose_arrows_still_fit_is_a_coefficient_problem():
    """Good R² means refitting will handle it, and `observe` refits on every observation."""
    universe = _inferred()                      # clean data: R² is 1.0
    _surprise(universe, _MIN_DEATHS + 2)
    assert universe.should_retire() is None


def test_a_model_with_nothing_of_its_own_to_give_up_is_not_retired():
    """Asserted directions came from outside the model; retiring them is discarding evidence."""
    universe = InternalUniverse()
    universe.declare("p.x", "p.y", sign=1)
    for i in range(10):
        universe.observe({"p.x": float(i), "p.y": 3.0 + 0.4 * i + ((7 * i) % 5 - 2)},
                         order=["p.x", "p.y"])
    # Fitted and usable — so it is the *orientation* that makes it unretirable, not a bad fit.
    assert universe.relations[("p.x", "p.y")].usable
    assert universe.retirable() == []
    _surprise(universe, _MIN_DEATHS + 2)
    assert universe.should_retire() is None


def test_retirable_needs_the_reverse_to_be_fitted():
    """Withdrawing an inference with nothing to re-derive from leaves a hole, not a question."""
    universe = _inferred()
    reverse = universe.relations.get(("p.y", "p.x"))
    assert reverse is not None
    reverse.n = 0                               # unfit the reverse: nothing to fall back on
    assert universe.retirable() == []


def test_only_inferred_directions_die():
    """Testimony and intervention did not come from the model under suspicion."""
    universe = _inferred()
    forward = universe.relations[("p.x", "p.y")]
    assert forward.orientation == Orientation.INFERRED
    forward.orientation = Orientation.VERIFIED
    assert universe.retirable() == []
    assert universe.retire(universe.compile().version, why="probe") == 0


def test_a_retirement_is_reversible_byte_for_byte():
    universe = _inferred()
    before = {k: r.orientation for k, r in universe.relations.items()}
    version = universe.compile().version
    universe.retire(version, why="probe")
    assert {k: r.orientation for k, r in universe.relations.items()} != before
    assert universe.restore(version)
    assert {k: r.orientation for k, r in universe.relations.items()} == before


# --------------------------------------------------------------------------- #
# ⑦ — the ambiguity a retirement opens becomes an experiment
# --------------------------------------------------------------------------- #
class _Designer:
    def __init__(self) -> None:
        self.proposed = []

    def propose(self, name, *, probability=0.5, predictions=None):
        self.proposed.append((name, dict(predictions or {})))
        return object()


def test_a_retirement_feeds_the_experiment_designer():
    """The chain the phase exists for: surprise → retirement → ambiguity → two rivals.

    `seed_ambiguities` had no caller anywhere in the package. This is the whole point of giving it
    one: the pairs it reads only exist in the window a retirement opens.
    """
    universe = _inferred()
    designer = _Designer()
    assert universe.seed_ambiguities(designer) == 0, "nothing is ambiguous before retirement"

    universe.retire(universe.compile().version, why="probe")
    seeded = universe.seed_ambiguities(designer)
    assert seeded == 2, "a pair is two rivals, or there is nothing to select between"
    names = {name for name, _p in designer.proposed}
    assert names == {"p.x->p.y", "p.y->p.x"}


def test_the_rivals_disagree_about_the_intervention():
    """Two hypotheses that predict the same thing carry no information — this is that check."""
    universe = _inferred()
    designer = _Designer()
    universe.retire(universe.compile().version, why="probe")
    universe.seed_ambiguities(designer)
    predictions = dict(designer.proposed)
    assert predictions["p.x->p.y"] != predictions["p.y->p.x"]


def test_seeding_is_safe_without_a_designer():
    assert _inferred().seed_ambiguities(None) == 0


# --------------------------------------------------------------------------- #
# The whole chain, in one object
# --------------------------------------------------------------------------- #
def test_the_full_chain_runs_from_surprise_to_a_seeded_experiment():
    """surprise → retirement → ambiguity → two rivals the designer can tell apart.

    Every link of this existed before Phase 4 and the chain had never run once, because its first
    link was gated on its third. Each assertion below is a link, in order, so a break says which
    one went.
    """
    from nyxara.njp.field import RecursiveCognitiveField

    universe = _inferred(noisy=True)
    designer = _Designer()
    field = RecursiveCognitiveField(None)
    field.universe = universe
    field.designer = designer

    # 1. Surprised repeatedly, by a model whose arrows do not fit well enough to refit out of it.
    _surprise(universe, _MIN_DEATHS + 2)
    version = universe.should_retire()
    assert version, "link 1: the model has failed enough to be given up"

    # 2. What dies is what she inferred, and it is archived rather than deleted.
    assert universe.retire(version, why="repeated surprise") == 1, "link 2: the inference is gone"
    assert version in universe.retired

    # 3. Giving it up opens the question.
    assert universe.ambiguous() == [("p.x", "p.y")], "link 3: the pair is Markov-equivalent again"

    # 4. And the question becomes an experiment, through the caller that did not exist.
    from nyxara.njp.field import CycleReport

    report = CycleReport()
    field._seed_ambiguities(report)
    assert report.ambiguities_seeded == 2, "link 4: two rivals, or nothing to select between"
    assert field.ambiguities_seeded == 2
    assert field.stats()["ambiguities_seeded"] == 2


def test_the_cycle_seeds_nothing_while_nothing_is_ambiguous():
    """The control. If this seeded anyway the counter would be measuring the call, not the state."""
    from nyxara.njp.field import CycleReport, RecursiveCognitiveField

    field = RecursiveCognitiveField(None)
    field.universe = _inferred()
    field.designer = _Designer()
    report = CycleReport()
    field._seed_ambiguities(report)
    assert report.ambiguities_seeded == 0
    assert field.stats()["ambiguities_seeded"] == 0
