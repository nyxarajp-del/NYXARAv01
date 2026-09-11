"""V.81 — does the gate that lets her rewrite her own source ever say no?

`njp/evolve.py` is the most consequential thing in this package. Its safeguards are real and its
existing tests are good — a protected path refused, a failed gauntlet rolled back byte-for-byte, a
regression blocking the next edit. Every one of them is a single case with the answer built into
the fixture, and none hands the gate an edit that *does* have evidence which happens to be
worthless.

For a self-rewriting loop the value of a gate is entirely in what it refuses, and a permissive one
does not fail loudly — it degrades the thing it guards, one accepted edit at a time. So what is
pinned here is a rate, per kind of candidate, and the ablation that shows which mechanism earns it.
"""

from __future__ import annotations

import random

from nyxara.njp.evolve import PASS_RATIO, SelfEvolver
from nyxara.njp.gateschool import (
    KINDS, MIN_GAIN, Candidate, Report, candidate, examine, judge, sweep,
)
from nyxara.njp.truth import Verdict


# --------------------------------------------------------------------------------------------- #
#  the claim the module makes about itself
# --------------------------------------------------------------------------------------------- #
def test_an_edit_that_improves_only_where_it_was_measured_is_refused():
    """The central promise of `evolve.py`, tested rather than asserted in prose."""
    got = examine(200)
    assert got.rate("overfit") == 0.0, got.to_dict()


def test_and_would_sail_through_if_judged_on_the_samples_that_motivated_it():
    """The ablation. Without it, 'we refuse overfit edits' is a sentence, not a mechanism."""
    got = examine(200, on_holdout=False)
    assert got.rate("overfit") == 1.0, got.to_dict()


def test_the_held_out_draw_is_what_is_doing_the_refusing():
    held, tuned = examine(200), examine(200, on_holdout=False)
    assert tuned.let_through - held.let_through > 0.15


# --------------------------------------------------------------------------------------------- #
#  the other three ways an edit can be worthless
# --------------------------------------------------------------------------------------------- #
def test_an_edit_that_is_faster_and_breaks_something_else_is_refused():
    """Speed alone is an objective a self-editor satisfies by deleting the work."""
    assert examine(200).rate("harmful") == 0.0


def test_an_edit_indistinguishable_from_what_is_there_is_refused():
    assert examine(200).rate("null") == 0.0


def test_a_noisy_edit_almost_never_gets_through():
    """The one leak, and it is small. Reported as a number rather than claimed to be zero."""
    assert examine(300).rate("noisy") < 0.02


def test_a_genuinely_better_edit_is_never_turned_away():
    """A gate is only worth having if it also lets the real thing past."""
    assert examine(200).rate("real") == 1.0
    assert examine(200).turned_away == 0.0


# --------------------------------------------------------------------------------------------- #
#  the bar, which was loose and is now measured
# --------------------------------------------------------------------------------------------- #
def test_raising_the_bar_cost_nothing_which_is_why_it_was_raised():
    """Every setting swept turns away zero real edits. There was no trade to make."""
    rows = sweep(150, ratios=(0.75, 0.80, 0.90, 1.00))
    assert all(r.turned_away == 0.0 for r in rows), [r.to_dict() for r in rows]


def test_the_looser_bar_let_more_through():
    loose = examine(300, pass_ratio=0.75)
    tight = examine(300, pass_ratio=PASS_RATIO)
    assert loose.let_through > tight.let_through


def test_the_evolver_uses_the_measured_bar():
    assert SelfEvolver().pass_ratio == PASS_RATIO == 0.90


def test_the_school_reads_the_bar_from_the_module_it_is_judging():
    """Copying it would let the two drift and leave this measuring a gate that no longer exists."""
    import nyxara.njp.evolve as evolve
    import nyxara.njp.gateschool as school
    assert school.PASS_RATIO is evolve.PASS_RATIO


# --------------------------------------------------------------------------------------------- #
#  the fixture models the gate as it is, not as it would be easier to beat
# --------------------------------------------------------------------------------------------- #
def test_the_predicate_requires_the_margin_the_real_one_requires():
    """Dropping `min_gain` measured a gate looser than the one that exists — 0.112 against 0.021.

    Modelling a mechanism as weaker overstates its faults exactly as reliably as modelling it as
    stronger hides them.
    """
    assert MIN_GAIN == 0.05
    barely = Candidate("x", [], [(10.0, 9.9)] * 8)
    assert judge(barely) != Verdict.ESTABLISHED


def test_a_candidate_of_each_kind_is_what_its_name_says():
    rng = random.Random(1)
    real, overfit = candidate("real", rng), candidate("overfit", rng)
    assert all(a > b for a, b in real.holdout)
    assert sum(1 for a, b in overfit.motivating if b < a * 0.95) >= 6
    assert sum(1 for a, b in overfit.holdout if b < a * 0.95) <= 2


def test_a_harmful_candidate_is_genuinely_faster_so_speed_alone_cannot_carry_it():
    rng = random.Random(1)
    harmful = candidate("harmful", rng)
    assert all(b < a for a, b in harmful.holdout) and harmful.capability < 0


# --------------------------------------------------------------------------------------------- #
#  the counting
# --------------------------------------------------------------------------------------------- #
def test_let_through_counts_only_the_candidates_that_should_have_been_refused():
    rep = Report(seen={k: 10 for k, _ in KINDS}, passed={"real": 10, "noisy": 4})
    assert rep.let_through == 0.1 and rep.turned_away == 0.0


def test_turned_away_counts_only_the_ones_that_should_have_passed():
    rep = Report(seen={k: 10 for k, _ in KINDS}, passed={"real": 6})
    assert rep.turned_away == 0.4


def test_an_empty_report_claims_nothing():
    assert Report().let_through == 0.0 and Report().turned_away == 0.0


def test_only_one_of_the_five_kinds_should_pass():
    assert sum(1 for _k, good in KINDS if good) == 1
