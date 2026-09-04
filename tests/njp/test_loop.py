"""The cycle that cannot comfortably remain wrong (NJP V.42).

The loop's parts are all measured elsewhere. What is tested here is the wiring, the refusals, and
the three defects the discovery gauntlet found in it.
"""
from __future__ import annotations

import pytest

from nyxara.njp.discovery import Observation as O
from nyxara.njp.loop import (
    Experiments, Latents, Loop, Model, Prediction, Reason, Status, Unknown, connected, fits,
    responds,
)

E = lambda *pairs: frozenset(pairs)  # noqa: E731


# --------------------------------------------------------------------------- #
# The pieces
# --------------------------------------------------------------------------- #
def test_an_intervention_cuts_the_incoming_edges():
    """The hinge: `A → B` and `A ← H → B` agree on every observation and differ here."""
    direct = Model(edges=E(("a", "b")), observed=("a", "b"))
    confounded = Model(edges=E(("h", "a"), ("h", "b")), observed=("a", "b"))
    assert responds(direct, "a", "b") is True
    assert responds(confounded, "a", "b") is False
    # and they are observationally identical
    seen = [O("a", "b", True)]
    assert fits(direct, seen) and fits(confounded, seen)


def test_a_latent_transmits_and_can_never_be_conditioned_on():
    edges = E(("h", "a"), ("h", "b"))
    assert connected(edges, "a", "b") is True
    assert connected(edges, "a", "b", ["h"]) is False   # only if someone could hold it — nobody can
    seen = [O("a", "b", True)]
    assert fits(Model(edges=edges, observed=("a", "b")), seen)


def test_a_proposed_latent_is_hypothetical_and_stays_that_way():
    base = Model(edges=E(("a", "b")), observed=("a", "b"))
    got = Latents().propose(base, [O("a", "b", True)], ("a", "b"))
    assert got and all(m.status is Status.HYPOTHETICAL for m in got)
    assert all(m.hypothetical and m.hidden for m in got)


def test_a_latent_that_does_not_fit_is_not_proposed():
    """Not every swap works: it must still imply everything observed."""
    base = Model(edges=E(("a", "b"), ("b", "c")), observed=("a", "b", "c"))
    seen = [O("a", "b", True), O("b", "c", True), O("a", "c", True),
            O("a", "c", False, frozenset({"b"}))]
    for model in Latents().propose(base, seen, ("a", "b", "c")):
        assert fits(model, seen)


# --------------------------------------------------------------------------- #
# The experiment, and the term that made it work
# --------------------------------------------------------------------------- #
def test_an_experiment_everyone_agrees_about_is_worth_nothing():
    models = [Model(edges=E(("a", "b")), observed=("a", "b", "c")),
              Model(edges=E(("a", "b"), ("a", "c")), observed=("a", "b", "c"))]
    assert Experiments().value(models, "a", "b") == 0.0


def test_an_experiment_that_cannot_move_the_aim_is_not_run():
    """It was run: the loop split model sets along axes nobody had asked about."""
    models = [Model(edges=E(("a", "b"), ("c", "d")), observed=("a", "b", "c", "d")),
              Model(edges=E(("b", "a"), ("c", "d")), observed=("a", "b", "c", "d"))]
    chooser = Experiments()
    assert chooser.value(models, "a", "b") > 0.0                       # splits the set
    assert chooser.value(models, "a", "b", toward=("c", "d")) == 0.0   # but not the question
    assert chooser.choose(models, toward=("c", "d")) is None


def test_the_question_itself_is_never_offered_as_the_experiment():
    models = [Model(edges=E(("a", "b")), observed=("a", "b")),
              Model(edges=E(("h", "a"), ("h", "b")), observed=("a", "b"))]
    assert Experiments().choose(models, toward=("a", "b")) is None


def test_the_aim_changes_which_experiment_is_chosen():
    models = [Model(edges=E(("a", "b"), ("b", "c")), observed=("a", "b", "c")),
              Model(edges=E(("b", "a"), ("b", "c")), observed=("a", "b", "c"))]
    blind = Experiments().choose(models)
    aimed = Experiments().choose(models, toward=("a", "c"))
    assert blind is not None
    assert aimed is None or aimed != ("a", "c")


# --------------------------------------------------------------------------- #
# Revision, in both directions
# --------------------------------------------------------------------------- #
def test_revision_kills_exactly_what_the_result_refutes():
    direct = Model(edges=E(("a", "b")), observed=("a", "b"))
    confounded = Model(edges=E(("h", "a"), ("h", "b")), observed=("a", "b"))
    loop = Loop()
    assert loop.revise([direct, confounded], "a", "b", True) == [direct]
    assert loop.revise([direct, confounded], "a", "b", False) == [confounded]


def test_a_refuted_round_is_recorded_with_the_edge_that_carried_it():
    direct = Model(edges=E(("a", "b")), observed=("a", "b"))
    confounded = Model(edges=E(("h", "a"), ("h", "b")), observed=("a", "b"))
    loop = Loop()
    loop.revise([direct, confounded], "a", "b", False)
    assert loop.autopsy.failures
    failure = loop.autopsy.failures[0]
    assert failure.blamed == [direct] and failure.edge == ("a", "b")
    assert loop.autopsy.seen_before("a", "b") is failure


# --------------------------------------------------------------------------- #
# Not knowing, with a reason
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("models,cause,effect,reason", [
    ([], "a", "b", Reason.NO_MODEL),
    ([Model(edges=E(("a", "b")), observed=("a", "b"))], "a", "z", Reason.OUT_OF_SCOPE),
    ([Model(edges=E(("a", "b"), ("c", "d")), observed=("a", "b", "c", "d"))],
     "a", "d", Reason.UNREACHED),
    ([Model(edges=E(("a", "b")), observed=("a", "b", "c")),
      Model(edges=E(("b", "a")), observed=("a", "b", "c"))], "a", "b", Reason.NO_EXPERIMENT),
])
def test_every_reason_for_not_knowing_is_reachable(models, cause, effect, reason):
    got = Loop().forecast(models, cause, effect)
    assert isinstance(got, Unknown) and got.reason is reason


def test_a_variable_no_model_mentions_is_not_answered_with_no():
    """It was: the mutilation reaches nothing, so "no" fell out of never having heard of it."""
    got = Loop().forecast([Model(edges=E(("a", "b")), observed=("a", "b"))], "a", "z")
    assert isinstance(got, Unknown) and got.reason is Reason.OUT_OF_SCOPE


def test_a_conflicted_set_is_never_resolved_by_majority():
    models = [Model(edges=E(("a", "b"), ("b", "c")), observed=("a", "b", "c")),
              Model(edges=E(("a", "b"), ("c", "b")), observed=("a", "b", "c")),
              Model(edges=E(("b", "a"), ("b", "c")), observed=("a", "b", "c"))]
    got = Loop().forecast(models, "a", "c")
    assert isinstance(got, Unknown)


def test_a_prediction_resting_on_a_hidden_cause_says_so():
    confounded = Model(edges=E(("h", "a"), ("h", "b")), observed=("a", "b"),
                       status=Status.HYPOTHETICAL)
    got = Loop().forecast([confounded], "a", "b")
    assert isinstance(got, Prediction) and got.status is Status.HYPOTHETICAL
    assert got.responds is False


def test_a_prediction_carries_the_models_it_came_from():
    direct = Model(edges=E(("a", "b")), observed=("a", "b"))
    got = Loop().forecast([direct], "a", "b")
    assert got.from_models == [direct] and got.status is Status.SUPPORTED


# --------------------------------------------------------------------------- #
# The defect that cost the most
# --------------------------------------------------------------------------- #
def test_no_observed_only_structure_fitting_is_evidence_of_a_latent_not_a_dead_end():
    """It returned zero models where the observations admitted dozens.

    A latent that confounds two variables can make **every** orientation of the skeleton imply a
    collider the data denies. The observed-only class is then correctly empty — and reading that as
    failure left nothing to propose a latent from.
    """
    # a — b — c fully dependent, and no conditioning separates any pair: a confounded triangle
    nodes = ("a", "b", "c")
    seen = [O("a", "b", True), O("b", "c", True), O("a", "c", True),
            O("a", "b", True, frozenset({"c"})), O("b", "c", True, frozenset({"a"})),
            O("a", "c", True, frozenset({"b"}))]
    got = Loop().models(nodes, seen)
    assert got, "no models at all is the defect this test exists for"
    assert all(fits(m, seen) for m in got)


def test_the_skeleton_is_read_off_the_observations_not_searched_for():
    nodes = ("a", "b", "c")
    seen = [O("a", "b", True), O("b", "c", True), O("a", "c", True),
            O("a", "c", False, frozenset({"b"}))]
    for edges in Loop()._orientations(list(nodes), seen):
        undirected = {frozenset(e) for e in edges}
        assert frozenset({"a", "c"}) not in undirected      # separated by b
        assert frozenset({"a", "b"}) in undirected


def test_the_loop_holds_both_kinds_where_nothing_separates_them():
    nodes = ("a", "b")
    seen = [O("a", "b", True)]
    got = Loop().models(nodes, seen)
    assert any(m.hypothetical for m in got), "refusing to consider a latent invents its absence"
    assert any(not m.hypothetical for m in got), "considering only latents commits just as hard"
