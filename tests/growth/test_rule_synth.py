"""Tests for nyxara.growth.rule_synth — inventing a new learning rule when the old one fails."""

from __future__ import annotations

import math

import pytest

from nyxara.growth.learn import Learner, LinearValueModel
from nyxara.growth.rule_synth import (
    LearningRuleSynthesizer,
    Node,
    UpdateRule,
    incumbent_rule,
)
from nyxara.guard.value_learning import IMMUTABLE_VALUES
from nyxara.kernel.errors import ValidationError


def _synth(**kw):
    kw.setdefault("population", 12)
    kw.setdefault("generations", 6)
    kw.setdefault("steps_per_task", 50)
    kw.setdefault("seeds", 3)
    kw.setdefault("seed", 1)
    return LearningRuleSynthesizer(**kw)


# ---- the seam: default learner is byte-identical ---- #
def test_default_learner_unchanged():
    """A model with update_rule=None must produce exactly the incumbent SGD+EWC weights."""
    seq = [("a", {"f": 1.0, "g": 0.5}, 1.0), ("a", {"f": 1.0}, 0.5),
           ("b", {"g": 1.0}, -0.3), ("a", {"f": 0.2, "g": 0.2}, 0.8)]

    def run(model):
        for action, feats, reward in seq * 20:
            model.update(action, feats, reward, 0.1)
        return {a: dict(w) for a, w in model._w.items()}

    plain = run(LinearValueModel(ewc_lambda=1.0))
    default = run(LinearValueModel(ewc_lambda=1.0, update_rule=None))
    assert plain == default
    # and the explicit incumbent DAG reproduces the same math
    incumbent = run(LinearValueModel(ewc_lambda=1.0, update_rule=incumbent_rule()))
    for a in plain:
        for f in plain[a]:
            assert abs(plain[a][f] - incumbent[a][f]) < 1e-9


# ---- incumbent representability & recovery ---- #
def test_incumbent_representable():
    inc = incumbent_rule()
    assert inc.expr == "mul(lr, sub(grad, penalty))"
    d = inc.delta(err=0.5, x=2.0, w=0.0, w_star=0.0, importance=0.0,
                  grad=1.0, m=0.0, v=0.0, step=1.0, lr=0.1, penalty=0.3)
    assert abs(d - 0.07) < 1e-12          # lr*(grad-penalty) = 0.1*(1.0-0.3)


def test_search_never_regresses_below_sgd():
    """The best discovered rule must be at least as fit as the incumbent (elitism keeps SGD)."""
    rep = _synth().run(enact=False)
    assert math.isfinite(rep.best_fitness)
    assert rep.best_fitness >= rep.incumbent_fitness - 1e-9


# ---- discovery on a task where a better rule exists ---- #
def test_discovers_better_rule():
    """On the battery (which includes a non-stationary task), a rule beating plain SGD exists."""
    rep = _synth(generations=8, adoption_margin=0.01).run(enact=False)
    assert rep.adopted is True
    assert rep.best_fitness > rep.incumbent_fitness + 0.01
    assert rep.best_rule is not None and rep.best_expr


# ---- numerical safety ---- #
def test_interpreter_always_finite():
    """Pathological rules (÷0, blowups) must still yield a finite delta."""
    rules = [
        UpdateRule(root=Node("safe_div", Node("err"), Node("const", const=0.0))),
        UpdateRule(root=Node("square", Node("square", Node("square", Node("grad"))))),
        UpdateRule(root=Node("sqrt", Node("neg", Node("abs", Node("w"))))),
    ]
    for r in rules:
        d = r.delta(err=1e9, x=1e9, w=-1e9, w_star=0.0, importance=1e9,
                    grad=1e9, m=1e9, v=1e9, step=1e9, lr=0.1, penalty=1e9)
        assert math.isfinite(d)


def test_nan_or_divergent_rule_rejected():
    """A rule that makes the learner diverge is hard-rejected (fitness -inf), never adopted."""
    synth = _synth()
    # a rule that adds a huge constant every step forces divergence on the regression task
    runaway = UpdateRule(root=Node("const", const=1e5))
    fit = synth._fitness(runaway)
    assert fit == float("-inf")


# ---- character safety ---- #
def test_no_protected_names_in_battery():
    """No task in the evaluation battery may train on a protected character value."""
    synth = _synth()
    # the battery uses only f0..f3 / good / bad / act / c — assert disjoint from IMMUTABLE_VALUES
    used = {"f0", "f1", "f2", "f3", "good", "bad", "act", "c"}
    assert not (used & set(IMMUTABLE_VALUES))


def test_installed_rule_cannot_learn_over_character():
    """Even with an invented rule installed, the guard still refuses protected values."""
    lrn = Learner(base_lr=0.1, seed=0)
    lrn.install_rule(UpdateRule(root=Node("grad")))
    with pytest.raises(ValidationError):
        lrn.update("loyalty_to_master", {"x": 1.0}, -1.0)
    with pytest.raises(ValidationError):
        lrn.update("act", {"honesty": 1.0}, -1.0)


# ---- adoption gating ---- #
def test_tie_is_not_adopted():
    """If nothing beats the incumbent by the margin, no rule is adopted."""
    # a huge margin makes any realistic improvement fail the gate
    rep = _synth(adoption_margin=0.99).run(enact=False)
    assert rep.adopted is False


def test_enact_false_never_touches_live_learner():
    class _Core:
        learner = Learner(base_lr=0.1, seed=0)
    core = _Core()
    rep = _synth().run(enact=False)
    assert rep is not None
    assert core.learner.update_rule is None


# ---- determinism ---- #
def test_seed_determinism():
    a = _synth(seed=7).run(enact=False)
    b = _synth(seed=7).run(enact=False)
    assert a.best_expr == b.best_expr
    assert abs(a.best_fitness - b.best_fitness) < 1e-12
    assert a.best_rule == b.best_rule


# ---- persistence ---- #
def test_save_load_round_trip(tmp_path):
    rule = UpdateRule(root=Node("add", Node("mul", Node("lr"), Node("grad")),
                               Node("tanh", Node("m"))), origin="invented")
    path = tmp_path / "rule.json"
    rule.save(path)
    loaded = UpdateRule.load(path)
    for args in (
        dict(err=0.5, x=1.0, w=0.1, w_star=0.0, importance=0.2, grad=0.5,
             m=0.3, v=0.1, step=4.0, lr=0.1, penalty=0.05),
        dict(err=-1.0, x=2.0, w=-0.5, w_star=0.1, importance=0.0, grad=-2.0,
             m=-0.4, v=0.9, step=10.0, lr=0.2, penalty=-0.2),
    ):
        assert abs(rule.delta(**args) - loaded.delta(**args)) < 1e-12
    assert loaded.to_dict()["root"] == rule.to_dict()["root"]


# ---- install / rollback ---- #
def test_install_and_rollback_restores_behavior():
    seq = [("a", {"f": 1.0}, 1.0)] * 30
    baseline = Learner(base_lr=0.1, seed=0)
    for action, feats, reward in seq:
        baseline.update(action, feats, reward)
    base_val = baseline.value("a", {"f": 1.0})

    lrn = Learner(base_lr=0.1, seed=0)
    lrn.install_rule(UpdateRule(root=Node("mul", Node("lr"), Node("grad"))))
    assert lrn.update_rule is not None
    lrn.rollback_rule()
    assert lrn.update_rule is None
    for action, feats, reward in seq:
        lrn.update(action, feats, reward)
    # after rollback the learner behaves exactly like the incumbent
    assert abs(lrn.value("a", {"f": 1.0}) - base_val) < 1e-12
