"""Tests for the learning-rule-synthesis wire.

NYXARA's learning-to-learn engine (growth/rule_synth.py) invents a NEW weight-update rule when her
fixed learning method has stalled, and installs it into her live learner. These tests pin the wire:
the config exists and is sealed OFF from live installs in the suite, the ``GrowthEngine`` exposes the
stage and surfaces it in the report, the plateau trigger stays dormant without a failure signal and
fires when one appears, and a hermetic growth pass never mutates the live learner.
"""

from __future__ import annotations

from nyxara.growth.autolearn import GrowthEngine, GrowthReport
from nyxara.growth.meta_engine import MetaDimension
from nyxara.kernel.config import NyxaraSettings, Profile, RuleSynthesisConfig
from nyxara.kernel.orchestrator import NyxaraCore


# -------------------- config present + hermetic hardening -------------------- #
def test_config_defaults_live():
    # the RAW model default (a plain BaseModel — no env) is the live DEV/PROD posture
    cfg = RuleSynthesisConfig()
    assert cfg.enabled is True
    assert cfg.autonomous_enact is True         # standing authorisation in live DEV/PROD
    assert cfg.every >= 1 and cfg.population >= 2 and cfg.generations >= 1


def test_config_present_on_settings():
    cfg = NyxaraSettings().rule_synthesis
    assert isinstance(cfg, RuleSynthesisConfig)
    assert cfg.enabled is True                  # search/measure stays on in the suite


def test_test_profile_seals_off_live_install():
    cfg = NyxaraSettings.for_profile(Profile.TEST).rule_synthesis
    assert cfg.autonomous_enact is False        # the suite never installs into the live learner


# -------------------- the stage is wired into the growth engine -------------------- #
def test_growth_report_carries_rule_synthesis_slot():
    assert "rule_synthesis" in GrowthReport().to_dict()


def test_growth_engine_exposes_the_stage():
    core = NyxaraCore()
    engine = GrowthEngine.from_core(core)
    assert hasattr(engine, "synthesize_rule")
    assert hasattr(engine, "should_invent")
    assert engine.enable_rule_synthesis is True


def test_core_exposes_convenience_method():
    core = NyxaraCore()
    assert hasattr(core, "synthesize_learning_rule")


# -------------------- the trigger stays dormant until learning actually fails -------------------- #
def test_should_invent_false_without_signal():
    core = NyxaraCore()
    engine = GrowthEngine.from_core(core)
    # a fresh core has no plateau/regression evidence -> the subsystem stays dormant
    assert engine.should_invent() is False


def test_should_invent_true_on_learning_plateau():
    core = NyxaraCore()
    engine = GrowthEngine.from_core(core)
    mle = core.meta_learning_engine
    assert mle is not None
    # feed a flat/declining LEARNING signal past the min_signal threshold -> a plateau
    for _ in range(int(core_settings_min_signal(core)) + 2):
        mle.observe(MetaDimension.LEARNING, 0.4)
    assert engine.should_invent() is True


def core_settings_min_signal(core) -> int:
    from nyxara.kernel.config import get_settings
    return get_settings().rule_synthesis.min_signal


# -------------------- a hermetic growth pass never mutates the live learner -------------------- #
def test_synthesize_rule_does_not_install_under_test_profile():
    core = NyxaraCore()
    engine = GrowthEngine.from_core(core)
    # force the trigger so a real synthesis pass runs, but the TEST profile has autonomous_enact off
    mle = core.meta_learning_engine
    for _ in range(int(core_settings_min_signal(core)) + 2):
        mle.observe(MetaDimension.LEARNING, 0.4)
    assert engine.should_invent() is True
    report = engine.synthesize_rule()
    assert report is not None                       # the pass ran and measured
    # ...but nothing was installed into the live learner (hermetic)
    assert core.learner.update_rule is None
