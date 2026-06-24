"""Tests for nyxara.growth.credit — the lifelong improvement ledger (credit assignment).

Hermetic and offline: Beta posteriors update in-process, persistence rides an in-process
MemoryStore, and Thompson ranking uses a seeded RNG for determinism.
"""

from __future__ import annotations

import random

from nyxara.growth.credit import (
    EditStrategyBandit,
    ImprovementLedger,
    LedgerState,
    arm_key,
)
from nyxara.memory.store import MemoryStore


class _W:
    """A minimal weakness stand-in (source + edit_strategy + severity)."""
    def __init__(self, source: str, severity: float = 0.5, strategy: str = "transform"):
        self.source = source
        self.edit_strategy = strategy
        self.severity = severity


def _ledger() -> ImprovementLedger:
    return ImprovementLedger(memory=MemoryStore(), rng=random.Random(0))


# --------------------------------------------------------------------------- #
# arm keys
# --------------------------------------------------------------------------- #
def test_arm_key_shapes():
    assert arm_key(source="cyclomatic", strategy="transform") == "edit:cyclomatic:transform"
    assert arm_key(action="train_self_model") == "action:train_self_model"
    # an action key takes precedence and a missing source degrades to 'unknown'
    assert arm_key() == "edit:unknown:transform"


# --------------------------------------------------------------------------- #
# reward shaping
# --------------------------------------------------------------------------- #
def test_reward_from_delta_is_monotone_and_centered():
    led = _ledger()
    assert led.reward_from_delta(0.05) > 0.6
    assert led.reward_from_delta(-0.05) < 0.4
    assert abs(led.reward_from_delta(0.0) - 0.5) < 1e-9
    # monotone in delta
    assert led.reward_from_delta(0.1) > led.reward_from_delta(0.01)


# --------------------------------------------------------------------------- #
# Beta updates
# --------------------------------------------------------------------------- #
def test_record_moves_posterior_means():
    led = _ledger()
    st = led.load()
    good = arm_key(source="cyclomatic")
    bad = arm_key(source="hygiene")
    for _ in range(25):
        led.record(st, good, 0.95)
        led.record(st, bad, 0.05)
    assert led.mean(st, good) > 0.8
    assert led.mean(st, bad) < 0.2
    assert st.t == 50
    # an untried arm sits at the uniform prior mean
    assert abs(led.mean(st, arm_key(source="never")) - 0.5) < 1e-9


def test_record_delta_credits_all_touched_arms():
    led = _ledger()
    st = led.load()
    arms = [arm_key(source="a"), arm_key(action="acquire_knowledge")]
    led.record_delta(st, arms, delta=0.08)        # a clear rise → reward > 0.5 for both
    assert led.mean(st, arms[0]) > 0.5
    assert led.mean(st, arms[1]) > 0.5


# --------------------------------------------------------------------------- #
# bandit ranking
# --------------------------------------------------------------------------- #
def test_bandit_prioritizes_learned_winner():
    led = _ledger()
    st = led.load()
    good, bad = arm_key(source="cyclomatic"), arm_key(source="hygiene")
    for _ in range(30):
        led.record(st, good, 0.95)
        led.record(st, bad, 0.05)
    bandit = EditStrategyBandit(led)
    wins = 0
    trials = 200
    for _ in range(trials):
        order = bandit.prioritize([_W("hygiene"), _W("cyclomatic")], state=st)
        if order[0].source == "cyclomatic":
            wins += 1
    assert wins > trials * 0.7


def test_bandit_empty_ledger_preserves_order_on_equal_severity():
    # with no learned signal and equal severity, the original order is preserved (stable)
    led = _ledger()
    bandit = EditStrategyBandit(led)
    items = [_W("a", 0.5), _W("b", 0.5), _W("c", 0.5)]
    order = bandit.prioritize(items, state=led.load())
    assert [w.source for w in order] == ["a", "b", "c"]


def test_bandit_severity_breaks_ties_for_untried_arms():
    led = _ledger()
    bandit = EditStrategyBandit(led)
    order = bandit.prioritize([_W("low", 0.1), _W("high", 0.9)], state=led.load())
    assert order[0].source == "high"


def test_bandit_severity_blend_is_config_driven():
    # the learned/severity blend is a real, evolvable algorithm knob (read from config). With a
    # historically-good but LOW-severity arm vs a never-tried HIGH-severity one, the blend decides
    # what gets fixed first: blend=1.0 → pure learned payoff; blend=0.0 → pure severity.
    led = _ledger()
    st = led.load()
    good = arm_key(source="learned_good")
    for _ in range(30):
        led.record(st, good, 0.95)               # learned to pay off, but we give it low severity
    bandit = EditStrategyBandit(led)
    items = lambda: [_W("learned_good", 0.05), _W("severe_new", 0.95)]

    led.settings.self_improvement.bandit_severity_blend = 1.0   # pure learned payoff
    learned_first = sum(bandit.prioritize(items(), state=st)[0].source == "learned_good"
                        for _ in range(200))
    led.settings.self_improvement.bandit_severity_blend = 0.0   # pure severity
    severe_first = bandit.prioritize(items(), state=st)[0].source

    assert learned_first > 140          # learned payoff dominates when blend favours it
    assert severe_first == "severe_new"  # severity dominates when blend favours it (deterministic)


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def test_persistence_round_trips_as_one_protected_record():
    led = ImprovementLedger(memory=MemoryStore())
    st = led.load()
    led.record(st, arm_key(source="x"), 0.9)
    assert led.save(st)
    led2 = ImprovementLedger(memory=led.memory)
    st2 = led2.load()
    assert st2.t == st.t
    assert abs(led2.mean(st2, arm_key(source="x")) - led.mean(st, arm_key(source="x"))) < 1e-6
    rec = led2._find_record()
    assert rec is not None and rec.importance >= 0.9
    assert led.settings.memory.deep_synapse_tag in rec.tags


def test_save_keeps_exactly_one_record():
    led = ImprovementLedger(memory=MemoryStore())
    st = led.load()
    for i in range(3):
        led.record(st, arm_key(source=f"s{i}"), 0.7)
        led.save(st)
    records = [r for r in led.memory._kv.values()
               if "improvement-ledger" in r.tags and "state" in r.metadata]
    assert len(records) == 1


def test_fresh_store_starts_empty():
    assert ImprovementLedger(memory=MemoryStore()).load().t == 0


def test_state_dict_round_trip():
    st = LedgerState(arms={"action:x": {"alpha": 3.0, "beta": 1.0, "n": 4}}, t=4,
                     history=[0.9, 0.8])
    back = LedgerState.from_dict(st.to_dict())
    assert back.t == 4 and back.arms["action:x"]["alpha"] == 3.0
    assert back.history == [0.9, 0.8]
