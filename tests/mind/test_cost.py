"""Tests for nyxara.mind.cost — the LLM token/cost ledger.

Every backend runs locally now (Qwen / self / mock), so real spend is always $0;
the budget-gate machinery is still exercised via a hand-priced entry monkeypatched
into PRICES.
"""

from __future__ import annotations

import pytest

from nyxara.mind.cost import PRICES, UsageLedger
from nyxara.mind.llm import LLMResponse, Usage


def test_estimate_cost_local_models_zero():
    assert UsageLedger.estimate_cost("Qwen/Qwen2.5-0.5B-Instruct", 1000, 1000) == 0.0
    assert UsageLedger.estimate_cost("nyxara-self", 5000, 5000) == 0.0
    assert UsageLedger.estimate_cost("native", 100, 100) == 0.0


def test_longest_prefix_matching():
    # the full Qwen model id must match the "Qwen" prefix entry
    assert UsageLedger.price_for("Qwen/Qwen2.5-0.5B-Instruct") == PRICES["Qwen"]


def test_unknown_model_costs_nothing():
    # everything runs on owned hardware — even an unknown model falls back to $0
    assert UsageLedger.estimate_cost("some-mystery-model", 1000, 1000) == 0.0


def test_record_from_llm_response():
    ledger = UsageLedger()
    resp = LLMResponse(text="hi", provider="qwen",
                       model="Qwen/Qwen2.5-0.5B-Instruct",
                       usage=Usage(prompt_tokens=1000, completion_tokens=1000))
    item = ledger.record(resp)
    assert item.cost_usd == 0.0
    assert item.provider == "qwen"
    assert item.total_tokens == 2000


def test_record_from_raw_usage():
    ledger = UsageLedger()
    item = ledger.record(Usage(prompt_tokens=2000, completion_tokens=0),
                         provider="self", model="nyxara-self")
    assert item.cost_usd == 0.0 and item.model == "nyxara-self"


def test_aggregations_track_tokens_per_model_and_provider():
    ledger = UsageLedger()
    ledger.record(Usage(1000, 1000), provider="qwen",
                  model="Qwen/Qwen2.5-0.5B-Instruct")
    ledger.record(Usage(1000, 0), provider="self", model="nyxara-self")
    assert ledger.total_cost() == 0.0
    assert ledger.total_tokens() == 3000
    assert "Qwen/Qwen2.5-0.5B-Instruct" in ledger.by_model()
    assert set(ledger.by_provider()) == {"qwen", "self"}


def test_budget_tracking_with_a_priced_entry(monkeypatch):
    # the daily-budget gate still works when a metered price exists
    monkeypatch.setitem(PRICES, "metered-test-model", (2.50, 10.0))
    ledger = UsageLedger(daily_budget=20.0)
    assert not ledger.over_budget()
    ledger.record(Usage(1000, 1000), provider="test", model="metered-test-model")  # $12.50
    assert abs(ledger.remaining() - 7.50) < 1e-6 and not ledger.over_budget()
    ledger.record(Usage(1000, 1000), provider="test", model="metered-test-model")  # -> $25
    assert ledger.over_budget()


def test_free_local_models_never_trip_the_budget():
    ledger = UsageLedger(daily_budget=1.0)
    ledger.record(Usage(10_000, 10_000), provider="qwen",
                  model="Qwen/Qwen2.5-0.5B-Instruct")
    assert ledger.total_cost() == 0.0
    assert not ledger.over_budget()
    assert ledger.remaining() == pytest.approx(1.0)
