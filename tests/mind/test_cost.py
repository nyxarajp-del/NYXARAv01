"""Tests for nyxara.mind.cost — the LLM token/cost ledger."""

from __future__ import annotations

from nyxara.mind.cost import PRICES, UsageLedger
from nyxara.mind.llm import LLMResponse, Usage


def test_estimate_cost_known_model():
    # gpt-4o: $2.50/1k prompt + $10.0/1k completion -> 1k+1k = 12.50
    assert abs(UsageLedger.estimate_cost("gpt-4o", 1000, 1000) - 12.50) < 1e-9


def test_longest_prefix_matching():
    # claude-3-5-sonnet must match its specific entry, not the generic "claude"
    p_specific = UsageLedger.price_for("claude-3-5-sonnet-20240620")
    assert p_specific == PRICES["claude-3-5-sonnet"]


def test_record_from_llm_response():
    ledger = UsageLedger()
    resp = LLMResponse(text="hi", provider="openai", model="gpt-4o",
                       usage=Usage(prompt_tokens=1000, completion_tokens=1000))
    item = ledger.record(resp)
    assert abs(item.cost_usd - 12.50) < 1e-9
    assert item.provider == "openai" and item.model == "gpt-4o"


def test_record_from_raw_usage():
    ledger = UsageLedger()
    item = ledger.record(Usage(prompt_tokens=2000, completion_tokens=0),
                         provider="anthropic", model="claude-opus")
    # claude-opus: $15/1k prompt -> 2k prompt = 30.0
    assert abs(item.cost_usd - 30.0) < 1e-9


def test_aggregations():
    ledger = UsageLedger()
    ledger.record(Usage(1000, 1000), provider="openai", model="gpt-4o")     # $12.50
    ledger.record(Usage(1000, 0), provider="anthropic", model="claude-opus")  # $15.00
    assert abs(ledger.total_cost() - 27.50) < 1e-6
    assert ledger.total_tokens() == 3000
    assert "gpt-4o" in ledger.by_model()
    assert "anthropic" in ledger.by_provider()


def test_budget_tracking():
    ledger = UsageLedger(daily_budget=20.0)
    assert not ledger.over_budget()
    ledger.record(Usage(1000, 1000), provider="openai", model="gpt-4o")  # $12.50
    assert abs(ledger.remaining() - 7.50) < 1e-6 and not ledger.over_budget()
    ledger.record(Usage(1000, 1000), provider="openai", model="gpt-4o")  # +12.50 -> 25
    assert ledger.over_budget()


def test_free_local_models_cost_nothing():
    ledger = UsageLedger()
    item = ledger.record(Usage(10000, 10000), provider="qwen", model="qwen")
    assert item.cost_usd == 0.0
