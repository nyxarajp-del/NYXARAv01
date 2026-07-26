"""Intent symbiote: predicts from history, speculative cache hit/evict, capped, gate still applies."""

from __future__ import annotations

from nyxara.nyx5.intent import IntentSymbiote


def _train(it):
    for a, b in [("code", "test"), ("code", "test"), ("code", "debug")]:
        it.observe(a)
        it.observe(b)


def test_predicts_most_likely_next():
    it = IntentSymbiote()
    _train(it)
    it.observe("code")
    preds = it.predict()
    assert preds and preds[0].label == "test"     # test followed code twice, debug once


def test_prefetch_and_consume():
    it = IntentSymbiote(cache_size=4)
    _train(it)
    it.observe("code")
    label = it.prefetch(lambda x: f"answer:{x}")
    assert label == "test"
    hit, value = it.consume("test")
    assert hit is True and value == "answer:test"


def test_cache_capped():
    it = IntentSymbiote(cache_size=2, speculate=True)
    # force several distinct predictions by rotating history
    for label in ("a", "b", "c", "d"):
        it.observe("root")
        it.observe(label)
    for _ in range(4):
        it.observe("root")
        it.prefetch(lambda x: x)
    assert len(it._cache) <= 2


def test_wrong_prediction_discarded():
    it = IntentSymbiote(cache_size=4)
    _train(it)
    it.observe("code")
    it.prefetch(lambda x: f"answer:{x}")
    hit, _ = it.consume("something-else")          # actual intent differs from the guess
    assert hit is False
