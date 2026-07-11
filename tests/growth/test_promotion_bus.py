"""Tests for the promotion event bus (growth/promotion.py) and the Foundry hooks.

The bus is the in-process half of the train→serve loop: any Foundry that promotes or
rolls back notifies every live listener, and a listener failure never fails a promotion.
The pointer write must be atomic (readers poll it per request)."""

from __future__ import annotations

from nyxara.growth import promotion
from nyxara.growth.foundry import Foundry
from nyxara.kernel.config import NyxaraSettings, Profile

CORPUS = ["the master is jp. nyxara serves the master with absolute loyalty."] * 6 + [
    "loyalty to the master is absolute and never changes, in every world."] * 6


def _foundry(tmp_path) -> Foundry:
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.llm.self_model_dir = tmp_path / "foundry"
    settings.foundry.backend = "ngram"
    settings.foundry.enabled = True
    return Foundry(settings=settings, seed_corpus=CORPUS)


def test_subscribe_notify_unsubscribe():
    events = []
    def listener(ev):
        events.append(ev)
    promotion.subscribe(listener)
    try:
        ev = promotion.PromotionEvent(action="promote", version=1, kind="ngram",
                                      path="/x/v1", root="/x", metrics={"perplexity": 3.0})
        assert promotion.notify(ev) == 1
        assert events == [ev]
        assert ev.to_dict()["version"] == 1
    finally:
        promotion.unsubscribe(listener)
    assert promotion.notify(ev) == 0     # unsubscribed: nobody called


def test_subscribe_is_idempotent():
    events = []
    def listener(ev):
        events.append(ev)
    promotion.subscribe(listener)
    promotion.subscribe(listener)
    try:
        promotion.notify(promotion.PromotionEvent(
            action="promote", version=1, kind="ngram", path="p", root="r"))
        assert len(events) == 1
    finally:
        promotion.unsubscribe(listener)


def test_listener_exception_is_swallowed():
    def boom(ev):
        raise RuntimeError("listener broke")
    promotion.subscribe(boom)
    try:
        # never raises into the promoter
        promotion.notify(promotion.PromotionEvent(
            action="promote", version=1, kind="ngram", path="p", root="r"))
    finally:
        promotion.unsubscribe(boom)


def test_dead_bound_method_drops_from_bus():
    events = []

    class Core:
        def on_promoted(self, ev):
            events.append(ev)

    core = Core()
    promotion.subscribe(core.on_promoted)
    del core
    import gc
    gc.collect()
    called = promotion.notify(promotion.PromotionEvent(
        action="promote", version=1, kind="ngram", path="p", root="r"))
    assert called == 0 and events == []


def test_foundry_promote_and_rollback_emit_events(tmp_path):
    events = []
    def listener(ev):
        events.append(ev)
    promotion.subscribe(listener)
    try:
        f = _foundry(tmp_path)
        _, v1 = f.train_candidate()
        f.promote(v1.version)
        assert [e.action for e in events] == ["promote"]
        assert events[0].version == v1.version
        assert events[0].kind == v1.kind
        assert events[0].path == v1.path
        assert "perplexity" in events[0].metrics

        _, v2 = f.train_candidate()
        f.promote(v2.version)
        f.rollback()
        assert [e.action for e in events] == ["promote", "promote", "rollback"]
        # the event always names the version the foundry actually made active
        assert events[-1].version == f.active_version
    finally:
        promotion.unsubscribe(listener)


def test_active_pointer_write_is_atomic(tmp_path):
    f = _foundry(tmp_path)
    _, v1 = f.train_candidate()
    f.promote(v1.version)
    root = f.root
    assert (root / "active").read_text(encoding="utf-8") == f"v{v1.version}"
    assert not (root / "active.tmp").exists()      # tmp file replaced, never left behind
