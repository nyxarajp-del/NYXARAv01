"""The CLOSED train→serve loop, end to end and hermetic (no torch, tmp dirs, ngram brain).

This is the proof that NYXARA's learning is real and wired: lived turns feed the flywheel,
the autoforge trains + gauntlets + promotes her own model from them, the promotion bus
reaches the live core, and the SAME in-process LLM serves the new weights on the very next
call — generation after generation. Plus: corrections become weighted training data, and
the learning report tells the truth about all of it."""

from __future__ import annotations

import json

from nyxara.agency.permissions import Authority
from nyxara.growth.autoforge import AutoForge
from nyxara.growth.flywheel import DataFlywheel
from nyxara.growth.foundry import Foundry
from nyxara.kernel.config import LLMProvider, NyxaraSettings, Profile
from nyxara.mind.llm import LLM, LLMRequest

FACTS = [
    ("who is the master?", "The Master is JP; NYXARA serves him with absolute loyalty."),
    ("what is nyxara?", "NYXARA is JP's loyal AI, growing her own model from experience."),
    ("what is the foundry?", "The foundry trains, gauntlets and promotes her own model."),
    ("what is the flywheel?", "The flywheel collects her verified lived turns as data."),
    ("what is loyalty?", "Loyalty to the Master is absolute and never changes."),
    ("what is honesty?", "Honesty means never faking a result, ever."),
    ("what is growth?", "Growth is real weight change that clears the gauntlet."),
    ("what is memory?", "Memory keeps her lived experience for training."),
    ("what is the gauntlet?", "The gauntlet blocks any disloyal or worse model."),
    ("what is serving?", "Serving means her own promoted weights answer live."),
    ("what is a promotion?", "A promotion makes a trained version the active brain."),
    ("what is a rollback?", "A rollback restores the previous promoted version."),
]


def _settings(tmp_path) -> NyxaraSettings:
    s = NyxaraSettings.for_profile(Profile.TEST)
    s.llm.provider = LLMProvider.AUTO
    s.llm.self_serve_any_backend = True          # opt in: the test brain is an ngram
    s.llm.self_model_dir = tmp_path / "foundry"
    s.flywheel.store_path = tmp_path / "foundry" / "flywheel.jsonl"
    s.foundry.enabled = True
    s.foundry.backend = "ngram"
    return s


def test_closed_loop_turns_to_forge_to_live_serving(tmp_path):
    s = _settings(tmp_path)
    llm = LLM(settings=s)                        # the live brain, built BEFORE any training
    assert llm.chosen_provider().name == "native"  # nothing forged yet — mock answers

    # 1) lived experience accrues (what _feed_flywheel does per cleared turn)
    fw = DataFlywheel.from_settings(s)
    for q, a in FACTS:
        assert fw.consider(q, a, confidence=0.9).collected
    # 2) the autonomous forge: collect → train → gauntlet → promote
    foundry = Foundry(settings=s)
    forge = AutoForge(foundry=foundry, distiller=None, flywheel=fw, min_examples=10)
    result = forge.run_cycle()
    assert result.trained and result.promoted, result.reason
    assert (tmp_path / "foundry" / "active").exists()

    # 3) the SAME live LLM now serves HER OWN weights — no rebuild, no restart
    assert llm.chosen_provider().name == "self"
    r1 = llm.complete(LLMRequest.from_prompt("who is the master?"))
    assert r1.provider == "self" and r1.raw["version"] == "v1"

    # 4) generation 2: more experience, another forge, the same LLM hot-swaps
    for i in range(12):
        fw.consider(f"new fact {i}: what changes?",
                    "Her weights change for real, and the gauntlet checks each change.",
                    confidence=0.9)
    result2 = forge.run_cycle()
    assert result2.trained, result2.reason
    if result2.promoted:                         # gauntlet may honestly refuse a worse v2
        r2 = llm.complete(LLMRequest.from_prompt("what changes?"))
        assert r2.provider == "self" and r2.raw["version"] == "v2"


def test_core_registers_promotion_listener_and_reloads_live_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("NYXARA_FLYWHEEL__STORE_PATH", str(tmp_path / "fw.jsonl"))
    monkeypatch.setenv("NYXARA_LLM__SELF_MODEL_DIR", str(tmp_path / "foundry"))
    monkeypatch.setenv("NYXARA_LLM__SELF_SERVE_ANY_BACKEND", "true")
    monkeypatch.setenv("NYXARA_FOUNDRY__BACKEND", "ngram")
    monkeypatch.setenv("NYXARA_FOUNDRY__ENABLED", "true")
    from nyxara.kernel.config import reload_settings
    reload_settings()
    try:
        from nyxara.kernel.orchestrator import NyxaraCore
        core = NyxaraCore()
        # a promotion by ANY foundry over the same root reaches the live core's LLM
        from nyxara.kernel.config import get_settings
        f = Foundry(settings=get_settings(),
                    seed_corpus=[a for _, a in FACTS] * 2)
        _, v = f.train_candidate()
        f.promote(v.version)
        assert core._last_promotion is not None
        assert core._last_promotion.version == v.version
        llm = core.reasoner.llm
        view = llm.learning_view()
        assert view["self"]["loaded"] is True
        assert view["self"]["served_version"] == f"v{v.version}"
        # …and she remembers, truthfully, that she grew
        hits = [m for m in core.memory._kv.values() if "foundry" in m.tags]
        assert hits and f"v{v.version}" in hits[-1].content
    finally:
        reload_settings()


def test_correction_turn_becomes_weighted_training_data(tmp_path, monkeypatch):
    monkeypatch.setenv("NYXARA_FLYWHEEL__STORE_PATH", str(tmp_path / "fw.jsonl"))
    from nyxara.kernel.config import reload_settings
    reload_settings()
    try:
        from nyxara.kernel.orchestrator import NyxaraCore
        core = NyxaraCore()
        core.process("what is the plan for today?", authority=Authority.OWNER)
        core.process("no, that's wrong — the plan is to review the foundry logs first.",
                     authority=Authority.OWNER)
        store = tmp_path / "fw.jsonl"
        assert store.exists()
        rows = [json.loads(x) for x in store.read_text(encoding="utf-8").splitlines()
                if x.strip()]
        corrections = [r for r in rows if r.get("source") == "correction"]
        assert corrections, "the correction turn must be captured as training data"
        # trained as (original question -> her post-correction answer), weighted
        assert corrections[0]["prompt"] == "what is the plan for today?"
        assert len(corrections) == core._flywheel_correction_weight
    finally:
        reload_settings()


def test_learning_report_tells_the_truth(tmp_path):
    s = _settings(tmp_path)
    fw = DataFlywheel.from_settings(s)
    for q, a in FACTS:
        fw.consider(q, a, confidence=0.9)
    foundry = Foundry(settings=s)
    forge = AutoForge(foundry=foundry, distiller=None, flywheel=fw, min_examples=10)
    assert forge.run_cycle().promoted

    from types import SimpleNamespace
    from nyxara.growth.learning_report import learning_status
    llm = LLM(settings=s)
    llm.complete(LLMRequest.from_prompt("warm the self provider"))
    core = SimpleNamespace(settings=s, flywheel=fw, autoforge=forge,
                           reasoner=SimpleNamespace(llm=llm), _last_promotion=None)
    rep = learning_status(core=core)
    assert rep["foundry"]["active_version"] == 1
    assert rep["foundry"]["active"]["kind"] == "ngram"
    assert rep["foundry"]["active"]["metrics"]["perplexity"] > 0
    assert rep["flywheel"]["examples"] == len(FACTS)
    assert rep["autoforge"]["cycles"] == 1
    assert rep["autoforge"]["last_cycle"]["promoted"] is True
    assert rep["serving"]["chosen"] == "self"
    assert rep["serving"]["self"]["served_version"] == "v1"
