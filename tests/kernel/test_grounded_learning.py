"""End-to-end tests for grounded understanding: NYXARA's OWN learning is actually wired.

Three substrates must genuinely learn from her real turns — no LLM involved:

* the self-learned embedder banks every exchange (corpus + contrastive pairs) and is
  trained by real SGD inside the consolidation cycle, then stale memory vectors migrate;
* the (grounded) world model turns every lived exchange into a learnable transition;
* the causal world model receives the turn's WHOLE context (action, tool, recall
  hit/miss, valued reward), not just two labels.
"""

from __future__ import annotations

from nyxara.agency.permissions import Authority
from nyxara.kernel.orchestrator import NyxaraCore


def _talked_core(n_turns: int = 3) -> NyxaraCore:
    nyx = NyxaraCore()
    lines = ["hello NYXARA, I am JP and I love astronomy",
             "what do you know about me?",
             "the server disk is nearly full",
             "please summarise today's events"]
    for line in lines[:n_turns]:
        nyx.process(line, authority=Authority.OWNER)
    return nyx


def test_default_embedder_is_self_learning():
    nyx = NyxaraCore()
    emb = nyx.memory.embedder
    assert hasattr(emb, "learn_pair") and hasattr(emb, "train")
    assert emb.space_version == 0 and emb.is_lexical


def test_turns_bank_experience_and_pairs():
    nyx = _talked_core(3)
    emb = nyx.memory.embedder
    assert emb.experience_size > 0                  # she banked what she lived
    assert len(emb._pairs) > 0                      # and mined her own supervision


def test_world_model_learns_from_real_turns():
    nyx = _talked_core(3)
    assert nyx.world_model is not None
    assert hasattr(nyx.world_model, "encode_state")  # the grounded wrapper is live
    assert len(nyx.world_model) >= 3                # every turn became a transition
    assert callable(getattr(nyx.world_model, "mean_epistemic", None))


def test_causal_stream_is_rich():
    nyx = _talked_core(2)
    stats = nyx.causal_world_model.stats()
    assert stats["events"] >= 2 * 4                 # several labels per turn, not two
    assert stats["interventions"] >= 2              # her actions are do-experiments
    assert stats["valued_labels"] >= 1              # the reward event carries a value
    labels = set(nyx.causal_world_model._by_label)
    assert any(lb.startswith("act:") for lb in labels)
    assert any(lb.startswith("recall:") for lb in labels)
    assert "outcome:reward" in labels


def test_consolidation_trains_and_migrates():
    nyx = _talked_core(3)
    emb = nyx.memory.embedder
    report = nyx.consolidator.run_cycle()
    assert report.embedder_trained is True
    assert emb.space_version >= 1
    # every indexed record has been migrated into (or born in) the current space
    stale = [r for r in nyx.memory._kv.values()
             if r.embedding is not None
             and int(r.metadata.get("emb_v", 0)) < emb.space_version]
    nyx.memory.reembed_stale(budget=10_000)
    remaining = [r for r in nyx.memory._kv.values()
                 if r.embedding is not None
                 and int(r.metadata.get("emb_v", 0)) < emb.space_version]
    assert len(remaining) <= len(stale)
    assert not remaining


def test_world_model_persists_via_autonomy_state(tmp_path, monkeypatch):
    nyx = _talked_core(2)
    monkeypatch.setattr(nyx, "_autonomy_state_dir", lambda: str(tmp_path))
    saved = nyx.persist_autonomy_state()
    assert saved.get("world_model") is True
    assert (tmp_path / "world_model.json").exists()


def test_causal_prior_is_attached_to_world_model():
    nyx = NyxaraCore()
    assert getattr(nyx.world_model, "causal_model", None) is nyx.causal_world_model
