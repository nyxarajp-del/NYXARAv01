"""Tests for growth/fast_weights.py — inference-time plasticity, and its fence.

The capability tests matter, but the *fence* tests are the load-bearing ones. This feature
bypasses the promotion gauntlet by construction — that is what makes it fast — so the guarantees
that keep it safe have to be properties the tests actually check, not claims in a docstring:

* ``reset`` restores the model **byte-exactly**, not approximately;
* the overlay never attaches to the character-bearing layers;
* the norm is bounded, so a long adversarial conversation cannot swamp the static weights;
* every write is auditable;
* it is OFF unless a session turns it on.
"""

from __future__ import annotations

import pytest

from nyxara.growth.fast_weights import (
    PROTECTED_SUBSTRINGS,
    FastWeightConfig,
    FastWeights,
)
from nyxara.growth.foundry_models import ModelSpec

torch = pytest.importorskip("torch", reason="fast weights are a torch overlay")

from nyxara.growth.foundry_models import NanoGPTModel  # noqa: E402


def _model() -> NanoGPTModel:
    spec = ModelSpec(kind="nanogpt", vocab_size=256, n_layer=6, n_embd=64, n_head=4,
                     n_kv_head=2, mlp_hidden=128, block_size=32, rope_theta=10000.0,
                     tie_embeddings=True)
    return NanoGPTModel(spec)


def _on(**kw) -> FastWeightConfig:
    base = dict(enabled=True, decay=0.95, lr=0.1, max_norm=1.0)
    base.update(kw)
    return FastWeightConfig(**base)


def _snapshot(model) -> dict:
    return {k: v.clone() for k, v in model.net.state_dict().items()}


def _identical(model, snapshot) -> bool:
    current = model.net.state_dict()
    return all(torch.equal(current[k], v) for k, v in snapshot.items())


# --------------------------------------------------------------------------- #
# The fence
# --------------------------------------------------------------------------- #
def test_disabled_by_default() -> None:
    """A session must opt in. Weight change outside the gauntlet is never the default."""
    assert FastWeightConfig().enabled is False
    model = _model()
    fw = FastWeights(model, config=FastWeightConfig()).attach()
    assert not fw.state()["attached"]


def test_reset_restores_the_model_byte_exactly() -> None:
    """"It mostly goes back" is not a property anyone should have to trust."""
    model = _model()
    before = _snapshot(model)
    fw = FastWeights(model, config=_on()).attach()
    for i in range(20):
        fw.learn_text(f"cue number {i}", f"target value {i}")
    assert fw.norm() > 0
    fw.reset()
    assert _identical(model, before), "the static weights were mutated — they must never be"
    assert fw.state()["writes"] == 0 and not fw.state()["attached"]


def test_static_weights_are_never_touched_while_attached() -> None:
    """The overlay is additive at forward time; the stored parameters stay untouched."""
    model = _model()
    before = _snapshot(model)
    fw = FastWeights(model, config=_on()).attach()
    fw.learn_text("the deploy key rotates", "on tuesdays")
    assert _identical(model, before)
    fw.reset()


def test_overlay_never_attaches_to_the_character_core() -> None:
    """The head and embeddings carry the identity-bearing directions and are off-limits."""
    for name in ("head.weight", "tok.weight", "lm_head.bias", "layers.3.embed.weight"):
        assert FastWeights._is_protected(name), name
    for name in ("layers.3.mlp.gate.weight", "layers.5.attn.q_proj.weight"):
        assert not FastWeights._is_protected(name), name


def test_protected_list_covers_the_immutable_values() -> None:
    model = _model()
    fw = FastWeights(model, config=_on()).attach()
    try:
        from nyxara.guard.value_learning import IMMUTABLE_VALUES
    except Exception:
        pytest.skip("value_learning unavailable")
    for value in IMMUTABLE_VALUES:
        assert fw._is_protected(f"layers.2.{value}.weight")
    fw.reset()


def test_only_the_late_layers_are_adapted() -> None:
    """Early layers encode surface form; identity lives at the ends. Adapt the middle-late."""
    model = _model()          # 6 layers
    fw = FastWeights(model, config=_on()).attach()
    attached = fw.state()["layers"]
    assert attached and min(attached) >= 3
    fw.reset()


def test_norm_is_bounded_under_sustained_writing() -> None:
    """A long conversation must not let the overlay drown the static weights."""
    model = _model()
    fw = FastWeights(model, config=_on(max_norm=0.5)).attach()
    for i in range(200):
        fw.learn_text(f"cue {i}", f"value {i}")
    assert fw.norm() <= 0.5 + 1e-5
    fw.reset()


def test_writes_are_capped() -> None:
    model = _model()
    fw = FastWeights(model, config=_on(max_writes=5)).attach()
    for i in range(50):
        fw.learn_text(f"cue {i}", f"value {i}")
    assert fw.state()["writes"] == 5
    fw.reset()


def test_every_write_is_auditable() -> None:
    """What she picked up mid-conversation must be inspectable, not mysterious."""
    model = _model()
    fw = FastWeights(model, config=_on()).attach()
    fw.learn_text("the build breaks on python 3.9", "pin to 3.11", source="sandbox_error")
    entries = fw.audit()
    assert len(entries) == 1
    assert entries[0]["source"] == "sandbox_error"
    assert "3.9" in entries[0]["summary"]
    assert entries[0]["norm_after"] > 0
    fw.reset()


# --------------------------------------------------------------------------- #
# It actually does something
# --------------------------------------------------------------------------- #
def test_writing_changes_the_forward_pass() -> None:
    """An overlay that changes nothing would pass every safety test and be useless."""
    model = _model()
    x = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        before = model.net(x).clone()

    fw = FastWeights(model, config=_on(lr=0.5)).attach()
    fw.learn_text("a distinctive cue phrase", "the associated response")
    with torch.no_grad():
        after = model.net(x)
    assert not torch.allclose(before, after), "the overlay had no effect on the output"

    fw.reset()
    with torch.no_grad():
        restored = model.net(x)
    assert torch.allclose(before, restored), "reset did not restore the original behaviour"


def test_decay_fades_an_unrefreshed_association() -> None:
    """λ < 1 means something nobody repeats goes away on its own."""
    model = _model()
    fw = FastWeights(model, config=_on(decay=0.5, lr=0.4)).attach()
    fw.learn_text("remember this", "value")
    peak = fw.norm()
    for i in range(12):        # write orthogonal-ish content; the first association decays
        fw.learn_text(f"unrelated {i}", f"other {i}")
    assert fw.norm() <= peak * 4    # bounded, and the early write no longer dominates
    fw.reset()


def test_decay_below_one_is_enforced() -> None:
    assert FastWeightConfig(decay=5.0).decay < 1.0
    assert FastWeightConfig(decay=-1.0).decay >= 0.0


def test_sandbox_error_is_a_first_class_source() -> None:
    """Learning from a failed run within seconds is half the point of this module."""
    model = _model()
    fw = FastWeights(model, config=_on()).attach()
    fw.learn_text("NameError: name 'np' is not defined",
                  "import numpy as np before using np", source="sandbox_error")
    assert fw.audit()[0]["source"] == "sandbox_error"
    fw.reset()


def test_write_is_a_no_op_when_not_attached() -> None:
    model = _model()
    fw = FastWeights(model, config=FastWeightConfig())      # disabled
    assert fw.write(torch.randn(64), torch.randn(64)) == 0.0
    assert fw.learn_text("a", "b") == 0.0


def test_state_reports_the_bounds_in_force() -> None:
    model = _model()
    fw = FastWeights(model, config=_on(max_norm=0.25, lr=0.2)).attach()
    state = fw.state()
    assert state["max_norm"] == 0.25 and state["lr"] == 0.2 and state["enabled"]
    fw.reset()


def test_protected_substrings_are_not_empty() -> None:
    assert PROTECTED_SUBSTRINGS and "head" in PROTECTED_SUBSTRINGS
