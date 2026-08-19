"""The converter itself: does a grafted layer compute what the original layer computed?

Everything here runs on synthetic arrays. That is the point — the numeric claim
``njp/graft.py`` makes is about the conversion, not about any particular 5.6 GB file, so it can
and should be checkable without one.
"""

from __future__ import annotations

import numpy as np
import pytest

from nyxara.njp.fabric import Fabric
from nyxara.njp.graft import (
    DEFAULT_GROUP, GraftedRegion, Grafter, QuantizedWeights, _pack_int4, _unpack_int4,
    balance_threshold, classify_tensor, read_gguf_tensors, verify_region,
)


def _layer(n_post=48, n_pre=96, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.08, size=(n_post, n_pre)).astype(np.float32)
    b = rng.normal(0, 0.02, size=n_post).astype(np.float32)
    cal = [rng.random(n_pre).astype(np.float32) for _ in range(24)]
    return w, b, cal


# --------------------------------------------------------------------------- #
# Quantisation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_pre", [16, 37, 64])
def test_int4_pack_roundtrip_is_exact_including_odd_widths(n_pre):
    rng = np.random.default_rng(1)
    q = rng.integers(-7, 8, size=(5, n_pre)).astype(np.int8)
    assert (_unpack_int4(_pack_int4(q), n_pre) == q).all()


@pytest.mark.parametrize("bits", [8, 4])
def test_matvec_matches_dequantised_matmul(bits):
    """``matvec`` must be the same function as ``dequantize`` then ``@`` — it is an optimisation,
    not a different arithmetic."""
    w, _b, _cal = _layer()
    x = np.random.default_rng(2).random(w.shape[1]).astype(np.float32)
    qw = QuantizedWeights.from_dense(w, bits=bits, group=DEFAULT_GROUP)
    assert np.allclose(qw.matvec(x), qw.dequantize() @ x, rtol=1e-4, atol=1e-4)


def test_quantisation_costs_the_advertised_memory():
    w, _b, _cal = _layer(n_post=64, n_pre=512)
    assert QuantizedWeights.from_dense(w, bits=8, group=32).bytes_per_param == pytest.approx(1.125)
    assert QuantizedWeights.from_dense(w, bits=4, group=32).bytes_per_param == pytest.approx(0.625)


def test_a_row_of_zeros_does_not_divide_by_zero():
    w = np.zeros((4, 64), dtype=np.float32)
    qw = QuantizedWeights.from_dense(w, bits=8)
    assert np.isfinite(qw.dequantize()).all()
    assert np.allclose(qw.matvec(np.ones(64, dtype=np.float32)), 0.0)


# --------------------------------------------------------------------------- #
# Threshold balancing
# --------------------------------------------------------------------------- #
def test_threshold_ignores_the_outlier_that_would_silence_the_layer():
    """The whole reason for a percentile rather than a max: one huge activation must not set the
    scale for everything else."""
    bulk = np.random.default_rng(3).normal(1.0, 0.1, size=10_000)
    with_outlier = np.concatenate([bulk, [1e6]])
    assert balance_threshold(with_outlier) < 10.0
    assert np.isclose(balance_threshold(bulk), balance_threshold(with_outlier), rtol=0.05)


def test_a_layer_that_never_activates_gets_a_usable_threshold():
    assert balance_threshold(np.zeros(100)) > 0.0
    assert balance_threshold(np.asarray([])) > 0.0


# --------------------------------------------------------------------------- #
# The conversion
# --------------------------------------------------------------------------- #
def test_converted_layer_reproduces_the_original_arithmetic():
    w, b, cal = _layer()
    g = Grafter(Fabric(seed=1), timesteps=64, min_fidelity=0.95)
    region = g.graft_linear("ffn", w, b, calibration=cal, verify_samples=cal)
    assert region is not None
    assert region.fidelity > 0.99, f"conversion only reached {region.fidelity}"


def test_fidelity_improves_with_simulation_length():
    """Rate-code error is O(1/T). If this inverts, the simulation is not doing what it claims."""
    w, b, cal = _layer()
    got = []
    for T in (8, 32, 128):
        g = Grafter(Fabric(seed=1), timesteps=T, min_fidelity=0.0)
        got.append(g.graft_linear(f"t{T}", w, b, calibration=cal, verify_samples=cal).fidelity)
    assert got[0] < got[1] < got[2]


def test_soft_reset_beats_the_fabrics_own_leaky_law_and_both_are_measured():
    """The deviation the module documents, checked rather than asserted: the leak discards the
    charge a rate code is counting, so ``fabric_lif`` must score worse — and be reportable."""
    w, b, cal = _layer()
    scores = {}
    for law in ("if_soft_reset", "fabric_lif"):
        g = Grafter(Fabric(seed=1), timesteps=64, min_fidelity=0.0, law=law)
        scores[law] = g.graft_linear(law, w, b, calibration=cal, verify_samples=cal).fidelity
    assert scores["if_soft_reset"] > scores["fabric_lif"]


def test_a_region_below_the_floor_is_refused_and_leaves_no_trace():
    """The refusal is the half that matters: a partial graft computes something plausible and
    wrong, and every later measurement would inherit it."""
    w, b, cal = _layer()
    fabric = Fabric(seed=1)
    before_cells, before_grafts = fabric.n_cells, len(fabric.grafts)
    g = Grafter(fabric, timesteps=64, min_fidelity=1.01)      # unreachable by construction
    assert g.graft_linear("doomed", w, b, calibration=cal, verify_samples=cal) is None
    assert fabric.n_cells == before_cells
    assert len(fabric.grafts) == before_grafts
    assert g.stats()["n_rejected"] == 1
    assert "fidelity" in g.stats()["rejected"][0]["reason"]


def test_certificate_reports_coverage_against_what_was_offered():
    w, b, cal = _layer()
    g = Grafter(Fabric(seed=1), timesteps=64, min_fidelity=0.95)
    g.graft_linear("kept", w, b, calibration=cal, verify_samples=cal)
    g.min_fidelity = 1.01
    g.graft_linear("refused", w, b, calibration=cal, verify_samples=cal)
    cert = g.stats()
    assert cert["params_available"] == 2 * w.size
    assert cert["params_grafted"] == w.size
    assert cert["coverage"] == pytest.approx(0.5)


def test_grafted_cells_never_collide_with_grown_ids():
    fabric = Fabric(seed=1)
    for i in range(1, 200):
        fabric.connect(i, i + 1, 0.4)
    grown = {c for c in fabric.cells}
    w, b, cal = _layer()
    region = Grafter(fabric, timesteps=32, min_fidelity=0.9).graft_linear(
        "ffn", w, b, calibration=cal, verify_samples=cal)
    assert region.post_lo > max(grown)
    # And her own counter must not have been dragged into the graft range.
    assert fabric.next_id <= max(grown) + 1


def test_the_gate_measures_quantisation_too_not_just_the_rate_code():
    """``GraftedRegion.reference`` is the region's own ANALOG output, so verifying against it
    compares quantised weights to quantised weights and cancels quantisation out of both sides.
    The acceptance gate must not use that: it builds its reference from the original float
    weights, and so must always be the stricter of the two numbers."""
    w, b, cal = _layer()
    g = Grafter(Fabric(seed=1), timesteps=64, bits=4, group=32, min_fidelity=0.0)
    region = g.graft_linear("ffn", w, b, calibration=cal, verify_samples=cal)
    rate_code_only = verify_region(region, cal).cosine
    assert region.fidelity <= rate_code_only
    assert region.fidelity < 0.999 < rate_code_only, "int4 must show a visible quantisation cost"


def test_verify_region_treats_mutual_silence_as_agreement():
    """A layer that should emit nothing and emits nothing has converted correctly. Scoring that
    zero would reject a working region."""
    region = GraftedRegion(
        name="silent", pre_lo=0, post_lo=100,
        weights=QuantizedWeights.from_dense(np.zeros((4, 8), dtype=np.float32)),
        bias=np.full(4, -5.0, dtype=np.float32), threshold=1.0)
    assert verify_region(region, [np.zeros(8, dtype=np.float32)]).cosine == 1.0


# --------------------------------------------------------------------------- #
# The GGUF side, which is allowed to be absent
# --------------------------------------------------------------------------- #
def test_tensor_policy_routes_each_family_to_its_honest_destination():
    assert classify_tensor("blk.12.ffn_gate.weight") == "graft"
    assert classify_tensor("blk.3.attn_q.weight") == "hybrid"
    assert classify_tensor("token_embd.weight") == "lookup"
    assert classify_tensor("rope_freqs") == "skip"


def test_the_policy_is_ordered_so_a_fused_projection_is_not_read_as_a_separate_one():
    """Found against the real file: `attn_qkv` contains `attn_q` as a substring, so an unordered
    table classified this model's FUSED qkv projection (24 of its 32 blocks) as a separate q
    projection. It landed on the right verdict for entirely the wrong reason — an accident that
    stops being harmless the moment the two verdicts differ."""
    assert classify_tensor("blk.0.attn_qkv.weight") == "hybrid"
    assert classify_tensor("output_norm.weight") == "scale"      # not swallowed by "output"
    assert classify_tensor("output.weight") == "lookup"


def test_state_space_tensors_are_counted_rather_than_skipped():
    """This model is a hybrid transformer/SSM: 24 blocks carry gated-deltanet `ssm_*` tensors.
    A recurrent state update is not a feedforward activation, so the conversion does not apply —
    but 400M+ parameters silently outside the denominator would inflate the coverage figure this
    module publishes."""
    assert classify_tensor("blk.0.ssm_out.weight") == "hybrid"
    assert classify_tensor("blk.0.ssm_norm.weight") == "scale"
    assert classify_tensor("blk.0.attn_gate.weight") == "hybrid"


def test_dequantize_refuses_anything_that_is_not_float_weights():
    """The silent failure this guards: GGUFReader hands back PACKED quantisation bytes, and
    `.astype(float32)` on them yields plausible numbers that are not weights. Every downstream
    stage — threshold, quantise, simulate, verify — stays internally consistent and issues a
    certificate for a layer computing nothing."""
    from nyxara.njp.graft import dequantize_tensor

    class _Raw:
        name = "blk.0.ffn_gate.weight"
        data = np.zeros((4, 8), dtype=np.uint8)
        tensor_type = None            # no qtype: passes straight through, must still be float32

    assert dequantize_tensor(_Raw()).dtype == np.float32


def test_an_unreadable_gguf_is_unavailability_not_an_exception():
    assert read_gguf_tensors("/definitely/not/here.gguf") == []
