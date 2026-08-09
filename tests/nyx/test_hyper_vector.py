"""NYX V.02 · vector-symbolic structure — what is bound to what, and where retrieval breaks.

`semantics` says how *close* two concepts are. It cannot say which role is bound to which
filler, and "the engine turns the flywheel" versus "the flywheel turns the engine" is exactly
the case where that difference is the whole meaning. The other half of this file is the honest
one: capacity is real, and `capacity_probe` measures where it breaks on this machine.
"""

from __future__ import annotations

import pytest

from nyxara.kernel.config import NyxConfig
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.hyper_vector import Bound, VectorSymbolic

pytestmark = pytest.mark.skipif(not VectorSymbolic().available(),
                                reason="the hyperdimensional algebra is unavailable")


def _vsa(**kw) -> VectorSymbolic:
    kw.setdefault("dim", 10000)
    return VectorSymbolic(**kw)


def _countries(vsa: VectorSymbolic) -> VectorSymbolic:
    vsa.encode("france", {"capital": "paris", "currency": "euro", "continent": "europe"})
    vsa.encode("japan", {"capital": "tokyo", "currency": "yen", "continent": "asia"})
    return vsa


# --------------------------------------------------------------------------- #
# Structure in one vector — the thing a bag of words cannot hold
# --------------------------------------------------------------------------- #
def test_a_role_binds_and_comes_back_out():
    vsa = _vsa()
    vsa.encode("s", {"agent": "engine", "action": "turns", "patient": "flywheel"})
    assert vsa.role_of("s", "agent").filler == "engine"
    assert vsa.role_of("s", "patient").filler == "flywheel"


def test_every_retrieval_carries_a_margin():
    vsa = _vsa()
    vsa.encode("s", {"agent": "engine", "patient": "flywheel"})
    got = vsa.role_of("s", "agent")
    assert got.trusted and got.margin > 0.0 and got.runner_up


def test_the_same_words_in_opposite_roles_are_different_structures():
    """This is the case similarity cannot see and structure can."""
    vsa = _vsa()
    vsa.encode("forward", {"agent": "engine", "patient": "flywheel"})
    vsa.encode("reverse", {"agent": "flywheel", "patient": "engine"})
    assert vsa.role_of("forward", "agent").filler == "engine"
    assert vsa.role_of("reverse", "agent").filler == "flywheel"


def test_unpacking_returns_every_role():
    vsa = _vsa()
    vsa.encode("s", {"agent": "engine", "action": "turns", "patient": "flywheel"})
    unpacked = vsa.unpack("s")
    assert {k: v.filler for k, v in unpacked.items()} == {
        "agent": "engine", "action": "turns", "patient": "flywheel"}


def test_asking_an_unknown_structure_is_reported_not_guessed():
    got = _vsa().role_of("nothing", "agent")
    assert got.filler is None and "nothing is bound" in got.note


def test_an_empty_encode_produces_nothing():
    assert _vsa().encode("s", {}) is None


# --------------------------------------------------------------------------- #
# Order — permutation makes sequence structural
# --------------------------------------------------------------------------- #
def test_a_then_b_is_not_the_same_vector_as_b_then_a():
    vsa = _vsa()
    different = vsa.order_differs(["a", "b"], ["b", "a"])
    same = vsa.order_differs(["a", "b"], ["a", "b"])
    assert same == pytest.approx(1.0, abs=1e-6)
    assert different < 0.2


def test_a_sequence_can_be_encoded_alongside_roles():
    vsa = _vsa()
    bound = vsa.encode("plan", {"goal": "ship"}, sequence=["test", "commit", "push"])
    assert bound is not None and bound.sequence == ["test", "commit", "push"]
    assert "▸" in bound.render()


# --------------------------------------------------------------------------- #
# Zero-shot analogy — over structures, and honest about bare symbols
# --------------------------------------------------------------------------- #
def test_the_mapping_between_two_structures_transports():
    """She was never told paris and tokyo correspond. That is the inference."""
    vsa = _countries(_vsa())
    got = vsa.analogy("france", "japan", "paris")
    assert got.filler == "tokyo" and got.trusted
    assert "never told these two correspond" in got.note


def test_the_same_mapping_transports_a_different_role():
    vsa = _countries(_vsa())
    assert vsa.analogy("france", "japan", "euro").filler == "yen"


def test_bare_symbols_have_no_relation_to_transport_and_she_says_so():
    """Two random vectors share no structure. Dressing that up would be the lie."""
    vsa = _vsa()
    for name in ("paris", "france", "tokyo", "japan", "italy", "rome"):
        vsa.symbol(name)
    got = vsa.analogy("paris", "france", "tokyo")
    assert got.trusted is False
    assert "bare symbols, not structures" in got.note


def test_an_analogy_with_nowhere_to_land_says_that_rather_than_raising():
    vsa = _vsa()
    for name in ("a", "b", "c"):
        vsa.symbol(name)
    got = vsa.analogy("a", "b", "c")
    assert got.filler is None and "nothing left in my vocabulary" in got.note


def test_an_analogy_over_a_symbol_she_does_not_hold_returns_nothing():
    vsa = _countries(_vsa())
    got = vsa.analogy("france", "japan", "berlin")
    assert got.filler is None and "not a symbol I hold" in got.note


# --------------------------------------------------------------------------- #
# Associative recall is a different claim, and has its own method
# --------------------------------------------------------------------------- #
def test_a_bundled_relation_recalls_what_she_was_given():
    vsa = _vsa()
    assert vsa.relate("capital-of", [("paris", "france"), ("tokyo", "japan"),
                                     ("rome", "italy")]) == 3
    got = vsa.across("capital-of", "tokyo")
    assert got.filler == "japan" and got.trusted


def test_associative_recall_says_it_is_not_inference():
    vsa = _vsa()
    vsa.relate("capital-of", [("paris", "france"), ("tokyo", "japan")])
    assert "not an inference" in vsa.across("capital-of", "tokyo").note


def test_querying_an_unbundled_relation_is_reported():
    assert "no relation is bundled" in _vsa().across("nope", "x").note


# --------------------------------------------------------------------------- #
# Capacity is real, and measured on this machine
# --------------------------------------------------------------------------- #
def test_capacity_breaks_measurably_at_a_small_dimension():
    """The honest counterpart to every '10,000-dimensional' claim."""
    cap = VectorSymbolic(dim=256).capacity_probe(up_to=24)
    assert cap.clean_up_to >= 1
    assert cap.broke_at > cap.clean_up_to
    assert cap.margins and cap.margins[0][1] > cap.margins[-1][1]


def test_a_large_dimension_holds_far_more():
    small = VectorSymbolic(dim=256).capacity_probe(up_to=24)
    large = VectorSymbolic(dim=10000).capacity_probe(up_to=24)
    assert large.clean_up_to > small.clean_up_to


def test_an_overfull_bundle_reports_itself_rather_than_returning_junk_silently():
    vsa = VectorSymbolic(dim=64, min_margin=0.2)
    roles = {f"r{i}": f"f{i}" for i in range(40)}
    vsa.encode("crowded", roles)
    results = [vsa.role_of("crowded", r) for r in list(roles)[:10]]
    assert any(not r.trusted for r in results)
    assert any("fuller than this dimension retrieves cleanly" in r.note
               for r in results if not r.trusted)


def test_the_untrusted_count_is_tracked():
    vsa = VectorSymbolic(dim=64, min_margin=0.5)
    vsa.encode("crowded", {f"r{i}": f"f{i}" for i in range(30)})
    for i in range(5):
        vsa.role_of("crowded", f"r{i}")
    assert vsa.stats()["untrusted"] > 0


# --------------------------------------------------------------------------- #
# Alongside semantics
# --------------------------------------------------------------------------- #
def test_similarity_and_structure_are_reported_side_by_side():
    brain = NyxBrain(NyxConfig())
    brain.bind_structure("forward", {"agent": "engine", "patient": "flywheel"})
    brain.bind_structure("reverse", {"agent": "flywheel", "patient": "engine"})
    got = brain.vsa.compare("forward", "reverse")
    assert got["structure"]["a"]["roles"] != got["structure"]["b"]["roles"]
    assert "opposite on the second" in got["note"]


# --------------------------------------------------------------------------- #
# Through the brain, and across a restart
# --------------------------------------------------------------------------- #
def test_bind_and_analogy_are_reachable_on_the_brain():
    brain = NyxBrain(NyxConfig())
    brain.bind_structure("france", {"capital": "paris"})
    brain.bind_structure("japan", {"capital": "tokyo"})
    assert brain.analogy("france", "japan", "paris").filler == "tokyo"


def test_structures_survive_a_restart():
    brain = NyxBrain(NyxConfig())
    brain.bind_structure("s", {"agent": "engine", "patient": "flywheel"})
    reborn = NyxBrain(NyxConfig())
    reborn.load_dict(brain.to_dict())
    assert reborn.vsa.role_of("s", "agent").filler == "engine"


def test_disabling_the_layer_leaves_v01_behaviour():
    brain = NyxBrain(NyxConfig(vsa_enabled=False))
    assert brain.vsa is None
    assert brain.bind_structure("s", {"a": "b"}) is None
    assert brain.analogy("a", "b", "c") is None


# --------------------------------------------------------------------------- #
# Honesty and fail-soft
# --------------------------------------------------------------------------- #
def test_stats_say_capacity_is_measured_not_absolute():
    note = _vsa().stats()["note"]
    assert "capacity is REAL" in note
    assert "measured retrieval, never 'absolute retrieval'" in note
    assert "Nothing here is trained" in note


def test_everything_is_fail_soft_on_junk():
    vsa = _vsa()
    assert vsa.encode(None, None) is None
    assert vsa.role_of(None, None).filler is None
    assert vsa.analogy(None, None, None).filler is None
    assert vsa.load_dict(None) is False
    assert vsa.load_dict({"bindings": ["junk", {"name": ""}]}) is True


def test_a_bound_structure_renders_and_serialises():
    bound = Bound(name="s", roles={"agent": "engine"}, slots=1, dim=10000)
    assert "⊗" in bound.render() and bound.to_dict()["slots"] == 1
