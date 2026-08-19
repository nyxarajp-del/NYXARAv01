"""A graft has to survive the thing the fabric does after every single turn.

``Fabric.expand`` potentiates, grows, depresses, prunes and apoptoses. Every one of those is
correct for structure she grew and destructive for structure that was converted: imported weights
would be depressed toward the floor and deleted, and grafted cells look unconnected to apoptosis
because their synapses live in the region rather than in ``out``/``inn``.
"""

from __future__ import annotations

import numpy as np

from nyxara.njp.fabric import Fabric
from nyxara.njp.graft import Grafter


def _grafted(*, plastic=False, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.08, size=(32, 64)).astype(np.float32)
    cal = [rng.random(64).astype(np.float32) for _ in range(16)]
    fabric = Fabric(seed=1)
    region = Grafter(fabric, timesteps=32, min_fidelity=0.9).graft_linear(
        "ffn", w, None, calibration=cal, verify_samples=cal, plastic=plastic)
    return fabric, region


def _churn(fabric, region, turns=30):
    for _ in range(turns):
        fabric.stimulate([region.pre_lo + k for k in range(0, region.n_pre, 3)])
        result = fabric.settle(max_steps=6)
        fabric.expand(outcome=1.0, result=result)
        fabric.tick += 100          # age past the 64-tick apoptosis grace window


def test_expansion_leaves_grafted_weights_byte_identical():
    fabric, region = _grafted()
    q, scale = region.weights.q.copy(), region.weights.scale.copy()
    _churn(fabric, region)
    assert (region.weights.q == q).all()
    assert (region.weights.scale == scale).all()


def test_expansion_leaves_grafted_thresholds_alone():
    fabric, region = _grafted()
    before = [fabric.cells[region.post_lo + k].threshold for k in range(region.n_post)]
    _churn(fabric, region)
    assert [fabric.cells[region.post_lo + k].threshold for k in range(region.n_post)] == before


def test_apoptosis_does_not_delete_grafted_cells():
    """The specific bug this guards: grafted cells have no entry in ``out``/``inn``, so the
    'connects to nothing and never fired' test is true of every one of them by construction."""
    fabric, region = _grafted()
    _churn(fabric, region, turns=40)
    assert all((region.post_lo + k) in fabric.cells for k in range(region.n_post))
    assert all((region.pre_lo + k) in fabric.cells for k in range(region.n_pre))


def test_synaptogenesis_does_not_grow_across_a_frozen_region():
    """Growth inside a converted layer would turn a verified function into a different one that
    no certificate describes."""
    fabric, region = _grafted()
    _churn(fabric, region)
    for pre, targets in fabric.out.items():
        for post in targets:
            assert not (fabric.is_frozen(pre) and fabric.is_frozen(post))


def test_a_plastic_region_is_deliberately_not_frozen():
    """The opt-in exists, and it must actually change the freeze — an option that silently
    does nothing is worse than no option."""
    fabric, region = _grafted(plastic=True)
    assert not fabric.is_frozen(region.post_lo)
    assert fabric.is_grafted(region.post_lo)      # still protected from apoptosis
    _churn(fabric, region)
    assert (region.post_lo + 0) in fabric.cells


def test_grown_and_grafted_are_counted_separately():
    """If these merge, ``growing`` becomes unfalsifiable: five billion declared synapses would
    swamp every number that is supposed to say whether the automaton is still growing."""
    fabric, region = _grafted()
    for i in range(1, 40):
        fabric.connect(i, i + 1, 0.5)
    stats = fabric.stats()
    assert stats["synapses"] == 39                       # hers, and only hers
    assert stats["grafted"]["parameters"] == region.n_params
    assert stats["grafted"]["cells"] == region.n_pre + region.n_post
    assert stats["cells"] == 40                          # grown cells, graft subtracted out


def test_stats_reports_no_graft_block_when_there_is_no_graft():
    """Absent, not a row of zeros — the distinction this package keeps everywhere."""
    assert Fabric(seed=1).stats()["grafted"] is None


def test_run_graft_reproduces_the_reference_and_registers_real_activity():
    """A settle is event-driven and gives one propagation step; a rate code needs the input held.
    ``run_graft`` is that driver, and the spikes must land on the fabric's own cells."""
    fabric, region = _grafted()
    x = np.random.default_rng(5).random(region.n_pre).astype(np.float32)
    rates = fabric.run_graft(region, x)
    ref = region.reference(x)
    cos = float(np.dot(rates, ref) / (np.linalg.norm(rates) * np.linalg.norm(ref)))
    assert cos > 0.99
    assert any(fabric.cells[region.post_lo + k].hits > 0 for k in range(region.n_post))


def test_a_save_reload_cycle_does_not_invent_growth():
    """The bug this pins: ``to_dict`` wrote grafted cells (they live in ``self.cells``) but not
    their regions, so on reload nothing claimed them and they were counted as cells she GREW.
    Measured before the fix: 20 grown cells with one small graft reloaded as 68 — the growth
    claim inflating by itself, silently, once per restart."""
    fabric, region = _grafted()
    for i in range(1, 20):
        fabric.connect(i, i + 1, 0.5)
    grown_cells = fabric.stats()["cells"]
    grown_synapses = fabric.stats()["synapses"]

    reloaded = Fabric(seed=1)
    reloaded.load_dict(fabric.to_dict())

    assert reloaded.stats()["cells"] == grown_cells
    assert reloaded.stats()["synapses"] == grown_synapses
    # And the graft is honestly gone rather than half-present: re-run the converter to restore it.
    assert reloaded.stats()["grafted"] is None
