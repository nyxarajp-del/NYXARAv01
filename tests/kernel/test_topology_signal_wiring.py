"""P0-B — dynamic topology growth can actually fire.

The autonomic tick used to call ``topology.maybe_grow()`` with no arguments, but ``maybe_grow``
returns ``None`` immediately without a real ``CapacitySignal`` and a ``source`` — so Net2Net
growth was wired yet could never trigger. The kernel now derives a *real* capacity signal from
lived telemetry (the journal's action failure rate + the honesty calibrator) and supplies the
live champion genome as the source. These tests prove the signal is genuine and that, under
sustained measured pressure, the growth monitor actually fires (capabilities #11, #15).
"""

from __future__ import annotations

from nyxara.growth.genesis import ArchitectureGenome, LayerGene
from nyxara.growth.topology import CapacityMonitor
from nyxara.kernel.config import TopologyConfig
from nyxara.kernel.orchestrator import NyxaraCore
from nyxara.planning.journal import ActionStatus


def _base_genome() -> ArchitectureGenome:
    return ArchitectureGenome(n_embd=64, block_size=16,
                              layers=[LayerGene(op="attention", n_head=4),
                                      LayerGene(op="gated_mlp")])


def _record_actions(nyx: NyxaraCore, status: ActionStatus, n: int) -> None:
    for i in range(n):
        aid = nyx.journal.record_action(f"act-{i}", goal="serve the Master", rationale="r",
                                        autonomous=False, confidence=0.9, reversibility=1.0)
        nyx.journal.record_outcome(aid, status=status)


# -------------------- no growth without lived evidence -------------------- #
def test_capacity_signal_none_without_enough_actions():
    nyx = NyxaraCore()
    assert nyx._capacity_signal() is None              # an idle tick stays a cheap no-op


# -------------------- the signal reflects REAL failure rate -------------------- #
def test_capacity_signal_tracks_real_failure_rate():
    nyx = NyxaraCore()
    _record_actions(nyx, ActionStatus.FAILED, 10)
    sig = nyx._capacity_signal()
    assert sig is not None
    assert sig.problem_difficulty >= 0.9               # nearly every action failed → hard
    assert sig.saturation >= 0.8                       # operating at the edge of capacity


# -------------------- a real signal actually triggers growth -------------------- #
def test_real_pressure_triggers_growth_monitor():
    nyx = NyxaraCore()
    _record_actions(nyx, ActionStatus.FAILED, 12)
    sig = nyx._capacity_signal()
    decision = CapacityMonitor(cfg=TopologyConfig()).should_grow(sig, genome=_base_genome())
    assert decision.should_grow is True                # the loop can now fire (it never could)


# -------------------- success means no growth (no fabricated pressure) -------------------- #
def test_healthy_history_does_not_grow():
    nyx = NyxaraCore()
    _record_actions(nyx, ActionStatus.SUCCEEDED, 20)
    sig = nyx._capacity_signal()
    assert sig is not None
    assert sig.problem_difficulty < 0.7                # her actions are succeeding → no pressure
    decision = CapacityMonitor(cfg=TopologyConfig()).should_grow(sig, genome=_base_genome())
    assert decision.should_grow is False
