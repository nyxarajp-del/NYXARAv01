"""Her settled beliefs must survive a restart, so her autonomous science compounds across
lives instead of resetting to zero each boot. Hermetic, no LLM.
"""

from __future__ import annotations

import os

from nyxara.growth.autonomous_scientist import AutonomousScientist


def test_beliefs_persist_across_a_restart(tmp_path):
    path = os.path.join(str(tmp_path), "beliefs.json")
    life1 = AutonomousScientist(path=path, save_every=2)
    life1.discover(cycles=6)
    life1.save()
    n = len(life1.belief_model())
    assert n > 0
    assert os.path.exists(path)

    life2 = AutonomousScientist(path=path)          # a fresh "boot" reloads her prior life
    assert len(life2.belief_model()) == n
    # a reloaded, already-settled proposition is not re-posed as brand new
    for belief in life2.belief_model().beliefs.values():
        if belief.statement:
            assert belief.statement in life2._seen


def test_missing_or_corrupt_file_never_bricks_the_mind(tmp_path):
    path = os.path.join(str(tmp_path), "beliefs.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{ this is not valid json")
    auto = AutonomousScientist(path=path)           # must not raise
    assert len(auto.belief_model()) == 0
    auto.discover(cycles=2)
    assert auto.save() is True                        # and can overwrite the corrupt file cleanly


def test_no_path_means_no_persistence_but_still_works():
    auto = AutonomousScientist()
    auto.discover(cycles=3)
    assert auto.save() is False                       # nothing to persist to, reported honestly
    assert len(auto.belief_model()) > 0
