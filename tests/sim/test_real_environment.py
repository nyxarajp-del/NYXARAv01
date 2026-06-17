"""Tests for nyxara.sim.real_environment — the real sensorimotor world-model fuel."""

from __future__ import annotations

import os
import random

import pytest

from nyxara.mind.world_model import WorldModel
from nyxara.sim.real_environment import (
    RealEnvironment,
    Transition,
    sensorimotor_tick,
)

_I_FILES = 0
_I_SUBDIRS = 5


def _drive(env: RealEnvironment, wm: WorldModel, n: int, seed: int = 1) -> None:
    rng = random.Random(seed)
    for _ in range(n):
        sensorimotor_tick(env, wm, rng=rng)


def test_sense_is_fixed_dim_numeric():
    with RealEnvironment() as env:
        s = env.sense()
        assert len(s) == 6
        assert all(isinstance(x, float) and not isinstance(x, bool) for x in s)
        assert s[_I_FILES] == 0.0           # fresh scratch dir
        assert s[4] >= 1.0                  # cpu_count


def test_create_file_changes_real_state():
    with RealEnvironment() as env:
        before = env.sense()
        tr = env.step("create_file")
        assert tr.next_state[_I_FILES] == before[_I_FILES] + 1.0
        assert tr.next_state[1] > before[1]            # total_kb grew
        # a real file exists on disk, strictly under the scratch root
        files = [e.path for e in os.scandir(env.root) if e.is_file()]
        assert len(files) == 1
        assert os.path.realpath(files[0]).startswith(env.root + os.sep)


def test_delete_is_reversible_via_restore():
    with RealEnvironment() as env:
        env.step("create_file")
        env.step("create_file")
        n_before = env.sense()[_I_FILES]
        env.step("delete_file")
        assert env.sense()[_I_FILES] == n_before - 1.0
        restored = env.restore_all()
        assert restored >= 1
        assert env.sense()[_I_FILES] == n_before


def test_guard_refuses_escape():
    with RealEnvironment() as env:
        with pytest.raises(ValueError):
            env._guard(os.path.join(env.root, "..", "escape.txt"))
        # a path well outside the root is refused too
        with pytest.raises(ValueError):
            env._guard("/etc/passwd")


def test_world_model_learns_real_dynamics():
    with RealEnvironment(seed=3) as env:
        wm = WorldModel(k=3, distance_scale=50.0)
        _drive(env, wm, 80)
        assert len(wm) == 80
        assert set(wm.actions()) <= set(RealEnvironment.ACTIONS)
        s = env.sense()
        pred = wm.predict(s, "create_file")
        # the model reproduces the REAL next state: file_count rises by ~1
        assert pred.next_state[_I_FILES] - s[_I_FILES] == pytest.approx(1.0, abs=0.3)
        assert pred.confidence > 0.6


def test_wrong_action_is_distinguishable():
    with RealEnvironment(seed=5) as env:
        wm = WorldModel(k=3, distance_scale=50.0)
        _drive(env, wm, 80)
        s = env.sense()
        d_create = wm.predict(s, "create_file").next_state[_I_FILES] - s[_I_FILES]
        d_delete = wm.predict(s, "delete_file").next_state[_I_FILES] - s[_I_FILES]
        d_noop = wm.predict(s, "noop").next_state[_I_FILES] - s[_I_FILES]
        # action-conditioned dynamics: delete < noop < create
        assert d_delete < d_noop < d_create


def test_counterfactual_prefers_higher_reward_policy():
    with RealEnvironment(seed=9) as env:
        wm = WorldModel(k=3, distance_scale=50.0)
        _drive(env, wm, 120)
        s = env.sense()
        cf = wm.counterfactual(s, ["create_file"] * 4, ["delete_file"] * 4, steps=4)
        # creating earns +0.5/step, deleting -0.3/step → policy A wins
        assert cf.trajectory_a.total_reward > cf.trajectory_b.total_reward


def test_reward_is_real():
    with RealEnvironment() as env:
        assert env.step("noop").reward == 0.0
        assert env.step("create_file").reward > 0.0
        # delete a real file: valid cleanup, mildly costly (negative but not the -1 failure)
        r_delete = env.step("delete_file").reward
        assert -0.5 < r_delete < 0.0


def test_failed_op_is_negative_not_crash():
    with RealEnvironment() as env:
        # no files yet → delete cannot do anything → failure reward, no exception
        tr = env.step("delete_file")
        assert isinstance(tr, Transition)
        assert tr.reward == -1.0
        assert tr.next_state == tr.state


def test_max_files_cap_self_regulates():
    with RealEnvironment(max_files=3) as env:
        for _ in range(10):
            env.step("create_file")
        assert env.sense()[_I_FILES] <= 3.0
        assert "create_file" not in env.safe_actions()


def test_cleanup_removes_scratch():
    env = RealEnvironment()
    root = env.root
    assert os.path.isdir(root)
    env.cleanup()
    assert not os.path.exists(root)


def test_provided_root_is_not_deleted(tmp_path):
    env = RealEnvironment(root=str(tmp_path))
    env.step("create_file")
    env.cleanup()
    assert os.path.isdir(str(tmp_path))   # we did not create it, so we do not remove it
