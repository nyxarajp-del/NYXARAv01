"""Tests for F8 — continuous latent world-simulators (mind/continuous_world.py).

She must fit a continuous vector field and predict a held-out tail, recover physical parameters by
gradient descent (differentiable physics), and stay honest (measured error; too little data → no fit).
Runs torch-free. No LLM anywhere.
"""

from __future__ import annotations

from nyxara.mind.continuous_world import (
    ContinuousDynamics,
    DifferentiablePhysics,
    torch_available,
)


def _decay(dt=0.1, n=40, rate=0.5, x0=1.0):
    traj = [[x0]]
    for _ in range(n):
        traj.append([traj[-1][0] + dt * (-rate * traj[-1][0])])
    return traj


def test_continuous_dynamics_predicts_holdout():
    err = ContinuousDynamics().holdout_error(_decay(), 0.1)
    assert err < 1e-3          # a learned vector field that genuinely generalises


def test_rollout_tracks_a_linear_system():
    cd = ContinuousDynamics()
    traj = _decay()
    assert cd.fit(traj, 0.1)
    roll = cd.rollout(traj[0], 40, 0.1)
    # the integrated field stays close to the true decay and heads toward 0
    assert abs(roll[-1][0]) < abs(roll[0][0])
    assert abs(roll[20][0] - traj[20][0]) < 0.05


def test_too_little_data_does_not_fit():
    assert not ContinuousDynamics().fit([[1.0], [0.9]], 0.1)


def test_differentiable_physics_recovers_parameters():
    dp = DifferentiablePhysics()
    obs = dp.simulate(1.0, 0.1, 1.0, 0.0, 60, 0.1)     # true a=1.0, b=0.1
    fit = dp.recover(obs, 0.1, iters=500, lr=0.02)
    assert abs(fit.a - 1.0) < 0.15
    assert abs(fit.b - 0.1) < 0.10


def test_recovery_reduces_loss():
    dp = DifferentiablePhysics()
    obs = dp.simulate(0.8, 0.2, 1.0, 0.0, 60, 0.1)
    start_loss = dp._loss(1.0, 0.5, obs, 0.1)          # the initial guess
    fit = dp.recover(obs, 0.1, iters=400, lr=0.02)
    assert fit.loss < start_loss                        # gradient descent actually descended


def test_torch_available_is_a_bool():
    assert isinstance(torch_available(), bool)          # honest capability report
