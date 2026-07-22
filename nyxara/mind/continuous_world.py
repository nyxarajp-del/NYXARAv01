"""NYXARA · mind/continuous_world.py — continuous latent world-simulators (⏳, Rule 4, F8).

Discrete transition tables see the world in frames; F8 lets NYXARA model it in **continuous time** and
**recover the physics behind it** — sharpening prediction and planning.

* **Continuous dynamics (a learned vector field).** She fits ``dx/dt ≈ W·φ(x)`` over a small
  polynomial feature basis by least squares on finite-difference derivatives — a SINDy-style continuous
  surrogate (the same sparse-dynamics idea as :mod:`nyxara.growth.law_discovery`) — then integrates it
  (RK4) to roll the world forward. Honest: prediction error is measured on a **held-out** tail of the
  trajectory, never assumed.

* **Differentiable physics (parameter recovery by gradient descent).** Given a parametric law
  (a damped oscillator ``x'' = −a·x − b·x'``) she recovers its physical parameters ``(a, b)`` — stiffness
  and damping — by descending the MSE between her simulation and the observation. Gradients are
  finite-difference here (torch would make them exact autodiff — an optional sharpening, not required),
  so the whole module runs **torch-free on any box**.

No "100× accuracy" is claimed — only the measured held-out improvement. Pure standard library; **no
LLM**; it models the world, touching no gate and no character value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

__all__ = [
    "torch_available",
    "ContinuousDynamics",
    "DifferentiablePhysics",
    "PhysicsFit",
]

Vec = Sequence[float]


def torch_available() -> bool:
    try:
        import torch  # type: ignore  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# tiny pure-python linear algebra (so F8 runs on a bare machine)
# --------------------------------------------------------------------------- #
def _solve(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """Solve A x = b (n×n) by Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        M[col], M[piv] = M[piv], M[col]
        pivval = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= pivval
        for r in range(n):
            if r != col and abs(M[r][col]) > 0:
                factor = M[r][col]
                for j in range(col, n + 1):
                    M[r][j] -= factor * M[col][j]
    return [M[i][n] for i in range(n)]


def _lstsq(Phi: List[List[float]], y: List[float]) -> Optional[List[float]]:
    """Least-squares w minimising ||Phi w − y|| via the normal equations (small feature count)."""
    m = len(Phi)
    if m == 0:
        return None
    n = len(Phi[0])
    AtA = [[sum(Phi[k][i] * Phi[k][j] for k in range(m)) for j in range(n)] for i in range(n)]
    for i in range(n):
        AtA[i][i] += 1e-9                       # ridge — keep it well-conditioned
    Aty = [sum(Phi[k][i] * y[k] for k in range(m)) for i in range(n)]
    return _solve(AtA, Aty)


# --------------------------------------------------------------------------- #
# continuous dynamics — a learned vector field, integrated with RK4
# --------------------------------------------------------------------------- #
def _features(x: Vec) -> List[float]:
    """Polynomial basis up to degree 2 (plus a bias) — the sparse-dynamics feature library."""
    x = list(x)
    feats = [1.0] + list(x)
    for i in range(len(x)):
        for j in range(i, len(x)):
            feats.append(x[i] * x[j])
    return feats


@dataclass
class ContinuousDynamics:
    dim: int = 1
    W: List[List[float]] = field(default_factory=list)   # per-dim feature weights
    fitted: bool = False
    train_error: float = math.inf

    def fit(self, traj: Sequence[Vec], dt: float) -> bool:
        """Fit dx/dt = W·φ(x) from a trajectory via finite-difference derivatives + least squares."""
        if len(traj) < 4:
            return False
        self.dim = len(traj[0])
        Phi = [_features(traj[t]) for t in range(len(traj) - 1)]
        self.W = []
        errs = 0.0
        for d in range(self.dim):
            dy = [(traj[t + 1][d] - traj[t][d]) / dt for t in range(len(traj) - 1)]
            w = _lstsq(Phi, dy)
            if w is None:
                return False
            self.W.append(w)
            errs += sum((sum(Phi[k][i] * w[i] for i in range(len(w))) - dy[k]) ** 2
                        for k in range(len(dy))) / max(1, len(dy))
        self.train_error = errs / self.dim
        self.fitted = True
        return True

    def _deriv(self, x: Vec) -> List[float]:
        f = _features(x)
        return [sum(f[i] * self.W[d][i] for i in range(len(f))) for d in range(self.dim)]

    def step(self, x: Vec, dt: float) -> List[float]:
        """One RK4 integration step of the learned vector field."""
        x = list(x)
        k1 = self._deriv(x)
        k2 = self._deriv([x[i] + 0.5 * dt * k1[i] for i in range(self.dim)])
        k3 = self._deriv([x[i] + 0.5 * dt * k2[i] for i in range(self.dim)])
        k4 = self._deriv([x[i] + dt * k3[i] for i in range(self.dim)])
        return [x[i] + dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(self.dim)]

    def rollout(self, x0: Vec, steps: int, dt: float) -> List[List[float]]:
        out = [list(x0)]
        for _ in range(steps):
            out.append(self.step(out[-1], dt))
        return out

    def holdout_error(self, traj: Sequence[Vec], dt: float, split: float = 0.6) -> float:
        """Fit on the first part, measure mean-squared prediction error on the held-out tail."""
        cut = max(3, int(len(traj) * split))
        if not self.fit(traj[:cut], dt):
            return math.inf
        pred = self.rollout(traj[cut - 1], len(traj) - cut, dt)
        errs = [sum((pred[i][d] - traj[cut - 1 + i][d]) ** 2 for d in range(self.dim))
                for i in range(len(traj) - cut + 1)]
        return sum(errs) / len(errs)


# --------------------------------------------------------------------------- #
# differentiable physics — recover (a, b) of x'' = -a x - b x' by gradient descent
# --------------------------------------------------------------------------- #
@dataclass
class PhysicsFit:
    a: float                 # stiffness (ω²)
    b: float                 # damping
    loss: float
    iters: int

    def to_dict(self) -> dict:
        return {"a": round(self.a, 4), "b": round(self.b, 4),
                "loss": round(self.loss, 6), "iters": self.iters}


class DifferentiablePhysics:
    """A parametric damped-oscillator simulator whose parameters she recovers by gradient descent."""

    def simulate(self, a: float, b: float, x0: float, v0: float, steps: int, dt: float) -> List[float]:
        xs = [x0]
        x, v = x0, v0
        for _ in range(steps):
            acc = -a * x - b * v
            v = v + dt * acc
            x = x + dt * v
            xs.append(x)
        return xs

    def _loss(self, a: float, b: float, obs: Sequence[float], dt: float) -> float:
        sim = self.simulate(a, b, obs[0], (obs[1] - obs[0]) / dt, len(obs) - 1, dt)
        return sum((sim[i] - obs[i]) ** 2 for i in range(len(obs))) / len(obs)

    def recover(self, obs: Sequence[float], dt: float, *, iters: int = 300,
                lr: float = 0.05, eps: float = 1e-4) -> PhysicsFit:
        """Descend the MSE between simulation and observation to recover (a, b) — finite-diff gradients."""
        a, b = 1.0, 0.5
        loss = self._loss(a, b, obs, dt)
        for it in range(iters):
            ga = (self._loss(a + eps, b, obs, dt) - self._loss(a - eps, b, obs, dt)) / (2 * eps)
            gb = (self._loss(a, b + eps, obs, dt) - self._loss(a, b - eps, obs, dt)) / (2 * eps)
            a = max(0.0, a - lr * ga)
            b = max(0.0, b - lr * gb)
            new = self._loss(a, b, obs, dt)
            if abs(loss - new) < 1e-12:
                loss = new
                return PhysicsFit(a, b, loss, it + 1)
            loss = new
        return PhysicsFit(a, b, loss, iters)
