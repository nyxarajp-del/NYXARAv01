"""Layer 0 — the substrate has to be *correct arithmetic*, or nothing above it means anything.

The load-bearing test here is :func:`test_added_ops_gradients_are_exact`: every op added in
``nyx001/substrate/autograd.py`` is checked against central differences. An autograd with one
wrong backward produces a model that trains to a plausible-looking loss and has learned the wrong
function — the hardest class of bug to notice downstream.
"""

from __future__ import annotations

import numpy as np
import pytest

from nyxara.nyx001 import substrate as S
from nyxara.nyx001.substrate import autograd as A


def _numeric_grad(fn, arrays, i, eps=1e-6):
    num = np.zeros_like(arrays[i])
    it = np.nditer(arrays[i], flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        up = [a.copy() for a in arrays]
        dn = [a.copy() for a in arrays]
        up[i][idx] += eps
        dn[i][idx] -= eps
        num[idx] = (fn(*[A.parameter(a) for a in up]).data
                    - fn(*[A.parameter(a) for a in dn]).data) / (2 * eps)
        it.iternext()
    return num


@pytest.mark.parametrize("name,fn,shapes,positive", [
    ("sub", lambda a, b: A.sum_all(A.sub(a, b)), [(3, 4), (3, 4)], False),
    ("div", lambda a, b: A.sum_all(A.div(a, b)), [(3, 4), (3, 4)], True),
    ("exp", lambda a: A.sum_all(A.exp(a)), [(3, 4)], False),
    ("log", lambda a: A.sum_all(A.log(a)), [(3, 4)], True),
    ("tanh", lambda a: A.sum_all(A.tanh(a)), [(3, 4)], False),
    ("square", lambda a: A.sum_all(A.square(a)), [(3, 4)], False),
    ("abs_val", lambda a: A.sum_all(A.abs_val(a)), [(3, 4)], True),
    ("mean_all", lambda a: A.mean_all(a), [(3, 4)], False),
    ("sum_lastdim", lambda a: A.sum_all(A.sum_lastdim(a)), [(3, 4)], False),
    ("mean_lastdim", lambda a: A.sum_all(A.mean_lastdim(a)), [(3, 4)], False),
    ("concat_last", lambda a, b: A.sum_all(A.square(A.concat_last([a, b]))),
     [(3, 2), (3, 5)], False),
    ("cosine", lambda a, b: A.sum_all(A.cosine(a, b)), [(3, 4), (3, 4)], False),
    ("composite", lambda a, b: A.mean_all(A.tanh(A.sub(A.matmul(a, b), a))),
     [(3, 3), (3, 3)], False),
])
def test_added_ops_gradients_are_exact(name, fn, shapes, positive):
    """Analytic gradients must match central differences. A wrong backward is a silent disaster."""
    rng = np.random.default_rng(0)
    arrays = [rng.random(s) + (0.5 if positive else -0.5) for s in shapes]
    tensors = [A.parameter(a.copy()) for a in arrays]
    A.backward(fn(*tensors))
    for i, t in enumerate(tensors):
        num = _numeric_grad(fn, arrays, i)
        scale = max(1.0, float(np.max(np.abs(num))))
        assert np.max(np.abs(num - t.grad)) / scale < 2e-5, f"{name} arg{i} gradient is wrong"


def test_mse_gradient_is_exact():
    rng = np.random.default_rng(1)
    target = np.full((3, 4), 0.25)
    arr = rng.random((3, 4))
    t = A.parameter(arr.copy())
    A.backward(A.mse(t, target))
    num = _numeric_grad(lambda a: A.mse(a, target), [arr], 0)
    assert np.max(np.abs(num - t.grad)) < 1e-5


def test_optimizer_is_plain_rmsprop_under_a_neutral_state():
    """The ablation baseline: with no signals reported, the cognitive gain is EXACTLY 1.0.

    Without this identity, any measured difference between "with modulation" and "without" could
    be a different base optimizer rather than the modulation.
    """
    p = [S.parameter(np.ones((2, 2)))]
    opt = S.AdaptiveOptimizer(p)
    assert opt.gain_for(None) == 1.0
    assert opt.gain_for(S.CognitiveState()) == 1.0
    assert S.CognitiveState().is_neutral


def test_modulation_directions_are_the_ones_documented():
    """Error raises the step; uncertainty LOWERS it. The second is the counter-intuitive one."""
    opt = S.AdaptiveOptimizer([S.parameter(np.ones((2, 2)))])
    assert opt.gain_for(S.CognitiveState(error=1.0)) > 1.0
    assert opt.gain_for(S.CognitiveState(novelty=1.0)) > 1.0
    assert opt.gain_for(S.CognitiveState(uncertainty=1.0)) < 1.0
    assert opt.gain_for(S.CognitiveState(memory_pressure=1.0)) < 1.0
    assert opt.gain_for(S.CognitiveState(error=1.0, time_budget=0.0)) < \
        opt.gain_for(S.CognitiveState(error=1.0, time_budget=1.0))


def test_gain_is_bounded_however_extreme_the_signals():
    """An unbounded cognitive gain is a divergence waiting for its input."""
    opt = S.AdaptiveOptimizer([S.parameter(np.ones((2, 2)))], gain_floor=0.1, gain_ceiling=3.0)
    for st in (S.CognitiveState(error=1.0, novelty=1.0, time_budget=1.0),
               S.CognitiveState(uncertainty=1.0, memory_pressure=1.0, time_budget=0.0)):
        assert 0.1 <= opt.gain_for(st) <= 3.0


def test_cognitive_state_clamps_out_of_range_inputs():
    st = S.CognitiveState(error=5.0, uncertainty=-2.0, time_budget=99.0)
    assert st.error == 1.0 and st.uncertainty == 0.0 and st.time_budget == 1.0


def test_substrate_actually_learns_a_nonlinear_function():
    """The whole package rests on this: does gradient descent on this substrate work at all?"""
    rng = S.rng_for(7)
    n, d, h = 256, 4, 64
    x = rng.normal(0, 1, (n, d))
    y = (np.sin(3 * x[:, 0]) * np.cos(2 * x[:, 1]) + x[:, 2] ** 2).reshape(n, 1)
    w1, b1 = S.he(rng, d, h), S.zeros(h)
    w2, b2 = S.he(rng, h, h), S.zeros(h)
    w3, b3 = S.xavier(rng, h, 1), S.zeros(1)
    params = [w1, b1, w2, b2, w3, b3]
    opt = S.AdaptiveOptimizer(params, lr=3e-3)
    obj = S.CompositeObjective(w_concept=0, w_compression=0, w_calibration=0, w_curiosity=0)

    def forward(xb):
        hid = A.gelu(A.add(A.matmul(A.constant(xb), w1), b1))
        hid = A.gelu(A.add(A.matmul(hid, w2), b2))
        return A.add(A.matmul(hid, w3), b3)

    first = None
    for i in range(400):
        idx = rng.integers(0, n, 32)
        opt.zero_grad()
        loss = obj(prediction=obj.prediction_term(forward(x[idx]), y[idx]))
        A.backward(loss.total)
        opt.step()
        if i == 0:
            first = loss.value
    final = float(np.mean((forward(x).data - y) ** 2))
    assert final < first * 0.25, f"did not learn: {first:.4f} → {final:.4f}"


def test_compression_term_is_l1_not_l2():
    """L1 drives coordinates TO zero; L2 only shrinks them. Layer 14 needs the former."""
    obj = S.CompositeObjective()
    code = S.parameter(np.array([[3.0, -4.0]]))
    A.backward(obj.compression_term(code))
    # |x| has a constant-magnitude subgradient; x² would give gradients proportional to x.
    assert np.allclose(np.abs(code.grad), np.abs(code.grad)[0, 0]), "not a constant subgradient"


def test_curiosity_term_saturates_and_its_gradient_vanishes():
    """The safety property: a huge information gain must stop pulling harder."""
    obj = S.CompositeObjective(curiosity_ceiling=1.0)
    small = S.parameter(np.array(0.1))
    big = S.parameter(np.array(50.0))
    A.backward(obj.curiosity_term(small))
    A.backward(obj.curiosity_term(big))
    assert abs(float(big.grad)) < abs(float(small.grad)) * 1e-3, "ceiling does not remove the pull"
    assert float(obj.curiosity_term(big).data) >= -1.0001, "value exceeded the ceiling"


def test_objective_reports_each_term_separately():
    """A composite loss whose terms cannot be told apart is one number in five hats."""
    obj = S.CompositeObjective()
    loss = obj(prediction=A.constant(np.array(1.0)), compression=A.constant(np.array(2.0)))
    assert set(loss.terms) == {"prediction", "compression"}
    assert "concept" not in loss.contributions(), "a term never computed must not be reported"
    assert abs(sum(loss.contributions().values()) - 1.0) < 1e-6


def test_missing_terms_are_absent_not_zero():
    obj = S.CompositeObjective()
    assert obj(prediction=None).total is None
    assert obj(prediction=None).value == 0.0        # a total of nothing, truthfully


def test_init_schemes_are_seeded_and_reproducible():
    a = S.xavier(S.rng_for(3, stream=1), 4, 4)
    b = S.xavier(S.rng_for(3, stream=1), 4, 4)
    assert np.allclose(a.data, b.data)
    c = S.xavier(S.rng_for(3, stream=2), 4, 4)
    assert not np.allclose(a.data, c.data), "streams must not collide"


def test_orthogonal_init_is_actually_orthonormal():
    q = S.orthogonal(S.rng_for(1), 8, 8)
    assert np.allclose(q.data @ q.data.T, np.eye(8), atol=1e-9)


def test_uniform_gate_stays_strictly_inside_zero_one():
    g = S.uniform_gate(S.rng_for(1), 64)
    assert float(g.data.min()) > 0.0 and float(g.data.max()) < 1.0
