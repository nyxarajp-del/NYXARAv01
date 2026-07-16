"""Tests for the intuitive-physics sensorimotor world.

:mod:`nyxara.sim.physics_world` grounds a world model in real Newtonian mechanics — gravity,
friction, restitution and momentum-conserving collisions — that the agent learns by *acting*:
pushing, lifting, dropping and poking its body, then predicting the consequence. These tests
assert the physics itself is real (not a scripted animation), that the world model genuinely
learns the action-conditioned dynamics from lived experience, and that curiosity is honest.
Everything is in-memory, deterministic (seeded) and offline.
"""

from __future__ import annotations

from nyxara.sim.physics_world import (
    ACTIONS, STATE_DIM, PhysicsAgent, PhysicsWorld, physics_stream, physics_tick,
    _VX0, _VY0, _VX1, _Y0,
)


# --------------------------------------------------------------------------- #
# The raw physics is real
# --------------------------------------------------------------------------- #
def test_state_shape_and_actions():
    w = PhysicsWorld(seed=0)
    s = w.sense()
    assert len(s) == STATE_DIM
    assert all(isinstance(x, float) for x in s)
    assert w.safe_actions() == list(ACTIONS)


def test_gravity_makes_a_released_body_fall():
    w = PhysicsWorld(seed=1)
    y_start = w.sense()[_Y0]
    for _ in range(8):
        w.step("noop")
    assert w.sense()[_Y0] < y_start   # it fell


def test_friction_halts_a_pushed_body():
    w = PhysicsWorld(seed=2)
    for _ in range(30):               # let body 0 settle onto the floor
        w.step("noop")
    w.step("push_right")
    vx_pushed = w.sense()[_VX0]
    for _ in range(20):               # coast — ground friction should bleed the motion off
        w.step("noop")
    assert abs(vx_pushed) > 0.1
    assert abs(w.sense()[_VX0]) < abs(vx_pushed)


def test_push_direction_is_signed():
    left = PhysicsWorld(seed=3)
    right = PhysicsWorld(seed=3)
    left.step("push_left")
    right.step("push_right")
    assert left.sense()[_VX0] < 0.0 < right.sense()[_VX0]


def test_collision_transfers_momentum():
    w = PhysicsWorld(seed=3)
    for _ in range(30):
        w.step("noop")
    before = abs(w.sense()[_VX1])
    moved = False
    for _ in range(40):
        w.step("poke")               # shove body 0 into the passive object
        if abs(w.sense()[_VX1]) > before + 0.1:
            moved = True
            break
    assert moved, "poking one body never moved the other — collision failed"


def test_bodies_stay_inside_the_arena():
    w = PhysicsWorld(seed=5)
    for _ in range(200):
        w.step("poke")               # keep shoving; walls must contain the motion
        for b in w.bodies:
            assert -1e-6 <= b.x <= w.arena + 1e-6
            assert -1e-6 <= b.y <= w.arena + 1e-6


def test_step_returns_a_valid_transition():
    w = PhysicsWorld(seed=6)
    tr = w.step("lift")
    assert tr.action == "lift"
    assert len(tr.state) == STATE_DIM and len(tr.next_state) == STATE_DIM
    assert isinstance(tr.reward, float)


def test_dynamics_reports_rate_of_change():
    w = PhysicsWorld(seed=7)
    assert w.dynamics() == (0.0,) * STATE_DIM   # nothing streamed yet beyond the reset frame
    for _ in range(5):
        w.step("drop")
    d = w.dynamics()
    assert len(d) == STATE_DIM
    assert any(abs(x) > 0.0 for x in d)          # the world is genuinely moving


# --------------------------------------------------------------------------- #
# The world model learns the physics from lived experience
# --------------------------------------------------------------------------- #
def test_agent_learns_action_conditioned_physics():
    agent = PhysicsAgent(seed=7, epsilon=0.3)
    for _ in range(400):
        agent.step()
    assert agent.status()["world_transitions"] >= 400

    s = agent.world.sense()
    p_right = agent.world_model.predict(s, "push_right")
    p_left = agent.world_model.predict(s, "push_left")
    p_drop = agent.world_model.predict(s, "drop")
    p_lift = agent.world_model.predict(s, "lift")
    # pushing right accelerates rightward, left leftward — the model learned the sign
    assert p_right.next_state[_VX0] > p_left.next_state[_VX0]
    # lifting rises, dropping falls — the model learned vertical dynamics
    assert p_lift.next_state[_VY0] > p_drop.next_state[_VY0]


def test_confidence_grows_with_experience():
    agent = PhysicsAgent(seed=9, epsilon=0.0)
    s0 = agent.world.sense()
    naive = agent.world_model.predict(s0, "push_right").confidence
    for _ in range(80):
        agent.step("push_right")
    trained = agent.world_model.predict(agent.world.sense(), "push_right").confidence
    assert trained > naive


# --------------------------------------------------------------------------- #
# Curiosity is honest and shared
# --------------------------------------------------------------------------- #
def test_learning_progress_prefers_the_unseen():
    agent = PhysicsAgent(seed=11, epsilon=0.0)
    for _ in range(60):
        agent.step("push_right")     # drill ONE action into competence
    s = agent.world.sense()
    known = agent.world_model.learning_progress(s, "push_right")
    unseen = agent.world_model.learning_progress(s, "lift")
    assert 0.0 <= known <= 1.0 and 0.0 <= unseen <= 1.0
    assert unseen >= known           # an un-modelled action is at least as worth exploring


# --------------------------------------------------------------------------- #
# Orchestrator entry points
# --------------------------------------------------------------------------- #
def test_physics_tick_and_stream_feed_a_shared_model():
    from nyxara.mind.world_model import build_world_model
    shared = build_world_model("auto", distance_scale=6.0)
    agent = PhysicsAgent(seed=2)
    tr = physics_tick(agent, shared)
    assert tr.action in ACTIONS
    assert len(shared) >= 1           # the tick mirrored its transition into the shared model
    stream = physics_stream(agent, shared, steps=5)
    assert len(stream) == 5
    assert len(shared) >= 6


def test_stream_never_raises_and_is_bounded():
    agent = PhysicsAgent(seed=1)
    stream = agent.run(steps=10)
    assert 1 <= len(stream) <= 10
    assert all(len(t.next_state) == STATE_DIM for t in stream)


# --------------------------------------------------------------------------- #
# The optional forces (spring / central / drag) — real, and OFF by default
# --------------------------------------------------------------------------- #
def test_extra_forces_are_off_by_default():
    """A default world is byte-for-byte the old rigid-body world: no spring, central, or drag."""
    w = PhysicsWorld(seed=0)
    assert w.spring_k == 0.0 and w.central_mu == 0.0 and w.drag_k == 0.0
    assert w.spring_anchor is None and w.central_at is None


def test_spring_force_produces_oscillation():
    """A Hooke spring makes body 0 oscillate across its anchor (simple-harmonic motion)."""
    import math
    w = PhysicsWorld(seed=1, gravity=0.0, air_drag=1.0, arena=1e9, dt=0.01, substeps=1,
                     spring_k=4.0, spring_anchor=(100.0, 100.0))
    w.reset()
    w.bodies[0].x, w.bodies[0].y = 103.0, 100.0
    w.bodies[0].vx = w.bodies[0].vy = 0.0
    w.bodies[0].on_ground = False
    xs = []
    for _ in range(1200):
        w.step("noop")
        xs.append(w.bodies[0].x - 100.0)
    crossings = sum(1 for i in range(1, len(xs))
                    if xs[i - 1] > 0 >= xs[i] or xs[i - 1] < 0 <= xs[i])
    assert crossings >= 4
    # measured period ≈ 2π√(m/k) = 2π√(1/4) = π
    from nyxara.growth.law_discovery import _period_of
    period = _period_of(xs, 0.01)
    assert period is not None and abs(period - math.pi) < 0.4


def test_central_force_makes_a_bound_orbit():
    """A central inverse-square force with tangential velocity yields a bound (non-escaping) orbit."""
    import math
    w = PhysicsWorld(seed=2, gravity=0.0, air_drag=1.0, arena=1e9, dt=0.005, substeps=2,
                     central_mu=40.0, central_at=(100.0, 100.0))
    w.reset()
    w.bodies[0].x, w.bodies[0].y = 104.0, 100.0
    w.bodies[0].vx, w.bodies[0].vy = 0.0, 3.0
    w.bodies[0].on_ground = False
    radii = []
    for _ in range(2000):
        w.step("noop")
        radii.append(math.hypot(w.bodies[0].x - 100.0, w.bodies[0].y - 100.0))
    assert max(radii) < 50.0 and min(radii) > 0.2      # stays in an annulus — it orbits, never escapes


def test_quadratic_drag_gives_terminal_velocity():
    """Under gravity + quadratic drag a falling body approaches a finite terminal speed √(mg/k)."""
    import math
    w = PhysicsWorld(seed=3, gravity=20.0, air_drag=1.0, arena=1e12, dt=0.002, substeps=1, drag_k=0.5)
    w.reset()
    w.bodies[0].mass = 1.0
    w.bodies[0].y = 1e9
    w.bodies[0].vx = w.bodies[0].vy = 0.0
    w.bodies[0].on_ground = False
    for _ in range(8000):
        w.step("noop")
    v_term = abs(w.bodies[0].vy)
    assert abs(v_term - math.sqrt(1.0 * 20.0 / 0.5)) < 0.5    # √(mg/k) = √40 ≈ 6.32
