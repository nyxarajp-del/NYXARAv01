"""NYXARA · sim/physics_world.py — intuitive-physics sensorimotor grounding (🌍🍎, stdlib-only).

NYXARA's world model (:mod:`nyxara.mind.world_model`) is a genuinely real learned-dynamics
engine, and :mod:`nyxara.sim.embodied` already gives it one *closed* sensorimotor loop — but that
loop is grounded in the **filesystem** (file counts, bytes, directories). The dynamics it learns
there are real, yet they are bookkeeping dynamics, not the physics of a lived world.

A powerful mind's world model is strong because it is built from **interaction, not text** — the
way a child learns physics by *dropping things and touching them*: let go of a cup and it falls,
push a ball and it rolls then stops, knock one marble into another and the motion passes across.
This module gives NYXARA exactly that: a real, deterministic **rigid-body micro-physics world**
it lives inside, acting with its body and predicting the consequences.

    perceive → decide → act (push / lift / drop / poke) → perceive the consequence → learn

* **The world** (:class:`PhysicsWorld`) is honest Newtonian mechanics: bodies with position,
  velocity, mass and radius under **gravity, friction and restitution**, integrated with
  semi-implicit Euler, colliding with the ground, the walls, and **each other** (momentum-
  conserving elastic collisions). Nothing here is a scripted animation — motion *emerges* from
  the forces, so the only way to predict it is to actually learn the physics.
* **The body** — a controllable "hand/ball" (body 0) the agent shoves with real impulses, and a
  passive object (body 1) it can *touch* by colliding into it. Every :meth:`~PhysicsWorld.step`
  applies a real force and returns a genuine ``(state, action, next_state, reward)`` transition,
  exactly the record :class:`nyxara.mind.world_model.Transition` consumes — the same interface
  :mod:`nyxara.sim.real_environment` uses, so it feeds the **same** shared world model with no
  new plumbing.
* **The drive** (:class:`PhysicsAgent`) is intrinsic curiosity: it favours the actions whose
  consequences it *cannot yet predict* (:meth:`~nyxara.mind.world_model.WorldModel.learning_progress`),
  blends the :class:`~nyxara.identity.motivation.MotivationSystem`'s drives just like the embodied
  agent, and is rewarded by the **competence** each interaction buys (how much better the world
  model now predicts it). It explores physics for its own sake — a curious child, not a task-doer.

Honesty, stated plainly: a server has no robot, so this body is a **high-fidelity deterministic
simulation**. It is embodied in the sense that matters — a *closed* loop where NYXARA's own
actions produce real cause→effect consequences it must predict and learn from — not text-derived
and not a degenerate synthetic signal. It is the honest ceiling of embodiment in a sandbox, and
it directly realises "learning physics by dropping and touching." Pure stdlib, fully in-memory,
zero filesystem/network side effects, and — like every core faculty — it never raises into the
cognitive cycle: a failed action is a negative-reward observation, never a crash. No LLM is
anywhere near this loop; the physics and the learning are entirely NYXARA's own numeric code.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.identity.motivation import MotivationSystem, Option
from nyxara.senses.predictive import SensoryPredictor

__all__ = [
    "Body", "PhysicsWorld", "PhysicsTransition", "PhysicsAgent",
    "physics_tick", "physics_stream",
]

# --------------------------------------------------------------------------- #
# The fully-numeric state vector fed to the world model. Two bodies (the agent-controlled
# hand/ball and one passive object), each with position + velocity, plus three contact flags.
# Units are chosen so positions (0..ARENA) and velocities live on a comparable numeric scale, so
# no single dimension drowns the others in the world model's distance metric (same discipline as
# real_environment.State):
#   0 x0  1 y0  2 vx0  3 vy0   |   4 x1  5 y1  6 vx1  7 vy1   |   8 on_ground0  9 on_ground1  10 touch
# --------------------------------------------------------------------------- #
STATE_DIM = 11
(_X0, _Y0, _VX0, _VY0, _X1, _Y1, _VX1, _VY1,
 _G0, _G1, _TOUCH) = range(STATE_DIM)

State = Tuple[float, ...]

# The motor repertoire — real impulses the agent applies to its own body (body 0). Every one is
# physically safe (in-memory) and self-explanatory: shove sideways, lift up, drop down, or poke
# hard toward the object to touch it. "noop" still advances physics (gravity keeps acting), which
# is itself a lesson: doing nothing is not the same as nothing happening.
ACTIONS: Tuple[str, ...] = (
    "noop", "push_left", "push_right", "lift", "drop", "poke",
)


@dataclass
class Body:
    """One rigid circular body in the world — real state, real mass, real size."""
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    mass: float = 1.0
    radius: float = 0.5
    on_ground: bool = False


@dataclass
class PhysicsTransition:
    """One lived physics step: ``(state, action, next_state, reward)`` with its learning story."""
    state: State
    action: str
    next_state: State
    reward: float = 0.0
    surprise: float = 0.0
    novelty: bool = False
    competence_gain: float = 0.0
    breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "reward": round(self.reward, 4),
                "surprise": round(self.surprise, 4), "novelty": self.novelty,
                "competence_gain": round(self.competence_gain, 4),
                "breakdown": {k: round(v, 4) for k, v in self.breakdown.items()}}


# --------------------------------------------------------------------------- #
# The world
# --------------------------------------------------------------------------- #
class PhysicsWorld:
    """A real, deterministic rigid-body micro-world the agent learns physics inside.

    Mirrors the :class:`nyxara.sim.real_environment.RealEnvironment` surface —
    ``sense`` / ``step`` / ``dynamics`` / ``safe_actions`` — so it feeds the world model through
    exactly the same path. Fully in-memory and seedable, so every run is reproducible.
    """

    ACTIONS = ACTIONS

    def __init__(self, *, arena: float = 10.0, gravity: float = 9.8, dt: float = 0.04,
                 substeps: int = 3, ground_friction: float = 0.82, air_drag: float = 0.999,
                 restitution: float = 0.6, push: float = 3.0, lift: float = 5.0,
                 drop: float = 3.0, poke: float = 6.0, seed: int = 0) -> None:
        self.arena = float(arena)
        self.g = float(gravity)
        self.dt = float(dt)
        self.substeps = max(1, int(substeps))
        self.ground_friction = min(1.0, max(0.0, ground_friction))
        self.air_drag = min(1.0, max(0.0, air_drag))
        self.restitution = min(1.0, max(0.0, restitution))
        self.push = float(push)
        self.lift = float(lift)
        self.drop = float(drop)
        self.poke = float(poke)
        self._rng = random.Random(seed)
        # a rolling window of recent states — turns discrete snapshots into a CONTINUOUS
        # sensorimotor stream the world model (and curiosity) read dynamics from.
        self._history: deque = deque(maxlen=64)
        self.reset()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def reset(self) -> State:
        """Place the two bodies in a deterministic, interesting starting configuration."""
        r = 0.5
        # body 0 (the "hand/ball" the agent controls) starts lifted, so gravity has something to
        # teach immediately; body 1 (the passive object) rests on the ground to its right.
        self.bodies: List[Body] = [
            Body(x=self.arena * 0.35, y=self.arena * 0.6, mass=1.0, radius=r),
            Body(x=self.arena * 0.65, y=r, mass=1.5, radius=r, on_ground=True),
        ]
        self._history.clear()
        s = self.sense()
        self._history.append(s)
        return s

    # ------------------------------------------------------------------ #
    # Sensing — the real, numeric state
    # ------------------------------------------------------------------ #
    def sense(self) -> State:
        """Read the fixed-dim numeric state from the true positions/velocities/contacts."""
        b0, b1 = self.bodies[0], self.bodies[1]
        dx, dy = b0.x - b1.x, b0.y - b1.y
        dist = math.hypot(dx, dy)
        touching = 1.0 if dist <= (b0.radius + b1.radius) + 1e-6 else 0.0
        return (
            b0.x, b0.y, b0.vx, b0.vy,
            b1.x, b1.y, b1.vx, b1.vy,
            1.0 if b0.on_ground else 0.0,
            1.0 if b1.on_ground else 0.0,
            touching,
        )

    # ------------------------------------------------------------------ #
    # Acting
    # ------------------------------------------------------------------ #
    def safe_actions(self) -> List[str]:
        """Every motor action is always safe here (pure in-memory simulation)."""
        return list(self.ACTIONS)

    def _apply_control(self, action: str) -> None:
        """Inject the action's real impulse into the agent-controlled body (body 0)."""
        b0 = self.bodies[0]
        if action == "push_left":
            b0.vx -= self.push
        elif action == "push_right":
            b0.vx += self.push
        elif action == "lift":
            b0.vy += self.lift
            b0.on_ground = False
        elif action == "drop":
            b0.vy -= self.drop
            b0.on_ground = False
        elif action == "poke":
            # a hard shove toward the object — the way to *touch* it and pass motion across
            direction = 1.0 if self.bodies[1].x >= b0.x else -1.0
            b0.vx += direction * self.poke
        # "noop" and anything unknown: apply nothing, but physics still advances below.

    def step(self, action: str) -> "Transition":
        """Apply the action's force, integrate real physics, return a genuine transition.

        Never raises: a malformed action injects no force (a no-op observation), and the physics
        integration is total. The reward here is the *environment-level* intrinsic signal
        (interaction + gentle homeostasis); the :class:`PhysicsAgent` adds curiosity/competence.
        """
        before = self.sense()
        if not self._history:
            self._history.append(before)
        try:
            self._apply_control(action)
            for _ in range(self.substeps):
                self._integrate(self.dt)
        except Exception:  # noqa: BLE001 — an integration slip is a null observation, not a crash
            pass
        after = self.sense()
        self._history.append(after)
        reward = self._reward(before, action, after)
        from nyxara.mind.world_model import Transition
        return Transition(state=before, action=action, next_state=after, reward=reward)

    def _integrate(self, dt: float) -> None:
        """One semi-implicit-Euler physics substep: gravity, drag, friction, motion, collisions."""
        for b in self.bodies:
            b.vy -= self.g * dt                      # gravity
            b.vx *= self.air_drag                     # air drag
            b.vy *= self.air_drag
            if b.on_ground:                           # ground friction opposes horizontal motion
                b.vx *= self.ground_friction
            b.x += b.vx * dt
            b.y += b.vy * dt
            self._resolve_bounds(b)
        self._resolve_body_collision(self.bodies[0], self.bodies[1])

    def _resolve_bounds(self, b: Body) -> None:
        """Keep a body inside the arena; bounce off ground/ceiling/walls with restitution."""
        lo, hi = b.radius, self.arena - b.radius
        # ground / ceiling
        if b.y <= lo:
            b.y = lo
            if b.vy < 0.0:
                b.vy = -b.vy * self.restitution
            if abs(b.vy) < 0.15:                      # settled — rest on the ground
                b.vy = 0.0
                b.on_ground = True
            else:
                b.on_ground = False
        elif b.y >= hi:
            b.y = hi
            if b.vy > 0.0:
                b.vy = -b.vy * self.restitution
            b.on_ground = False
        else:
            b.on_ground = False
        # walls
        if b.x <= lo:
            b.x = lo
            if b.vx < 0.0:
                b.vx = -b.vx * self.restitution
        elif b.x >= hi:
            b.x = hi
            if b.vx > 0.0:
                b.vx = -b.vx * self.restitution

    def _resolve_body_collision(self, a: Body, b: Body) -> None:
        """Momentum-conserving collision between two circles along their contact normal."""
        dx, dy = b.x - a.x, b.y - a.y
        dist = math.hypot(dx, dy)
        overlap = (a.radius + b.radius) - dist
        if overlap <= 0.0 or dist <= 1e-9:
            return
        nx, ny = dx / dist, dy / dist                 # unit contact normal a→b
        # positional correction (split by inverse mass) so they stop overlapping
        inv_a, inv_b = 1.0 / a.mass, 1.0 / b.mass
        corr = overlap / (inv_a + inv_b)
        a.x -= nx * corr * inv_a
        a.y -= ny * corr * inv_a
        b.x += nx * corr * inv_b
        b.y += ny * corr * inv_b
        # relative velocity along the normal
        rvx, rvy = b.vx - a.vx, b.vy - a.vy
        rel = rvx * nx + rvy * ny
        if rel >= 0.0:                                # already separating — no impulse
            return
        e = self.restitution
        j = -(1.0 + e) * rel / (inv_a + inv_b)        # scalar impulse magnitude
        jx, jy = j * nx, j * ny
        a.vx -= jx * inv_a
        a.vy -= jy * inv_a
        b.vx += jx * inv_b
        b.vy += jy * inv_b

    def _reward(self, before: State, action: str, after: State) -> float:
        """Honest environment-level reward: interaction is good, wild energy is gently penalised.

        There is no external task — the point is to *understand* physics — so the signal is thin
        and intrinsic. Touching the object (a genuine causal event to learn from) is rewarded; a
        tiny cost on kinetic energy keeps the body from flinging itself into the walls forever.
        """
        touched = after[_TOUCH] > 0.5 and before[_TOUCH] <= 0.5
        r = 0.2 if touched else 0.0
        kinetic = (after[_VX0] ** 2 + after[_VY0] ** 2 + after[_VX1] ** 2 + after[_VY1] ** 2)
        r -= 0.002 * kinetic
        return r

    def dynamics(self) -> State:
        """Per-step rate-of-change of each state dim over the rolling window (a live velocity)."""
        if len(self._history) < 2:
            return (0.0,) * STATE_DIM
        first, last = self._history[0], self._history[-1]
        n = len(self._history) - 1
        return tuple((last[i] - first[i]) / n for i in range(STATE_DIM))


# --------------------------------------------------------------------------- #
# The agent: perceive → decide → act → consequence → learn
# --------------------------------------------------------------------------- #
class PhysicsAgent:
    """A curiosity-driven agent that grounds the world model in lived intuitive physics."""

    def __init__(self, world_model: Any = None, *, world: Optional[PhysicsWorld] = None,
                 motivation: Optional[MotivationSystem] = None, seed: int = 0,
                 epsilon: float = 0.15, explore_weight: float = 0.8,
                 reward_weight: float = 1.0) -> None:
        self.world = world if world is not None else PhysicsWorld(seed=seed)
        if world_model is None:                       # the engine — built here if not supplied
            from nyxara.mind.world_model import build_world_model
            world_model = build_world_model("auto", distance_scale=6.0)
        self.world_model = world_model
        self.motivation = motivation if motivation is not None else MotivationSystem()
        self.state_predictor = SensoryPredictor()     # honest surprise over the state vector
        self._rng = random.Random(seed)
        self.epsilon = max(0.0, min(1.0, epsilon))
        self.explore_weight = explore_weight
        self.reward_weight = reward_weight
        self.steps_taken = 0
        self._novelty_ewma = 0.0

    # ------------------------------------------------------------------ #
    # Decide — curiosity + world model + motivation choose the action
    # ------------------------------------------------------------------ #
    def decide(self, state: State) -> str:
        options = self.world.safe_actions() or ["noop"]
        if self._rng.random() < self.epsilon:         # ε-exploration keeps the physics fresh
            return self._rng.choice(options)
        best, best_score = options[0], float("-inf")
        for a in options:
            pred = self.world_model.predict(state, a)
            # ONE shared, honest curiosity signal (also used by the filesystem embodied loop):
            # how much taking this action here would teach the model.
            info_gain = self.world_model.learning_progress(state, a)
            conf = max(0.0, min(1.0, pred.confidence))
            opt = Option(name=a, signature=a, info_gain=info_gain, competence_gain=conf,
                         predicted_options=self._empowerment_hint(a),
                         owner_relevance=0.0)          # embodied curiosity is owner-neutral
            impulse = self.motivation.impulse(opt)
            score = (impulse.total + self.reward_weight * pred.reward
                     + self.explore_weight * info_gain)
            if score > best_score:
                best, best_score = a, score
        return best

    @staticmethod
    def _empowerment_hint(action: str) -> int:
        """A cheap empowerment proxy: forceful actions open up more reachable futures."""
        if action in ("poke", "push_left", "push_right"):
            return 6
        if action in ("lift", "drop"):
            return 4
        return 1                                       # noop changes little of its own accord

    # ------------------------------------------------------------------ #
    # The loop
    # ------------------------------------------------------------------ #
    def step(self, action: Optional[str] = None) -> PhysicsTransition:
        """Live one full physics cycle and learn from the genuine transition. Never raises."""
        before = self.world.sense()
        chosen = action or self.decide(before)

        conf_before = self.world_model.predict(before, chosen).confidence
        try:
            tr = self.world.step(chosen)
        except Exception:  # noqa: BLE001 — a failed action is a negative observation, never fatal
            after = self.world.sense()
            from nyxara.mind.world_model import Transition
            tr = Transition(state=before, action=chosen, next_state=after, reward=-0.5)
        after = tr.next_state

        # global surprise: how far did the consequence violate the agent's expectation?
        sr = self.state_predictor.observe(after)
        surprise = sr.surprise
        novelty = bool(sr.novelty)
        self._novelty_ewma = 0.85 * self._novelty_ewma + 0.15 * (1.0 if novelty else 0.0)

        # learn the real dynamics, then measure the competence this interaction bought
        self.world_model.observe(before, chosen, after, tr.reward)
        conf_after = self.world_model.predict(before, chosen).confidence
        competence_gain = max(0.0, conf_after - conf_before)

        bd: Dict[str, float] = {"env": tr.reward}
        reward = tr.reward
        if competence_gain > 0.0:
            reward += 0.5 * competence_gain           # rewarded for getting better at the world
            bd["competence"] = 0.5 * competence_gain
        # curiosity: resolved surprise is intrinsically rewarding (a lesson learned)
        curiosity = 0.1 * min(1.0, surprise)
        reward += curiosity
        bd["curiosity"] = curiosity

        self.motivation.record_outcome(
            Option(name=chosen, signature=chosen),
            competence_gain=competence_gain, skill=chosen)

        self.steps_taken += 1
        return PhysicsTransition(state=before, action=chosen, next_state=after, reward=reward,
                                 surprise=surprise, novelty=novelty,
                                 competence_gain=competence_gain, breakdown=bd)

    def run(self, steps: int = 8) -> List[PhysicsTransition]:
        """Drive a continuous burst of physics steps (a lived trajectory, not a snapshot)."""
        out: List[PhysicsTransition] = []
        for _ in range(max(1, int(steps))):
            try:
                out.append(self.step())
            except Exception:  # noqa: BLE001 — the stream is best-effort, never fatal
                break
        return out

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def status(self) -> Dict[str, Any]:
        return {"steps": self.steps_taken,
                "world_transitions": len(self.world_model),
                "novelty_rate": round(self._novelty_ewma, 3),
                "actions_learned": len(self.world_model.actions())}


# --------------------------------------------------------------------------- #
# Orchestrator entry points — mirror sensorimotor_tick / sensorimotor_stream
# --------------------------------------------------------------------------- #
def physics_tick(agent: PhysicsAgent, world_model: Any = None) -> PhysicsTransition:
    """Live one physics step and feed the genuine transition to the shared model too.

    ``world_model`` is accepted for symmetry with :func:`physics_stream`; the agent already learns
    into its own model, but when a *different* (shared/planning) model is passed the transition is
    mirrored into it so planning gets the same real fuel.
    """
    tr = agent.step()
    if world_model is not None and world_model is not agent.world_model:
        try:
            world_model.observe(tr.state, tr.action, tr.next_state, reward=tr.reward)
        except Exception:  # noqa: BLE001 — mirroring is best-effort
            pass
    return tr


def physics_stream(agent: PhysicsAgent, world_model: Any = None, *,
                   steps: int = 6) -> List[PhysicsTransition]:
    """Drive a continuous burst of physics steps; mirror each into ``world_model`` if given."""
    stream = agent.run(steps=steps)
    if world_model is not None and world_model is not agent.world_model:
        for tr in stream:
            try:
                world_model.observe(tr.state, tr.action, tr.next_state, reward=tr.reward)
            except Exception:  # noqa: BLE001 — mirroring is best-effort
                pass
    return stream


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA intuitive-physics self-test (learning by dropping & touching)")
    print("=" * 70)

    # 1) raw physics is real: a released body falls under gravity
    w = PhysicsWorld(seed=1)
    y_start = w.sense()[_Y0]
    for _ in range(8):
        w.step("noop")
    y_after = w.sense()[_Y0]
    print(f"\ngravity        : y0 {y_start:.2f} -> {y_after:.2f} (fell {y_start - y_after:+.2f})")
    assert y_after < y_start, "gravity failed — a released body did not fall"

    # 2) friction halts a pushed body resting on the ground
    w2 = PhysicsWorld(seed=2)
    for _ in range(30):                                # let body 0 settle onto the floor
        w2.step("noop")
    w2.step("push_right")
    vx_pushed = w2.sense()[_VX0]
    for _ in range(20):
        w2.step("noop")
    vx_settled = w2.sense()[_VX0]
    print(f"friction       : vx0 {vx_pushed:+.2f} -> {vx_settled:+.2f} after coasting")
    assert abs(vx_settled) < abs(vx_pushed), "friction failed — a pushed body never slowed"

    # 3) momentum transfers: poking body 0 into body 1 sets body 1 moving
    w3 = PhysicsWorld(seed=3)
    for _ in range(30):
        w3.step("noop")
    b1_vx_before = w3.sense()[_VX1]
    moved = False
    for _ in range(40):
        w3.step("poke")
        if abs(w3.sense()[_VX1]) > abs(b1_vx_before) + 0.1:
            moved = True
            break
    print(f"collision      : object vx1 reached {w3.sense()[_VX1]:+.2f} after being poked")
    assert moved, "collision failed — poking one body never moved the other"

    # 4) the world model LEARNS action-conditioned physics from lived experience
    agent = PhysicsAgent(seed=7, epsilon=0.3)
    for _ in range(400):
        agent.step()
    st = agent.status()
    print(f"\nsteps lived    : {st['steps']}")
    print(f"transitions    : {st['world_transitions']}  over {st['actions_learned']} actions")

    s = agent.world.sense()
    p_right = agent.world_model.predict(s, "push_right")
    p_left = agent.world_model.predict(s, "push_left")
    p_drop = agent.world_model.predict(s, "drop")
    p_lift = agent.world_model.predict(s, "lift")
    print(f"predict push_right : Δvx0={p_right.next_state[_VX0]-s[_VX0]:+.2f} conf={p_right.confidence:.2f}")
    print(f"predict push_left  : Δvx0={p_left.next_state[_VX0]-s[_VX0]:+.2f} conf={p_left.confidence:.2f}")
    print(f"predict drop       : Δvy0={p_drop.next_state[_VY0]-s[_VY0]:+.2f} conf={p_drop.confidence:.2f}")
    print(f"predict lift       : Δvy0={p_lift.next_state[_VY0]-s[_VY0]:+.2f} conf={p_lift.confidence:.2f}")
    assert p_right.next_state[_VX0] > p_left.next_state[_VX0], \
        "world model failed to learn that pushing right/left moves the body opposite ways"
    assert p_lift.next_state[_VY0] > p_drop.next_state[_VY0], \
        "world model failed to learn that lifting rises and dropping falls"

    # 5) curiosity is honest: an unseen action is more worth exploring than a well-drilled one
    fresh = PhysicsAgent(seed=11, epsilon=0.0)
    for _ in range(60):
        fresh.step("push_right")                       # drill ONE action into competence
    s2 = fresh.world.sense()
    lp_known = fresh.world_model.learning_progress(s2, "push_right")
    lp_unknown = fresh.world_model.learning_progress(s2, "lift")
    print(f"\ncuriosity      : learning_progress(known)={lp_known:.2f} < (unseen)={lp_unknown:.2f}")
    assert lp_unknown >= lp_known, "curiosity failed — an unseen action was not more informative"

    print("\nALL SELF-TESTS PASSED ✓")
