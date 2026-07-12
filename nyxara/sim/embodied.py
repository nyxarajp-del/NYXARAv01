"""NYXARA · sim/embodied.py — a real, closed perception→action→consequence→learning loop (🧠🌍).

For a long time NYXARA's *grounding* was two disconnected islands:

* :mod:`nyxara.senses` (vision / audio / nlp / web / …) were **one-shot processors** — call
  ``analyze(path)``, get a result back, done. They produced :class:`~nyxara.senses.binding.Percept`
  s that coloured attention as passive data, but they never *acted*, never saw a consequence,
  never closed a loop.
* :mod:`nyxara.sim.real_environment` *was* a loop, but a narrow one: it sensed six raw filesystem
  numbers and acted with five toy file ops that wrote **random bytes** (perceivable as nothing).

This module fuses the two into **one embodied agent** that genuinely lives a sensorimotor cycle:

    perceive → decide → act → perceive the consequence → reward → learn

* **Perceive** — the agent reads the artifacts in its world *for real* with the senses
  (:class:`~nyxara.senses.nlp.NLP`, :class:`~nyxara.senses.vision.Vision`,
  :class:`~nyxara.senses.web.Web`), binds them through the
  :class:`~nyxara.senses.binding.Binder`, and scores **surprise / novelty** with the predictive
  coder (:class:`~nyxara.senses.predictive.PerceptualPredictor`). Its state is the physical
  filesystem dims **plus** these perceptual features — a richer, fully-numeric vector.
* **Decide** — actions are chosen, not random: the world model
  (:mod:`nyxara.mind.world_model`) predicts each action's reward + confidence, the
  :class:`~nyxara.identity.motivation.MotivationSystem` blends intrinsic drives (curiosity,
  competence, empowerment, novelty), and the agent picks for a *reason*.
* **Act** — it authors perceivable content (real text notes, generated images), reads/perceives
  its world, organizes it, or (oversight-gated) perceives the live web — every action genuinely
  changes real, re-observable state.
* **Reward** is intrinsic and honest: curiosity (resolved surprise), competence (the world model
  getting better at predicting the action), and homeostasis (no unbounded disk growth).
* **Learn** — every real ``(state, action, next_state, reward)`` transition trains the world
  model, so its dynamics reflect a *lived* world instead of a synthetic toy signal.

Safety is inherited from :class:`~nyxara.sim.real_environment.RealEnvironment`: scratch-confined,
reversible (journalled), capped, and oversight-gated. The live web is screened for injection and
bound as **low-trust, data-only** perception — never executed, never acted upon. Every fault is a
negative-reward observation, never a crash. Stdlib-first; heavy senses degrade gracefully.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from nyxara.senses.binding import Binder, Percept
from nyxara.senses.predictive import PerceptualPredictor, SensoryPredictor
from nyxara.identity.motivation import MotivationSystem, Option
from nyxara.sim.envmodel import EnvironmentModel
from nyxara.sim.real_environment import RealEnvironment

__all__ = ["Perception", "EmbodiedTransition", "EmbodiedAgent", "embodied_stream"]

# The embodied state: a fixed, fully-numeric vector blending the controllable *physical* world
# with a summary of what the agent has *perceived*. Comparable scales; learned dynamics are real.
#   0 file_count   1 total_kb   2 subdir_count
#   3 n_percepts   4 distinct_entities   5 mean_salience   6 novelty_rate   7 surprise
STATE_DIM = 8
(_S_FILES, _S_KB, _S_SUBDIRS, _S_PERCEPTS, _S_ENTITIES,
 _S_SALIENCE, _S_NOVELTY, _S_SURPRISE) = range(STATE_DIM)

# A tiny corpus of authored facts — proper nouns, dates and numbers an NLP pass extracts as real
# entities, so a perceived note carries genuine *meaning* (unlike random bytes). Public so the
# orchestrator/tests can inspect or extend it.
NOTE_CORPUS: Tuple[str, ...] = (
    "Ada Lovelace wrote the first published algorithm in 1843 for Babbage's Analytical Engine.",
    "Mount Everest rises 8849 metres above sea level on the border of Nepal and Tibet.",
    "Marie Curie won Nobel Prizes in both Physics in 1903 and Chemistry in 1911.",
    "The Pacific Ocean spans roughly 165 million square kilometres of the Earth's surface.",
    "Jupiter has 95 confirmed moons, the largest being Ganymede, Callisto, Io and Europa.",
    "Photosynthesis converts sunlight, water and carbon dioxide into glucose inside chloroplasts.",
    "The Amazon River in Brazil discharges more freshwater than the next seven rivers combined.",
    "Tokyo is the largest metropolitan area on Earth, home to about 37 million people.",
    "Alan Turing described a universal computing machine in his 1936 paper On Computable Numbers.",
    "The speed of light in a vacuum is exactly 299792458 metres per second.",
)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class Perception:
    """A bound snapshot of the agent's world: physical reading + perceptual summary."""
    state: Tuple[float, ...]                 # the STATE_DIM-vector fed to the world model
    n_percepts: int
    distinct_entities: int
    mean_salience: float
    novelty: float
    surprise: float

    def to_dict(self) -> Dict[str, Any]:
        return {"state": [round(x, 3) for x in self.state], "n_percepts": self.n_percepts,
                "distinct_entities": self.distinct_entities,
                "mean_salience": round(self.mean_salience, 3),
                "novelty": round(self.novelty, 3), "surprise": round(self.surprise, 4)}


@dataclass
class EmbodiedTransition:
    """One lived ``(state, action, next_state, reward)`` step, with its perceptual story."""
    state: Tuple[float, ...]
    action: str
    next_state: Tuple[float, ...]
    reward: float = 0.0
    surprise: float = 0.0
    novelty: bool = False
    competence_gain: float = 0.0
    percept: Optional[Dict[str, Any]] = None
    breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "reward": round(self.reward, 4),
                "surprise": round(self.surprise, 4), "novelty": self.novelty,
                "competence_gain": round(self.competence_gain, 4),
                "percept": self.percept, "breakdown": {k: round(v, 4)
                                                       for k, v in self.breakdown.items()}}


# --------------------------------------------------------------------------- #
# The embodied agent
# --------------------------------------------------------------------------- #
class EmbodiedAgent:
    """An agent that lives a real perceive→decide→act→consequence→learn loop in a safe world."""

    # full repertoire; :meth:`safe_actions` self-regulates which are sensible right now
    ACTIONS: Tuple[str, ...] = (
        "noop", "observe", "write_note", "write_image", "perceive",
        "create_file", "append", "delete_file", "organize", "web_perceive",
        "look", "watch", "listen",
    )

    # live real-world perception actions → the live modality they sense
    _LIVE_ACTIONS: Dict[str, str] = {"look": "camera", "watch": "screen", "listen": "mic"}

    def __init__(self, world_model: Any = None, *, env: Optional[RealEnvironment] = None,
                 binder: Optional[Binder] = None, motivation: Optional[MotivationSystem] = None,
                 seed: int = 0, soft_clutter: int = 24, frame_cap: int = 96,
                 web_enabled: bool = False, web_urls: Sequence[str] = (),
                 web: Any = None, gate: Optional[Callable[[], bool]] = None,
                 live_enabled: bool = False, live: Any = None,
                 explore_weight: float = 0.6, reward_weight: float = 1.0,
                 epsilon: float = 0.1, env_model: Optional[EnvironmentModel] = None,
                 planner: Optional[Callable[[Tuple[float, ...], List[str]],
                                            Optional[str]]] = None) -> None:
        self.env = env if env is not None else RealEnvironment(seed=seed)
        if world_model is None:                       # the engine — built here if not supplied
            from nyxara.mind.world_model import build_world_model
            world_model = build_world_model("auto", distance_scale=8.0)
        self.world_model = world_model
        self.binder = binder if binder is not None else Binder(window=120.0)
        self.motivation = motivation if motivation is not None else MotivationSystem()
        self.percept_predictor = PerceptualPredictor()
        self.state_predictor = SensoryPredictor()     # surprise over the state vector itself
        self._rng = random.Random(seed)
        self.soft_clutter = max(1, soft_clutter)
        self.frame_cap = max(8, frame_cap)
        self.explore_weight = explore_weight
        self.reward_weight = reward_weight
        self.epsilon = max(0.0, min(1.0, epsilon))
        # Long-horizon planner (#6): an optional ``(state, options) -> action`` that looks
        # *many* steps ahead (e.g. the Abyss timeline simulator) instead of the greedy
        # one-step world-model lookahead. None keeps the original myopic decide. When set, it
        # is consulted on exploit steps; ε-exploration and the greedy floor still stand.
        self.planner = planner

        # live-web perception (oversight + flag gated, data-only)
        self.web_enabled = bool(web_enabled)
        self.web_urls = list(web_urls)
        self._web = web
        self._gate = gate or (lambda: True)

        # live real-world perception (oversight + flag gated): a genuine camera/screen/mic
        # snapshot each step, sensed through the SAME senses as everything else. Sensitive, so
        # off unless explicitly enabled; built lazily and injectable for tests.
        self.live_enabled = bool(live_enabled)
        self._live = live

        # perceptual long-term bookkeeping
        self._perceived: set = set()                  # artifact paths already sensed
        self._distinct_entities: set = set()          # cumulative entity vocabulary
        self._novelty_ewma = 0.0
        self._last_state_surprise = 0.0
        self._note_seq = 0
        self._img_seq = 0
        self.steps_taken = 0

        # structured environment model: a *learned* template per action over the workspace,
        # fuelled by the SAME real before→after transitions the numeric world model sees. The
        # numeric model captures dynamics for control; this captures the symbolic effect of each
        # action (what attributes change, reversibility, confidence) for planning + dry-runs
        # (#25 World Model, #53 Simulation). Pure-stdlib; never fatal.
        self.env_model = env_model if env_model is not None else EnvironmentModel(
            "embodied workspace", "workspace")
        if self.env_model.get_entity("workspace") is None:
            self.env_model.add_entity("workspace", "filesystem",
                                      file_count=0.0, total_kb=0.0, subdir_count=0.0)

    # ------------------------------------------------------------------ #
    # Perception — build the rich numeric state
    # ------------------------------------------------------------------ #
    def perceive(self) -> Perception:
        """Sense the physical world and summarise the current perceptual frame into the state."""
        phys = self.env.sense()
        frame = self.binder.frame
        percepts = frame.percepts()
        n = len(percepts)
        mean_sal = (sum(p.salience for p in percepts) / n) if n else 0.0
        state = (
            phys[0],                                  # file_count
            phys[1],                                  # total_kb
            phys[5],                                  # subdir_count
            float(n),                                 # percepts in working memory
            float(len(self._distinct_entities)),      # entities understood so far
            mean_sal,                                 # current attention level
            self._novelty_ewma,                       # recent novelty rate
            self._last_state_surprise,                # how surprising the world has been
        )
        return Perception(state=state, n_percepts=n,
                          distinct_entities=len(self._distinct_entities),
                          mean_salience=mean_sal, novelty=self._novelty_ewma,
                          surprise=self._last_state_surprise)

    # ------------------------------------------------------------------ #
    # Action repertoire
    # ------------------------------------------------------------------ #
    def safe_actions(self) -> List[str]:
        """Only the actions that are sensible and safe right now (self-regulating)."""
        actions = ["noop", "observe"]
        n_files = sum(1 for p in self.env._created if os.path.isfile(p))
        artifacts = self._artifacts()
        if n_files < self.env.max_files:
            actions += ["write_note", "write_image", "create_file"]
        if artifacts:
            actions.append("perceive")
        if n_files > 0:
            actions += ["append", "delete_file"]
        if len(self.env._dirs) < self.env.max_files:
            actions.append("organize")
        if self.web_enabled and self.web_urls and self._gate():
            actions.append("web_perceive")
        if self.live_enabled and self._gate():
            avail = self._live_availability()
            for act, kind in self._LIVE_ACTIONS.items():
                if avail.get(kind):
                    actions.append(act)
        return actions

    def live_sensor(self) -> Any:
        """The live real-world sensor, built on first use (never fatal)."""
        if self._live is None:
            try:
                from nyxara.senses.live import LiveSensor
                self._live = LiveSensor()
            except Exception:  # noqa: BLE001 — live perception is a capability, never required
                self._live = False
        return self._live or None

    def _live_availability(self) -> Dict[str, bool]:
        sensor = self.live_sensor()
        if sensor is None:
            return {}
        try:
            return sensor.available()
        except Exception:  # noqa: BLE001
            return {}

    def _artifacts(self) -> List[str]:
        """Files in the scratch world that could be perceived, newest first."""
        try:
            files = [e.path for e in os.scandir(self.env.root)
                     if e.is_file(follow_symlinks=False)]
        except OSError:
            return []
        files.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0.0,
                   reverse=True)
        return files

    # ------------------------------------------------------------------ #
    # Decide — perception + world model + motivation choose the action
    # ------------------------------------------------------------------ #
    def decide(self, state: Tuple[float, ...]) -> str:
        options = self.safe_actions() or ["noop"]
        if self._rng.random() < self.epsilon:        # ε-exploration keeps the world fresh
            return self._rng.choice(options)
        # long-horizon lookahead first: roll each option many steps through the world model
        # and pick the one whose *distribution of futures* is best (risk-aware), not the one
        # that merely looks best for a single step. Falls through to greedy on any miss.
        if self.planner is not None:
            try:
                planned = self.planner(state, list(options))
                if planned in options:
                    return planned
            except Exception:  # noqa: BLE001 — planning is advisory; the greedy floor stands
                pass
        best, best_score = options[0], float("-inf")
        for a in options:
            pred = self.world_model.predict(state, a)
            conf = max(0.0, min(1.0, pred.confidence))
            explore = 1.0 - conf                      # curiosity toward the un-modelled
            opt = Option(name=a, signature=self._signature(a),
                         info_gain=explore, competence_gain=conf,
                         predicted_options=self._empowerment_hint(a, options),
                         owner_relevance=0.0)         # embodied curiosity is owner-neutral
            impulse = self.motivation.impulse(opt)
            # blend intrinsic motivation with the world model's predicted reward
            score = (impulse.total + self.reward_weight * pred.reward
                     + self.explore_weight * explore)
            if score > best_score:
                best, best_score = a, score
        return best

    def _signature(self, action: str) -> str:
        return action

    @staticmethod
    def _empowerment_hint(action: str, options: Sequence[str]) -> int:
        """A cheap empowerment proxy: do-something-new actions open up more reachable states."""
        n = len(options)
        if action in ("write_note", "write_image", "create_file"):
            return n + 3                              # authoring grows the perceivable world
        if action == "organize":
            return n + 1
        if action == "delete_file":
            return max(1, n - 2)
        if action in ("perceive", "web_perceive", "look", "watch", "listen"):
            return n
        return 1                                      # noop / observe change nothing

    # ------------------------------------------------------------------ #
    # Act — perform the action for real; return (reward, surprise, novelty, percept)
    # ------------------------------------------------------------------ #
    def _act(self, action: str) -> Tuple[float, float, bool, Optional[Dict[str, Any]],
                                         Dict[str, float]]:
        bd: Dict[str, float] = {}
        if action == "noop":
            return 0.0, 0.0, False, None, bd
        if action == "observe":
            self.env.sense()                          # a free, honest look at the machine
            bd["attend"] = 0.01
            return 0.01, 0.0, False, None, bd
        if action == "write_note":
            text = NOTE_CORPUS[self._note_seq % len(NOTE_CORPUS)]
            self._note_seq += 1
            path = self.env.write_text(f"note{self._note_seq}.txt", text)
            bd["author"] = 0.15 if path else -1.0
            return bd["author"], 0.0, False, None, bd
        if action == "write_image":
            r = self._write_image()
            bd["author"] = r
            return r, 0.0, False, None, bd
        if action == "perceive":
            return self._perceive_artifact()
        if action == "web_perceive":
            return self._perceive_web()
        if action in self._LIVE_ACTIONS:
            return self._perceive_live(self._LIVE_ACTIONS[action])
        # the original physical ops, delegated to the safe RealEnvironment
        if action in ("create_file", "append", "delete_file", "organize"):
            env_action = "make_dir" if action == "organize" else action
            before = self.env.sense()
            applied = False
            try:
                applied = self.env._apply(env_action)
            except Exception:  # noqa: BLE001 — a failed op is a negative-reward observation
                applied = False
            after = self.env.sense()
            r = self._physical_reward(action, before, after, applied)
            bd["physical"] = r
            return r, 0.0, False, None, bd
        return 0.0, 0.0, False, None, bd

    def _write_image(self) -> float:
        """Author a small, deterministic identicon PNG — a perceivable visual artifact."""
        try:
            from nyxara.senses.generate import identicon_rows, write_png
        except Exception:  # noqa: BLE001
            return 0.0
        self._img_seq += 1
        prompt = f"nyxara-embodied-{self._img_seq}"
        name = f"img{self._img_seq}.png"
        path = self.env._guard(os.path.join(self.env.root, name))
        if sum(1 for p in self.env._created if os.path.isfile(p)) >= self.env.max_files:
            return -1.0
        try:
            write_png(path, 32, 32, identicon_rows(prompt, size=32, grid=8))
            with open(path, "rb") as fh:
                data = fh.read()
            self.env._created.append(path)
            self.env._journal[path] = data            # journalled ⇒ reversible
            return 0.1
        except Exception:  # noqa: BLE001
            return -1.0

    def _perceive_artifact(self) -> Tuple[float, float, bool, Optional[Dict[str, Any]],
                                          Dict[str, float]]:
        """Read one artifact with the right sense, bind it, and reward resolved curiosity."""
        candidates = [p for p in self._artifacts() if p not in self._perceived] \
            or self._artifacts()
        if not candidates:
            return 0.0, 0.0, False, None, {"curiosity": 0.0}
        path = candidates[0]
        fresh = path not in self._perceived
        self._perceived.add(path)
        percept = self._sense_file(path)
        if percept is None:
            return -0.05, 0.0, False, None, {"curiosity": -0.05}
        bound, is_new = self.binder.perceive(percept)
        surprise = self.percept_predictor.observe(bound, boost_salience=True)
        for e in bound.entities:
            self._distinct_entities.add(e.lower())
        nov = bool(surprise.novelty) and fresh
        # curiosity reward: novelty + resolved surprise, habituating on re-perception
        base = 0.2 if fresh else 0.02
        curiosity = base + 0.4 * min(1.0, surprise.surprise) + (0.2 if nov else 0.0)
        self._note_novelty(1.0 if nov else 0.0)
        self.binder.frame  # keep ref; maybe reset below
        self._maybe_reset_frame()
        bd = {"curiosity": curiosity}
        return curiosity, surprise.surprise, nov, bound.to_dict(), bd

    def _perceive_web(self) -> Tuple[float, float, bool, Optional[Dict[str, Any]],
                                     Dict[str, float]]:
        """Fetch + sense one live web page as low-trust, data-only perception (gated, fail-soft)."""
        if not (self.web_enabled and self.web_urls and self._gate()):
            return 0.0, 0.0, False, None, {"curiosity": 0.0}
        url = self.web_urls[self.steps_taken % len(self.web_urls)]
        try:
            if self._web is None:
                from nyxara.senses.web import Web
                self._web = Web()
            page = self._web.page(url)
        except Exception:  # noqa: BLE001 — a network failure is a negative observation, not a crash
            return -0.05, 0.0, False, None, {"curiosity": -0.05}
        if not getattr(page, "text", ""):
            return -0.05, 0.0, False, None, {"curiosity": -0.05}
        percept = Percept.from_webpage(page)          # low trust; injection already screened
        bound, _ = self.binder.perceive(percept)
        surprise = self.percept_predictor.observe(bound, boost_salience=True)
        hostile = bool(getattr(page, "injection", None) and page.injection.suspicious)
        nov = bool(surprise.novelty)
        # genuine curiosity, but a hostile page is mildly aversive (learned caution — still data)
        curiosity = (0.0 if hostile else 0.2) + 0.4 * min(1.0, surprise.surprise)
        if hostile:
            curiosity -= 0.1
        self._note_novelty(1.0 if nov else 0.0)
        self._maybe_reset_frame()
        return curiosity, surprise.surprise, nov, bound.to_dict(), {"curiosity": curiosity}

    def _perceive_live(self, kind: str) -> Tuple[float, float, bool, Optional[Dict[str, Any]],
                                                 Dict[str, float]]:
        """Sense the *live* world (camera/screen/mic) through the real senses (gated, fail-soft).

        A genuine device snapshot is captured as bytes (PNG for camera/screen, WAV for mic),
        analysed by the SAME :class:`~nyxara.senses.vision.Vision` /
        :class:`~nyxara.senses.audio.Audio` used everywhere else, bound, and scored for
        surprise — so live intake drives curiosity, novelty and world-model learning identically
        to reading a file. A missing device is an honest negative observation, never a crash."""
        if not (self.live_enabled and self._gate()):
            return 0.0, 0.0, False, None, {"curiosity": 0.0}
        sensor = self.live_sensor()
        if sensor is None:
            return -0.05, 0.0, False, None, {"curiosity": -0.05}
        try:
            data, note = sensor.capture(kind)
        except Exception:  # noqa: BLE001 — a device fault is a negative observation, not a crash
            data, note = None, "capture error"
        if not data:
            # honest absence: NYXARA looked/listened and there was nothing there to perceive
            return -0.05, 0.0, False, None, {"curiosity": -0.05, "live_miss": 1.0}
        percept = self._sense_live_bytes(kind, data, note)
        if percept is None:
            return -0.05, 0.0, False, None, {"curiosity": -0.05}
        bound, _ = self.binder.perceive(percept)
        surprise = self.percept_predictor.observe(bound, boost_salience=True)
        for e in bound.entities:
            self._distinct_entities.add(e.lower())
        nov = bool(surprise.novelty)
        curiosity = 0.2 + 0.4 * min(1.0, surprise.surprise) + (0.2 if nov else 0.0)
        self._note_novelty(1.0 if nov else 0.0)
        self._maybe_reset_frame()
        return curiosity, surprise.surprise, nov, bound.to_dict(), {"curiosity": curiosity}

    def _sense_live_bytes(self, kind: str, data: bytes, note: str) -> Optional[Percept]:
        """Turn a live capture (PNG/WAV bytes) into a bound-ready percept via the right sense."""
        src = f"live:{kind}"
        try:
            if kind == "mic":
                from nyxara.senses.audio import Audio
                return Percept.from_audio(Audio().analyze_bytes(data, transcribe=True), source=src)
            from nyxara.senses.vision import Vision
            return Percept.from_image(Vision().analyze_bytes(data, ocr=True), source=src)
        except Exception:  # noqa: BLE001 — an undecodable capture is just low-information
            return None

    def _sense_file(self, path: str) -> Optional[Percept]:
        """Pick the right sense for an artifact and return a bound-ready percept (best-effort)."""
        name = os.path.basename(path)
        src = f"scratch:{name}"
        try:
            if name.endswith(".png") or name.endswith(".bmp"):
                from nyxara.senses.vision import Vision
                return Percept.from_image(Vision().analyze(path), source=src)
            if name.endswith(".wav"):
                from nyxara.senses.audio import Audio
                return Percept.from_audio(Audio().analyze(path), source=src)
            with open(path, "rb") as fh:
                raw = fh.read(4096)
            text = raw.decode("utf-8", errors="replace")
            from nyxara.senses.nlp import NLP
            return Percept.from_analysis(NLP().analyze(text), source=src)
        except Exception:  # noqa: BLE001 — an unreadable artifact is just low-information
            return None

    # ------------------------------------------------------------------ #
    # Reward helpers
    # ------------------------------------------------------------------ #
    def _physical_reward(self, action: str, before: Sequence[float], after: Sequence[float],
                         applied: bool) -> float:
        if not applied:
            return -1.0
        d_files = after[0] - before[0]
        d_kb = after[1] - before[1]
        d_dirs = after[5] - before[5]
        if action == "create_file":
            r = 0.1 if d_files > 0 else -1.0          # bytes, but less valued than an authored note
        elif action == "append":
            r = 0.1 if d_kb > 0 else -1.0
        elif action == "organize":
            r = 0.1 if d_dirs > 0 else -1.0
        elif action == "delete_file":
            # cleanup: costly when tidy, but *homeostatically good* when the world is cluttered
            r = (0.2 if before[0] > self.soft_clutter else -0.2) if d_files < 0 else -1.0
        else:
            r = 0.0
        return r

    def _homeostasis(self, before: Tuple[float, ...], after: Tuple[float, ...]) -> float:
        d_kb = after[_S_KB] - before[_S_KB]
        return -0.001 * max(0.0, d_kb / 1024.0)       # honest penalty on unbounded growth

    def _note_novelty(self, x: float) -> None:
        self._novelty_ewma = 0.85 * self._novelty_ewma + 0.15 * x

    def _maybe_reset_frame(self) -> None:
        """Keep working memory bounded; long-term vocabulary (entities) survives the refresh."""
        if len(self.binder.frame) > self.frame_cap:
            self.binder.new_frame()

    # ------------------------------------------------------------------ #
    # The loop: perceive → decide → act → perceive → reward → learn
    # ------------------------------------------------------------------ #
    # actions whose effect on the filesystem cannot be undone by the agent itself
    _IRREVERSIBLE_ACTIONS: FrozenSet[str] = frozenset({"delete_file"})

    def _observe_environment(self, action: str, before: "Perception",
                             after: "Perception") -> None:
        """Refine the structured workspace model from one real before→after transition.

        The first three state dims are the physical filesystem facts (file_count, total_kb,
        subdir_count); we present them as workspace attributes so the model learns each action's
        symbolic effect (the attribute diff, reversibility, and a confidence that grows with
        observations). Best-effort — a modelling slip never disturbs the lived loop."""
        try:
            b = {"file_count": round(before.state[0], 3),
                 "total_kb": round(before.state[1], 3),
                 "subdir_count": round(before.state[2], 3)}
            a = {"file_count": round(after.state[0], 3),
                 "total_kb": round(after.state[1], 3),
                 "subdir_count": round(after.state[2], 3)}
            self.env_model.observe(action, "workspace", b, a,
                                   reversible=action not in self._IRREVERSIBLE_ACTIONS)
        except Exception:  # noqa: BLE001 — structured modelling is a bonus, never load-bearing
            pass

    def env_coverage(self) -> float:
        """How well-modelled the workspace is now (mean learned-template confidence)."""
        try:
            return self.env_model.coverage()
        except Exception:  # noqa: BLE001
            return 0.0

    def step(self, action: Optional[str] = None) -> EmbodiedTransition:
        """Live one full sensorimotor cycle and learn from the genuine transition. Never raises."""
        before = self.perceive()
        chosen = action or self.decide(before.state)

        conf_before = self.world_model.predict(before.state, chosen).confidence
        try:
            r_act, surprise, novelty, percept, bd = self._act(chosen)
        except Exception:  # noqa: BLE001 — a failed action is a negative observation, never fatal
            r_act, surprise, novelty, percept, bd = -1.0, 0.0, False, None, {"error": -1.0}

        after = self.perceive()
        # global surprise: how much did the *consequence* violate the agent's expectation?
        sr = self.state_predictor.observe(after.state)
        self._last_state_surprise = sr.surprise

        homeo = self._homeostasis(before.state, after.state)
        reward = r_act + homeo
        bd["homeostasis"] = homeo

        # learn the real dynamics, then measure the competence the action just bought
        self.world_model.observe(before.state, chosen, after.state, reward)
        conf_after = self.world_model.predict(before.state, chosen).confidence
        competence_gain = max(0.0, conf_after - conf_before)
        if competence_gain > 0:
            reward += 0.5 * competence_gain
            bd["competence"] = 0.5 * competence_gain
        self.motivation.record_outcome(
            Option(name=chosen, signature=self._signature(chosen)),
            competence_gain=competence_gain, skill=chosen)

        # learn the structured (symbolic) effect of this action on the workspace, too
        self._observe_environment(chosen, before, after)

        self.steps_taken += 1
        return EmbodiedTransition(state=before.state, action=chosen, next_state=after.state,
                                  reward=reward, surprise=surprise or sr.surprise,
                                  novelty=novelty, competence_gain=competence_gain,
                                  percept=percept, breakdown=bd)

    def run(self, steps: int = 8) -> List[EmbodiedTransition]:
        """Drive a continuous burst of embodied steps (a lived trajectory, not a snapshot)."""
        out: List[EmbodiedTransition] = []
        for _ in range(max(1, int(steps))):
            try:
                out.append(self.step())
            except Exception:  # noqa: BLE001 — the stream is best-effort, never fatal
                break
        return out

    # ------------------------------------------------------------------ #
    # Introspection / teardown
    # ------------------------------------------------------------------ #
    def status(self) -> Dict[str, Any]:
        return {"steps": self.steps_taken, "percepts": len(self.binder.frame),
                "distinct_entities": len(self._distinct_entities),
                "perceived_artifacts": len(self._perceived),
                "novelty_rate": round(self._novelty_ewma, 3),
                "world_transitions": len(self.world_model),
                "env_model_coverage": round(self.env_coverage(), 3),
                "env_actions_learned": len(self.env_model.templates),
                "web_enabled": self.web_enabled, "live_enabled": self.live_enabled,
                "live_available": self._live_availability() if self.live_enabled else {}}

    def cleanup(self) -> None:
        self.env.cleanup()

    def __enter__(self) -> "EmbodiedAgent":
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()


def embodied_stream(agent: EmbodiedAgent, world_model: Any = None, *,
                    steps: int = 8) -> List[EmbodiedTransition]:
    """Orchestrator entry point: run an embodied burst.

    ``world_model`` is accepted for symmetry with ``sensorimotor_stream`` but the agent already
    owns (and learns into) its world model; if a different model is passed, transitions are
    mirrored into it too, so a shared planning model gets the same real fuel.
    """
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
    print("NYXARA embodied-agent self-test (perceive→decide→act→learn)")
    print("=" * 70)

    with EmbodiedAgent(seed=7, epsilon=0.15) as agent:
        # 1) seed the world with some authored content so there is something to perceive
        for _ in range(6):
            agent.step("write_note")
        # 2) live a free trajectory: the agent decides for itself
        traj = agent.run(steps=120)
        total_reward = sum(t.reward for t in traj)
        actions = {}
        for t in traj:
            actions[t.action] = actions.get(t.action, 0) + 1

        st = agent.status()
        print(f"\nsteps lived          : {agent.steps_taken}")
        print(f"action histogram     : {actions}")
        print(f"distinct entities    : {st['distinct_entities']}  (read from authored notes)")
        print(f"world transitions    : {st['world_transitions']}")
        print(f"trajectory reward    : {total_reward:+.2f}")

        # 3) the senses really fed the loop: perceiving notes extracted real entities
        assert st["distinct_entities"] > 0, "senses never grounded any meaning into the loop"
        assert st["perceived_artifacts"] > 0, "the agent never perceived its world"

        # 4) the world model learned real, action-conditioned dynamics from lived experience
        s = agent.perceive().state
        up = agent.world_model.predict(s, "write_note")
        down = agent.world_model.predict(s, "delete_file")
        print(f"\npredict write_note   : Δfiles={up.next_state[_S_FILES]-s[_S_FILES]:+.2f} "
              f"conf={up.confidence:.2f}")
        print(f"predict delete_file  : Δfiles={down.next_state[_S_FILES]-s[_S_FILES]:+.2f} "
              f"conf={down.confidence:.2f}")
        assert up.next_state[_S_FILES] > down.next_state[_S_FILES], \
            "world model failed to learn that authoring grows the world and deleting shrinks it"

        # 5) it is a genuine *closed* loop — perception is an action with a consequence
        assert "perceive" in actions, "perception never entered the action loop"

    # 6) LIVE real-world perception closes the loop too. On this box there may be no camera/mic,
    #    so we inject a stub sensor that returns REAL PNG/WAV bytes (crafted, not fabricated
    #    percepts) to prove the full capture→sense→bind→learn path end-to-end.
    from nyxara.senses.generate import identicon_rows, encode_png
    from nyxara.senses.live import pcm16_to_wav_bytes

    class _StubSensor:
        def available(self):
            return {"camera": True, "screen": False, "mic": True}

        def capture(self, kind, **kw):
            if kind == "camera":
                return encode_png(32, 32, identicon_rows("live-demo", size=32, grid=8)), "stub cam"
            if kind == "mic":
                return pcm16_to_wav_bytes(b"\x11\x00" * 8000, rate=16000), "stub mic"
            return None, "stub: unavailable"

    with EmbodiedAgent(seed=3, live_enabled=True, live=_StubSensor()) as live_agent:
        opts = live_agent.safe_actions()
        assert "look" in opts and "listen" in opts and "watch" not in opts, \
            f"live actions not gated on availability: {opts}"
        before_tx = len(live_agent.world_model)
        tr_look = live_agent.step("look")
        tr_listen = live_agent.step("listen")
        print(f"\nlive look reward     : {tr_look.reward:+.3f}  percept={bool(tr_look.percept)}")
        print(f"live listen reward   : {tr_listen.reward:+.3f}  percept={bool(tr_listen.percept)}")
        assert tr_look.percept and tr_listen.percept, "live capture never produced a percept"
        assert len(live_agent.binder.frame) > 0, "live percepts never entered working memory"
        assert len(live_agent.world_model) > before_tx, "live steps taught the world model nothing"
        print(f"live status          : live_enabled={live_agent.status()['live_enabled']} "
              f"available={live_agent.status()['live_available']}")

    print("\nALL SELF-TESTS PASSED ✓")
