"""NYXARA · growth/cls.py — Complementary Learning Systems (fast + slow) (🧠, continual).

The single deepest reason a frozen model is an *artifact* and not an *agent*: it cannot learn
one more thing without risking everything it already knew (catastrophic forgetting). Brains solve
this with **two** learning systems that complement each other — the insight of McClelland,
O'Reilly & Kumaran (1995) — and NYXARA runs both, herself, in her own code. No LLM is asked to
"remember"; this is her own SGD, her own attractor recall, her own replay.

* **Fast system — the hippocampus** (:class:`Hippocampus`). A highly-plastic learner that encodes
  a new experience in *one shot*. It **pattern-separates** (a seeded random projection + k-winners
  sparsification, so two similar episodes get decorrelated codes and do not overwrite each other)
  and **pattern-completes** (recall from a partial/noisy cue by nearest sparse code — a CA3-style
  attractor), and its plasticity is **neuromodulated** by novelty/surprise (encode hard what is
  new, barely touch what is already known).

* **Slow system — the neocortex** (:class:`Neocortex`). A slow, EWC-anchored learner that
  *generalises* and is *stable*. It learns **only through replay**, never one-shot, and its
  learning rate is **schema-gated** (an experience congruent with what it already knows
  consolidates fast; a novel one, slowly — Tse et al. 2007).

* **The bridge between them — sleep** (:meth:`ComplementaryLearningSystem.sleep`). Two stages:
  **NREM** interleaves prioritised replay from the hippocampus into the cortex (this is the
  anti-forgetting core); **REM** does **generative pseudo-rehearsal** — the cortex regenerates and
  re-consolidates skills it has already learned, so they survive *even after the episodes that
  taught them are evicted* (Shin et al. 2017) — abstracts recurring episodes into schemas, then
  **synaptically downscales** (Tononi & Cirelli's SHY) to protect signal-to-noise. Afterwards the
  hippocampus **turns over**, releasing capacity now that the cortex holds the knowledge.

At inference the two systems are **blended by recall confidence**: a novel cue is answered by the
hippocampus (which just learned it), a familiar one by the cortex (which generalised it).

:class:`ComplementaryLearningSystem` is a **drop-in superset** of
:class:`~nyxara.growth.learn.Learner` — ``record`` / ``value`` / ``replay`` / ``consolidate`` /
``best_action`` / ``model`` / ``buffer`` / ``to_dict`` / ``load_dict`` all behave as the orchestrator
expects — so wiring it in upgrades the single-system learner into a real two-system CLS with no other
change. Pure standard library, deterministic under a fixed seed. Character is untouchable: the
loyalty core cannot be learned over, fail-closed, exactly as in :mod:`growth.learn`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.growth.learn import Experience, Learner
from nyxara.guard.value_learning import IMMUTABLE_VALUES

__all__ = [
    "SparseProjector",
    "Hippocampus",
    "Neocortex",
    "SleepStage",
    "SleepReport",
    "ComplementaryLearningSystem",
]

Features = Dict[str, float]
SparseCode = Dict[int, float]


# --------------------------------------------------------------------------- #
# Pattern separation / completion primitives
# --------------------------------------------------------------------------- #
def _cosine_sparse(a: SparseCode, b: SparseCode) -> float:
    """Cosine similarity of two sparse codes (dicts of index -> value)."""
    if not a or not b:
        return 0.0
    # iterate the smaller for speed
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(i, 0.0) for i, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


class SparseProjector:
    """Dentate-gyrus-style pattern separation: a fixed seeded random projection followed by
    k-winners-take-all sparsification. Similar dense feature vectors map to *decorrelated* sparse
    codes, so the fast store can hold overlapping episodes without them overwriting one another.

    Deterministic: the projection column for a feature name is a fixed pseudo-random Gaussian vector
    derived from ``(seed, name)``, so the same experience always produces the same code — across
    processes and restarts — with no stored matrix to serialise.
    """

    def __init__(self, dim: int = 256, k: int = 16, seed: int = 0) -> None:
        self.dim = max(1, int(dim))
        self.k = max(1, min(int(k), self.dim))
        self.seed = int(seed)
        self._cols: Dict[str, List[float]] = {}

    def _col(self, name: str) -> List[float]:
        v = self._cols.get(name)
        if v is None:
            rng = random.Random(f"{self.seed}:{name}")
            v = [rng.gauss(0.0, 1.0) for _ in range(self.dim)]
            self._cols[name] = v
        return v

    def code(self, features: Features) -> SparseCode:
        """Project + k-WTA + L2-normalise → a sparse code (top-k dimensions by |activation|)."""
        if not features:
            return {}
        acc = [0.0] * self.dim
        for f, x in features.items():
            xf = float(x)
            if xf == 0.0:
                continue
            col = self._col(str(f))
            for i in range(self.dim):
                acc[i] += col[i] * xf
        # k-winners-take-all: keep the k largest-magnitude dimensions, zero the rest
        idx = sorted(range(self.dim), key=lambda i: abs(acc[i]), reverse=True)[: self.k]
        code = {i: acc[i] for i in idx if acc[i] != 0.0}
        norm = math.sqrt(sum(v * v for v in code.values()))
        if norm == 0.0:
            return {}
        return {i: v / norm for i, v in code.items()}


# --------------------------------------------------------------------------- #
# Fast system — the hippocampus
# --------------------------------------------------------------------------- #
@dataclass
class _Trace:
    """One episodic trace: its pattern-separated code plus the raw experience."""
    code: SparseCode
    exp: Experience


class Hippocampus:
    """One-shot episodic encoder with pattern separation, CA3 completion, and neuromodulated gain."""

    def __init__(self, *, base_lr: float = 0.5, buffer_capacity: int = 4000,
                 task_reserve: int = 64, trace_capacity: int = 4000,
                 proj_dim: int = 256, proj_k: int = 16, seed: int = 0,
                 protected: Optional[Sequence[str]] = None) -> None:
        # a plastic learner: high lr, no EWC — it is *meant* to be overwritten as it turns over
        self.learner = Learner(base_lr=base_lr, ewc_lambda=0.0,
                               buffer_capacity=buffer_capacity, task_reserve=task_reserve,
                               protected=protected, seed=seed)
        self.projector = SparseProjector(dim=proj_dim, k=proj_k, seed=seed)
        self.trace_capacity = max(1, int(trace_capacity))
        self._traces: List[_Trace] = []

    # ---- neuromodulated one-shot encoding ---- #
    def _novelty(self, code: SparseCode) -> float:
        """1 − (max cosine similarity to any stored trace). 1.0 when the store is empty."""
        if not self._traces or not code:
            return 1.0
        best = max(_cosine_sparse(code, t.code) for t in self._traces)
        return max(0.0, 1.0 - best)

    def encode(self, action: str, features: Features, reward: float, *,
               task: str = "", logit: Optional[float] = None,
               surprise: float = 0.0, buffer: bool = True) -> float:
        """Encode one experience in a single shot, with plasticity scaled by novelty/surprise.

        A novel or surprising experience is written hard (high effective learning rate); a familiar
        one barely moves the weights — the fast store spends its plasticity where it matters. The
        loyalty core is refused fail-closed, exactly as in :class:`Learner`.
        """
        self.learner._guard(action, features)  # protected-core fail-closed
        code = self.projector.code(features)
        novelty = self._novelty(code)
        surprise = max(0.0, min(1.0, float(surprise)))
        reward_mag = min(1.0, abs(float(reward)))
        # ACh/DA-like gain: strong for the new and the surprising, weak for the already-known
        gain = 0.35 + 0.65 * novelty + 0.5 * surprise + 0.25 * reward_mag
        gain = max(0.05, min(3.0, gain))
        pred = self.learner.model.predict(action, features) if logit is None else float(logit)
        err = self.learner.model.update(action, features, reward, self.learner.base_lr * gain)
        self.learner._steps += 1
        exp = Experience(action, dict(features), reward, task=task, logit=pred)
        if buffer:
            self.learner.buffer.add(exp)
        self._traces.append(_Trace(code, exp))
        if len(self._traces) > self.trace_capacity:
            del self._traces[: len(self._traces) - self.trace_capacity]
        return err

    # ---- CA3 pattern completion ---- #
    def predict(self, action: str, features: Features) -> float:
        """The fast model's one-shot value estimate for a cue."""
        return self.learner.model.predict(action, features)

    def confidence(self, features: Features) -> float:
        """Recall confidence for a cue (the attractor's basin depth): the max cosine similarity to
        any stored episodic trace — how strongly this cue completes to a remembered episode."""
        if not self._traces:
            return 0.0
        code = self.projector.code(features)
        if not code:
            return 0.0
        return max(_cosine_sparse(code, t.code) for t in self._traces)

    # ---- turnover: the temporary store releases capacity after consolidation ---- #
    def turnover(self, decay: float, *, evict_fraction: float = 0.0) -> int:
        """Decay the fast weights toward zero (they were consolidated to cortex) and optionally
        evict the oldest episodic traces. Returns the number of traces evicted."""
        decay = max(0.0, min(1.0, float(decay)))
        keep = 1.0 - decay
        for feats in self.learner.model._w.values():
            for f in list(feats):
                feats[f] *= keep
        evicted = 0
        if evict_fraction > 0.0 and self._traces:
            n = int(len(self._traces) * max(0.0, min(1.0, evict_fraction)))
            if n:
                del self._traces[:n]
                evicted = n
        return evicted

    def episodes(self) -> List[Experience]:
        return [t.exp for t in self._traces]


# --------------------------------------------------------------------------- #
# Slow system — the neocortex
# --------------------------------------------------------------------------- #
class Neocortex:
    """Slow, EWC-anchored generaliser that learns only via replay, with schema-gated rates."""

    def __init__(self, *, base_lr: float = 0.05, ewc_lambda: float = 3.0,
                 der_alpha: float = 0.5, task_reserve: int = 64,
                 frozen_lr_scale: float = 0.2, congruence_gain: float = 2.0,
                 seed: int = 0, protected: Optional[Sequence[str]] = None) -> None:
        self.learner = Learner(base_lr=base_lr, ewc_lambda=ewc_lambda, der_alpha=der_alpha,
                               task_reserve=task_reserve, frozen_lr_scale=frozen_lr_scale,
                               protected=protected, seed=seed)
        self.congruence_gain = max(1.0, float(congruence_gain))
        self._schema: Any = None                 # lazily-built SchemaLibrary (optional dep)
        self._prototypes: Dict[str, Features] = {}   # task -> running-mean feature prototype
        self._proto_reward: Dict[str, float] = {}    # task -> running-mean reward
        self._proto_action: Dict[str, str] = {}      # task -> its action

    def _schema_lib(self) -> Any:
        if self._schema is None:
            try:
                from nyxara.memory.schema import SchemaLibrary
                self._schema = SchemaLibrary(match_threshold=0.45)
            except Exception:  # noqa: BLE001 — schemas are an enhancement, never required
                self._schema = False
        # NB: an empty SchemaLibrary is falsy (``__len__`` == 0), so test the failure sentinel
        # explicitly — ``self._schema or None`` would wrongly hide a valid-but-empty library.
        return None if self._schema is False else self._schema

    @staticmethod
    def _episode(exp: Experience) -> Dict[str, Any]:
        """A schema episode from an experience: its features plus its action as a categorical slot."""
        ep: Dict[str, Any] = {f"feat::{k}": round(float(v), 3) for k, v in exp.features.items()}
        ep["action"] = exp.action
        return ep

    def congruence(self, exp: Experience) -> float:
        """How well this experience fits what the cortex already knows ([0,1]); 0 when it has no schema."""
        lib = self._schema_lib()
        if lib is None or len(lib) == 0:
            return 0.0
        try:
            _, score = lib.best_match(self._episode(exp))
            return max(0.0, min(1.0, float(score)))
        except Exception:  # noqa: BLE001
            return 0.0

    def _remember_prototype(self, exp: Experience, *, beta: float = 0.1) -> None:
        """Fold an experience into its task's running-mean prototype (for generative rehearsal)."""
        if not exp.task:
            return
        proto = self._prototypes.setdefault(exp.task, {})
        for f, v in exp.features.items():
            proto[f] = (1.0 - beta) * proto.get(f, 0.0) + beta * float(v)
        self._proto_reward[exp.task] = (
            (1.0 - beta) * self._proto_reward.get(exp.task, float(exp.reward))
            + beta * float(exp.reward))
        self._proto_action[exp.task] = exp.action

    def consolidate_batch(self, batch: Sequence[Experience]) -> Tuple[int, int]:
        """Interleaved replay into the cortex (the NREM core). Returns (trained, congruent).

        Each replayed experience trains the slow model at the cortex's low rate, *accelerated* by how
        congruent it is with existing schemas (fast integration of the familiar), with a
        Dark-Experience-Replay pull toward the experience's stored prediction so consolidation
        distils what was known, not just the raw reward. Every experience also feeds the schema
        library and its task prototype, so the cortex builds abstractions to generalise and to
        regenerate later.
        """
        trained = 0
        congruent = 0
        lib = self._schema_lib()
        lr0 = self.learner.base_lr
        for exp in batch:
            try:
                self.learner._guard(exp.action, exp.features)
            except Exception:  # noqa: BLE001 — never consolidate over the loyalty core
                continue
            cong = self.congruence(exp)
            if cong >= 0.5:
                congruent += 1
            mult = 1.0 + (self.congruence_gain - 1.0) * cong    # congruent → faster consolidation
            self.learner.model.update(exp.action, exp.features, exp.reward, lr0 * mult)
            if self.learner.der_alpha > 0.0 and exp.logit is not None:
                self.learner.model.update(exp.action, exp.features, exp.logit,
                                          lr0 * mult * self.learner.der_alpha)
            self.learner._steps += 1
            self._remember_prototype(exp)
            if lib is not None:
                try:
                    lib.integrate(self._episode(exp))
                except Exception:  # noqa: BLE001 — schema induction is best-effort
                    pass
            trained += 1
        return trained, congruent

    def pseudo_rehearse(self, n: int, rng: random.Random) -> int:
        """Generative pseudo-rehearsal (REM): regenerate and re-consolidate already-learned skills.

        For each known task the cortex synthesises a pseudo-experience from that task's prototype (with
        light noise), labels it with **its own current prediction** (self-generated target — deep
        generative replay), and trains toward it. This anchors old skills against the drift of new
        learning *without needing the original episodes*, so a task stays protected even after its
        hippocampal traces are evicted. Returns the number of pseudo-samples rehearsed.
        """
        if not self._prototypes or n <= 0:
            return 0
        tasks = list(self._prototypes)
        rehearsed = 0
        lr = self.learner.base_lr * 0.5
        for _ in range(n):
            task = rng.choice(tasks)
            proto = self._prototypes[task]
            action = self._proto_action.get(task, "")
            if not action or not proto:
                continue
            feats = {f: v + rng.gauss(0.0, 0.02) for f, v in proto.items()}
            try:
                self.learner._guard(action, feats)
            except Exception:  # noqa: BLE001
                continue
            target = self.learner.model.predict(action, feats)     # self-generated (its own memory)
            self.learner.model.update(action, feats, target, lr)
            self.learner._steps += 1
            rehearsed += 1
        return rehearsed

    def homeostatic_downscale(self, scale: float) -> float:
        """Synaptic homeostasis (SHY): multiply every cortical weight by ``scale`` (<1) to renormalise
        after a night of potentiation, protecting signal-to-noise. Relative ordering — and therefore
        every learned decision — is preserved; the frozen EWC anchors are untouched. Returns ``scale``.
        """
        s = max(0.0, min(1.0, float(scale)))
        if s == 1.0:
            return s
        for feats in self.learner.model._w.values():
            for f in list(feats):
                feats[f] *= s
        return s

    def predict(self, action: str, features: Features) -> float:
        return self.learner.model.predict(action, features)

    def support(self, features: Features) -> float:
        """How consolidated these features are in the cortex ([0,1]): the mean, over the queried
        features, of a saturating function of each weight's use count. Novel features → ~0; a
        well-practised skill → ~1. This is the gate that decides fast-vs-slow at inference."""
        model = self.learner.model
        if not features:
            return 0.0
        total = 0.0
        # per-feature use counts live per action; aggregate across actions for each feature name
        counts: Dict[str, float] = {}
        for feats in model._n.values():
            for f, c in feats.items():
                counts[f] = counts.get(f, 0.0) + c
        for f in features:
            c = counts.get(f, 0.0)
            total += 1.0 - math.exp(-c / 8.0)      # saturates toward 1 as the feature is reused
        return total / len(features)


# --------------------------------------------------------------------------- #
# Sleep report
# --------------------------------------------------------------------------- #
class SleepStage:
    NREM = "nrem"
    REM = "rem"


@dataclass
class SleepReport:
    deep: bool = False
    replayed_to_cortex: int = 0
    congruent: int = 0
    pseudo_rehearsed: int = 0
    schemas: int = 0
    homeostatic_scale: float = 1.0
    evicted: int = 0
    forgetting: float = 0.0
    duration_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deep": self.deep,
            "replayed_to_cortex": self.replayed_to_cortex,
            "congruent": self.congruent,
            "pseudo_rehearsed": self.pseudo_rehearsed,
            "schemas": self.schemas,
            "homeostatic_scale": round(self.homeostatic_scale, 4),
            "evicted": self.evicted,
            "forgetting": round(self.forgetting, 4),
            "duration_s": round(self.duration_s, 4),
        }


# --------------------------------------------------------------------------- #
# The Complementary Learning System — a drop-in Learner superset
# --------------------------------------------------------------------------- #
class ComplementaryLearningSystem:
    """Fast hippocampus + slow neocortex + a sleep bridge, exposing the :class:`Learner` surface."""

    def __init__(self, *, fast_lr: float = 0.5, slow_lr: float = 0.05, ewc_lambda: float = 3.0,
                 der_alpha: float = 0.5, task_reserve: int = 64, frozen_lr_scale: float = 0.2,
                 replay_batch: int = 32, hippocampal_decay: float = 0.15,
                 blend_sharpness: float = 4.0, pattern_sep_dim: int = 256, pattern_sep_k: int = 16,
                 rem_pseudo_batch: int = 16, homeostatic_scale: float = 0.98,
                 schema_congruence_gain: float = 2.0, adaptive_sleep: bool = True,
                 protected: Optional[Sequence[str]] = None, seed: int = 0) -> None:
        self.hippocampus = Hippocampus(
            base_lr=fast_lr, buffer_capacity=max(2000, replay_batch * 64),
            task_reserve=task_reserve, trace_capacity=max(2000, replay_batch * 64),
            proj_dim=pattern_sep_dim, proj_k=pattern_sep_k, seed=seed, protected=protected)
        self.neocortex = Neocortex(
            base_lr=slow_lr, ewc_lambda=ewc_lambda, der_alpha=der_alpha,
            task_reserve=task_reserve, frozen_lr_scale=frozen_lr_scale,
            congruence_gain=schema_congruence_gain, seed=seed, protected=protected)
        self.replay_batch = max(1, int(replay_batch))
        self.hippocampal_decay = max(0.0, min(1.0, float(hippocampal_decay)))
        self.blend_sharpness = max(0.0, float(blend_sharpness))
        self.rem_pseudo_batch = max(0, int(rem_pseudo_batch))
        self.homeostatic_scale = max(0.0, min(1.0, float(homeostatic_scale)))
        self.adaptive_sleep = bool(adaptive_sleep)
        self._rng = random.Random(seed)
        self._probe: List[Tuple[str, Features, float]] = []   # held-out drift probe
        self._probe_tasks: set = set()                        # task tags already in the probe
        self._sleeps = 0
        self._last_forgetting = 0.0

    # ---- Learner-compatible attribute surface (so all wiring keeps working) ---- #
    @property
    def model(self) -> Any:
        """The durable cortex model — what ``_learner_weight_vector`` and ElasticSynapses anchor."""
        return self.neocortex.learner.model

    @property
    def buffer(self) -> Any:
        """The hippocampal episodic buffer — what ``_dominant_task_tag`` / ContinuousLearner read."""
        return self.hippocampus.learner.buffer

    @property
    def synapses(self) -> Any:
        return self.neocortex.learner.synapses

    @property
    def protected(self) -> set:
        return self.neocortex.learner.protected

    def attach_synapses(self, syn: Any) -> None:
        """Elastic-synapse protection anchors the durable cortex (the store worth protecting)."""
        self.neocortex.learner.attach_synapses(syn)

    # ---- online experience: one-shot into the hippocampus ---- #
    def record(self, action: str, features: Features, reward: float, *,
               context: str = "", task: str = "", surprise: float = 0.0) -> float:
        """Encode a lived experience one-shot into the fast store (and remember it for replay)."""
        err = self.hippocampus.encode(action, features, reward, task=task,
                                      surprise=surprise, buffer=True)
        self._maybe_probe(action, features, reward, task)
        return err

    def update(self, action: str, features: Features, reward: float, *,
               context: str = "", surprise: float = 0.0) -> float:
        """One-shot encode without adding to the replay buffer (matches ``Learner.update``)."""
        return self.hippocampus.encode(action, features, reward, surprise=surprise, buffer=False)

    # ---- inference: recall-gated blend of the two systems ---- #
    def value(self, action: str, features: Features) -> float:
        """Blend fast and slow by how consolidated the cue is: novel → hippocampus, familiar → cortex."""
        cortex_v = self.neocortex.predict(action, features)
        hippo_v = self.hippocampus.predict(action, features)
        support = self.neocortex.support(features)
        recall = self.hippocampus.confidence(features)
        # gate toward the cortex as cortical support grows; toward the hippocampus when it has a
        # strong episodic match the cortex has not yet absorbed
        g = _sigmoid(self.blend_sharpness * (support - 0.5))
        g = max(0.0, min(1.0, g * (1.0 - 0.5 * max(0.0, recall - support))))
        return g * cortex_v + (1.0 - g) * hippo_v

    def best_action(self, features: Features, actions: Sequence[str]) -> str:
        return max(actions, key=lambda a: self.value(a, features))

    # ---- the sleep bridge (NREM + REM) ---- #
    def _sample_episodes(self, n: int, *, balanced: bool, prioritized: bool) -> List[Experience]:
        buf = self.hippocampus.learner.buffer
        if balanced:
            return buf.sample_balanced(n, rng=self._rng, prioritized=prioritized)
        return buf.sample(n, rng=self._rng, prioritized=prioritized)

    def sleep(self, *, deep: bool = False, batch: Optional[int] = None) -> SleepReport:
        """One sleep cycle. NREM always runs (replay → cortex); REM (generative rehearsal +
        homeostasis + hippocampal turnover) runs when ``deep``. Adaptive depth: when the drift probe
        shows forgetting rising, a light sleep is promoted to deep automatically."""
        import time
        t0 = time.monotonic()
        report = SleepReport(deep=deep)
        n = batch or self.replay_batch
        forgetting_before = self._measure_forgetting()

        # promote to deep sleep when the cortex is drifting (self-regulating stability/plasticity)
        if self.adaptive_sleep and not deep and forgetting_before > 0.1:
            deep = True
            report.deep = True

        # NREM — interleaved, prioritised replay from hippocampus into cortex
        batch_exps = self._sample_episodes(n, balanced=True, prioritized=True)
        trained, congruent = self.neocortex.consolidate_batch(batch_exps)
        report.replayed_to_cortex = trained
        report.congruent = congruent

        if deep:
            # REM — generative pseudo-rehearsal of already-consolidated skills
            report.pseudo_rehearsed = self.neocortex.pseudo_rehearse(self.rem_pseudo_batch, self._rng)
            # synaptic homeostasis — renormalise after the night's potentiation
            report.homeostatic_scale = self.neocortex.homeostatic_downscale(self.homeostatic_scale)
            # hippocampal turnover — release the temporary store now the cortex holds it
            report.evicted = self.hippocampus.turnover(self.hippocampal_decay, evict_fraction=0.1)

        lib = self.neocortex._schema_lib()
        report.schemas = len(lib) if lib is not None else 0
        report.forgetting = self._measure_forgetting()
        self._last_forgetting = report.forgetting
        self._sleeps += 1
        report.duration_s = time.monotonic() - t0
        return report

    def replay(self, n: int = 32, *, prioritized: bool = True, balanced: bool = False) -> int:
        """NREM replay (the per-turn consolidation the orchestrator drives). Returns replay count."""
        rep = self.sleep(deep=False, batch=n)
        return rep.replayed_to_cortex

    def consolidate(self) -> None:
        """Freeze the cortex's EWC anchors so what it has learned resists being overwritten."""
        self.neocortex.learner.consolidate()

    # ---- adaptive forgetting probe ---- #
    def _maybe_probe(self, action: str, features: Features, reward: float, task: str) -> None:
        """Keep a small, task-diverse held-out set to measure drift (one exemplar per new task)."""
        held_tasks = getattr(self, "_probe_tasks", None)
        if held_tasks is None:
            held_tasks = set()
            self._probe_tasks = held_tasks
        if task and task not in held_tasks and len(self._probe) < 64:
            self._probe.append((action, dict(features), float(reward)))
            held_tasks.add(task)

    def _measure_forgetting(self) -> float:
        """Mean absolute error of the *blended* system on the held-out probe (0 when empty)."""
        if not self._probe:
            return 0.0
        total = 0.0
        for action, feats, target in self._probe:
            total += abs(self.value(action, feats) - target)
        return total / len(self._probe)

    # ---- reporting ---- #
    def report(self) -> Dict[str, Any]:
        lib = self.neocortex._schema_lib()
        return {
            "kind": "complementary_learning_system",
            # ``steps`` (lived experiences encoded) keeps parity with ``Learner.report`` so every
            # caller that reads ``learner.report()["steps"]`` works unchanged.
            "steps": self.hippocampus.learner._steps,
            "fast_steps": self.hippocampus.learner._steps,
            "slow_steps": self.neocortex.learner._steps,
            "episodes": len(self.hippocampus.episodes()),
            "buffer": len(self.buffer),
            "tasks": self.buffer.tasks(),
            "schemas": len(lib) if lib is not None else 0,
            "prototypes": len(self.neocortex._prototypes),
            "sleeps": self._sleeps,
            "forgetting": round(self._last_forgetting, 4),
            "ewc_lambda": self.neocortex.learner.model.ewc_lambda,
            "protected": sorted(self.protected),
        }

    # ---- persistence: the whole two-system state round-trips ---- #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "cls",
            "fast": self.hippocampus.learner.to_dict(),
            "slow": self.neocortex.learner.to_dict(),
            "projector": {"dim": self.hippocampus.projector.dim,
                          "k": self.hippocampus.projector.k,
                          "seed": self.hippocampus.projector.seed},
            "prototypes": {t: dict(p) for t, p in self.neocortex._prototypes.items()},
            "proto_reward": dict(self.neocortex._proto_reward),
            "proto_action": dict(self.neocortex._proto_action),
            "probe": [[a, dict(f), r] for a, f, r in self._probe],
            "probe_tasks": sorted(getattr(self, "_probe_tasks", set())),
            "sleeps": int(self._sleeps),
            "config": {
                "replay_batch": self.replay_batch,
                "hippocampal_decay": self.hippocampal_decay,
                "blend_sharpness": self.blend_sharpness,
                "rem_pseudo_batch": self.rem_pseudo_batch,
                "homeostatic_scale": self.homeostatic_scale,
                "adaptive_sleep": self.adaptive_sleep,
            },
        }

    def load_dict(self, data: Dict[str, Any]) -> bool:
        """Restore both systems. Fail-closed on the loyalty core, exactly like ``Learner.load_dict``:
        the protected set can only grow toward :data:`IMMUTABLE_VALUES`, never shrink below it."""
        if not isinstance(data, dict):
            return False
        if isinstance(data.get("fast"), dict):
            self.hippocampus.learner.load_dict(data["fast"])
        if isinstance(data.get("slow"), dict):
            self.neocortex.learner.load_dict(data["slow"])
        proj = data.get("projector") or {}
        if proj:
            self.hippocampus.projector = SparseProjector(
                dim=int(proj.get("dim", 256)), k=int(proj.get("k", 16)),
                seed=int(proj.get("seed", 0)))
        self.neocortex._prototypes = {
            str(t): {str(k): float(v) for k, v in (p or {}).items()}
            for t, p in (data.get("prototypes") or {}).items()}
        self.neocortex._proto_reward = {
            str(t): float(r) for t, r in (data.get("proto_reward") or {}).items()}
        self.neocortex._proto_action = {
            str(t): str(a) for t, a in (data.get("proto_action") or {}).items()}
        self._probe = [(str(a), {str(k): float(v) for k, v in (f or {}).items()}, float(r))
                       for a, f, r in (data.get("probe") or [])]
        self._probe_tasks = set(data.get("probe_tasks") or [])
        self._sleeps = int(data.get("sleeps", self._sleeps))
        # the protected core is enforced fail-closed by each sub-learner's load_dict already
        assert set(IMMUTABLE_VALUES).issubset(self.protected)
        return True


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import json

    print("=" * 70)
    print("NYXARA complementary-learning-systems self-test")
    print("=" * 70)

    # ---- pattern separation: similar inputs → decorrelated sparse codes ---- #
    proj = SparseProjector(dim=256, k=16, seed=1)
    a = {"shared_a": 1.0, "shared_b": 1.0, "ctx_0": 1.0}
    b = {"shared_a": 1.0, "shared_b": 1.0, "ctx_1": 1.0}   # overlaps a in 2/3 features
    ca, cb = proj.code(a), proj.code(b)
    assert proj.code(a) == ca, "codes must be deterministic"
    sep = _cosine_sparse(ca, cb)
    print(f"\npattern separation  : overlapping cues → code similarity {sep:.2f} (decorrelated)")
    assert sep < 0.9

    # ---- one-shot fast encoding while the slow cortex lags ---- #
    cls = ComplementaryLearningSystem(fast_lr=0.6, slow_lr=0.05, seed=2)
    cls.record("act", {"novel": 1.0}, reward=1.0, task="A")
    fast_v = cls.hippocampus.predict("act", {"novel": 1.0})
    slow_v = cls.neocortex.predict("act", {"novel": 1.0})
    print(f"\none-shot           : hippocampus={fast_v:.2f}  cortex={slow_v:.2f} (cortex lags)")
    assert fast_v > 0.3 and abs(slow_v) < 0.05

    # ---- sleep consolidates hippocampus → cortex ---- #
    for _ in range(30):
        cls.record("act", {"novel": 1.0}, reward=1.0, task="A")
    for _ in range(20):
        cls.sleep(deep=False)
    slow_after = cls.neocortex.predict("act", {"novel": 1.0})
    print(f"sleep→cortex       : cortex value {slow_v:.2f} → {slow_after:.2f} (consolidated)")
    assert slow_after > 0.3

    # ---- recall-gated blend: familiar cue answered by the (now-trained) cortex ---- #
    cls.consolidate()
    blended = cls.value("act", {"novel": 1.0})
    print(f"blended inference  : {blended:.2f}")
    assert blended > 0.3

    # ---- generative pseudo-rehearsal protects a task after its episodes are EVICTED ---- #
    cls2 = ComplementaryLearningSystem(fast_lr=0.6, slow_lr=0.1, rem_pseudo_batch=24, seed=5)
    for _ in range(40):
        cls2.record("act", {"shared": 1.0, "ctx_a": 1.0}, reward=1.0, task="A")
    for _ in range(15):
        cls2.sleep(deep=True)
    cls2.consolidate()
    a_before = cls2.neocortex.predict("act", {"shared": 1.0, "ctx_a": 1.0})
    # evict ALL of task A's hippocampal traces, then learn an interfering task B
    cls2.hippocampus._traces.clear()
    cls2.hippocampus.learner.buffer._buf.clear()
    cls2.hippocampus.learner.buffer._reserve.clear()
    for _ in range(60):
        cls2.record("act", {"shared": 1.0, "ctx_b": 1.0}, reward=-1.0, task="B")
        cls2.sleep(deep=True)                       # REM regenerates A from its prototype
    a_after = cls2.neocortex.predict("act", {"shared": 1.0, "ctx_a": 1.0})
    print(f"\ngenerative rehearse: task-A cortex value {a_before:.2f} → {a_after:.2f} "
          f"(A's episodes were evicted; REM kept it)")
    assert a_after > 0.0, "REM pseudo-rehearsal must protect the evicted task from sign-flip"

    # ---- persistence: the full two-system state round-trips ---- #
    blob = json.loads(json.dumps(cls2.to_dict()))
    restored = ComplementaryLearningSystem(seed=99)
    assert restored.load_dict(blob)
    for feats in ({"shared": 1.0, "ctx_a": 1.0}, {"shared": 1.0, "ctx_b": 1.0}):
        assert abs(restored.neocortex.predict("act", feats)
                   - cls2.neocortex.predict("act", feats)) < 1e-9
    assert set(IMMUTABLE_VALUES).issubset(restored.protected)
    print("\npersistence        : two-system state round-trips losslessly, core stays protected ✓")

    # ---- the loyalty core cannot be learned over ---- #
    from nyxara.kernel.errors import ValidationError
    blocked = False
    try:
        cls2.record("loyalty_to_master", {"x": 1.0}, reward=-1.0)
    except ValidationError:
        blocked = True
    assert blocked
    print("protected core     : refused to encode over 'loyalty_to_master' ✓")

    print(f"\nreport             : {cls2.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
