"""NYXARA · eval/continual.py — the continual-learning benchmark (📏, forgetting measured).

"She never forgets what she learned" is a *claim*; this module makes it a **measurement**.
It builds a sequence of tasks that interfere by construction (they share features, so plain
gradient descent on task N drags the weights that task 1 depends on), trains a learner on
them strictly one after another — the continual-learning setting — and computes the standard
metrics of the field from the accuracy matrix ``R[i][j]`` (accuracy on task *j* after
finishing task *i*):

* **Average accuracy** ``ACC = mean_j R[T-1][j]`` — how good she is at *everything* at the end.
* **Forgetting** ``F_j = max_i R[i][j] − R[T-1][j]`` — how much of each earned skill was lost.
* **Backward transfer** ``BWT = mean_{j<T-1} (R[T-1][j] − R[j][j])`` — negative = forgetting.
* **Forward transfer** ``FWT = mean_{j>0} (R[j-1][j] − b_j)`` — did old learning help new tasks
  (``b_j`` is an untrained learner's accuracy on task *j*).

:func:`evaluate_continual` runs the whole battery twice — a **baseline** learner with no
protection, and a **protected** learner wired the way the orchestrator wires NYXARA herself
(elastic synapses attached to every update step, per-skill anchors consolidated at task
boundaries, balanced dark-experience replay, plasticity gating) — and reports both, so the
protection is *proven* against its own ablation, not asserted. Pure standard library,
deterministic under a fixed seed, fast enough for CI.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.growth.learn import Learner

__all__ = [
    "ContinualTask",
    "ContinualMetrics",
    "ContinualReport",
    "make_task_suite",
    "run_sequential",
    "matrix_metrics",
    "evaluate_continual",
    "evaluate_cls",
]

Features = Dict[str, float]
Sample = Tuple[Features, float]

_ACTION = "act"          # one shared action ⇒ every task competes for the same weights
_TARGET_SCALE = 2.0      # targets live in [-1, 1]; error is normalised by the full span


# --------------------------------------------------------------------------- #
# The task suite — sequential tasks that interfere by construction
# --------------------------------------------------------------------------- #
@dataclass
class ContinualTask:
    """One task in the sequence: named, with train and held-out test samples."""

    name: str
    train: List[Sample] = field(default_factory=list)
    test: List[Sample] = field(default_factory=list)


def make_task_suite(n_tasks: int = 4, n_train: int = 60, *, n_test: int = 12,
                    seed: int = 7) -> List[ContinualTask]:
    """Build ``n_tasks`` sequential regression tasks over sparse features.

    Every task uses the same ``shared_*`` features (that is the interference: later tasks
    drag them) plus its own ``ctx_i`` context feature, and asks for an alternating target
    (±1) — so a learner with no protection provably forgets task 0 while learning task 1.
    Deterministic for a given seed.
    """
    rng = random.Random(seed)
    tasks: List[ContinualTask] = []
    for i in range(max(1, n_tasks)):
        target = 1.0 if i % 2 == 0 else -1.0

        def sample(k: int, *, _i: int = i, _t: float = target) -> Sample:
            feats = {
                "shared_a": 1.0 + rng.uniform(-0.1, 0.1),
                "shared_b": 1.0 + rng.uniform(-0.1, 0.1),
                f"ctx_{_i}": 1.0 + rng.uniform(-0.1, 0.1),
            }
            return feats, _t

        tasks.append(ContinualTask(
            name=f"task-{i}",
            train=[sample(k) for k in range(max(1, n_train))],
            test=[sample(k) for k in range(max(1, n_test))],
        ))
    return tasks


def _accuracy(learner: Learner, task: ContinualTask) -> float:
    """Mean accuracy on the task's held-out test set: 1 − normalised absolute error."""
    if not task.test:
        return 0.0
    total = 0.0
    for feats, target in task.test:
        err = abs(learner.value(_ACTION, feats) - target) / _TARGET_SCALE
        total += 1.0 - min(1.0, err)
    return total / len(task.test)


# --------------------------------------------------------------------------- #
# Sequential training → the accuracy matrix R[i][j]
# --------------------------------------------------------------------------- #
def run_sequential(learner_factory: Callable[[], Learner],
                   tasks: Sequence[ContinualTask], *,
                   rehearse_every: int = 0, rehearse_n: int = 16,
                   balanced: bool = False,
                   on_task_end: Optional[Callable[[Learner, ContinualTask], None]] = None,
                   ) -> Tuple[List[List[float]], Learner]:
    """Train on the tasks strictly in sequence; return (accuracy matrix, final learner).

    ``R[i][j]`` is the accuracy on task ``j`` measured right after finishing task ``i`` —
    the raw material for every continual-learning metric. ``rehearse_every`` interleaves a
    replay pass every that-many training steps; ``on_task_end`` is the consolidation hook.
    """
    learner = learner_factory()
    matrix: List[List[float]] = []
    for task in tasks:
        for step, (feats, target) in enumerate(task.train):
            learner.record(_ACTION, feats, target, task=task.name)
            if rehearse_every and (step + 1) % rehearse_every == 0:
                learner.replay(rehearse_n, balanced=balanced)
        if on_task_end is not None:
            on_task_end(learner, task)
        matrix.append([_accuracy(learner, t) for t in tasks])
    return matrix, learner


# --------------------------------------------------------------------------- #
# The standard continual-learning metrics
# --------------------------------------------------------------------------- #
@dataclass
class ContinualMetrics:
    """ACC / forgetting / BWT / FWT computed from one accuracy matrix."""

    average_accuracy: float
    forgetting: float                      # mean over tasks (lower is better; ≥ 0)
    forgetting_per_task: List[float]
    bwt: float                             # backward transfer (negative = forgot)
    fwt: float                             # forward transfer vs an untrained baseline
    matrix: List[List[float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "average_accuracy": round(self.average_accuracy, 4),
            "forgetting": round(self.forgetting, 4),
            "forgetting_per_task": [round(f, 4) for f in self.forgetting_per_task],
            "bwt": round(self.bwt, 4),
            "fwt": round(self.fwt, 4),
            "matrix": [[round(v, 4) for v in row] for row in self.matrix],
        }


def matrix_metrics(matrix: Sequence[Sequence[float]],
                   untrained: Optional[Sequence[float]] = None) -> ContinualMetrics:
    """Compute the standard CL metrics from an accuracy matrix ``R[i][j]``."""
    T = len(matrix)
    final = list(matrix[-1]) if T else []
    n = len(final)
    acc = sum(final) / n if n else 0.0
    forgetting_per_task: List[float] = []
    for j in range(n):
        best = max(matrix[i][j] for i in range(T))
        forgetting_per_task.append(max(0.0, best - final[j]))
    forgetting = (sum(forgetting_per_task) / n) if n else 0.0
    older = [final[j] - matrix[j][j] for j in range(min(T, n) - 1)]
    bwt = sum(older) / len(older) if older else 0.0
    if untrained is not None and T > 1:
        fwd = [matrix[j - 1][j] - untrained[j] for j in range(1, min(T, n))]
        fwt = sum(fwd) / len(fwd) if fwd else 0.0
    else:
        fwt = 0.0
    return ContinualMetrics(average_accuracy=acc, forgetting=forgetting,
                            forgetting_per_task=forgetting_per_task, bwt=bwt, fwt=fwt,
                            matrix=[list(r) for r in matrix])


# --------------------------------------------------------------------------- #
# The head-to-head evaluation: protected NYXARA vs her own ablation
# --------------------------------------------------------------------------- #
@dataclass
class ContinualReport:
    """Baseline vs protected metrics on the same task suite, plus the deltas."""

    baseline: ContinualMetrics
    protected: ContinualMetrics
    n_tasks: int
    seed: int

    @property
    def forgetting_reduction(self) -> float:
        return self.baseline.forgetting - self.protected.forgetting

    @property
    def accuracy_gain(self) -> float:
        return self.protected.average_accuracy - self.baseline.average_accuracy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_tasks": self.n_tasks,
            "seed": self.seed,
            "baseline": self.baseline.to_dict(),
            "protected": self.protected.to_dict(),
            "forgetting_reduction": round(self.forgetting_reduction, 4),
            "accuracy_gain": round(self.accuracy_gain, 4),
        }


def _flatten_weights(learner: Learner) -> Dict[str, float]:
    """The learner's weights in the engine's ``action::feature`` naming scheme."""
    out: Dict[str, float] = {}
    for action, feats in learner.model._w.items():
        for feat, val in feats.items():
            out[f"{action}::{feat}"] = float(val)
    return out


def evaluate_continual(*, n_tasks: int = 4, n_train: int = 60,
                       seed: int = 7) -> ContinualReport:
    """Run the full battery: an unprotected learner vs the full protection stack.

    The protected learner is wired exactly the way the orchestrator wires NYXARA: an
    :class:`~nyxara.memory.elastic_synapses.ElasticSynapses` engine attached to every update
    step (per-step elastic pull, plasticity gating, SI+Fisher importance), per-skill anchors
    consolidated at each task boundary, and balanced DER replay across everything she has
    lived. Same tasks, same seed, same budget — the only difference is the protection.
    """
    from nyxara.memory.elastic_synapses import ElasticSynapses

    tasks = make_task_suite(n_tasks, n_train, seed=seed)
    untrained = [_accuracy(Learner(seed=seed), t) for t in tasks]

    # ---- baseline: plain online SGD, no protection of any kind ---- #
    base_matrix, _ = run_sequential(lambda: Learner(base_lr=0.1, seed=seed), tasks)

    # ---- protected: the full continual-learning stack ---- #
    def protected_factory() -> Learner:
        learner = Learner(base_lr=0.1, seed=seed, der_alpha=0.5, frozen_lr_scale=0.2,
                          task_reserve=32)
        syn = ElasticSynapses(ewc_lambda=4.0, per_skill_anchors=True,
                              freeze_threshold=0.6, max_tasks=32)
        learner.attach_synapses(syn)
        return learner

    def consolidate(learner: Learner, task: ContinualTask) -> None:
        learner.replay(32, balanced=True)
        syn = learner.synapses
        if syn is not None:
            syn.consolidate(_flatten_weights(learner), task=task.name)

    prot_matrix, _ = run_sequential(protected_factory, tasks,
                                    rehearse_every=10, rehearse_n=16, balanced=True,
                                    on_task_end=consolidate)

    return ContinualReport(
        baseline=matrix_metrics(base_matrix, untrained),
        protected=matrix_metrics(prot_matrix, untrained),
        n_tasks=len(tasks), seed=seed,
    )


# --------------------------------------------------------------------------- #
# The Complementary Learning Systems arm — fast + slow vs a single online learner
# --------------------------------------------------------------------------- #
def evaluate_cls(*, n_tasks: int = 4, n_train: int = 60, seed: int = 7,
                 evict: bool = False) -> ContinualReport:
    """Run the interfering suite through a real CLS (fast hippocampus + slow cortex + sleep) and
    compare its forgetting against a single unprotected online learner on the *same* tasks/seed.

    Each experience is encoded one-shot into the hippocampus; between and after tasks she **sleeps**
    (NREM replay into the cortex + REM generative pseudo-rehearsal + synaptic homeostasis). With
    ``evict=True`` the hippocampal store is wiped after every task, so old tasks can *only* be
    retained by REM's self-generated rehearsal — proof that the cortex holds the skill even once the
    episodes that taught it are gone. The only difference from the baseline is the architecture.
    """
    from nyxara.growth.cls import ComplementaryLearningSystem

    tasks = make_task_suite(n_tasks, n_train, seed=seed)
    untrained = [_accuracy(Learner(seed=seed), t) for t in tasks]

    # ---- baseline: plain online SGD, single system, no protection ---- #
    base_matrix, _ = run_sequential(lambda: Learner(base_lr=0.1, seed=seed), tasks)

    # ---- CLS: two systems bridged by sleep ---- #
    def cls_factory() -> Any:
        return ComplementaryLearningSystem(
            fast_lr=0.5, slow_lr=0.1, ewc_lambda=4.0, der_alpha=0.5, task_reserve=32,
            replay_batch=16, rem_pseudo_batch=24, schema_congruence_gain=2.0, seed=seed)

    def sleep_and_consolidate(learner: Any, task: ContinualTask) -> None:
        for _ in range(8):
            learner.sleep(deep=True)          # NREM replay + REM pseudo-rehearsal + homeostasis
        learner.consolidate()                 # freeze the cortex's EWC anchors for this skill
        if evict:                             # wipe the fast store — only the cortex may remember now
            learner.hippocampus._traces.clear()
            learner.hippocampus.learner.buffer._buf.clear()
            learner.hippocampus.learner.buffer._reserve.clear()

    cls_matrix, _ = run_sequential(cls_factory, tasks, rehearse_every=10, rehearse_n=16,
                                   balanced=True, on_task_end=sleep_and_consolidate)

    return ContinualReport(
        baseline=matrix_metrics(base_matrix, untrained),
        protected=matrix_metrics(cls_matrix, untrained),
        n_tasks=len(tasks), seed=seed,
    )


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import json

    print("=" * 70)
    print("NYXARA continual-learning benchmark (forgetting, measured)")
    print("=" * 70)

    report = evaluate_continual()

    def show(label: str, m: ContinualMetrics) -> None:
        print(f"\n{label}:")
        print(f"  average accuracy : {m.average_accuracy:.3f}")
        print(f"  forgetting       : {m.forgetting:.3f}  (per task: "
              + ", ".join(f"{f:.3f}" for f in m.forgetting_per_task) + ")")
        print(f"  BWT / FWT        : {m.bwt:+.3f} / {m.fwt:+.3f}")

    show("baseline (no protection)", report.baseline)
    show("protected (EWC+SI + per-skill anchors + balanced DER replay)", report.protected)

    print(f"\nforgetting reduction : {report.forgetting_reduction:+.3f}")
    print(f"accuracy gain        : {report.accuracy_gain:+.3f}")

    assert report.protected.forgetting < report.baseline.forgetting, \
        "protection must forget less than the ablation"
    assert report.protected.average_accuracy > report.baseline.average_accuracy, \
        "protection must end more capable than the ablation"
    json.dumps(report.to_dict())   # must be serialisable

    print("\nALL SELF-TESTS PASSED ✓ — protection beats its own ablation, measurably")

    # ---- Complementary Learning Systems: fast + slow, bridged by sleep ---- #
    print("\n" + "=" * 70)
    print("NYXARA CLS benchmark (fast hippocampus + slow cortex + sleep)")
    print("=" * 70)
    cls_report = evaluate_cls()
    show("baseline (single online learner)", cls_report.baseline)
    show("CLS (hippocampus + cortex + NREM/REM sleep)", cls_report.protected)
    print(f"\nforgetting reduction : {cls_report.forgetting_reduction:+.3f}")
    print(f"accuracy gain        : {cls_report.accuracy_gain:+.3f}")
    assert cls_report.protected.forgetting < cls_report.baseline.forgetting, \
        "the CLS must forget less than a single online learner"

    # ---- generative rehearsal: retention holds even when the fast store is wiped each task ---- #
    evicted = evaluate_cls(evict=True)
    print(f"\nevicted-episodes CLS : forgetting {evicted.protected.forgetting:.3f} "
          f"vs baseline {evicted.baseline.forgetting:.3f} "
          f"(reduction {evicted.forgetting_reduction:+.3f})")
    assert evicted.protected.forgetting < evicted.baseline.forgetting, \
        "REM pseudo-rehearsal must protect old tasks even after their episodes are evicted"
    json.dumps(cls_report.to_dict()); json.dumps(evicted.to_dict())

    print("\nALL CLS SELF-TESTS PASSED ✓ — two systems forget less than one, even under eviction")
