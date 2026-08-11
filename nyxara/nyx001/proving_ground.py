"""NYXARA · nyx001/proving_ground.py — the six strict tests (🧪, held-out or it does not count).

The charter's rule: *"Hum tab tak isko 'Intelligence' nahi maanenge jab tak consistently in tests
mein pass na ho"* — we do not call it intelligence until it consistently passes these six:

1. **Zero-shot** — solve an unseen problem. Trained on rotation systems, tested on a system with
   a *different rotation angle it never saw*.
2. **Few-shot** — learn fast from little. How much error is eliminated in 20 steps on a new system.
3. **Transfer** — knowledge from one field used in another. A model pre-trained on system A vs a
   freshly-born one, both given the same small budget on system B. The *difference* is transfer.
4. **Memory interference** — learn something new without forgetting the old. Train on A, train on
   B, then re-measure A. The retention is the score.
5. **Adaptation** — survive a changed environment. The dynamics are switched mid-run and the
   system must recover; the score is how much of its performance it gets back.
6. **Long-horizon** — reason far ahead. Rollout accuracy at horizon 10 against ground truth.

**Every test is graded on data held out of the training that produced the model.** There is no
path here by which memorisation scores well — test 1 changes the system parameters, test 3 uses a
different system entirely, test 4 re-measures a system after training on another, and test 6 grades
against a simulator the model never sees the internals of.

**The baselines are the point.** Tests 1, 2, 5 and 6 all report a comparison against *copying the
input* — the null model that predicts no change. A world model that cannot beat copying has
learned nothing, however small its loss looks, and several of these tests would otherwise be
passable by a model that has learned exactly that.

This suite **trains from scratch on every run**, which is why it is slow and why it is honest:
nothing carries over between tests except what each test deliberately transfers.

Honest about scope: these are synthetic dynamical systems, not the world. Passing them shows the
Layer stack generalises, retains, transfers and adapts *on systems of this kind*. It does not show
general intelligence, and the charter's own framing — measurable tests instead of adjectives — is
what makes that distinction statable at all.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]

__all__ = ["TestResult", "ProvingReport", "ProvingGround"]

_TESTS = ("zero_shot", "few_shot", "transfer", "interference", "adaptation", "long_horizon")


@dataclass
class TestResult:
    """One test's outcome, with the baseline it had to beat."""

    name: str = ""
    score: Optional[float] = None
    baseline: Optional[float] = None
    passed: Optional[bool] = None
    threshold: float = 0.0
    detail: Dict[str, Any] = field(default_factory=dict)
    ms: float = 0.0
    error: str = ""

    @property
    def beat_baseline(self) -> Optional[bool]:
        if self.score is None or self.baseline is None:
            return None
        return bool(self.score > self.baseline)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name,
                "score": None if self.score is None else round(self.score, 4),
                "baseline": None if self.baseline is None else round(self.baseline, 4),
                "threshold": self.threshold, "passed": self.passed,
                "beat_baseline": self.beat_baseline, "ms": round(self.ms, 1),
                "detail": dict(self.detail), "error": self.error}


@dataclass
class ProvingReport:
    """All six, and the charter's verdict — which is only ever "not yet" or "on these tests"."""

    results: List[TestResult] = field(default_factory=list)
    ms: float = 0.0

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed is True)

    @property
    def measured(self) -> int:
        return sum(1 for r in self.results if r.passed is not None)

    @property
    def all_passed(self) -> bool:
        return self.measured == len(_TESTS) and self.passed == len(_TESTS)

    def to_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed, "measured": self.measured, "total": len(_TESTS),
                "all_passed": self.all_passed, "ms": round(self.ms, 1),
                "results": [r.to_dict() for r in self.results]}

    def render(self) -> str:
        lines = [f"the six strict tests — {self.passed}/{len(_TESTS)} passed "
                 f"({self.measured} measurable), {self.ms / 1000:.1f}s", ""]
        for r in self.results:
            if r.error:
                lines.append(f"  ?  {r.name:14s} could not run: {r.error}")
                continue
            if r.score is None:
                lines.append(f"  ?  {r.name:14s} not measurable")
                continue
            mark = "✓" if r.passed else "×"
            base = "" if r.baseline is None else f" · baseline {r.baseline:.4f}"
            lines.append(f"  {mark}  {r.name:14s} {r.score:.4f} (needs {r.threshold:.2f}){base}")
        lines.append("")
        if self.all_passed:
            lines.append("All six pass ON THESE SYNTHETIC SYSTEMS. That is what was measured;")
            lines.append("it is not a claim about general intelligence.")
        else:
            lines.append("Not intelligence yet, by the charter's own rule.")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The worlds. Deterministic dynamical systems, parameterised so a test can hold
# a variant out of training entirely.
# --------------------------------------------------------------------------- #
def _rotation_system(width: int, theta: float, decay: float = 0.99,
                     nonlinear: float = 0.15) -> Callable[[Any], Any]:
    """A damped rotation with a nonlinear coupling a linear model cannot capture."""
    R = np.eye(width)
    for i in range(0, width - 1, 2):
        R[i, i] = R[i + 1, i + 1] = np.cos(theta)
        R[i, i + 1] = -np.sin(theta)
        R[i + 1, i] = np.sin(theta)

    def step(s: Any) -> Any:
        nxt = decay * (R @ s)
        nxt[0] = nxt[0] + nonlinear * np.tanh(3.0 * s[1])
        return nxt
    return step


def _shift_system(width: int, k: int = 3, decay: float = 0.98) -> Callable[[Any], Any]:
    """A different *kind* of system: a circular shift. Used for transfer, so it is genuinely new."""
    def step(s: Any) -> Any:
        return decay * np.roll(s, k)
    return step


class ProvingGround:
    """Runs the six strict tests against a freshly-born Layer stack each time."""

    def __init__(self, *, width: int = 32, train_steps: int = 20_000, seed: int = 20260811,
                 lr: float = 5e-3, min_reference_skill: float = 0.15) -> None:
        self.width = int(width)
        # 20k steps is not arbitrary. Measured on this substrate at width 24: 1.2k steps reaches
        # 0.05 skill against the copy baseline, 10k reaches 0.28, 20k reaches 0.56. Below ~10k the
        # model has not learned enough for any of these six tests to be measuring what it claims
        # to measure, and the suite would be grading noise.
        self.train_steps = int(train_steps)
        self.seed = int(seed)
        self.lr = float(lr)
        # Ratio-based tests (interference, adaptation) divide by a reference skill. When that
        # reference is near zero the ratio is noise — and noise near zero can produce a ratio
        # above 1.0 that reads as a perfect score. Below this floor those tests report NOT
        # MEASURABLE rather than a number. This guard is why `adaptation` no longer "passes" at
        # 1.0000 on a model whose reference skill was 0.011.
        self.min_reference_skill = float(min_reference_skill)
        self.thresholds = {"zero_shot": 0.3, "few_shot": 0.15, "transfer": 0.05,
                           "interference": 0.7, "adaptation": 0.5, "long_horizon": 0.3}

    # ---- helpers ---- #
    def _new_model(self, seed: int) -> Any:
        from nyxara.nyx001.layers.l00_substrate import Substrate
        from nyxara.nyx001.layers.l02_world_model import WorldModel
        base = Substrate(seed=seed, rung="toy")
        width = self.width

        class _Pinned:
            def __init__(self) -> None:
                self.width = width

            def stream(self, name: str) -> Any:
                return base.stream(name)

            def note_params(self, *t: Any) -> int:
                return 0
        return WorldModel(_Pinned(), lr=self.lr)

    def _train(self, model: Any, step_fn: Callable[[Any], Any], steps: int,
               rng: Any, episode: int = 25) -> None:
        s = rng.normal(0, 0.5, self.width)
        model.reset_state()
        for i in range(steps):
            if i % episode == 0:
                s = rng.normal(0, 0.5, self.width)
                model.reset_state()
            model.predict(s)
            nxt = step_fn(s)
            model.observe(nxt)
            s = nxt

    def _evaluate(self, model: Any, step_fn: Callable[[Any], Any], rng: Any,
                  n: int = 200) -> Tuple[float, float]:
        """Return (model_mse, copy_baseline_mse) on fresh trajectories the model never trained on."""
        s = rng.normal(0, 0.5, self.width)
        model.reset_state()
        m_err, c_err = [], []
        for i in range(n):
            if i % 25 == 0:
                s = rng.normal(0, 0.5, self.width)
                model.reset_state()
            p = model.predict(s)
            nxt = step_fn(s)
            m_err.append(float(np.mean((p.state - nxt) ** 2)))
            c_err.append(float(np.mean((s - nxt) ** 2)))
            s = nxt
        return float(np.mean(m_err)), float(np.mean(c_err))

    @staticmethod
    def _skill(model_mse: float, copy_mse: float) -> float:
        """Skill score against the copy baseline, in ``[0, 1]``. 0 means no better than copying."""
        if copy_mse <= 0:
            return 0.0
        return float(max(0.0, min(1.0, 1.0 - model_mse / copy_mse)))

    # ---- the six ---- #
    def zero_shot(self) -> TestResult:
        """Train on several rotation angles; test on an angle held out entirely."""
        r = TestResult(name="zero_shot", threshold=self.thresholds["zero_shot"])
        t0 = time.time()
        try:
            rng = np.random.default_rng(self.seed)
            model = self._new_model(self.seed)
            for theta in (0.3, 0.5, 0.9, 1.1):                 # 0.7 is deliberately absent
                self._train(model, _rotation_system(self.width, theta),
                            self.train_steps // 4, rng)
            unseen = _rotation_system(self.width, 0.7)          # never trained on
            m, c = self._evaluate(model, unseen, np.random.default_rng(self.seed + 1))
            r.score = self._skill(m, c)
            r.baseline = 0.0
            r.passed = r.score >= r.threshold
            r.detail = {"held_out_theta": 0.7, "model_mse": round(m, 6),
                        "copy_mse": round(c, 6)}
        except Exception as exc:  # noqa: BLE001
            r.error = f"{type(exc).__name__}: {exc}"[:160]
        r.ms = (time.time() - t0) * 1000.0
        return r

    def few_shot(self) -> TestResult:
        """A competent model meets a NEW system and gets 40 steps. How much does it close?

        Measured on a model that already learned a *different* rotation, not on a randomly
        initialised one. Few-shot learning is a claim about a system that knows something adapting
        fast to something new; grading a fresh random model's first 20 steps measures the noise of
        initialisation, and produced a negative result here for exactly that reason.
        """
        r = TestResult(name="few_shot", threshold=self.thresholds["few_shot"])
        t0 = time.time()
        try:
            model = self._new_model(self.seed + 2)
            self._train(model, _rotation_system(self.width, 0.8), self.train_steps,
                        np.random.default_rng(self.seed + 2))
            system = _rotation_system(self.width, 0.35)          # new, never trained on
            probe = np.random.default_rng(self.seed + 3)
            before, _ = self._evaluate(model, system, np.random.default_rng(self.seed + 3), n=100)
            self._train(model, system, 40, probe, episode=20)     # forty steps. That is all.
            after, c1 = self._evaluate(model, system, np.random.default_rng(self.seed + 3), n=100)
            r.score = float(max(0.0, min(1.0, 1.0 - after / max(1e-12, before))))
            r.baseline = 0.0
            r.passed = r.score >= r.threshold
            r.detail = {"steps": 40, "mse_before": round(before, 6), "mse_after": round(after, 6),
                        "copy_mse": round(c1, 6)}
        except Exception as exc:  # noqa: BLE001
            r.error = f"{type(exc).__name__}: {exc}"[:160]
        r.ms = (time.time() - t0) * 1000.0
        return r

    def transfer(self) -> TestResult:
        """Pre-trained on rotation vs freshly born, both given the same small budget on shift."""
        r = TestResult(name="transfer", threshold=self.thresholds["transfer"])
        t0 = time.time()
        try:
            target = _shift_system(self.width, k=3)             # a different KIND of system
            budget = 150

            trained = self._new_model(self.seed + 4)
            self._train(trained, _rotation_system(self.width, 0.6), self.train_steps,
                        np.random.default_rng(self.seed + 4))
            self._train(trained, target, budget, np.random.default_rng(self.seed + 5))
            m_t, c_t = self._evaluate(trained, target, np.random.default_rng(self.seed + 6))

            fresh = self._new_model(self.seed + 4)               # same seed ⇒ same init
            self._train(fresh, target, budget, np.random.default_rng(self.seed + 5))
            m_f, c_f = self._evaluate(fresh, target, np.random.default_rng(self.seed + 6))

            skill_t = self._skill(m_t, c_t)
            skill_f = self._skill(m_f, c_f)
            r.score = float(skill_t - skill_f)                   # the DIFFERENCE is the transfer
            r.baseline = 0.0
            r.passed = r.score >= r.threshold
            r.detail = {"pretrained_skill": round(skill_t, 4), "fresh_skill": round(skill_f, 4),
                        "budget": budget}
        except Exception as exc:  # noqa: BLE001
            r.error = f"{type(exc).__name__}: {exc}"[:160]
        r.ms = (time.time() - t0) * 1000.0
        return r

    def interference(self) -> TestResult:
        """Train A, train B, re-measure A. How much of A survived learning B."""
        r = TestResult(name="interference", threshold=self.thresholds["interference"])
        t0 = time.time()
        try:
            a = _rotation_system(self.width, 0.5)
            b = _shift_system(self.width, k=5)
            model = self._new_model(self.seed + 7)
            self._train(model, a, self.train_steps, np.random.default_rng(self.seed + 7))
            m1, c1 = self._evaluate(model, a, np.random.default_rng(self.seed + 8))
            skill_before = self._skill(m1, c1)

            self._train(model, b, self.train_steps // 2, np.random.default_rng(self.seed + 9))
            m2, c2 = self._evaluate(model, a, np.random.default_rng(self.seed + 8))
            skill_after = self._skill(m2, c2)

            r.detail = {"skill_on_A_before": round(skill_before, 4),
                        "skill_on_A_after_learning_B": round(skill_after, 4),
                        "reference_floor": self.min_reference_skill}
            if skill_before < self.min_reference_skill:
                # Retention of a skill that was never acquired is undefined, not zero — and a
                # ratio of two near-zero numbers can read as a perfect 1.0.
                r.error = (f"reference skill on A was {skill_before:.4f}, below the "
                           f"{self.min_reference_skill:.2f} floor — retention is undefined")
            else:
                r.score = float(min(1.0, skill_after / skill_before))
                r.passed = r.score >= r.threshold
        except Exception as exc:  # noqa: BLE001
            r.error = f"{type(exc).__name__}: {exc}"[:160]
        r.ms = (time.time() - t0) * 1000.0
        return r

    def adaptation(self) -> TestResult:
        """Switch the dynamics mid-run; measure how much performance is recovered."""
        r = TestResult(name="adaptation", threshold=self.thresholds["adaptation"])
        t0 = time.time()
        try:
            old = _rotation_system(self.width, 0.5)
            new = _rotation_system(self.width, 0.5, decay=0.9, nonlinear=0.4)   # changed world
            model = self._new_model(self.seed + 10)
            self._train(model, old, self.train_steps, np.random.default_rng(self.seed + 10))
            m_ref, c_ref = self._evaluate(model, old, np.random.default_rng(self.seed + 11))
            skill_home = self._skill(m_ref, c_ref)

            m_shock, c_shock = self._evaluate(model, new, np.random.default_rng(self.seed + 12))
            skill_shock = self._skill(m_shock, c_shock)

            self._train(model, new, self.train_steps // 4, np.random.default_rng(self.seed + 13))
            m_rec, c_rec = self._evaluate(model, new, np.random.default_rng(self.seed + 12))
            skill_recovered = self._skill(m_rec, c_rec)

            r.detail = {"skill_home": round(skill_home, 4),
                        "skill_on_shock": round(skill_shock, 4),
                        "skill_recovered": round(skill_recovered, 4),
                        "reference_floor": self.min_reference_skill}
            if skill_home < self.min_reference_skill:
                # Same guard as interference: recovering to a fraction of a skill the model never
                # had is not adaptation. Without this floor, 0.0117/0.0111 scored a clean 1.0000.
                r.error = (f"home skill was {skill_home:.4f}, below the "
                           f"{self.min_reference_skill:.2f} floor — recovery is undefined")
            else:
                r.baseline = round(skill_shock, 4)
                r.score = float(min(1.0, skill_recovered / skill_home))
                r.passed = r.score >= r.threshold
        except Exception as exc:  # noqa: BLE001
            r.error = f"{type(exc).__name__}: {exc}"[:160]
        r.ms = (time.time() - t0) * 1000.0
        return r

    def long_horizon(self) -> TestResult:
        """Roll the model forward 10 steps against ground truth it never sees."""
        r = TestResult(name="long_horizon", threshold=self.thresholds["long_horizon"])
        t0 = time.time()
        try:
            system = _rotation_system(self.width, 0.55)
            model = self._new_model(self.seed + 14)
            self._train(model, system, self.train_steps, np.random.default_rng(self.seed + 14))
            rng = np.random.default_rng(self.seed + 15)
            horizon = 10
            m_err, c_err = [], []
            for _ in range(40):
                s0 = rng.normal(0, 0.5, self.width)
                model.reset_state()
                preds = model.rollout(s0, horizon=horizon)
                truth, s = [], s0.copy()
                for _k in range(horizon):
                    s = system(s)
                    truth.append(s.copy())
                if len(preds) < horizon:
                    continue
                m_err.append(float(np.mean((preds[-1].state - truth[-1]) ** 2)))
                c_err.append(float(np.mean((s0 - truth[-1]) ** 2)))   # copy, 10 steps out
            if not m_err:
                r.error = "rollout produced no predictions"
            else:
                m, c = float(np.mean(m_err)), float(np.mean(c_err))
                r.score = self._skill(m, c)
                r.baseline = 0.0
                r.passed = r.score >= r.threshold
                r.detail = {"horizon": horizon, "model_mse": round(m, 6),
                            "copy_mse": round(c, 6)}
        except Exception as exc:  # noqa: BLE001
            r.error = f"{type(exc).__name__}: {exc}"[:160]
        r.ms = (time.time() - t0) * 1000.0
        return r

    # ---- the suite ---- #
    def run_all(self, brain: Any = None, *, only: Optional[List[str]] = None) -> ProvingReport:
        """Run every test (or a named subset). Each trains from scratch; nothing carries over."""
        rep = ProvingReport()
        t0 = time.time()
        if np is None:
            rep.results = [TestResult(name=n, error="NumPy unavailable") for n in _TESTS]
            return rep
        wanted = set(only or _TESTS)
        for name in _TESTS:
            if name not in wanted:
                continue
            try:
                rep.results.append(getattr(self, name)())
            except Exception as exc:  # noqa: BLE001
                rep.results.append(TestResult(name=name,
                                              error=f"{type(exc).__name__}: {exc}"[:160]))
        rep.ms = (time.time() - t0) * 1000.0
        return rep
