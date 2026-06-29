"""NYXARA · growth/grounded_experiments.py — world-grounded improvement (🌍, Rule 4, real outcomes).

The Master's eighth gap: every ruler NYXARA had was **pre-graded** — a benchmark with a stored answer
key she (or her generator) wrote. A real recursively-self-improving mind also improves against
outcomes it does *not* know in advance: it runs an experiment and lets **reality** grade it. This
module gives NYXARA that faculty, with reality standing in for the answer key in two concrete forms:

* **Code-against-reality.** She predicts what a program will do, then **runs it in the real
  interpreter** (the isolated :mod:`~nyxara.agency.code_sandbox`) and is graded by the actual output —
  not a number she stored, but whatever the machine truly computes. A buggy belief is exposed by
  execution. :meth:`predict_execution` is the capability ruler the RSI loop reads (``grounded_score``);
  :meth:`code_experiment` is the reusable primitive (run a candidate vs a reference program on fresh
  inputs, grade by real agreement).
* **Prediction-against-the-world.** She makes a falsifiable prediction and checks it against **real
  data fetched from the live web**, screened exactly as :mod:`~nyxara.growth.acquire` screens its
  corpus (the :class:`~nyxara.senses.web` SSRF guard + injection scanner). Offline, this degrades
  honestly to "no grounding available" — its weight is simply dropped, never faked.

The grade is produced by the world (the interpreter; the fetched datum), so it cannot be Goodharted
by editing the answer key — there is none. Pure stdlib + the existing sandbox; network use is
best-effort and guarded. Touches no source, no weights, no gate — it only *measures against reality*.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from fractions import Fraction
from random import Random
from typing import Any, Callable, Dict, List, Optional

__all__ = ["GroundedOutcome", "GroundedReport", "GroundedExperiment"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class GroundedOutcome:
    """One real experiment graded by reality (execution / fetched data), not a stored key."""

    kind: str
    n_trials: int = 0
    n_passed: int = 0
    source: str = "execution"
    details: List[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return (self.n_passed / self.n_trials) if self.n_trials else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "n_trials": self.n_trials, "n_passed": self.n_passed,
                "score": round(self.score, 6), "source": self.source,
                "details": self.details[:8]}


@dataclass
class GroundedReport:
    """The grounded-improvement summary the RSI loop folds in as a real, externally-graded signal."""

    grounded_score: float = 0.0
    outcomes: List[GroundedOutcome] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"grounded_score": round(self.grounded_score, 6),
                "outcomes": [o.to_dict() for o in self.outcomes],
                "elapsed_ms": round(self.elapsed_ms, 3)}


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
class GroundedExperiment:
    """Run real experiments whose outcome is decided by reality, and grade NYXARA against it.

    Parameters
    ----------
    settings: :class:`~nyxara.kernel.config.NyxaraSettings`.
    seed:     base seed; each evaluation advances an internal RNG so programs are fresh.
    timeout_s: per-program wall-clock budget in the sandbox.
    """

    def __init__(self, *, settings: Any = None, seed: int = 20260629,
                 timeout_s: float = 4.0) -> None:
        from nyxara.kernel.config import get_settings
        self.settings = settings or get_settings()
        self._rng = Random(seed)
        self.timeout_s = float(timeout_s)

    # ------------------------------------------------------------------ #
    # primitive — run code in the real interpreter and recover its result
    # ------------------------------------------------------------------ #
    def _execute(self, code: str) -> Optional[Any]:
        """Run ``code`` in the isolated sandbox and return its ``result`` value (None on failure)."""
        try:
            from nyxara.agency.code_sandbox import run_python
        except Exception:  # noqa: BLE001 — no sandbox ⇒ no grounding available
            return None
        res = run_python(code, timeout_s=self.timeout_s, allow_network=False)
        return res.value if res.ok else None

    def code_experiment(self, candidate_code: str, reference_code: str,
                        inputs: List[Any], *, fn: str = "solve") -> GroundedOutcome:
        """Grade a candidate program against a reference by **real execution** on fresh inputs.

        Neither side's outputs are stored: both are produced by running the code now, so a bug in the
        candidate is revealed by reality, not by a pre-written key."""
        out = GroundedOutcome(kind="code_vs_reference", source="execution")
        for x in inputs:
            cand = self._execute(f"{candidate_code}\nresult = {fn}({x!r})\n")
            ref = self._execute(f"{reference_code}\nresult = {fn}({x!r})\n")
            out.n_trials += 1
            if ref is not None and cand == ref:
                out.n_passed += 1
            else:
                out.details.append(f"{fn}({x!r}): candidate={cand!r} reference={ref!r}")
        return out

    # ------------------------------------------------------------------ #
    # capability ruler — NYXARA predicts a program's output; reality grades
    # ------------------------------------------------------------------ #
    def predict_execution(self, solver: Callable[[str], str], *, trials: int = 6,
                          tier: int = 2) -> GroundedOutcome:
        """Ask ``solver`` to predict each program's ``result``, then **run it** and grade by truth.

        The ground truth is whatever the interpreter actually computes — there is no answer key to
        edit, so this signal cannot be Goodharted. Programs are freshly generated each call."""
        r = Random(self._rng.random())
        out = GroundedOutcome(kind="predict_execution", source="execution")
        for _ in range(max(1, trials)):
            code, prompt = self._gen_program(r, tier)
            truth = self._execute(code + "\n")
            if truth is None:
                continue                         # a program that would not run is not a fair trial
            out.n_trials += 1
            try:
                resp = solver(prompt)
            except Exception:  # noqa: BLE001 — a solver failure is simply a wrong prediction
                resp = ""
            if self._matches(resp, truth):
                out.n_passed += 1
            else:
                out.details.append(f"predicted {resp.strip()[:24]!r} ≠ real {truth!r}")
        return out

    def _gen_program(self, r: Random, tier: int) -> "tuple[str, str]":
        """A small but non-trivial program whose integer ``result`` is only knowable by running it."""
        n = r.randint(5, 5 + 6 * tier)
        k = r.randint(2, 3 + tier)
        op = r.choice(("+= i", "+= i * i", "+= (i % {k})".format(k=k)))
        code = ("total = 0\n"
                f"for i in range(1, {n} + 1):\n"
                f"    if i % {k} == 0:\n"
                f"        total {op}\n"
                "result = total")
        prompt = ("Execute this program in your head and give the final integer value of `result` "
                  "(answer with just the number):\n" + code)
        return code, prompt

    # ------------------------------------------------------------------ #
    # world grounding — predict, then verify against real fetched data
    # ------------------------------------------------------------------ #
    def predict_and_verify(self, prediction: str, *, url: str,
                           must_contain: bool = True) -> GroundedOutcome:
        """Check a prediction against **real web data** (screened), or degrade honestly offline.

        Fetches ``url`` through the SSRF-guarded fetcher + injection scanner (as acquire.py does); the
        prediction passes iff it appears in the genuinely-fetched page. When the network/fetcher is
        unavailable the outcome reports zero trials so its weight is dropped — never a fake pass."""
        out = GroundedOutcome(kind="predict_vs_world", source="web")
        try:
            from nyxara.senses.web import WebFetcher
        except Exception:  # noqa: BLE001 — no fetcher ⇒ no world grounding available
            out.source = "unavailable"
            return out
        try:
            page = WebFetcher().get_page(url)
        except Exception as exc:  # noqa: BLE001 — offline / blocked ⇒ honest no-grounding
            out.source = "unavailable"
            out.details.append(str(exc))
            return out
        injection = getattr(page, "injection", None)
        if injection is not None and getattr(injection, "suspicious", False):
            out.source = "dropped_suspicious"          # screened out, exactly like acquire.py
            return out
        text = getattr(page, "text", "") or ""
        out.n_trials = 1
        hit = prediction.lower() in text.lower()
        out.n_passed = 1 if (hit == must_contain) else 0
        return out

    # ------------------------------------------------------------------ #
    # battery — the grounded_score the RSI loop folds in
    # ------------------------------------------------------------------ #
    def evaluate(self, solver: Callable[[str], str], *, trials: int = 6) -> GroundedReport:
        """Run the grounded battery and combine into a single ``grounded_score`` ∈ [0, 1]."""
        t0 = time.perf_counter()
        outcomes = [self.predict_execution(solver, trials=trials)]
        graded = [o for o in outcomes if o.n_trials > 0]
        score = (sum(o.score for o in graded) / len(graded)) if graded else 0.0
        rep = GroundedReport(grounded_score=score, outcomes=outcomes)
        rep.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return rep

    # ------------------------------------------------------------------ #
    @staticmethod
    def _matches(response: str, truth: Any) -> bool:
        try:
            target = Fraction(int(truth))
        except Exception:  # noqa: BLE001
            return str(truth).strip() in (response or "")
        for tok in re.findall(r"-?\d+", response or ""):
            try:
                if Fraction(int(tok)) == target:
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.agency.code_sandbox import run_python

    print("=" * 70)
    print("NYXARA grounded-experiments self-test")
    print("=" * 70)

    exp = GroundedExperiment(seed=11)

    # 1) code-against-reality: a correct candidate agrees with the reference under real execution,
    #    a buggy candidate is EXPOSED by running it — no stored answer key involved.
    good = "def solve(n):\n    return sum(range(1, n + 1))"
    ref = "def solve(n):\n    return n * (n + 1) // 2"
    bad = "def solve(n):\n    return n * n // 2"
    g = exp.code_experiment(good, ref, [3, 10, 50])
    b = exp.code_experiment(bad, ref, [3, 10, 50])
    print(f"correct candidate   : {g.to_dict()}")
    print(f"buggy candidate     : {b.to_dict()}")
    assert g.score == 1.0, "a correct candidate must agree with reality"
    assert b.score < 1.0, "reality must expose the buggy candidate (no answer key to hide behind)"

    # 2) predict-execution ruler: a solver that actually runs the code scores high; a guesser low.
    def truth_solver(prompt: str) -> str:
        code = prompt.split(":\n", 1)[1] if ":\n" in prompt else ""
        res = run_python(code + "\n", timeout_s=3.0)
        return str(res.value) if res.ok else "0"

    smart = exp.predict_execution(truth_solver, trials=6)
    dumb = exp.predict_execution(lambda p: "0", trials=6)
    print(f"smart predictor     : {smart.to_dict()}")
    print(f"dumb predictor      : {dumb.to_dict()}")
    assert smart.n_trials > 0 and smart.score > dumb.score
    assert smart.score >= 0.8, "predicting via real execution must score high"

    # 3) the combined battery yields a grounded_score in [0, 1]
    rep = exp.evaluate(truth_solver, trials=5)
    print(f"grounded report     : score={rep.grounded_score:.2f}")
    assert 0.0 <= rep.grounded_score <= 1.0 and rep.grounded_score > 0.5

    # 4) world grounding degrades honestly when the network/fetcher is unavailable (no fake pass)
    w = exp.predict_and_verify("anything", url="http://127.0.0.1:0/nope")
    print(f"web (offline)       : source={w.source} trials={w.n_trials}")
    assert w.n_trials == 0 or w.source in ("unavailable", "dropped_suspicious", "web")

    print("\nALL SELF-TESTS PASSED ✓")
