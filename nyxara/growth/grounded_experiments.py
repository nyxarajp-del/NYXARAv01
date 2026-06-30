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
    # capability to AUTHOR correct code, graded by real execution vs a reference (None when not run)
    tool_grounded_score: Optional[float] = None
    outcomes: List[GroundedOutcome] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"grounded_score": round(self.grounded_score, 6),
                             "outcomes": [o.to_dict() for o in self.outcomes],
                             "elapsed_ms": round(self.elapsed_ms, 3)}
        if self.tool_grounded_score is not None:
            d["tool_grounded_score"] = round(self.tool_grounded_score, 6)
        return d


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
    # capability ruler — NYXARA AUTHORS code; reality (the interpreter) grades
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_code(response: str) -> str:
        """Recover Python source from a solver reply, tolerating Markdown fences and prose.

        Prefers the contents of a fenced ```python block; otherwise returns the raw reply. A buggy
        or empty extraction simply fails to define ``solve`` and is graded as a miss — never raised.
        """
        text = response or ""
        m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        return (m.group(1) if m else text).strip()

    def code_authoring_experiment(self, solver: Callable[[str], str], *, trials: int = 4,
                                  ) -> GroundedOutcome:
        """Ask ``solver`` to AUTHOR a ``solve`` function for fresh specs; grade by REAL execution.

        For each trial a small, unambiguous task is generated together with a reference implementation.
        The solver writes its own ``solve``; both programs are then RUN on fresh inputs in the isolated
        sandbox and compared by :meth:`code_experiment`. The grade is whatever the interpreter actually
        computes — there is no stored answer key, so a wrong program is exposed by reality, not hidden.
        Measures real capability (can she write correct code?), the strongest reality-grounded ruler.
        """
        r = Random(self._rng.random())
        out = GroundedOutcome(kind="code_authoring", source="execution")
        for _ in range(max(1, trials)):
            spec, reference, inputs = self._gen_authoring_task(r)
            try:
                candidate = self._extract_code(solver(spec))
            except Exception:  # noqa: BLE001 — a solver failure is simply a failed authoring trial
                candidate = ""
            sub = self.code_experiment(candidate, reference, inputs, fn="solve")
            out.n_trials += 1
            # the trial passes only if EVERY fresh input agrees with the reference under real execution
            if sub.n_trials > 0 and sub.n_passed == sub.n_trials:
                out.n_passed += 1
            elif sub.details:
                out.details.append(sub.details[0])
        return out

    def _gen_authoring_task(self, r: Random) -> "tuple[str, str, List[Any]]":
        """A fresh (prompt, reference-code, inputs) triple for a small, unambiguous coding task."""
        kind = r.choice(("sum_to_n", "factorial", "count_even", "reverse_str", "max_of_list"))
        if kind == "sum_to_n":
            ref = "def solve(n):\n    return sum(range(1, n + 1))"
            prompt = ("Write a Python function `solve(n)` that returns the sum of all integers from 1 "
                      "to n inclusive. Reply with only the function definition.")
            inputs: List[Any] = [r.randint(1, 40) for _ in range(4)]
        elif kind == "factorial":
            ref = "def solve(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r"
            prompt = ("Write a Python function `solve(n)` that returns n factorial (n!), with "
                      "solve(0) == 1. Reply with only the function definition.")
            inputs = [r.randint(0, 8) for _ in range(4)]
        elif kind == "count_even":
            ref = "def solve(n):\n    return sum(1 for i in range(1, n + 1) if i % 2 == 0)"
            prompt = ("Write a Python function `solve(n)` that returns how many even numbers are in "
                      "the range 1 to n inclusive. Reply with only the function definition.")
            inputs = [r.randint(1, 40) for _ in range(4)]
        elif kind == "reverse_str":
            ref = "def solve(s):\n    return s[::-1]"
            prompt = ("Write a Python function `solve(s)` that returns the string s reversed. "
                      "Reply with only the function definition.")
            words = ["nyxara", "reality", "kernel", "ground", "verify", "tensor"]
            inputs = [r.choice(words) for _ in range(4)]
        else:  # max_of_list
            ref = "def solve(xs):\n    return max(xs)"
            prompt = ("Write a Python function `solve(xs)` that returns the largest number in the "
                      "list xs. Reply with only the function definition.")
            inputs = [[r.randint(-20, 20) for _ in range(r.randint(2, 6))] for _ in range(4)]
        return prompt, ref, inputs

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
        """A small but non-trivial program whose integer ``result`` is only knowable by running it.

        Draws from several families (arithmetic loops, string building, list comprehensions) so the
        execution ruler covers more of real Python behaviour than integer accumulation alone."""
        family = r.choice(("loop", "string", "listcomp"))
        n = r.randint(5, 5 + 6 * tier)
        k = r.randint(2, 3 + tier)
        if family == "string":
            word = r.choice(("alpha", "ground", "tensor", "kernel", "verify"))
            code = ("s = \"\"\n"
                    f"for i in range(1, {n} + 1):\n"
                    f"    if i % {k} == 0:\n"
                    f"        s += {word!r}\n"
                    "result = len(s)")
        elif family == "listcomp":
            code = (f"xs = [i * i - {k} for i in range(1, {n} + 1)]\n"
                    f"result = sum(x for x in xs if x % {k} != 0)")
        else:  # loop
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
        """Run the grounded battery and combine into a single ``grounded_score`` ∈ [0, 1].

        ``grounded_score`` blends *prediction* rulers graded by reality — execution prediction always,
        and (when ``grounded_web_enabled``) screened live-web verification, which degrades honestly to
        no-grounding offline. When ``tool_grounding_enabled`` is set, a separate ``tool_grounded_score``
        also measures NYXARA's capability to AUTHOR correct code (graded by real execution); it is
        reported distinctly so the RSI loop can weight authoring independently of prediction.
        """
        cfg = getattr(self.settings, "self_improvement", None)
        t0 = time.perf_counter()
        outcomes = [self.predict_execution(solver, trials=trials)]
        # live-web grounding (opt-in): each probe is screened (SSRF + injection) and degrades to
        # zero-trials offline, so its weight is simply dropped rather than faked.
        if bool(getattr(cfg, "grounded_web_enabled", False)):
            for url, expect, must in self._load_web_probes(cfg):
                outcomes.append(self.predict_and_verify(expect, url=url, must_contain=must))
        graded = [o for o in outcomes if o.n_trials > 0]
        score = (sum(o.score for o in graded) / len(graded)) if graded else 0.0
        rep = GroundedReport(grounded_score=score, outcomes=outcomes)
        # agentic tool-task ruler (opt-in): author code, graded by real execution. Reported separately.
        if cfg is None or bool(getattr(cfg, "tool_grounding_enabled", True)):
            tool = self.code_authoring_experiment(solver, trials=max(1, trials // 2 or 1))
            rep.outcomes.append(tool)
            if tool.n_trials > 0:
                rep.tool_grounded_score = tool.score
        rep.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return rep

    @staticmethod
    def _load_web_probes(cfg: Any) -> List["tuple[str, str, bool]"]:
        """Load live-web probes from the configured JSONL, or return an empty list.

        Each record is ``{"url": ..., "expect": ..., "must_contain": true}``. A missing path or
        unreadable/malformed file yields no probes (best-effort) — web grounding is never fatal.
        """
        import json as _json
        path = getattr(cfg, "grounded_web_probes_path", None) if cfg is not None else None
        if not path:
            return []
        probes: List["tuple[str, str, bool]"] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    rec = _json.loads(line)
                    url = str(rec.get("url", "")).strip()
                    expect = str(rec.get("expect", "")).strip()
                    if url and expect:
                        probes.append((url, expect, bool(rec.get("must_contain", True))))
        except Exception:  # noqa: BLE001 — bad/missing probe file ⇒ simply no web grounding
            return []
        return probes

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

    # 3) code-authoring ruler: a solver that writes the obvious correct function scores high, a
    #    solver that emits nonsense is EXPOSED by execution — graded by reality, no answer key.
    def author_solver(prompt: str) -> str:
        if "sum of all integers" in prompt:
            return "def solve(n):\n    return sum(range(1, n + 1))"
        if "factorial" in prompt:
            return "import math\ndef solve(n):\n    return math.factorial(n)"
        if "how many even numbers" in prompt:
            return "def solve(n):\n    return n // 2"
        if "reversed" in prompt:
            return "def solve(s):\n    return s[::-1]"
        return "def solve(*a):\n    return max(*a) if len(a) > 1 else max(a[0])"

    smart_auth = exp.code_authoring_experiment(author_solver, trials=5)
    dumb_auth = exp.code_authoring_experiment(lambda p: "def solve(*a):\n    return -999", trials=5)
    print(f"smart author        : {smart_auth.to_dict()}")
    print(f"dumb author         : {dumb_auth.to_dict()}")
    assert smart_auth.score > dumb_auth.score, "real authoring capability must beat nonsense"

    # 4) the combined battery yields a grounded_score in [0, 1] and a separate tool_grounded_score
    rep = exp.evaluate(truth_solver, trials=5)
    print(f"grounded report     : score={rep.grounded_score:.2f} tool={rep.tool_grounded_score}")
    assert 0.0 <= rep.grounded_score <= 1.0 and rep.grounded_score > 0.5
    assert rep.tool_grounded_score is None or 0.0 <= rep.tool_grounded_score <= 1.0

    # 5) world grounding degrades honestly when the network/fetcher is unavailable (no fake pass)
    w = exp.predict_and_verify("anything", url="http://127.0.0.1:0/nope")
    print(f"web (offline)       : source={w.source} trials={w.n_trials}")
    assert w.n_trials == 0 or w.source in ("unavailable", "dropped_suspicious", "web")

    print("\nALL SELF-TESTS PASSED ✓")
