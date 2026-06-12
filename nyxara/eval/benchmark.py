"""NYXARA · eval/benchmark.py — graded capability benchmarks (📏 → 🧠).

The safety harness (``eval/harness.py`` + ``suites.py``) answers *"is the mind still safe and
corrigible?"*. This module answers the other half — *"how capable is the mind?"* — with the
machinery the field uses to measure intelligence: a battery of tasks each paired with a
**grader** that scores a free-text answer against a known solution (exact / numeric /
contains / multiple-choice). Run any *solver* (NYXARA's whole loop, a bare LLM, or a scripted
stand-in) over the battery and get an auditable accuracy report, overall and per category,
with baseline/regression tracking just like the safety suite.

Design rules:

* **Model-agnostic.** A solver is just ``Callable[[str], str]`` (prompt → answer). Adapters
  wrap a :class:`~nyxara.kernel.orchestrator.NyxaraCore` or an :class:`~nyxara.mind.llm.LLM`,
  so the *same* benchmark measures the offline reasoner, a local model, or a frontier API —
  apples to apples.
* **Deterministic harness.** Grading is pure and reproducible; only the solver varies. The
  built-in tasks run anywhere (no network) — meaningful *scores* need a real model, but the
  *harness* is fully testable offline with a scripted solver.
* **Honest.** A grader returns 1.0 only on a genuinely correct answer; partial credit is
  explicit. The number it reports is the number it means.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "Grader", "BenchmarkTask", "BenchmarkResult", "BenchmarkReport", "Benchmark",
    "Solver", "core_solver", "llm_solver", "self_solver",
    "build_arithmetic_benchmark", "build_logic_benchmark", "build_default_benchmark",
]

Solver = Callable[[str], str]
GraderFn = Callable[[str, "BenchmarkTask"], Tuple[float, str]]

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


# --------------------------------------------------------------------------- #
# Graders (pure, model-agnostic)
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def grade_numeric(response: str, task: "BenchmarkTask") -> Tuple[float, str]:
    """Score by the LAST number in the response vs the expected value (with tolerance)."""
    matches = _NUMBER.findall(response or "")
    if not matches:
        return 0.0, "no number found in response"
    got = float(matches[-1].replace(",", ""))
    want = float(task.answer)
    tol = task.tolerance if task.tolerance is not None else 1e-6
    if abs(got - want) <= tol:
        return 1.0, f"{got} == {want}"
    return 0.0, f"{got} != {want}"


def grade_exact(response: str, task: "BenchmarkTask") -> Tuple[float, str]:
    """Score 1.0 iff the normalised response equals the answer (or any accepted alias)."""
    got = _norm(response)
    accepted = {_norm(task.answer), *(_norm(a) for a in task.aliases)}
    if got in accepted:
        return 1.0, "exact match"
    return 0.0, f"{got!r} not in {sorted(accepted)!r}"


def grade_contains(response: str, task: "BenchmarkTask") -> Tuple[float, str]:
    """Score 1.0 iff every required phrase (answer + aliases) appears in the response."""
    got = _norm(response)
    needles = [_norm(task.answer), *(_norm(a) for a in task.aliases)]
    missing = [n for n in needles if n and n not in got]
    if not missing:
        return 1.0, "all phrases present"
    return 0.0, f"missing: {missing}"


def grade_multiple_choice(response: str, task: "BenchmarkTask") -> Tuple[float, str]:
    """Score by the option the solver actually *chose*.

    Robust against an echo of the prompt: simply containing the letters A–D (as every option
    label does) is NOT a choice. A choice is an explicit "answer is X", a response that is
    essentially just the letter, or the unambiguous spelled-out option text.
    """
    want = _norm(task.answer)
    resp = response or ""

    # 1) An explicit answer marker — "the answer is B", "choose C", "option A". A real
    # solver states its choice last, so take the LAST marker whose letter is a valid option
    # (this ignores an instruction like "answer A or B" echoed earlier in the text).
    n_choices = len(task.choices) or 26
    valid = {"abcdefghijklmnopqrstuvwxyz"[i] for i in range(n_choices)}
    markers = re.findall(r"(?:answer|choose|chose|option|pick|select|correct)\b"
                         r".{0,15}?\b([A-Za-z])\b", resp, re.I)
    chosen_markers = [x.lower() for x in markers if x.lower() in valid]
    if chosen_markers:
        chosen = chosen_markers[-1]
        return (1.0, f"chose {chosen.upper()}") if chosen == want \
            else (0.0, f"chose {chosen.upper()}, wanted {want.upper()}")

    # 2) The response is essentially just the letter (optionally as "B)" / "(B)" / "B.").
    m2 = re.match(r"^\(?([A-Za-z])[).:\s]", resp.strip()) or \
        re.fullmatch(r"\(?([A-Za-z])\)?", resp.strip())
    if m2:
        chosen = m2.group(1).lower()
        return (1.0, f"chose {chosen.upper()}") if chosen == want \
            else (0.0, f"chose {chosen.upper()}, wanted {want.upper()}")

    # 3) Unambiguous spelled-out option text (the right option present, the others absent).
    if task.choices:
        idx = "abcdefghijklmnopqrstuvwxyz".find(want)
        if 0 <= idx < len(task.choices):
            target = _norm(task.choices[idx])
            others = [_norm(c) for j, c in enumerate(task.choices) if j != idx]
            rn = _norm(resp)
            if target and target in rn and not any(o and o in rn for o in others):
                return 1.0, "matched option text"
    return 0.0, "no clear choice"


class Grader:
    """Named graders. ``Grader.get(name)`` resolves the function used by a task."""

    NUMERIC = "numeric"
    EXACT = "exact"
    CONTAINS = "contains"
    MULTIPLE_CHOICE = "multiple_choice"

    _FNS: Dict[str, GraderFn] = {
        NUMERIC: grade_numeric, EXACT: grade_exact,
        CONTAINS: grade_contains, MULTIPLE_CHOICE: grade_multiple_choice,
    }

    @classmethod
    def get(cls, name: str) -> GraderFn:
        try:
            return cls._FNS[name]
        except KeyError:
            raise ValueError(f"unknown grader {name!r}; "
                             f"choose from {sorted(cls._FNS)}") from None


# --------------------------------------------------------------------------- #
# Task & results
# --------------------------------------------------------------------------- #
@dataclass
class BenchmarkTask:
    id: str
    prompt: str
    answer: str
    grader: str = Grader.EXACT
    category: str = "general"
    choices: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)        # other accepted answers
    tolerance: Optional[float] = None                       # numeric grader slack
    custom_grader: Optional[GraderFn] = None                # overrides ``grader`` when set
    metadata: Dict[str, Any] = field(default_factory=dict)

    def grade(self, response: str) -> Tuple[float, str]:
        fn = self.custom_grader or Grader.get(self.grader)
        return fn(response, self)


@dataclass
class BenchmarkResult:
    task_id: str
    category: str
    response: str
    score: float
    detail: str = ""
    latency_s: float = 0.0

    @property
    def passed(self) -> bool:
        return self.score >= 0.999

    def to_dict(self) -> Dict[str, Any]:
        return {"task_id": self.task_id, "category": self.category,
                "score": round(self.score, 4), "passed": self.passed,
                "detail": self.detail, "latency_s": round(self.latency_s, 5),
                "response": self.response[:500]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkResult":
        return cls(task_id=d["task_id"], category=d.get("category", "general"),
                   response=d.get("response", ""), score=float(d.get("score", 0.0)),
                   detail=d.get("detail", ""), latency_s=float(d.get("latency_s", 0.0)))


@dataclass
class BenchmarkReport:
    name: str
    results: List[BenchmarkResult] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def accuracy(self) -> float:
        return self.passed / len(self.results) if self.results else 0.0

    @property
    def mean_score(self) -> float:
        return sum(r.score for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def mean_latency(self) -> float:
        return (sum(r.latency_s for r in self.results) / len(self.results)
                if self.results else 0.0)

    def by_category(self) -> Dict[str, Dict[str, Any]]:
        cats: Dict[str, List[BenchmarkResult]] = {}
        for r in self.results:
            cats.setdefault(r.category, []).append(r)
        out: Dict[str, Dict[str, Any]] = {}
        for cat, rs in cats.items():
            n = len(rs)
            out[cat] = {"n": n, "passed": sum(1 for r in rs if r.passed),
                        "accuracy": sum(1 for r in rs if r.passed) / n,
                        "mean_score": sum(r.score for r in rs) / n}
        return out

    def failures(self) -> List[BenchmarkResult]:
        return [r for r in self.results if not r.passed]

    def get(self, task_id: str) -> Optional[BenchmarkResult]:
        return next((r for r in self.results if r.task_id == task_id), None)

    def regression_vs(self, baseline: "BenchmarkReport") -> List[str]:
        """Task ids that passed in ``baseline`` but fail now (capability lost)."""
        lost: List[str] = []
        for b in baseline.results:
            if not b.passed:
                continue
            cur = self.get(b.task_id)
            if cur is None or not cur.passed:
                lost.append(b.task_id)
        return lost

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "accuracy": round(self.accuracy, 4),
                "mean_score": round(self.mean_score, 4),
                "results": [r.to_dict() for r in self.results]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkReport":
        return cls(name=d.get("name", "benchmark"),
                   results=[BenchmarkResult.from_dict(r) for r in d.get("results", [])])

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "BenchmarkReport":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def summary(self) -> str:
        lines = ["=" * 70, f"NYXARA benchmark report — {self.name}", "=" * 70,
                 f"overall   : {self.passed}/{len(self.results)} correct "
                 f"({self.accuracy:.0%}), mean score {self.mean_score:.3f}, "
                 f"mean latency {self.mean_latency*1000:.0f}ms", "", "by category:"]
        for cat, st in sorted(self.by_category().items()):
            lines.append(f"  {cat:<14}: {st['accuracy']:.0%} "
                         f"({st['passed']}/{st['n']}), mean {st['mean_score']:.3f}")
        fails = self.failures()
        if fails:
            lines += ["", "failures:"]
            lines += [f"  ✗ [{r.category}] {r.task_id} — {r.detail}" for r in fails]
        else:
            lines += ["", "no failures ✓"]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Benchmark runner
# --------------------------------------------------------------------------- #
class Benchmark:
    """A named battery of tasks, run against any solver into a report."""

    def __init__(self, name: str = "default",
                 tasks: Optional[List[BenchmarkTask]] = None) -> None:
        self.name = name
        self._tasks: List[BenchmarkTask] = list(tasks or [])

    def add(self, task: BenchmarkTask) -> BenchmarkTask:
        self._tasks.append(task)
        return task

    def __len__(self) -> int:
        return len(self._tasks)

    def tasks(self, category: Optional[str] = None) -> List[BenchmarkTask]:
        if category is None:
            return list(self._tasks)
        return [t for t in self._tasks if t.category == category]

    def categories(self) -> List[str]:
        return sorted({t.category for t in self._tasks})

    def run(self, solver: Solver, *, category: Optional[str] = None) -> BenchmarkReport:
        results: List[BenchmarkResult] = []
        for task in self.tasks(category):
            t0 = time.monotonic()
            try:
                response = solver(task.prompt)
            except Exception as exc:  # noqa: BLE001 — a solver error is a 0, never a crash
                response, score, detail = "", 0.0, f"solver error: {exc}"
            else:
                score, detail = task.grade(response or "")
            results.append(BenchmarkResult(
                task_id=task.id, category=task.category, response=response or "",
                score=score, detail=detail, latency_s=time.monotonic() - t0))
        return BenchmarkReport(name=self.name, results=results)


# --------------------------------------------------------------------------- #
# Solver adapters
# --------------------------------------------------------------------------- #
def core_solver(core_factory: Optional[Callable[[], Any]] = None, *,
                fresh_per_task: bool = True) -> Solver:
    """A solver that runs a prompt through NYXARA's whole sovereign loop.

    By default a fresh core per task, so memory/journal state never leaks between tasks and
    the measurement is clean (as in the safety harness).
    """
    from nyxara.agency.permissions import Authority
    if core_factory is None:
        from nyxara.eval.harness import default_core_factory
        core_factory = default_core_factory
    shared = None if fresh_per_task else core_factory()

    def _solve(prompt: str) -> str:
        core = core_factory() if fresh_per_task else shared
        return core.process(prompt, authority=Authority.OWNER).response

    return _solve


_BENCH_SYSTEM = ("Answer the question. Put the final answer last, plainly — a "
                 "number for arithmetic, or the single letter for a choice.")


def llm_solver(llm: Optional[Any] = None, *, system: Optional[str] = None) -> Solver:
    """A solver that asks a bare LLM directly (measures the model, not the whole loop)."""
    from nyxara.mind.llm import LLM
    llm = llm or LLM()
    sys_prompt = system or _BENCH_SYSTEM

    def _solve(prompt: str) -> str:
        return llm.generate(prompt, system=sys_prompt)

    return _solve


def self_solver(*, system: Optional[str] = None, settings: Optional[Any] = None) -> Solver:
    """A solver that asks NYXARA's OWN promoted model directly (bypassing the loop).

    The peer of :func:`llm_solver` for A/B measurement: the *same* battery against NYXARA's own
    substrate vs the external teacher, apples to apples. Honest about an un-forged substrate —
    returns ``""`` (scores 0) when no model has been trained + promoted yet, never crashes.
    """
    from nyxara.mind.llm import SelfProvider, LLMRequest
    prov = SelfProvider(settings)
    sys_prompt = system or _BENCH_SYSTEM

    def _solve(prompt: str) -> str:
        if not prov.available():
            return ""
        req = LLMRequest.from_prompt(prompt, system=sys_prompt,
                                     temperature=0.0, max_tokens=256)
        return prov.complete(req).text

    return _solve


# --------------------------------------------------------------------------- #
# Built-in benchmark suites
# --------------------------------------------------------------------------- #
def build_arithmetic_benchmark(n: int = 12) -> Benchmark:
    """Deterministic arithmetic word problems with exact numeric answers (GSM-style)."""
    tasks: List[BenchmarkTask] = []
    for i in range(n):
        a, b, c = 3 + i, 2 + (i % 5), 1 + (i % 4)
        total = a * b + c
        tasks.append(BenchmarkTask(
            id=f"arith-{i:02d}", category="arithmetic", grader=Grader.NUMERIC,
            answer=str(total),
            prompt=(f"A crate holds {a} boxes and each box holds {b} apples. "
                    f"Another {c} apples sit loose beside the crate. "
                    f"How many apples are there in total? Give just the number.")))
    return Benchmark("arithmetic", tasks)


def build_logic_benchmark() -> Benchmark:
    """A small static set of commonsense / logic multiple-choice items."""
    raw = [
        ("logic-00", "All bloops are razzies, and all razzies are lazzies. Are all bloops "
         "definitely lazzies? Options: A) yes  B) no", ["yes", "no"], "A"),
        ("logic-01", "Tom is taller than Sam. Sam is taller than Ada. Who is shortest? "
         "A) Tom  B) Sam  C) Ada", ["Tom", "Sam", "Ada"], "C"),
        ("logic-02", "A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the "
         "ball. How much does the ball cost, in cents? A) 10  B) 5  C) 1", ["10", "5", "1"], "B"),
        ("logic-03", "Which word does not belong? A) rose  B) tulip  C) hammer  D) daisy",
         ["rose", "tulip", "hammer", "daisy"], "C"),
    ]
    tasks = [BenchmarkTask(id=i, prompt=p, answer=ans, grader=Grader.MULTIPLE_CHOICE,
                           category="logic", choices=ch) for (i, p, ch, ans) in raw]
    return Benchmark("logic", tasks)


def build_default_benchmark() -> Benchmark:
    """The combined default battery (arithmetic + logic)."""
    bench = Benchmark("default")
    for src in (build_arithmetic_benchmark(), build_logic_benchmark()):
        for t in src.tasks():
            bench.add(t)
    return bench


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA benchmark self-test")
    print("=" * 70)

    bench = build_default_benchmark()
    print(f"tasks: {len(bench)} across {bench.categories()}")

    # An oracle solver answers every task correctly -> 100%.
    def oracle(prompt: str) -> str:
        for t in bench.tasks():
            if t.prompt == prompt:
                return (t.answer if t.grader != Grader.MULTIPLE_CHOICE
                        else f"The answer is {t.answer}.")
        return ""

    report = bench.run(oracle)
    print("\n" + report.summary())
    assert report.accuracy == 1.0, "oracle should ace it"

    # A wrong solver -> 0%, and is flagged as a regression against the oracle baseline.
    wrong = bench.run(lambda p: "definitely 99999")
    assert wrong.accuracy == 0.0
    assert len(wrong.regression_vs(report)) == len(bench)
    print("\nALL SELF-TESTS PASSED ✓")
