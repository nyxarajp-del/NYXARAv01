"""NYXARA · njp/arithmetic.py — worked sums, recomputed rather than believed (🧮).

The chain-of-thought submix holds 21,886 word problems whose rationale is a chain of stated sums:

    Wendy's dentist bill was 5 * $120 = $600. She got two fillings at 2 * $120 = $240.
    Thus, Wendy paid $600 - $240 - $70 = $290 for the tooth extraction. The answer is 290.

This is the half of the dataset where an answer is **checkable**. :mod:`nyxara.njp.entail` learns
that a hypothesis made of the premise's own words *usually* follows, and *usually* is the honest
word there. Here there is no usually: ``5 * 120`` is 600 or it is not, and
:mod:`nyxara.njp.calculate` settles it in one call.

So the first thing this organ does is not learn from the dataset. It **audits** it: every stated
sum recomputed, every chain followed to its end, and the result compared against the answer the
row claims. A corpus is evidence, and evidence that has never been checked is a rumour with a
licence attached.

The second thing is harder and is reported as what it is. A chain can be abstracted into a
**shape** — where each operand came from, a number in the question or the result of an earlier
step — and a shape is a little program. Whether the right shape can be picked for a problem she
has not seen is a question with a number attached, and the number is not flattering.

Nothing is ``eval``\\ ed: sums reach :func:`nyxara.njp.calculate.evaluate`, which parses with
``ast`` against a fixed operator table, and the operands are numbers this module has already
parsed out of the text.

Pure standard library.
"""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["Problem", "Check", "Shape", "Arithmetic", "CORPUS", "read_problems"]

CORPUS = Path(__file__).with_name("data") / "flan_maths.jsonl.gz"

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_OPS = {"+": "+", "-": "-", "*": "*", "/": "/", "×": "*", "÷": "/"}
_TWO = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)\s*([-+*/×÷])\s*(-?\d[\d,]*(?:\.\d+)?)")

#: How close a recomputed value has to be to the stated one to count as agreeing. Not zero, because
#: the corpus rounds: a chain that divides and then reports two decimal places is not wrong about
#: the arithmetic.
TOLERANCE = Fraction(1, 100)


def _number(text: str) -> Optional[Fraction]:
    try:
        return Fraction(str(text).replace(",", "").strip())
    except Exception:  # noqa: BLE001
        return None


@dataclass(frozen=True)
class Problem:
    """A word problem, the chain someone worked, and the answer they reached."""

    question: str = ""
    worked: str = ""
    answer: str = ""
    #: ``(expression, the value claimed for it)``, as written. Whole expressions rather than
    #: operand pairs: ``3/10 * 20/11 = 6/11`` is one step, and cutting it into two was how an
    #: earlier version reported the corpus wrong about arithmetic it had got right.
    steps: Tuple[Tuple[str, str], ...] = ()
    numbers: Tuple[str, ...] = ()
    task: str = ""

    @property
    def stated(self) -> Optional[Fraction]:
        """The answer as a number, when it is one. Multiple choice answers are letters."""
        match = _NUMBER.search(self.answer)
        return _number(match.group(0)) if match else None

    def to_dict(self) -> Dict[str, Any]:
        return {"question": self.question[:120], "answer": self.answer,
                "steps": len(self.steps), "task": self.task}


def read_problems(path: Optional[Path] = None) -> List[Problem]:
    out: List[Problem] = []
    source = Path(path) if path is not None else CORPUS
    try:
        if not source.exists():
            return out
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                out.append(Problem(question=str(row.get("question") or ""),
                                   worked=str(row.get("worked") or ""),
                                   answer=str(row.get("answer") or ""),
                                   steps=tuple(tuple(s) for s in (row.get("steps") or ())),
                                   numbers=tuple(row.get("numbers") or ()),
                                   task=str(row.get("task") or "")))
    except Exception:  # noqa: BLE001
        return out
    return out


# --------------------------------------------------------------------------------------------- #
#  auditing what the corpus says
# --------------------------------------------------------------------------------------------- #
@dataclass
class Check:
    """What recomputing a worked chain found."""

    steps: int = 0
    computed: int = 0
    agreed: int = 0
    #: Sums whose stated value is the rounded one. Counted apart from both agreement and
    #: disagreement, because a chain that divides and writes ``45/4 = 11`` has not made a mistake
    #: about arithmetic — it has rounded — and filing that as an error would be a claim about the
    #: corpus that the corpus does not deserve.
    rounded: int = 0
    reaches: bool = False
    #: The answer this problem states is a number at all. Multiple-choice rows answer "(D)", and
    #: counting those as chains that failed to reach their answer understated the corpus by a
    #: quarter.
    numeric: bool = False
    #: The first sum whose stated result is not what it computes to, verbatim.
    wrong: str = ""

    @property
    def sound(self) -> bool:
        """Every sum in the chain computes to what it says it does, exactly."""
        return self.computed > 0 and self.agreed == self.computed

    @property
    def sound_allowing_rounding(self) -> bool:
        return self.computed > 0 and self.agreed + self.rounded == self.computed

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": self.steps, "checked": self.computed, "agreed": self.agreed,
                "rounded": self.rounded, "sound": self.sound,
                "sound_allowing_rounding": self.sound_allowing_rounding,
                "reaches": self.reaches, "numeric": self.numeric, "wrong": self.wrong}


def _evaluate_expression(expression: str) -> Optional[Fraction]:
    """A whole stated expression, through her own calculator. Never `eval`."""
    try:
        from nyxara.njp.calculate import evaluate

        got = evaluate(expression.replace(",", "").replace("×", "*").replace("÷", "/"))
        if not getattr(got, "ok", False) or got.value is None:
            return None
        return Fraction(str(got.value))
    except Exception:  # noqa: BLE001
        return None


def _evaluate(left: str, op: str, right: str) -> Optional[Fraction]:
    """One sum, through her own calculator. Never `eval`.

    Written without the defensive brackets it wanted at first: ``njp.calculate`` refuses
    ``(3) * (40)`` and accepts ``3 * 40``, and with the brackets in place **every one** of 21,886
    audits came back with nothing computed and nothing wrong — a clean sheet that meant the
    checker had never run. The operands here are single numbers already parsed out of the text, so
    there is nothing for a bracket to protect.
    """
    try:
        from nyxara.njp.calculate import evaluate

        got = evaluate(f"{left.replace(',', '')} {_OPS.get(op, op)} {right.replace(',', '')}")
        if not getattr(got, "ok", False):
            return None
        return Fraction(str(getattr(got, "value", ""))) if got.value is not None else None
    except Exception:  # noqa: BLE001
        return None


class Arithmetic:
    """Recomputes worked chains, and tries to learn which chain a problem calls for."""

    def __init__(self, *, tolerance: Fraction = TOLERANCE) -> None:
        self.tolerance = tolerance
        self.shapes: Dict[str, int] = {}

    # -- the audit ----------------------------------------------------------- #
    def check(self, problem: Problem) -> Check:
        """Recompute every stated sum and follow the chain to its end."""
        out = Check(steps=len(problem.steps))
        last: Optional[Fraction] = None
        for expression, said in problem.steps:
            got = _evaluate_expression(expression)
            claimed = _number(said)
            if got is None or claimed is None:
                continue
            out.computed += 1
            if abs(got - claimed) <= self.tolerance:
                out.agreed += 1
                last = claimed
            elif abs(got - claimed) < Fraction(1, 2) or claimed == round(got):
                out.rounded += 1
                last = claimed
            elif not out.wrong:
                out.wrong = f"{expression} = {said}, but it is {got}"
        stated = problem.stated
        out.numeric = stated is not None
        out.reaches = bool(last is not None and stated is not None
                           and abs(last - stated) < Fraction(1, 2))
        return out

    def audit(self, problems: Sequence[Problem]) -> Dict[str, Any]:
        """The corpus, checked. What it says about the corpus, not about her."""
        out: Dict[str, Any] = {"problems": 0, "with_steps": 0, "sound": 0, "rounded": 0,
                               "numeric": 0, "reaches": 0, "unsound": 0, "examples": []}
        for problem in problems:
            out["problems"] += 1
            if not problem.steps:
                continue
            out["with_steps"] += 1
            got = self.check(problem)
            if got.sound:
                out["sound"] += 1
            elif got.sound_allowing_rounding:
                out["rounded"] += 1
            elif got.wrong:
                out["unsound"] += 1
                if len(out["examples"]) < 8:
                    out["examples"].append(got.wrong)
            if got.numeric:
                out["numeric"] += 1
                if got.reaches:
                    out["reaches"] += 1
        return out

    # -- the shape of a chain ------------------------------------------------- #
    def shape_of(self, problem: Problem) -> Optional["Shape"]:
        """Where each operand came from: a number in the question, or an earlier result."""
        givens = [g for g in (_number(n) for n in problem.numbers) if g is not None]
        results: List[Fraction] = []
        moves: List[Tuple[str, str, str]] = []
        for expression, said in problem.steps:
            simple = _TWO.fullmatch(expression.strip())
            if simple is None:
                return None            # a step with more than one operator has no simple shape
            left, op, right = simple.group(1), simple.group(2), simple.group(3)
            a, b, c = _number(left), _number(right), _number(said)
            if a is None or b is None or c is None:
                return None
            source_a = _source(a, givens, results)
            source_b = _source(b, givens, results)
            if source_a is None or source_b is None:
                return None
            moves.append((source_a, _OPS.get(op, op), source_b))
            results.append(c)
        return Shape(moves=tuple(moves)) if moves else None

    def learn_shapes(self, problems: Sequence[Problem]) -> Dict[str, int]:
        """How often each chain shape is the one a problem needed."""
        counts: Dict[str, int] = {}
        for problem in problems:
            shape = self.shape_of(problem)
            if shape is None:
                continue
            counts[shape.render()] = counts.get(shape.render(), 0) + 1
        self.shapes = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
        return self.shapes


def _source(value: Fraction, givens: Sequence[Fraction],
            results: Sequence[Fraction]) -> Optional[str]:
    for index, got in enumerate(reversed(results)):
        if got == value:
            return f"r{len(results) - 1 - index}"
    for index, given in enumerate(givens):
        if given == value:
            return f"q{index}"
    return None


@dataclass(frozen=True)
class Shape:
    """A chain with its numbers taken out — a little program over the question's own numbers."""

    moves: Tuple[Tuple[str, str, str], ...] = ()

    def render(self) -> str:
        return "; ".join(f"{a} {op} {b}" for a, op, b in self.moves)

    def run(self, givens: Sequence[Fraction]) -> Optional[Fraction]:
        """Apply it to a problem's numbers. ``None`` where a slot has nothing to bind to."""
        results: List[Fraction] = []
        for left, op, right in self.moves:
            a = _bind(left, givens, results)
            b = _bind(right, givens, results)
            if a is None or b is None:
                return None
            try:
                if op == "+":
                    results.append(a + b)
                elif op == "-":
                    results.append(a - b)
                elif op == "*":
                    results.append(a * b)
                elif op == "/":
                    if b == 0:
                        return None
                    results.append(a / b)
                else:
                    return None
            except Exception:  # noqa: BLE001
                return None
        return results[-1] if results else None


def _bind(slot: str, givens: Sequence[Fraction],
          results: Sequence[Fraction]) -> Optional[Fraction]:
    try:
        index = int(slot[1:])
    except Exception:  # noqa: BLE001
        return None
    pool = givens if slot.startswith("q") else results
    return pool[index] if 0 <= index < len(pool) else None
