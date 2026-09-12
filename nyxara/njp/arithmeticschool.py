"""NYXARA · njp/arithmeticschool.py — the corpus audited, and then the harder half (📐).

Two questions, and only one of them is about her.

**Is the worked reasoning in the dataset actually right?** 23,371 chains, every stated sum
recomputed by :mod:`nyxara.njp.calculate`. This is an audit of the corpus, not a score for the
reader, and it is worth taking first: evidence nobody has checked is a rumour with a licence
attached.

**Can she solve one she has not seen?** A chain abstracted into a :class:`~nyxara.njp.arithmetic.Shape`
is a little program over the question's own numbers — ``q0 * q1; r0 + q2`` — and solving is picking
the right one. Measured against two floors: the commonest shape applied to every problem, and no
shape at all. The gap between those two and her is the whole of what shape-picking is worth, and it
is reported whatever it comes to.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.arithmetic import Arithmetic, Problem, Shape, read_problems

__all__ = ["Audit", "Solving", "audit", "solve_paper", "run"]

TRAIN = 0.7
#: How many shapes she will consider. A shape seen once is not a method, it is a coincidence.
KEEP_SHAPES = 40


@dataclass
class Audit:
    problems: int = 0
    with_steps: int = 0
    sound: int = 0
    rounded: int = 0
    unsound: int = 0
    numeric: int = 0
    reaches: int = 0
    examples: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        n = max(1, self.with_steps)
        m = max(1, self.numeric)
        return {"problems": self.problems, "with_steps": self.with_steps,
                "exact": round(self.sound / n, 4),
                "allowing_rounding": round((self.sound + self.rounded) / n, 4),
                "unverified": round(self.unsound / n, 4),
                "reaches_its_answer": round(self.reaches / m, 4)}


@dataclass
class Solving:
    name: str = ""
    right: int = 0
    asked: int = 0
    answered: int = 0

    @property
    def accuracy(self) -> float:
        return round(self.right / self.asked, 4) if self.asked else 0.0

    @property
    def when_answered(self) -> float:
        return round(self.right / self.answered, 4) if self.answered else 0.0

    @property
    def coverage(self) -> float:
        return round(self.answered / self.asked, 4) if self.asked else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"accuracy": self.accuracy, "coverage": self.coverage,
                "when_answered": self.when_answered, "asked": self.asked}

    def render(self) -> str:
        return (f"{self.name:<26} {self.accuracy:.3f}   answered {self.coverage:.3f} "
                f"at {self.when_answered:.3f}")


def split(problems: Optional[Sequence[Problem]] = None,
          train: float = TRAIN) -> Tuple[List[Problem], List[Problem]]:
    rows = list(problems if problems is not None else read_problems())
    cut = int(len(rows) * float(train))
    return rows[:cut], rows[cut:]


def audit(problems: Optional[Sequence[Problem]] = None) -> Audit:
    rows = list(problems if problems is not None else read_problems())
    got = Arithmetic().audit(rows)
    return Audit(problems=got["problems"], with_steps=got["with_steps"], sound=got["sound"],
                 rounded=got["rounded"], unsound=got["unsound"], numeric=got["numeric"],
                 reaches=got["reaches"], examples=tuple(got["examples"]))


def _numbers(problem: Problem) -> List[Fraction]:
    from nyxara.njp.arithmetic import _number

    return [n for n in (_number(one) for one in problem.numbers) if n is not None]


def _try(shape: Shape, problem: Problem) -> Optional[Fraction]:
    return shape.run(_numbers(problem))


def solve_paper(learn: Sequence[Problem], held: Sequence[Problem]) -> Dict[str, Solving]:
    """Learn which chain shapes exist, then try to answer problems never seen."""
    reader = Arithmetic()
    counts = reader.learn_shapes(learn)
    ranked = [Shape(moves=tuple(tuple(m.split(" ")) for m in name.split("; ")))
              for name in list(counts)[:KEEP_SHAPES]]
    out: Dict[str, Solving] = {}

    # Her: try the shapes in order of how often they were the one a problem needed, and take the
    # first that binds. No reading of the question at all — which is exactly the point of measuring
    # it, because it says how much of the score is shape frequency rather than shape *choice*.
    hers = Solving(name="commonest shape that binds")
    for problem in held:
        hers.asked += 1
        stated = problem.stated
        if stated is None:
            continue
        for shape in ranked:
            got = _try(shape, problem)
            if got is None:
                continue
            hers.answered += 1
            if abs(got - stated) < Fraction(1, 2):
                hers.right += 1
            break
    out["shapes"] = hers

    only = Solving(name="one shape for everything")
    first = ranked[0] if ranked else None
    for problem in held:
        only.asked += 1
        stated = problem.stated
        if stated is None or first is None:
            continue
        got = _try(first, problem)
        if got is None:
            continue
        only.answered += 1
        if abs(got - stated) < Fraction(1, 2):
            only.right += 1
    out["one_shape"] = only

    none = Solving(name="no shape at all")
    none.asked = len(held)
    out["nothing"] = none
    out["_shapes_learned"] = Solving(name=f"{len(counts)} distinct shapes")
    return out


def run() -> Dict[str, Any]:
    problems = read_problems()
    learn, held = split(problems)
    got = solve_paper(learn, held)
    return {"audit": audit(problems).to_dict(),
            "solving": {k: v.to_dict() for k, v in got.items() if not k.startswith("_")}}


def main() -> None:  # pragma: no cover — a report, not a test
    problems = read_problems()
    if not problems:
        print("no corpus; run scripts/build_reasoning_corpus.py")
        return
    marked = audit(problems)
    row = marked.to_dict()
    print(f"{marked.problems} worked problems, {marked.numeric} answering with a number\n")
    print("=== the corpus, audited by recomputing every stated sum ===")
    print(f"  every sum exact                  {row['exact']:.3f}")
    print(f"  every sum, allowing rounding     {row['allowing_rounding']:.3f}")
    print(f"  a sum this parser cannot verify  {row['unverified']:.3f}")
    print(f"  the chain reaches its answer     {row['reaches_its_answer']:.3f}")
    if marked.examples:
        print("  what could not be verified:")
        for one in marked.examples[:4]:
            print(f"     {one}")

    learn, held = split(problems)
    print(f"\n=== solving {len(held)} problems never seen ===")
    got = solve_paper(learn, held)
    print(f"  ({got['_shapes_learned'].name} learned from {len(learn)} problems)")
    for key in ("nothing", "one_shape", "shapes"):
        print("  " + got[key].render())
    counts = Arithmetic().learn_shapes(learn)
    print("\n  the commonest chain shapes:")
    for name, count in list(counts.items())[:6]:
        print(f"     {count:>6}  {name}")


if __name__ == "__main__":  # pragma: no cover
    main()
