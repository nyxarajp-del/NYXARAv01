"""NYXARA · njp/taskschool.py — how many of the dataset's tasks can she actually do (📏).

One question, asked of every task in FLAN separately, and answered by counting:

    Shown seventy examples of this task and asked thirty it has never seen, does she beat
    **always saying whichever answer was commonest**?

The majority baseline is per task and it is not a formality. A two-way task whose answers run
nine-to-one is 0.9 for a machine that has learned nothing, and an accuracy of 0.85 on it is a
failure dressed as a pass. So nothing here reports accuracy without the majority beside it, and
`learned something` means *beat its own task's floor*, not *scored well*.

Three groups, and all three are counted:

* **in scope** — the answers are a small set, so there is something to choose between.
* **out of scope** — the answer is free text. Translate this, summarise this, write a question.
  This machinery picks among answers it has seen and cannot compose a new one, so these are not
  attempted and not counted as failures. They are counted as *not attempted*, which is a different
  and more honest thing.
* **too few examples** — fewer than thirty instances collected. Beating a majority computed on
  twenty rows means nothing.

And one ablation, which is the only reason :mod:`nyxara.njp.shapes` is wired in here at all:
``no_shape`` learns from the **whole prompt** instead of from what the shape says are the slots. A
task's instruction is identical in every row of it, so its words carry no signal and should be
pure noise in the reading. If reading the slots alone is worth nothing, then the shape induction
bought nothing downstream and that is the finding.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.shapes import Shape, induce, read_groups
from nyxara.njp.tasks import Learned, TaskLearner, read_examples

__all__ = ["Report", "shapes_by_task", "examine", "run"]


@dataclass
class Report:
    name: str = ""
    in_scope: int = 0
    free_text: int = 0
    too_few: int = 0
    won: int = 0
    results: List[Learned] = field(default_factory=list)

    @property
    def tasks(self) -> int:
        return self.in_scope + self.free_text + self.too_few

    @property
    def beat_majority(self) -> float:
        return round(self.won / self.in_scope, 4) if self.in_scope else 0.0

    @property
    def accuracy(self) -> float:
        return round(statistics.mean([r.accuracy for r in self.results]), 4) \
            if self.results else 0.0

    @property
    def majority(self) -> float:
        return round(statistics.mean([r.majority for r in self.results]), 4) \
            if self.results else 0.0

    @property
    def lift(self) -> float:
        """How far above its own floor the average task ends up. The number that matters."""
        return round(self.accuracy - self.majority, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "tasks": self.tasks, "in_scope": self.in_scope,
                "free_text": self.free_text, "too_few": self.too_few,
                "beat_majority": self.beat_majority, "accuracy": self.accuracy,
                "majority": self.majority, "lift": self.lift}

    def render(self) -> str:
        return (f"{self.name:<12} in scope {self.in_scope:>4}   beat their own floor "
                f"{self.won:>4} = {self.beat_majority:.3f}   accuracy {self.accuracy:.3f} "
                f"vs majority {self.majority:.3f}   lift {self.lift:+.3f}")


def shapes_by_task(paths: Sequence[Path]) -> Dict[str, Shape]:
    """One induced shape per task, from the alignment collection.

    A task may have several templates; the one with the most rows behind it is taken, because a
    shape induced from six rows is better evidence than one induced from two.
    """
    best: Dict[str, Tuple[int, Shape]] = {}
    for path in paths:
        for group in read_groups(Path(path)):
            if len(group.prompts) < 3:
                continue
            shape = induce(group)
            if shape is None:
                continue
            held = best.get(group.task)
            if held is None or len(group.prompts) > held[0]:
                best[group.task] = (len(group.prompts), shape)
    return {task: shape for task, (_n, shape) in best.items()}


def examine(paths: Sequence[Path], shapes: Optional[Dict[str, Shape]] = None,
            *, learner: Optional[TaskLearner] = None,
            use_shapes: bool = True, name: str = "") -> Report:
    """Learn every task the collection holds, and count what was learned."""
    engine = learner or TaskLearner()
    known = shapes or {}
    out = Report(name=name or ("with shapes" if use_shapes else "whole prompt"))
    for path in paths:
        for task, examples in read_examples(Path(path)):
            answers = {e.answer.strip() for e in examples if e.answer.strip()}
            if len(examples) < 30:
                out.too_few += 1
                continue
            if not 1 < len(answers) <= 8:
                out.free_text += 1
                continue
            shape = known.get(task) if use_shapes else None
            got = engine.learn(examples, shape)
            if got is None:
                out.too_few += 1
                continue
            out.in_scope += 1
            out.results.append(got)
            out.won += int(got.learned_something)
    return out


def run(learn_dir: str, shape_dir: str = "") -> Dict[str, Any]:  # pragma: no cover — a report
    learn_paths = sorted(Path(learn_dir).glob("rows.*.jsonl.gz"))
    if not learn_paths:
        print("no examples; run scripts/collect_task_rows.py --by-task")
        return {}
    shapes: Dict[str, Shape] = {}
    if shape_dir:
        shapes = shapes_by_task(sorted(Path(shape_dir).glob("*.jsonl.gz")))
        print(f"{len(shapes):,} task shapes induced\n")

    with_shape = examine(learn_paths, shapes, use_shapes=True)
    without = examine(learn_paths, shapes, use_shapes=False)
    print("  " + with_shape.render())
    print("  " + without.render())
    print(f"\n  free text, not attempted : {with_shape.free_text:,}")
    print(f"  too few examples         : {with_shape.too_few:,}")
    print(f"  tasks seen               : {with_shape.tasks:,}")

    print("\nwhat she worked out, best first:")
    for got in sorted(with_shape.results, key=lambda r: -(r.accuracy - r.majority))[:20]:
        print(got.render())
        for rule in got.rules[:2]:
            print(f"        {rule.label!r} when {rule.render()}")
    return {"with_shapes": with_shape.to_dict(), "whole_prompt": without.to_dict()}
