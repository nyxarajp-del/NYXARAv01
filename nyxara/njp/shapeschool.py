"""NYXARA · njp/shapeschool.py — is the shape she found the shape that made the row (📏).

This exam needs no answer key and no hand marking, which makes it the strictest one in the
package. A template is induced from four rows of a group. Two rows of that group are held back.
The question asked of each held-out row is:

    Fill the induced template with what it reads out of this row. Do you get the row back —
    character for character?

There is no judgement in that. Either the template plus the fields it recovered reconstruct the
original string exactly, or the shape is wrong about how the row was made. A shape that parses a
row but reconstructs something else has found a pattern that is not the template.

Four things are reported and the last two are the ones that stop the first two being gamed:

* **shaped** — groups that yielded a shape at all. A group that yields none is counted, not hidden.
* **reconstructs** — held-out rows returned exactly by template-plus-fields. The headline.
* **slots** — how many holes the average shape has. A template with one hole covering the whole
  prompt reconstructs *everything* perfectly and has learned nothing, so this is printed beside
  the reconstruction rate and a shape with a single all-consuming slot is counted apart.
* **constant** — how much of the row the template accounts for. The same guard from the other
  side: a shape whose constant text is two characters long is not a template either.

And a floor to beat, because "reconstructs 0.9" means nothing on its own: **one slot** is the
degenerate shape that treats the entire prompt as a hole. It reconstructs perfectly by
construction and knows nothing, and the distance between it and the induced shapes is what the
alignment is worth.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.shapes import MIN_ANCHOR, TRIM, Group, Shape, induce

__all__ = ["Report", "SEED", "LEARN_ROWS", "grade", "examine", "sweep",
           "sweep_trim", "run"]

SEED = 56

#: How many rows of a group the alignment may see. The rest are held back. Four is the fewest that
#: makes a coincidental anchor unlikely while still leaving rows to examine on.
LEARN_ROWS = 4


@dataclass
class Report:
    name: str = ""
    groups: int = 0
    shaped: int = 0
    rows: int = 0
    parsed: int = 0
    rebuilt: int = 0
    slots: int = 0
    constant: int = 0
    whole: int = 0
    spaces: int = 0
    examples: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return round(self.shaped / self.groups, 4) if self.groups else 0.0

    @property
    def reconstructs(self) -> float:
        return round(self.rebuilt / self.rows, 4) if self.rows else 0.0

    @property
    def parses(self) -> float:
        return round(self.parsed / self.rows, 4) if self.rows else 0.0

    @property
    def per_shape(self) -> float:
        return round(self.slots / self.shaped, 3) if self.shaped else 0.0

    @property
    def held(self) -> float:
        """Average characters of the row the template itself accounts for."""
        return round(self.constant / self.shaped, 1) if self.shaped else 0.0

    @property
    def degenerate(self) -> float:
        """Shapes that are one hole and nothing else — perfect and worthless."""
        return round(self.whole / self.shaped, 4) if self.shaped else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "groups": self.groups, "coverage": self.coverage,
                "reconstructs": self.reconstructs, "parses": self.parses,
                "slots_per_shape": self.per_shape, "template_chars": self.held,
                "degenerate": self.degenerate, "with_answer_space": self.spaces}

    def render(self) -> str:
        return (f"{self.name:<16} shaped {self.coverage:.3f}   reconstructs "
                f"{self.reconstructs:.3f}   slots {self.per_shape:.2f}   "
                f"template {self.held:.0f} chars   degenerate {self.degenerate:.3f}")


def _split(group: Group) -> Tuple[Group, Sequence[str]]:
    """Four rows to align on, the rest held back."""
    order = list(range(len(group.prompts)))
    random.Random(SEED).shuffle(order)
    learn = [group.prompts[i] for i in order[:LEARN_ROWS]]
    held = [group.prompts[i] for i in order[LEARN_ROWS:]]
    return Group(task=group.task, template=group.template, source=group.source,
                 prompts=tuple(learn),
                 targets=tuple(group.targets[i] for i in order[:LEARN_ROWS]
                               if i < len(group.targets))), held


def _rebuild(shape: Shape, fields: Sequence[str]) -> str:
    """The template with these fields poured back in."""
    out: List[str] = []
    at = 0
    for part in shape.parts:
        if isinstance(part, str):
            out.append(part)
        else:
            out.append(fields[at] if at < len(fields) else "")
            at += 1
    return "".join(out)


def _one_slot(group: Group) -> Shape:
    """The degenerate shape: the whole prompt is a hole. Perfect, and it knows nothing."""
    return Shape(task=group.task, template=group.template, source=group.source,
                 parts=(0,), slots=(), answers=tuple(group.targets),
                 rows=len(group.prompts))


def grade(groups: Sequence[Group], name: str = "", *, least: int = MIN_ANCHOR,
          degenerate: bool = False, trim: bool = TRIM) -> Report:
    """Induce on four rows of each group and reconstruct the rest."""
    out = Report(name=name)
    for group in groups:
        if len(group.prompts) < LEARN_ROWS + 1:
            continue
        out.groups += 1
        seen, held = _split(group)
        shape = _one_slot(seen) if degenerate else induce(seen, least, trim=trim)
        if shape is None:
            continue
        out.shaped += 1
        out.slots += len(shape.slots) or sum(1 for p in shape.parts if isinstance(p, int))
        out.constant += len("".join(p for p in shape.parts if isinstance(p, str)))
        if len(shape.parts) == 1 and isinstance(shape.parts[0], int):
            out.whole += 1
        if shape.answer_space:
            out.spaces += 1
        for prompt in held:
            out.rows += 1
            fields = shape.read(prompt)
            if fields is None:
                continue
            out.parsed += 1
            if _rebuild(shape, fields) == prompt:
                out.rebuilt += 1
            elif len(out.examples) < 8:
                # Where they diverge, not where they start. Two strings that agree for four
                # hundred characters and differ at the four hundred and first look identical
                # when printed from the front, which is what the first version showed.
                built = _rebuild(shape, fields)
                at = next((i for i, (x, y) in enumerate(zip(built, prompt)) if x != y),
                          min(len(built), len(prompt)))
                out.examples.append((f"…{built[max(0, at - 40):at + 40]!r}",
                                     f"…{prompt[max(0, at - 40):at + 40]!r}"))
    return out


def examine(groups: Sequence[Group]) -> Dict[str, Report]:
    rows = list(groups)
    return {"induced": grade(rows, "induced"),
            "one_slot": grade(rows, "one slot", degenerate=True)}


def sweep(groups: Sequence[Group],
          values: Sequence[int] = (4, 8, 12, 16, 24, 40)) -> List[Tuple[int, Report]]:
    """What the anchor length is worth, so it is a measurement rather than a preference."""
    rows = list(groups)
    return [(value, grade(rows, f"anchor {value}", least=value)) for value in values]


def sweep_trim(groups: Sequence[Group]) -> List[Tuple[bool, Report]]:
    """What shortening a rejected anchor is worth, against dropping it whole.

    The mechanism has to be shown to matter by being taken away, which is what the ``False`` row
    is. If the two rows are the same, the trimming buys nothing and should go.
    """
    rows = list(groups)
    return [(value, grade(rows, f"trim {value}", trim=value)) for value in (False, True)]


def run(path: Optional[Path] = None) -> Dict[str, Any]:  # pragma: no cover — a report
    from nyxara.njp.shapes import read_groups
    if path is None:
        print("give a path to the collector's output")
        return {}
    groups = [g for g in read_groups(Path(path)) if len(g.prompts) >= LEARN_ROWS + 1]
    if not groups:
        print("no groups; run scripts/collect_task_rows.py")
        return {}
    print(f"{len(groups):,} groups with rows to spare\n")
    print("anchor   shaped   reconstructs   slots   template")
    for value, report in sweep(groups):
        print(f"  {value:>3}     {report.coverage:.3f}      {report.reconstructs:.3f}"
              f"      {report.per_shape:.2f}     {report.held:.0f}")
    print()
    got = examine(groups)
    for key in ("one_slot", "induced"):
        print("  " + got[key].render())
    print(f"\nshapes with an induced answer space: {got['induced'].spaces}")
    print("\nwhere the reconstruction failed, at the character it went wrong:")
    for built, prompt in got["induced"].examples[:5]:
        print(f"    rebuilt {built}")
        print(f"    row     {prompt}")
    return {k: v.to_dict() for k, v in got.items()}
