"""NYXARA · njp/closureschool.py — is it really new, or is it an identity away from what you have (🔒).

Six candidates whose answer is known before anything runs, and half of them are quantities somebody
would plausibly set out to *discover*: a stride-two difference, a smoothing average. Both are
already there, exactly, and the exam's job is to say so with the identity rather than with a
correlation.

The other half are primitives that break superposition — a median, a rank, a sort. Nothing in the
algebra reaches them, and that is not a compliment: it is a statement about where a real primitive
would have to live, made before any data is looked at.

A module that answered *new* to everything would score three of six here, and so would one that
answered *redundant* to everything. Both halves are needed and neither is decoration.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import median
from typing import Any, Callable, Dict, List, Sequence, Tuple

from nyxara.njp.closure import IDENTITIES, judge, obeys_superposition

__all__ = ["Case", "KNOWN", "sliding", "retrodict", "examine", "run", "SEED", "WINDOW"]

SEED = 88
WINDOW = 3


def sliding(pick: Callable[[List[float]], float], k: int = WINDOW):
    """A window primitive built from any rule for reducing ``k`` numbers to one.

    ``pick`` is where linearity is won or lost: a weighted sum keeps it, anything that sorts or
    compares does not, and :func:`~nyxara.njp.closure.obeys_superposition` finds out by trying
    rather than by reading the name.
    """
    def _run(xs: Sequence[float]) -> List[float]:
        return [pick(list(xs[i:i + k])) for i in range(len(xs) - k + 1)]
    return _run


def _weighted(xs: List[float]) -> float:
    return 0.5 * xs[0] - 1.0 * xs[1] + 0.5 * xs[2]


@dataclass
class Case:
    name: str = ""
    what: Any = None
    #: ``redundant`` — the algebra already contains it. ``nonlinear`` — it is outside the family.
    truth: str = ""
    note: str = ""


KNOWN: Tuple[Case, ...] = (
    Case("the stride-two difference", (1.0, 0.0, -1.0), "redundant",
         "looks like a new primitive; is 4·d − d∘d exactly"),
    Case("a three-tap average", (1.0, 1.0, 1.0), "redundant",
         "smoothing, which is 9 − 6·d + d∘d — the first fixture died on this"),
    Case("a weighted window", sliding(_weighted), "redundant",
         "a callable, but a linear one, so the algebra still owns it — and its taps are read "
         "off it by impulse, never off its name"),
    Case("a sliding median", sliding(median), "nonlinear",
         "sorts, so no stencil reproduces it — this is where a primitive could be new"),
    Case("a sliding largest", sliding(max), "nonlinear", "an order statistic, not a sum"),
    Case("a sliding gap", sliding(lambda xs: max(xs) - min(xs)), "nonlinear",
         "a difference of order statistics; still nothing linear reaches it"),
)


def _look(case: Case) -> Any:
    what = case.what
    if callable(what) and not isinstance(what, (tuple, list)):
        if obeys_superposition(what, rng=random.Random(SEED)):
            # A linear callable: recover its stencil by probing it with unit impulses, then let the
            # algebra decide. Reading the taps off the function rather than off its name is the
            # whole point — a primitive does not get to say what it is.
            taps = _impulse(what)
            return judge(taps, name=case.name)
        return judge(what, name=case.name)
    return judge(what, name=case.name)


def _impulse(primitive: Callable[[Sequence[float]], Sequence[float]], width: int = 8
             ) -> Tuple[float, ...]:
    """Recover a linear primitive's taps by feeding it impulses. Its name is never consulted."""
    out: List[float] = []
    base = primitive([0.0] * width)
    for i in range(width):
        spike = [0.0] * width
        spike[i] = 1.0
        got = primitive(spike)
        if not got:
            break
        out.append(round(got[0] - (base[0] if base else 0.0), 10))
    while out and abs(out[-1]) < 1e-12:
        out.pop()
    return tuple(reversed(out)) if out else ()


def retrodict(cases: Sequence[Case] = KNOWN) -> Dict[str, Any]:
    """Six candidates whose standing is known before anything runs."""
    rows: List[Dict[str, Any]] = []
    right = 0
    flattered = 0     # called something new that the algebra already contains
    buried = 0        # called something redundant that is outside the family
    for case in cases:
        got = _look(case)
        ok = got.stands == case.truth
        right += int(ok)
        flattered += int(got.new and case.truth == "redundant")
        buried += int(not got.new and case.truth == "nonlinear")
        rows.append({"case": case.name, "want": case.truth, "got": got.stands,
                     "identity": got.identity, "residual": got.residual,
                     "note": case.note, "verdict": got})
    return {"rows": rows, "right": right, "of": len(cases),
            "flattered": flattered, "buried": buried,
            "redundant": sum(1 for c in cases if c.truth == "redundant")}


def examine(cases: Sequence[Case] = KNOWN) -> Dict[str, Any]:
    """The exam, with its pass condition written down rather than implied.

    ``flattered`` is the number that matters: a candidate the algebra already contains, reported as
    a discovery. That is the failure this module exists to prevent, and it is the one a search with
    no algebra makes silently every time it rediscovers ``4·d − d∘d`` under a new name.
    """
    got = retrodict(cases)
    got["passes"] = bool(got["right"] == got["of"]
                         and got["flattered"] == 0 and got["buried"] == 0)
    return got


def run() -> Dict[str, Any]:  # pragma: no cover — a report
    got = examine()
    for row in got["rows"]:
        print(f"  {row['case']:<28} want {row['want']:<10} got {row['got']:<10} "
              f"residual {row['residual']:.1e}")
    print(f"\n  right {got['right']}/{got['of']}, flattered {got['flattered']}, "
          f"buried {got['buried']}, passes {got['passes']}\n")
    print("  the identities, written out:")
    for name, taps in IDENTITIES:
        print(judge(taps, name=name).render())
    return got


if __name__ == "__main__":  # pragma: no cover
    run()
