"""NYXARA · growth/grammar.py — programmatic morphogenesis: self-evolving grammar (🧬, Rule 4, F6).

Noēsis's SLEEP grows the library at the **term** level — new reusable programs. F6 grows it at the
**type** level: when the current language simply *cannot express* a problem, NYXARA invents a new
**meta-type** and the primitives that operate on it, expanding her expressive power at its root.

She does not mutate the grammar blindly. From a small, sound **catalog of type-system extensions**
(a product type ``pair``, a nested-list type ``intlistlist``, …) she adopts one **only if it makes
previously-unsolvable held-out tasks solvable** — strictly more capability — and its primitives are
typed, guarded definitions the interpreter's type discipline already enforces (F9). This is genuine
morphogenesis with a verified gate, not caprice.

Pure standard library; **no LLM**; it extends what she can *compute and represent*, never what she
values — an extension can add no character/value symbol (the same ``_FORBIDDEN`` guard as elsewhere).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.growth.noesis import (
    INT,
    INTLIST,
    Library,
    Primitive,
    Task,
    _FORBIDDEN,
    abs_name_tokens,
    synthesize,
)

__all__ = ["TypeExtension", "CATALOG", "install_extension", "is_installed",
           "clone_library", "GrammarEvolver", "meta_tasks"]

PAIR = "pair"
INTLISTLIST = "intlistlist"


# --------------------------------------------------------------------------- #
# meta-type coercers (validate a self-invented type's runtime values)
# --------------------------------------------------------------------------- #
def _coerce_pair(v: Any) -> Optional[Tuple[int, int]]:
    if isinstance(v, (tuple, list)) and len(v) == 2:
        a, b = v
        if isinstance(a, int) and not isinstance(a, bool) and isinstance(b, int) and not isinstance(b, bool):
            return (int(a), int(b))
    return None


def _coerce_intlistlist(v: Any) -> Optional[List[List[int]]]:
    if not isinstance(v, list) or len(v) > 32:
        return None
    out: List[List[int]] = []
    for xs in v:
        if not isinstance(xs, list) or len(xs) > 32:
            return None
        row: List[int] = []
        for x in xs:
            if isinstance(x, int) and not isinstance(x, bool):
                row.append(int(x))
            else:
                return None
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# a type-system extension: new types + the primitives that operate on them
# --------------------------------------------------------------------------- #
@dataclass
class TypeExtension:
    name: str
    types: Dict[str, Callable[[Any], Any]]
    primitives: List[Primitive] = field(default_factory=list)
    motive: str = ""

    def __post_init__(self) -> None:
        tokens = abs_name_tokens(self.name) + [t for ty in self.types for t in abs_name_tokens(ty)]
        if _FORBIDDEN.intersection(tokens):
            raise ValueError(f"grammar extension '{self.name}' touches the character core")


def _pair_extension() -> TypeExtension:
    P = Primitive
    return TypeExtension(
        name="product_type",
        types={PAIR: _coerce_pair},
        primitives=[
            P("mk_pair", (INT, INT), PAIR, lambda a, b: (a, b)),
            P("fst", (PAIR,), INT, lambda p: p[0]),
            P("snd", (PAIR,), INT, lambda p: p[1]),
        ],
        motive="a product type — carry two results at once (e.g. (min, max))",
    )


def _nested_extension() -> TypeExtension:
    P = Primitive
    return TypeExtension(
        name="nested_list_type",
        types={INTLISTLIST: _coerce_intlistlist},
        primitives=[
            P("wrap", (INTLIST,), INTLISTLIST, lambda xs: [xs]),
            P("append_row", (INTLISTLIST, INTLIST), INTLISTLIST, lambda xss, xs: xss + [xs]),
            P("flatten", (INTLISTLIST,), INTLIST, lambda xss: [x for xs in xss for x in xs]),
            P("row_count", (INTLISTLIST,), INT, lambda xss: len(xss)),
        ],
        motive="a nested list — represent grouped/2-D structure",
    )


CATALOG: Tuple[Callable[[], TypeExtension], ...] = (_pair_extension, _nested_extension)


# --------------------------------------------------------------------------- #
# install / inspect / clone
# --------------------------------------------------------------------------- #
def is_installed(lib: Library, ext: TypeExtension) -> bool:
    return all(t in lib.extra_types for t in ext.types)


def install_extension(lib: Library, ext: TypeExtension) -> bool:
    """Extend the library with a new meta-type + its primitives (idempotent)."""
    if is_installed(lib, ext):
        return False
    lib.extra_types.update(ext.types)
    for p in ext.primitives:
        lib.primitives.setdefault(p.name, p)
        lib.counts.setdefault(p.name, 0.0)
    return True


def clone_library(lib: Library) -> Library:
    return Library(primitives=dict(lib.primitives), abstractions=dict(lib.abstractions),
                   counts=dict(lib.counts), archived=dict(lib.archived),
                   extra_types=dict(lib.extra_types))


# --------------------------------------------------------------------------- #
# the evolver: adopt an extension only if it enables new solves
# --------------------------------------------------------------------------- #
class GrammarEvolver:
    """Grow the grammar/type-system, adopting an extension only on a strict capability gain."""

    def _solved_count(self, lib: Library, tasks: Sequence[Task]) -> int:
        return sum(1 for t in tasks if synthesize(t, lib, max_rounds=3, beam=200,
                                                   max_expansions=40000).solved)

    def evolve(self, lib: Library, tasks: Sequence[Task]) -> List[str]:
        """Try each catalog extension; install (on the real lib) those that solve strictly more."""
        if not tasks:
            return []
        adopted: List[str] = []
        baseline = self._solved_count(lib, tasks)
        for make in CATALOG:
            ext = make()
            if is_installed(lib, ext):
                continue
            trial = clone_library(lib)
            install_extension(trial, ext)
            if self._solved_count(trial, tasks) > baseline:
                install_extension(lib, ext)           # strict capability gain → adopt for real
                adopted.append(ext.name)
                baseline = self._solved_count(lib, tasks)
        return adopted


# --------------------------------------------------------------------------- #
# tasks that REQUIRE a meta-type (unsolvable in the base grammar)
# --------------------------------------------------------------------------- #
def meta_tasks(rnd, n: int) -> List[Task]:
    """Fantasise tasks that the base first-order grammar cannot express (they need ``pair``)."""
    from nyxara.growth.noesis import app, evaluate, lit, var, base_library
    lib = base_library()
    install_extension(lib, _pair_extension())
    tasks: List[Task] = []
    # (min, max) as a pair — impossible without the product type
    oracle = app("mk_pair", PAIR, [app("minimum", INT, [var(INTLIST)]),
                                    app("maximum", INT, [var(INTLIST)])])
    for _ in range(n):
        examples = []
        tries = 0
        while len(examples) < 4 and tries < 40:
            tries += 1
            xs = [rnd.randint(-5, 9) for _ in range(rnd.randint(1, 6))]
            out = evaluate(oracle, xs, lib)
            if out is not None:
                examples.append((xs, out))
        if len(examples) >= 3:
            tasks.append(Task("minmax_pair", INTLIST, PAIR, tuple(examples), oracle=oracle))
    return tasks
