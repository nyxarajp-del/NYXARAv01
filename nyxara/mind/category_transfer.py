"""NYXARA · mind/category_transfer.py — category-theoretic functorial transfer (🏛️, Rule 4, F7).

A discovery in one domain often has an exact twin in another — the same *shape* over different
operations. NYXARA makes that precise with a **functor**: a structure-preserving map between the small
categories her DSL generates (objects = types, morphisms = primitives). A functor here is **identity on
objects** and a **signature-preserving map on morphisms** (each primitive maps to one of the *same*
type, so domains/codomains — hence composition — are preserved *by construction*). Its image of a
learned abstraction is a **zero-shot** candidate concept in the sibling domain.

She never *asserts* an analogy. A transferred abstraction is adopted **only if it verifiably works** —
it type-checks (F9) and the bounded search actually **uses it to solve a held-out sibling task**. When
it cannot be verified she **abstains**: an unverified analogy is a hypothesis, not knowledge.

Concretely: from ``sum(scale_each(x, k))`` the functor ``{scale_each ↦ inc_each}`` (same signature)
yields ``sum(inc_each(x, k))`` — a correct, ready-made concept for a family she has not yet studied,
obtained for free and verified. Pure standard library; **no LLM**; touches no gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.growth.formal_proof import typecheck
from nyxara.growth.noesis import Abstraction, Library, Prog, Task, synthesize

__all__ = ["Functor", "discover_functors", "apply_functor", "TransferResult", "FunctorTransfer"]


# --------------------------------------------------------------------------- #
# functor: identity on objects (types), a signature-preserving map on morphisms
# --------------------------------------------------------------------------- #
@dataclass
class Functor:
    name: str
    morphism_map: Dict[str, str]      # source primitive name -> target primitive name

    def well_defined(self, lib: Library) -> bool:
        """True iff every mapped morphism preserves its domain & codomain (a real functor)."""
        for src, tgt in self.morphism_map.items():
            es, et = lib.entry(src), lib.entry(tgt)
            if es is None or et is None:
                return False
            if tuple(es.arg_types) != tuple(et.arg_types) or es.ret_type != et.ret_type:
                return False
        return True


def apply_functor(prog: Prog, fmap: Dict[str, str]) -> Prog:
    """The functor's image of a program: rename each morphism to its target (structure preserved)."""
    name = fmap.get(prog.name, prog.name) if prog.kind == "app" else prog.name
    if not prog.args:
        return Prog(prog.kind, prog.rtype, name=name, value=prog.value)
    return Prog(prog.kind, prog.rtype, name=name, value=prog.value,
                args=tuple(apply_functor(a, fmap) for a in prog.args))


def discover_functors(lib: Library) -> List[Functor]:
    """Every non-trivial single-morphism swap between primitives of identical signature."""
    by_sig: Dict[Tuple[Any, ...], List[str]] = {}
    for name, p in lib.primitives.items():
        by_sig.setdefault((tuple(p.arg_types), p.ret_type), []).append(name)
    functors: List[Functor] = []
    for names in by_sig.values():
        if len(names) < 2:
            continue
        for a in names:
            for b in names:
                if a != b:
                    functors.append(Functor(f"{a}->{b}", {a: b}))
    return functors


# --------------------------------------------------------------------------- #
# transfer + verification
# --------------------------------------------------------------------------- #
def _uses(name: str, prog: Prog) -> bool:
    if prog.kind == "app" and prog.name == name:
        return True
    return any(_uses(name, a) for a in prog.args)


@dataclass
class TransferResult:
    source: str
    functor: str
    verified: bool
    abstraction: Optional[Abstraction] = None
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "functor": self.functor, "verified": self.verified,
                "abstraction": self.abstraction.name if self.abstraction else None,
                "detail": self.detail}


class FunctorTransfer:
    """Transfer proven abstractions across domains via verified functors (F7)."""

    def __init__(self) -> None:
        self.results: List[TransferResult] = []
        self._n = 0

    def _verify_useful(self, body: Prog, arg_types: Sequence[str], ret_type: str,
                       lib: Library, probes: Sequence[Task]) -> bool:
        """The transferred concept is kept only if search actually USES it to solve a sibling task."""
        from nyxara.growth.grammar import clone_library
        trial = clone_library(lib)
        tmp = "xfer_probe"
        trial.abstractions[tmp] = Abstraction(tmp, tuple(arg_types), ret_type, body)
        for t in probes:
            if t.ret_type != ret_type:
                continue
            res = synthesize(t, trial, max_rounds=3, beam=200, max_expansions=40000)
            if res.solved and res.program is not None and _uses(tmp, res.program):
                return True
        return False

    def transfer(self, source: Abstraction, functor: Functor, lib: Library,
                 probes: Sequence[Task]) -> TransferResult:
        if not functor.well_defined(lib):
            return TransferResult(source.name, functor.name, False, detail="functor not well-defined")
        image_body = apply_functor(source.body, functor.morphism_map)
        if image_body.equal(source.body):
            return TransferResult(source.name, functor.name, False, detail="functor fixes the body")
        if not typecheck(image_body, lib, source.arg_types):
            return TransferResult(source.name, functor.name, False, detail="image ill-typed")
        # already held (syntactically)?  then nothing new
        for a in lib.abstractions.values():
            if a.body.equal(image_body):
                return TransferResult(source.name, functor.name, False, detail="already known")
        if not self._verify_useful(image_body, source.arg_types, source.ret_type, lib, probes):
            return TransferResult(source.name, functor.name, False,
                                  detail="unverified — abstains (hypothesis, not knowledge)")
        name = f"xfer{self._n}"
        self._n += 1
        absn = Abstraction(name, source.arg_types, source.ret_type, image_body)
        return TransferResult(source.name, functor.name, True, absn,
                              detail="verified: search used it to solve a held-out sibling task")

    def transfer_all(self, lib: Library, sources: Sequence[Abstraction],
                     probes: Sequence[Task]) -> List[Abstraction]:
        """Try every source × functor; adopt the verified zero-shot transfers into the library."""
        adopted: List[Abstraction] = []
        functors = discover_functors(lib)
        for src in sources:
            for f in functors:
                # only functors that touch a morphism the source actually uses
                if not any(_uses(m, src.body) for m in f.morphism_map):
                    continue
                r = self.transfer(src, f, lib, probes)
                self.results.append(r)
                if r.verified and r.abstraction is not None:
                    lib.add_abstraction(r.abstraction)
                    adopted.append(r.abstraction)
        return adopted
