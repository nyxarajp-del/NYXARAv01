"""NYXARA · growth/ecosystem.py — the symbiotic internal ecosystem (🌐, Rule 4, F11).

Instead of one SLEEP loop deciding everything, Noēsis's abstraction learning becomes a small **internal
ecosystem** of three specialised roles in tension — competitive evolution where only robust, efficient,
*proven* concepts survive:

* **Explorer** (high-temperature novelty) — proposes a *wider* candidate pool than the default pairwise
  anti-unification: it also mines **sub-program fragments** (any recurring sub-tree, its constants
  abstracted to holes), so structure that only appears *inside* solutions can still become a concept.
* **Skeptic** (adversarial critic) — tries to break each candidate: it must **type-check** (F9), be
  **non-degenerate** (actually reference the data, not fold to a constant), and have a clean hole
  signature. Whatever the Skeptic breaks never reaches the contest.
* **Synthesizer** (MDL compressor) — the survivors compete on the **held-out description-length** gate
  (the proven `sleep_abstract` rule); only a strict, generalising win is folded into the shared library.

The ecosystem is a thin orchestration over the existing, tested SLEEP machinery — it changes *which
candidates compete*, never the adoption rule. Pure standard library; **no LLM**; touches no gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from nyxara.growth.formal_proof import typecheck
from nyxara.growth.noesis import (
    INT,
    Abstraction,
    Library,
    Prog,
    _candidate_bodies,
    _hole_types,
    hole,
    sleep_abstract,
)

__all__ = ["Explorer", "Skeptic", "Synthesizer", "Ecosystem"]


def _has_var(prog: Prog) -> bool:
    if prog.kind == "var":
        return True
    return any(_has_var(a) for a in prog.args)


def _lits_to_holes(p: Prog) -> Prog:
    """Abstract every integer literal in a sub-tree to a fresh, in-order hole."""
    counter = [0]

    def rec(node: Prog) -> Prog:
        if node.kind == "lit" and node.rtype == INT:
            idx = counter[0]
            counter[0] += 1
            return hole(idx, INT)
        if not node.args:
            return node
        return Prog(node.kind, node.rtype, name=node.name, value=node.value,
                    args=tuple(rec(a) for a in node.args))

    return rec(p)


def _fragments(prog: Prog) -> List[Prog]:
    """Every app sub-tree of ``prog``, with its constants abstracted to holes (a candidate concept)."""
    out: List[Prog] = []

    def walk(p: Prog) -> None:
        if p.kind == "app" and p.size() >= 2:
            frag = _lits_to_holes(p)
            nh = frag.leaves_holes()
            if 1 <= nh <= 2 and frag.has_app() and _has_var(frag):
                out.append(frag)
        for a in p.args:
            walk(a)

    walk(prog)
    return out


class Explorer:
    """Propose a broad candidate pool — pairwise LGG + sub-program fragments (novelty search)."""

    def __init__(self) -> None:
        self.proposed = 0

    def propose(self, corpus: Sequence[Prog]) -> List[Prog]:
        seen: set = set()
        out: List[Prog] = []
        for body in list(_candidate_bodies(corpus)) + [f for p in corpus for f in _fragments(p)]:
            key = body.pretty()
            if key not in seen:
                seen.add(key)
                out.append(body)
        self.proposed = len(out)
        return out


class Skeptic:
    """Break unsound / degenerate candidates before they can compete."""

    def __init__(self) -> None:
        self.rejected = 0

    def accept(self, body: Prog, lib: Library) -> bool:
        arg_types = _hole_types(body)
        ok = (body.has_app()
              and _has_var(body)                                   # must be about the data
              and 1 <= body.leaves_holes() <= 2
              and len(arg_types) == body.leaves_holes()            # clean, contiguous holes
              and typecheck(body, lib, arg_types))                 # F9 well-typed
        if not ok:
            self.rejected += 1
        return ok


class Synthesizer:
    """Fold survivors into the library on the strict held-out MDL gate (the proven SLEEP rule)."""

    def integrate(self, corpus: Sequence[Prog], lib: Library, *, next_id: int, seed: int,
                  explorer: Explorer, skeptic: Skeptic) -> List[Abstraction]:
        return sleep_abstract(corpus, lib, next_id=next_id, seed=seed,
                              candidate_fn=explorer.propose, skeptic=skeptic.accept)


@dataclass
class EcosystemReport:
    adopted: int
    proposed: int
    rejected: int

    def to_dict(self) -> Dict[str, int]:
        return {"adopted": self.adopted, "proposed": self.proposed, "rejected": self.rejected}


class Ecosystem:
    """Explorer → Skeptic → Synthesizer: competitive evolution over abstractions (F11)."""

    def __init__(self) -> None:
        self.explorer = Explorer()
        self.skeptic = Skeptic()
        self.synthesizer = Synthesizer()
        self.last: EcosystemReport = EcosystemReport(0, 0, 0)

    def evolve(self, corpus: Sequence[Prog], lib: Library, *, next_id: int = 0,
               seed: int = 0) -> List[Abstraction]:
        self.explorer.proposed = 0
        self.skeptic.rejected = 0
        adopted = self.synthesizer.integrate(corpus, lib, next_id=next_id, seed=seed,
                                             explorer=self.explorer, skeptic=self.skeptic)
        self.last = EcosystemReport(len(adopted), self.explorer.proposed, self.skeptic.rejected)
        return adopted
