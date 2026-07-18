"""NYXARA · cognition/program_library.py — a discrete program library, DreamCoder-style (✷).

A mind that only ever *searches* relearns the same structure on every task. DreamCoder's
insight was to split learning in two: a **search** proposes verified programs for one task at a
time, and a separate **library-learning** step looks across everything solved so far, finds the
program fragments that keep recurring, and **compiles them into new, permanent, reusable
primitives** — so the *next* search starts from a richer vocabulary, not the same fixed one.
Compositional generalization comes from that growing, reused library, not from throwing more
search budget (scale) at each task.

NYXARA already had real, verified program synthesis
(:class:`~nyxara.cognition.skill_induction.SkillInductionEngine`: breadth-first search over a
small string-op DSL, every candidate checked against every demonstration) — but every task was
searched from the *same fixed* primitive set, and a solved task's program was cached under its
own name and never fed back into the vocabulary. This module closes that loop:

* **wake** (:meth:`ProgramLibrary.induce`) — hand a task to the wrapped engine, whose search now
  also tries every macro the library has learned so far, ranked by a small learned
  :class:`PrimitiveScorer` (a minimal, honest, non-LLM stand-in for DreamCoder's neural
  recognition model — it never proposes programs itself, only learns which of the vocabulary is
  worth trying first from what has actually worked before);
* **sleep — abstraction** (:meth:`ProgramLibrary.compress`) — look across every verified
  :class:`~nyxara.cognition.skill_induction.InductiveSkill` solved so far for programs that share
  the same *shape* (the same operation names, in the same order) across **independently solved**
  tasks. Where those programs' literal parameters agree, anti-unify them
  (:func:`nyxara.cognition.abstraction.anti_unify`, over a :mod:`nyxara.mind.lot` term encoding)
  to find what is permanently fixed and what remains a hole re-induced per task — an honest,
  structural discovery, not a heuristic guess;
* **sleep — compression** (:meth:`ProgramLibrary.promote`) — make a mined pattern **permanent**:
  register it as a first-class, nameable :class:`Macro`, persist it through the
  :class:`~nyxara.memory.store.MemoryStore` (so every future session's search starts from an
  ever-growing library, never a fixed one), and wire it into the live engine as one more
  candidate closing op.

The payoff is concrete and testable: a task that needs two verified search steps the first few
times it is solved can close in **one** step forever after, because the *library* — not the
search budget — now carries the compressed primitive. See the ``__main__`` self-test: a depth-0
(single-step-only) search, which cannot possibly discover a genuine two-step program on its own,
solves one anyway once the library carries the compressed macro — including across a fresh
process, from macros rehydrated out of the store.

Pure standard library. Skill/knowledge only — never touches the loyalty/corrigibility/honesty
core.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.cognition.abstraction import anti_unify
from nyxara.cognition.skill_induction import (
    _ATOMS, _INDUCERS_BY_OP, _MACRO_REGISTRY, InductiveSkill, Operation, SkillInductionEngine,
)
from nyxara.mind.lot import Const, Function

__all__ = ["Macro", "LibraryReport", "PrimitiveScorer", "ProgramLibrary", "apply_macro"]


# --------------------------------------------------------------------------- #
# The permanent library primitive
# --------------------------------------------------------------------------- #
@dataclass
class Macro:
    """One permanent, symbolic library primitive — a compressed program fragment.

    Mined from >=2 *independently solved* tasks whose programs turned out to share the same
    shape (the same op names, same order): positions where their induced literal parameters
    agreed become permanently ``fixed_params`` (baked into the macro); positions where they
    disagreed stay ``hole_param_keys`` — free parameters re-induced from evidence each time the
    macro is tried on a new task, exactly like the base inducer it wraps would.
    """

    name: str
    atom_prefix: Tuple[str, ...]
    closing_op_name: str
    param_keys: Tuple[str, ...] = ()
    fixed_params: Dict[str, str] = field(default_factory=dict)
    hole_param_keys: Tuple[str, ...] = ()
    provenance: List[str] = field(default_factory=list)
    support: int = 0
    created_at: float = field(default_factory=time.time)
    uses: int = 0

    def describe(self) -> str:
        prefix = " ∘ ".join(self.atom_prefix)
        closing = self.closing_op_name
        if self.fixed_params:
            fixed = ", ".join(f"{k}={v!r}" for k, v in self.fixed_params.items())
            closing = f"{closing}({fixed})"
        chain = " ∘ ".join(x for x in (prefix, closing) if x)
        return f"macro {self.name!r}: {chain}  (from {self.support} task(s): {', '.join(self.provenance)})"

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "atom_prefix": list(self.atom_prefix),
                "closing_op_name": self.closing_op_name, "param_keys": list(self.param_keys),
                "fixed_params": dict(self.fixed_params),
                "hole_param_keys": list(self.hole_param_keys),
                "provenance": list(self.provenance), "support": self.support,
                "created_at": self.created_at, "uses": self.uses}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Macro":
        return cls(name=str(d.get("name", "macro")),
                   atom_prefix=tuple(d.get("atom_prefix", [])),
                   closing_op_name=str(d.get("closing_op_name", "")),
                   param_keys=tuple(d.get("param_keys", [])),
                   fixed_params=dict(d.get("fixed_params", {})),
                   hole_param_keys=tuple(d.get("hole_param_keys", [])),
                   provenance=list(d.get("provenance", [])),
                   support=int(d.get("support", 0)),
                   created_at=float(d.get("created_at", time.time())),
                   uses=int(d.get("uses", 0)))


# --------------------------------------------------------------------------- #
# Registry — resolving a "macro" Operation at apply() time
# --------------------------------------------------------------------------- #
# The registry dict itself lives in skill_induction.py (imported above): Operation.apply reads
# it directly there, which also keeps a single canonical copy no matter how this module is
# loaded (its own __main__ self-test included — see the comment at its definition).
def apply_macro(macro_name: str, params: Dict[str, Any], text: str) -> str:
    """Replay a registered :class:`Macro`'s compressed program on ``text``.

    A thin convenience wrapper around :meth:`~nyxara.cognition.skill_induction.Operation.apply`
    for callers outside the engine's own search/apply path. An unknown/unregistered macro name
    is inert (returns ``text`` unchanged).
    """
    return Operation("macro", {**params, "macro": macro_name}).apply(text)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class LibraryReport:
    macros_before: int = 0
    macros_after: int = 0
    new_macros: List[str] = field(default_factory=list)
    skills_examined: int = 0
    reason: str = ""

    @property
    def grown(self) -> bool:
        return self.macros_after > self.macros_before

    def to_dict(self) -> Dict[str, Any]:
        return {"macros_before": self.macros_before, "macros_after": self.macros_after,
                "new_macros": list(self.new_macros), "skills_examined": self.skills_examined,
                "reason": self.reason}


# --------------------------------------------------------------------------- #
# The learned search guide — deliberately small and honest about what it is
# --------------------------------------------------------------------------- #
class PrimitiveScorer:
    """A minimal learned preference over primitive/macro names — NYXARA's neural-*lite* search
    guide.

    One scalar weight per name, pulled toward +1 whenever that primitive took part in a
    verified solution and left to decay toward 0 otherwise. It never proposes programs itself
    (the breadth-first search in :mod:`skill_induction` does that) — it only learns, from
    outcomes, which member of the library's growing vocabulary is worth trying *first*. That is
    the whole and honest scope of the "neural" half of this wake/sleep loop: no LLM, no opaque
    black box, a single trainable weight vector any part of NYXARA can inspect. Pure stdlib.
    """

    def __init__(self, *, lr: float = 0.2, decay: float = 0.01) -> None:
        self.weights: Dict[str, float] = {}
        self.lr = float(lr)
        self.decay = float(decay)

    def score(self, name: str) -> float:
        return self.weights.get(name, 0.0)

    def reinforce(self, names: Sequence[str]) -> None:
        """Pull every name in ``names`` toward +1; everything else decays gently toward 0."""
        seen = set(names)
        for n in seen:
            w = self.weights.get(n, 0.0)
            self.weights[n] = max(-1.0, min(1.0, w + self.lr * (1.0 - w)))
        for n in list(self.weights):
            if n not in seen:
                self.weights[n] *= (1.0 - self.decay)

    def to_dict(self) -> Dict[str, float]:
        return dict(self.weights)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PrimitiveScorer":
        s = cls()
        s.weights = {str(k): float(v) for k, v in dict(d or {}).items()}
        return s


# --------------------------------------------------------------------------- #
# The library
# --------------------------------------------------------------------------- #
class ProgramLibrary:
    """NYXARA's discrete program library — the DreamCoder-style wake/sleep loop, made real.

    Wraps a :class:`~nyxara.cognition.skill_induction.SkillInductionEngine`: :meth:`induce`
    forwards to it (empowered by whatever the library has learned so far and by the learned
    :class:`PrimitiveScorer`), and :meth:`consolidate` periodically mines its corpus of verified
    solutions for recurring structure and permanently compiles it into new primitives. Persists
    through the same :class:`~nyxara.memory.store.MemoryStore` the rest of NYXARA's long-term
    memory uses, so the library survives restarts and keeps growing across her whole lifetime.
    """

    TAG = "program_macro"

    def __init__(self, *, engine: Optional[SkillInductionEngine] = None, store: Any = None,
                 min_support: int = 2) -> None:
        self.engine = engine
        self.store = store if store is not None else getattr(engine, "store", None)
        self.min_support = max(2, int(min_support))
        self.scorer = PrimitiveScorer()
        self._macros: Dict[str, Macro] = {}
        self._signatures: set = set()
        self._hydrated = False
        if self.engine is not None:
            self.engine.set_bias(self.scorer.score)

    # ---- wake: search, empowered by the library ---- #
    def induce(self, name: str, pairs: Sequence[Tuple[str, str]]) -> Optional[InductiveSkill]:
        """Induce a skill through the wrapped engine, then reinforce the primitives it used."""
        self._hydrate()
        if self.engine is None:
            return None
        skill = self.engine.induce(name, pairs)
        if skill is not None:
            self.scorer.reinforce([op.name for op in skill.program])
            for op in skill.program:
                if op.name == "macro":
                    m = self._macros.get(str(op.params.get("macro", "")))
                    if m is not None:
                        m.uses += 1
        return skill

    # ---- sleep / abstraction: mine recurring shapes into candidate macros ---- #
    def compress(self, skills: Sequence[InductiveSkill], *,
                 min_support: Optional[int] = None) -> List[Macro]:
        """Mine :class:`Macro`\\ s from programs that recur, in the same shape, across at least
        ``min_support`` *distinct* solved tasks. Deterministic and verified — a shape is only
        ever compressed on evidence actually present in the corpus."""
        min_support = self.min_support if min_support is None else max(2, int(min_support))
        groups: Dict[Tuple[str, ...], List[InductiveSkill]] = {}
        for sk in skills:
            if not sk.program:
                continue
            groups.setdefault(tuple(op.name for op in sk.program), []).append(sk)
        found: List[Macro] = []
        for shape, group in groups.items():
            names = sorted({sk.name for sk in group})
            if len(names) < min_support:
                continue
            macro = self._build_macro(shape, group, names)
            if macro is not None:
                found.append(macro)
        return found

    def _build_macro(self, shape: Tuple[str, ...], group: List[InductiveSkill],
                     names: List[str]) -> Optional[Macro]:
        closing_ops = [sk.program[-1] for sk in group]
        keys = list(closing_ops[0].params.keys())
        if any(list(op.params.keys()) != keys for op in closing_ops):
            return None                                  # inconsistent param shape — defensive
        atom_prefix = shape[:-1]
        fixed_params: Dict[str, str] = {}
        hole_keys: List[str] = []
        if keys:
            # encode each task's closing op as a LoT term and fold anti-unify (LGG) across the
            # group: positions where every term agrees stay a Const (fixed); any disagreement
            # collapses that position to a fresh Var (a hole) — real structural discovery.
            terms = [Function(op.name, tuple(Const(str(op.params[k])) for k in keys))
                     for op in closing_ops]
            lgg = terms[0]
            for t in terms[1:]:
                lgg = anti_unify(lgg, t)
            if not isinstance(lgg, Function):
                return None
            for k, arg in zip(keys, lgg.args):
                if isinstance(arg, Const):
                    fixed_params[k] = arg.name
                else:
                    hole_keys.append(k)
        if not fixed_params and not atom_prefix:
            return None                        # no compression over an already-existing primitive
        sig = (shape, tuple(sorted(fixed_params.items())))
        if sig in self._signatures:
            return None                        # this exact pattern is already a library primitive
        # globally unique — _MACRO_REGISTRY is process-wide, so two different ProgramLibrary
        # instances (e.g. two SampleEfficientMinds) must never mint the same macro name and
        # silently overwrite each other's registered primitive.
        macro_name = f"macro_{closing_ops[0].name}_{uuid.uuid4().hex[:8]}"
        return Macro(name=macro_name, atom_prefix=atom_prefix, closing_op_name=closing_ops[0].name,
                    param_keys=tuple(keys), fixed_params=fixed_params,
                    hole_param_keys=tuple(hole_keys), provenance=names, support=len(names))

    # ---- sleep / compression: make a mined pattern permanent ---- #
    def promote(self, macro: Macro) -> None:
        """Register, persist, and wire ``macro`` into live search — the compile step."""
        self._macros[macro.name] = macro
        self._signatures.add((macro.atom_prefix + (macro.closing_op_name,),
                              tuple(sorted(macro.fixed_params.items()))))
        _MACRO_REGISTRY[macro.name] = macro
        if self.engine is not None:
            self.engine.add_inducer(self._inducer_for(macro))
        self.scorer.weights.setdefault(macro.name, 0.3)   # a mild prior: reuse over rediscovery
        self._persist(macro)

    def consolidate(self) -> LibraryReport:
        """One sleep-abstraction pass: mine the current skill corpus, promote what's new."""
        self._hydrate()
        report = LibraryReport(macros_before=len(self._macros))
        if self.engine is None:
            report.reason = "no engine bound"
            return report
        skills = self.engine.skills()
        report.skills_examined = len(skills)
        for macro in self.compress(skills):
            self.promote(macro)
            report.new_macros.append(macro.name)
        report.macros_after = len(self._macros)
        report.reason = (f"promoted {len(report.new_macros)} new macro(s) from "
                         f"{report.skills_examined} skill(s)" if report.new_macros
                         else "no new compressible pattern")
        return report

    def maybe_run(self) -> Optional[Dict[str, Any]]:
        """Idle-cycle entry point: run one consolidation pass. Mirrors the rest of the growth
        loop's ``maybe_run`` convention — cheap, best-effort, never raises."""
        try:
            return self.consolidate().to_dict()
        except Exception as exc:  # noqa: BLE001 — a failed pass reports, never crashes idle
            return {"macros_after": len(self._macros), "reason": f"consolidation failed: {exc}"}

    # ---- the inducer a promoted macro contributes to future search ---- #
    def _inducer_for(self, macro: Macro):
        base = _INDUCERS_BY_OP.get(macro.closing_op_name)

        def inducer(xs: Sequence[str], ys: Sequence[str]) -> Optional[Operation]:
            states = list(xs)
            for op_name in macro.atom_prefix:
                atom = _ATOMS.get(op_name)
                if atom is None:
                    return None
                states = [atom(s) for s in states]
            if not macro.param_keys:
                atom = _ATOMS.get(macro.closing_op_name)
                if atom is None:
                    return None
                if all(atom(s) == y for s, y in zip(states, ys)):
                    return Operation("macro", {"macro": macro.name})
                return None
            if base is None:
                return None
            induced = base(states, ys)
            if induced is None:
                return None
            for k, v in macro.fixed_params.items():
                if str(induced.params.get(k)) != str(v):
                    return None                            # this task doesn't match the macro
            params: Dict[str, Any] = {"macro": macro.name}
            params.update(macro.fixed_params)
            params.update({k: induced.params.get(k) for k in macro.hole_param_keys})
            return Operation("macro", params)

        inducer.macro_name = macro.name                    # lets bias-based ordering find it
        return inducer

    # ---- introspection ---- #
    def macros(self) -> List[Macro]:
        self._hydrate()
        return list(self._macros.values())

    def stats(self) -> Dict[str, Any]:
        self._hydrate()
        return {"macros": len(self._macros),
                "total_uses": sum(m.uses for m in self._macros.values())}

    def __len__(self) -> int:
        self._hydrate()
        return len(self._macros)

    # ---- persistence ---- #
    def _persist(self, macro: Macro) -> None:
        if self.store is None:
            return
        try:
            from nyxara.memory.store import MemoryType
            self.store.remember(
                macro.describe(), mem_type=MemoryType.PROCEDURAL, importance=0.7,
                tags=[self.TAG], metadata={"macro": macro.to_dict(), "name": macro.name})
        except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
            pass

    def _hydrate(self) -> None:
        if self._hydrated:
            return
        self._hydrated = True
        if self.store is None:
            return
        try:
            hits = self.store.recall("", k=1000, tags=[self.TAG], strengthen=False)
            for rec, _ in hits:
                data = rec.metadata.get("macro") if isinstance(rec.metadata, dict) else None
                if not isinstance(data, dict):
                    continue
                macro = Macro.from_dict(data)
                if macro.name in self._macros:
                    continue
                self._macros[macro.name] = macro
                self._signatures.add((macro.atom_prefix + (macro.closing_op_name,),
                                      tuple(sorted(macro.fixed_params.items()))))
                _MACRO_REGISTRY[macro.name] = macro
                if self.engine is not None:
                    self.engine.add_inducer(self._inducer_for(macro))
        except Exception:  # noqa: BLE001 — hydration is best-effort
            pass


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA discrete program library (DreamCoder-style wake/sleep) self-test")
    print("=" * 70)

    # ---- wake: solve 3 INDEPENDENT tasks that happen to share a 2-step shape ---- #
    # reverse_words, then wrap with a FIXED prefix and a VARYING suffix
    engine = SkillInductionEngine(max_depth=1, min_demos=2)
    lib = ProgramLibrary(engine=engine, min_support=3)

    a = lib.induce("shout", [("hello world", "RESULT: world hello!"),
                             ("go now", "RESULT: now go!")])
    b = lib.induce("ask", [("foo bar", "RESULT: bar foo?"),
                           ("up down", "RESULT: down up?")])
    c = lib.induce("state", [("cat dog mouse", "RESULT: mouse dog cat."),
                             ("red blue", "RESULT: blue red.")])
    assert a is not None and b is not None and c is not None
    shape = [op.name for op in a.program]
    print(f"\nsolved 3 tasks      : shape={shape}")
    assert shape == ["reverse_words", "affix"], f"unexpected shape {shape}"
    assert [op.name for op in b.program] == shape
    assert [op.name for op in c.program] == shape

    # ---- sleep-abstraction + compression: mine the shared shape into a permanent macro ---- #
    report = lib.consolidate()
    print(f"consolidate         : {report.to_dict()}")
    assert report.grown and len(lib.macros()) == 1
    macro = lib.macros()[0]
    print(f"promoted            : {macro.describe()}")
    assert macro.atom_prefix == ("reverse_words",)
    assert macro.fixed_params.get("prefix") == "RESULT: "
    assert "suffix" in macro.hole_param_keys, "suffix varied across tasks -> must stay a hole"

    # ---- THE ACID TEST ----
    # a depth-0 (single-step-only) engine CANNOT find a genuine 2-step program on its own...
    never_seen_demos = [("m n", "RESULT: n m#"), ("j k", "RESULT: k j#")]
    bare = SkillInductionEngine(max_depth=0, min_demos=2)
    unsolved = bare.induce("never_seen_bare", never_seen_demos)
    print(f"\ndepth-0, no library : {unsolved}")
    assert unsolved is None, "a depth-0 search must NOT be able to invent a 2-step program"

    # ...but once the SAME macro is handed to a depth-0 engine, it solves the never-seen task
    # in ONE step — real compositional reuse, not more search budget.
    shallow_engine = SkillInductionEngine(max_depth=0, min_demos=2)
    shallow_lib = ProgramLibrary(engine=shallow_engine)
    for m in lib.macros():
        shallow_lib.promote(m)
    solved = shallow_lib.induce("never_seen", never_seen_demos)
    print(f"depth-0, WITH library: {solved.render() if solved else None}")
    assert solved is not None, "the compressed macro must let a depth-0 search close in one step"
    assert solved.program[0].name == "macro"
    # generalizes to a HELD-OUT input never shown to induce() at all
    assert solved.apply("p q") == "RESULT: q p#", "the hole (suffix) generalises to a new input"
    print("acid test           : depth-0 search solves a 2-step task via the library ✓")

    # ---- persistence: the library survives a restart and keeps empowering search ---- #
    from nyxara.memory.store import MemoryStore

    store = MemoryStore()
    # (two-word inputs with a genuine ORDER SWAP — the template inducer can fill one captured
    # slot into a fixed frame in a single step, but it cannot re-order two, so these still need
    # the real reverse_words + affix pair, exactly like the tasks above.)
    e1 = SkillInductionEngine(max_depth=1, min_demos=2, store=store)
    lib1 = ProgramLibrary(engine=e1, store=store, min_support=2)
    lib1.induce("greetA", [("alpha beta", "LOUD: beta alpha!!"),
                           ("gamma delta", "LOUD: delta gamma!!")])
    lib1.induce("greetB", [("kappa lambda", "LOUD: lambda kappa??"),
                           ("sigma tau", "LOUD: tau sigma??")])
    rep1 = lib1.consolidate()
    assert rep1.grown, "two independently-solved same-shape tasks should compress"
    print(f"\nsession 1 macro     : {lib1.macros()[0].describe()}")

    e2 = SkillInductionEngine(max_depth=0, min_demos=2, store=store)   # a FRESH, shallow engine
    lib2 = ProgramLibrary(engine=e2, store=store)                      # a FRESH library instance
    solved2 = lib2.induce("greetC", [("omega psi", "LOUD: psi omega::"),
                                     ("chi phi", "LOUD: phi chi::")])
    print(f"session 2 (restart) : {solved2.render() if solved2 else None}")
    assert solved2 is not None and solved2.program[0].name == "macro", (
        "a fresh process, over the same store, must rehydrate the macro and reuse it")
    assert solved2.apply("nu xi") == "LOUD: xi nu::"
    print("persisted reuse     : macro survives a restart and solves a depth-0 task ✓")

    print("\nALL SELF-TESTS PASSED ✓")
