"""NYXARA · mind/domain_genesis.py — domain mastery *from scratch* by her OWN faculties (⚙⇶).

The critique this module answers: *"NYXARA masters a new domain only when its structure maps
onto an old one. On a genuinely ALIEN problem — whose relational topology corresponds to none of
her known domains — she honestly falls back to the base LLM."* That was true. Her
:mod:`nyxara.mind.transfer` engine is cross-domain: it can only fire when the alien structure
aligns with one of the ~9 seed domains. Nothing built a *fresh* theory of a field from the
field's **own** internal regularities.

:class:`DomainGenesisEngine` closes that gap. When transfer-to-a-known-base declines, this does
what a scientist does with a truly new system: it reads the alien domain's own relational
skeleton and **induces the laws that hold *within* it** — transitivity/chaining, feedback loops,
symmetry, and inverse relations — then **projects held-out predictions by closure over those
induced laws**. The reasoning content is hers, derived by a deterministic algorithm from the
problem's own stated structure — not sampled from a language model. With no LLM at all, the
induced theory *is* the answer.

Deliberately honest about its reach:

* It only fires when it can recover **≥ ``min_relations`` real relations** from the query *and*
  induce at least one law that projects a genuinely new inference. On free-form chat with no
  extractable structure it returns ``None`` and the normal LLM path runs — no faked mastery.
* Every projected inference is a *conjecture from an induced internal law* ("so it likely
  follows that …"), surfaced as such. Confidence is held moderate and always below a verified
  faculty, so it never overclaims.
* On success it **distils the alien domain into a** :class:`~nyxara.mind.transfer.DomainSchema`
  **and learns it** into the shared, memory-backed store — so the field is recognised next time
  and can itself serve as a transfer base. Genuine growth from scratch, by her own action
  (Rule 4).

Pure standard library. Reuses the shared open-domain extractor
(:func:`nyxara.mind.transfer.extract_relations`), the Language of Thought
(:mod:`nyxara.mind.lot`), and :func:`nyxara.mind.analogy.relation`/``entity``. Wired as a stage
of the unified :class:`nyxara.mind.generalization.GeneralizationEngine`, so it reaches both the
conversational router and the domain solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.mind.analogy import entity, relation
from nyxara.mind.lot import Const, Predicate, Term

__all__ = [
    "DomainLaw",
    "DomainTheory",
    "GenesisResult",
    "DomainGenesisEngine",
]

# Surface words that never name a domain — used only for labelling a distilled field.
_STOP = frozenset("""
a an the of to in on at by for with from into onto over under and or but if then than as is are
was were be been being do does did will would can could should may might must this that these
those it its their there here what which who whom whose why how when where some any each every
both all no not more most much many few this device system thing then continuously
""".split())


# --------------------------------------------------------------------------- #
# A binary relation as a directed edge (left → right), for law induction
# --------------------------------------------------------------------------- #
def _binary(p: Predicate) -> Optional[Tuple[str, str, str]]:
    """``rel(a, b)`` over two entities → ``(rel_name, a, b)``; anything else → ``None``."""
    if not isinstance(p, Predicate) or len(p.args) != 2:
        return None
    a, b = p.args
    if isinstance(a, Const) and isinstance(b, Const):
        return (p.name, a.name, b.name)
    return None


# --------------------------------------------------------------------------- #
# Induced laws + the theory they form
# --------------------------------------------------------------------------- #
@dataclass
class DomainLaw:
    """One regularity induced from the alien domain's *own* internal structure."""

    kind: str          # "transitivity" | "feedback_loop" | "symmetry" | "inverse" | "hub"
    statement: str     # human-readable law
    relation: str = ""  # the relation name(s) it concerns
    support: int = 1    # how many instances in the domain corroborate it

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "statement": self.statement,
                "relation": self.relation, "support": int(self.support)}


@dataclass
class DomainTheory:
    """A self-consistent model of a genuinely new field, built from its own structure."""

    label: str
    relations: List[Predicate] = field(default_factory=list)
    laws: List[DomainLaw] = field(default_factory=list)
    inferences: List[Predicate] = field(default_factory=list)  # held-out projections

    def entities(self) -> List[str]:
        seen: List[str] = []
        for p in self.relations:
            b = _binary(p)
            if b:
                for e in (b[1], b[2]):
                    if e not in seen:
                        seen.append(e)
        return seen

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "relations": [str(p) for p in self.relations],
            "laws": [law.to_dict() for law in self.laws],
            "inferences": [str(p) for p in self.inferences],
            "entities": self.entities(),
        }


@dataclass
class GenesisResult:
    """The answer NYXARA derived from a domain theory she built from scratch."""

    answer: str
    confidence: float
    theory: DomainTheory
    analogical: bool = True   # induced conjecture, not a strict verified fact
    detail: Dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        return self.answer

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "confidence": round(float(self.confidence), 3),
            "analogical": bool(self.analogical),
            "theory": self.theory.to_dict(),
            "detail": dict(self.detail),
        }


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
class DomainGenesisEngine:
    """Master a genuinely alien domain from its OWN internal structure — no known base, no LLM.

    ``min_relations`` is the least extractable structure accepted; below it the engine declines.
    ``min_confidence`` gates whether the result is surfaced at all. When a ``transfer_engine`` is
    supplied, a successfully-modelled domain is distilled and learned into its shared, memory-
    backed :class:`~nyxara.mind.transfer.DomainSchemaStore`, so the field is recognised next time.
    """

    def __init__(self, *, transfer_engine: Any = None, min_relations: int = 2,
                 min_confidence: float = 0.4, learn_from_experience: bool = True) -> None:
        self.transfer_engine = transfer_engine
        self.min_relations = max(2, int(min_relations))
        self.min_confidence = float(min_confidence)
        self.learn_from_experience = bool(learn_from_experience)

    # ---- public entry point ---- #
    def generalize(self, query: str) -> Optional[GenesisResult]:
        """Build a from-scratch theory of ``query``'s domain and answer from it, or ``None``.

        Returns ``None`` (so the LLM path runs) when there is too little structure to model, or
        when no induced law projects a genuinely new inference — never a faked mastery."""
        theory = self.build(query)
        if theory is None or not theory.inferences:
            return None
        conf = self._confidence(theory)
        if conf < self.min_confidence:
            return None
        # grow: a field modelled from scratch is worth keeping so it is mastered next time.
        if self.learn_from_experience:
            self._learn(theory)
        answer = self._render(theory)
        return GenesisResult(answer=answer, confidence=conf, theory=theory,
                             detail={"own_faculty": "domain_genesis",
                                     "laws": [law.kind for law in theory.laws],
                                     "learned_field": theory.label})

    # ---- construct the theory ---- #
    def build(self, query: str) -> Optional[DomainTheory]:
        """Extract the alien domain's relations, induce its own laws, project held-out facts."""
        rels = self._extract(query)
        if len(rels) < self.min_relations:
            return None
        label = self._label(query, rels)
        laws, inferences = self._induce(rels)
        return DomainTheory(label=label, relations=rels, laws=laws, inferences=inferences)

    def _extract(self, query: str) -> List[Predicate]:
        """Open-domain relational skeleton via the shared extractor (novel verbs welcome)."""
        try:
            from nyxara.mind.transfer import extract_relations
            return extract_relations(query)  # pure open-domain (no known vocab)
        except Exception:  # noqa: BLE001 — extraction is best-effort, never fatal
            return []

    # ---- law induction from the domain's OWN internal structure ---- #
    def _induce(self, rels: Sequence[Predicate]) -> Tuple[List[DomainLaw], List[Predicate]]:
        edges = [b for b in (_binary(p) for p in rels) if b is not None]
        stated = {(n, a, b) for (n, a, b) in edges}
        laws: List[DomainLaw] = []
        inferences: List[Predicate] = []
        seen_inf: set = set()

        def add_inf(p: Predicate) -> None:
            if str(p) not in seen_inf:
                seen_inf.add(str(p))
                inferences.append(p)

        # per-relation adjacency, for transitivity and symmetry
        by_rel: Dict[str, List[Tuple[str, str]]] = {}
        for (n, a, b) in edges:
            by_rel.setdefault(n, []).append((a, b))

        # 1) TRANSITIVITY / chaining: R(a,b) ∧ R(b,c) ⇒ R(a,c) (when unstated)
        for name, pairs in by_rel.items():
            succ: Dict[str, List[str]] = {}
            for a, b in pairs:
                succ.setdefault(a, []).append(b)
            chains = 0
            for a, b in pairs:
                for c in succ.get(b, []):
                    if a != c and (name, a, c) not in stated:
                        add_inf(relation(name, entity(a), entity(c)))
                        chains += 1
            if chains:
                laws.append(DomainLaw("transitivity",
                                      f"'{name}' chains through the domain (a→b, b→c ⇒ a→c)",
                                      relation=name, support=chains))

        # 2) SYMMETRY: R(a,b) ∧ R(b,a) ⇒ R is symmetric; project the missing mirror elsewhere
        for name, pairs in by_rel.items():
            pset = set(pairs)
            if any((b, a) in pset for (a, b) in pairs):
                laws.append(DomainLaw("symmetry", f"'{name}' is symmetric (holds both ways)",
                                      relation=name, support=1))
                for a, b in pairs:
                    if (b, a) not in pset and a != b and (name, b, a) not in stated:
                        add_inf(relation(name, entity(b), entity(a)))

        # 3) INVERSE: R(a,b) ∧ S(b,a) with R≠S ⇒ S is the converse of R; project S over other R.
        #    Each unordered {R, S} pair yields one converse law (deduped), not two mirror copies.
        inverse_seen: set = set()
        for (rn, ra, rb) in edges:
            for (sn, sa, sb) in edges:
                if rn != sn and (sa, sb) == (rb, ra):
                    key = tuple(sorted((rn, sn)))
                    if key not in inverse_seen:
                        inverse_seen.add(key)
                        laws.append(DomainLaw(
                            "inverse",
                            f"'{sn}' is the converse of '{rn}' (S(b,a) mirrors R(a,b))",
                            relation=f"{rn}/{sn}", support=1))
                    for (n2, a2, b2) in edges:
                        if n2 == rn and (sn, b2, a2) not in stated and a2 != b2:
                            add_inf(relation(sn, entity(b2), entity(a2)))

        # 4) FEEDBACK LOOP: a cycle over the (name-agnostic) relational graph is self-sustaining —
        #    project the higher-order closure (the last step sustains the first), the systematic
        #    glue every seed schema carries, here induced from the alien domain itself.
        cycle = self._find_cycle(edges)
        if cycle:
            first, last = cycle[0], cycle[-1]
            fp = relation(first[0], entity(first[1]), entity(first[2]))
            lp = relation(last[0], entity(last[1]), entity(last[2]))
            closure = relation("sustains", lp, fp)
            add_inf(closure)
            names = "→".join(e[1] for e in cycle) + "→" + cycle[-1][2]
            laws.append(DomainLaw("feedback_loop",
                                  f"the domain closes a self-sustaining loop ({names})",
                                  relation="sustains", support=len(cycle)))

        # 5) HUB / driver: the highest-degree entity governs the domain (descriptive, not projected
        #    as a fabricated edge — kept honest by leaving it out of `inferences`).
        degree: Dict[str, int] = {}
        for (_n, a, b) in edges:
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
        if degree:
            hub = max(degree, key=lambda e: degree[e])
            if degree[hub] >= 3:
                laws.append(DomainLaw("hub", f"'{hub}' is the central hub of the domain "
                                             f"(participates in {degree[hub]} relations)",
                                      relation="", support=degree[hub]))
        return laws, inferences

    @staticmethod
    def _find_cycle(edges: Sequence[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """Return one directed cycle as an ordered edge list (empty if the graph is acyclic).

        Name-agnostic: a loop of *any* relations closing back on itself signals feedback."""
        adj: Dict[str, List[Tuple[str, str, str]]] = {}
        for e in edges:
            adj.setdefault(e[1], []).append(e)
        WHITE, GREY, BLACK = 0, 1, 2
        color: Dict[str, int] = {}
        stack: List[Tuple[str, str, str]] = []

        def dfs(node: str) -> List[Tuple[str, str, str]]:
            color[node] = GREY
            for e in adj.get(node, []):
                nxt = e[2]
                if color.get(nxt, WHITE) == GREY:
                    # back-edge → close the cycle from where nxt entered the stack
                    cyc = [e]
                    for se in reversed(stack):
                        cyc.append(se)
                        if se[1] == nxt:
                            break
                    return list(reversed(cyc))
                if color.get(nxt, WHITE) == WHITE:
                    stack.append(e)
                    got = dfs(nxt)
                    stack.pop()
                    if got:
                        return got
            color[node] = BLACK
            return []

        for start in list(adj):
            if color.get(start, WHITE) == WHITE:
                got = dfs(start)
                if got:
                    return got
        return []

    # ---- confidence, rendering, growth ---- #
    def _confidence(self, theory: DomainTheory) -> float:
        """Moderate and corroboration-scaled — always below a verified faculty."""
        corroboration = sum(law.support for law in theory.laws)
        conf = 0.45 + 0.04 * min(corroboration, 5)
        # a higher-order feedback closure is the strongest structural signal
        if any(law.kind == "feedback_loop" for law in theory.laws):
            conf += 0.03
        return min(0.66, conf)

    def _render(self, theory: DomainTheory) -> str:
        head = ("This is a genuinely new field. I built a model of it from its own internal "
                "structure (no prior domain to borrow from) and induced these laws:")
        law_lines = [f"  • {law.statement}" for law in theory.laws] or \
                    ["  • (structure present, but no strong regularity)"]
        inf_head = ("From those laws it likely follows (conjectures to verify, not asserted "
                    "facts):")
        inf_lines = [f"  • {_humanize(p)}" for p in theory.inferences]
        return "\n".join([head, *law_lines, "", inf_head, *inf_lines])

    def _label(self, query: str, rels: Sequence[Predicate]) -> str:
        """A short, stable label for the field, from its most distinctive entities/verbs."""
        ents: List[str] = []
        for p in rels:
            b = _binary(p)
            if b:
                for e in (b[1], b[2]):
                    if e not in ents and e not in _STOP and len(e) >= 4:
                        ents.append(e)
        ents.sort(key=lambda w: (-len(w), w))
        picked = ents[:2] or [b[0] for b in (_binary(p) for p in rels) if b][:2]
        return "genesis_" + "_".join(sorted(picked)) if picked else "genesis_field"

    def _learn(self, theory: DomainTheory) -> None:
        """Distil the modelled domain into a DomainSchema and learn it into the shared store."""
        eng = self.transfer_engine
        if eng is None or not hasattr(eng, "learn_domain"):
            return
        try:
            keywords = [e for e in theory.entities() if e not in _STOP][:8]
            # learn the stated relations *plus* the higher-order closure so the field carries the
            # same systematic glue the seed schemas do and can serve as a transfer base next time.
            rels = list(theory.relations) + [
                p for p in theory.inferences
                if isinstance(p, Predicate) and p.name == "sustains"]
            eng.learn_domain(theory.label, rels, keywords)
        except Exception:  # noqa: BLE001 — growth is best-effort, never breaks a turn
            pass


def _humanize(p: Term) -> str:
    """Render a relation as a readable clause: ``causes(a, b)`` → ``a causes b``."""
    if isinstance(p, Const):
        return p.name
    if not isinstance(p, Predicate):
        return str(p)
    verb = p.name.replace("_", " ")
    args = [_humanize(a) for a in p.args]
    if len(args) == 2:
        return f"{args[0]} {verb} {args[1]}"
    if len(args) == 1:
        return f"{args[0]} is {verb}"
    return f"{verb}({', '.join(args)})"


# --------------------------------------------------------------------------- #
# Self-test / demo (offline, no LLM anywhere)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA domain-genesis self-test (mastery from scratch, no LLM)")
    print("=" * 70)

    from nyxara.mind.transfer import RelationalTransferEngine

    # A GENUINELY ALIEN domain: fresh verbs, and a relational topology that maps onto NONE of the
    # seed schemas — so cross-domain transfer declines. NYXARA must model it from its OWN structure.
    alien = ("in this device the glorp zephs the drine and the drine zephs the quon "
             "and the quon flumps the glorp")

    # 1) transfer alone cannot help — the honest limitation this module lifts.
    tr = RelationalTransferEngine().generalize(alien)
    print(f"\ncross-domain transfer : {'declined (None)' if tr is None else tr.base_domain}")

    eng = DomainGenesisEngine()
    theory = eng.build(alien)
    assert theory is not None, "must recover a relational skeleton from the alien text"
    print(f"extracted relations   : {[str(p) for p in theory.relations]}")
    assert len(theory.relations) >= 3
    kinds = {law.kind for law in theory.laws}
    print(f"induced laws          : {sorted(kinds)}")
    assert "transitivity" in kinds, "zeph chains: glorp→drine→quon"
    assert "feedback_loop" in kinds, "glorp→drine→quon→glorp is a self-sustaining loop"

    proj = {str(p) for p in theory.inferences}
    print(f"projected inferences  : {sorted(proj)}")
    # held-out transitive fact (never stated) + the higher-order feedback closure
    assert any(s.startswith("zeph(glorp, quon)") for s in proj), proj
    assert any(s.startswith("sustains(") for s in proj), proj

    res = eng.generalize(alien)
    assert res is not None and res.confidence >= 0.4
    print(f"\n{res.render()}")
    print(f"\nconfidence            : {res.confidence:.2f}  (moderate — an induced conjecture)")

    # 2) honest decline on free-form chat — no structure, so the LLM path would run.
    assert eng.generalize("hi how are you today") is None
    print("\nno structure          : declines (None) — LLM path would run  ✓")

    # 3) GROWTH (Rule 4): with a transfer engine, the alien field is learned and recognised next
    #    time — it even becomes a transfer base.
    shared = RelationalTransferEngine()
    grow = DomainGenesisEngine(transfer_engine=shared)
    grow.generalize(alien)
    assert shared.store.get(theory.label) is not None, "the modelled field must be learned"
    print(f"learned + recognised  : '{theory.label}' now in the transfer store  ✓")

    print("\nALL SELF-TESTS PASSED ✓")
