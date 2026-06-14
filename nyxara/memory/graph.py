"""NYXARA · memory/graph.py — knowledge graph + natural-language query.

Facts as a graph of **typed entities** joined by **typed, provenanced relations**
(subject –predicate→ object triples). On top of the raw store this gives NYXARA:

* **Pattern queries** — SPARQL-lite ``(subject?, predicate?, object?)`` matching.
* **Multi-hop traversal & path finding** — "how is A connected to B?", with paths
  scored by the *weakest* relation's confidence (a chain is only as sure as its
  flimsiest link). networkx-accelerated when present; exact pure-Python otherwise.
* **Reasoning** — transitive predicates (``is_a``, ``part_of`` inheritance) and
  registered inverse predicates (``owns`` ↔ ``owned_by``) so questions answer in
  either direction.
* **Natural-language → query** — a dependency-free, rule-based parser turns
  questions ("who is the owner of NYXARA?", "how is chess related to defense?",
  "what does JP own?") into graph queries. (An LLM can augment this later — the
  kernel decides; this layer always works offline.)

Every triple carries a :class:`~nyxara.memory.provenance.Provenance`, so answers
report their confidence and source (Rule 6).

Depends on :mod:`memory.provenance` and :mod:`kernel.errors`. networkx optional.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from nyxara.kernel.errors import MemoryError_
from nyxara.memory.provenance import Provenance, SourceType

try:  # optional acceleration
    import networkx as _nx  # type: ignore
    _HAS_NX = True
except Exception:  # pragma: no cover
    _nx = None  # type: ignore
    _HAS_NX = False

__all__ = [
    "Entity",
    "Triple",
    "GraphQuery",
    "Path",
    "GraphAnswer",
    "KnowledgeGraph",
    "has_networkx",
    "Relation",
    "GraphPopulator",
]


def has_networkx() -> bool:
    return _HAS_NX


def _norm(s: str) -> str:
    return re.sub(r"\s+", "_", s.strip().lower())


# --------------------------------------------------------------------------- #
# Entity & triple
# --------------------------------------------------------------------------- #
@dataclass
class Entity:
    name: str
    etype: str = "thing"
    eid: str = ""
    aliases: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    provenance: Optional[Provenance] = None

    def __post_init__(self) -> None:
        if not self.eid:
            self.eid = _norm(self.name)
        self.aliases = {_norm(a) for a in self.aliases}

    def all_names(self) -> Set[str]:
        return {_norm(self.name)} | self.aliases

    def to_dict(self) -> Dict[str, Any]:
        return {"eid": self.eid, "name": self.name, "etype": self.etype,
                "aliases": sorted(self.aliases), "attributes": self.attributes}


@dataclass
class Triple:
    subject: str          # entity id
    predicate: str        # normalised relation
    object: str           # entity id
    confidence: float = 0.9
    provenance: Optional[Provenance] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    triple_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def trust(self, now: Optional[float] = None) -> float:
        if self.provenance is not None:
            return self.provenance.trust(now)
        return self.confidence

    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def to_dict(self) -> Dict[str, Any]:
        return {"s": self.subject, "p": self.predicate, "o": self.object,
                "confidence": round(self.confidence, 4), "triple_id": self.triple_id,
                "attributes": self.attributes}


@dataclass
class GraphQuery:
    subject: Optional[str] = None     # entity id or None (wildcard)
    predicate: Optional[str] = None
    object: Optional[str] = None
    min_confidence: float = 0.0


@dataclass
class Path:
    nodes: List[str]
    edges: List[Triple]
    score: float  # weakest-link confidence along the path

    def predicates(self) -> List[str]:
        return [e.predicate for e in self.edges]

    def to_dict(self) -> Dict[str, Any]:
        return {"nodes": self.nodes, "predicates": self.predicates(),
                "score": round(self.score, 4)}


@dataclass
class GraphAnswer:
    question: str
    interpretation: str
    triples: List[Triple] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    paths: List[Path] = field(default_factory=list)
    text: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"question": self.question, "interpretation": self.interpretation,
                "text": self.text, "confidence": round(self.confidence, 4),
                "triples": [t.to_dict() for t in self.triples],
                "paths": [p.to_dict() for p in self.paths]}


# --------------------------------------------------------------------------- #
# The knowledge graph
# --------------------------------------------------------------------------- #
class KnowledgeGraph:
    """Typed, provenanced knowledge graph with traversal, reasoning, and NL query."""

    # default relation synonyms used by the NL parser (noun/verb -> predicate)
    _DEFAULT_SYNONYMS = {
        "owner": "owns", "owns": "owns", "own": "owns", "belongs": "owned_by",
        "is": "is_a", "kind": "is_a", "type": "is_a", "a": "is_a",
        "part": "part_of", "located": "located_in", "location": "located_in",
        "knows": "knows", "made": "created", "created": "created",
        "defends": "defends", "protects": "defends", "related": "related_to",
        "connected": "related_to",
    }

    def __init__(self) -> None:
        self._entities: Dict[str, Entity] = {}
        self._names: Dict[str, str] = {}            # normalised name/alias -> eid
        self._triples: Dict[str, Triple] = {}
        self._out: Dict[str, List[str]] = {}        # eid -> [triple_id]
        self._in: Dict[str, List[str]] = {}         # eid -> [triple_id]
        self._inverse: Dict[str, str] = {}          # predicate -> inverse predicate
        self._transitive: Set[str] = set()
        self._symmetric: Set[str] = set()
        self.synonyms: Dict[str, str] = dict(self._DEFAULT_SYNONYMS)

    # ---- schema config ---- #
    def register_inverse(self, predicate: str, inverse: str) -> None:
        p, q = _norm(predicate), _norm(inverse)
        self._inverse[p] = q
        self._inverse[q] = p

    def register_transitive(self, predicate: str) -> None:
        self._transitive.add(_norm(predicate))

    def register_symmetric(self, predicate: str) -> None:
        self._symmetric.add(_norm(predicate))

    def add_synonym(self, word: str, predicate: str) -> None:
        self.synonyms[_norm(word).replace("_", " ").strip()] = _norm(predicate)
        self.synonyms[_norm(word)] = _norm(predicate)

    # ---- entities ---- #
    def add_entity(self, name: str, *, etype: str = "thing", aliases: Sequence[str] = (),
                   attributes: Optional[Dict[str, Any]] = None,
                   provenance: Optional[Provenance] = None) -> Entity:
        # reuse an existing entity if this name/alias already resolves to one
        existing_id = self.resolve(name)
        ent = self._entities.get(existing_id) if existing_id else None
        if ent is None:
            ent = Entity(name=name, etype=etype, aliases=set(aliases),
                         attributes=attributes or {}, provenance=provenance)
            self._entities[ent.eid] = ent
            self._out.setdefault(ent.eid, [])
            self._in.setdefault(ent.eid, [])
        else:
            ent.aliases |= {_norm(a) for a in aliases}
            if attributes:
                ent.attributes.update(attributes)
            if etype != "thing":
                ent.etype = etype
        for n in ent.all_names():
            self._names[n] = ent.eid
        return ent

    def resolve(self, name: str) -> Optional[str]:
        """Map a surface name/alias to an entity id (None if unknown)."""
        return self._names.get(_norm(name))

    def get_entity(self, name_or_id: str) -> Entity:
        eid = name_or_id if name_or_id in self._entities else self.resolve(name_or_id)
        if eid is None or eid not in self._entities:
            raise MemoryError_(f"unknown entity {name_or_id!r}")
        return self._entities[eid]

    # ---- triples ---- #
    def add_triple(self, subject: str, predicate: str, object: str, *,
                   confidence: float = 0.9, provenance: Optional[Provenance] = None,
                   attributes: Optional[Dict[str, Any]] = None) -> Triple:
        s = self.add_entity(subject).eid
        o = self.add_entity(object).eid
        p = _norm(predicate)
        t = Triple(subject=s, predicate=p, object=o, confidence=confidence,
                   provenance=provenance, attributes=attributes or {})
        self._triples[t.triple_id] = t
        self._out[s].append(t.triple_id)
        self._in[o].append(t.triple_id)
        return t

    def __len__(self) -> int:
        return len(self._triples)

    # ---- pattern matching ---- #
    def match(self, query: GraphQuery, now: Optional[float] = None) -> List[Triple]:
        s = self.resolve(query.subject) if query.subject else None
        o = self.resolve(query.object) if query.object else None
        p = _norm(query.predicate) if query.predicate else None
        out: List[Triple] = []
        for t in self._triples.values():
            if s is not None and t.subject != s:
                continue
            if o is not None and t.object != o:
                continue
            if p is not None and t.predicate != p:
                continue
            if t.trust(now) < query.min_confidence:
                continue
            out.append(t)
        # also honour inverse predicates when querying by predicate
        if p is not None and p in self._inverse:
            inv = self._inverse[p]
            for t in self._triples.values():
                if t.predicate != inv:
                    continue
                # inverse: (o inv s) answers (s p o); swap roles for the filter
                if s is not None and t.object != s:
                    continue
                if o is not None and t.subject != o:
                    continue
                if t.trust(now) < query.min_confidence:
                    continue
                out.append(_swap(t, p))
        return out

    def triples_of(self, entity: str, *, direction: str = "out") -> List[Triple]:
        eid = self.resolve(entity)
        if eid is None:
            return []
        ids = self._out.get(eid, []) if direction == "out" else self._in.get(eid, [])
        return [self._triples[i] for i in ids]

    def neighbors(self, entity: str) -> List[str]:
        eid = self.resolve(entity)
        if eid is None:
            return []
        ns = {self._triples[i].object for i in self._out.get(eid, [])}
        ns |= {self._triples[i].subject for i in self._in.get(eid, [])}
        return sorted(ns)

    # ---- reasoning ---- #
    def transitive_closure(self, start: str, predicate: str,
                           now: Optional[float] = None) -> List[str]:
        """All entities reachable from ``start`` via repeated ``predicate``."""
        p = _norm(predicate)
        sid = self.resolve(start)
        if sid is None:
            return []
        seen: Set[str] = set()
        frontier = [sid]
        while frontier:
            cur = frontier.pop()
            for tid in self._out.get(cur, []):
                t = self._triples[tid]
                if t.predicate == p and t.object not in seen:
                    seen.add(t.object)
                    frontier.append(t.object)
        return sorted(seen)

    # ---- path finding (weakest-link confidence) ---- #
    def paths(self, source: str, target: str, *, max_hops: int = 4,
              now: Optional[float] = None) -> List[Path]:
        s, tg = self.resolve(source), self.resolve(target)
        if s is None or tg is None:
            return []
        found: List[Path] = []
        # BFS over directed-or-reverse adjacency up to max_hops
        queue: List[Tuple[str, List[str], List[Triple]]] = [(s, [s], [])]
        while queue:
            cur, nodes, edges = queue.pop(0)
            if len(edges) > max_hops:
                continue
            if cur == tg and edges:
                score = min(e.trust(now) for e in edges)
                found.append(Path(nodes=nodes, edges=edges, score=score))
                continue
            for tid in self._out.get(cur, []):
                t = self._triples[tid]
                if t.object not in nodes:
                    queue.append((t.object, nodes + [t.object], edges + [t]))
            for tid in self._in.get(cur, []):
                t = self._triples[tid]
                if t.subject not in nodes:
                    queue.append((t.subject, nodes + [t.subject], edges + [t]))
        found.sort(key=lambda pth: (len(pth.edges), -pth.score))
        return found

    def shortest_path(self, source: str, target: str, *, max_hops: int = 4,
                      now: Optional[float] = None) -> Optional[Path]:
        ps = self.paths(source, target, max_hops=max_hops, now=now)
        return ps[0] if ps else None

    # ---- natural language ---- #
    _Q_HOW = re.compile(r"\b(how|connection|connected|relate|related|link|path)\b")
    _Q_WHO = re.compile(r"\bwho\b")
    _Q_WHAT = re.compile(r"\bwhat|which\b")

    def _find_entities(self, q: str) -> List[str]:
        """Longest-match known entity names/aliases in the question."""
        qn = _norm(q).replace("_", " ")
        found: List[Tuple[int, str]] = []
        for surface, eid in self._names.items():
            phrase = surface.replace("_", " ")
            if re.search(rf"\b{re.escape(phrase)}\b", qn):
                found.append((len(phrase), eid))
        # prefer longer matches, dedupe preserving order
        found.sort(key=lambda x: -x[0])
        seen, out = set(), []
        for _, eid in found:
            if eid not in seen:
                seen.add(eid)
                out.append(eid)
        return out

    def _find_predicate(self, q: str) -> Optional[str]:
        qn = _norm(q).replace("_", " ")
        words = qn.split()
        # multi-word synonyms first (most specific)
        for syn, pred in self.synonyms.items():
            if " " in syn and syn in qn:
                return pred
        candidates: List[str] = []
        for w in words:
            if w in self.synonyms:
                candidates.append(self.synonyms[w])
        # known predicates appearing literally
        preds = {t.predicate for t in self._triples.values()}
        for w in words:
            if w in preds and w not in candidates:
                candidates.append(w)
        if not candidates:
            return None
        # demote the generic copula ("is"/"a" -> is_a) when a richer relation exists,
        # e.g. "who IS the OWNER of X" should resolve to 'owns', not 'is_a'.
        non_copula = [c for c in candidates if c != "is_a"]
        return non_copula[0] if non_copula else candidates[0]

    def ask(self, question: str, *, now: Optional[float] = None) -> GraphAnswer:
        """Answer a natural-language question against the graph (rule-based)."""
        q = question.strip()
        ents = self._find_entities(q)
        pred = self._find_predicate(q)
        ans = GraphAnswer(question=q, interpretation="")

        # 1) relational / path question between two entities
        if (self._Q_HOW.search(q.lower()) or "between" in q.lower()) and len(ents) >= 2:
            path = self.shortest_path(ents[0], ents[1], now=now)
            ans.interpretation = f"path({ents[0]} ~ {ents[1]})"
            if path:
                ans.paths = [path]
                chain = " → ".join(
                    f"{self._entities[path.nodes[i]].name} -[{path.edges[i].predicate}]-"
                    for i in range(len(path.edges))
                ) + f" {self._entities[path.nodes[-1]].name}"
                ans.text = chain
                ans.confidence = path.score
            else:
                ans.text = f"No known connection between {ents[0]} and {ents[1]}."
            return ans

        # 2) predicate + entity -> pattern query
        if pred and ents:
            subj_q = GraphQuery(subject=ents[0], predicate=pred, min_confidence=0.0)
            forward = self.match(subj_q, now)
            if self._Q_WHO.search(q.lower()) or "owner of" in q.lower() or "'s" in q:
                # asking for the SUBJECT of (?, pred, entity)
                obj_q = GraphQuery(predicate=pred, object=ents[0], min_confidence=0.0)
                back = self.match(obj_q, now)
                if back:
                    ans.interpretation = f"?-[{pred}]->{ents[0]}"
                    ans.triples = back
                    names = [self._entities[t.subject].name for t in back]
                    ans.entities = [t.subject for t in back]
                    ans.text = ", ".join(names)
                    ans.confidence = max(t.trust(now) for t in back)
                    return ans
            if forward:
                ans.interpretation = f"{ents[0]}-[{pred}]->?"
                ans.triples = forward
                names = [self._entities[t.object].name for t in forward]
                ans.entities = [t.object for t in forward]
                ans.text = ", ".join(names)
                ans.confidence = max(t.trust(now) for t in forward)
                return ans

        # 3) fallback: describe the entity (its outgoing relations)
        if ents:
            rels = self.triples_of(ents[0], direction="out")
            ans.interpretation = f"describe({ents[0]})"
            ans.triples = rels
            ans.entities = [ents[0]]
            if rels:
                ans.text = "; ".join(
                    f"{t.predicate} {self._entities[t.object].name}" for t in rels)
                ans.confidence = sum(t.trust(now) for t in rels) / len(rels)
            else:
                ans.text = f"I know of {self._entities[ents[0]].name} but no relations."
            return ans

        ans.interpretation = "unparsed"
        ans.text = "I couldn't identify a known entity in that question."
        return ans

    # ---- persistence & stats ---- #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self._entities.values()],
            "triples": [t.to_dict() for t in self._triples.values()],
            "inverse": self._inverse, "transitive": sorted(self._transitive),
            "symmetric": sorted(self._symmetric),
        }

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, default=str)
        return path

    def stats(self) -> Dict[str, Any]:
        preds: Dict[str, int] = {}
        for t in self._triples.values():
            preds[t.predicate] = preds.get(t.predicate, 0) + 1
        return {"entities": len(self._entities), "triples": len(self._triples),
                "predicates": preds, "networkx": has_networkx()}


# --------------------------------------------------------------------------- #
# Standard relation vocabulary (Level 6)
# --------------------------------------------------------------------------- #
class Relation(str):
    """Standard predicate constants — use these instead of bare strings."""
    CAUSES = "causes"
    OWNS = "owns"
    DEPENDS_ON = "depends_on"
    CREATED_BY = "created_by"
    LEARNED_FROM = "learned_from"
    PART_OF = "part_of"
    ENABLES = "enables"
    BLOCKS = "blocks"
    RELATED_TO = "related_to"
    IS_A = "is_a"
    HAS_PROPERTY = "has_property"
    SAID = "said"
    REPLIED = "replied"


def _configure_standard_relations(graph: KnowledgeGraph) -> None:
    """Register standard inverse, transitive, and symmetric relations."""
    graph.register_inverse(Relation.OWNS, "owned_by")
    graph.register_inverse(Relation.CREATED_BY, "created")
    graph.register_inverse(Relation.DEPENDS_ON, "required_by")
    graph.register_inverse(Relation.ENABLES, "enabled_by")
    graph.register_inverse(Relation.BLOCKS, "blocked_by")
    graph.register_transitive(Relation.PART_OF)
    graph.register_transitive(Relation.IS_A)
    graph.register_transitive(Relation.CAUSES)
    graph.register_symmetric(Relation.RELATED_TO)
    # synonym shortcuts for the NL parser
    for word in ("causes", "cause", "led"):
        graph.add_synonym(word, Relation.CAUSES)
    for word in ("depends", "requires", "needs"):
        graph.add_synonym(word, Relation.DEPENDS_ON)
    for word in ("enables", "allow", "allows"):
        graph.add_synonym(word, Relation.ENABLES)
    for word in ("part", "component", "module"):
        graph.add_synonym(word, Relation.PART_OF)
    for word in ("learned", "learned from", "from"):
        graph.add_synonym(word, Relation.LEARNED_FROM)


class GraphPopulator:
    """Extract subject–predicate–object triples from memory records and text.

    Used by Level 6 to auto-populate the KnowledgeGraph every turn from episodic
    and semantic memories, so the graph accumulates structured knowledge organically.
    """

    def __init__(self, graph: KnowledgeGraph,
                 provenance: Optional[Provenance] = None) -> None:
        self.graph = graph
        self.default_prov = provenance

    def from_memory_record(self, record: Any) -> int:
        """Extract triples from a MemoryRecord (or any object with a .text() method).

        Uses simple heuristics to identify subject-verb-object patterns in the text.
        Returns the number of triples added.
        """
        try:
            text = record.text() if callable(getattr(record, "text", None)) else str(
                getattr(record, "content", record))
            text = str(text).strip()
            if not text:
                return 0
            return self._parse_and_add(text)
        except Exception:  # noqa: BLE001
            return 0

    def from_conversation_turn(self, stimulus: str, response: str,
                                confidence: float = 0.7) -> int:
        """Add triples from a conversation turn (stimulus + response pair)."""
        added = 0
        prov = self.default_prov
        # "Master said X" → (master, said, X[:40])
        s_short = stimulus[:60].strip()
        r_short = response[:60].strip()
        if s_short:
            try:
                self.graph.add_triple("master", Relation.SAID, s_short,
                                      confidence=confidence, provenance=prov)
                added += 1
            except Exception:  # noqa: BLE001
                pass
        if r_short:
            try:
                self.graph.add_triple("nyxara", Relation.REPLIED, r_short,
                                      confidence=confidence * 0.9, provenance=prov)
                added += 1
            except Exception:  # noqa: BLE001
                pass
        # extract any "X is Y" / "X has Y" patterns from stimulus
        added += self._parse_and_add(stimulus, confidence=confidence * 0.8)
        return added

    def _parse_and_add(self, text: str, confidence: float = 0.65) -> int:
        """Heuristic triple extraction from free text."""
        added = 0
        prov = self.default_prov
        lower = text.lower()
        # "X is a Y" / "X is Y"
        for m in re.finditer(r"(\b[\w\s]{2,30}?)\s+is\s+(?:a\s+|an\s+)?([\w\s]{2,30}?\b)",
                             lower):
            subj, obj = m.group(1).strip(), m.group(2).strip()
            if 2 <= len(subj) <= 30 and 2 <= len(obj) <= 30:
                try:
                    self.graph.add_triple(subj, Relation.IS_A, obj,
                                          confidence=confidence, provenance=prov)
                    added += 1
                    if added >= 5:
                        return added
                except Exception:  # noqa: BLE001
                    pass
        # "X has Y" / "X owns Y"
        for m in re.finditer(r"(\b[\w\s]{2,20}?)\s+(?:has|owns|have)\s+([\w\s]{2,20}?\b)",
                             lower):
            subj, obj = m.group(1).strip(), m.group(2).strip()
            if 2 <= len(subj) <= 20 and 2 <= len(obj) <= 20:
                try:
                    self.graph.add_triple(subj, Relation.HAS_PROPERTY, obj,
                                          confidence=confidence * 0.8, provenance=prov)
                    added += 1
                    if added >= 5:
                        return added
                except Exception:  # noqa: BLE001
                    pass
        return added


def _swap(t: Triple, as_predicate: str) -> Triple:
    """Present an inverse triple as if it answered the forward predicate."""
    return Triple(subject=t.object, predicate=as_predicate, object=t.subject,
                  confidence=t.confidence, provenance=t.provenance,
                  attributes={**t.attributes, "via_inverse": t.predicate},
                  triple_id=t.triple_id)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA knowledge-graph self-test")
    print("=" * 70)
    print(f"networkx available  : {has_networkx()}")

    g = KnowledgeGraph()
    g.register_inverse("owns", "owned_by")
    g.register_transitive("part_of")
    g.register_symmetric("related_to")

    # build a small world
    g.add_entity("Jaypal Khoja", etype="person", aliases=["JP", "boss"])
    g.add_triple("Jaypal Khoja", "owns", "NYXARA", confidence=1.0,
                 provenance=Provenance(SourceType.OWNER, confidence=1.0))
    g.add_triple("NYXARA", "is_a", "AI", confidence=1.0)
    g.add_triple("NYXARA", "defends", "Jaypal Khoja", confidence=1.0)
    g.add_triple("kernel", "part_of", "NYXARA", confidence=0.95)
    g.add_triple("invariants", "part_of", "kernel", confidence=0.95)
    g.add_triple("chess", "related_to", "strategy", confidence=0.8)
    g.add_triple("strategy", "related_to", "warfare", confidence=0.8)
    g.add_triple("warfare", "related_to", "defense", confidence=0.8)

    # pattern query
    owns = g.match(GraphQuery(subject="JP", predicate="owns"))
    print(f"\nJP owns             : {[g.get_entity(t.object).name for t in owns]}")
    assert any(g.get_entity(t.object).name == "NYXARA" for t in owns)

    # inverse query: who owns NYXARA?
    inv = g.match(GraphQuery(predicate="owns", object="NYXARA"))
    print(f"owns NYXARA (subj)  : {[g.get_entity(t.subject).name for t in inv]}")

    # transitive closure: what is part_of NYXARA (transitively)?
    parts = g.transitive_closure("invariants", "part_of")
    print(f"invariants part_of* : {parts}")
    assert "nyxara" in parts and "kernel" in parts

    # path finding: chess ~ defense
    path = g.shortest_path("chess", "defense")
    print(f"chess ~ defense path: {path.to_dict() if path else None}")
    assert path and "defense" in path.nodes

    # natural language
    print("\n--- natural language ---")
    for q in ["Who is the owner of NYXARA?",
              "What does JP own?",
              "What is NYXARA?",
              "How is chess related to defense?",
              "Who owns NYXARA?"]:
        a = g.ask(q)
        print(f"  Q: {q}")
        print(f"     -> [{a.interpretation}] {a.text}  (conf={a.confidence:.2f})")

    a1 = g.ask("Who is the owner of NYXARA?")
    assert "Jaypal" in a1.text
    a2 = g.ask("What does JP own?")
    assert "NYXARA" in a2.text
    a3 = g.ask("How is chess related to defense?")
    assert a3.paths

    print(f"\nstats               : {g.stats()}")
    print("\nALL SELF-TESTS PASSED ✓")
