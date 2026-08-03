"""NYXARA · growth/dataset_hyperspace.py — a dataset becomes geometry (📚→🌐).

:mod:`~nyxara.mind.concept_hyperspace` can fit a Poincaré embedding, but only if something hands it
``(child, parent)`` relations. Nothing did: ``manifold_sync_graph`` seeds from the knowledge graph
and its own docstring says it is *"never wired into the live turn path"*. This is the wire from an
ingested dataset to a fitted concept space.

Relations are read with a deterministic **is-a cue grammar** — "a poodle is a dog", "dogs are a
kind of mammal", "mammals include dogs", "cats, dogs and wolves are mammals" — the same style of
explicit, auditable extraction :mod:`~nyxara.growth.text_causal` uses for causal claims, and for
the same reason: an extraction you cannot inspect is an extraction you cannot trust.

Deliberately narrow. "is a" is heavily overloaded in English ("the answer is a mystery", "today is
a Tuesday"), so the patterns require a taxonomic frame and the copula alone is not enough. Missing
a real relation costs a little structure; inventing one corrupts the hierarchy for every concept
downstream of it.

Depends on ``growth/dataset`` (the store), ``senses/nlp`` (sentence splitting) and
``mind/concept_hyperspace`` (the fit). Nothing imports back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["IsARelation", "HyperspaceIngestReport", "extract_is_a", "build_from_dataset_store"]

_STOPWORDS = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "all", "every",
    "kind", "kinds", "type", "types", "sort", "sorts", "form", "forms", "example", "examples",
    "of", "is", "are", "was", "were", "be", "been", "called", "known", "considered",
})
_MAX_LABEL_WORDS = 3

# (name, pattern, child_group, parent_group, child_is_list)
_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]", int, int, bool], ...] = (
    # "a poodle is a kind of dog" / "dogs are a type of mammal"
    ("kind_of", re.compile(
        r"^(.+?)\s+(?:is|are)\s+(?:a|an)?\s*(?:kind|type|sort|form|species|subclass|subset)"
        r"\s+of\s+(.+)$", re.I), 1, 2, True),
    # "mammals include dogs, cats and wolves"  (parent first, children listed)
    ("includes", re.compile(
        r"^(.+?)\s+(?:include|includes|comprise|comprises|encompass|encompasses)\s+(.+)$", re.I),
     2, 1, True),
    # "a poodle is a dog" — the bare copula, tightly fenced by the article requirement
    ("is_a", re.compile(r"^(?:a|an)\s+(.+?)\s+is\s+(?:a|an)\s+(.+)$", re.I), 1, 2, False),
    # "dogs are mammals" — bare plural copula, both sides plural
    ("are", re.compile(r"^(.+?s)\s+are\s+(.+?s)$", re.I), 1, 2, True),
    # "X, a kind of Y," appositive
    ("appositive", re.compile(r"^(.+?),\s+(?:a|an)\s+(?:kind|type|form)\s+of\s+(.+?),", re.I),
     1, 2, False),
)

# Copula sentences that look taxonomic but are not — the frame is right, the meaning is not.
_NOT_TAXONOMIC = re.compile(
    r"\b(mystery|problem|question|answer|matter|shame|pity|surprise|reminder|"
    r"way|means|result|consequence|sign|symbol|metaphor|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday)\b", re.I)


def _normalize(phrase: str) -> str:
    """Reduce a phrase to a stable concept label, singularising a trailing plural."""
    words = re.findall(r"[a-z0-9'-]+", (phrase or "").lower())
    while words and words[0] in _STOPWORDS:
        words.pop(0)
    while words and words[-1] in _STOPWORDS:
        words.pop()
    if not words:
        return ""
    words = words[:_MAX_LABEL_WORDS]
    # Crude singularisation so "dogs" and "dog" are one concept rather than two. Rule-based and
    # therefore imperfect ("knives" lands on "knife" only by luck of ordering, "mice" not at all);
    # the property that matters is *consistency* — both spellings must reach the same label — and
    # an occasional odd label costs far less than splitting one concept across two points.
    last = words[-1]
    if len(last) > 3 and last.endswith("ies"):
        words[-1] = last[:-3] + "y"
    elif len(last) > 3 and last.endswith("ves"):
        words[-1] = last[:-3] + "f"          # wolves -> wolf, leaves -> leaf
    elif len(last) > 3 and last.endswith(("ses", "xes", "zes", "ches", "shes")):
        words[-1] = last[:-2]
    elif len(last) > 3 and last.endswith("s") and not last.endswith("ss"):
        words[-1] = last[:-1]
    return " ".join(words)


def _split_list(phrase: str) -> List[str]:
    """Split "cats, dogs and wolves" into its members."""
    parts = re.split(r",\s*|\s+and\s+|\s+or\s+", phrase or "")
    return [p for p in (part.strip() for part in parts) if p]


@dataclass
class IsARelation:
    """One taxonomic relation read from a sentence, with the sentence kept for audit."""

    child: str
    parent: str
    cue: str = ""
    quote: str = ""
    source: str = ""

    def as_pair(self) -> Tuple[str, str]:
        return self.child, self.parent

    def to_dict(self) -> Dict[str, Any]:
        return {"child": self.child, "parent": self.parent, "cue": self.cue,
                "quote": self.quote, "source": self.source}


@dataclass
class HyperspaceIngestReport:
    """What was extracted and what the fit achieved."""

    sentences: int = 0
    relations: List[IsARelation] = field(default_factory=list)
    rejected: int = 0
    fit: Any = None                    # a HyperspaceReport, when a fit ran
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"· hyperspace ingest: {len(self.relations)} is-a relation(s) from "
                 f"{self.sentences} sentence(s); {self.rejected} non-taxonomic copula(s) skipped"]
        for relation in self.relations[:10]:
            lines.append(f"    {relation.child} ⊂ {relation.parent}  [{relation.cue}]")
        if self.fit is not None:
            lines.append(self.fit.summary())
        lines.extend(f"    · {n}" for n in self.notes)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"sentences": self.sentences, "rejected": self.rejected,
                "relations": [r.to_dict() for r in self.relations],
                "fit": self.fit.to_dict() if self.fit is not None else None,
                "notes": list(self.notes)}


def extract_is_a(text: str, *, source: str = "") -> Tuple[List[IsARelation], int, int]:
    """Read taxonomic relations from prose. Returns ``(relations, n_sentences, n_rejected)``."""
    try:
        from nyxara.senses.nlp import sentences as split_sentences
        sentences = [s for s in split_sentences(text) if s.strip()]
    except Exception:  # noqa: BLE001 — the regex fallback keeps this working anywhere
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]

    out: List[IsARelation] = []
    rejected = 0
    for sentence in sentences:
        clean = sentence.strip().rstrip(".!?")
        if not clean:
            continue
        for cue, pattern, child_group, parent_group, child_is_list in _PATTERNS:
            match = pattern.match(clean)
            if not match:
                continue
            parent = _normalize(match.group(parent_group))
            if not parent:
                break
            raw_children = (_split_list(match.group(child_group)) if child_is_list
                            else [match.group(child_group)])
            if _NOT_TAXONOMIC.search(parent) or any(_NOT_TAXONOMIC.search(c) for c in raw_children):
                rejected += 1       # right frame, wrong meaning: "today is a Tuesday"
                break
            for raw in raw_children:
                child = _normalize(raw)
                if child and child != parent:
                    out.append(IsARelation(child=child, parent=parent, cue=cue,
                                           quote=sentence.strip()[:200], source=source))
            break
    return out, len(sentences), rejected


def build_from_dataset_store(store_path: Any, *, space: Any = None, domain: str = "",
                             dim: int = 64, seed: int = 0, epochs: int = 200,
                             limit: Optional[int] = None) -> HyperspaceIngestReport:
    """Read an ingested dataset, mine its taxonomic relations, and fit the concept space."""
    from nyxara.growth.dataset import load_dataset_docs
    from nyxara.mind.concept_hyperspace import ConceptHyperspace

    report = HyperspaceIngestReport()
    for doc in load_dataset_docs(store_path, limit=limit):
        relations, n_sentences, rejected = extract_is_a(doc, source=str(store_path))
        report.sentences += n_sentences
        report.rejected += rejected
        report.relations.extend(relations)

    if not report.relations:
        report.notes.append("no taxonomic relations found — nothing to fit")
        return report

    space = space if space is not None else ConceptHyperspace(dim=dim, seed=seed, lr=0.1)
    pairs: List[Tuple[str, str]] = []
    seen = set()
    for relation in report.relations:      # de-duplicate; a repeated claim is not more evidence
        pair = relation.as_pair()
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    report.fit = space.fit(pairs, epochs=epochs, burn_in=max(1, epochs // 10), domain=domain)
    report.notes.append(f"space has {len(space.concepts)} concept(s)")
    return report
