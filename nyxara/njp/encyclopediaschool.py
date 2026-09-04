"""NYXARA · njp/encyclopediaschool.py — what the reader is worth on prose she did not write (📏).

:mod:`nyxara.njp.passageschool` scores 0.919 on twenty-six passages and 0.918 on a sealed nine.
Every one of those was written by one hand in one week. This is the same reader on Wikipedia,
which was not.

Three measurements, and they are deliberately different in kind because at this scale no single
one is honest on its own:

* **coverage** needs no gold at all. How many articles yielded a definition, a kind, any relation;
  how many sentences produced anything. It says nothing about correctness and does not pretend to
  — it says how much of the corpus the reader had anything to say about, which is the number that
  moves first when prose gets harder.
* **precision by audit** marks what she produced, one relation at a time, against the sentence it
  came from. Judging output is a far weaker act of authorship than writing gold from scratch, so
  this is the number least contaminated by the person taking it.
* **recall on an annotated set** is the expensive one, so it is small and it is fixed. Both sets
  are deterministic slices of the corpus: an audit whose items move between runs is an audit
  nobody can repeat.

The audit is taken **before** any fix and re-taken after on the same items, and then once on a
second slice that was never looked at. The first two numbers say whether a fix worked; only the
third says whether it generalised.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from nyxara.njp.encyclopedia import Article, Coverage, Encyclopedia, taught_on_wikipedia
from nyxara.njp.passage import PassageReader, _bare, _toks, _words

__all__ = ["Audit", "Marked", "Report", "AUDIT", "coverage", "audit", "recall",
           "kinds", "grounder_on", "slices", "run"]

#: Where the hand-marked verdicts live. A JSON file rather than a table in this module, because it
#: is data somebody sat and marked and it should be diffable as data.
AUDIT = Path(__file__).with_name("data") / "wikipedia_audit.json"

#: The two slices, by construction and not by choice: one article in ``EVERY`` across the whole
#: corpus, first ``FIRST`` for the audit that fixes were made against, the next ``SEALED`` for the
#: one that was not looked at.
EVERY = 7
FIRST = 25
SEALED = 25


def _key(text: str) -> str:
    return " ".join(_words(_bare(_toks(str(text or "")))))


@dataclass
class Marked:
    """One produced relation and the verdict a person gave it."""

    title: str = ""
    predicate: str = ""
    object: str = ""
    verdict: str = ""          # "right" | "wrong"
    sentence: str = ""
    note: str = ""

    @property
    def id(self) -> Tuple[str, str, str]:
        return (_key(self.title), self.predicate, _key(self.object))


@dataclass
class Audit:
    """Precision on real prose, and what was still unmarked when it was taken."""

    marked: int = 0
    right: int = 0
    wrong: int = 0
    unmarked: Tuple[str, ...] = ()
    worst: Tuple[str, ...] = ()

    @property
    def precision(self) -> float:
        return round(self.right / self.marked, 4) if self.marked else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"precision": self.precision, "marked": self.marked, "right": self.right,
                "wrong": self.wrong, "unmarked": len(self.unmarked)}


@dataclass
class Report:
    coverage: Optional[Coverage] = None
    audit: Optional[Audit] = None
    sealed: Optional[Audit] = None
    recall: float = 0.0
    recall_asked: int = 0
    baseline: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"coverage": self.coverage.to_dict() if self.coverage else {},
                "audit": self.audit.to_dict() if self.audit else {},
                "sealed": self.sealed.to_dict() if self.sealed else {},
                "recall": self.recall, "recall_asked": self.recall_asked,
                "baseline": self.baseline}


# --------------------------------------------------------------------------------------------- #
#  the slices
# --------------------------------------------------------------------------------------------- #
def slices(book: Optional[Encyclopedia] = None) -> Tuple[List[Article], List[Article]]:
    """The audited slice and the sealed one. Deterministic, and disjoint by construction."""
    book = book or Encyclopedia()
    spread = book.sample(0, every=EVERY)
    return spread[:FIRST], spread[FIRST:FIRST + SEALED]


def _load_marks() -> Dict[Tuple[str, str, str], Marked]:
    try:
        rows = json.loads(AUDIT.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — an unmarked corpus audits to nothing, which is honest
        return {}
    out: Dict[Tuple[str, str, str], Marked] = {}
    for row in rows.get("marks", []):
        mark = Marked(title=str(row.get("title") or ""),
                      predicate=str(row.get("predicate") or ""),
                      object=str(row.get("object") or ""),
                      verdict=str(row.get("verdict") or ""),
                      sentence=str(row.get("sentence") or ""),
                      note=str(row.get("note") or ""))
        out[mark.id] = mark
    return out


def _marked(field: str) -> Dict[str, str]:
    """Per-article verdicts: ``definitions`` for the gloss, ``kinds`` for the head it names."""
    try:
        rows = json.loads(AUDIT.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return {str(k): str(v) for k, v in (rows.get(field) or {}).items()}


def _definitions() -> Dict[str, str]:
    return _marked("definitions")


# --------------------------------------------------------------------------------------------- #
#  the measurements
# --------------------------------------------------------------------------------------------- #
def coverage(reader: Optional[PassageReader] = None,
             book: Optional[Encyclopedia] = None) -> Coverage:
    book = book or Encyclopedia()
    return book.coverage(reader or taught_on_wikipedia(book))


def audit(articles: Sequence[Article], reader: Optional[PassageReader] = None,
          book: Optional[Encyclopedia] = None) -> Audit:
    """Mark every relation produced on these articles against what a person said of it."""
    book = book or Encyclopedia()
    reader = reader or taught_on_wikipedia(book)
    marks = _load_marks()
    out = Audit()
    unmarked: List[str] = []
    worst: List[str] = []
    for article, relation in book.claims(reader, articles):
        mark = marks.get((_key(article.title), relation.predicate, _key(relation.object)))
        if mark is None:
            unmarked.append(f"{article.title}: {relation.predicate}={relation.object}")
            continue
        out.marked += 1
        if mark.verdict == "right":
            out.right += 1
        else:
            out.wrong += 1
            worst.append(f"{article.title}: {relation.predicate}={relation.object}"
                         + (f"  [{mark.note}]" if mark.note else ""))
    out.unmarked = tuple(unmarked)
    out.worst = tuple(worst)
    return out


def recall(reader: Optional[PassageReader] = None,
           book: Optional[Encyclopedia] = None) -> Tuple[float, int]:
    """Definition accuracy: of the articles marked, how many got their **own** subject right?

    Every lead in the corpus defines its subject, so the denominator is every marked article and
    this is a recall and a precision at once: an article scores only if the reader produced a
    definition and that definition was of the article. Producing nothing scores the same as
    producing the wrong one, which is the honest treatment here — a reader that says nothing about
    the lung has not read the lead on the lung.
    """
    book = book or Encyclopedia()
    reader = reader or taught_on_wikipedia(book)
    marked = _definitions()
    if not marked:
        return 0.0, 0
    return (round(sum(1 for v in marked.values() if v == "right") / len(marked), 4),
            len(marked))


def kinds() -> Tuple[float, int]:
    """How often the structurally derived head noun is the category the subject falls under.

    Reported separately and not folded into anything, because it is the weakest link in the
    layered reading: the head of a noun phrase in English is not recoverable from the closed
    class alone, and ``a free and open-source software project`` gives ``free`` to any rule that
    has no way to know which of those words is the noun.
    """
    marked = _marked("kinds")
    if not marked:
        return 0.0, 0
    return (round(sum(1 for v in marked.values() if v == "right") / len(marked), 4),
            len(marked))


def grounder_on(articles: Sequence[Article]) -> Dict[str, Any]:
    """The same real leads through the sentence parser, counted the same way."""
    try:
        from nyxara.njp.grounding import Grounder
    except Exception:  # noqa: BLE001 - pragma: no cover
        return {}
    grounder = Grounder()
    produced = defined = read = 0
    for article in articles:
        got = 0
        for sentence in [s.strip() for s in article.text.split(". ") if s.strip()]:
            try:
                grounded = grounder.ground(sentence if sentence.endswith(".") else sentence + ".")
            except Exception:  # noqa: BLE001
                continue
            triples = list(getattr(grounded, "triples", ()) or ())
            got += len(triples)
            if any(t.predicate == "is_a" for t in triples):
                defined += 1
        produced += got
        read += 1 if got else 0
    n = max(1, len(articles))
    return {"articles": len(articles), "relations": produced,
            "per_article": round(produced / n, 4), "any_relation": round(read / n, 4)}


def run() -> Report:
    book = Encyclopedia()
    reader = taught_on_wikipedia(book)
    first, sealed = slices(book)
    got, asked = recall(reader, book)
    return Report(coverage=book.coverage(reader), audit=audit(first, reader, book),
                  sealed=audit(sealed, reader, book), recall=got, recall_asked=asked,
                  baseline=grounder_on(first))


def main() -> None:  # pragma: no cover — a report, not a test
    book = Encyclopedia()
    articles = book.load()
    if not articles:
        print(f"no corpus at {book.path}; run scripts/build_wikipedia_corpus.py")
        return
    print(f"{len(articles)} articles across {len(book.domains())} domains")
    report = run()
    print("\n=== coverage (no gold needed) ===")
    print(report.coverage.render() if report.coverage else "-")
    print("\n=== the sentence parser on the same leads ===")
    print("  " + json.dumps(report.baseline))
    print("\n=== precision by audit ===")
    print(f"  audited slice  {report.audit.precision:.3f} "
          f"({report.audit.right}/{report.audit.marked}), "
          f"{len(report.audit.unmarked)} unmarked")
    print(f"  sealed slice   {report.sealed.precision:.3f} "
          f"({report.sealed.right}/{report.sealed.marked}), "
          f"{len(report.sealed.unmarked)} unmarked")
    got, asked = kinds()
    print(f"\n=== the article's own definition === {report.recall:.3f} "
          f"of {report.recall_asked} articles")
    print(f"=== the kind it falls under    === {got:.3f} of {asked} articles")
    if report.audit.worst:
        print("\nmarked wrong:")
        for row in report.audit.worst[:20]:
            print("   ", row)


if __name__ == "__main__":  # pragma: no cover
    main()
