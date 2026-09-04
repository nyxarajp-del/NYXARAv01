"""NYXARA · njp/encyclopedia.py — real prose, from a real encyclopedia, with its citation (📚).

:mod:`nyxara.njp.passage` was taught on six passages and examined on twenty-six more, and every
one of them was written by the same hand in the same week. Passing that exam says the shapes
generalise across *subjects*; it says nothing at all about whether they survive prose somebody
else wrote. Wikipedia's own lead for the very concept the lessons start from:

    "Photosynthesis is a system of biological processes by which photopigment-bearing autotrophic
     organisms, such as most plants, algae and cyanobacteria, convert light energy—typically from
     sunlight—into the chemical energy necessary to fuel their metabolism."

An em-dash parenthetical inside the verb phrase, an appositive list inside the subject, and a
definition that is a clause with three of its own. Nothing in the lessons looks like this. That is
the point of this module: it is the source that can say what the reader is actually worth, because
it did not come from here.

**It downloads nothing.** ``scripts/fetch_wikipedia.py`` does that, once, into a corpus file; this
reads that file. The split is deliberate — a measurement that needs the network is a measurement
that changes between runs, and a test that hits an API is a test that fails when somebody else is
using it.

Every article carries **its citation**: the title, the URL, the revision date it was taken on, and
the licence it is under. Those travel onto every claim read out of it, because a claim that cannot
say where it came from is one nobody can later check, retract or date — and the plan is explicit
that a medical or legal fact without a date is a differently shaped fact, not a weaker one.

What this module does *not* do is decide what is true. It reads, it counts, and it hands the
result to the caller. Filing is :meth:`~nyxara.njp.brain.NJPBrain.learn_passage`'s decision and
quarantine is :mod:`nyxara.njp.immune`'s, and a pipeline that fused all three would leave nobody
able to audit any of them.

Pure standard library.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

__all__ = ["Article", "Coverage", "Encyclopedia", "CORPUS", "WIKIPEDIA_LESSONS",
           "taught_on_wikipedia"]

#: Where ``scripts/build_wikipedia_corpus.py`` writes and this reads.
CORPUS = Path(__file__).with_name("data") / "wikipedia_leads.jsonl.gz"

#: Seven lessons in the encyclopedia's own voice, and the measurement that made them necessary:
#: taught only on prose written here, the reader produced a definition for 55% of real articles
#: and almost nothing else -- ``uses`` fired **zero** times across 2,771 leads. The constructions
#: the lessons had were the ones the lessons' author writes. Wikipedia's are ``are``, ``was``,
#: ``were``, ``refers to``, ``is used to``, ``consists of`` and ``occurs when``, and none of them
#: appeared in a single demonstration.
#:
#: **Only the expectations are written here.** The text is pulled from the corpus by title at
#: construction, so a lesson cannot quietly become a sentence that suits the reader better than
#: the one the encyclopedia actually contains — and a test asserts every title is real and that
#: none of them is in the audited or sealed slices.
WIKIPEDIA_LESSONS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "Merge algorithm": {
        "definition": ("a family of algorithms that take multiple sorted lists as input and "
                       "produce a single list as output",),
        "uses": ("multiple sorted lists",),
        "produces": ("a single list",)},
    "Beastie Boys": {
        "definition": ("an American hip-hop and punk group formed in New York City in 1981",)},
    "Osiris": {
        "definition": ("the god of fertility, agriculture, the afterlife, the dead, "
                       "resurrection, life, and vegetation in ancient Egyptian religion",)},
    "Byte": {
        "definition": ("a unit of digital information that most commonly consists of eight bits",),
        "consists_of": ("eight bits",)},
    "Feedback": {
        "occurs_when": ("outputs of a system are routed back as inputs",)},
    "Plant cell": {
        "definition": ("the cells present in green plants, photosynthetic eukaryotes of the "
                       "kingdom Plantae",)},
    "Key size": {
        "definition": ("the number of bits in a key used by a cryptographic algorithm",)},
    "Flavoring": {
        "definition": ("a food additive that is used to improve the taste or smell of food",),
        "purpose": ("improve the taste or smell of food",)},
}


@dataclass(frozen=True)
class Article:
    """One article's lead paragraph and the citation that makes it checkable."""

    title: str = ""
    text: str = ""
    url: str = ""
    pageid: int = 0
    domain: str = ""
    category: str = ""
    fetched: str = ""
    licence: str = ""
    source: str = ""

    @property
    def sentences(self) -> int:
        return sum(1 for c in self.text if c in ".!?")

    def to_dict(self) -> Dict[str, Any]:
        return {"title": self.title, "url": self.url, "domain": self.domain,
                "fetched": self.fetched, "licence": self.licence, "words": len(self.text.split())}


@dataclass
class Coverage:
    """What a reading run produced, counted without needing anyone to say what was right.

    Coverage is not accuracy and this class does not pretend otherwise. It answers *how much of
    the corpus the reader had anything to say about*, which is the half of the picture a gold set
    cannot give at this scale — and it is the half that moves first when prose gets harder.
    """

    articles: int = 0
    read: int = 0
    defined: int = 0
    kinded: int = 0
    relations: int = 0
    sentences: int = 0
    sentences_read: int = 0
    by_predicate: Dict[str, int] = field(default_factory=dict)
    by_domain: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    unresolved: int = 0

    @property
    def any_relation(self) -> float:
        return round(self.read / self.articles, 4) if self.articles else 0.0

    @property
    def definition_rate(self) -> float:
        return round(self.defined / self.articles, 4) if self.articles else 0.0

    @property
    def kind_rate(self) -> float:
        return round(self.kinded / self.articles, 4) if self.articles else 0.0

    @property
    def per_article(self) -> float:
        return round(self.relations / self.articles, 4) if self.articles else 0.0

    @property
    def sentence_rate(self) -> float:
        return round(self.sentences_read / self.sentences, 4) if self.sentences else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"articles": self.articles, "any_relation": self.any_relation,
                "definition_rate": self.definition_rate, "kind_rate": self.kind_rate,
                "relations": self.relations, "per_article": self.per_article,
                "sentence_rate": self.sentence_rate, "unresolved": self.unresolved,
                "by_predicate": dict(sorted(self.by_predicate.items(),
                                            key=lambda kv: -kv[1]))}

    def render(self) -> str:
        rows = [f"{self.articles} articles, {self.relations} relations "
                f"({self.per_article:.2f} each)",
                f"  a definition        {self.definition_rate:.3f}",
                f"  a kind              {self.kind_rate:.3f}",
                f"  any relation        {self.any_relation:.3f}",
                f"  sentences read      {self.sentence_rate:.3f} "
                f"({self.sentences_read}/{self.sentences})"]
        if self.by_predicate:
            top = ", ".join(f"{k} {v}" for k, v in
                            sorted(self.by_predicate.items(), key=lambda kv: -kv[1])[:6])
            rows.append(f"  by predicate        {top}")
        return "\n".join(rows)


class Encyclopedia:
    """The cached corpus, and the reader pointed at it."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else CORPUS
        self._articles: List[Article] = []

    # -- the corpus ---------------------------------------------------------- #
    def load(self) -> List[Article]:
        """Read the corpus file. A missing file is an empty encyclopedia, not an error."""
        if self._articles:
            return self._articles
        out: List[Article] = []
        try:
            if not self.path.exists():
                return out
            opener = gzip.open if str(self.path).endswith(".gz") else open
            with opener(self.path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:  # noqa: BLE001 — a malformed row is skipped, not fatal
                        continue
                    out.append(Article(title=str(row.get("title") or ""),
                                       text=str(row.get("text") or ""),
                                       url=str(row.get("url") or ""),
                                       pageid=int(row.get("pageid") or 0),
                                       domain=str(row.get("domain") or ""),
                                       category=str(row.get("category") or ""),
                                       fetched=str(row.get("fetched") or ""),
                                       licence=str(row.get("licence") or ""),
                                       source=str(row.get("source") or "")))
        except Exception:  # noqa: BLE001
            return out
        self._articles = out
        return out

    def domains(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for article in self.load():
            counts[article.domain] = counts.get(article.domain, 0) + 1
        return dict(sorted(counts.items()))

    def sample(self, n: int, *, every: int = 0, domain: str = "") -> List[Article]:
        """A deterministic slice. ``every`` takes one article in N, spread across the corpus.

        Deterministic on purpose: a held-out set that moves between runs cannot be held out, and
        an audit whose items change is an audit nobody can repeat.
        """
        rows = [a for a in self.load() if not domain or a.domain == domain]
        if every > 1:
            rows = rows[::every]
        return rows[:n] if n > 0 else rows

    # -- reading it ---------------------------------------------------------- #
    #: A trailing bracket in a title is a disambiguator, not part of the name: the article
    #: "Indiana Jones (character)" is about Indiana Jones. Left in, the concept's head noun came
    #: out as ``character`` and no sentence in the lead was about the topic.
    _DISAMBIGUATOR = __import__("re").compile(r"\s*\([^()]*\)\s*$")

    def name_of(self, article: Article) -> str:
        return self._DISAMBIGUATOR.sub("", article.title).strip() or article.title

    def read(self, article: Article, reader: Any) -> Any:
        """One article through the passage reader, with its citation attached to the result."""
        obj = reader.read(article.text, concept=self.name_of(article), source=article.source,
                          domain=article.domain)
        if obj is not None:
            obj.url = article.url
            obj.retrieved = article.fetched
            obj.licence = article.licence
        return obj

    def coverage(self, reader: Any, articles: Optional[Sequence[Article]] = None) -> Coverage:
        """Read a run of articles and count what came back. Says nothing about what was right."""
        out = Coverage()
        rows = list(articles) if articles is not None else self.load()
        for article in rows:
            out.articles += 1
            out.sentences += max(1, article.sentences)
            obj = self.read(article, reader)
            if obj is None:
                continue
            out.unresolved += len(obj.unresolved)
            if obj.definition:
                out.defined += 1
            if obj.kind:
                out.kinded += 1
            if obj.relations:
                out.read += 1
            out.relations += len(obj.relations)
            out.sentences_read += sum(1 for read_by in obj.provenance.values() if read_by)
            for relation in obj.relations:
                out.by_predicate[relation.predicate] = \
                    out.by_predicate.get(relation.predicate, 0) + 1
            hit, seen = out.by_domain.get(article.domain, (0, 0))
            out.by_domain[article.domain] = (hit + (1 if obj.relations else 0), seen + 1)
        return out

    def lessons(self) -> List[Any]:
        """:data:`WIKIPEDIA_LESSONS` as demonstrations, with their text taken from the corpus."""
        from nyxara.njp.passage import Demonstration

        by_title = {a.title: a for a in self.load()}
        out: List[Any] = []
        for title, expect in WIKIPEDIA_LESSONS.items():
            article = by_title.get(title)
            if article is None:
                continue
            out.append(Demonstration(name=f"wikipedia:{title}", concept=self.name_of(article),
                                     text=article.text, expect=dict(expect)))
        return out

    def claims(self, reader: Any,
               articles: Optional[Sequence[Article]] = None) -> Iterator[Tuple[Article, Any]]:
        """Every relation the reader produced, with the article and the sentence behind it."""
        for article in (articles if articles is not None else self.load()):
            obj = self.read(article, reader)
            if obj is None:
                continue
            for relation in obj.relations:
                yield article, relation


def taught_on_wikipedia(book: Optional[Encyclopedia] = None, **kwargs: Any) -> Any:
    """A reader taught on the synthetic lessons **and** on the encyclopedia's own sentences.

    Both, not one or the other. The synthetic lessons carry the ``by which`` constructions that
    no Wikipedia lead in the corpus states as cleanly, and the encyclopedia's carry the copulas
    and the ``refers to`` that no synthetic lesson has. Which words end up naming a relation is
    recomputed over all of them together, so adding real prose can and does change what the
    reader thinks ``produces`` means.
    """
    from nyxara.njp.passage import taught_reader

    reader = taught_reader(**kwargs)
    for lesson in (book or Encyclopedia()).lessons():
        reader.teach(lesson)
    return reader
