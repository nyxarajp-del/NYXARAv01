"""NYXARA · njp/breadth.py — how much of the world she has anything to say about (🌐, NJP V.41).

Every other examination in this package measures **how well she reasons over what she has**.
:mod:`nyxara.njp.general` asks questions whose answers are not in the store and grades the
derivation; :mod:`nyxara.njp.explaingauntlet` mints worlds she has never seen. Both are the right
measurements and neither of them can see the thing the Master actually asked about, because both
draw their subjects from what she already knows.

This one asks the other question, and it is the uncomfortable one:

    Of the things a person might mention, what fraction does she have **any fact at all** about?

Three papers, and they measure three different failures
--------------------------------------------------------

``coverage``
    Subjects drawn from a source **outside** the shipped corpus. How many does the store hold a
    single triple about? This is breadth and nothing else — it does not ask whether the fact is
    useful, only whether there is one. A low number here cannot be fixed by better reasoning.

``reachable``
    Of the subjects she *does* hold, how many can an English question actually retrieve? Knowing
    and being askable are different, and this package has measured the gap before:
    :mod:`nyxara.njp.general` records ``inheritance`` at 400/400 with **every one of them arriving
    through the derivation ladder and none through English**. A fact nobody can ask for is a fact
    she does not have, from where the person asking is standing.

``derived``
    Two-hop questions built from a **held-out** slice of the same source: the slice is not
    ingested, and its facts are the answers. This is the only paper of the three that can tell
    whether more knowledge buys more *reasoning* rather than more storage — the question that
    decides whether scaling the corpus is worth anything at all.

What this cannot measure, said plainly
--------------------------------------

A fact store and a language model do not have the same shape of knowledge and no single number
compares them. What is on this side is **named entities and stated relations between them**, which
is what can be asked for, checked, contradicted and derived from. What is not on this side is
everything a model carries that was never a triple: how a sentence is usually finished, what a
paragraph of legal prose sounds like, the shape of an argument, the thousand unstated regularities
that come from reading rather than from being told.

So ``coverage`` is a real number about a real thing and it is **not** a percentage of "what an LLM
knows". Nothing in this file will produce that percentage, because the denominator does not exist.
What it will produce is the honest one: of a list of things somebody might name, this many have a
fact behind them, and this many of those can be asked for in English.

Pure standard library, deterministic per seed.
"""

from __future__ import annotations

import gzip
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

__all__ = ["Paper", "Report", "Breadth", "measure", "render", "main",
           "DEFAULT_SAMPLE", "DEFAULT_SEED"]

DEFAULT_SAMPLE = 2000
DEFAULT_SEED = 20260906


@dataclass
class Paper:
    name: str
    asked: int = 0
    hit: int = 0
    examples_hit: List[str] = field(default_factory=list)
    examples_missed: List[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return round(self.hit / self.asked, 4) if self.asked else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"paper": self.name, "asked": self.asked, "hit": self.hit, "score": self.score,
                "hit_examples": self.examples_hit[:8],
                "missed_examples": self.examples_missed[:8]}


@dataclass
class Report:
    papers: List[Paper] = field(default_factory=list)
    facts: int = 0
    subjects: int = 0
    source: str = ""
    seed: int = DEFAULT_SEED

    def paper(self, name: str) -> Optional[Paper]:
        return next((p for p in self.papers if p.name == name), None)

    def to_dict(self) -> Dict[str, Any]:
        return {"facts": self.facts, "subjects": self.subjects, "source": self.source,
                "seed": self.seed, "papers": [p.to_dict() for p in self.papers]}


class Breadth:
    """Three papers over a brain and a list of subjects from somewhere else."""

    def __init__(self, brain: Any, *, seed: int = DEFAULT_SEED) -> None:
        self.brain = brain
        self.grounder = getattr(brain, "grounder", brain)
        self.seed = int(seed)
        self.by_sp: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.subjects: Set[str] = set()
        facts = getattr(self.grounder, "facts", None)
        if isinstance(facts, dict):
            for key, triples in facts.items():
                try:
                    subject, predicate = key
                except Exception:  # noqa: BLE001
                    continue
                for triple in triples:
                    if getattr(triple, "superseded", False):
                        continue
                    obj = str(getattr(triple, "object", "") or "").strip()
                    if obj:
                        self.by_sp[(subject, predicate)].append(obj)
                        self.subjects.add(subject)

    @property
    def facts(self) -> int:
        return sum(len(v) for v in self.by_sp.values())

    def _key(self, text: str) -> str:
        try:
            return self.grounder._key(text)
        except Exception:  # noqa: BLE001
            return " ".join(str(text or "").split()).lower()

    def knows(self, subject: str) -> bool:
        return self._key(subject) in self.subjects

    # ---- the papers ------------------------------------------------------ #
    def coverage(self, names: Sequence[str]) -> Paper:
        """Does the store hold **any** triple about this? Breadth, and nothing else."""
        out = Paper(name="coverage")
        for name in names:
            out.asked += 1
            if self.knows(name):
                out.hit += 1
                if len(out.examples_hit) < 20:
                    out.examples_hit.append(name)
            elif len(out.examples_missed) < 20:
                out.examples_missed.append(name)
        return out

    def reachable(self, names: Sequence[str]) -> Paper:
        """Of what she holds, how much can an English question actually get back?

        Only subjects she *does* know are asked, because a miss here has to mean *"the fact is in
        there and the question could not reach it"* rather than *"there was no fact"*. Mixing the
        two produces a number that moves when either changes and says which neither.
        """
        out = Paper(name="reachable")
        for name in names:
            if not self.knows(name):
                continue
            gold = self.by_sp.get((self._key(name), "is_a"), ())
            if not gold:
                continue
            out.asked += 1
            try:
                got = self.grounder.answer(f"what is {name}?")
            except Exception:  # noqa: BLE001
                got = None
            said = str(getattr(got, "text", "") or "").strip().lower()
            if said and any(said in g.lower() or g.lower() in said for g in gold):
                out.hit += 1
                if len(out.examples_hit) < 20:
                    out.examples_hit.append(name)
            elif len(out.examples_missed) < 20:
                out.examples_missed.append(f"{name} -> {said or '(silence)'} want {gold[0]}")
        return out

    def derived(self, held_out: Sequence[Tuple[str, str, str]]) -> Paper:
        """Two hops, where the second hop was **never ingested**.

        The only one of the three that says whether more knowledge buys more reasoning. An item is
        used only when the store holds the first hop and does **not** hold the answer in any form,
        which is the rule every held-out paper in this package rests on.
        """
        out = Paper(name="derived")
        learner = getattr(self.brain, "core", None) or getattr(self.brain, "learner", None)
        for subject, predicate, obj in held_out:
            key = self._key(subject)
            if key not in self.subjects:
                continue
            if any(o.lower() == obj.lower() for o in self.by_sp.get((key, predicate), ())):
                continue                # she was told it: not held out
            out.asked += 1
            said = ""
            try:
                got = self.grounder.answer(f"what is {subject}?" if predicate == "is_a"
                                           else f"what does {subject} {predicate}?")
                said = str(getattr(got, "text", "") or "").strip().lower()
            except Exception:  # noqa: BLE001
                said = ""
            if not said and learner is not None:
                try:
                    guess = learner.predict(key, predicate)
                    said = str(getattr(guess, "answer", "") or "").lower() if guess.ok else ""
                except Exception:  # noqa: BLE001
                    said = ""
            if said and (said in obj.lower() or obj.lower() in said):
                out.hit += 1
                if len(out.examples_hit) < 20:
                    out.examples_hit.append(f"{subject} {predicate} {obj}")
            elif len(out.examples_missed) < 20:
                out.examples_missed.append(f"{subject} {predicate} {obj} -> {said or '(silence)'}")
        return out


# --------------------------------------------------------------------------- #
# Sampling a list of subjects from somewhere that is not her corpus
# --------------------------------------------------------------------------- #
def sample_subjects(path: str, *, count: int = DEFAULT_SAMPLE,
                    seed: int = DEFAULT_SEED) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """``(subjects, held-out triples)`` from a triple JSONL, reservoir-sampled in one pass.

    One pass and a reservoir because the sources worth measuring against are hundreds of megabytes
    and a session that loads one into a list to shuffle it has spent its memory on the sampling
    rather than on the brain.
    """
    rng = random.Random(seed)
    subjects: List[str] = []
    triples: List[Tuple[str, str, str]] = []
    seen = 0
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:      # type: ignore[operator]
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            subject = str(row.get("subject") or "").strip()
            if not subject:
                continue
            seen += 1
            got = (subject, str(row.get("predicate") or ""), str(row.get("object") or ""))
            if len(subjects) < count:
                subjects.append(subject)
                triples.append(got)
            else:
                at = rng.randrange(seen)
                if at < count:
                    subjects[at] = subject
                    triples[at] = got
    return subjects, triples


def measure(brain: Any, *, names: Sequence[str],
            held_out: Sequence[Tuple[str, str, str]] = (),
            source: str = "", seed: int = DEFAULT_SEED) -> Report:
    got = Breadth(brain, seed=seed)
    report = Report(facts=got.facts, subjects=len(got.subjects), source=source, seed=seed)
    report.papers.append(got.coverage(names))
    report.papers.append(got.reachable(names))
    if held_out:
        report.papers.append(got.derived(held_out))
    return report


def render(report: Report) -> str:
    lines = [f"breadth — {report.facts} facts over {report.subjects} subjects"
             + (f", sampled against {report.source}" if report.source else ""), "",
             f"{'paper':12} {'asked':>7} {'hit':>7} {'score':>7}"]
    for paper in report.papers:
        lines.append(f"{paper.name:12} {paper.asked:7} {paper.hit:7} {paper.score:7.3f}")
    missed = report.paper("coverage")
    if missed and missed.examples_missed:
        lines += ["", "not known at all: " + ", ".join(missed.examples_missed[:10])]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="how much of the world she has a fact about")
    ap.add_argument("--against", required=True,
                    help="a triple JSONL to sample subjects from — a source, not her corpus")
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--ingest", default="", help="load this triple file first")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    from nyxara.njp.general import load_brain
    from nyxara.njp.ingest import ingest_triples

    brain = load_brain()
    if args.ingest:
        got = ingest_triples(brain, args.ingest, source="breadth")
        print(f"ingested {got.asserted} facts from {args.ingest}")
    names, triples = sample_subjects(args.against, count=args.sample, seed=args.seed)
    report = measure(brain, names=names, held_out=triples,
                     source=os.path.basename(args.against), seed=args.seed)
    print(json.dumps(report.to_dict(), indent=1) if args.json else render(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
