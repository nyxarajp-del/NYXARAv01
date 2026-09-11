"""NYXARA · njp/fieldwork.py — the diagnostic loop taken out of its fixtures (🔬, NJP V.83).

V.74 through V.82 built a stack that measures a measurement, attributes a failure, finds where a
capability stops reaching, and runs the whole thing as a loop. Every one of them was validated the
same way: a **fixture** whose answer was known because it had been built that way, retrodicted, and
scored on catches and on false alarms together.

That is the right way to build such a thing and it is not sufficient, for one reason. A fixture is
written by the same hand that writes the organ, so it inherits that hand's idea of what can go
wrong. The failure modes it cannot contain are exactly the ones nobody had thought of — which is
the set that matters.

So this module points the finished stack at a **real** organ and reports what came back. The target
is the span stage of :mod:`nyxara.njp.finding`: given the sentence that contains the answer, pick
the span inside it. It is chosen because its cause is already known by hand — two versions of work
went into the ranker before a decomposition found that the candidate generator proposes about a
hundred and twenty-six spans per sentence, and the ranker was never the thing holding the number
down. If the loop is worth anything it should find that by itself.

**What it found instead, and this is the version's whole content:**

1. It declined to name a cause, and it was right to. Every repair offered was genuinely refuted:
   more data, another ranker and richer readings each moved the score by 0.0000. *These levers do
   not reach this* is a true and useful thing to be told, and a diagnostician that produced a word
   anyway would have sent the next version somewhere.

2. ``reachability`` came back **refuted**, which surprised me and should not have. The right answer
   is in the pool for 0.700 of items and the stage scores 0.032, so the ceiling is nowhere near
   binding. *The pool is too big* and *the answer is not in the pool* are different claims and only
   the second is a ceiling. My hand-diagnosis was the first, and the organ has no hypothesis for
   it, which is a real gap and is named below rather than papered over.

3. The one repair built from my own hand-diagnosis — keep the twelve shortest candidates instead of
   a hundred and twenty-six — made the score **worse**, 0.0320 → 0.0160. Knowing the cause does not
   hand you the repair.

4. And that last repair is why this version changed the organ. Capping the pool also threw the
   right answer out of it: reachable fell 0.700 → 0.160. The attributor recorded ``budget:
   refuted``, meaning *more search would not have helped*, on the strength of an experiment that
   had changed two things at once. The module's own first rule is that an experiment changes
   exactly one thing, and it was checking its callers for everything except that. V.83 adds
   :data:`~nyxara.njp.attribution.SPOILED`: the ceiling is re-read after every repair, and one that
   moved it is reported as ``spoiled`` — ran, tested nothing — instead of as a refutation. Where
   every repair is refuted and one is spoiled, ``floor`` is withheld, because *no lever reaches
   this* is not shown by a lever nobody pulled.

The gap in point 2 is the honest debt of this version. The organ tests *is the answer producible*
and does not test *is it producible among so many alternatives that nothing could pick it out* —
a distractor-count hypothesis, which needs an experiment that thins the pool **without** dropping
what it is thinning toward. That experiment is not written here, because the obvious way to write
it is the one that just failed.

Needs the reading corpus, so it is a report rather than a test: nothing here runs in CI.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.attribution import Attribution, Failure, attribute
from nyxara.njp.finding import (Finder, Reading, Setting, Span, candidates, gold_key, probe,
                                read_passages)
from nyxara.njp.findingschool import SEED, TRAIN, gold_sentence, taught_finder
from nyxara.njp.measurement import Benchmark, Critique
from nyxara.njp.space import Held, Intervention, Knob, Map, gaps, propose, survey

__all__ = ["Stage", "Fieldwork", "span_stage", "diagnose", "explore", "run", "HELD",
           "SMALL_POOL", "LEARN_FROM", "MORE", "SPAN_KNOBS"]

#: How many readings the stage is taught from as found, and how many the `more data` experiment
#: gets. They must differ, and :func:`diagnose` refuses to run if they do not — see :func:`_cut`.
LEARN_FROM = 1_500
MORE = 6_000

#: How many held-out readings the stage is examined on. Every experiment uses the same ones — the
#: rule the fixtures broke twice before the organ did.
HELD = 250

#: How many candidates the "smaller pool" repair keeps. Twelve, against the hundred and twenty-six
#: the generator proposes. It is the repair my own hand-diagnosis implies, and it is kept in this
#: module after failing, because a repair that failed is the evidence.
SMALL_POOL = 12


@dataclass
class Stage:
    """One run of the span stage, with everything the critic needs to check the number."""

    name: str = ""
    bench: Optional[Benchmark] = None
    #: Average candidates offered per item — the quantity my hand-diagnosis was about, and the one
    #: the attributor has no hypothesis for.
    pool: float = 0.0
    #: Share of items whose right answer was among those candidates at all.
    reachable: float = 0.0
    rows: int = 0
    #: How many readings the finder was taught from. Recorded because the first version of this
    #: module asked for twice the data and got the same data — see :func:`_cut`.
    learned_from: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "pool": self.pool, "reachable": self.reachable,
                "rows": self.rows, "learned_from": self.learned_from,
                "score": self.bench.score() if self.bench else 0.0}

    def render(self) -> str:
        got = self.bench.score() if self.bench else 0.0
        return (f"  {self.name:<18} {self.rows} rows from {self.learned_from} taught, "
                f"{self.pool:.1f} candidates each, gold reachable {self.reachable:.3f}, "
                f"scored {got:.4f}")


def _shortest(spans: Sequence[Span], _reading: Reading) -> List[Span]:
    """The repair my hand-diagnosis implies: fewer things to choose between.

    It is also, and this was not noticed until the attributor was made to check, a repair that
    throws the right answer away for most items on its way past. Kept exactly as it was written.
    """
    return sorted(spans, key=lambda s: len(s.text))[:SMALL_POOL]


def _cut(readings: Sequence[Reading]) -> List[List[Reading]]:
    """The passage-level split :mod:`~nyxara.njp.findingschool` makes, **without its caps**.

    ``findingschool.split`` trims the learn side to 1,500 rows and the held side to 600, which is
    right for a fixed comparison between versions and wrong here. The first draft of this module
    asked for twice the data by writing ``learn[:LEARN_FROM * 2]`` against that function's output —
    and got the same 1,500 rows back, because they had already been cut. The `more data` experiment
    therefore changed nothing, came back flat, and was recorded as ``data: refuted``: *more examples
    would not have helped*, concluded from an experiment that showed no more examples.

    That is the same defect this version found in the attributor, in the file that found it, on the
    same afternoon — which says something about how quietly it happens. It was caught by reading
    the call rather than by reading the number, so :func:`diagnose` now asserts the two training
    sets differ in size instead of trusting that they do.

    Split by **passage**, not by row, for the reason ``split`` gives: SQuAD asks a dozen questions
    of one paragraph, so cutting rows puts the same paragraph on both sides.
    """
    rows = [r for r in readings if r.holds_its_answer()]
    by_passage: Dict[str, List[Reading]] = {}
    for row in rows:
        by_passage.setdefault(row.passage, []).append(row)
    passages = sorted(by_passage)
    random.Random(SEED).shuffle(passages)
    at = int(len(passages) * TRAIN)
    return [[r for passage in passages[:at] for r in by_passage[passage]],
            [r for passage in passages[at:] for r in by_passage[passage]]]


def _thinned(spans: Sequence[Span], reading: Reading) -> List[Span]:
    """The experiment V.83 ended owing: thin the pool **without dropping what it is thinning toward**.

    :func:`_shortest` asked *does the ranker do better with fewer things to choose between* and
    could not answer, because it also threw the right answer away for most items. This keeps the
    same twelve and forces the gold span in among them, so the only thing that changes is **how
    many wrong candidates are competing** — which is the quantity my hand-diagnosis was about and
    the one no hypothesis in :mod:`nyxara.njp.attribution` tests.

    It consults the gold answer, so it is an **oracle** and can never be a repair: at run time
    nobody knows which span is right. It is an experiment, and confusing the two is how a
    measurement gets announced as a fix.
    """
    want = gold_key(reading.answer)
    short = sorted(spans, key=lambda s: len(s.text))[:SMALL_POOL]
    if any(s.key == want for s in short):
        return short
    gold = next((s for s in spans if s.key == want), None)
    return short if gold is None else [gold] + short[:SMALL_POOL - 1]


def span_stage(engine: Finder, held: Sequence[Reading], *, name: str = "as found",
               learned_from: int = 0, taught: Sequence[Reading] = (),
               pool: Optional[Callable[[Sequence[Span], Reading], Sequence[Span]]] = None
               ) -> Stage:
    """Score the span stage with the gold sentence handed to it, so only the span choice is tested.

    The sentence is given rather than found. That is deliberate and it is what makes this a stage
    rather than the whole reader: a number that mixes *which sentence* with *which span inside it*
    cannot say which of them is failing, and the whole point of the stack above is that such a
    number is not evidence about either.
    """
    items: List[int] = []
    gold: List[str] = []
    said: List[str] = []
    reach: List[bool] = []
    sizes: List[int] = []
    kept: List[Reading] = []
    for reading in held:
        at = gold_sentence(reading)
        fixed = Setting.of(reading)
        if at >= len(fixed.spans):
            continue
        sentence, start = fixed.spans[at]
        offered = candidates(sentence)
        if not offered:
            continue
        want = gold_key(reading.answer)
        use = list(pool(offered, reading)) if pool else offered
        if not use:
            continue
        sizes.append(len(use))
        shape = engine.wants(reading.question)
        best: Any = (0, 0.0)
        chosen: Optional[Span] = None
        for span in use:
            placed = Span(text=span.text, start=span.start + start,
                          end=span.end + start, sentence=at)
            score = engine._score(probe(reading, placed, shape, use_shape=engine.use_shape,
                                        setting=fixed), engine.rules)
            if score > best:
                best, chosen = score, span
        items.append(len(items))
        kept.append(reading)
        gold.append(want)
        said.append(chosen.key if chosen is not None else "")
        reach.append(any(s.key == want for s in use))
    answers = dict(zip(items, said))
    # Identity here is the **passage**, not the row, for the reason `_cut` splits on it: SQuAD asks
    # a dozen questions of one paragraph, and two rows sharing a paragraph are not two independent
    # items however differently they are worded.
    where = {i: r.passage for i, r in zip(items, kept)}
    return Stage(name=name, rows=len(items), learned_from=learned_from,
                 pool=round(sum(sizes) / max(1, len(sizes)), 4),
                 reachable=round(sum(reach) / max(1, len(reach)), 4),
                 bench=Benchmark(name=name, items=items, gold=gold,
                                 predict=lambda i: answers.get(i, ""),
                                 reachable=lambda i, _w: reach[i],
                                 train=[r.passage for r in taught],
                                 key=lambda x: where.get(x, x)))


@dataclass
class Fieldwork:
    """What the stack made of a real organ, kept together with the measurements it ran on."""

    stages: List[Stage] = field(default_factory=list)
    attribution: Optional[Attribution] = None
    critique: Optional[Critique] = None

    @property
    def as_found(self) -> Optional[Stage]:
        return self.stages[0] if self.stages else None

    def to_dict(self) -> Dict[str, Any]:
        return {"stages": [s.to_dict() for s in self.stages],
                "root": self.attribution.root if self.attribution else "",
                "spoiled": self.attribution.spoiled if self.attribution else [],
                "refuted": self.attribution.refuted if self.attribution else [],
                "untested": self.attribution.untested if self.attribution else []}

    def render(self) -> str:
        lines = [s.render() for s in self.stages]
        if self.attribution is not None:
            lines.append(self.attribution.render())
        return "\n".join(lines)


def diagnose(readings: Optional[Sequence[Reading]] = None) -> Fieldwork:
    """Point the V.74–V.83 stack at the span stage and report what it says, including nothing.

    Four experiments, and each one is checked here for being the single change it claims to be
    before it is handed over: `more data` must genuinely see more rows, and the attributor checks
    the rest by re-reading the ceiling. Neither check was in the first draft and both caught
    something.
    """
    learn, held = _cut(readings if readings is not None else read_passages())
    held = list(held[:HELD])
    if len(learn) <= LEARN_FROM:
        raise ValueError(f"only {len(learn)} rows to learn from, so `more data` is the same data")
    more = min(MORE, len(learn))
    base = taught_finder(learn[:LEARN_FROM])
    found = span_stage(base, held, learned_from=LEARN_FROM, taught=learn[:LEARN_FROM])
    seen: List[Stage] = [found]

    def _run(build: Callable[[], Stage]) -> Benchmark:
        stage = build()
        seen.append(stage)
        return stage.bench

    failure = Failure(
        name="span stage, inside the gold sentence",
        bench=found.bench,
        more_data=lambda: _run(lambda: span_stage(
            taught_finder(learn[:more]), held, name="more data", learned_from=more,
            taught=learn[:more])),
        other_algorithm=lambda: _run(lambda: span_stage(
            taught_finder(learn[:LEARN_FROM], purity=0.5), held,
            name="other algorithm", learned_from=LEARN_FROM, taught=learn[:LEARN_FROM])),
        richer_reading=lambda: _run(lambda: span_stage(
            taught_finder(learn[:LEARN_FROM], use_shape=True), held,
            name="richer readings", learned_from=LEARN_FROM, taught=learn[:LEARN_FROM])),
        more_budget=lambda: _run(lambda: span_stage(
            base, held, name="a smaller pool", learned_from=LEARN_FROM,
            taught=learn[:LEARN_FROM], pool=_shortest)))
    got = attribute(failure)
    return Fieldwork(stages=seen, attribution=got, critique=got.critique)


def run() -> Dict[str, Any]:  # pragma: no cover — a report over the corpus, not a test
    out = diagnose()
    print(out.render())
    print(f"\n  the pool the hand-diagnosis was about: {out.as_found.pool:.1f} candidates an item, "
          f"and no hypothesis here tests it")
    return out.to_dict()


if __name__ == "__main__":  # pragma: no cover
    run()


# --------------------------------------------------------------------------------------------- #
#  V.84 — the same organ, drawn as a map instead of a list
# --------------------------------------------------------------------------------------------- #
#: What could be varied about a run of the span stage. Supplied, like every other map in
#: :mod:`nyxara.njp.space` — the claim is not that these were discovered, it is that four
#: experiments landing in two of five regions is visible here and was not visible as a list.
SPAN_KNOBS: Tuple[Knob, ...] = (
    Knob("how many examples", "given", "readings the finder is taught from"),
    Knob("which rule family", "method", "the purity bar the rules must clear"),
    Knob("which readings", "shown", "whether the answer-shape organ is consulted"),
    Knob("how answers are compared", "scored", "`Span.key` against `gold_key`"),
    Knob("what counts as answering", "scored", "abstention against a wrong span"),
    Knob("how candidates were made", "built", "the generator that proposes spans"),
    Knob("how many candidates compete", "built", "how many wrong ones are in the way"),
    Knob("how the split was cut", "built", "by passage, and where"),
)


def explore(readings: Optional[Sequence[Reading]] = None) -> Map:
    """Run the four experiments the attributor holds, plus the one V.83 ended owing, on a map.

    The fifth is the point. ``remove(how many candidates compete) hold(the right answer stays
    producible)`` is the experiment that could not be written before :mod:`nyxara.njp.space`,
    because holding something fixed was a property of the checker rather than of the experiment.
    It is an oracle and therefore never a repair — but if it moves the number, the cause is in
    `built`, and if it does not, my hand-diagnosis of this organ was wrong.
    """
    learn, held = _cut(readings if readings is not None else read_passages())
    held = list(held[:HELD])
    if len(learn) <= LEARN_FROM:
        raise ValueError(f"only {len(learn)} rows to learn from, so `more data` is the same data")
    more = min(MORE, len(learn))
    base = taught_finder(learn[:LEARN_FROM])
    found = span_stage(base, held, learned_from=LEARN_FROM, taught=learn[:LEARN_FROM])

    keeps = Held(name="the right answer stays producible", read=lambda st: st.reachable)
    same_rows = Held(name="the same items are examined", read=lambda st: float(st.rows))
    both = (keeps, same_rows)
    what: List[Intervention] = [
        Intervention(verb="add", on="how many examples", holds=both,
                     run=lambda: span_stage(taught_finder(learn[:more]), held,
                                            name="more data", learned_from=more,
                                            taught=learn[:more])),
        Intervention(verb="replace", on="which rule family", holds=both,
                     run=lambda: span_stage(taught_finder(learn[:LEARN_FROM], purity=0.5), held,
                                            name="other algorithm", learned_from=LEARN_FROM,
                                            taught=learn[:LEARN_FROM])),
        Intervention(verb="add", on="which readings", holds=both,
                     run=lambda: span_stage(taught_finder(learn[:LEARN_FROM], use_shape=True),
                                            held, name="richer readings",
                                            learned_from=LEARN_FROM, taught=learn[:LEARN_FROM])),
        Intervention(verb="remove", on="how many candidates compete", holds=both,
                     says="the repair my hand-diagnosis implies, exactly as it failed",
                     run=lambda: span_stage(base, held, name="a smaller pool",
                                            learned_from=LEARN_FROM, taught=learn[:LEARN_FROM],
                                            pool=_shortest)),
        Intervention(verb="remove", on="how many candidates compete", holds=both,
                     deployable=False,
                     says="the same thinning, with the right answer held in the pool",
                     run=lambda: span_stage(base, held, name="a smaller pool, answer kept",
                                            learned_from=LEARN_FROM, taught=learn[:LEARN_FROM],
                                            pool=_thinned)),
    ]
    return survey("the span stage, as a map", SPAN_KNOBS, what, found,
                  score=lambda st: st.bench.score() if st.bench else 0.0)


def chart() -> Dict[str, Any]:  # pragma: no cover — a report over the corpus, not a test
    drawn = explore()
    print(drawn.render())
    asked = propose(drawn)
    empty = [g.region for g in gaps(drawn)]
    print(f"\n  {len(asked)} experiments the map says are worth building in {empty or 'nowhere'};"
          f" the first four:")
    for one in asked[:4]:
        print(f"    {one.name}")
    return drawn.to_dict()
