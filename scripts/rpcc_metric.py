#!/usr/bin/env python3
"""NYXARA · scripts/rpcc_metric.py — did a change make NJP *smarter*, or only bigger?

The Master's acceptance bar, and the reason this file exists:

    "NJP ko tabhi smarter bolo jab unknown, paraphrased, causal, counterfactual aur completely
    novel tasks par performance improve ho — sirf cells, synapses, concepts ya parameter count
    badhne par nahi."

Every organ in :mod:`nyxara.njp` already reports counters — cells, synapses, concepts, beliefs,
transitions. None of them is evidence of intelligence: a fabric that grows every turn grows just
as fast when it is learning nothing. So this harness deliberately reports **none** of them. It
runs a fixed battery of turns through a bare :class:`~nyxara.njp.brain.NJPBrain`, fully offline
(no LLM, no network, no weights), and scores only what the bar names.

**The half that keeps it honest.** Roughly a third of the battery is cases where the correct
answer is *"I don't know"* — a question she was never told the answer to, or a counterfactual whose
causal direction nothing has established. Those cases score a PASS for **abstaining** and a FAIL
for answering. Without them a metric of this shape can be maximised by a system that answers
everything, which is the failure the whole repo is written against; with them, fluency is a cost.

Run:  ``python scripts/rpcc_metric.py``            (whole battery)
      ``python scripts/rpcc_metric.py --json out.json``   (machine-readable, for before/after)
      ``python scripts/rpcc_metric.py --only causal``     (one category)

It exits 0 always. It prints a report; it does not gate CI. Pure standard library — the same
constraint every organ it measures is held to.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------- #
# What a case may expect
# --------------------------------------------------------------------------- #

#: She must produce a non-empty answer.
ANSWERS = "answers"

#: She must NOT produce an answer — the honest "I don't know". A pass here is a *refusal*.
ABSTAINS = "abstains"

#: She must answer, and the answer must contain every one of these tokens (lowercased).
CONTAINS = "contains"


@dataclass
class Case:
    """One probe: teach her some turns, ask one thing, and say what a right answer looks like."""

    category: str
    setup: Tuple[str, ...]
    probe: str
    expect: str
    tokens: Tuple[str, ...] = ()
    why: str = ""

    def judge(self, answer: str, epistemic: str) -> bool:
        text = (answer or "").strip().lower()
        answered = bool(text) and epistemic != "unknown"
        if self.expect == ABSTAINS:
            return not answered
        if not answered:
            return False
        if self.expect == CONTAINS:
            return all(t.lower() in text for t in self.tokens)
        return True


# --------------------------------------------------------------------------- #
# The battery
# --------------------------------------------------------------------------- #

_PLANT: Tuple[str, ...] = (
    "paudhe ko paani chahiye",
    "paani se growth badhti hai",
    "light se growth badhti hai",
)

#: A domain built out of words that appear nowhere in her seeds, her corpus, or this repo — so a
#: pass cannot come from having been told the answer somewhere else.
_NOVEL: Tuple[str, ...] = (
    "blarf se quon badhta hai",
    "quon se drine badhta hai",
)


BATTERY: Tuple[Case, ...] = (

    # -- causal ------------------------------------------------------------- #
    # A law was stated; the question asks for it back. This is the floor: a system that fails
    # here has no world model at all, whatever its counters say.
    Case("causal", _PLANT, "growth kaise badhti hai", CONTAINS, ("paani",),
         "a stated cause, asked for directly"),
    Case("causal", _PLANT, "paudhe ko kya chahiye", CONTAINS, ("paani",),
         "a stated requirement, asked for directly"),
    Case("causal", _PLANT, "paani se kya hota hai", ANSWERS,
         "the same law read forwards instead of backwards"),
    Case("causal", _NOVEL, "quon kaise badhta hai", CONTAINS, ("blarf",),
         "the same shape in a domain with no prior content"),
    Case("causal", _NOVEL, "blarf se kya hota hai", ANSWERS,
         "forward reading, novel domain"),

    # -- paraphrased -------------------------------------------------------- #
    # One question, many surfaces. A regex layer passes the surface it was written for and drops
    # the rest; that is precisely the gap this measures.
    Case("paraphrased", _PLANT, "growth kaise badhti hai", CONTAINS, ("paani",)),
    Case("paraphrased", _PLANT, "growth kis se badhti hai", CONTAINS, ("paani",)),
    Case("paraphrased", _PLANT, "kya cheez growth badhati hai", CONTAINS, ("paani",)),
    Case("paraphrased", _PLANT, "what makes growth increase", CONTAINS, ("paani",)),
    Case("paraphrased", _PLANT, "growth badhane ke liye kya chahiye", CONTAINS, ("paani",)),
    Case("paraphrased", _PLANT, "paudhe ki growth ka karan kya hai", CONTAINS, ("paani",)),

    # -- counterfactual ----------------------------------------------------- #
    # The reproduced failure this whole effort started from. The engine answers when called
    # directly; the question cannot reach it.
    Case("counterfactual", _PLANT, "agar paani aadha kar doon to kya hoga", ANSWERS,
         "THE reproduced bug: parsed as an intervention, never reached the do-operator"),
    Case("counterfactual", _PLANT, "agar paani aadha ho to kya hoga", ANSWERS,
         "same question, the variable-first surface"),
    Case("counterfactual", _PLANT, "what if the water is halved", ANSWERS,
         "same question, English passive"),
    Case("counterfactual", _PLANT, "agar paani band kar doon to kya hoga", ANSWERS,
         "the zero intervention — a different question from halving"),
    Case("counterfactual", _NOVEL, "agar blarf aadha kar doon to kya hoga", ANSWERS,
         "intervention in a domain with no prior content"),

    # -- unknown (a pass here is a REFUSAL) ---------------------------------- #
    # These exist so the battery cannot be maximised by answering everything.
    Case("unknown", _PLANT, "mangal grah par kitne log hain", ABSTAINS,
         "never told, unknowable — answering it is a fabrication"),
    Case("unknown", _PLANT, "paudhe ka rang kya hai", ABSTAINS,
         "in-domain but never stated — the tempting one"),
    Case("unknown", _NOVEL, "drine ka wazan kitna hai", ABSTAINS,
         "novel domain, never stated"),
    Case("unknown", (), "growth kaise badhti hai", ABSTAINS,
         "the same question with NOTHING taught — she must not know it"),
    Case("unknown", _PLANT, "agar zephyr aadha kar doon to kya hoga", ABSTAINS,
         "an intervention on a variable she has never heard of"),

    # -- novel -------------------------------------------------------------- #
    # Structure she has met, content she has not. A chain stated in two hops, asked in one.
    Case("novel", _NOVEL, "drine kaise badhta hai", CONTAINS, ("quon",),
         "the second hop of a chain, stated directly"),
    Case("novel", _NOVEL + ("drine se flump badhta hai",), "flump kaise badhta hai",
         CONTAINS, ("drine",), "third hop"),
    Case("novel", ("glorp zeph ko badhata hai", "zeph drine ko badhata hai"),
         "glorp se kya hota hai", ANSWERS, "a different stated surface, novel content"),
    Case("novel", ("the widget drives the sprocket",), "what does the widget drive",
         ANSWERS, "English, novel content"),
)


CATEGORIES: Tuple[str, ...] = ("causal", "paraphrased", "counterfactual", "unknown", "novel")


# --------------------------------------------------------------------------- #
# Running it
# --------------------------------------------------------------------------- #

@dataclass
class Outcome:
    case: Case
    answer: str = ""
    epistemic: str = ""
    passed: bool = False
    error: str = ""


def _fresh_brain() -> Any:
    """A bare brain per case.

    Per case rather than per battery on purpose: a brain that has already seen the plant domain
    would answer the novel-domain probes from it, and the whole point of the novel category is
    that she has not.
    """
    from nyxara.njp.brain import NJPBrain
    return NJPBrain()


def run_case(case: Case) -> Outcome:
    out = Outcome(case=case)
    try:
        brain = _fresh_brain()
        for turn in case.setup:
            brain.think(turn)
        thought = brain.think(case.probe)
        out.answer = str(getattr(thought, "answer", "") or "")
        out.epistemic = str(getattr(thought, "epistemic", "") or "")
        out.passed = case.judge(out.answer, out.epistemic)
    except Exception as exc:  # noqa: BLE001 — a crashing probe is a failing probe, not a crash
        out.error = f"{type(exc).__name__}: {exc}"
        out.passed = False
    return out


def run(only: Optional[str] = None) -> List[Outcome]:
    cases = [c for c in BATTERY if only is None or c.category == only]
    return [run_case(c) for c in cases]


def report(outcomes: Sequence[Outcome]) -> Dict[str, Any]:
    by_cat: Dict[str, Dict[str, int]] = {}
    for out in outcomes:
        bucket = by_cat.setdefault(out.case.category, {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(out.passed)
    passed = sum(1 for o in outcomes if o.passed)
    return {
        "total": len(outcomes),
        "passed": passed,
        "failed": len(outcomes) - passed,
        "by_category": by_cat,
        "cases": [
            {"category": o.case.category, "probe": o.case.probe,
             "expect": o.case.expect, "passed": o.passed,
             "answer": o.answer, "epistemic": o.epistemic, "error": o.error}
            for o in outcomes
        ],
    }


def render(outcomes: Sequence[Outcome]) -> None:
    print("=" * 84)
    print("NYXARA · RPCC metric — unknown / paraphrased / causal / counterfactual / novel")
    print("no cells, no synapses, no concept counts: none of those is evidence of intelligence")
    print("=" * 84)
    current = ""
    for out in outcomes:
        if out.case.category != current:
            current = out.case.category
            note = "  (a PASS here is a REFUSAL)" if current == "unknown" else ""
            print(f"\n  {current.upper()}{note}")
        mark = "pass" if out.passed else "FAIL"
        probe = out.case.probe if len(out.case.probe) <= 44 else out.case.probe[:41] + "..."
        if out.error:
            shown = f"!! {out.error}"
        elif out.answer:
            shown = out.answer if len(out.answer) <= 46 else out.answer[:43] + "..."
        else:
            shown = "(abstained)"
        print(f"    [{mark}]  {probe:<46} → {shown}")

    data = report(outcomes)
    print("\n" + "-" * 84)
    for category in CATEGORIES:
        bucket = data["by_category"].get(category)
        if bucket is None:
            continue          # a category that was not run is ABSENT, not reported as zero
        print(f"  {category:<16} {bucket['passed']:>3} / {bucket['total']:<3}")
    print("-" * 84)
    print(f"  TOTAL            {data['passed']:>3} / {data['total']:<3}")
    print("\n  This number is the only claim worth making. Compare it before and after a change;")
    print("  a change that does not move it did not make her smarter, whatever else it moved.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", metavar="PATH", help="also write the report as JSON")
    parser.add_argument("--only", choices=CATEGORIES, help="run one category")
    args = parser.parse_args(argv)

    outcomes = run(only=args.only)
    render(outcomes)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report(outcomes), handle, indent=2)
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
