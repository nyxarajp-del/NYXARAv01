"""NYXARA · njp/entailschool.py — is a rule about a pair of sentences worth anything (📏).

7,226 inference pairs, split once and deterministically: the first seven tenths to learn from, the
last three tenths never seen. Four numbers, and three exist to keep the first honest.

* **accuracy** over every held-out pair, counting an abstention as a miss. The number a caller
  actually gets.
* **the base rate** — always answer the commonest label. On this corpus that is *it is not possible
  to tell*, and it is 0.42, which any learner must beat to have done anything.
* **no induction at all** — the same reasoner with learning off. It abstains on everything, which
  is the honest floor for something that has been shown pairs and concluded nothing.
* **when she does answer** — accuracy over the pairs she did not abstain on, printed beside the
  share she answered. A reasoner that answers a tenth of the pairs at 0.9 and one that answers all
  of them at 0.5 are different things and one number cannot say which is which.

And the sweep. ``purity`` is how clean a rule must be to be kept, and it is not a taste: it is set
by running the whole examination at each value and reading the held-out column. A threshold chosen
any other way is a knob turned until the training number looked nice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.entail import Pair, Reasoner, read_pairs

__all__ = ["Result", "split", "examine", "sweep", "knowledge_gap", "run"]

#: Relations the fact store holds that could link two words of a pair. Nothing narrower: the point
#: of :func:`knowledge_gap` is to be generous about what would count as knowing something.
LINKS: Tuple[str, ...] = ("excludes", "is_a", "has_property", "capable_of", "part_of",
                          "used_for", "at_location")

#: The share of the corpus learned from. The rest is never shown to the reasoner.
TRAIN = 0.7


@dataclass
class Result:
    name: str = ""
    right: int = 0
    asked: int = 0
    answered: int = 0
    rules: int = 0
    by_label: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        """Over every pair, an abstention counting as a miss."""
        return round(self.right / self.asked, 4) if self.asked else 0.0

    @property
    def when_answered(self) -> float:
        return round(self.right / self.answered, 4) if self.answered else 0.0

    @property
    def coverage(self) -> float:
        return round(self.answered / self.asked, 4) if self.asked else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"accuracy": self.accuracy, "when_answered": self.when_answered,
                "coverage": self.coverage, "asked": self.asked, "rules": self.rules,
                "by_label": {k: {"right": v[0], "asked": v[1]} for k, v in self.by_label.items()}}

    def render(self) -> str:
        return (f"{self.name:<16} {self.accuracy:.3f}   "
                f"answered {self.coverage:.3f} of them at {self.when_answered:.3f}   "
                f"({self.rules} rules)")


def split(pairs: Optional[Sequence[Pair]] = None,
          train: float = TRAIN) -> Tuple[List[Pair], List[Pair]]:
    """One deterministic cut. A held-out set that moves between runs is not held out."""
    rows = list(pairs if pairs is not None else read_pairs())
    cut = int(len(rows) * float(train))
    return rows[:cut], rows[cut:]


def _mark(reasoner: Reasoner, held: Sequence[Pair], name: str, *,
          fallback: bool = False) -> Result:
    out = Result(name=name, rules=len(reasoner.rules))
    speak = reasoner.guess if fallback else reasoner.answer
    for pair in held:
        out.asked += 1
        right, asked = out.by_label.get(pair.label, (0, 0))
        answer, _why = speak(pair.premise, pair.hypothesis)
        hit = int(answer == pair.label)
        if answer != "unknown":
            out.answered += 1
        out.right += hit
        out.by_label[pair.label] = (right + hit, asked + 1)
    return out


def examine(purity: float = 0.72, **kwargs: Any) -> Dict[str, Result]:
    learn, held = split()
    out: Dict[str, Result] = {}

    taught = Reasoner(purity=purity, **kwargs)
    taught.learn_from(learn)
    out["taught"] = _mark(taught, held, "taught")
    out["with_fallback"] = _mark(taught, held, "taught + fallback", fallback=True)

    blind = Reasoner(purity=purity, learn=False, **kwargs)
    blind.learn_from(learn)
    out["no_rules"] = _mark(blind, held, "no rules")

    commonest = max({p.label for p in learn},
                    key=lambda label: sum(1 for p in learn if p.label == label))
    base = Result(name=f"base rate ({commonest})")
    for pair in held:
        base.asked += 1
        base.answered += 1
        base.right += int(pair.label == commonest)
    out["base_rate"] = base
    return out


def sweep(values: Sequence[float] = (0.55, 0.60, 0.65, 0.72, 0.80, 0.90, 1.00),
          **kwargs: Any) -> List[Tuple[float, Result]]:
    """The examination at each threshold. What sets ``purity`` is this table, not a preference."""
    return [(value, examine(value, **kwargs)["taught"]) for value in values]


def knowledge_gap(brain: Any = None, sample: int = 300) -> Dict[str, Any]:
    """Is the knowledge the hard cases need actually in the store? Asked, not assumed.

    Separating *no* from *it is not possible to tell* needs to know that performing in a
    competition and watching television are incompatible. Word overlap cannot see that; a fact
    store could. So this counts, over held-out pairs of each label, how often the store holds
    **any** relation at all between a premise word and a hypothesis word.

    The answer decides whether the near misses are a defect of this module or a gap in the corpus,
    and it is not the same question as whether she has heard of the words.
    """
    from nyxara.njp.entail import _content

    if brain is None:
        from nyxara.njp.general import load_brain
        brain = load_brain(broad=True)
    grounder = getattr(brain, "grounder", None)
    if grounder is None:
        return {}
    _learn, held = split()
    out: Dict[str, Any] = {"facts": len(getattr(grounder, "facts", {}) or {})}

    def linked(pair: Pair) -> bool:
        hypothesis = _content(pair.hypothesis)
        for word in _content(pair.premise):
            for relation in LINKS:
                for triple in grounder.facts.get((grounder._key(word), relation), ()):
                    if str(triple.object).lower() in hypothesis:
                        return True
        return False

    words: set = set()
    for label in ("no", "it is not possible to tell", "yes"):
        rows = [p for p in held if p.label == label][:sample]
        out[label] = {"linked": sum(1 for p in rows if linked(p)), "of": len(rows)}
        for pair in rows[:150]:
            words |= _content(pair.premise) | _content(pair.hypothesis)
    out["words_known"] = sum(
        1 for word in words
        if any((grounder._key(word), relation) in grounder.facts for relation in LINKS))
    out["words"] = len(words)
    return out


def run() -> Dict[str, Any]:
    got = examine()
    return {"held_out": {name: result.to_dict() for name, result in got.items()},
            "sweep": [{"purity": value, **result.to_dict()} for value, result in sweep()]}


def main() -> None:  # pragma: no cover — a report, not a test
    pairs = read_pairs()
    if not pairs:
        print("no corpus; run scripts/build_reasoning_corpus.py")
        return
    learn, held = split(pairs)
    print(f"{len(pairs)} pairs — {len(learn)} learned from, {len(held)} held out\n")
    print("purity   held-out   answered   when answered   rules")
    for value, result in sweep():
        print(f"  {value:.2f}     {result.accuracy:.3f}      {result.coverage:.3f}"
              f"          {result.when_answered:.3f}        {result.rules}")
    print()
    got = examine()
    for name in ("base_rate", "no_rules", "taught", "with_fallback"):
        print("  " + got[name].render())
    print("\nby label, taught:")
    for label, (right, asked) in sorted(got["taught"].by_label.items()):
        print(f"    {label:<28} {right}/{asked}  {right / max(1, asked):.3f}")
    reasoner = Reasoner()
    reasoner.learn_from(learn)
    print("\nwhat she worked out:")
    print("\n".join(reasoner.render().splitlines()[:6]))
    print("\nis the knowledge the hard cases need even in the store?")
    gap = knowledge_gap()
    if gap:
        print(f"    {gap['facts']} facts; she has heard of "
              f"{gap['words_known']}/{gap['words']} of the content words")
        for label in ("no", "it is not possible to tell", "yes"):
            row = gap.get(label, {})
            print(f"    {label:<28} a stored relation links the two sentences in "
                  f"{row.get('linked')}/{row.get('of')}")


if __name__ == "__main__":  # pragma: no cover
    main()
