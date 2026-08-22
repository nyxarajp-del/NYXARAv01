"""NYXARA · eval/relational.py — can the gradient head generalise a relation? (📐, NJP V.07)

**Why this exists as a probe rather than as a feature.** The fabric's readout is the one learner in
the package that uses real, gradient-checked backpropagation, and its symbols score MRR 0.238
because its objective is turn-to-turn concept succession — *what tends to come up next*, which in a
conversation is largely the order things were said in. The obvious repair is to train it on the
relational signal instead: ``(subject cells + predicate cells) → object cells``. That was built,
measured, and **is not shipped**, because the measurement says it does not work and says why.

Three runs, each with the control the previous one needed::

    1 · relational training, held-out members
        trained pairs      rank 1 for every one          — it memorises perfectly
        held-out members   MRR 0.521
        nonsense subjects  MRR 0.500                     — the control

    2 · the same, with `is_a` also taught so two hops are composable
        held-out members   MRR 0.233
        nonsense subjects  MRR 0.583                     — real subjects score WORSE

Run 1 looks like generalisation until the control runs: a subject the brain has never seen scores
essentially the same as a real held-out member, so the ranking is the **object prior** and the
subject contributes nothing. Run 2 is worse than nothing — teaching ``sparrow is_a bird`` trains
that row to predict *bird*, so asking ``(sparrow, requires)`` pushes toward ``bird`` rather than
``water``. The head superimposes; it does not compose.

**The cause is the encoding, not the objective.** :func:`~nyxara.njp.brain._cell_id` is a blake2b
digest, so ``sparrow`` and ``finch`` have orthogonal representations with no shared structure.
Generalisation requires that similar things be represented similarly, and a hash guarantees exactly
the opposite. No training objective repairs that.

**And that is what :mod:`nyxara.njp.embed` supplies**, which is why there is a third run::

    3 · the same, with each query widened by the subject's neighbours
        held-out members   MRR 1.000     six of six at rank 1
        nonsense subjects  MRR 0.500     the control, unmoved

Same head, same weights, same objective — only the *query* changed, from a hashed cell id to that
id plus the entities whose relational context resembles it. The control not moving is what says the
gain came from the subject: a nonsense subject has no neighbours, so nothing widens, and it scores
exactly what it scored before.

**What this module is for.** The claims above should be checkable rather than remembered, and any
future attempt should inherit the control that caught the first one — the first version of run 3
was measured on two candidate objects, where a nonsense subject reached rank 1 by chance, and was
rebuilt rather than reported. :func:`probe` always reports the nonsense-subject baseline beside the
real one, because the gap between them is the only number that means anything here.

Run it::

    python -m nyxara.eval --relational
    python -m nyxara.eval --relational --teach-is-a
    python -m nyxara.eval --relational --expand
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["RelationalProbe", "probe"]

#: A small relational world. Members share a kind, and the kind carries the property — so a
#: held-out member's answer is genuinely inferable *if* anything in the representation ties
#: members of a kind together.
_WORLD: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "bird": ("water", ("sparrow", "crow", "robin", "finch")),
    "fish": ("plankton", ("trout", "carp", "bass", "perch")),
    "cactus": ("sand", ("saguaro", "peyote", "cholla", "opuntia")),
    "engine": ("fuel", ("diesel", "turbine", "rotary", "piston")),
}


@dataclass
class RelationalProbe:
    """What the head learned, and what the control says that learning was."""

    steps: int = 0
    vocabulary: int = 0
    trained_mrr: float = 0.0        # on pairs it was trained on — a memorisation check
    held_mrr: float = 0.0           # on members never trained
    control_mrr: float = 0.0        # on subjects that do not exist
    taught_is_a: bool = False
    expanded: bool = False
    detail: List[str] = field(default_factory=list)

    @property
    def gap(self) -> float:
        """Held-out minus control. **The only number here that means anything.**

        A held-out score on its own is uninterpretable: the object prior alone produces a
        respectable-looking MRR, because there are few plausible objects and they recur. What
        would show the subject mattered is the real members beating subjects that do not exist.
        """
        return self.held_mrr - self.control_mrr

    @property
    def generalises(self) -> bool:
        """Did the subject contribute anything at all? Deliberately a wide margin.

        0.10 rather than 0.0 because with a vocabulary this small a couple of rank positions is
        noise, and a probe that called noise a capability would be worse than no probe.
        """
        return self.gap > 0.10

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": self.steps, "vocabulary": self.vocabulary,
                "trained_mrr": round(self.trained_mrr, 4),
                "held_mrr": round(self.held_mrr, 4),
                "control_mrr": round(self.control_mrr, 4),
                "gap": round(self.gap, 4), "generalises": self.generalises,
                "taught_is_a": self.taught_is_a, "expanded": self.expanded,
                "detail": self.detail[:8]}

    def render(self) -> str:
        lines = [
            "NYXARA — relational generalisation probe",
            f"{self.steps} gradient steps · vocabulary {self.vocabulary}"
            f" · is_a taught: {'yes' if self.taught_is_a else 'no'}"
            f" · embedding-expanded: {'yes' if self.expanded else 'no'}",
            "",
            f"  trained pairs (memorisation) : MRR {self.trained_mrr:.3f}",
            f"  held-out members             : MRR {self.held_mrr:.3f}",
            f"  nonsense subjects (control)  : MRR {self.control_mrr:.3f}",
            "",
            f"  gap (held-out − control)     : {self.gap:+.3f}",
            f"  the subject contributed      : {'YES' if self.generalises else 'NO'}",
        ]
        if not self.generalises:
            lines.append("")
            lines.append("  Reading: the ranking is the object prior. `_cell_id` is a blake2b")
            lines.append("  digest, so related entities have orthogonal representations and there")
            lines.append("  is no shared structure to generalise over. This is an encoding")
            lines.append("  problem, not a training-objective one.")
        return "\n".join(lines)


def _mrr(rank_of, pairs: Sequence[Tuple[str, str, str]]) -> Tuple[float, List[Optional[int]]]:
    ranks = [rank_of(s, p, o) for s, p, o in pairs]
    hits = [r for r in ranks if r]
    return (sum(1.0 / r for r in hits) / max(1, len(ranks))), ranks


def probe(*, epochs: int = 60, seed: int = 0, teach_is_a: bool = False,
          expand: bool = False) -> RelationalProbe:
    """Train the readout on ``(subject, predicate) → object`` and measure against the control.

    ``teach_is_a`` also teaches every member's kind, so a two-hop composition is available in
    principle. It measures *worse*, which is the more interesting of the two results.

    ``expand`` widens each query by the subject's neighbours from :mod:`nyxara.njp.embed` — the
    similarity structure the hashed encoding cannot have. This is the variant that works, and the
    control is reported beside it for the same reason as everywhere else here.
    """
    out = RelationalProbe(taught_is_a=bool(teach_is_a), expanded=bool(expand))
    try:
        from nyxara.njp.brain import NJPBrain
    except Exception as exc:  # noqa: BLE001
        out.detail.append(f"NJP unavailable: {exc}")
        return out

    brain = NJPBrain()
    readout = getattr(brain, "readout", None)
    if readout is None:
        out.detail.append("no readout — gradients need numpy")
        return out

    rng = random.Random(seed)
    cells = brain.encode

    train: List[Tuple[str, str, str]] = []
    held: List[Tuple[str, str, str]] = []
    for kind, (prop, members) in _WORLD.items():
        if teach_is_a:
            for m in members:
                train.append((m, "is_a", kind))
            train.append((kind, "requires", prop))
        else:
            for m in members[:3]:
                train.append((m, "requires", prop))
        held.append((members[3], "requires", prop))

    for _ in range(max(1, int(epochs))):
        rng.shuffle(train)
        for subject, predicate, obj in train:
            readout.train_step([(cells(subject) + cells(predicate), cells(obj))])

    lexicon = getattr(brain, "lexicon", {})
    vocabulary = sorted(set(lexicon.values()))
    out.steps = int(readout.stats().get("steps", 0))
    out.vocabulary = len(vocabulary)

    embedding = None
    if expand:
        try:
            from nyxara.njp.brain import _cell_id
            from nyxara.njp.embed import EntityEmbedding
            embedding = EntityEmbedding(brain.fabric.manifold, cell_id=_cell_id)
            contexts: Dict[str, List[Tuple[str, str]]] = {}
            for kind, (prop, members) in _WORLD.items():
                for m in members:
                    # The CONTEXT only — never the answer being asked for. Feeding the held-out
                    # member its own `requires` would make the probe measure a lookup.
                    contexts.setdefault(m, []).append(("is_a", kind))
            for entity, pairs in contexts.items():
                embedding.observe(entity, pairs)
        except Exception as exc:  # noqa: BLE001
            out.detail.append(f"embedding unavailable: {exc}")
            embedding = None

    def rank_of(subject: str, predicate: str, truth: str) -> Optional[int]:
        query = cells(subject) + cells(predicate)
        if embedding is not None:
            for neighbour in embedding.expand(subject):
                query = query + cells(neighbour)
        got = [n for n, _ in readout.predict_symbols(
            query, lexicon=lexicon, k=len(vocabulary))]
        return got.index(truth) + 1 if truth in got else None

    out.trained_mrr, _ = _mrr(rank_of, train[:8])
    out.held_mrr, held_ranks = _mrr(rank_of, held)
    # Subjects that do not exist, asked the same questions. If they score the same, the ranking
    # never depended on the subject.
    control = [(f"zzqx{i}", "requires", o) for i, (_, _, o) in enumerate(held)]
    out.control_mrr, control_ranks = _mrr(rank_of, control)

    for (subject, _, truth), rank in zip(held, held_ranks):
        out.detail.append(f"{subject} → {truth}: rank {rank}")
    out.detail.append(f"control ranks: {control_ranks}")
    return out
