"""NYXARA · njp/propose.py — the next rung, proposed from measured weakness (🪜, NJP V.02).

:mod:`nyxara.njp.curriculum` is a **fixed ladder of nine rungs**, walked in order. It measures
honestly and it reports honestly, and it cannot do the one thing Phase 5 asks for: propose a stage
that is not on the list.

The cost of that is not theoretical. Measured on a brain that had absorbed both corpora and taken
a handful of turns::

    mastered   B C D E H          five rungs met
    current    A                  blocked: "only 3 of 20 required predictive.scored"
    depth      0
    next_stage A                  …and A, and A, forever

``depth`` is consecutive-from-A on purpose and the reasoning behind that is sound — mastering G on
a floor that is not there is not depth seven. But the consequence is that one rung blocked on a
**sample count** pins the whole ladder while five later rungs are already met, and ``next_stage``
hands back that same rung on every assessment. :meth:`nyxara.njp.doing.CognitiveAgency.deficit_goal`
had already run into this and said so plainly: *"every curriculum rung here is gated on a sample
count, and every affordance rearranges what she has rather than making experience — so without
this she is honestly, permanently, stuck at zero."*

**What a generator must not do.** The plan's rule for this phase is one line and it is the whole
design constraint: *generation and evaluation must stay separate — a system that writes its own
tests will optimise them.* Two things follow, and both are enforced rather than intended:

* **The generator never sees the held-out split.** It reads ``brain.stats()`` — numbers every
  organ already computes for its own reasons — and its own high-water marks, and nothing else.
  ``tests/njp/test_stage_generator.py`` inspects this module's source and fails if
  :mod:`nyxara.eval.capability` appears anywhere in it.
* **The generator never chooses a threshold.** This is the subtler half. A proposal whose bar is
  set relative to the current reading is a test written to be passed, so every source below takes
  its threshold from somewhere the generator had no hand in:

  ===============  ==========================================================================
  source           where the bar comes from
  ===============  ==========================================================================
  ``regression``   her own recorded best for that metric. She set it by achieving it, and a
                   number she has already hit is not a bar she can argue with.
  ``starved``      the ladder's own ``min_samples``, unchanged. The weakness is that a rung
                   cannot be *judged*, so the proposal is about producing the evidence, and
                   the amount required is the one the curriculum already declared.
  ``weakness``     :class:`~nyxara.njp.selfmodel.SelfModel`'s own reliability floor, for a
                   capability its own posterior calls weak.
  ===============  ==========================================================================

**And the evaluator is untouched.** A :class:`Proposal` carries a real
:class:`~nyxara.njp.curriculum.Stage`, so :meth:`~nyxara.njp.curriculum.Curriculum.assess` scores
it with exactly the code that scored the nine — which is what keeps generation and evaluation
apart in practice rather than only in the docstring.

Pure standard library. Fail-soft: an unreadable organ proposes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from nyxara.njp.curriculum import STAGES, Stage

__all__ = ["Source", "Proposal", "StageGenerator"]

#: How many proposals one pass returns. Bounded like everything else here.
_LIMIT = 4

#: A metric must fall this far below its own best before the drop is a regression rather than
#: the ordinary jitter of a rate computed over a growing denominator.
_REGRESSION = 0.05

#: Readings kept per metric before a best is trusted. One reading is not a record.
_MIN_READINGS = 3

#: The line below which :class:`~nyxara.njp.selfmodel.Capability` calls itself weak. Imported in
#: spirit rather than copied by choice — the organ's own rule, restated here as the bar a
#: ``weakness`` proposal has to clear, so the generator is not the thing that decided it.
_WEAK_FLOOR = 0.5


class Source:
    """Where a proposal came from. Each takes its threshold from somewhere else."""

    REGRESSION = "regression"
    STARVED = "starved"
    WEAKNESS = "weakness"


@dataclass(frozen=True)
class Proposal:
    """A rung she is not on, proposed because a number says she should be."""

    stage: Stage = field(default_factory=Stage)
    source: str = ""
    why: str = ""
    evidence: str = ""

    @property
    def named(self) -> bool:
        return bool(self.stage.organ and self.stage.metric)

    def to_dict(self) -> Dict[str, Any]:
        return {"stage": self.stage.to_dict(), "source": self.source,
                "why": self.why[:200], "evidence": self.evidence[:160]}


class StageGenerator:
    """Proposes the next rung from what her own organs already measure about her."""

    def __init__(self, brain: Any = None, *, stages: Tuple[Stage, ...] = STAGES,
                 limit: int = _LIMIT) -> None:
        self.brain = brain
        self.stages = tuple(stages)
        self.limit = max(1, int(limit))

        self.passes = 0
        self.proposed = 0
        self.by_source: Dict[str, int] = {}
        #: metric → (best seen, how many readings). The bar a regression is measured against, and
        #: the reason it is not one the generator gets to pick: she set it by achieving it.
        self.best: Dict[str, Tuple[float, int]] = {}
        self.last: List[Proposal] = []

    # ---- watching her own numbers -------------------------------------------- #
    def observe(self, brain: Any = None) -> int:
        """Record a reading for every ladder metric. Returns how many moved to a new best.

        Called before proposing, and separately from it, because a high-water mark that is only
        updated when someone asks for a proposal is a mark that misses every peak between asks.
        """
        moved = 0
        try:
            stats = self._stats(brain if brain is not None else self.brain)
            for stage in self.stages:
                key = f"{stage.organ}.{stage.metric}"
                value = self._dig(stats, stage.organ, stage.metric)
                if value is None:
                    continue
                best, seen = self.best.get(key, (value, 0))
                if value > best:
                    best, moved = value, moved + 1
                self.best[key] = (best, seen + 1)
        except Exception:  # noqa: BLE001
            return moved
        return moved

    @staticmethod
    def _stats(brain: Any) -> Dict[str, Any]:
        try:
            return dict(brain.stats() or {}) if hasattr(brain, "stats") else {}
        except Exception:  # noqa: BLE001
            return {}

    @staticmethod
    def _dig(stats: Dict[str, Any], organ: str, key: str) -> Optional[float]:
        block = stats.get(organ)
        if not isinstance(block, dict):
            return None
        value = block.get(key)
        if value is None or isinstance(value, bool):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # ---- proposing ------------------------------------------------------------ #
    def propose(self, brain: Any = None, report: Any = None) -> List[Proposal]:
        """Rungs worth working on, from measured weakness. Strongest evidence first."""
        out: List[Proposal] = []
        try:
            self.passes += 1
            brain = brain if brain is not None else self.brain
            stats = self._stats(brain)
            out.extend(self._regressed(stats))
            out.extend(self._starved(stats, report))
            out.extend(self._weakness(brain))
            out = out[: self.limit]
            for proposal in out:
                self.proposed += 1
                self.by_source[proposal.source] = self.by_source.get(proposal.source, 0) + 1
            self.last = list(out)
            return out
        except Exception:  # noqa: BLE001 — a failed pass proposes nothing
            return out

    def _regressed(self, stats: Dict[str, Any]) -> List[Proposal]:
        """A metric now below a level she has already reached.

        The strongest kind of weakness there is, and the only one whose bar needs no justifying:
        she is being asked for a number she has personally produced. Nothing here can be gamed by
        lowering expectations, because the expectation is a record rather than a target.
        """
        out: List[Proposal] = []
        for stage in self.stages:
            key = f"{stage.organ}.{stage.metric}"
            best, seen = self.best.get(key, (None, 0))
            value = self._dig(stats, stage.organ, stage.metric)
            if best is None or value is None or seen < _MIN_READINGS:
                continue
            if value >= best - _REGRESSION:
                continue
            out.append(Proposal(
                # Her own best becomes the bar, and the evidence requirement is the ladder's.
                stage=Stage(letter=stage.letter, name=f"{stage.name} (recover)",
                            teaches=f"get back to {key} {best:g}",
                            organ=stage.organ, metric=stage.metric, threshold=best,
                            sample_organ=stage.sample_organ,
                            sample_metric=stage.sample_metric,
                            min_samples=stage.min_samples),
                source=Source.REGRESSION,
                why=f"{key} has fallen from {best:g} to {value:g}",
                evidence=f"best {best:g} over {seen} readings, now {value:g}"))
        return out

    def _starved(self, stats: Dict[str, Any], report: Any) -> List[Proposal]:
        """A rung that cannot be *judged*, because the evidence for it does not exist yet.

        This is the state the ladder actually spends its life in, and the one the fixed list has
        no way to express. Stage A wants twenty scored predictions and has three; the metric it
        would be scored on is not failing, it is *unmeasured*, and "practise prediction" is not
        the work — producing predictions that can be scored at all is.

        So the proposed rung is about the sample counter rather than the metric, and its bar is
        the number the curriculum itself already declared. The generator sets nothing.
        """
        out: List[Proposal] = []
        for stage in self.stages:
            if not stage.sample_organ or not stage.sample_metric:
                continue
            samples = self._dig(stats, stage.sample_organ, stage.sample_metric)
            value = self._dig(stats, stage.organ, stage.metric)
            if samples is None or samples >= stage.min_samples:
                continue
            # A rung that is both nearly-evidenced and clearly failing is a *failing* rung, and
            # proposing "make more of it" would answer the wrong question. But "nearly" is
            # load-bearing and the first version of this omitted it — the check read *"any
            # samples at all, and the metric under half the bar"*, which is self-contradictory:
            # a starved rung has too little evidence to be called failing by definition, and the
            # test threw out exactly the rung the whole mechanism exists for. Measured, stage A
            # at ``accuracy 0.0`` on **3 of 20** required samples was silently skipped.
            if (value is not None and samples >= stage.min_samples * 0.5
                    and value < stage.threshold * 0.5):
                continue
            out.append(Proposal(
                stage=Stage(letter=stage.letter, name=f"{stage.name} (evidence)",
                            teaches=f"produce {stage.sample_organ}.{stage.sample_metric}",
                            organ=stage.sample_organ, metric=stage.sample_metric,
                            threshold=stage.min_samples,
                            sample_organ=stage.sample_organ,
                            sample_metric=stage.sample_metric, min_samples=0.0),
                source=Source.STARVED,
                why=(f"stage {stage.letter} cannot be judged: "
                     f"{stage.sample_organ}.{stage.sample_metric} is {samples:g}, "
                     f"wants {stage.min_samples:g}"),
                evidence=f"metric reads {'—' if value is None else format(value, 'g')}"))
        return out

    def _weakness(self, brain: Any) -> List[Proposal]:
        """A capability the self-model's own posterior calls weak.

        The bar is the self-model's reliability floor, which is a property of how that organ
        decides anything is reliable at all — not a number chosen here to be reachable.
        """
        out: List[Proposal] = []
        try:
            model = getattr(brain, "self_model", None)
            weakest = model.weakest() if model is not None else None
            if weakest is None or not getattr(weakest, "weak", False):
                return out
            name = str(getattr(weakest, "name", "") or "")
            level = float(getattr(weakest, "level", 0.0) or 0.0)
            if not name:
                return out
            # `Capability.weak` is `level < 0.5`; clearing it is the bar, and the organ owns that
            # number rather than this one. Scored on `self_model.weakest_level`, which is a flat
            # reading — a stage whose metric cannot be dug out of a stats block is a stage that
            # can never be assessed, and proposing one is worse than proposing nothing.
            floor = _WEAK_FLOOR
            out.append(Proposal(
                stage=Stage(letter="?", name=f"{name} (weak)",
                            teaches=f"raise measured {name} past the weak line",
                            organ="self_model", metric="weakest_level", threshold=floor,
                            sample_organ="self_model", sample_metric="measured",
                            min_samples=1.0),
                source=Source.WEAKNESS,
                why=f"{name} is measured at {level:.2f}, below the weak line {floor:.2f}",
                evidence=f"self-model posterior for {name}"))
        except Exception:  # noqa: BLE001
            return out
        return out

    # ---- reporting -------------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        return {
            "passes": self.passes,
            "proposed": self.proposed,
            "by_source": dict(self.by_source),
            "tracked": len(self.best),
            # The falsifier, and it is deliberately about *reaching past the ladder*: if every
            # proposal is the rung `next_stage` already returns, this organ is an alias.
            "last": [p.to_dict() for p in self.last[:4]],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"best": {k: list(v) for k, v in self.best.items()},
                "counters": {"passes": self.passes, "proposed": self.proposed,
                             "by_source": dict(self.by_source)}}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            for key, pair in ((d or {}).get("best") or {}).items():
                self.best[str(key)] = (float(pair[0]), int(pair[1]))
            counters = (d or {}).get("counters") or {}
            self.passes = int(counters.get("passes", 0))
            self.proposed = int(counters.get("proposed", 0))
            self.by_source = dict(counters.get("by_source") or {})
        except Exception:  # noqa: BLE001
            pass
