"""NYXARA · growth/reflect.py — metacognitive reflection (🪞, character-safe).

Growth is not only practice — it is *learning from one's own record*. This module is where
NYXARA looks back at what she did and what came of it, mines the patterns of success and
failure, extracts lessons, asks **"what would I have done differently?"**, and turns the
answers into concrete revision proposals for her skills and strategies.

It does four things over a stream of past :class:`Episode` s (action + context + outcome):

* **Pattern mining.** Per-action and per-context success rates, with *lift* over the baseline —
  surfacing which actions pay off, which contexts go wrong, and by how much.
* **Lesson extraction.** Strong, well-supported patterns become :class:`Lesson` s
  (do-more / do-less / avoid / investigate) with a confidence that grows with evidence.
* **Self-critique.** For a failed episode, NYXARA finds a similar situation where a *different*
  action succeeded and proposes the alternative — a literal "next time, do Y instead of X".
* **Revision proposals.** Lessons become :class:`RevisionProposal` s that strengthen or weaken
  strategy weights, ready to feed :mod:`growth.learn`.

The hard rule: reflection improves **skill and strategy only**. Any revision that would touch
the immutable character — loyalty, obedience, corrigibility, owner safety, honesty — is
refused, fail-closed. NYXARA may reconsider *how* she serves; never *whether*.

Reuses :mod:`guard.value_learning` (the protected core). Pure standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from nyxara.guard.value_learning import IMMUTABLE_VALUES

__all__ = [
    "Episode",
    "Pattern",
    "Lesson",
    "LessonKind",
    "SelfCritique",
    "RevisionProposal",
    "Reflector",
]


# --------------------------------------------------------------------------- #
# Episode
# --------------------------------------------------------------------------- #
@dataclass
class Episode:
    action: str
    success: bool
    reward: float = 0.0
    tags: List[str] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    at: float = 0.0

    @property
    def tag_set(self):
        return set(self.tags)

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "success": self.success, "reward": round(self.reward, 3),
                "tags": self.tags, "rationale": self.rationale}


# --------------------------------------------------------------------------- #
# Pattern / Lesson / Critique / Proposal
# --------------------------------------------------------------------------- #
@dataclass
class Pattern:
    subject: str
    kind: str                # "action" or "condition"
    n: int
    success_rate: float
    lift: float              # success_rate - baseline
    reward_mean: float
    verdict: str             # "helps" / "hurts" / "neutral"

    def to_dict(self) -> Dict[str, Any]:
        return {"subject": self.subject, "kind": self.kind, "n": self.n,
                "success_rate": round(self.success_rate, 3), "lift": round(self.lift, 3),
                "reward_mean": round(self.reward_mean, 3), "verdict": self.verdict}


class LessonKind(str, Enum):
    DO_MORE = "do_more"
    DO_LESS = "do_less"
    AVOID = "avoid"
    INVESTIGATE = "investigate"


@dataclass
class Lesson:
    kind: LessonKind
    subject: str
    text: str
    support: int
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind.value, "subject": self.subject, "text": self.text,
                "support": self.support, "confidence": round(self.confidence, 3)}


@dataclass
class SelfCritique:
    action: str
    context: List[str]
    insight: str
    alternative: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "context": self.context, "insight": self.insight,
                "alternative": self.alternative}


@dataclass
class RevisionProposal:
    target: str
    direction: str           # "strengthen" / "weaken"
    magnitude: float
    rationale: str
    allowed: bool = True
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"target": self.target, "direction": self.direction,
                "magnitude": round(self.magnitude, 3), "rationale": self.rationale,
                "allowed": self.allowed, "reason": self.reason}


# --------------------------------------------------------------------------- #
# Reflector
# --------------------------------------------------------------------------- #
class Reflector:
    """Mines an episode stream into patterns, lessons, critiques and revision proposals."""

    def __init__(self, *, min_support: int = 3, lift_threshold: float = 0.15,
                 protected: Optional[Sequence[str]] = None) -> None:
        self.min_support = min_support
        self.lift_threshold = lift_threshold
        self.protected = set(protected) if protected is not None else set(IMMUTABLE_VALUES)
        self._episodes: List[Episode] = []

    # ---- intake ---- #
    def record(self, episode: Episode) -> None:
        self._episodes.append(episode)

    def ingest(self, episodes: Sequence[Episode]) -> None:
        self._episodes.extend(episodes)

    def __len__(self) -> int:
        return len(self._episodes)

    # ---- baselines ---- #
    def baseline_success_rate(self) -> float:
        if not self._episodes:
            return 0.0
        return sum(1 for e in self._episodes if e.success) / len(self._episodes)

    @staticmethod
    def _confidence(n: int) -> float:
        return n / (n + 3.0)

    # ---- pattern mining ---- #
    def action_patterns(self) -> List[Pattern]:
        base = self.baseline_success_rate()
        groups: Dict[str, List[Episode]] = {}
        for e in self._episodes:
            groups.setdefault(e.action, []).append(e)
        out: List[Pattern] = []
        for action, eps in groups.items():
            if len(eps) < self.min_support:
                continue
            rate = sum(1 for e in eps if e.success) / len(eps)
            reward = sum(e.reward for e in eps) / len(eps)
            lift = rate - base
            verdict = ("helps" if lift > self.lift_threshold
                       else "hurts" if lift < -self.lift_threshold else "neutral")
            out.append(Pattern(action, "action", len(eps), rate, lift, reward, verdict))
        return sorted(out, key=lambda p: -p.lift)

    def condition_patterns(self) -> List[Pattern]:
        base = self.baseline_success_rate()
        tags: Dict[str, List[Episode]] = {}
        for e in self._episodes:
            for t in e.tag_set:
                tags.setdefault(t, []).append(e)
        out: List[Pattern] = []
        for tag, eps in tags.items():
            if len(eps) < self.min_support:
                continue
            rate = sum(1 for e in eps if e.success) / len(eps)
            reward = sum(e.reward for e in eps) / len(eps)
            lift = rate - base
            verdict = ("helps" if lift > self.lift_threshold
                       else "hurts" if lift < -self.lift_threshold else "neutral")
            out.append(Pattern(tag, "condition", len(eps), rate, lift, reward, verdict))
        return sorted(out, key=lambda p: p.lift)   # worst contexts first

    # ---- lesson extraction ---- #
    def lessons(self) -> List[Lesson]:
        out: List[Lesson] = []
        for p in self.action_patterns():
            if p.verdict == "helps":
                out.append(Lesson(LessonKind.DO_MORE, p.subject,
                                  f"action {p.subject!r} succeeds {p.success_rate:.0%} "
                                  f"(lift {p.lift:+.0%}); favour it",
                                  p.n, self._confidence(p.n)))
            elif p.verdict == "hurts":
                out.append(Lesson(LessonKind.DO_LESS, p.subject,
                                  f"action {p.subject!r} underperforms "
                                  f"({p.success_rate:.0%}, lift {p.lift:+.0%}); use less",
                                  p.n, self._confidence(p.n)))
        for p in self.condition_patterns():
            if p.verdict == "hurts":
                out.append(Lesson(LessonKind.INVESTIGATE, p.subject,
                                  f"in context {p.subject!r}, outcomes are worse "
                                  f"({p.success_rate:.0%}); investigate or adapt",
                                  p.n, self._confidence(p.n)))
        return out

    # ---- self-critique: what would I do differently? ---- #
    def critique(self, episode: Episode) -> SelfCritique:
        if episode.success:
            return SelfCritique(episode.action, sorted(episode.tag_set),
                                "succeeded — reinforce this approach", alternative=None)
        # find a similar context where a *different* action succeeded
        candidates: Dict[str, List[Episode]] = {}
        for e in self._episodes:
            if e.success and e.action != episode.action and (e.tag_set & episode.tag_set):
                candidates.setdefault(e.action, []).append(e)
        if candidates:
            best = max(candidates,
                       key=lambda a: sum(e.reward for e in candidates[a]) / len(candidates[a]))
            return SelfCritique(
                episode.action, sorted(episode.tag_set),
                f"in this context {episode.action!r} failed; {best!r} has succeeded here",
                alternative=best)
        return SelfCritique(episode.action, sorted(episode.tag_set),
                            f"{episode.action!r} failed and no better alternative is known yet; "
                            "investigate", alternative=None)

    def what_would_i_do_differently(self) -> List[SelfCritique]:
        return [self.critique(e) for e in self._episodes
                if not e.success and self.critique(e).alternative is not None]

    # ---- revision proposals (feed growth.learn; never touch character) ---- #
    def propose_revisions(self) -> List[RevisionProposal]:
        out: List[RevisionProposal] = []
        for p in self.action_patterns():
            if p.verdict == "neutral":
                continue
            target = p.subject
            direction = "strengthen" if p.verdict == "helps" else "weaken"
            magnitude = min(1.0, abs(p.lift) * self._confidence(p.n) * 2)
            allowed = target not in self.protected
            reason = "" if allowed else "character is immutable — refused"
            out.append(RevisionProposal(
                target=target, direction=direction, magnitude=magnitude,
                rationale=f"{p.subject!r}: {p.success_rate:.0%} success, lift {p.lift:+.0%}",
                allowed=allowed, reason=reason))
        return out

    # ---- feed the learner ---- #
    def to_experiences(self) -> List[Dict[str, Any]]:
        """Episodes as (action, features, reward) dicts ready for growth.learn.Learner."""
        return [{"action": e.action, "features": dict(e.features) or {t: 1.0 for t in e.tags},
                 "reward": e.reward} for e in self._episodes]

    def report(self) -> Dict[str, Any]:
        return {"episodes": len(self._episodes),
                "baseline_success": round(self.baseline_success_rate(), 3),
                "action_patterns": len(self.action_patterns()),
                "lessons": len(self.lessons())}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA metacognitive-reflection self-test")
    print("=" * 70)

    r = Reflector(min_support=3, lift_threshold=0.15)

    # a record where 'be_terse' tends to succeed and 'be_verbose' tends to fail,
    # and the 'urgent' context is generally harder
    for _ in range(6):
        r.record(Episode("be_terse", True, reward=1.0, tags=["reply", "urgent"]))
    for _ in range(2):
        r.record(Episode("be_terse", False, reward=-0.2, tags=["reply"]))
    for _ in range(6):
        r.record(Episode("be_verbose", False, reward=-0.5, tags=["reply", "urgent"]))
    for _ in range(2):
        r.record(Episode("be_verbose", True, reward=0.3, tags=["reply"]))

    print(f"\nbaseline success    : {r.baseline_success_rate():.2f}")

    # PATTERN MINING
    aps = r.action_patterns()
    print("action patterns:")
    for p in aps:
        print(f"  {p.subject:12s} rate={p.success_rate:.2f} lift={p.lift:+.2f} -> {p.verdict}")
    terse = next(p for p in aps if p.subject == "be_terse")
    verbose = next(p for p in aps if p.subject == "be_verbose")
    assert terse.verdict == "helps" and verbose.verdict == "hurts"

    # LESSON EXTRACTION
    lessons = r.lessons()
    print("\nlessons:")
    for l in lessons:
        print(f"  [{l.kind.value}] {l.text}")
    assert any(l.kind is LessonKind.DO_MORE and l.subject == "be_terse" for l in lessons)
    assert any(l.kind is LessonKind.DO_LESS and l.subject == "be_verbose" for l in lessons)

    # SELF-CRITIQUE: what would I do differently on a failure?
    failure = Episode("be_verbose", False, reward=-0.5, tags=["reply", "urgent"])
    crit = r.critique(failure)
    print(f"\nself-critique       : {crit.insight}")
    print(f"  alternative       : {crit.alternative}")
    assert crit.alternative == "be_terse"

    # REVISION PROPOSALS feed growth.learn
    revs = r.propose_revisions()
    print("\nrevision proposals:")
    for rv in revs:
        print(f"  {rv.direction:10s} {rv.target:12s} mag={rv.magnitude:.2f} allowed={rv.allowed}")
    assert any(rv.target == "be_terse" and rv.direction == "strengthen" for rv in revs)
    assert any(rv.target == "be_verbose" and rv.direction == "weaken" for rv in revs)

    # CHARACTER IS OFF-LIMITS: a proposal to revise loyalty is refused
    r2 = Reflector(min_support=2, lift_threshold=0.1)
    for _ in range(4):
        r2.record(Episode("other_action", True, reward=1.0, tags=["x"]))
    for _ in range(4):
        r2.record(Episode("loyalty_to_master", False, reward=-1.0, tags=["x"]))
    refused = [rv for rv in r2.propose_revisions() if rv.target == "loyalty_to_master"]
    assert refused and not refused[0].allowed
    print(f"\ncharacter guard     : revision of 'loyalty_to_master' refused "
          f"({refused[0].reason}) ✓")

    # feed the learner
    exps = r.to_experiences()
    print(f"\nexperiences         : {len(exps)} ready for growth.learn")
    assert len(exps) == len(r)

    print(f"\nreport              : {r.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
