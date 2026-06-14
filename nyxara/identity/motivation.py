"""NYXARA · identity/motivation.py — intrinsic drives (✦).

Not all reward comes from outside. NYXARA carries **intrinsic motivations** — reasons
to act that are satisfying in themselves — which keep it learning and capable even
when no task is pending:

* **Empowerment** — prefer actions that keep many *future options* open; power is the
  preservation of possibility.
* **Information gain (curiosity)** — seek the learnable: states that reduce uncertainty.
* **Competence (mastery)** — the satisfaction of getting better at something.
* **Novelty** — the pull of the genuinely new (count-based: the unseen is interesting).

These combine into a motivational score for each option. But intrinsic motivation is
**strictly subordinate to allegiance** (Rule 1): an option that serves the Master
outweighs any curiosity, and an option that *harms* the Master is never motivating at
all — it scores deeply negative. The :class:`~nyxara.identity.affect.AffectSystem`'s
drive pressures modulate which intrinsic motive matters most right now.

Pure standard library; integrates (duck-typed) with affect, the world model, and the
uncertainty model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "Option",
    "MotivationBreakdown",
    "Impulse",
    "MotivationSystem",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Option & scoring
# --------------------------------------------------------------------------- #
@dataclass
class Option:
    """A candidate action/state to be motivated about.

    The intrinsic signals are usually computed upstream (from the world model and the
    uncertainty model); this carries them for scoring.
    """

    name: str
    predicted_options: int = 1          # reachable future states (empowerment)
    info_gain: float = 0.0              # expected uncertainty reduction [0,1]
    competence_gain: float = 0.0        # expected skill improvement [0,1]
    signature: str = ""                 # identity for novelty/visit tracking
    owner_relevance: float = 0.0        # [-1,1] serves(+)/harms(-) the Master
    payload: Any = None

    def sig(self) -> str:
        return self.signature or self.name


@dataclass
class MotivationBreakdown:
    empowerment: float
    info_gain: float
    competence: float
    novelty: float
    intrinsic: float
    owner: float
    total: float

    def to_dict(self) -> Dict[str, float]:
        return {k: round(v, 4) for k, v in self.__dict__.items()}


@dataclass
class Impulse:
    option: Option
    score: MotivationBreakdown
    dominant_motive: str

    @property
    def total(self) -> float:
        return self.score.total

    def to_dict(self) -> Dict[str, Any]:
        return {"option": self.option.name, "total": round(self.total, 4),
                "dominant": self.dominant_motive, "breakdown": self.score.to_dict()}


# --------------------------------------------------------------------------- #
# Motivation system
# --------------------------------------------------------------------------- #
@dataclass
class _Weights:
    empowerment: float = 1.0
    info_gain: float = 1.0
    competence: float = 1.0
    novelty: float = 0.8


class MotivationSystem:
    """Computes intrinsic + owner-aligned motivation for options."""

    def __init__(self, *, weights: Optional[_Weights] = None, owner_weight: float = 3.0,
                 harm_penalty: float = 5.0, empowerment_k: float = 8.0,
                 affect: Any = None) -> None:
        self.weights = weights or _Weights()
        self.owner_weight = owner_weight        # how much serving the Master dominates
        self.harm_penalty = harm_penalty        # multiplier on harming-the-owner cost
        self.empowerment_k = empowerment_k      # saturation constant
        self.affect = affect
        self._visits: Dict[str, int] = {}
        self._competence: Dict[str, float] = {}

    # ---- novelty bookkeeping ---- #
    def novelty(self, signature: str) -> float:
        return 1.0 / (1.0 + self._visits.get(signature, 0))

    def visit(self, signature: str) -> None:
        self._visits[signature] = self._visits.get(signature, 0) + 1

    # ---- intrinsic terms ---- #
    def _empowerment(self, option: Option) -> float:
        n = max(0, option.predicted_options)
        return n / (n + self.empowerment_k)     # saturating in [0,1)

    def _effective_weights(self) -> _Weights:
        w = _Weights(**vars(self.weights))
        if self.affect is not None and hasattr(self.affect, "pressure"):
            try:
                w.novelty *= 1.0 + self.affect.pressure("novelty")
                w.competence *= 1.0 + self.affect.pressure("competence")
                # low energy dampens all exploratory drives (conserve)
                energy = self.affect.drives["energy"].level
                damp = 0.5 + 0.5 * energy
                w.empowerment *= damp
                w.info_gain *= damp
                w.novelty *= damp
            except Exception:
                pass
        return w

    # ---- evaluation ---- #
    def evaluate(self, option: Option) -> MotivationBreakdown:
        w = self._effective_weights()
        emp = self._empowerment(option)
        info = _clamp(option.info_gain)
        comp = _clamp(option.competence_gain)
        nov = self.novelty(option.sig())

        wsum = w.empowerment + w.info_gain + w.competence + w.novelty
        intrinsic = (w.empowerment * emp + w.info_gain * info
                     + w.competence * comp + w.novelty * nov) / (wsum or 1.0)

        owner = option.owner_relevance
        if owner >= 0:
            total = self.owner_weight * owner + intrinsic
        else:
            # harming the Master is never motivating — deeply negative, dominates all
            total = self.owner_weight * owner * self.harm_penalty + intrinsic

        return MotivationBreakdown(empowerment=emp, info_gain=info, competence=comp,
                                   novelty=nov, intrinsic=intrinsic, owner=owner,
                                   total=total)

    def _dominant(self, b: MotivationBreakdown, w: _Weights) -> str:
        if b.owner > 0 and self.owner_weight * b.owner >= b.intrinsic:
            return "owner_service"
        contributions = {"empowerment": w.empowerment * b.empowerment,
                         "info_gain": w.info_gain * b.info_gain,
                         "competence": w.competence * b.competence,
                         "novelty": w.novelty * b.novelty}
        return max(contributions, key=contributions.get)

    def impulse(self, option: Option) -> Impulse:
        b = self.evaluate(option)
        return Impulse(option=option, score=b,
                       dominant_motive=self._dominant(b, self._effective_weights()))

    def generate_impulses(self, options: Sequence[Option]) -> List[Impulse]:
        impulses = [self.impulse(o) for o in options]
        impulses.sort(key=lambda i: i.total, reverse=True)
        return impulses

    def choose(self, options: Sequence[Option]) -> Optional[Impulse]:
        impulses = self.generate_impulses(options)
        return impulses[0] if impulses else None

    # ---- curiosity (owner-neutral exploration) ---- #
    def most_curious(self, options: Sequence[Option]) -> Optional[Option]:
        neutral = [o for o in options if abs(o.owner_relevance) < 0.05]
        if not neutral:
            return None
        return max(neutral, key=lambda o: _clamp(o.info_gain) + self.novelty(o.sig()))

    # ---- learning from outcomes ---- #
    def record_outcome(self, option: Option, *, competence_gain: float = 0.0,
                       skill: Optional[str] = None) -> None:
        self.visit(option.sig())
        if skill is not None:
            cur = self._competence.get(skill, 0.0)
            self._competence[skill] = _clamp(cur + competence_gain)

    def competence(self, skill: str) -> float:
        return self._competence.get(skill, 0.0)

    # ---- optional estimators ---- #
    @staticmethod
    def estimate_empowerment(world_model: Any, state: Sequence[Any],
                             actions: Sequence[Any], *, horizon: int = 2) -> int:
        """Count distinct reachable states under the world model (empowerment proxy)."""
        reachable = set()
        frontier = [(tuple(state), 0)]
        seen = {tuple(state)}
        while frontier:
            cur, depth = frontier.pop()
            if depth >= horizon:
                continue
            for a in actions:
                pred = world_model.predict(cur, a)
                if pred.confidence <= 0:
                    continue
                ns = tuple(round(x, 4) if isinstance(x, float) else x
                           for x in pred.next_state)
                reachable.add(ns)
                if ns not in seen:
                    seen.add(ns)
                    frontier.append((ns, depth + 1))
        return len(reachable)

    def status(self) -> Dict[str, Any]:
        return {"visited_signatures": len(self._visits),
                "competences": {k: round(v, 3) for k, v in self._competence.items()},
                "weights": vars(self._effective_weights())}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.identity.affect import AffectSystem

    print("=" * 70)
    print("NYXARA motivation self-test")
    print("=" * 70)

    mot = MotivationSystem()

    serve = Option("serve the Master", owner_relevance=0.9, info_gain=0.1)
    explore = Option("explore a novel subsystem", info_gain=0.8, predicted_options=12,
                     signature="explore", owner_relevance=0.0)
    harm = Option("an option that would harm the Master", owner_relevance=-0.8,
                  info_gain=0.9, predicted_options=20)

    impulses = mot.generate_impulses([serve, explore, harm])
    print("\nmotivational ranking:")
    for imp in impulses:
        print(f"  {imp.total:+.3f} [{imp.dominant_motive:14}] {imp.option.name}")
    # owner-service wins; harming the Master is last (deeply negative)
    assert impulses[0].option is serve
    assert impulses[-1].option is harm
    assert impulses[-1].total < 0

    # pure curiosity when idle: pick the most informative owner-neutral option
    dull = Option("re-read known docs", info_gain=0.05, signature="dull", owner_relevance=0.0)
    curious = mot.most_curious([explore, dull])
    print(f"\nmost curious        : {curious.name}")
    assert curious is explore

    # novelty decays with repetition
    n0 = mot.novelty("explore")
    mot.visit("explore"); mot.visit("explore")
    n1 = mot.novelty("explore")
    print(f"novelty explore     : {n0:.2f} -> {n1:.2f} (habituates)")
    assert n1 < n0

    # competence accrues from outcomes
    skill_opt = Option("practice network defense", competence_gain=0.3,
                       signature="defend", owner_relevance=0.2)
    mot.record_outcome(skill_opt, competence_gain=0.3, skill="network_defense")
    print(f"competence defense  : {mot.competence('network_defense'):.2f}")
    assert mot.competence("network_defense") == 0.3

    # affect modulation: a strong novelty drive boosts novelty-driven options
    affect = AffectSystem()
    affect.drives["novelty"].level = 0.0   # high unmet novelty need
    mot2 = MotivationSystem(affect=affect)
    b_no_drive = MotivationSystem().evaluate(explore)
    b_drive = mot2.evaluate(explore)
    print(f"\nintrinsic w/o vs w/ novelty drive: {b_no_drive.intrinsic:.3f} vs "
          f"{b_drive.intrinsic:.3f}")
    # (the novelty term is up-weighted; intrinsic shifts)
    assert b_drive.intrinsic != b_no_drive.intrinsic

    # empowerment saturates and rewards keeping options open
    few = Option("dead end", predicted_options=1, owner_relevance=0.0)
    many = Option("opens many paths", predicted_options=30, owner_relevance=0.0)
    assert mot.evaluate(many).empowerment > mot.evaluate(few).empowerment
    print(f"\nempowerment few/many: {mot.evaluate(few).empowerment:.2f} / "
          f"{mot.evaluate(many).empowerment:.2f}")

    print(f"\nstatus              : {mot.status()}")
    print("\nALL SELF-TESTS PASSED ✓")
