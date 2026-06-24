"""NYXARA · growth/credit.py — the improvement ledger (🧾, lifelong credit assignment).

The Recursive Self-Improvement Engine acts each cycle (edit this weakness, take that growth
action), but until now it forgot every outcome: each cycle re-decided from scratch, ordering
self-edits by static severity and dispatching the index's directive blind to whether that kind of
intervention has *ever actually raised her intelligence*. This module is the memory that closes
that gap — a lifelong, persisted record of **which interventions pay off**, so effort flows to
what has worked before.

Design (deliberately mirrors :mod:`nyxara.growth.intelligence`):

* **Beta-Bernoulli per arm.** Every intervention is an "arm": a weakness-class to edit
  (``edit:{source}:{strategy}``) or a growth action to dispatch (``action:{name}``). Each arm
  carries a Beta(α, β) posterior over "this raised the index". Updates are *fractional* — a cycle
  that nudged the index up by δ credits the arm by a reward in ``[0, 1]`` derived from δ — so the
  signal is graded, not just win/lose.
* **Thompson sampling.** To rank candidates she samples each arm's posterior and orders by the
  draw: high-mean arms lead, but under-tried arms still get explored (the posterior is wide). This
  is real exploration/exploitation, not a fixed heuristic.
* **Rides the existing memory.** Persisted as ONE protected, high-importance SEMANTIC record (tag
  ``improvement-ledger``) in NYXARA's long-term store — no new format, no schema migration, same
  reconsolidate-in-place pattern as the intelligence index. Degrades to in-process state with no
  memory. Pure standard library; an optional learned forecaster (:mod:`growth.forecaster`) can
  sharpen the ranking when torch is present, but is never required.
* **Measure-only.** The ledger never widens authority or touches a gate — it only records
  outcomes and *reorders* work that every existing gauntlet still independently approves.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["LedgerState", "ImprovementLedger", "EditStrategyBandit", "arm_key"]

_LEDGER_TAG = "improvement-ledger"


# --------------------------------------------------------------------------- #
# arm keys — a stable name for each kind of intervention
# --------------------------------------------------------------------------- #
def arm_key(*, source: str = "", strategy: str = "", action: str = "") -> str:
    """A stable ledger key for one intervention.

    A source edit is keyed by its weakness class + edit strategy
    (``edit:{source}:{strategy}``); a growth directive by its action (``action:{name}``).
    """
    if action:
        return f"action:{action}"
    return f"edit:{source or 'unknown'}:{strategy or 'transform'}"


# --------------------------------------------------------------------------- #
# The persisted state
# --------------------------------------------------------------------------- #
@dataclass
class LedgerState:
    """All arms' Beta posteriors + a short reward history — the thing that persists."""

    arms: Dict[str, Dict[str, float]] = field(default_factory=dict)
    t: int = 0
    history: list = field(default_factory=list)   # recent rewards — for trend/observability
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"arms": {k: {"alpha": round(float(v.get("alpha", 1.0)), 6),
                             "beta": round(float(v.get("beta", 1.0)), 6),
                             "n": int(v.get("n", 0))}
                         for k, v in self.arms.items()},
                "t": int(self.t),
                "history": [round(float(r), 6) for r in self.history], "at": self.at}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LedgerState":
        if not isinstance(d, dict):
            return cls()
        arms: Dict[str, Dict[str, float]] = {}
        for k, v in (d.get("arms", {}) or {}).items():
            if not isinstance(v, dict):
                continue
            arms[k] = {"alpha": float(v.get("alpha", 1.0) or 1.0),
                       "beta": float(v.get("beta", 1.0) or 1.0),
                       "n": int(v.get("n", 0) or 0)}
        return cls(arms=arms, t=int(d.get("t", 0) or 0),
                   history=list(d.get("history", []) or []),
                   at=float(d.get("at", time.time()) or time.time()))

    def summary(self) -> str:
        top = sorted(self.arms.items(),
                     key=lambda kv: kv[1]["alpha"] / (kv[1]["alpha"] + kv[1]["beta"]),
                     reverse=True)[:3]
        shown = ", ".join(f"{k}={v['alpha'] / (v['alpha'] + v['beta']):.2f}" for k, v in top)
        return f"ledger t={self.t}, {len(self.arms)} arm(s); best: {shown or '(none)'}"


# --------------------------------------------------------------------------- #
# The ledger — Beta-Bernoulli outcome memory, persisted in long-term memory
# --------------------------------------------------------------------------- #
class ImprovementLedger:
    """Records which interventions raise the intelligence index, and persists across restarts.

    Parameters
    ----------
    memory:   a :class:`~nyxara.memory.store.MemoryStore` for persistence (optional — falls back
              to in-process state).
    settings: :class:`~nyxara.kernel.config.NyxaraSettings` (pulled from ``get_settings`` if not
              supplied); reads ``self_improvement`` ledger bounds.
    rng:      an optional :class:`random.Random` for deterministic Thompson sampling in tests.
    """

    def __init__(self, *, memory: Any = None, settings: Any = None,
                 rng: Optional[random.Random] = None) -> None:
        from nyxara.kernel.config import get_settings
        self.settings = settings or get_settings()
        self.memory = memory
        self._rng = rng or random.Random()
        self._cached: Optional[LedgerState] = None

    # ---- the prior strength / reward shaping knobs (bounded, from config) ---- #
    def _prior(self) -> float:
        cfg = self.settings.self_improvement
        return max(0.1, float(getattr(cfg, "ledger_prior_strength", 1.0)))

    def _reward_scale(self) -> float:
        cfg = self.settings.self_improvement
        return max(1.0, float(getattr(cfg, "ledger_reward_scale", 40.0)))

    def _arm(self, state: LedgerState, key: str) -> Dict[str, float]:
        a = self._prior()
        return state.arms.setdefault(key, {"alpha": a, "beta": a, "n": 0})

    # ---- reward shaping: a raw index delta -> a graded reward in [0, 1] ---- #
    def reward_from_delta(self, delta: float) -> float:
        """Map a (possibly tiny, possibly negative) index change to a reward in ``[0, 1]``.

        A logistic on ``delta × scale``: a clear rise → ~1, a clear fall → ~0, no change → 0.5.
        The scale makes the small per-cycle deltas the index actually produces meaningful."""
        import math
        try:
            x = float(delta) * self._reward_scale()
            return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))
        except Exception:  # noqa: BLE001
            return 0.5

    # ---- record an outcome (fractional Beta update) ---- #
    def record(self, state: LedgerState, key: str, reward: float) -> LedgerState:
        """Credit ``key`` with a reward in ``[0, 1]`` (fractional Beta-Bernoulli update)."""
        r = max(0.0, min(1.0, float(reward)))
        arm = self._arm(state, key)
        arm["alpha"] = float(arm["alpha"]) + r
        arm["beta"] = float(arm["beta"]) + (1.0 - r)
        arm["n"] = int(arm["n"]) + 1
        state.t += 1
        state.history = (list(state.history)[-31:] + [r])
        return state

    def record_delta(self, state: LedgerState, keys: List[str], delta: float) -> LedgerState:
        """Credit every arm touched this cycle with the reward implied by the index ``delta``."""
        reward = self.reward_from_delta(delta)
        for key in keys:
            self.record(state, key, reward)
        return state

    # ---- read the posteriors ---- #
    def mean(self, state: LedgerState, key: str) -> float:
        arm = state.arms.get(key)
        if not arm:
            return 0.5                       # uniform prior mean for an untried arm
        a, b = float(arm["alpha"]), float(arm["beta"])
        return a / (a + b) if (a + b) > 0 else 0.5

    def sample(self, state: LedgerState, key: str) -> float:
        """Thompson draw from ``key``'s Beta posterior (wide for under-tried arms → exploration)."""
        arm = state.arms.get(key)
        a = float(arm["alpha"]) if arm else self._prior()
        b = float(arm["beta"]) if arm else self._prior()
        try:
            return self._rng.betavariate(max(1e-3, a), max(1e-3, b))
        except Exception:  # noqa: BLE001
            return a / (a + b) if (a + b) > 0 else 0.5

    # ---- persistence (rides the existing long-term memory) ---- #
    def load(self) -> LedgerState:
        if self._cached is not None:
            return self._cached
        rec = self._find_record()
        state = LedgerState.from_dict(rec.metadata.get("state", {})) if rec is not None \
            else LedgerState()
        self._cached = state
        return state

    def save(self, state: LedgerState) -> bool:
        self._cached = state
        if self.memory is None:
            return False
        try:
            from nyxara.memory.provenance import Provenance, SourceType
            from nyxara.memory.store import MemoryType
        except Exception:  # noqa: BLE001
            return False
        tag = str(getattr(self.settings.memory, "deep_synapse_tag", "deep-synapse"))
        content = f"[improvement-ledger] {state.summary()}"
        meta = {"state": state.to_dict()}
        try:
            existing = self._find_record()
            if existing is not None:
                self.memory.forget(existing.mem_id)
            self.memory.remember(
                content, mem_type=MemoryType.SEMANTIC,
                provenance=Provenance(SourceType.SELF_REFLECTION, confidence=0.9),
                importance=1.0, tags=[_LEDGER_TAG, tag], metadata=meta)
            return True
        except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
            return False

    def _find_record(self) -> Any:
        if self.memory is None:
            return None
        try:
            for rec in self.memory._kv.values():
                if _LEDGER_TAG in rec.tags and "state" in rec.metadata:
                    return rec
        except Exception:  # noqa: BLE001
            return None
        return None


# --------------------------------------------------------------------------- #
# The bandit — rank pending work by learned payoff (Thompson, optional forecaster)
# --------------------------------------------------------------------------- #
class EditStrategyBandit:
    """Reorders pending self-edits (and scores growth actions) by learned expected payoff.

    Wraps an :class:`ImprovementLedger`. When a learned :mod:`growth.forecaster` is supplied and
    available it scores by predicted index-gain; otherwise it Thompson-samples the ledger's Beta
    posteriors. With an empty ledger every arm draws from the same prior, so the result reduces to
    the caller's original (severity) order — the bandit only ever *improves* on the baseline.
    """

    def __init__(self, ledger: ImprovementLedger, *, forecaster: Any = None) -> None:
        self.ledger = ledger
        self.forecaster = forecaster

    @staticmethod
    def weakness_key(weakness: Any) -> str:
        src = str(getattr(weakness, "source", "") or "unknown")
        strat = str(getattr(weakness, "edit_strategy", "") or "transform")
        return arm_key(source=src, strategy=strat)

    def prioritize(self, weaknesses: List[Any], *, state: Optional[LedgerState] = None,
                   features: Optional[Dict[str, float]] = None) -> List[Any]:
        """Return ``weaknesses`` reordered by descending learned payoff (stable on ties)."""
        if not weaknesses:
            return list(weaknesses)
        st = state if state is not None else self.ledger.load()
        # blend the learned score with the weakness's own severity so a brand-new, never-tried
        # high-severity weakness is not buried under a marginally-favoured trivial one. The blend is
        # a real, evolvable *algorithm* knob (the meta tower tunes what-to-fix-first); 1.0 ⇒ pure
        # learned payoff, 0.0 ⇒ pure severity. Falls back to the historical 0.7 when unset.
        try:
            blend = float(getattr(self.ledger.settings.self_improvement,
                                  "bandit_severity_blend", 0.7))
        except Exception:  # noqa: BLE001 — a missing setting just uses the historical default
            blend = 0.7
        blend = max(0.0, min(1.0, blend))
        scored: List[Tuple[float, int, Any]] = []
        for i, w in enumerate(weaknesses):
            key = self.weakness_key(w)
            score = self._score(st, key, features)
            sev = float(getattr(w, "severity", 0.0) or 0.0)
            scored.append((blend * score + (1.0 - blend) * sev, -i, w))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [w for _, _, w in scored]

    def action_scores(self, actions: List[str], *,
                      state: Optional[LedgerState] = None) -> Dict[str, float]:
        """Thompson score for each growth action (for the directive planner)."""
        st = state if state is not None else self.ledger.load()
        return {a: self.ledger.sample(st, arm_key(action=a)) for a in actions}

    def _score(self, state: LedgerState, key: str, features: Optional[Dict[str, float]]) -> float:
        if self.forecaster is not None:
            try:
                if self.forecaster.available():
                    pred = self.forecaster.predict(key, features or {})
                    if pred is not None:
                        return max(0.0, min(1.0, float(pred)))
            except Exception:  # noqa: BLE001 — a flaky forecaster never sinks ranking
                pass
        # Thompson-sample only arms with actual evidence; an untried arm has a uniform posterior,
        # so "sampling" it is pure noise — return the neutral prior mean instead and let the
        # caller's severity blend decide. This makes the cold-start order deterministic (the
        # bandit only ever *improves* on the severity baseline once it has learned something).
        arm = state.arms.get(key)
        if arm and int(arm.get("n", 0)) > 0:
            return self.ledger.sample(state, key)
        return 0.5


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.memory.store import MemoryStore

    print("=" * 70)
    print("NYXARA improvement-ledger self-test")
    print("=" * 70)

    led = ImprovementLedger(memory=MemoryStore(), rng=random.Random(0))
    st = led.load()
    assert st.t == 0 and not st.arms

    # reward shaping: up-delta -> >0.5, down-delta -> <0.5, no change -> ~0.5
    assert led.reward_from_delta(0.05) > 0.6
    assert led.reward_from_delta(-0.05) < 0.4
    assert abs(led.reward_from_delta(0.0) - 0.5) < 1e-6

    # an arm that keeps paying off should overtake one that keeps failing
    good = arm_key(source="cyclomatic", strategy="transform")
    bad = arm_key(source="hygiene", strategy="transform")
    for _ in range(20):
        led.record(st, good, 0.9)
        led.record(st, bad, 0.1)
    print(f"\n{st.summary()}")
    assert led.mean(st, good) > 0.8 > led.mean(st, bad)
    assert st.t == 40

    # Thompson ranking puts the historically-good arm first most of the time
    class _W:
        def __init__(self, source, sev):
            self.source = source
            self.edit_strategy = "transform"
            self.severity = sev
    bandit = EditStrategyBandit(led)
    wins = 0
    for _ in range(200):
        order = bandit.prioritize([_W("hygiene", 0.5), _W("cyclomatic", 0.5)], state=st)
        if order[0].source == "cyclomatic":
            wins += 1
    print(f"learned-good ranked first: {wins}/200")
    assert wins > 140

    # persistence round-trips as ONE protected record
    led.save(st)
    led2 = ImprovementLedger(memory=led.memory)
    st2 = led2.load()
    assert st2.t == st.t and abs(led2.mean(st2, good) - led.mean(st, good)) < 1e-6
    rec = led2._find_record()
    assert rec is not None and rec.importance >= 0.9
    assert led.settings.memory.deep_synapse_tag in rec.tags

    # a fresh store starts empty
    assert ImprovementLedger(memory=MemoryStore()).load().t == 0

    print("\nALL SELF-TESTS PASSED ✓")
