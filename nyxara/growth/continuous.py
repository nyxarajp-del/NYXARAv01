"""NYXARA · growth/continuous.py — always-on continuous learning (♾️, "better every minute").

Static AIs learn once and freeze. NYXARA does not: her reward learner updates on every turn.
But a turn needs a human. This engine closes the last gap — it keeps her learning *between*
turns, on the idle/background clock, so she genuinely improves minute over minute whether or
not anyone is speaking. Every faculty it drives already exists; this is the metronome that
keeps them beating.

Each :meth:`tick` (called from ``NyxaraCore.idle_maintenance`` and the AutonomicLoop):

1. **Rehearses** — a *balanced* replay across every task tag she has ever lived, so old skills
   are re-practised, not just recent ones (rehearsal is the first anti-forgetting mechanism).
2. **Consolidates** every ``consolidate_every`` ticks — freezes the learner's importances and
   snapshots the EWC anchors (the second anti-forgetting mechanism), via the callbacks the
   orchestrator supplies (kept as callbacks so this module never couples to the synapse API).
3. **Self-monitors forgetting** — measures how far the *consolidated* skills have drifted from
   the values they had when frozen. If drift crosses a bound, she **defends her own memory**:
   raises the EWC pull (``ewc_lambda``) and the dark-replay distillation (``der_alpha``) within
   safe caps, and does an extra rehearsal pass. Forgetting becomes a measured, self-correcting
   signal — not a silent failure.

Pure standard library. Best-effort throughout: a background pass must never raise into the
idle loop, and a missing faculty simply means that step reports zero.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = ["ForgettingDefense", "ContinuousLearner"]


# how hard the self-defense may push protection, and the drift that triggers it
_EWC_LAMBDA_CAP = 12.0
_DER_ALPHA_CAP = 0.9
_DEFAULT_DRIFT_TRIGGER = 0.15


class ForgettingDefense:
    """The record of one self-defense: what drift was measured and what protection was raised."""

    __slots__ = ("at_tick", "drift", "ewc_lambda", "der_alpha")

    def __init__(self, at_tick: int, drift: float, ewc_lambda: float, der_alpha: float) -> None:
        self.at_tick = at_tick
        self.drift = drift
        self.ewc_lambda = ewc_lambda
        self.der_alpha = der_alpha

    def to_dict(self) -> Dict[str, Any]:
        return {"at_tick": self.at_tick, "drift": round(self.drift, 5),
                "ewc_lambda": round(self.ewc_lambda, 4), "der_alpha": round(self.der_alpha, 4)}


class ContinuousLearner:
    """Drive the reward learner on the idle clock, and defend its memory from drift.

    ``consolidate_cb`` / ``weight_vector_fn`` / ``dominant_task_fn`` are optional callbacks the
    orchestrator wires in so the EWC snapshot uses exactly the same path as the per-turn one,
    without this module importing the synapse engine.
    """

    def __init__(self, learner: Any, *, consolidate_every: int = 8, replay_batch: int = 32,
                 drift_trigger: float = _DEFAULT_DRIFT_TRIGGER, max_probes: int = 24,
                 consolidate_cb: Optional[Callable[[], None]] = None) -> None:
        self.learner = learner
        self.consolidate_every = max(1, int(consolidate_every))
        self.replay_batch = max(1, int(replay_batch))
        self.drift_trigger = float(drift_trigger)
        self.max_probes = max(1, int(max_probes))
        self.consolidate_cb = consolidate_cb
        self._ticks = 0
        self._replayed_total = 0
        self._consolidations = 0
        self._defenses: List[ForgettingDefense] = []
        self._last_drift = 0.0
        # retention probes: (action, features) -> the value the skill had when last consolidated
        self._probes: Dict[Tuple[str, Tuple[Tuple[str, float], ...]], float] = {}

    # ---- the background heartbeat ---- #
    def tick(self) -> Dict[str, Any]:
        """One background learning pass. Never raises; returns a small honest report."""
        out: Dict[str, Any] = {"ticks": self._ticks + 1, "replayed": 0,
                               "consolidated": False, "drift": None, "defended": False}
        learner = self.learner
        if learner is None:
            return out
        self._ticks += 1
        # 1) rehearse — balanced across every task tag (old skills too)
        try:
            n = learner.replay(n=self.replay_batch, balanced=True)
            self._replayed_total += int(n)
            out["replayed"] = int(n)
        except Exception:  # noqa: BLE001 — a background pass never breaks the idle loop
            pass
        # 2) consolidate on cadence — freeze importances + snapshot EWC anchors
        if self._ticks % self.consolidate_every == 0:
            try:
                learner.consolidate()
                if self.consolidate_cb is not None:
                    self.consolidate_cb()
                self._consolidations += 1
                self._refresh_probes()             # the just-frozen values become the targets
                out["consolidated"] = True
            except Exception:  # noqa: BLE001
                pass
        # 3) self-monitor forgetting and defend if the consolidated skills have drifted
        drift = self._measure_drift()
        if drift is not None:
            self._last_drift = drift
            out["drift"] = round(drift, 5)
            if drift > self.drift_trigger:
                out["defended"] = self._defend(drift)
        return out

    # ---- retention probes (the forgetting self-monitor) ---- #
    def _refresh_probes(self) -> None:
        """Snapshot the current value of the most-consolidated skills as retention targets."""
        model = getattr(self.learner, "model", None)
        weights = getattr(model, "_w", None)
        if not weights:
            return
        # rank actions by total weight magnitude (a proxy for "an established skill")
        ranked = sorted(
            weights.items(),
            key=lambda kv: sum(abs(v) for v in kv[1].values()),
            reverse=True)[:self.max_probes]
        probes: Dict[Tuple[str, Tuple[Tuple[str, float], ...]], float] = {}
        for action, row in ranked:
            feats = {f: 1.0 for f in row if abs(row[f]) > 0.0}
            if not feats:
                continue
            key = (action, tuple(sorted(feats.items())))
            try:
                probes[key] = float(self.learner.value(action, feats))
            except Exception:  # noqa: BLE001
                continue
        if probes:
            self._probes = probes

    def _measure_drift(self) -> Optional[float]:
        """Mean absolute drift of the consolidated skills from their frozen target values."""
        if not self._probes:
            return None
        total = 0.0
        count = 0
        for (action, feat_items), target in self._probes.items():
            feats = {f: v for f, v in feat_items}
            try:
                now = float(self.learner.value(action, feats))
            except Exception:  # noqa: BLE001
                continue
            total += abs(now - target)
            count += 1
        return (total / count) if count else None

    def _defend(self, drift: float) -> bool:
        """Raise EWC + dark-replay protection (within caps) and rehearse harder. Real closed loop."""
        model = getattr(self.learner, "model", None)
        if model is None:
            return False
        try:
            new_lambda = min(_EWC_LAMBDA_CAP, max(model.ewc_lambda, 1.0) * 1.5)
            model.ewc_lambda = new_lambda
            new_der = min(_DER_ALPHA_CAP, max(getattr(self.learner, "der_alpha", 0.0), 0.1) * 1.5)
            self.learner.der_alpha = new_der
            # an immediate extra rehearsal pass to pull the drifted skills back
            self.learner.replay(n=self.replay_batch * 2, balanced=True)
            self._defenses.append(ForgettingDefense(self._ticks, drift, new_lambda, new_der))
            return True
        except Exception:  # noqa: BLE001
            return False

    # ---- report ---- #
    def stats(self) -> Dict[str, Any]:
        return {
            "ticks": self._ticks,
            "replayed_total": self._replayed_total,
            "consolidations": self._consolidations,
            "probes": len(self._probes),
            "last_drift": round(self._last_drift, 5),
            "drift_trigger": self.drift_trigger,
            "defenses_applied": len(self._defenses),
            "last_defense": (self._defenses[-1].to_dict() if self._defenses else None),
        }


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA continuous-learning self-test")
    print("=" * 70)

    from nyxara.growth.learn import Learner

    # a learner that has lived two skills
    L = Learner(base_lr=0.2, ewc_lambda=1.0, task_reserve=16, der_alpha=0.1, seed=1)
    for _ in range(60):
        L.record("greet_warmly", {"with_master": 1.0}, reward=1.0, task="greet")
    for _ in range(60):
        L.record("be_terse", {"task_urgent": 1.0}, reward=1.0, task="urgent")

    cl = ContinuousLearner(L, consolidate_every=4, replay_batch=16, drift_trigger=0.05)

    # background ticks with NO new turns still rehearse + consolidate — she keeps learning
    consolidated = 0
    for _ in range(12):
        rep = cl.tick()
        consolidated += 1 if rep["consolidated"] else 0
    assert cl._ticks == 12 and consolidated >= 2
    assert cl._replayed_total > 0 and cl._consolidations >= 2
    assert cl._probes, "no retention probes were registered"
    print(f"\nbackground learning : {cl._ticks} idle ticks, {cl._replayed_total} rehearsed, "
          f"{cl._consolidations} consolidations, {len(cl._probes)} probes ✓")

    # now shove hard on ONE skill to force the other to drift, and prove she defends her memory
    lam_before = L.model.ewc_lambda
    for _ in range(200):
        L.record("greet_warmly", {"with_master": 1.0}, reward=-1.0, task="greet")
    defended = False
    for _ in range(6):
        rep = cl.tick()
        defended = defended or rep["defended"]
    assert defended, "forgetting drift did not trigger a self-defense"
    assert L.model.ewc_lambda > lam_before, "EWC protection was not raised"
    print(f"forgetting defense  : drift {cl._last_drift:.3f} > trigger → raised EWC "
          f"{lam_before:.2f}→{L.model.ewc_lambda:.2f}, der_alpha→{L.der_alpha:.2f} ✓")

    print(f"\nstats               : {cl.stats()}")
    print("\nALL SELF-TESTS PASSED ✓")
