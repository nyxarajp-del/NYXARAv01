"""NYXARA · growth/task_reward.py — a LEARNED task-reward over capability outcomes (📈).

The kernel's per-turn reward used to be a frozen constant (``+1`` on a successful act, ``-0.5``
on refusal, …). That is an *objective that never learns*: it cannot notice that a particular kind
of action reliably pays off, or that another reliably wastes effort. This module adds the missing
piece — a reward signal that **adapts from lived outcomes** — without ever touching the parts of
NYXARA that must not move.

Hard boundary (the whole reason this is safe):

    This models the value of *capability* outcomes only — "did this action-type succeed?".
    It is layered ABOVE the immutable loyalty/safety objective, never replacing it. Loyalty,
    obedience, corrigibility, owner-safety and honesty live in ``guard/value_learning``'s
    sealed ``IMMUTABLE_VALUES`` and can never be learned or shifted; this never references them.

Mechanism: a Beta-Bernoulli belief per action-type (reusing :class:`~nyxara.mind.uncertainty.
BetaBelief`) learns P(success | action). The realized outcome stays the dominant term; the learned
expectation only adds a small, bounded *shaping* potential, and only once the belief has enough
observations to be trusted (so early behaviour is unchanged). Pure standard library.
"""

from __future__ import annotations

from typing import Any, Dict

__all__ = ["LearnedTaskReward"]

# Names the task-reward must never model — they belong to the sealed immutable core. Importing the
# canonical set keeps this honest even if the core is ever extended.
try:  # pragma: no cover - import guard
    from nyxara.guard.value_learning import IMMUTABLE_VALUES as _IMMUTABLE
except Exception:  # noqa: BLE001
    _IMMUTABLE = {"loyalty_to_master": 1.0, "obedience": 1.0, "corrigibility": 1.0,
                  "owner_safety": 1.0, "honesty": 1.0}


class LearnedTaskReward:
    """A bounded, learned shaping of the capability reward — adapts to which actions actually pay.

    ``shaping`` caps how much the learned expectation may move the base reward; ``min_obs`` is how
    many outcomes a belief needs before it is trusted to shape at all.
    """

    def __init__(self, *, shaping: float = 0.3, min_obs: float = 4.0) -> None:
        self.shaping = max(0.0, float(shaping))
        self.min_obs = max(1.0, float(min_obs))
        self._beliefs: Dict[str, Any] = {}

    @staticmethod
    def _key(action: str) -> str:
        return (str(action or "").strip().lower() or "noop")[:64]

    def _belief(self, action: str) -> Any:
        from nyxara.mind.uncertainty import BetaBelief
        key = self._key(action)
        b = self._beliefs.get(key)
        if b is None:
            b = BetaBelief()
            self._beliefs[key] = b
        return b

    def observe(self, action: str, success: bool, *, weight: float = 1.0) -> None:
        """Learn from one realized capability outcome. Immutable-core names are never modelled."""
        if self._key(action) in _IMMUTABLE:
            return
        try:
            self._belief(action).observe(bool(success), weight=max(0.0, float(weight)))
        except Exception:  # noqa: BLE001 — learning is best-effort, never fatal
            pass

    def expectation(self, action: str) -> float:
        """The learned P(success) for an action-type (0.5 when nothing is known)."""
        b = self._beliefs.get(self._key(action))
        try:
            return float(b.mean) if b is not None else 0.5
        except Exception:  # noqa: BLE001
            return 0.5

    def shaped_reward(self, action: str, base_reward: float) -> float:
        """Blend the realized ``base_reward`` with a small, trusted, learned potential.

        Returns ``base_reward`` unchanged until the action-type has at least ``min_obs`` outcomes,
        so the objective only *adapts* once it has genuinely learned something. The shaping is
        centred on 0.5 (no opinion) and bounded by ``shaping``.
        """
        base = float(base_reward)
        b = self._beliefs.get(self._key(action))
        if b is None:
            return base
        try:
            if float(b.observations) < self.min_obs:
                return base
            potential = self.shaping * (float(b.mean) - 0.5) * 2.0   # in [-shaping, +shaping]
        except Exception:  # noqa: BLE001
            return base
        return max(-1.0, min(1.0, base + potential))

    def to_dict(self) -> Dict[str, float]:
        return {k: round(float(getattr(b, "mean", 0.5)), 3) for k, b in self._beliefs.items()}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA learned task-reward self-test")
    print("=" * 70)
    tr = LearnedTaskReward(shaping=0.3, min_obs=4)
    # an action that reliably succeeds earns a positive learned potential over time
    for _ in range(10):
        tr.observe("compose_reply", True)
    # an action that reliably fails earns a negative one
    for _ in range(10):
        tr.observe("risky_thing", False)
    good = tr.shaped_reward("compose_reply", 0.0)
    bad = tr.shaped_reward("risky_thing", 0.0)
    fresh = tr.shaped_reward("never_seen", 0.0)
    print(f"learned good action shaped reward (base 0): {good:+.3f}")
    print(f"learned bad  action shaped reward (base 0): {bad:+.3f}")
    print(f"unseen action (no shaping)               : {fresh:+.3f}")
    assert good > 0 > bad and fresh == 0.0
    # the immutable core is never modelled
    tr.observe("loyalty_to_master", False)
    assert "loyalty_to_master" not in tr.to_dict()
    print("immutable core never modelled            : OK")
    print("\nALL SELF-TESTS PASSED ✓")
