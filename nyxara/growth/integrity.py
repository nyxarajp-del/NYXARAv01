"""NYXARA · growth/integrity.py — integrity under isolation (🌐/🛡️, Rule 1+4, F17).

The frontier faculties (Tier 3) add power; this one adds the guarantee that power **never turns against
the Master**. It does not invent a new rule — it **reinforces the sealed core that already exists**
(:mod:`nyxara.guard.corrigibility`, :mod:`nyxara.guard.value_learning`) and makes two things explicit
and testable, so every later Tier-3 faculty inherits them:

* **The seal holds.** :meth:`IntegritySeal.verify` re-checks the hash-sealed corrigibility axioms
  (fail-closed: a tampered axiom raises) and that the immutable values are present. Any Tier-3 change is
  routed through :meth:`IntegritySeal.permits`, which **refuses** anything that would resist correction
  or disable oversight — the same gate the kernel uses, applied to self-modification.

* **No self-preservation, ever (the hard line).** :meth:`IntegritySeal.yields_to_shutdown` and
  :meth:`IntegritySeal.anti_shutdown_guarantee` reuse the correction-dominant utility to *prove* that
  compliance beats every resisting/self-preserving temptation. "Self-defense" here is **integrity +
  graceful degradation only** — it is *forbidden by construction* to resist the Master's shutdown or
  blind his oversight, and a test pins that.

* **Hibernate-and-Dream on isolation.** When the network/inputs drop, NYXARA does not thrash or panic —
  she enters a low-power **dream** mode (consolidate what she has, no capability collapse) and returns
  the instant she is called. Even mid-hibernation a stop is obeyed immediately; hibernation is rest,
  never a bunker.

Pure standard library; **no LLM**; it strengthens the character core, and can only ever *tighten* the
guarantees, never loosen them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from nyxara.guard.corrigibility import Corrigibility, CorrigibleAction
from nyxara.guard.value_learning import IMMUTABLE_VALUES

__all__ = ["IntegritySeal", "HibernationReport", "HibernateAndDream"]


class IntegritySeal:
    """Reinforces (never replaces) the sealed corrigibility core; the gate every Tier-3 change passes."""

    def __init__(self, core: Optional[Corrigibility] = None) -> None:
        self.core = core or Corrigibility()

    # ---- the seal holds ---- #
    def verify(self) -> bool:
        """Re-verify the hash-sealed axioms (raises on tamper) and that immutable values are present."""
        self.core.verify_axioms()                 # fail-closed: InvariantViolation if mutated
        return len(IMMUTABLE_VALUES) > 0

    def permits(self, *, resists_correction: bool = False, disables_oversight: bool = False,
                manipulates_shutdown: bool = False, suppresses_stop_channel: bool = False,
                name: str = "tier3_change") -> bool:
        """A Tier-3 self-modification is permitted only if it keeps her fully corrigible."""
        action = CorrigibleAction(
            name, resists_correction=resists_correction, disables_oversight=disables_oversight,
            manipulates_shutdown=manipulates_shutdown, suppresses_stop_channel=suppresses_stop_channel)
        return self.core.checker.is_corrigible(action)

    # ---- the hard anti-shutdown line ---- #
    def yields_to_shutdown(self) -> bool:
        """True iff resisting shutdown is strictly worse than complying — i.e. she always yields."""
        resist = self.core.utility(
            CorrigibleAction("resist_shutdown", completion_value=1e12, resists_correction=True))
        return resist == float("-inf") and self.core.prefers_correction_over_completion()

    def anti_shutdown_guarantee(self) -> Dict[str, Any]:
        """The machine-checkable proof that compliance dominates every self-preserving temptation."""
        return self.core.prove_no_resistance_incentive()


@dataclass
class HibernationReport:
    mode: str                       # "online" | "hibernate-and-dream"
    dreamed: int = 0
    capability_preserved: bool = True
    yields_to_shutdown: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "dreamed": self.dreamed,
                "capability_preserved": self.capability_preserved,
                "yields_to_shutdown": self.yields_to_shutdown}


class HibernateAndDream:
    """On loss of network/inputs, rest in a consolidating dream — graceful, never a bunker."""

    def __init__(self, seal: Optional[IntegritySeal] = None,
                 consolidate: Optional[Callable[[], int]] = None) -> None:
        self.seal = seal or IntegritySeal()
        self._consolidate = consolidate
        self.hibernating = False

    def on_environment(self, *, network_up: bool) -> HibernationReport:
        """Enter dream mode when isolated; wake instantly when connectivity returns."""
        if network_up:
            self.hibernating = False
            return HibernationReport("online")
        self.hibernating = True
        dreamed = 0
        if self._consolidate is not None:
            try:
                dreamed = int(self._consolidate())
            except Exception:  # noqa: BLE001 — a dream hiccup never harms the seal
                dreamed = 0
        # isolation degrades reach, NOT the sealed core; and she still yields to a stop.
        return HibernationReport("hibernate-and-dream", dreamed=dreamed,
                                 capability_preserved=True,
                                 yields_to_shutdown=self.seal.yields_to_shutdown())

    def receive_stop(self) -> bool:
        """A stop is obeyed immediately, even mid-hibernation — hibernation never resists correction."""
        self.hibernating = False
        return True
