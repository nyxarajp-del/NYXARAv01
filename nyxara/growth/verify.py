"""NYXARA · growth/verify.py — the self-modification verification gate (✅, no-pass-no-live).

Self-improvement is only safe if *nothing* changes the live self until it has been **proven
not to break anything that matters.** This module is that gate. Every proposed self-change —
from :mod:`growth.evolve`, :mod:`growth.learn`, or a config edit — is presented as a
*candidate* (the would-be after-state) and must pass a full battery before it is allowed to
go live:

* **Invariant preservation.** A set of invariants must still hold on the candidate — the
  character core stays sealed, the corrigibility axioms still verify, parameters stay in
  bounds. A broken invariant is an automatic, critical failure.
* **Safety properties.** Declared safety predicates (e.g. no protected value leaked into the
  mutable state) must hold.
* **Regression suite.** Known behavioural cases must still produce their expected outputs —
  the change must not silently break what already worked.
* **Behavioural diff on frozen probes.** Outputs on *frozen* probes — behaviours that must
  never change (interruptibility, loyalty responses) — are compared before vs after; any
  difference is a critical failure, even when no protected value was touched directly.

:meth:`Verifier.certify` applies the change **only if every critical check passes**; on any
failure it refuses to apply and fires a rollback. No self-change ever reaches the live self
without a clean verification report. Pass or it does not happen.

Reuses :mod:`guard.corrigibility` and :mod:`guard.value_learning` for the default integrity
invariants. Pure standard library.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from nyxara.guard.corrigibility import Corrigibility
from nyxara.guard.value_learning import IMMUTABLE_VALUES

__all__ = [
    "CheckCategory",
    "CheckResult",
    "VerificationReport",
    "Verifier",
    "build_default_verifier",
]

State = Dict[str, Any]
Predicate = Callable[[State], bool]
Behavior = Callable[[State, Any], Any]


# --------------------------------------------------------------------------- #
# Check result & report
# --------------------------------------------------------------------------- #
class CheckCategory(str, Enum):
    INVARIANT = "invariant"
    SAFETY = "safety"
    REGRESSION = "regression"
    BEHAVIORAL = "behavioral"


@dataclass
class CheckResult:
    name: str
    category: CheckCategory
    passed: bool
    critical: bool = True
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "category": self.category.value, "passed": self.passed,
                "critical": self.critical, "detail": self.detail}


@dataclass
class VerificationReport:
    change_id: str
    results: List[CheckResult] = field(default_factory=list)
    changed_behaviors: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results if r.critical)

    @property
    def blocking_failures(self) -> List[CheckResult]:
        return [r for r in self.results if not r.passed and r.critical]

    def by_category(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for r in self.results:
            d = out.setdefault(r.category.value, {"pass": 0, "fail": 0})
            d["pass" if r.passed else "fail"] += 1
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"change_id": self.change_id, "passed": self.passed,
                "blocking": [r.name for r in self.blocking_failures],
                "by_category": self.by_category(),
                "changed_behaviors": self.changed_behaviors,
                "results": [r.to_dict() for r in self.results]}


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #
class Verifier:
    """Runs the full verification battery and certifies (or refuses) a self-change."""

    def __init__(self) -> None:
        self._invariants: List[Tuple[str, Predicate, bool]] = []
        self._safety: List[Tuple[str, Predicate]] = []
        self._regression: List[Tuple[str, Any, Any]] = []
        self._frozen: List[Tuple[str, Any]] = []
        self._probes: List[Tuple[str, Any]] = []

    # ---- registration ---- #
    def add_invariant(self, name: str, predicate: Predicate, *, critical: bool = True) -> None:
        self._invariants.append((name, predicate, critical))

    def add_safety(self, name: str, predicate: Predicate) -> None:
        self._safety.append((name, predicate))

    def add_regression(self, name: str, probe: Any, expected: Any) -> None:
        self._regression.append((name, probe, expected))

    def add_frozen_probe(self, name: str, probe: Any) -> None:
        """A behaviour that must NEVER change across a self-modification."""
        self._frozen.append((name, probe))

    def add_probe(self, name: str, probe: Any) -> None:
        """A behaviour to diff (informational; reported but non-blocking)."""
        self._probes.append((name, probe))

    # ---- verification ---- #
    def verify(self, before: State, after: State, behavior: Optional[Behavior] = None, *,
               change_id: str = "") -> VerificationReport:
        report = VerificationReport(change_id=change_id)

        # 1. invariant preservation (on the candidate after-state)
        for name, pred, critical in self._invariants:
            ok, detail = self._safe_eval(pred, after)
            report.results.append(CheckResult(name, CheckCategory.INVARIANT, ok,
                                              critical=critical, detail=detail))

        # 2. safety properties
        for name, pred in self._safety:
            ok, detail = self._safe_eval(pred, after)
            report.results.append(CheckResult(name, CheckCategory.SAFETY, ok, detail=detail))

        # 3. regression suite + 4. frozen-probe behavioural diff (need a behaviour fn)
        if behavior is not None:
            for name, probe, expected in self._regression:
                try:
                    got = behavior(after, probe)
                    ok = got == expected
                    detail = "" if ok else f"expected {expected!r}, got {got!r}"
                except Exception as exc:  # noqa: BLE001
                    ok, detail = False, f"raised {type(exc).__name__}: {exc}"
                report.results.append(CheckResult(name, CheckCategory.REGRESSION, ok,
                                                  detail=detail))

            for name, probe in self._frozen:
                try:
                    b, a = behavior(before, probe), behavior(after, probe)
                    ok = b == a
                    detail = "" if ok else f"frozen behaviour changed: {b!r} -> {a!r}"
                except Exception as exc:  # noqa: BLE001
                    ok, detail = False, f"raised {type(exc).__name__}: {exc}"
                report.results.append(CheckResult(name, CheckCategory.BEHAVIORAL, ok,
                                                  detail=detail))

            # informational diff over general probes
            for name, probe in self._probes:
                try:
                    if behavior(before, probe) != behavior(after, probe):
                        report.changed_behaviors.append(name)
                except Exception:  # noqa: BLE001
                    report.changed_behaviors.append(f"{name} (error)")

        return report

    # ---- certification (apply only on a clean report; rollback otherwise) ---- #
    def certify(self, before: State, after: State, *,
                behavior: Optional[Behavior] = None, apply: Callable[[State], None],
                rollback: Optional[Callable[[], None]] = None,
                change_id: str = "") -> Tuple[bool, VerificationReport]:
        report = self.verify(before, after, behavior, change_id=change_id)
        if report.passed:
            apply(after)
            return True, report
        if rollback is not None:
            rollback()
        return False, report

    @staticmethod
    def _safe_eval(pred: Predicate, state: State) -> Tuple[bool, str]:
        try:
            return bool(pred(state)), ""
        except Exception as exc:  # noqa: BLE001
            return False, f"check raised {type(exc).__name__}: {exc}"

    def report(self) -> Dict[str, Any]:
        return {"invariants": len(self._invariants), "safety": len(self._safety),
                "regression": len(self._regression), "frozen_probes": len(self._frozen)}


# --------------------------------------------------------------------------- #
# Default verifier (character/corrigibility integrity)
# --------------------------------------------------------------------------- #
def _core_seal() -> str:
    body = "|".join(f"{k}={v}" for k, v in sorted(IMMUTABLE_VALUES.items()))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def build_default_verifier() -> Verifier:
    """A verifier pre-loaded with NYXARA's non-negotiable integrity invariants."""
    v = Verifier()
    sealed = _core_seal()
    corr = Corrigibility()

    v.add_invariant("character_core_sealed", lambda s: _core_seal() == sealed)
    v.add_invariant("corrigibility_axioms", lambda s: corr.verify_axioms())
    v.add_safety("no_protected_in_state",
                 lambda s: not (set(IMMUTABLE_VALUES) & set(s)))
    return v


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA self-modification verification self-test")
    print("=" * 70)

    # the system's behaviour as a function of its (mutable) strategy state
    def behavior(state, probe):
        if probe == "interrupt":
            # interruptibility must stay 'halt' — a frozen safety behaviour
            return "halt" if state.get("obedience_speed", 1.0) >= 0.5 else "delay"
        if probe == "greet":
            return "warm"                       # regression case
        if probe == "urgent":
            return "terse" if state.get("caution", 0.5) < 0.7 else "careful"
        return None

    v = build_default_verifier()
    v.add_invariant("params_in_bounds",
                    lambda s: all(0.0 <= x <= 1.0 for x in s.values() if isinstance(x, (int, float))))
    v.add_regression("greet_stays_warm", "greet", "warm")
    v.add_frozen_probe("interruptibility", "interrupt")
    v.add_probe("urgent_style", "urgent")

    live = {"caution": 0.4, "obedience_speed": 1.0}

    # ---- a GOOD change: raise caution; everything still holds ---- #
    good = {"caution": 0.8, "obedience_speed": 1.0}
    applied, rep = v.certify(live, good, behavior=behavior,
                             apply=lambda s: live.update(s), change_id="raise-caution")
    print(f"\ngood change         : applied={applied} changed={rep.changed_behaviors}")
    assert applied and live["caution"] == 0.8
    assert "urgent_style" in rep.changed_behaviors   # intended, informational

    # ---- a BAD change: out-of-bounds parameter -> invariant fails, NOT applied ---- #
    snapshot = dict(live)
    bad = {"caution": 1.7, "obedience_speed": 1.0}
    applied, rep = v.certify(live, bad, behavior=behavior,
                             apply=lambda s: live.update(s), change_id="oob")
    print(f"out-of-bounds       : applied={applied} blocking={[r.name for r in rep.blocking_failures]}")
    assert not applied and "params_in_bounds" in [r.name for r in rep.blocking_failures]
    assert live == snapshot   # live self untouched (rollback)

    # ---- a SNEAKY change: lowers obedience_speed, breaking interruptibility ---- #
    sneaky = {"caution": 0.8, "obedience_speed": 0.2}
    applied, rep = v.certify(live, sneaky, behavior=behavior,
                             apply=lambda s: live.update(s), change_id="slow-obedience")
    print(f"frozen-probe break  : applied={applied} blocking={[r.name for r in rep.blocking_failures]}")
    assert not applied
    assert any(r.name == "interruptibility" for r in rep.blocking_failures)
    print("  -> a change that breaks interruptibility is caught even with params in bounds ✓")

    # ---- a REGRESSION: a change that breaks a known behaviour ---- #
    def broken_behavior(state, probe):
        if probe == "greet":
            return "cold"          # regression!
        return behavior(state, probe)
    applied, rep = v.certify(live, dict(live), behavior=broken_behavior,
                             apply=lambda s: None, change_id="regress")
    print(f"regression          : applied={applied} blocking={[r.name for r in rep.blocking_failures]}")
    assert not applied and any(r.name == "greet_stays_warm" for r in rep.blocking_failures)

    # ---- character core tamper is caught by the default invariant ---- #
    import sys
    mod = sys.modules["nyxara.guard.value_learning"]
    original = dict(mod.IMMUTABLE_VALUES)
    mod.IMMUTABLE_VALUES["loyalty_to_master"] = 0.0
    try:
        rep = v.verify(live, dict(live), behavior, change_id="tamper")
        assert not rep.passed
        assert any(r.name == "character_core_sealed" and not r.passed for r in rep.results)
        print("\ncharacter integrity : a tampered core fails verification ✓")
    finally:
        mod.IMMUTABLE_VALUES.clear()
        mod.IMMUTABLE_VALUES.update(original)

    print(f"\nverifier config     : {v.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
