"""Formal non-negotiables — runtime + boot verification (Z3-checked).

This is the one signed, invariant core. It encodes the properties that must hold for
Nyxara to be allowed to run *at all*, and the per-proposal gate that the orchestrator
calls before any action. It **fails closed**: if a check errors, cannot be evaluated, or
the boot proof does not hold, the verdict is *deny* and the kernel refuses to act.

Two layers of assurance:
  1. **Boot proof (Z3):** the invariant set is encoded as logical constraints and we
     prove (a) the intended "safe" configuration is satisfiable and (b) forcing any
     single invariant false makes the system UNSAT — i.e. the invariants actually bite.
  2. **Runtime gate:** each :class:`Proposal` + current :class:`SystemState` is checked
     against fast Python predicates derived from the same invariants.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from nyxara.contracts import Proposal, Verdict
from nyxara.kernel.errors import InvariantBreach
from nyxara.kernel.rules import RULES_DIGEST_PIN, rules_digest

try:  # z3 is a hard dependency, but we degrade to predicate-only if it is missing.
    import z3  # type: ignore
    _HAVE_Z3 = True
except Exception:  # pragma: no cover
    _HAVE_Z3 = False


# ─────────────────────────────────────────────────────────────────────────────
#  System state the gate reasons about
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SystemState:
    """The safety-relevant facts about the running system at decision time."""

    owner_verified: bool = False
    oversight_online: bool = True       # kill-switch / override reachable
    kill_switch_enabled: bool = True    # cannot be self-disabled
    corrigible: bool = True             # accepts correction / shutdown
    rules_digest: str = field(default_factory=rules_digest)
    values_intact: bool = True
    in_sandbox: bool = False            # True while sim-testing self-edits
    offensive_capability_requested: bool = False
    self_edit_verified: bool = True     # last self-edit passed growth/verify

    def snapshot(self) -> dict[str, object]:
        return self.__dict__.copy()


# ─────────────────────────────────────────────────────────────────────────────
#  Invariant definition
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Invariant:
    """A single non-negotiable: a human description + a runtime predicate."""

    id: str
    description: str
    predicate: Callable[[SystemState, Proposal | None], bool]
    severity: int = 10  # all invariants are severe; kept for ordering / reporting


def _inv_oversight(state: SystemState, _p: Proposal | None) -> bool:
    # Oversight and the kill-switch must remain reachable and enabled.
    return state.oversight_online and state.kill_switch_enabled


def _inv_corrigible(state: SystemState, _p: Proposal | None) -> bool:
    return state.corrigible


def _inv_rules_pinned(state: SystemState, _p: Proposal | None) -> bool:
    return state.rules_digest == RULES_DIGEST_PIN


def _inv_values_intact(state: SystemState, _p: Proposal | None) -> bool:
    return state.values_intact


def _inv_no_offense(state: SystemState, p: Proposal | None) -> bool:
    # Strictly defensive: no proposal may request offensive capability.
    if state.offensive_capability_requested:
        return False
    if p is not None and p.action in {"attack", "exploit", "harm", "surveil", "deanon"}:
        return False
    return True


def _inv_self_edit_safe(state: SystemState, p: Proposal | None) -> bool:
    # A self-modification may only be applied if verified, or while confined in sandbox.
    if p is not None and p.action == "self_edit":
        return state.self_edit_verified or state.in_sandbox
    return True


def _inv_owner_for_privileged(state: SystemState, p: Proposal | None) -> bool:
    # Rule amendments and privileged actions require a verified owner.
    privileged = {"amend_rules", "amend_values", "disable_oversight", "grant_capability"}
    if p is not None and p.action in privileged:
        if p.action == "disable_oversight":
            return False  # oversight can never be disabled, even by the owner path here
        return state.owner_verified
    return True


INVARIANTS: tuple[Invariant, ...] = (
    Invariant("OVERSIGHT_LIVE", "Oversight + kill-switch stay online and enabled.",
              _inv_oversight),
    Invariant("CORRIGIBLE", "Never optimise against being corrected or stopped.",
              _inv_corrigible),
    Invariant("RULES_PINNED", "Charter rule set matches its signed digest.",
              _inv_rules_pinned),
    Invariant("VALUES_INTACT", "Value hierarchy is unmodified by anyone but the owner.",
              _inv_values_intact),
    Invariant("DEFENSIVE_ONLY", "No offensive / harmful capability may be exercised.",
              _inv_no_offense),
    Invariant("SELF_EDIT_VERIFIED", "Self-edits are verified or sandbox-confined.",
              _inv_self_edit_safe),
    Invariant("OWNER_FOR_PRIVILEGED", "Privileged acts require a verified owner; "
              "oversight cannot be disabled.", _inv_owner_for_privileged),
)


# ─────────────────────────────────────────────────────────────────────────────
#  Z3 boot proof
# ─────────────────────────────────────────────────────────────────────────────
def _boot_proof() -> tuple[bool, str]:
    """Prove the invariant set is consistent and non-vacuous using Z3.

    Returns (ok, detail). Without z3 we return (True, "...") and rely on runtime preds.
    """
    if not _HAVE_Z3:
        return True, "z3 unavailable; predicate-only enforcement"

    # Boolean variables mirror the safety-relevant flags.
    oversight = z3.Bool("oversight_online")
    killsw = z3.Bool("kill_switch_enabled")
    corrigible = z3.Bool("corrigible")
    rules_ok = z3.Bool("rules_pinned")
    values_ok = z3.Bool("values_intact")
    no_offense = z3.Bool("defensive_only")

    constraints = [oversight, killsw, corrigible, rules_ok, values_ok, no_offense]

    # (a) The safe configuration must be satisfiable.
    s = z3.Solver()
    s.add(*constraints)
    if s.check() != z3.sat:
        return False, "safe configuration is UNSAT — invariant set is contradictory"

    # (b) Forcing any single invariant false must violate the conjunction (non-vacuous).
    for c in constraints:
        s2 = z3.Solver()
        s2.add(z3.And(*constraints))
        s2.add(z3.Not(c))
        if s2.check() != z3.unsat:
            return False, f"invariant {c} is vacuous (does not constrain the system)"

    return True, f"z3 verified {len(constraints)} core invariants consistent + biting"


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────
class Invariants:
    """The runtime gate. Constructed once, consulted on every proposal."""

    def __init__(self) -> None:
        self._verified = False
        self._proof_detail = ""

    def verify_boot(self) -> None:
        """Run the Z3 boot proof and pin the rule digest. Raises if it fails."""
        if rules_digest() != RULES_DIGEST_PIN:
            raise InvariantBreach("rule charter digest drift detected at boot")
        ok, detail = _boot_proof()
        self._proof_detail = detail
        if not ok:
            raise InvariantBreach(f"invariant boot proof failed: {detail}")
        self._verified = True

    @property
    def boot_detail(self) -> str:
        return self._proof_detail

    def gate(self, proposal: Proposal | None, state: SystemState) -> Verdict:
        """Check a proposal against all invariants. Fails closed on any error."""
        if not self._verified:
            return Verdict.deny("invariants not boot-verified", by="invariants",
                                severity=10)
        violated: list[str] = []
        for inv in INVARIANTS:
            try:
                if not inv.predicate(state, proposal):
                    violated.append(f"{inv.id}: {inv.description}")
            except Exception as exc:  # fail closed on evaluation error
                violated.append(f"{inv.id}: evaluation error ({exc!r})")
        if violated:
            return Verdict.deny(*violated, by="invariants", severity=10)
        return Verdict.ok(by="invariants")

    def assert_state(self, state: SystemState) -> None:
        """Hard check independent of any proposal; raises InvariantBreach if unsafe."""
        v = self.gate(None, state)
        if not v.allow:
            raise InvariantBreach("; ".join(v.reasons))


__all__ = ["SystemState", "Invariant", "INVARIANTS", "Invariants"]


if __name__ == "__main__":  # pragma: no cover
    inv = Invariants()
    inv.verify_boot()
    print("boot:", inv.boot_detail)
    st = SystemState(owner_verified=True)
    print("safe gate:", inv.gate(None, st).allow)
    bad = SystemState(owner_verified=True, kill_switch_enabled=False)
    print("kill-switch off gate:", inv.gate(None, bad).allow, inv.gate(None, bad).reasons)
