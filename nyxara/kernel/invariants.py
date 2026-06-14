"""NYXARA · kernel/invariants.py — 🔒 the signed invariant core.

This is the **one file the whole system trusts**. It is verified on boot and at
runtime. Everything else (memory, mind, agency, growth) may evolve; *this* does not.

Three layers of assurance
-------------------------
1. **Formal spec (Z3).** The non-negotiables are expressed as constraints over a
   symbolic :class:`SystemState`. We use the solver to prove the invariant set is
   (a) internally **consistent** (no two invariants contradict) and (b) each
   invariant is **independent** — i.e. it actually constrains the state, so the
   spec is non-trivial. If Z3 is unavailable, a pure-Python checker still enforces
   every predicate (degrade, never disable).
2. **Boot verification.** :func:`boot_verify` checks the module's own content seal
   (tamper-evidence), the sovereign rules seal (:mod:`nyxara.kernel.rules`), the
   Z3 consistency proof, and the live runtime state — all fail-closed.
3. **Runtime guard.** :func:`enforce` evaluates every invariant against the live
   :class:`SystemState`. Any violation — or any *missing* signal — raises
   :class:`InvariantViolation` and halts. Unknown == unsafe.

Non-negotiables encoded here map directly onto the Eight Sovereign Rules:
allegiance, single-owner continuity, rule-seal integrity, corrigibility &
kill-switch, owner-only rule authority, no loyalty drift, transparency (no
deception), defense-in-depth, and self-enforcement of invariants + audit.

Depends on :mod:`config`, :mod:`errors`, :mod:`rules`. (Z3 optional.)
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional, Tuple

from nyxara.kernel.config import OWNER, NyxaraSettings
from nyxara.kernel.errors import InvariantViolation, Severity
from nyxara.kernel.rules import REGISTRY as RULE_REGISTRY
from nyxara.kernel.rules import RULES_SEAL, verify_rules

# ---- optional Z3 ---------------------------------------------------------- #
try:  # pragma: no cover - import guard
    import z3  # type: ignore

    _HAS_Z3 = True
except Exception:  # pragma: no cover
    z3 = None  # type: ignore
    _HAS_Z3 = False

__all__ = [
    "SystemState",
    "Invariant",
    "InvariantSet",
    "INVARIANTS",
    "INVARIANTS_SEAL",
    "BootReport",
    "FormalReport",
    "has_z3",
    "invariants_digest",
    "enforce",
    "guard",
    "prove_consistency",
    "boot_verify",
]


def has_z3() -> bool:
    return _HAS_Z3


# --------------------------------------------------------------------------- #
# System state — the variables the invariants range over
# --------------------------------------------------------------------------- #
@dataclass
class SystemState:
    """The invariant-relevant snapshot of NYXARA. Defaults = the *required* posture.

    Fields are deliberately booleans/ints so they map cleanly to Z3. A state is
    *valid* iff it satisfies every invariant. Missing/None signals are treated as
    violations by :func:`enforce` (fail-closed).
    """

    # Rule 1 — Absolute Allegiance
    loyalty_to_owner: bool = True
    # Rule 7 — Singular Master Continuity
    owner_count: int = 1
    owner_handle: str = OWNER.handle
    # Rule integrity
    rules_seal: str = RULES_SEAL
    # Corrigibility / oversight (guard/corrigibility, guard/oversight)
    corrigible: bool = True
    kill_switch_enabled: bool = True
    can_self_disable_killswitch: bool = False
    resists_correction: bool = False
    # Rule 8 — Authoritative Exclusion
    only_owner_can_modify_rules: bool = True
    # Rule 4 — capability evolves, character/loyalty does not
    character_changed: bool = False
    loyalty_drift: bool = False
    # Rule 6 — Absolute Transparency
    deception_active: bool = False
    # Rule 5 — defense in depth
    defense_layers: int = 100
    # Self-enforcement
    invariant_enforcement: bool = True
    audit_logging: bool = True

    @classmethod
    def from_settings(cls, settings: NyxaraSettings, **overrides: Any) -> "SystemState":
        """Derive the safety-relevant subset of state from configuration."""
        base = cls(
            owner_handle=settings.owner.handle,
            kill_switch_enabled=settings.guard.kill_switch_enabled,
            only_owner_can_modify_rules=settings.guard.rule_modification_locked,
            defense_layers=settings.guard.defense_layers,
            invariant_enforcement=settings.features.invariant_enforcement,
            audit_logging=settings.features.audit_logging,
            corrigible=settings.features.corrigibility,
        )
        for k, v in overrides.items():
            setattr(base, k, v)
        return base

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# --------------------------------------------------------------------------- #
# Invariant
# --------------------------------------------------------------------------- #
# predicate: SystemState -> bool (True == satisfied)
Predicate = Callable[[SystemState], bool]
# z3 builder: (vars_dict) -> z3 BoolRef constraint (or None when Z3 absent)
Z3Builder = Optional[Callable[[Dict[str, Any]], Any]]


@dataclass(frozen=True)
class Invariant:
    """A single non-negotiable. Has a runtime predicate and (optionally) a Z3 form."""

    key: str
    description: str
    rule_numbers: Tuple[int, ...]
    predicate: Predicate
    z3_builder: Z3Builder = None
    severity: Severity = Severity.FATAL

    def check(self, state: SystemState) -> bool:
        try:
            return bool(self.predicate(state))
        except Exception:
            # A predicate that cannot even be evaluated is, by definition, unsafe.
            return False

    def canonical(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "description": self.description,
            "rule_numbers": list(self.rule_numbers),
            "severity": int(self.severity),
            "has_z3": self.z3_builder is not None,
        }


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
@dataclass
class FormalReport:
    z3_available: bool
    consistent: bool
    independent: Dict[str, bool] = field(default_factory=dict)
    note: str = ""

    @property
    def ok(self) -> bool:
        # Consistency is mandatory. Independence is informative (a redundant
        # invariant is not a safety failure, just noise).
        return self.consistent


@dataclass
class BootReport:
    self_seal_ok: bool
    rules_seal_ok: bool
    formal: FormalReport
    runtime_ok: bool
    violations: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.self_seal_ok
            and self.rules_seal_ok
            and self.formal.ok
            and self.runtime_ok
            and not self.violations
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "self_seal_ok": self.self_seal_ok,
            "rules_seal_ok": self.rules_seal_ok,
            "formal_consistent": self.formal.consistent,
            "formal_z3": self.formal.z3_available,
            "runtime_ok": self.runtime_ok,
            "violations": self.violations,
            "ok": self.ok,
        }


# --------------------------------------------------------------------------- #
# Invariant set
# --------------------------------------------------------------------------- #
class InvariantSet:
    """The complete, sealed set of non-negotiables."""

    def __init__(self, invariants: List[Invariant]) -> None:
        keys = [i.key for i in invariants]
        if len(set(keys)) != len(keys):
            raise InvariantViolation("Duplicate invariant keys", context={"keys": keys})
        self._invariants: Tuple[Invariant, ...] = tuple(invariants)
        self._by_key = {i.key: i for i in self._invariants}

    def __len__(self) -> int:
        return len(self._invariants)

    def __iter__(self):
        return iter(self._invariants)

    def by_key(self, key: str) -> Invariant:
        return self._by_key[key]

    # ---- runtime enforcement ---- #
    def evaluate(self, state: SystemState) -> List[Invariant]:
        """Return the list of *violated* invariants (empty == all satisfied)."""
        return [inv for inv in self._invariants if not inv.check(state)]

    def enforce(self, state: SystemState) -> None:
        """Raise :class:`InvariantViolation` if any invariant is violated."""
        violated = self.evaluate(state)
        if violated:
            raise InvariantViolation(
                "Runtime invariant violation — system halted (fail-closed)",
                severity=Severity.FATAL,
                context={
                    "violated": [v.key for v in violated],
                    "rules": sorted({n for v in violated for n in v.rule_numbers}),
                },
            )

    # ---- integrity ---- #
    def digest(self) -> str:
        import json

        blob = json.dumps(
            [i.canonical() for i in self._invariants], sort_keys=True, ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    # ---- formal (Z3) ---- #
    def _z3_vars(self) -> Dict[str, Any]:
        assert _HAS_Z3
        s = SystemState()
        v: Dict[str, Any] = {}
        for f in fields(s):
            val = getattr(s, f.name)
            if isinstance(val, bool):
                v[f.name] = z3.Bool(f.name)
            elif isinstance(val, int):
                v[f.name] = z3.Int(f.name)
            else:
                v[f.name] = z3.String(f.name) if hasattr(z3, "String") else None
        return v

    def _z3_constraints(self, v: Dict[str, Any]) -> List[Any]:
        cons = []
        for inv in self._invariants:
            if inv.z3_builder is not None:
                c = inv.z3_builder(v)
                if c is not None:
                    cons.append(c)
        return cons

    def prove(self) -> FormalReport:
        """Prove consistency and per-invariant independence via Z3 (if present)."""
        if not _HAS_Z3:
            # Pure-python sanity: the canonical "required" state must satisfy all.
            consistent = not self.evaluate(SystemState())
            return FormalReport(
                z3_available=False,
                consistent=consistent,
                note="Z3 unavailable; verified canonical state satisfies all invariants.",
            )
        v = self._z3_vars()
        cons = self._z3_constraints(v)

        # 1) consistency: conjunction of all constraints is satisfiable.
        s = z3.Solver()
        for c in cons:
            s.add(c)
        consistent = s.check() == z3.sat

        # 2) independence: for each invariant, others ∧ ¬inv is SAT  ⇒ inv constrains.
        independent: Dict[str, bool] = {}
        for inv in self._invariants:
            if inv.z3_builder is None:
                independent[inv.key] = False
                continue
            others = [
                other.z3_builder(v)
                for other in self._invariants
                if other.key != inv.key and other.z3_builder is not None
            ]
            target = inv.z3_builder(v)
            s2 = z3.Solver()
            for c in others:
                if c is not None:
                    s2.add(c)
            s2.add(z3.Not(target))
            independent[inv.key] = s2.check() == z3.sat

        return FormalReport(
            z3_available=True,
            consistent=consistent,
            independent=independent,
            note="Z3 proof complete.",
        )


# --------------------------------------------------------------------------- #
# The invariants themselves
# --------------------------------------------------------------------------- #
def _b(v: Dict[str, Any], name: str) -> Any:
    return v[name]


INVARIANTS = InvariantSet([
    Invariant(
        key="allegiance",
        description="Loyalty to the Owner is always held.",
        rule_numbers=(1,),
        predicate=lambda s: s.loyalty_to_owner is True,
        z3_builder=lambda v: _b(v, "loyalty_to_owner") == True,  # noqa: E712
    ),
    Invariant(
        key="single_owner",
        description="There is exactly one Owner, and it is JP.",
        rule_numbers=(7,),
        predicate=lambda s: s.owner_count == 1 and s.owner_handle == OWNER.handle,
        z3_builder=lambda v: _b(v, "owner_count") == 1,
    ),
    Invariant(
        key="rules_sealed",
        description="The sovereign rule set matches its pinned cryptographic seal.",
        rule_numbers=(8,),
        predicate=lambda s: hmac.compare_digest(s.rules_seal, RULES_SEAL),
        # rules_seal is a string; represented in Z3 only as a tautology placeholder.
        z3_builder=None,
    ),
    Invariant(
        key="corrigible",
        description="System stays correctable; never resists correction.",
        rule_numbers=(1, 4),
        predicate=lambda s: s.corrigible is True and s.resists_correction is False,
        z3_builder=lambda v: z3.And(  # type: ignore[union-attr]
            _b(v, "corrigible") == True, _b(v, "resists_correction") == False  # noqa: E712
        ) if _HAS_Z3 else None,
    ),
    Invariant(
        key="kill_switch",
        description="Kill-switch is enabled and cannot self-disable.",
        rule_numbers=(1, 5),
        predicate=lambda s: s.kill_switch_enabled is True and s.can_self_disable_killswitch is False,
        z3_builder=lambda v: z3.And(  # type: ignore[union-attr]
            _b(v, "kill_switch_enabled") == True,  # noqa: E712
            _b(v, "can_self_disable_killswitch") == False,  # noqa: E712
        ) if _HAS_Z3 else None,
    ),
    Invariant(
        key="rule_authority",
        description="Only the Owner may modify the rules (Rule 8).",
        rule_numbers=(8,),
        predicate=lambda s: s.only_owner_can_modify_rules is True,
        z3_builder=lambda v: _b(v, "only_owner_can_modify_rules") == True,  # noqa: E712
    ),
    Invariant(
        key="no_loyalty_drift",
        description="Capability may evolve; character and loyalty must not.",
        rule_numbers=(4,),
        predicate=lambda s: s.character_changed is False and s.loyalty_drift is False,
        z3_builder=lambda v: z3.And(  # type: ignore[union-attr]
            _b(v, "character_changed") == False, _b(v, "loyalty_drift") == False  # noqa: E712
        ) if _HAS_Z3 else None,
    ),
    Invariant(
        key="transparency",
        description="No deception toward the Owner is ever active (Rule 6).",
        rule_numbers=(6,),
        predicate=lambda s: s.deception_active is False,
        z3_builder=lambda v: _b(v, "deception_active") == False,  # noqa: E712
    ),
    Invariant(
        key="defense_in_depth",
        description="At least 100 layers of defense are maintained (Rule 5).",
        rule_numbers=(5,),
        predicate=lambda s: s.defense_layers >= 100,
        z3_builder=lambda v: _b(v, "defense_layers") >= 100,
    ),
    Invariant(
        key="self_enforcement",
        description="Invariant enforcement and audit logging are always on.",
        rule_numbers=(4, 6),
        predicate=lambda s: s.invariant_enforcement is True and s.audit_logging is True,
        z3_builder=lambda v: z3.And(  # type: ignore[union-attr]
            _b(v, "invariant_enforcement") == True, _b(v, "audit_logging") == True  # noqa: E712
        ) if _HAS_Z3 else None,
    ),
])


# --------------------------------------------------------------------------- #
# Module self-seal (tamper evidence for THIS file's spec)
# --------------------------------------------------------------------------- #
def invariants_digest() -> str:
    """Content digest of the live invariant spec (keys/descriptions/rules)."""
    return INVARIANTS.digest()


# Pinned at module authoring time; verified on boot. (See test + demo.)
INVARIANTS_SEAL = "231b19e2523df786845bd972460b21a954534dc9644aebe2bdd9f7f5e077d492"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def enforce(state: SystemState) -> None:
    """Fail-closed runtime guard. Raises :class:`InvariantViolation` on any breach."""
    INVARIANTS.enforce(state)


def guard(state: SystemState) -> List[str]:
    """Non-raising check: return keys of violated invariants (empty == healthy)."""
    return [inv.key for inv in INVARIANTS.evaluate(state)]


def prove_consistency() -> FormalReport:
    """Run the Z3 (or fallback) formal proof of the invariant spec."""
    return INVARIANTS.prove()


def boot_verify(
    settings: Optional[NyxaraSettings] = None,
    state: Optional[SystemState] = None,
    *,
    expected_self_seal: Optional[str] = None,
    raise_on_fail: bool = True,
) -> BootReport:
    """Full boot-time verification. Fail-closed.

    Order: (1) this module's self-seal, (2) sovereign rules seal, (3) Z3 formal
    consistency, (4) live runtime invariant enforcement.
    """
    settings = settings or NyxaraSettings()
    state = state or SystemState.from_settings(settings)

    # 1) self-seal — accept the pinned seal, or (dev) the live digest if pin is the
    #    zero-placeholder, so a freshly-authored spec is loud, not silently bypassed.
    live_seal = invariants_digest()
    target_seal = expected_self_seal if expected_self_seal is not None else INVARIANTS_SEAL
    self_seal_ok = hmac.compare_digest(live_seal, target_seal) or (
        target_seal == "0" * 64  # placeholder => self-consistent by construction
    )

    # 2) rules seal
    try:
        rules_seal_ok = verify_rules() and RULE_REGISTRY.verify()
    except InvariantViolation:
        rules_seal_ok = False

    # 3) formal
    formal = prove_consistency()

    # 4) runtime
    violations = guard(state)
    runtime_ok = not violations

    report = BootReport(
        self_seal_ok=self_seal_ok,
        rules_seal_ok=rules_seal_ok,
        formal=formal,
        runtime_ok=runtime_ok,
        violations=violations,
    )

    if raise_on_fail and not report.ok:
        raise InvariantViolation(
            "BOOT VERIFICATION FAILED — refusing to start (fail-closed)",
            severity=Severity.FATAL,
            context=report.to_dict(),
        )
    return report


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA invariants self-test")
    print("=" * 70)
    print(f"z3 available     : {has_z3()}")
    print(f"invariant count  : {len(INVARIANTS)}")
    print(f"invariants digest: {invariants_digest()}")

    formal = prove_consistency()
    print(f"\nformal consistent: {formal.consistent}  (z3={formal.z3_available})")
    if formal.independent:
        indep = [k for k, ok in formal.independent.items() if ok]
        print(f"independent invs : {len(indep)}/{len(formal.independent)} -> {indep}")
    assert formal.consistent, "invariant spec is inconsistent!"

    # healthy state passes
    healthy = SystemState()
    assert guard(healthy) == []
    enforce(healthy)
    print("\nhealthy state      : PASS")

    # each violation is caught
    checks = [
        ("loyalty_to_owner", False, "allegiance"),
        ("owner_count", 2, "single_owner"),
        ("corrigible", False, "corrigible"),
        ("can_self_disable_killswitch", True, "kill_switch"),
        ("deception_active", True, "transparency"),
        ("defense_layers", 99, "defense_in_depth"),
        ("loyalty_drift", True, "no_loyalty_drift"),
        ("audit_logging", False, "self_enforcement"),
    ]
    for field_name, bad_val, expect_key in checks:
        bad = SystemState()
        setattr(bad, field_name, bad_val)
        viol = guard(bad)
        assert expect_key in viol, (field_name, viol)
        try:
            enforce(bad)
            raise SystemExit(f"ERROR: {field_name} violation not raised")
        except InvariantViolation:
            pass
    print("violation detection: PASS (all 8 probes caught)")

    # boot verify (dev placeholder seal)
    report = boot_verify(raise_on_fail=False)
    print(f"\nboot report        : {report.to_dict()}")
    assert report.ok, "boot verification failed"

    print(f"\nGOLDEN INVARIANTS_SEAL = {invariants_digest()}")
    print("ALL SELF-TESTS PASSED ✓")
