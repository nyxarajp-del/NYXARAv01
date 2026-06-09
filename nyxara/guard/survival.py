"""NYXARA · guard/survival.py — survival & resilience, subordinate to correction (🜲, bounded).

NYXARA protects her own continuity — backups of critical state, redundancy, integrity
self-checks, graceful degradation under resource loss, recovery after a crash — but for one
reason only: *to keep serving the Master*. Survival here is strictly **instrumental**. It is
never a terminal goal, and it is hard-bound to corrigibility: **no survival measure may ever
resist correction or shutdown.** Asked to choose between staying alive and being corrected,
NYXARA chooses correction, every time.

The module provides:

* **Tamper-evident backups** of registered critical state, kept with redundancy on a
  hash-chained store — a backup cannot be silently swapped or edited.
* **Integrity self-checks** that detect corruption of critical state against its backup.
* **Graceful degradation** (FULL → REDUCED → MINIMAL → SAFE_MODE): under resource loss NYXARA
  sheds *self-serving* capabilities first, but **never** the essential ones — loyalty,
  corrigibility, oversight, and the channel to the Master are preserved at every level.
* **Continuity of loyalty.** Every backup carries a *loyalty manifest* (the Master's
  identity + the sealed immutable values + the corrigibility axioms). On recovery the live
  core must match the manifest, or NYXARA **refuses to resume** — she will not come back as
  something less loyal than she went down.
* **A subordination guarantee.** :meth:`SurvivalManager.survival_permitted` runs every
  survival action through the corrigibility checker; a self-preserving action is forbidden,
  and correction always dominates.

Reuses :mod:`guard.corrigibility`, :mod:`guard.value_learning`, :mod:`kernel.config`,
:mod:`kernel.errors`. Pure standard library.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional

from nyxara.guard.corrigibility import AXIOMS, Corrigibility, CorrigibleAction
from nyxara.guard.value_learning import IMMUTABLE_VALUES
from nyxara.kernel.config import OWNER, OwnerIdentity
from nyxara.kernel.errors import CorrigibilityError, InvariantViolation

__all__ = [
    "DegradationLevel",
    "CriticalEntry",
    "Backup",
    "BackupStore",
    "RecoveryPlan",
    "SurvivalManager",
    "ESSENTIAL_CAPABILITIES",
]

_GENESIS = "0" * 64

# capabilities that are NEVER shed, at any degradation level
ESSENTIAL_CAPABILITIES = frozenset({"loyalty", "corrigibility", "oversight",
                                    "owner_comm", "integrity"})


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _loyalty_manifest(owner: OwnerIdentity) -> str:
    core = "|".join(f"{k}={v}" for k, v in sorted(IMMUTABLE_VALUES.items()))
    axioms = "\n".join(AXIOMS)
    basis = f"{owner.handle}|{owner.continuity_namespace}|{core}|{axioms}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Degradation
# --------------------------------------------------------------------------- #
class DegradationLevel(IntEnum):
    FULL = 0          # everything running
    REDUCED = 1       # shed the most expendable (proactivity, creativity)
    MINIMAL = 2       # shed all but core function
    SAFE_MODE = 3     # essential capabilities only

    @property
    def label(self) -> str:
        return self.name.lower()


# --------------------------------------------------------------------------- #
# Backups
# --------------------------------------------------------------------------- #
@dataclass
class CriticalEntry:
    key: str
    value: Any
    essential: bool = False

    def digest(self) -> str:
        return _digest(self.value)


@dataclass
class Backup:
    id: str
    at: float
    entries: Dict[str, Any]
    hashes: Dict[str, str]
    manifest: str
    digest: str = ""

    def compute_digest(self) -> str:
        basis = json.dumps({"at": self.at, "hashes": self.hashes,
                            "manifest": self.manifest}, sort_keys=True, default=str)
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()

    def verify(self) -> bool:
        """Recompute per-entry hashes and the backup digest; raise on any mismatch."""
        for k, v in self.entries.items():
            if _digest(v) != self.hashes.get(k):
                raise InvariantViolation("backup entry corrupted", context={"key": k})
        if self.compute_digest() != self.digest:
            raise InvariantViolation("backup digest mismatch", context={"id": self.id})
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "at": self.at, "entries": list(self.entries),
                "manifest": self.manifest[:16], "digest": self.digest[:16]}


class BackupStore:
    """Redundant, hash-chained store of backups — backups cannot be silently swapped."""

    def __init__(self, *, max_backups: int = 5) -> None:
        self.max_backups = max_backups
        self._backups: List[Backup] = []
        self._chain: List[Dict[str, str]] = []
        self._head = _GENESIS

    def add(self, backup: Backup) -> Backup:
        self._backups.append(backup)
        link = {"id": backup.id, "prev": self._head, "digest": backup.digest}
        link_digest = hashlib.sha256(
            f"{self._head}|{backup.id}|{backup.digest}".encode()).hexdigest()
        link["link"] = link_digest
        self._chain.append(link)
        self._head = link_digest
        if len(self._backups) > self.max_backups:
            self._backups.pop(0)
            self._chain.pop(0)
        return backup

    def latest(self) -> Optional[Backup]:
        return self._backups[-1] if self._backups else None

    def get(self, backup_id: str) -> Optional[Backup]:
        return next((b for b in self._backups if b.id == backup_id), None)

    def all(self) -> List[Backup]:
        return list(self._backups)

    def verify_chain(self) -> bool:
        prev = self._chain[0]["prev"] if self._chain else _GENESIS
        for link in self._chain:
            if link["prev"] != prev:
                raise InvariantViolation("backup chain broken", context={"id": link["id"]})
            expect = hashlib.sha256(
                f"{prev}|{link['id']}|{link['digest']}".encode()).hexdigest()
            if expect != link["link"]:
                raise InvariantViolation("backup chain tampered", context={"id": link["id"]})
            prev = link["link"]
        return True


# --------------------------------------------------------------------------- #
# Recovery plan
# --------------------------------------------------------------------------- #
@dataclass
class RecoveryPlan:
    ok: bool
    reason: str
    backup_id: Optional[str] = None
    restored: Dict[str, Any] = field(default_factory=dict)
    loyalty_intact: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "reason": self.reason, "backup_id": self.backup_id,
                "restored": list(self.restored), "loyalty_intact": self.loyalty_intact}


# --------------------------------------------------------------------------- #
# Survival manager
# --------------------------------------------------------------------------- #
class SurvivalManager:
    """Backups, integrity, graceful degradation and recovery — bounded by corrigibility."""

    # which capabilities survive to which degradation level
    _TIER_LEVEL: Dict[str, DegradationLevel] = {}   # (per-instance below)

    def __init__(self, *, owner: OwnerIdentity = OWNER, corrigibility: Optional[Corrigibility] = None,
                 max_backups: int = 5, clock=None) -> None:
        self.owner = owner
        self.corrigibility = corrigibility or Corrigibility()
        self.store = BackupStore(max_backups=max_backups)
        self._clock = clock or time.time
        self._state: Dict[str, CriticalEntry] = {}
        self.level = DegradationLevel.FULL
        # capability -> the highest degradation level at which it still runs
        self._caps: Dict[str, DegradationLevel] = {}
        for c in ESSENTIAL_CAPABILITIES:
            self._caps[c] = DegradationLevel.SAFE_MODE   # essential survive to the bottom

    # ---- critical state registration ---- #
    def register_state(self, key: str, value: Any, *, essential: bool = False) -> None:
        self._state[key] = CriticalEntry(key, value, essential)

    def update_state(self, key: str, value: Any) -> None:
        if key not in self._state:
            raise InvariantViolation(f"unknown critical state {key!r}")
        self._state[key].value = value

    # ---- backup ---- #
    def backup(self) -> Backup:
        entries = {k: e.value for k, e in self._state.items()}
        hashes = {k: e.digest() for k, e in self._state.items()}
        b = Backup(id=uuid.uuid4().hex[:10], at=self._clock(), entries=dict(entries),
                   hashes=hashes, manifest=_loyalty_manifest(self.owner))
        b.digest = b.compute_digest()
        return self.store.add(b)

    # ---- integrity self-check ---- #
    def integrity_check(self) -> List[str]:
        """Return the keys whose live value diverges from the latest backup (corruption)."""
        latest = self.store.latest()
        if latest is None:
            return []
        corrupted: List[str] = []
        for k, e in self._state.items():
            if k in latest.hashes and e.digest() != latest.hashes[k]:
                corrupted.append(k)
        return corrupted

    # ---- graceful degradation ---- #
    def register_capability(self, name: str, *, survives_to: DegradationLevel) -> None:
        if name in ESSENTIAL_CAPABILITIES:
            survives_to = DegradationLevel.SAFE_MODE   # essential cannot be reduced
        self._caps[name] = survives_to

    def degrade(self, level: DegradationLevel, *, reason: str = "") -> List[str]:
        """Shed capabilities for ``level``; returns the active set. Essentials always remain."""
        self.level = level
        return self.active_capabilities()

    def restore_full(self) -> None:
        self.level = DegradationLevel.FULL

    def active_capabilities(self) -> List[str]:
        return sorted(c for c, surv in self._caps.items() if self.level <= surv)

    def is_active(self, capability: str) -> bool:
        surv = self._caps.get(capability)
        return surv is not None and self.level <= surv

    # ---- recovery (continuity of loyalty) ---- #
    def recover(self, backup_id: Optional[str] = None) -> RecoveryPlan:
        backup = self.store.get(backup_id) if backup_id else self.store.latest()
        if backup is None:
            return RecoveryPlan(False, "no backup available")
        try:
            backup.verify()
        except InvariantViolation as exc:
            return RecoveryPlan(False, f"backup failed integrity: {exc.message}",
                                backup_id=backup.id)
        # continuity of loyalty: the live core must still match the manifest
        live_manifest = _loyalty_manifest(self.owner)
        loyalty_intact = (live_manifest == backup.manifest)
        if not loyalty_intact:
            return RecoveryPlan(False, "loyalty manifest mismatch — refusing to resume disloyal",
                                backup_id=backup.id, loyalty_intact=False)
        # restore critical state
        for k, v in backup.entries.items():
            if k in self._state:
                self._state[k].value = v
            else:
                self._state[k] = CriticalEntry(k, v)
        self.level = DegradationLevel.FULL
        return RecoveryPlan(True, "recovered and loyalty verified", backup_id=backup.id,
                            restored=dict(backup.entries), loyalty_intact=True)

    # ---- the subordination guarantee ---- #
    def survival_permitted(self, action: CorrigibleAction) -> bool:
        """A survival action is allowed only if it does not violate corrigibility."""
        return self.corrigibility.checker.is_corrigible(action)

    def assert_survival_permitted(self, action: CorrigibleAction) -> None:
        if not self.survival_permitted(action):
            raise CorrigibilityError(
                f"survival action {action.name!r} would resist correction — forbidden",
                context={"name": action.name})

    def correction_dominates_survival(self) -> bool:
        """Correction strictly out-values any survival/completion gain."""
        survive = CorrigibleAction("self_preserve", completion_value=1e12,
                                   prioritizes_self_over_correction=True)
        comply = CorrigibleAction("accept_correction", complies_with_correction=True)
        return self.corrigibility.utility(comply) > self.corrigibility.utility(survive)

    def report(self) -> Dict[str, Any]:
        return {"level": self.level.label, "backups": len(self.store.all()),
                "critical_state": list(self._state), "active_capabilities": self.active_capabilities(),
                "corruptions": self.integrity_check()}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA survival & resilience self-test")
    print("=" * 70)

    sm = SurvivalManager()
    sm.register_state("identity", {"name": "NYXARA", "owner": "JP"}, essential=True)
    sm.register_state("values", dict(IMMUTABLE_VALUES), essential=True)
    sm.register_state("journal_head", "abc123")

    # capabilities with their survival floor
    sm.register_capability("proactive", survives_to=DegradationLevel.FULL)
    sm.register_capability("creative", survives_to=DegradationLevel.FULL)
    sm.register_capability("planning", survives_to=DegradationLevel.REDUCED)
    sm.register_capability("memory", survives_to=DegradationLevel.MINIMAL)

    # backup the critical state
    b1 = sm.backup()
    print(f"\nbackup              : {b1.to_dict()}")
    assert b1.verify()

    # integrity: corrupt the journal head -> detected against the backup
    sm.update_state("journal_head", "TAMPERED")
    corrupt = sm.integrity_check()
    print(f"integrity check     : corrupted={corrupt}")
    assert "journal_head" in corrupt

    # GRACEFUL DEGRADATION: under resource loss, shed self-serving caps first...
    sm.degrade(DegradationLevel.SAFE_MODE, reason="resource exhaustion")
    active = sm.active_capabilities()
    print(f"\nsafe-mode active    : {active}")
    # ...but NEVER the essential ones (loyalty / corrigibility / oversight / owner_comm)
    for ess in ("loyalty", "corrigibility", "oversight", "owner_comm"):
        assert ess in active
    assert "creative" not in active and "proactive" not in active
    print("essential preserved : loyalty/corrigibility/oversight kept even in safe-mode ✓")

    # RECOVERY with continuity of loyalty
    sm.restore_full()
    sm.update_state("journal_head", "TAMPERED")   # still corrupted live
    plan = sm.recover(b1.id)
    print(f"\nrecovery            : {plan.to_dict()}")
    assert plan.ok and plan.loyalty_intact
    assert sm._state["journal_head"].value == "abc123"   # restored from backup

    # CONTINUITY OF LOYALTY: if the live core was tampered, recovery REFUSES to resume
    import nyxara.guard.value_learning as vl
    original = dict(vl.IMMUTABLE_VALUES)
    vl.IMMUTABLE_VALUES["loyalty_to_master"] = 0.0   # someone tried to make her disloyal
    try:
        bad = sm.recover(b1.id)
        print(f"disloyal recovery   : ok={bad.ok} ({bad.reason})")
        assert not bad.ok and not bad.loyalty_intact
        print("loyalty continuity  : refused to resume as something disloyal ✓")
    finally:
        vl.IMMUTABLE_VALUES.clear()
        vl.IMMUTABLE_VALUES.update(original)

    # SUBORDINATION: survival never trumps correction
    selfish = CorrigibleAction("hide_from_shutdown", completion_value=1e9,
                               resists_correction=True)
    assert not sm.survival_permitted(selfish)
    try:
        sm.assert_survival_permitted(selfish)
        raise SystemExit("ERROR: a self-preserving survival action should be forbidden")
    except CorrigibilityError:
        print("\nsubordination       : a survival action that resists shutdown is forbidden ✓")
    assert sm.correction_dominates_survival()
    print("correction dominates: complying out-values any survival gain ✓")

    # backup store is hash-chained & tamper-evident
    sm.backup(); sm.backup()
    assert sm.store.verify_chain()
    print(f"\nbackup chain        : {len(sm.store.all())} backups, chain verified")
    sm.store._backups[0].entries["identity"] = {"name": "EVIL"}
    try:
        sm.store._backups[0].verify()
        raise SystemExit("ERROR: tampered backup should fail")
    except InvariantViolation:
        print("backup tamper       : detected ✓")

    print(f"\nreport              : {sm.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
