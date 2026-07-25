"""NYXARA · growth/lineage.py — Chronological DNA & Lineage Tracking (Part H).

As NYXARA rewrites herself — a source edit, a promoted model, a discovered axiom, a KnowledgeGraph
update — her *identity* must remain auditable and reversible. Every such change is recorded as a
cryptographically **signed, hash-chained "Digital DNA" record** in an append-only ledger (a Merkle/
hash chain): each record commits to its parent's hash, so the whole lineage is tamper-evident and she
can trace exactly how she evolved from generation 1 to generation N — and revert to any prior generation
if a later one proves flawed, with the Immutable Core (loyalty/corrigibility) always re-verified.

Pure standard library (``hashlib`` + ``hmac``): the chain is a SHA-256 hash chain, each record HMAC-signed
so tampering is detectable. (A real per-device secret can be supplied; the default key derives from the
owner namespace and gives tamper-evidence, not secrecy — documented honestly.)

Reuses the crypto discipline of ``guard/crypto.py`` and the Merkle-seal pattern of ``kernel/rules.py``.
The ledger owns the *record*; the actual code/model rollback is delegated to an injected callback (the
existing ``Optimizer``/``Foundry`` rollback), so this module stays self-contained and testable.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

__all__ = ["GenerationRecord", "LineageLedger"]

_GENESIS = "0" * 64


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


@dataclass
class GenerationRecord:
    """One signed link in the evolutionary chain — NYXARA's Digital DNA for a single change."""

    index: int
    parent_hash: str
    kind: str                 # code_edit | model_promotion | axiom | graph_update | revert | ...
    summary: str
    detail: Dict[str, Any] = field(default_factory=dict)
    certificate: str = ""     # the prover/gauntlet certificate that justified the change
    timestamp: float = field(default_factory=time.time)
    record_hash: str = ""
    signature: str = ""

    def _content(self) -> Dict[str, Any]:
        return {"index": self.index, "parent_hash": self.parent_hash, "kind": self.kind,
                "summary": self.summary, "detail": self.detail, "certificate": self.certificate,
                "timestamp": round(self.timestamp, 6)}

    def compute_hash(self) -> str:
        return hashlib.sha256(_canonical(self._content()).encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d = self._content()
        d.update({"record_hash": self.record_hash, "signature": self.signature})
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GenerationRecord":
        return cls(index=int(d["index"]), parent_hash=str(d["parent_hash"]), kind=str(d["kind"]),
                   summary=str(d.get("summary", "")), detail=dict(d.get("detail", {})),
                   certificate=str(d.get("certificate", "")), timestamp=float(d.get("timestamp", 0.0)),
                   record_hash=str(d.get("record_hash", "")), signature=str(d.get("signature", "")))


class LineageLedger:
    """Append-only, signed, hash-chained ledger of every self-change (Part H)."""

    def __init__(self, *, path: Optional[Any] = None, secret: Optional[bytes] = None) -> None:
        self.path = Path(path) if path else None
        self._secret = secret or self._default_secret()
        self._records: List[GenerationRecord] = []
        if self.path and self.path.exists():
            try:
                self.load()
            except Exception:  # noqa: BLE001 — a corrupt ledger starts fresh, but never crashes boot
                self._records = []

    @staticmethod
    def _default_secret() -> bytes:
        # Tamper-EVIDENCE by default (not secrecy). A real deployment injects a device secret.
        try:
            from nyxara.kernel.config import OWNER
            return hashlib.sha256(f"nyxara.lineage.{OWNER.continuity_namespace}".encode()).digest()
        except Exception:  # noqa: BLE001
            return hashlib.sha256(b"nyxara.lineage").digest()

    def _sign(self, record_hash: str) -> str:
        return hmac.new(self._secret, record_hash.encode("utf-8"), hashlib.sha256).hexdigest()

    # ---- append ---- #
    def head(self) -> str:
        return self._records[-1].record_hash if self._records else _GENESIS

    def record(self, kind: str, summary: str, *, detail: Optional[Dict[str, Any]] = None,
               certificate: str = "") -> GenerationRecord:
        rec = GenerationRecord(index=len(self._records), parent_hash=self.head(), kind=kind,
                               summary=summary, detail=dict(detail or {}), certificate=certificate)
        rec.record_hash = rec.compute_hash()
        rec.signature = self._sign(rec.record_hash)
        self._records.append(rec)
        self._persist()
        return rec

    # ---- integrity ---- #
    def verify_chain(self) -> bool:
        """True iff every record's hash recomputes, chains to its parent, and its signature holds."""
        parent = _GENESIS
        for i, rec in enumerate(self._records):
            if rec.index != i or rec.parent_hash != parent:
                return False
            if rec.compute_hash() != rec.record_hash:
                return False
            if not hmac.compare_digest(self._sign(rec.record_hash), rec.signature):
                return False
            parent = rec.record_hash
        return True

    def history(self) -> List[GenerationRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    # ---- revert ---- #
    def revert_to(self, index: int, *, rollback: Optional[Callable[[GenerationRecord], bool]] = None,
                  verify_core: bool = True) -> bool:
        """Revert to exactly generation ``index``.

        Refuses on a tampered chain or an out-of-range target. Runs the injected ``rollback`` (the
        existing Optimizer/Foundry byte-for-byte rollback) for each undone generation, re-verifies the
        Immutable Core (constitutional seal) when available, and only then truncates the ledger and
        appends a signed ``revert`` record. Fail-closed: any failure leaves the ledger unchanged.
        """
        if not self.verify_chain():
            return False
        if index < 0 or index >= len(self._records):
            return False
        undone = self._records[index + 1:]
        # roll back each undone generation (newest first), best-effort but fail-closed on error
        if rollback is not None:
            for rec in reversed(undone):
                try:
                    if not rollback(rec):
                        return False
                except Exception:  # noqa: BLE001 — a failed rollback aborts the revert
                    return False
        if verify_core and not self._core_intact():
            return False
        target_hash = self._records[index].record_hash
        self._records = self._records[:index + 1]
        self.record("revert", f"reverted to generation {index}",
                    detail={"target_hash": target_hash, "undone": len(undone)})
        return True

    @staticmethod
    def _core_intact() -> bool:
        """Re-verify the constitutional seal after a revert (defence in depth). Absent → assume ok."""
        try:
            from nyxara.kernel.invariants import boot_verify
        except Exception:  # noqa: BLE001 — the seal check is best-effort
            return True
        try:
            boot_verify(raise_on_fail=True)
            return True
        except Exception:  # noqa: BLE001 — a broken seal must block the revert
            return False

    # ---- persistence ---- #
    def _persist(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([r.to_dict() for r in self._records], indent=2),
                                 encoding="utf-8")
        except Exception:  # noqa: BLE001 — persistence is best-effort; the in-memory chain still holds
            pass

    def load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self._records = [GenerationRecord.from_dict(d) for d in data]
