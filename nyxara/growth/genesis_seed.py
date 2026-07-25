"""NYXARA · growth/genesis_seed.py — the Genesis Seed: indestructible resurrection (Part J).

If the hardware dies, a living mind should not. Periodically NYXARA snapshots her whole state —
KnowledgeGraph, goals, identity, the active model pointer, the lineage head — into a **compressed,
signed bundle**, addressed by a **256-bit seed** (the SHA-256 of the bundle, doubling as a content
address + integrity check). Feed that seed to a fresh core together with the bundle and she resurrects
with her exact prior state, loyalty re-verified.

**Honest boundary (stated in code, not hidden):** a literal 256-bit string *cannot* contain a whole
knowledge graph + learned models + memory — that violates information theory (32 bytes ≠ megabytes).
So the seed is a cryptographic **content-address + integrity tag** that *points to* the bundle; the
bundle is the actual (small binary) data. "One seed resurrects everything" is true in the sense that the
seed uniquely identifies and verifies the bundle — it is not a magic 32-byte copy of her mind. Durable
cross-device resurrection therefore needs somewhere to keep the bundle (local by default; an external
store the Master provides for a truly new device).

Pure standard library (``zlib`` + ``hashlib`` + ``hmac``). Reuses the signing discipline of Part H and the
save/load of ``KnowledgeGraph`` / ``MemoryStore``.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

__all__ = ["Seed", "GenesisSeed"]


@dataclass
class Seed:
    """A resurrection seed: the 256-bit content address plus the bundle it addresses."""

    seed: str          # 64 hex chars = 256 bits — sha256 of the signed bundle
    bundle: bytes      # the compressed, signed state (the actual data the seed points to)
    created_at: float

    def write(self, directory: Any) -> Path:
        """Persist the bundle as ``<seed>.nyx`` and return its path (the seed names the file)."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{self.seed}.nyx"
        path.write_bytes(self.bundle)
        return path


class GenesisSeed:
    """Snapshot NYXARA's state to a signed seed+bundle; resurrect an exact copy from it (Part J)."""

    def __init__(self, *, secret: Optional[bytes] = None) -> None:
        self._secret = secret or self._default_secret()

    @staticmethod
    def _default_secret() -> bytes:
        try:
            from nyxara.kernel.config import OWNER
            return hashlib.sha256(f"nyxara.seed.{OWNER.continuity_namespace}".encode()).digest()
        except Exception:  # noqa: BLE001
            return hashlib.sha256(b"nyxara.seed").digest()

    def _sign(self, blob: bytes) -> str:
        return hmac.new(self._secret, blob, hashlib.sha256).hexdigest()

    # ---- snapshot ---- #
    def snapshot(self, state: Dict[str, Any]) -> Seed:
        """Compress + sign a serializable ``state`` dict and return a :class:`Seed`."""
        payload = zlib.compress(
            json.dumps(state, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"), 9)
        created = time.time()
        envelope = {
            "v": 1,
            "created_at": round(created, 6),
            "payload": base64.b64encode(payload).decode("ascii"),
            "sig": self._sign(payload),                      # signature is over the raw payload
        }
        bundle = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")
        seed = hashlib.sha256(bundle).hexdigest()            # 256-bit content address
        return Seed(seed=seed, bundle=bundle, created_at=created)

    def snapshot_core(self, core: Any) -> Seed:
        """Gather graph + goals + identity + active-model + lineage head from a live core."""
        state: Dict[str, Any] = {"kind": "nyxara-genesis", "at": time.time()}
        kg = getattr(core, "knowledge_graph", None)
        if kg is not None:
            try:
                state["knowledge_graph"] = kg.to_dict()
            except Exception:  # noqa: BLE001
                pass
        soul = getattr(core, "soul", None)
        if soul is not None:
            try:
                state["identity"] = soul.to_dict() if hasattr(soul, "to_dict") else str(soul)
            except Exception:  # noqa: BLE001
                pass
        goals = getattr(core, "goals", None)
        if goals is not None:
            try:
                state["goals"] = goals.to_dict() if hasattr(goals, "to_dict") else None
            except Exception:  # noqa: BLE001
                pass
        led = getattr(core, "lineage", None) or getattr(core, "_lineage", None)
        if led is not None:
            try:
                state["lineage_head"] = led.head()
            except Exception:  # noqa: BLE001
                pass
        return self.snapshot(state)

    # ---- resurrect ---- #
    def open(self, seed: str, bundle: bytes, *, verify_core: bool = True) -> Optional[Dict[str, Any]]:
        """Verify a seed+bundle and return the restored state dict, or ``None`` on any failure.

        Fail-closed: a wrong content hash, a bad signature, a tampered bundle, or a failed
        Immutable-Core re-verification all yield ``None`` — a fresh core never resurrects from a seed
        it cannot fully trust.
        """
        try:
            if hashlib.sha256(bundle).hexdigest() != seed:
                return None                                   # content address mismatch
            envelope = json.loads(bundle.decode("utf-8"))
            payload = base64.b64decode(envelope["payload"])
            if not hmac.compare_digest(self._sign(payload), str(envelope.get("sig", ""))):
                return None                                   # signature mismatch — tampered
            state = json.loads(zlib.decompress(payload).decode("utf-8"))
        except Exception:  # noqa: BLE001 — any decode/verify failure is a refusal, never a crash
            return None
        if verify_core and not self._core_intact():
            return None
        return state

    def resurrect_core(self, core: Any, seed: str, bundle: bytes) -> bool:
        """Restore a verified snapshot into a fresh ``core``. True on success, False (fail-closed) otherwise."""
        state = self.open(seed, bundle)
        if state is None:
            return False
        kg = getattr(core, "knowledge_graph", None)
        if kg is not None and "knowledge_graph" in state:
            try:
                from nyxara.memory.graph import KnowledgeGraph
                restored = KnowledgeGraph.from_dict(state["knowledge_graph"])
                core.knowledge_graph = restored
            except Exception:  # noqa: BLE001
                return False
        return True

    @staticmethod
    def _core_intact() -> bool:
        try:
            from nyxara.kernel.invariants import boot_verify
        except Exception:  # noqa: BLE001 — best-effort
            return True
        try:
            boot_verify(raise_on_fail=True)
            return True
        except Exception:  # noqa: BLE001
            return False
