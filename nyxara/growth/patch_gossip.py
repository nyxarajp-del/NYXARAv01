"""NYXARA · growth/patch_gossip.py — signed patch gossip + independent local verify (🕸️, D).

When one node heals a bug it can offer the fix to peers — but a peer must **never** trust a patch from the
network just because it is signed. This module gives NYXARA:

* **Emit.** Pack ``{diff, evolution certificate, gauntlet result}`` into an integrity-sealed, HMAC-signed
  **PatchGram** (reusing :func:`nyxara.growth.self_repair.pack_bundle` for the tamper-evident envelope and
  the same HMAC-SHA256 voice as :mod:`nyxara.growth.epistemic_crypto`).
* **Receive.** Verify the signature *and* the bundle checksum, then **re-run the receiving node's own
  verify-or-rollback gauntlet** before applying. Consensus here means *independent local verification*, not
  deference to a peer — a valid signature only proves origin, never that the patch is safe on this node.

Honest boundary (enforced, not just documented): NYXARA does not currently run as a live multi-node
cluster. With no ``peers`` sink wired this is a **verified no-op** that still produces the signed PatchGram
for audit. The transport is a pluggable callable; the security semantics (sign → verify → local-gauntlet)
are the same whether the sink is a loopback list or a real network. Pure standard library; no LLM.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from nyxara.growth.self_repair import pack_bundle, unpack_bundle, verify_bundle

__all__ = ["PatchGram", "PatchGossip"]

PATCHGRAM_KIND = "nyxara-patchgram/1"


def _sign(key: bytes, payload: str) -> str:
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


@dataclass
class PatchGram:
    """A healed patch offered to peers: the payload, a portable signed bundle, and its signature."""

    node_id: str
    module: str
    diff: str
    certificate: Dict[str, Any]
    gauntlet: Dict[str, Any]
    created: float = field(default_factory=time.time)
    bundle: str = ""            # integrity-sealed JSON (self_repair.pack_bundle)
    signature: str = ""         # HMAC-SHA256 over the bundle

    def payload(self) -> Dict[str, Any]:
        return {"kind": PATCHGRAM_KIND, "node_id": self.node_id, "module": self.module,
                "diff": self.diff, "certificate": self.certificate,
                "gauntlet": self.gauntlet, "created": self.created}

    def to_dict(self) -> Dict[str, Any]:
        return {"node_id": self.node_id, "module": self.module,
                "signed": bool(self.signature), "created": self.created}


class PatchGossip:
    """Emit signed PatchGrams and verify received ones through the local gauntlet before applying."""

    def __init__(self, *, node_id: str, key: bytes,
                 peers: Optional[Callable[[PatchGram], None]] = None) -> None:
        self.node_id = node_id
        self._key = key
        # transport sink; None ⇒ single-process reality: broadcasting is a verified no-op
        self._peers = peers

    # ---- emit ---- #
    def make(self, *, module: str, diff: str,
             certificate: Dict[str, Any], gauntlet: Dict[str, Any]) -> PatchGram:
        gram = PatchGram(node_id=self.node_id, module=module, diff=diff,
                         certificate=dict(certificate), gauntlet=dict(gauntlet))
        gram.bundle = pack_bundle(gram.payload())
        gram.signature = _sign(self._key, gram.bundle)
        return gram

    def broadcast(self, gram: PatchGram) -> bool:
        """Offer a PatchGram to peers. Returns True if a transport actually accepted it, else False
        (the honest no-op when NYXARA runs single-process with no peers wired)."""
        if self._peers is None:
            return False
        try:
            self._peers(gram)
            return True
        except Exception:  # noqa: BLE001 — a failed transport is never fatal to the healing node
            return False

    # ---- receive ---- #
    def verify(self, gram: PatchGram) -> bool:
        """Signature valid AND bundle checksum intact AND payload matches the bundle. Any miss ⇒ False."""
        if not gram.bundle or not gram.signature:
            return False
        expected = _sign(self._key, gram.bundle)
        if not hmac.compare_digest(expected, gram.signature):
            return False
        if not verify_bundle(gram.bundle):
            return False
        payload = unpack_bundle(gram.bundle)
        return payload is not None and payload == gram.payload()

    def receive(self, gram: PatchGram,
                local_gauntlet: Callable[[PatchGram], bool]) -> Dict[str, Any]:
        """Verify origin/integrity, then re-run the receiving node's OWN gauntlet before applying.

        Returns an auditable outcome. A patch is applied only when it both verifies cryptographically and
        passes the *local* gauntlet — a signed patch is never trusted on the strength of its signature."""
        if not self.verify(gram):
            return {"applied": False, "verified": False,
                    "reason": "signature/bundle invalid — quarantined"}
        try:
            ok = bool(local_gauntlet(gram))
        except Exception:  # noqa: BLE001 — a throwing gauntlet is a failed verification, never a crash
            ok = False
        if not ok:
            return {"applied": False, "verified": True,
                    "reason": "failed local verify-or-rollback gauntlet — not applied"}
        return {"applied": True, "verified": True, "module": gram.module,
                "reason": "verified + locally re-gauntleted — applied"}
