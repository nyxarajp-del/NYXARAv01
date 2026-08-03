"""NYXARA · growth/self_repair.py — epistemic self-repair + substrate portability (🛡️, Rule 7, F14).

A mind that lives across restarts must survive **bit-rot** and move between machines without losing
itself. F14 gives NYXARA two honest, bounded capabilities:

* **Self-healing (checkpoint / verify / restore).** Every checkpoint is an **integrity-sealed bundle**
  — its payload carries a SHA-256 over a canonical serialisation. On load she **verifies** it; if the
  current checkpoint is corrupt she **restores from the last verified backup** (a rolling set). This is
  detect-and-restore, **not** a magic live hot-patch — she never pretends a corrupted state is fine.

* **Substrate portability.** The bundle is pure-stdlib **JSON** (never pickle), platform-agnostic, so a
  learned library + proofs boot **zero-setup on x86 / ARM** — anywhere the pure-Python core runs.

The honest boundary, stated and pinned by a test: **no self-replication** (spawning copies onto other
hosts is worm-like and unsafe — deliberately not built), and **no "digital immortality"** claim. This
is verified checkpoint/restore + portable serialisation, nothing more. Pure standard library; **no
LLM**; it protects continuity, touching no gate or character value.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

__all__ = ["BUNDLE_FORMAT", "pack_bundle", "verify_bundle", "unpack_bundle",
           "HealOutcome", "SelfRepair"]

BUNDLE_FORMAT = "nyxara-bundle/1"


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def pack_bundle(state: Dict[str, Any]) -> str:
    """Serialise state into a portable, integrity-sealed JSON bundle (arch-agnostic, stdlib-only)."""
    bundle = {
        "format": BUNDLE_FORMAT,
        "arch": "any",              # pure JSON — boots zero-setup on x86 / ARM alike
        "created": time.time(),
        "sha256": _digest(state),
        "payload": state,
    }
    return json.dumps(bundle)


def verify_bundle(blob: str) -> bool:
    """True iff the blob parses, is the right format, and its checksum matches its payload."""
    try:
        b = json.loads(blob)
        return (b.get("format") == BUNDLE_FORMAT
                and b.get("sha256") == _digest(b.get("payload")))
    except Exception:  # noqa: BLE001 — any parse fault is a failed verification, never a crash
        return False


def unpack_bundle(blob: str) -> Optional[Dict[str, Any]]:
    """Return the payload only if the bundle verifies — a tampered bundle yields ``None``, never data."""
    if not verify_bundle(blob):
        return None
    return json.loads(blob)["payload"]


@dataclass
class HealOutcome:
    status: str                 # "ok" | "healed" | "lost"
    source: str = ""            # which file the good state came from
    state: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "source": os.path.basename(self.source) if self.source else "",
                "recovered": self.state is not None}


class SelfRepair:
    """Checksummed, self-healing checkpoints with a rolling set of verified backups."""

    def __init__(self, path: str, *, keep: int = 3) -> None:
        self.path = path
        self.keep = max(1, keep)

    def _backups(self) -> List[str]:
        return [f"{self.path}.bak{i}" for i in range(self.keep)]

    def _write(self, path: str, blob: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(blob)

    def _read(self, path: str) -> Optional[str]:
        try:
            with open(path, encoding="utf-8") as f:
                return f.read()
        except Exception:  # noqa: BLE001
            return None

    # ---- checkpoint (rotate the old, verified primary into backups) ---- #
    def checkpoint(self, state: Dict[str, Any]) -> str:
        cur = self._read(self.path)
        if cur is not None and verify_bundle(cur):
            baks = self._backups()
            for i in range(len(baks) - 1, 0, -1):        # shift bak(n-1)..bak1 → bak(n)..bak2
                prev = self._read(baks[i - 1])
                if prev is not None:
                    self._write(baks[i], prev)
            self._write(baks[0], cur)                    # newest backup = the last good primary
        blob = pack_bundle(state)
        self._write(self.path, blob)
        return self.path

    # ---- verify / restore / heal ---- #
    def verify_current(self) -> bool:
        blob = self._read(self.path)
        return blob is not None and verify_bundle(blob)

    def restore(self) -> HealOutcome:
        """Return the newest VERIFIED state — primary if intact, else walk back through the backups."""
        blob = self._read(self.path)
        if blob is not None and verify_bundle(blob):
            return HealOutcome("ok", self.path, unpack_bundle(blob))
        for bak in self._backups():
            b = self._read(bak)
            if b is not None and verify_bundle(b):
                return HealOutcome("healed", bak, unpack_bundle(b))
        return HealOutcome("lost")

    def heal(self) -> HealOutcome:
        """If the current checkpoint is corrupt, rewrite it from the last verified backup."""
        outcome = self.restore()
        if outcome.status == "healed" and outcome.state is not None:
            self._write(self.path, pack_bundle(outcome.state))   # restore primary from the good copy
        return outcome
