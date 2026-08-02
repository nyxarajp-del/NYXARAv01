"""Tests for F14 — epistemic self-repair + substrate portability (growth/self_repair.py).

She must round-trip state portably, DETECT bit-rot via checksum (never trust corrupt data), restore
from the last verified backup, honestly report total loss, and expose NO self-replication. No LLM.
"""

from __future__ import annotations

import json

import nyxara.growth.self_repair as sr_mod
from nyxara.growth.self_repair import (
    BUNDLE_FORMAT,
    SelfRepair,
    pack_bundle,
    unpack_bundle,
    verify_bundle,
)


def test_bundle_roundtrips_and_is_portable():
    state = {"library": {"abs": ["a", "b"]}, "n": 3}
    blob = pack_bundle(state)
    b = json.loads(blob)
    assert b["format"] == BUNDLE_FORMAT and b["arch"] == "any"     # arch-agnostic, stdlib JSON
    assert unpack_bundle(blob) == state


def test_checksum_detects_tampering():
    blob = pack_bundle({"x": 1})
    b = json.loads(blob)
    b["payload"]["x"] = 999                                        # tamper without fixing the digest
    tampered = json.dumps(b)
    assert not verify_bundle(tampered)
    assert unpack_bundle(tampered) is None                        # never returns corrupted data


def test_heals_from_last_verified_backup(tmp_path):
    p = str(tmp_path / "state.json")
    sr = SelfRepair(p, keep=3)
    sr.checkpoint({"v": 1})
    sr.checkpoint({"v": 2})                                        # v1 → backup, v2 primary
    # corrupt the primary
    with open(p, "w", encoding="utf-8") as f:
        f.write("{ this is not valid json")
    assert not sr.verify_current()
    out = sr.heal()
    assert out.status == "healed"
    assert out.state == {"v": 1}                                   # recovered the last good state
    assert sr.verify_current()                                     # primary rewritten from backup


def test_reports_total_loss_honestly(tmp_path):
    p = str(tmp_path / "state.json")
    sr = SelfRepair(p, keep=2)
    sr.checkpoint({"v": 1})
    # corrupt everything that exists
    for path in [p] + sr._backups():
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("garbage")
        except FileNotFoundError:
            pass
    out = sr.heal()
    assert out.status == "lost" and out.state is None             # no bluffing a recovery


def test_no_self_replication_capability_is_exposed():
    names = " ".join(dir(sr_mod)).lower()
    assert "replicat" not in names and "spawn" not in names and "worm" not in names
