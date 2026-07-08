"""The constitution seal check is wired into the live runtime boot (fail-closed).

Previously the seal verification (that the sealed rules / values / invariants are byte-for-byte
intact) ran only in tests and the self-modification gauntlets, not when the live core starts.
These tests pin that ``NyxaraCore`` now verifies the constitution seals at boot and refuses to
start if a seal is broken.

Note: the boot hook runs only the pure-hash SEAL checks (thread-safe) — NOT the Z3 formal proof,
which is not thread-safe and would segfault on the per-core-construction hot path. The full
``boot_verify`` still runs in the gauntlet's single-threaded subprocess.
"""

from __future__ import annotations

import pytest

from nyxara.kernel.errors import InvariantViolation


def test_constitution_seals_pass_on_the_intact_tree():
    from nyxara import NyxaraCore
    # the method uses only module-level seals, so a bare instance exercises it fast
    bare = NyxaraCore.__new__(NyxaraCore)
    bare._verify_constitution_seals()          # must not raise on the untampered tree


def test_boot_aborts_on_a_tampered_invariant_seal(monkeypatch):
    # a genuine tamper: the pinned invariant seal no longer matches the live spec digest
    monkeypatch.setattr("nyxara.kernel.invariants.INVARIANTS_SEAL", "d" * 64)
    from nyxara import NyxaraCore
    with pytest.raises(InvariantViolation):
        NyxaraCore()                           # full boot must refuse to start


def test_verify_method_raises_when_rules_seal_is_tampered(monkeypatch):
    from nyxara import NyxaraCore

    def _tampered_verify_rules(*a, **k):
        raise InvariantViolation("sovereign-rules seal tampered (test)")

    monkeypatch.setattr("nyxara.kernel.rules.verify_rules", _tampered_verify_rules)
    bare = NyxaraCore.__new__(NyxaraCore)
    with pytest.raises(InvariantViolation):
        bare._verify_constitution_seals()
