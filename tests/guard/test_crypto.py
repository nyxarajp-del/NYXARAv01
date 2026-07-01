"""Tests for nyxara.guard.crypto."""

from __future__ import annotations

import base64

import pytest

from nyxara.guard.crypto import (
    CRYPTO_AVAILABLE,
    MasterKey,
    SecretBox,
    mint_ssh_ed25519,
    rotate_envelope,
    ssh_fingerprint,
)
from nyxara.kernel.errors import SecurityError

pytestmark = pytest.mark.skipif(not CRYPTO_AVAILABLE,
                                reason="cryptography backend not installed")


def _box(version: int = 1) -> SecretBox:
    return SecretBox(MasterKey.from_passphrase("pw", salt="nyxara.owner.jp", version=version))


def test_master_key_requires_32_bytes():
    with pytest.raises(SecurityError):
        MasterKey(key=b"too short")
    assert MasterKey.from_bytes(b"\x00" * 32).version == 1


def test_seal_open_round_trip_and_no_plaintext_leak():
    box = _box()
    env = box.seal("sk-ant-SECRET", aad="anthropic|api_key")
    assert "SECRET" not in str(env)
    assert env["alg"] == "aes-256-gcm" and env["kv"] == 1
    assert box.open_str(env, aad="anthropic|api_key") == "sk-ant-SECRET"


def test_nonce_is_unique_per_seal():
    box = _box()
    a = box.seal("x", aad="r|k")
    b = box.seal("x", aad="r|k")
    assert a["nonce"] != b["nonce"] and a["ct"] != b["ct"]


def test_aad_binding_prevents_transplant():
    box = _box()
    env = box.seal("secret", aad="record-a|api_key")
    with pytest.raises(SecurityError):
        box.open(env, aad="record-b|api_key")


def test_tamper_fails_auth_tag():
    box = _box()
    env = box.seal("secret", aad="r|k")
    raw = bytearray(base64.b64decode(env["ct"]))
    raw[0] ^= 0x01
    env["ct"] = base64.b64encode(bytes(raw)).decode()
    with pytest.raises(SecurityError):
        box.open(env, aad="r|k")


def test_wrong_key_version_refused():
    env = _box(version=1).seal("secret", aad="r|k")
    with pytest.raises(SecurityError):
        _box(version=2).open(env, aad="r|k")


def test_rotation_reencrypts_and_preserves_plaintext():
    old = _box(version=1)
    new = SecretBox(old._master.bumped())  # noqa: SLF001 — intentional in test
    env = old.seal("value", aad="r|k")
    rot = rotate_envelope(env, old=old, new=new, aad="r|k")
    assert rot["kv"] == 2 and rot["ct"] != env["ct"]
    assert new.open_str(rot, aad="r|k") == "value"
    with pytest.raises(SecurityError):  # old key can no longer open the rotated envelope
        old.open(rot, aad="r|k")


def test_ssh_keygen_is_valid_ed25519():
    priv, pub = mint_ssh_ed25519("nyxara@test")
    assert priv.startswith("-----BEGIN OPENSSH PRIVATE KEY-----")
    assert pub.startswith("ssh-ed25519 ") and pub.endswith("nyxara@test")
    fp = ssh_fingerprint(pub)
    assert fp.startswith("SHA256:")
    # distinct keys each time
    _, pub2 = mint_ssh_ed25519()
    assert ssh_fingerprint(pub2) != fp


def test_master_key_repr_masks_material():
    assert "***" in repr(MasterKey.from_bytes(b"\x01" * 32))
