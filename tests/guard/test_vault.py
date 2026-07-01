"""Tests for nyxara.guard.vault — the sovereign Credential Vault."""

from __future__ import annotations

import json

import pytest

from nyxara.agency.permissions import Authority
from nyxara.guard.crypto import CRYPTO_AVAILABLE
from nyxara.guard.vault import CredentialKind, CredentialVault
from nyxara.kernel.errors import AuthorizationError, SecurityError, ValidationError

pytestmark = pytest.mark.skipif(not CRYPTO_AVAILABLE,
                                reason="cryptography backend not installed")

OWNER = Authority.OWNER


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _vault(tmp_path, clock=None, **kw):
    return CredentialVault.from_passphrase(
        "test-master", path=tmp_path / "vault.json",
        audit_path=tmp_path / "audit.json", clock=clock or FakeClock(), **kw)


def test_put_and_reveal_round_trip(tmp_path):
    v = _vault(tmp_path)
    v.put("anthropic", CredentialKind.API_KEY, "sk-ant-XYZ", authority=OWNER)
    assert v.reveal("anthropic", authority=OWNER) == "sk-ant-XYZ"
    assert "anthropic" in v and len(v) == 1


def test_list_and_metadata_are_redacted(tmp_path):
    v = _vault(tmp_path)
    v.put("k", CredentialKind.API_KEY, "sk-ant-TOPSECRET", authority=OWNER, scope="api")
    rows = v.list(authority=Authority.AUTONOMOUS)
    assert rows[0]["secret"] == "***"
    assert "TOPSECRET" not in json.dumps(rows)
    assert "TOPSECRET" not in json.dumps(v.metadata("k"))


def test_owner_gate_blocks_autonomous_mutation_and_reveal(tmp_path):
    v = _vault(tmp_path)
    v.put("k", CredentialKind.PASSWORD, "pw", authority=OWNER)
    with pytest.raises(AuthorizationError):
        v.put("x", CredentialKind.PASSWORD, "pw", authority=Authority.AUTONOMOUS)
    with pytest.raises(AuthorizationError):
        v.reveal("k", authority=Authority.AUTONOMOUS)
    with pytest.raises(AuthorizationError):
        v.revoke("k", authority=Authority.DELEGATED)


def test_use_applies_secret_without_returning_it(tmp_path):
    v = _vault(tmp_path)
    v.put("k", CredentialKind.API_KEY, "sk-secret", authority=OWNER)
    captured = {}
    result = v.use("k", lambda s: captured.setdefault("s", s) and "ok",
                   authority=Authority.DELEGATED)
    assert captured["s"] == "sk-secret" and result == "ok"
    # autonomous cannot use
    with pytest.raises(AuthorizationError):
        v.use("k", lambda s: s, authority=Authority.AUTONOMOUS)


def test_rotate_reencrypts_but_preserves_value(tmp_path):
    v = _vault(tmp_path)
    v.put("k", CredentialKind.API_KEY, "v1", authority=OWNER)
    before = dict(v._records["k"].envelope)  # noqa: SLF001
    v.rotate("k", "v2", authority=OWNER)
    assert v._records["k"].envelope != before  # noqa: SLF001
    assert v.reveal("k", authority=OWNER) == "v2"
    # re-seal in place keeps the value, changes ciphertext
    ct = v._records["k"].envelope["ct"]  # noqa: SLF001
    v.rotate("k", authority=OWNER)
    assert v._records["k"].envelope["ct"] != ct  # noqa: SLF001
    assert v.reveal("k", authority=OWNER) == "v2"


def test_rotate_key_reencrypts_whole_store(tmp_path):
    v = _vault(tmp_path)
    v.put("a", CredentialKind.API_KEY, "aa", authority=OWNER)
    v.put("b", CredentialKind.PASSWORD, "bb", authority=OWNER)
    n = v.rotate_key(v._box._master.bumped(), authority=OWNER)  # noqa: SLF001
    assert n == 2
    assert v.reveal("a", authority=OWNER) == "aa"
    assert v.reveal("b", authority=OWNER) == "bb"


def test_revoke_removes_credential(tmp_path):
    v = _vault(tmp_path)
    v.put("k", CredentialKind.PASSWORD, "pw", authority=OWNER)
    assert v.revoke("k", authority=OWNER) is True
    assert "k" not in v
    assert v.revoke("k", authority=OWNER) is False


def test_mint_ssh_stores_private_returns_public(tmp_path):
    v = _vault(tmp_path)
    info = v.mint_ssh("deploy", authority=OWNER, comment="nyxara@deploy")
    assert info["public_key"].startswith("ssh-ed25519 ")
    assert info["fingerprint"].startswith("SHA256:")
    priv = v.reveal("deploy", authority=OWNER)
    assert priv.startswith("-----BEGIN OPENSSH PRIVATE KEY-----")


def test_audit_chain_verifies_and_detects_tamper(tmp_path):
    v = _vault(tmp_path)
    v.put("k", CredentialKind.API_KEY, "x", authority=OWNER)
    v.list(authority=Authority.AUTONOMOUS)
    assert v.verify_audit() is True
    v._audit[0]["rec"]["op"] = "forged"  # noqa: SLF001
    with pytest.raises(SecurityError):
        v.verify_audit()


def test_denied_access_is_audited(tmp_path):
    v = _vault(tmp_path)
    with pytest.raises(AuthorizationError):
        v.put("x", CredentialKind.PASSWORD, "p", authority=Authority.AUTONOMOUS)
    ops = [e["rec"] for e in v.audit_log()]
    assert any(o["op"] == "put" and o["ok"] is False for o in ops)


def test_persistence_reload(tmp_path):
    clk = FakeClock()
    v = _vault(tmp_path, clock=clk)
    v.put("k", CredentialKind.API_KEY, "sk-persist", authority=OWNER)
    reopened = CredentialVault.from_passphrase(
        "test-master", path=tmp_path / "vault.json",
        audit_path=tmp_path / "audit.json", clock=clk)
    assert "k" in reopened
    assert reopened.reveal("k", authority=OWNER) == "sk-persist"


def test_vault_object_is_truthy_when_empty(tmp_path):
    v = _vault(tmp_path)
    assert bool(v) is True and len(v) == 0


def test_missing_credential_raises(tmp_path):
    v = _vault(tmp_path)
    with pytest.raises(ValidationError):
        v.reveal("nope", authority=OWNER)


def test_oauth_refresh_uses_injected_http(tmp_path):
    calls = {}

    def fake_http(url, **kw):
        calls["url"] = url
        calls["body"] = kw.get("body", "")
        return {"ok": True, "status": 200,
                "body": json.dumps({"access_token": "new-access", "expires_in": 3600})}

    clk = FakeClock()
    v = _vault(tmp_path, clock=clk, http=fake_http)
    v.put("svc", CredentialKind.OAUTH_TOKEN, "old-access", authority=OWNER,
          oauth_token_endpoint="https://auth.example.com/token",
          oauth_client_id="client-1", oauth_refresh_token="refresh-abc")
    meta = v.refresh_oauth("svc", authority=Authority.DELEGATED)
    assert calls["url"] == "https://auth.example.com/token"
    assert "refresh-abc" in calls["body"] and "grant_type=refresh_token" in calls["body"]
    assert v.reveal("svc", authority=OWNER) == "new-access"
    assert meta["expires_at"] == clk.t + 3600


def test_expired_oauth_auto_refreshes_on_use(tmp_path):
    def fake_http(url, **kw):
        return {"ok": True, "status": 200,
                "body": json.dumps({"access_token": "fresh", "expires_in": 3600})}

    clk = FakeClock(t=1000.0)
    v = _vault(tmp_path, clock=clk, http=fake_http)
    v.put("svc", CredentialKind.OAUTH_TOKEN, "stale", authority=OWNER,
          expires_at=500.0,  # already expired at t=1000
          oauth_token_endpoint="https://auth.example.com/token",
          oauth_refresh_token="r")
    seen = {}
    v.use("svc", lambda s: seen.setdefault("s", s), authority=Authority.DELEGATED)
    assert seen["s"] == "fresh"
