"""NYXARA · guard/crypto.py — the cryptographic core (🔐, Rules 5-8).

The dangling promise made in :mod:`nyxara.kernel.config` (``GuardConfig.key_rotation_hours``
carries a ``# crypto.py`` comment) is fulfilled here. This module is the deterministic,
verifiable primitive on which the credential vault stands — **NYXARA's own cryptography, not
the LLM's**. It gives the rest of the guard tier three things and nothing else:

* **Authenticated envelope encryption** (:class:`SecretBox`) — AES-256-GCM with a per-record
  random 96-bit nonce and *additional authenticated data* (AAD) that binds each ciphertext to
  the exact record (name + kind + key version) it belongs to, so a sealed secret cannot be
  silently transplanted onto another record. Decryption verifies the GCM tag: any tamper — one
  flipped bit of ciphertext, a swapped nonce, a moved envelope — fails the open, never returns
  garbage.
* **A Master key** (:class:`MasterKey`) — 32 bytes, either derived from the Master's passphrase
  (PBKDF2-HMAC-SHA256, same construction as :class:`nyxara.guard.auth.AuthGuard`) or handed the
  live auth root key so the vault and the identity gate share one sovereign secret. The key is
  never serialised, never logged, never printed (``__repr__`` masks it).
* **Key rotation** (:func:`rotate_envelope`) and **SSH keypair minting**
  (:func:`mint_ssh_ed25519`) so NYXARA can re-seal her store under a fresh key and mint her own
  Ed25519 identities, not merely hold the Master's.

**Fail-closed by construction.** The real cryptography is provided by the ``cryptography``
package (a declared dependency). If it is somehow unavailable, sealing *raises* — this module
will never write a secret in the clear to satisfy a caller. Verifiable beats probabilistic.

Depends only on :mod:`nyxara.kernel.errors`.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from nyxara.kernel.errors import SecurityError

__all__ = [
    "CRYPTO_AVAILABLE",
    "MasterKey",
    "SecretBox",
    "rotate_envelope",
    "mint_ssh_ed25519",
    "ssh_fingerprint",
    "random_token",
]

# --------------------------------------------------------------------------- #
# Import-guarded real cryptography. We NEVER fall back to a home-rolled cipher —
# absence means fail-closed (see SecretBox.seal), never plaintext-at-rest.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised by whichever environment lacks the dep
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    CRYPTO_AVAILABLE = True
except Exception:  # noqa: BLE001 — any import failure degrades to fail-closed
    serialization = None  # type: ignore[assignment]
    Ed25519PrivateKey = None  # type: ignore[assignment]
    AESGCM = None  # type: ignore[assignment]
    CRYPTO_AVAILABLE = False


_ENVELOPE_VERSION = 1
_ALG = "aes-256-gcm"
_AAD_PREFIX = "nyxara.vault.v1"
_PBKDF2_ITERATIONS = 200_000


def _require_crypto(op: str) -> None:
    """Fail closed: refuse any cryptographic operation if the primitive is missing."""
    if not CRYPTO_AVAILABLE:
        raise SecurityError(
            f"cryptography backend unavailable — refusing to {op} (fail-closed; "
            "install the 'cryptography' package)",
            context={"available": False},
        )


# --------------------------------------------------------------------------- #
# Master key
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MasterKey:
    """A 32-byte symmetric Master key with a monotonic version tag.

    The version is written into every envelope so rotation is unambiguous: an envelope sealed
    under key vN can only be opened by the MasterKey of the same version, and
    :func:`rotate_envelope` re-wraps it under vN+1.
    """

    key: bytes
    version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.key, (bytes, bytearray)) or len(self.key) != 32:
            raise SecurityError("MasterKey requires exactly 32 bytes of key material")
        if self.version < 1:
            raise SecurityError("MasterKey version must be >= 1")

    @classmethod
    def from_passphrase(
        cls, passphrase: str, *, salt: bytes | str, version: int = 1,
        iterations: int = _PBKDF2_ITERATIONS,
    ) -> "MasterKey":
        """Derive a key from the Master's passphrase (PBKDF2-HMAC-SHA256).

        Mirrors :meth:`nyxara.guard.auth.AuthGuard.from_passphrase`; pass
        ``salt=owner.continuity_namespace`` to bind the two to the same Master identity.
        """
        salt_b = salt.encode("utf-8") if isinstance(salt, str) else bytes(salt)
        raw = hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), salt_b, iterations)
        return cls(key=raw, version=version)

    @classmethod
    def from_bytes(cls, raw: bytes, *, version: int = 1) -> "MasterKey":
        """Adopt an existing 32-byte key (e.g. the live ``AuthGuard`` root key)."""
        return cls(key=bytes(raw), version=version)

    def bumped(self, new_key: Optional[bytes] = None) -> "MasterKey":
        """Return the next key version. Fresh random material unless one is supplied."""
        return MasterKey(key=bytes(new_key) if new_key is not None else os.urandom(32),
                         version=self.version + 1)

    def __repr__(self) -> str:  # never leak key material
        return f"MasterKey(v{self.version}, ***)"


# --------------------------------------------------------------------------- #
# Authenticated envelope encryption
# --------------------------------------------------------------------------- #
class SecretBox:
    """Seal/open secrets under a :class:`MasterKey` with AES-256-GCM + bound AAD."""

    def __init__(self, master: MasterKey) -> None:
        self._master = master

    @property
    def version(self) -> int:
        return self._master.version

    @staticmethod
    def available() -> bool:
        return CRYPTO_AVAILABLE

    def _aad(self, aad: str) -> bytes:
        # Bind ciphertext to record identity AND key version — no cross-record transplant.
        return f"{_AAD_PREFIX}|kv={self._master.version}|{aad}".encode("utf-8")

    def seal(self, plaintext: str | bytes, *, aad: str = "") -> Dict[str, Any]:
        """Encrypt ``plaintext`` into a JSON-serialisable envelope. Fail-closed."""
        _require_crypto("seal a secret")
        pt = plaintext.encode("utf-8") if isinstance(plaintext, str) else bytes(plaintext)
        nonce = os.urandom(12)  # 96-bit GCM nonce, unique per seal
        ct = AESGCM(self._master.key).encrypt(nonce, pt, self._aad(aad))
        return {
            "v": _ENVELOPE_VERSION,
            "alg": _ALG,
            "kv": self._master.version,
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ct": base64.b64encode(ct).decode("ascii"),
        }

    def open(self, envelope: Dict[str, Any], *, aad: str = "") -> bytes:
        """Decrypt an envelope; raise on any tamper / wrong key / wrong record."""
        _require_crypto("open a secret")
        if envelope.get("v") != _ENVELOPE_VERSION or envelope.get("alg") != _ALG:
            raise SecurityError("unsupported secret envelope", context={"env": envelope.get("alg")})
        if envelope.get("kv") != self._master.version:
            raise SecurityError(
                "secret was sealed under a different key version — rotate first",
                context={"envelope_kv": envelope.get("kv"), "key_kv": self._master.version},
            )
        try:
            nonce = base64.b64decode(envelope["nonce"])
            ct = base64.b64decode(envelope["ct"])
        except Exception as exc:  # noqa: BLE001
            raise SecurityError("malformed secret envelope") from exc
        try:
            return AESGCM(self._master.key).decrypt(nonce, ct, self._aad(aad))
        except Exception as exc:  # noqa: BLE001 — InvalidTag et al.
            raise SecurityError(
                "secret decryption failed — envelope tampered or wrong key (fail-closed)",
            ) from exc

    def open_str(self, envelope: Dict[str, Any], *, aad: str = "") -> str:
        return self.open(envelope, aad=aad).decode("utf-8")


def rotate_envelope(
    envelope: Dict[str, Any], *, old: SecretBox, new: SecretBox, aad: str = "",
) -> Dict[str, Any]:
    """Re-seal an envelope from the ``old`` key version to the ``new`` one (same AAD)."""
    return new.seal(old.open(envelope, aad=aad), aad=aad)


# --------------------------------------------------------------------------- #
# SSH keypair minting (NYXARA's own identities)
# --------------------------------------------------------------------------- #
def mint_ssh_ed25519(comment: str = "nyxara") -> Tuple[str, str]:
    """Generate a real Ed25519 keypair. Returns ``(openssh_private_pem, openssh_public)``."""
    _require_crypto("mint an SSH keypair")
    priv = Ed25519PrivateKey.generate()
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.OpenSSH,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")
    pub_line = priv.public_key().public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH,
    ).decode("ascii")
    if comment:
        pub_line = f"{pub_line} {comment}"
    return priv_pem, pub_line


def ssh_fingerprint(public_openssh: str) -> str:
    """SHA-256 fingerprint of an OpenSSH public key line (``SHA256:...``), as ssh-keygen prints."""
    parts = public_openssh.split()
    if len(parts) < 2:
        raise SecurityError("not an OpenSSH public key line")
    try:
        blob = base64.b64decode(parts[1])
    except Exception as exc:  # noqa: BLE001
        raise SecurityError("malformed OpenSSH public key") from exc
    digest = hashlib.sha256(blob).digest()
    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
def random_token(nbytes: int = 32) -> str:
    """A URL-safe random token (e.g. a freshly-minted API secret)."""
    return base64.urlsafe_b64encode(os.urandom(max(1, nbytes))).decode("ascii").rstrip("=")


def constant_time_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA guard/crypto self-test")
    print("=" * 70)
    print(f"cryptography available: {CRYPTO_AVAILABLE}")

    mk = MasterKey.from_passphrase("correct horse battery staple", salt="nyxara.owner.jp")
    box = SecretBox(mk)

    env = box.seal("sk-ant-SUPER-SECRET", aad="anthropic|api_key")
    assert "SUPER" not in str(env), "plaintext leaked into envelope!"
    assert box.open_str(env, aad="anthropic|api_key") == "sk-ant-SUPER-SECRET"
    print("seal/open round-trip : ok ✓")

    # wrong AAD (transplant onto another record) is refused
    try:
        box.open(env, aad="openai|api_key")
        raise AssertionError("transplant across records must fail")
    except SecurityError:
        print("record-binding (AAD): transplant refused ✓")

    # a single flipped bit of ciphertext fails the GCM tag
    tampered = dict(env)
    raw = bytearray(base64.b64decode(tampered["ct"]))
    raw[0] ^= 0x01
    tampered["ct"] = base64.b64encode(bytes(raw)).decode()
    try:
        box.open(tampered, aad="anthropic|api_key")
        raise AssertionError("tampered ciphertext must fail")
    except SecurityError:
        print("tamper detection    : flipped bit refused ✓")

    # rotation re-seals under a new key version; old key can no longer open the new envelope
    mk2 = mk.bumped()
    box2 = SecretBox(mk2)
    rot = rotate_envelope(env, old=box, new=box2, aad="anthropic|api_key")
    assert rot["kv"] == 2 and rot["ct"] != env["ct"]
    assert box2.open_str(rot, aad="anthropic|api_key") == "sk-ant-SUPER-SECRET"
    print("key rotation        : re-sealed under v2, plaintext preserved ✓")

    # SSH keypair minting
    priv, pub = mint_ssh_ed25519("nyxara@sovereign")
    assert priv.startswith("-----BEGIN OPENSSH PRIVATE KEY-----")
    assert pub.startswith("ssh-ed25519 ")
    print(f"ssh keygen          : {ssh_fingerprint(pub)} ✓")

    print("\nALL SELF-TESTS PASSED ✓")
