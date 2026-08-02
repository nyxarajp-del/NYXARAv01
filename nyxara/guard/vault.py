"""NYXARA · guard/vault.py — the sovereign Credential Vault (🔐, Rules 1·3·5·6·7·8).

The Master's command: *"Passwords, API keys, SSH keys, OAuth tokens ko NYXARA ke direct
control me rakho — under the Master Rules."* This is that control, made real: a single
encrypted store that NYXARA — **not the LLM** — owns, gates, and audits.

What it guarantees:

* **Encrypted at rest.** Every secret is sealed by :class:`~nyxara.guard.crypto.SecretBox`
  (AES-256-GCM) before it ever touches disk. The vault file holds ciphertext and clear
  *metadata only*; the key lives in memory, anchored to the same Master key as the identity
  gate (:mod:`nyxara.guard.auth`). Fail-closed: with no crypto backend, nothing is stored.

* **Owner-gated (Rules 1 & 7).** Storing, revealing, rotating, revoking, exporting and
  minting are **the Master's alone** (``Authority.OWNER``). NYXARA's autonomous self may
  *list* redacted metadata and, under a standing owner blessing (``Authority.DELEGATED``),
  *use* a credential — but the plaintext is handed only to a deterministic in-kernel callback
  and is **never returned to the caller**, so it can never surface in the LLM's context. This
  is the literal "LLM proposes, kernel disposes" boundary for secrets.

* **Transparent & tamper-evident (Rule 6).** Every operation — who, what authority, which
  record, when, outcome — is appended to a SHA-256 hash-chained audit log (never the secret
  itself). :meth:`verify_audit` detects any after-the-fact edit.

* **Zero-trust (Rule 3) & defensive (Rule 5).** A denied access is not merely refused: it is
  logged and raised as a :class:`ThreatEvent` the :class:`~nyxara.guard.guardian.Guardian`
  can escalate. Secrets rotate and revoke on command.

Depends on :mod:`nyxara.guard.crypto`, :mod:`nyxara.agency.permissions` (Authority),
:mod:`nyxara.kernel.config` (owner + paths) and :mod:`nyxara.kernel.errors`.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.parse
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from nyxara.agency.permissions import Authority
from nyxara.guard.crypto import (
    CRYPTO_AVAILABLE,
    MasterKey,
    SecretBox,
    mint_ssh_ed25519,
    rotate_envelope,
    ssh_fingerprint,
)
from nyxara.kernel.config import OWNER, OwnerIdentity, get_settings
from nyxara.kernel.errors import AuthorizationError, SecurityError, ValidationError

__all__ = [
    "CredentialKind",
    "CredentialRecord",
    "CredentialVault",
    "get_process_vault",
    "resolve_api_key",
]

_GENESIS = "0" * 64
_REDACTED = "***"


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #
class CredentialKind(str, Enum):
    PASSWORD = "password"
    API_KEY = "api_key"
    SSH_KEY = "ssh_key"          # secret = OpenSSH private key; public_key kept in clear
    OAUTH_TOKEN = "oauth_token"  # secret = access token; refresh token sealed separately


# --------------------------------------------------------------------------- #
# Record
# --------------------------------------------------------------------------- #
@dataclass
class CredentialRecord:
    """One stored credential: clear metadata + a sealed secret envelope.

    The plaintext secret is *never* a field on this object — only its ciphertext
    ``envelope``. ``to_metadata`` is redaction-safe (Rule 6) and is what ``list`` and every
    log/report emits.
    """

    name: str
    kind: CredentialKind
    envelope: Dict[str, Any]                 # AES-GCM envelope of the secret (ciphertext)
    created_at: float
    rotated_at: float
    username: str = ""
    scope: str = ""                          # host / service / URL this credential is for
    tags: List[str] = field(default_factory=list)
    expires_at: Optional[float] = None
    last_accessed: float = 0.0
    access_count: int = 0
    # SSH-only: the public half is not a secret.
    public_key: str = ""
    fingerprint: str = ""
    # OAuth-only.
    oauth_token_endpoint: str = ""
    oauth_client_id: str = ""
    oauth_refresh_envelope: Optional[Dict[str, Any]] = None

    def expired(self, now: float) -> bool:
        return self.expires_at is not None and now >= self.expires_at

    def aad(self) -> str:
        """The additional-authenticated-data that binds the envelope to this record."""
        return f"{self.name}|{self.kind.value}"

    def refresh_aad(self) -> str:
        return f"{self.name}|{self.kind.value}|refresh"

    def to_metadata(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Redaction-safe view (Rule 6). Secret material never appears."""
        now = now if now is not None else time.time()
        return {
            "name": self.name,
            "kind": self.kind.value,
            "username": self.username,
            "scope": self.scope,
            "tags": list(self.tags),
            "secret": _REDACTED,
            "created_at": self.created_at,
            "rotated_at": self.rotated_at,
            "expires_at": self.expires_at,
            "expired": self.expired(now),
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "public_key": self.public_key,          # not secret
            "fingerprint": self.fingerprint,
            "has_refresh_token": self.oauth_refresh_envelope is not None,
        }

    def to_storage(self) -> Dict[str, Any]:
        """Full on-disk form (ciphertext included — safe, it is encrypted)."""
        d = asdict(self)
        d["kind"] = self.kind.value
        return d

    @classmethod
    def from_storage(cls, d: Dict[str, Any]) -> "CredentialRecord":
        d = dict(d)
        d["kind"] = CredentialKind(d["kind"])
        d["tags"] = list(d.get("tags", []))
        return cls(**d)


# --------------------------------------------------------------------------- #
# Vault
# --------------------------------------------------------------------------- #
class CredentialVault:
    """Encrypted, owner-gated, tamper-evident credential store.

    Parameters
    ----------
    box:      the :class:`SecretBox` (holds the Master key) that seals/opens secrets.
    path:     vault file (defaults to ``paths.keys_dir/vault.json``).
    audit_path: hash-chained audit log (defaults to ``paths.audit_dir/vault_audit.json``).
    guardian: optional :class:`~nyxara.guard.guardian.Guardian`; denied access is raised to it.
    http:     injectable HTTP callable for OAuth refresh (defaults to the guarded net layer).
    """

    def __init__(
        self,
        box: SecretBox,
        *,
        path: Optional[Path | str] = None,
        audit_path: Optional[Path | str] = None,
        owner: OwnerIdentity = OWNER,
        guardian: Any = None,
        http: Optional[Callable[..., Dict[str, Any]]] = None,
        clock: Callable[[], float] = time.time,
        autosave: bool = True,
    ) -> None:
        self._box = box
        self.owner = owner
        self._guardian = guardian
        self._http = http
        self._clock = clock
        self._autosave = autosave
        self._records: Dict[str, CredentialRecord] = {}

        self._path = Path(path) if path is not None else self._default_vault_path()
        self._audit_path = (
            Path(audit_path) if audit_path is not None else self._default_audit_path()
        )
        self._audit: List[Dict[str, Any]] = []
        self._audit_head = _GENESIS

        self._load()
        self._load_audit()

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_master_key(cls, key: bytes, *, version: int = 1, **kw: Any) -> "CredentialVault":
        """Build a vault from raw 32-byte Master key material (e.g. the AuthGuard root key)."""
        return cls(SecretBox(MasterKey.from_bytes(key, version=version)), **kw)

    @classmethod
    def from_passphrase(cls, passphrase: str, *, owner: OwnerIdentity = OWNER,
                        version: int = 1, **kw: Any) -> "CredentialVault":
        mk = MasterKey.from_passphrase(passphrase, salt=owner.continuity_namespace,
                                       version=version)
        return cls(SecretBox(mk), owner=owner, **kw)

    @classmethod
    def bootstrap(cls, *, owner: OwnerIdentity = OWNER, guardian: Any = None,
                  settings: Any = None, **kw: Any) -> "CredentialVault":
        """Build the process vault, resolving the Master key durably (survives restarts).

        Key precedence (strongest first):

        1. ``NYXARA_MASTER_PASSPHRASE`` — derived via PBKDF2, **never written to disk** (best).
        2. A persisted ``paths.keys_dir/master.key`` (32 random bytes, mode 0600) — created
           once so the vault works out of the box; document that a passphrase is stronger.

        Paths/overrides come from :class:`~nyxara.kernel.config.VaultConfig` when present.
        """
        if settings is None:
            try:
                settings = get_settings()
            except Exception:  # noqa: BLE001
                settings = None
        vcfg = getattr(settings, "vault", None)
        path = getattr(vcfg, "path", None) if vcfg else None
        audit_path = getattr(vcfg, "audit_path", None) if vcfg else None

        passphrase = os.getenv("NYXARA_MASTER_PASSPHRASE")
        if passphrase:
            mk = MasterKey.from_passphrase(passphrase, salt=owner.continuity_namespace)
        else:
            mk = MasterKey.from_bytes(cls._resolve_persisted_key(settings))
        return cls(SecretBox(mk), owner=owner, guardian=guardian,
                   path=path, audit_path=audit_path, **kw)

    @staticmethod
    def _resolve_persisted_key(settings: Any) -> bytes:
        """Load (or create once) a 32-byte machine key in keys_dir, locked to 0600."""
        try:
            keys_dir = Path(settings.paths.ensure().keys_dir)
        except Exception:  # noqa: BLE001
            keys_dir = Path.home() / ".nyxara" / "keys"
        keys_dir.mkdir(parents=True, exist_ok=True)
        key_file = keys_dir / "master.key"
        if key_file.exists():
            raw = key_file.read_bytes()
            if len(raw) == 32:
                return raw
        raw = os.urandom(32)
        tmp = key_file.with_suffix(".tmp")
        tmp.write_bytes(raw)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, key_file)
        try:
            os.chmod(key_file, 0o600)
        except OSError:
            pass
        return raw

    @staticmethod
    def _default_vault_path() -> Path:
        try:
            paths = get_settings().paths.ensure()
            return Path(paths.keys_dir) / "vault.json"
        except Exception:  # noqa: BLE001 — config optional in unit tests
            return Path.home() / ".nyxara" / "keys" / "vault.json"

    @staticmethod
    def _default_audit_path() -> Path:
        try:
            paths = get_settings().paths.ensure()
            return Path(paths.audit_dir) / "vault_audit.json"
        except Exception:  # noqa: BLE001
            return Path.home() / ".nyxara" / "audit" / "vault_audit.json"

    # ------------------------------------------------------------------ #
    # Authority gate (Rules 1, 3, 7)
    # ------------------------------------------------------------------ #
    def _gate(self, authority: Authority, minimum: Authority, op: str,
              name: str = "") -> None:
        """Fail-closed authority check. On denial: audit + raise a threat + refuse."""
        if authority.trust < minimum.trust:
            self._record_audit(op, name, authority, ok=False,
                               reason=f"requires {minimum.value}")
            self._raise_threat(op, name, authority, minimum)
            raise AuthorizationError(
                f"credential operation {op!r} requires {minimum.value} authority "
                f"(got {authority.value}) — refused",
                context={"op": op, "name": name, "authority": authority.value},
            )

    def _raise_threat(self, op: str, name: str, authority: Authority,
                      minimum: Authority) -> None:
        if self._guardian is None:
            return
        try:  # lazy import to avoid any import-order coupling
            from nyxara.guard.guardian import ThreatCategory, ThreatEvent
            cat = (ThreatCategory.IMPERSONATION if authority is Authority.UNTRUSTED
                   else ThreatCategory.DATA_EXFIL)
            self._guardian.handle(ThreatEvent(
                category=cat, source="credential_vault",
                description=f"denied {op} on {name!r}: {authority.value} < {minimum.value}",
                severity=0.7, confidence=0.9,
                data={"op": op, "name": name, "authority": authority.value},
            ))
        except Exception:  # noqa: BLE001 — the guardian is a capability, never a hard dep
            pass

    # ------------------------------------------------------------------ #
    # Mutating operations (OWNER only)
    # ------------------------------------------------------------------ #
    def put(
        self,
        name: str,
        kind: CredentialKind | str,
        secret: str,
        *,
        authority: Authority = Authority.AUTONOMOUS,
        username: str = "",
        scope: str = "",
        tags: Tuple[str, ...] | List[str] = (),
        expires_at: Optional[float] = None,
        public_key: str = "",
        oauth_token_endpoint: str = "",
        oauth_client_id: str = "",
        oauth_refresh_token: Optional[str] = None,
        overwrite: bool = True,
    ) -> Dict[str, Any]:
        """Store (seal) a secret under ``name``. The Master's authority alone (Rules 1 & 7)."""
        self._gate(authority, Authority.OWNER, "put", name)
        if not name or not isinstance(name, str):
            raise ValidationError("credential name must be a non-empty string")
        kind = CredentialKind(kind) if not isinstance(kind, CredentialKind) else kind
        if name in self._records and not overwrite:
            raise ValidationError(f"credential {name!r} already exists", context={"name": name})
        now = self._clock()
        rec = CredentialRecord(
            name=name, kind=kind, envelope={}, created_at=now, rotated_at=now,
            username=username, scope=scope, tags=list(tags), expires_at=expires_at,
            public_key=public_key, oauth_token_endpoint=oauth_token_endpoint,
            oauth_client_id=oauth_client_id,
        )
        rec.envelope = self._box.seal(secret, aad=rec.aad())
        if oauth_refresh_token is not None:
            rec.oauth_refresh_envelope = self._box.seal(oauth_refresh_token,
                                                        aad=rec.refresh_aad())
        self._records[name] = rec
        self._save()
        self._record_audit("put", name, authority, ok=True,
                           reason=f"kind={kind.value}")
        return rec.to_metadata(now)

    def mint_ssh(self, name: str, *, authority: Authority = Authority.AUTONOMOUS,
                 comment: str = "nyxara", scope: str = "",
                 tags: Tuple[str, ...] = ()) -> Dict[str, Any]:
        """Generate a fresh Ed25519 keypair and store it (Master only). Returns the public key."""
        self._gate(authority, Authority.OWNER, "mint_ssh", name)
        priv_pem, pub_line = mint_ssh_ed25519(comment)
        fp = ssh_fingerprint(pub_line)
        self.put(name, CredentialKind.SSH_KEY, priv_pem, authority=authority,
                 scope=scope, tags=tags, public_key=pub_line)
        self._records[name].fingerprint = fp
        self._save()
        self._record_audit("mint_ssh", name, authority, ok=True, reason=fp)
        return {"name": name, "public_key": pub_line, "fingerprint": fp}

    def rotate(self, name: str, new_secret: Optional[str] = None, *,
               authority: Authority = Authority.AUTONOMOUS) -> Dict[str, Any]:
        """Replace/re-seal a secret. If ``new_secret`` is None, re-seal the same value."""
        self._gate(authority, Authority.OWNER, "rotate", name)
        rec = self._must(name)
        now = self._clock()
        value = new_secret if new_secret is not None else self._box.open_str(
            rec.envelope, aad=rec.aad())
        rec.envelope = self._box.seal(value, aad=rec.aad())
        rec.rotated_at = now
        self._save()
        self._record_audit("rotate", name, authority, ok=True,
                           reason="new value" if new_secret is not None else "re-seal")
        return rec.to_metadata(now)

    def rotate_key(self, new_master: MasterKey, *,
                   authority: Authority = Authority.AUTONOMOUS) -> int:
        """Re-encrypt EVERY record under a new Master key version (Rule 4 defensive rotation)."""
        self._gate(authority, Authority.OWNER, "rotate_key", "*")
        new_box = SecretBox(new_master)
        for rec in self._records.values():
            rec.envelope = rotate_envelope(rec.envelope, old=self._box, new=new_box,
                                           aad=rec.aad())
            if rec.oauth_refresh_envelope is not None:
                rec.oauth_refresh_envelope = rotate_envelope(
                    rec.oauth_refresh_envelope, old=self._box, new=new_box,
                    aad=rec.refresh_aad())
        self._box = new_box
        self._save()
        self._record_audit("rotate_key", "*", authority, ok=True,
                           reason=f"kv->{new_master.version}")
        return len(self._records)

    def revoke(self, name: str, *, authority: Authority = Authority.AUTONOMOUS) -> bool:
        """Permanently remove a credential (Rule 5 defensive revocation)."""
        self._gate(authority, Authority.OWNER, "revoke", name)
        existed = self._records.pop(name, None) is not None
        if existed:
            self._save()
        self._record_audit("revoke", name, authority, ok=existed,
                           reason="removed" if existed else "absent")
        return existed

    def purge(self, *, authority: Authority = Authority.AUTONOMOUS) -> int:
        """Wipe the entire vault (Master only)."""
        self._gate(authority, Authority.OWNER, "purge", "*")
        n = len(self._records)
        self._records.clear()
        self._save()
        self._record_audit("purge", "*", authority, ok=True, reason=f"{n} removed")
        return n

    # ------------------------------------------------------------------ #
    # Reveal (OWNER only) — the one path that returns plaintext
    # ------------------------------------------------------------------ #
    def reveal(self, name: str, *, authority: Authority = Authority.AUTONOMOUS) -> str:
        """Decrypt and RETURN the plaintext secret. Master-direct only; never wired to a tool."""
        self._gate(authority, Authority.OWNER, "reveal", name)
        rec = self._must(name)
        value = self._decrypt(rec, refresh_if_expired=True, authority=authority)
        self._touch(rec)
        self._record_audit("reveal", name, authority, ok=True, reason="plaintext returned")
        return value

    def export(self, *, authority: Authority = Authority.AUTONOMOUS,
               include_secrets: bool = False) -> Dict[str, Any]:
        """Owner-only export. With ``include_secrets`` the plaintext is decrypted out."""
        self._gate(authority, Authority.OWNER, "export", "*")
        out: Dict[str, Any] = {"version": 1, "records": {}}
        for name, rec in self._records.items():
            meta = rec.to_metadata(self._clock())
            if include_secrets:
                meta["secret"] = self._box.open_str(rec.envelope, aad=rec.aad())
            out["records"][name] = meta
        self._record_audit("export", "*", authority, ok=True,
                           reason=f"secrets={include_secrets}")
        return out

    # ------------------------------------------------------------------ #
    # Use (DELEGATED) — apply a secret without ever returning it
    # ------------------------------------------------------------------ #
    def use(self, name: str, action: Callable[[str], Any], *,
            authority: Authority = Authority.AUTONOMOUS) -> Any:
        """Hand the plaintext to ``action`` and return only ``action``'s result.

        This is how NYXARA *uses* a credential (open an SSH session, sign a request) under a
        standing owner blessing (``DELEGATED``) while the secret stays inside the kernel and
        never returns to the caller / the LLM.
        """
        self._gate(authority, Authority.DELEGATED, "use", name)
        rec = self._must(name)
        value = self._decrypt(rec, refresh_if_expired=True, authority=authority)
        try:
            result = action(value)
        finally:
            self._touch(rec)
            self._record_audit("use", name, authority, ok=True,
                               reason=f"applied via {getattr(action, '__name__', 'callback')}")
        return result

    # ------------------------------------------------------------------ #
    # Read metadata (AUTONOMOUS) — redacted, no secret material
    # ------------------------------------------------------------------ #
    def list(self, *, authority: Authority = Authority.AUTONOMOUS,
             kind: Optional[CredentialKind | str] = None) -> List[Dict[str, Any]]:
        """Redacted metadata for every credential (Rule 6). Safe for autonomous / logs."""
        self._gate(authority, Authority.AUTONOMOUS, "list")
        now = self._clock()
        want = CredentialKind(kind) if (kind and not isinstance(kind, CredentialKind)) else kind
        rows = [r.to_metadata(now) for r in self._records.values()
                if want is None or r.kind is want]
        return sorted(rows, key=lambda r: r["name"])

    def metadata(self, name: str, *,
                 authority: Authority = Authority.AUTONOMOUS) -> Dict[str, Any]:
        self._gate(authority, Authority.AUTONOMOUS, "metadata", name)
        return self._must(name).to_metadata(self._clock())

    def names(self) -> List[str]:
        return sorted(self._records)

    def __contains__(self, name: str) -> bool:
        return name in self._records

    def __len__(self) -> int:
        return len(self._records)

    def __bool__(self) -> bool:
        # A vault object is always truthy: existence, not emptiness. Prevents the footgun of
        # `if core.vault:` reading as False for an empty (but present) vault via __len__.
        return True

    # ------------------------------------------------------------------ #
    # OAuth refresh (DELEGATED) — real token refresh over the guarded net layer
    # ------------------------------------------------------------------ #
    def refresh_oauth(self, name: str, *,
                      authority: Authority = Authority.AUTONOMOUS) -> Dict[str, Any]:
        """Exchange the sealed refresh token for a fresh access token; re-seal it."""
        self._gate(authority, Authority.DELEGATED, "refresh_oauth", name)
        rec = self._must(name)
        if rec.kind is not CredentialKind.OAUTH_TOKEN:
            raise ValidationError(f"{name!r} is not an OAuth token", context={"kind": rec.kind.value})
        if rec.oauth_refresh_envelope is None or not rec.oauth_token_endpoint:
            raise ValidationError(f"{name!r} has no refresh token / token endpoint")
        refresh_token = self._box.open_str(rec.oauth_refresh_envelope, aad=rec.refresh_aad())
        form = {"grant_type": "refresh_token", "refresh_token": refresh_token}
        if rec.oauth_client_id:
            form["client_id"] = rec.oauth_client_id
        resp = self._do_http(
            rec.oauth_token_endpoint, method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            body=urllib.parse.urlencode(form),
        )
        if not resp.get("ok"):
            self._record_audit("refresh_oauth", name, authority, ok=False,
                               reason=str(resp.get("error"))[:120])
            raise SecurityError(f"OAuth refresh failed for {name!r}",
                                context={"status": resp.get("status")})
        try:
            payload = json.loads(resp.get("body") or "{}")
        except Exception as exc:  # noqa: BLE001
            raise SecurityError("OAuth token endpoint returned non-JSON") from exc
        access = payload.get("access_token")
        if not access:
            raise SecurityError("OAuth refresh response missing access_token")
        now = self._clock()
        rec.envelope = self._box.seal(access, aad=rec.aad())
        rec.rotated_at = now
        if "expires_in" in payload:
            try:
                rec.expires_at = now + float(payload["expires_in"])
            except (TypeError, ValueError):
                pass
        if payload.get("refresh_token"):  # rotated refresh token
            rec.oauth_refresh_envelope = self._box.seal(payload["refresh_token"],
                                                        aad=rec.refresh_aad())
        self._save()
        self._record_audit("refresh_oauth", name, authority, ok=True, reason="access refreshed")
        return rec.to_metadata(now)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _must(self, name: str) -> CredentialRecord:
        rec = self._records.get(name)
        if rec is None:
            raise ValidationError(f"no such credential {name!r}",
                                  context={"known": self.names()})
        return rec

    def _decrypt(self, rec: CredentialRecord, *, refresh_if_expired: bool,
                 authority: Authority) -> str:
        if (refresh_if_expired and rec.kind is CredentialKind.OAUTH_TOKEN
                and rec.expired(self._clock()) and rec.oauth_refresh_envelope is not None):
            self.refresh_oauth(rec.name, authority=authority)
        return self._box.open_str(rec.envelope, aad=rec.aad())

    def _touch(self, rec: CredentialRecord) -> None:
        rec.last_accessed = self._clock()
        rec.access_count += 1
        self._save()

    def _do_http(self, url: str, **kw: Any) -> Dict[str, Any]:
        if self._http is not None:
            return self._http(url, **kw)
        from nyxara.agency.net_request import http_request
        return http_request(url, **kw)

    # ---- persistence (atomic, 0600) ---- #
    def _save(self) -> None:
        if not self._autosave:
            return
        self._write_json(self._path, {
            "version": 1,
            "kv": self._box.version,
            "records": {n: r.to_storage() for n, r in self._records.items()},
        })

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — a corrupt vault is not a silent success
            raise SecurityError("vault file is unreadable/corrupt (fail-closed)",
                                context={"path": str(self._path)})
        for name, raw in (data.get("records") or {}).items():
            self._records[name] = CredentialRecord.from_storage(raw)

    @staticmethod
    def _write_json(path: Path, obj: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex[:8]}")
        tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, path)  # atomic
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    # ---- tamper-evident audit (Rule 6) ---- #
    def _record_audit(self, op: str, name: str, authority: Authority, *,
                      ok: bool, reason: str = "") -> None:
        rec = {"ts": self._clock(), "op": op, "name": name,
               "authority": authority.value, "ok": ok, "reason": reason}
        body = json.dumps(rec, sort_keys=True, default=str)
        seq = len(self._audit)
        digest = hashlib.sha256(f"{self._audit_head}|{seq}|{body}".encode()).hexdigest()
        self._audit.append({"seq": seq, "prev": self._audit_head, "rec": rec,
                            "digest": digest})
        self._audit_head = digest
        self._persist_audit()

    def _persist_audit(self) -> None:
        if not self._autosave:
            return
        try:
            self._write_json(self._audit_path, {"chain": self._audit})
        except Exception:  # noqa: BLE001 — never let logging crash an operation
            pass

    def _load_audit(self) -> None:
        if not self._audit_path.exists():
            return
        try:
            data = json.loads(self._audit_path.read_text(encoding="utf-8"))
            self._audit = list(data.get("chain") or [])
            if self._audit:
                self._audit_head = self._audit[-1]["digest"]
        except Exception:  # noqa: BLE001
            self._audit, self._audit_head = [], _GENESIS

    def verify_audit(self) -> bool:
        """Return True iff the audit chain is intact; raise :class:`SecurityError` on tamper."""
        prev = _GENESIS
        for e in self._audit:
            if e.get("prev") != prev:
                raise SecurityError("vault audit chain broken", context={"seq": e.get("seq")})
            body = json.dumps(e["rec"], sort_keys=True, default=str)
            if hashlib.sha256(f"{prev}|{e['seq']}|{body}".encode()).hexdigest() != e["digest"]:
                raise SecurityError("vault audit log tampered", context={"seq": e.get("seq")})
            prev = e["digest"]
        return True

    def audit_log(self) -> List[Dict[str, Any]]:
        return [dict(e) for e in self._audit]

    def report(self) -> Dict[str, Any]:
        now = self._clock()
        kinds: Dict[str, int] = {}
        for r in self._records.values():
            kinds[r.kind.value] = kinds.get(r.kind.value, 0) + 1
        return {
            "records": len(self._records),
            "by_kind": kinds,
            "key_version": self._box.version,
            "crypto_available": CRYPTO_AVAILABLE,
            "audit_entries": len(self._audit),
            "expired": sum(1 for r in self._records.values() if r.expired(now)),
            "path": str(self._path),
        }


# --------------------------------------------------------------------------- #
# Process-wide vault + internal API-key resolution (kernel faculties only)
# --------------------------------------------------------------------------- #
_PROCESS_VAULT: Optional[CredentialVault] = None


def get_process_vault(**kw: Any) -> Optional[CredentialVault]:
    """Lazily bootstrap and cache one process-wide vault (kernel-internal use)."""
    global _PROCESS_VAULT
    if _PROCESS_VAULT is None:
        try:
            _PROCESS_VAULT = CredentialVault.bootstrap(**kw)
        except Exception:  # noqa: BLE001 — the vault is a capability, never a hard dep
            return None
    return _PROCESS_VAULT


def resolve_api_key(provider: str) -> Optional[str]:
    """Kernel-internal: fetch a service API key from the vault, or None.

    Trusted in-process faculties (e.g. web search) may call this *only* as a fallback
    when neither config nor env supplies a key. (The LLM itself is fully local —
    DistilGPT-2 — and needs no key.) It never runs unless
    ``VaultConfig.provider_key_fallback`` is on, and only reads pre-existing records — it does
    not create a vault where one is not warranted. The plaintext is used to construct the API
    client inside the kernel; it is never returned to the model.
    """
    try:
        settings = get_settings()
        if not getattr(settings.vault, "provider_key_fallback", False):
            return None
        vault_file = CredentialVault._default_vault_path()  # noqa: SLF001
        if getattr(settings.vault, "path", None):
            vault_file = Path(settings.vault.path)
        if not vault_file.exists():
            return None  # no store yet — don't bootstrap one just to miss
    except Exception:  # noqa: BLE001
        return None
    v = get_process_vault()
    if v is None:
        return None
    for name in (f"{provider}_api_key", provider):
        if name in v:
            try:
                return v.reveal(name, authority=Authority.OWNER)
            except Exception:  # noqa: BLE001
                return None
    return None


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import tempfile

    print("=" * 70)
    print("NYXARA Credential Vault self-test")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as d:
        vp = Path(d) / "vault.json"
        ap = Path(d) / "audit.json"
        v = CredentialVault.from_passphrase("correct horse battery staple",
                                            path=vp, audit_path=ap)
        OWNER_A = Authority.OWNER

        # store an API key (Master only)
        v.put("anthropic", CredentialKind.API_KEY, "sk-ant-SUPER-SECRET",
              authority=OWNER_A, scope="api.anthropic.com")
        print("put api_key         : ok ✓")

        # autonomous self may list — but only redacted metadata
        rows = v.list(authority=Authority.AUTONOMOUS)
        assert rows and rows[0]["secret"] == _REDACTED
        assert "SUPER" not in json.dumps(rows), "plaintext leaked into list()!"
        print("list (autonomous)   : redacted, no plaintext ✓")

        # autonomous self may NOT store or reveal
        for bad in ("put", "reveal"):
            try:
                if bad == "put":
                    v.put("x", CredentialKind.PASSWORD, "p", authority=Authority.AUTONOMOUS)
                else:
                    v.reveal("anthropic", authority=Authority.AUTONOMOUS)
                raise AssertionError(f"{bad} must be owner-gated")
            except AuthorizationError:
                pass
        print("owner gate          : autonomous put/reveal refused ✓")

        # DELEGATED self may USE the key — but never receives the plaintext
        seen: Dict[str, str] = {}
        masked = v.use("anthropic",
                       lambda secret: seen.setdefault("s", secret) and "HTTP 200",
                       authority=Authority.DELEGATED)
        assert seen["s"] == "sk-ant-SUPER-SECRET" and masked == "HTTP 200"
        print("use (delegated)     : secret applied, not returned ✓")

        # the Master may reveal
        assert v.reveal("anthropic", authority=OWNER_A) == "sk-ant-SUPER-SECRET"
        print("reveal (master)     : plaintext returned to Master ✓")

        # rotate the value, then rotate the whole key
        v.rotate("anthropic", "sk-ant-ROTATED", authority=OWNER_A)
        assert v.reveal("anthropic", authority=OWNER_A) == "sk-ant-ROTATED"
        v.rotate_key(v._box._master.bumped(), authority=OWNER_A)  # noqa: SLF001 (demo)
        assert v.reveal("anthropic", authority=OWNER_A) == "sk-ant-ROTATED"
        print("rotate + rekey      : plaintext preserved across rotation ✓")

        # mint an SSH key NYXARA owns
        info = v.mint_ssh("deploy", authority=OWNER_A, comment="nyxara@deploy")
        assert info["public_key"].startswith("ssh-ed25519 ")
        print(f"mint ssh            : {info['fingerprint']} ✓")

        # tamper-evidence: chain verifies, and an edit is detected
        assert v.verify_audit()
        v._audit[1]["rec"]["op"] = "forged"  # noqa: SLF001 (demo tamper)
        try:
            v.verify_audit()
            raise AssertionError("tampered audit must be detected")
        except SecurityError:
            print("audit chain         : tamper detected ✓")

        # persistence: a reopened vault sees the same records
        v._audit[1]["rec"]["op"] = "use"  # noqa: SLF001 — undo demo tamper before reload
        v._persist_audit()  # noqa: SLF001
        reopened = CredentialVault(v._box, path=vp, audit_path=ap)
        assert set(reopened.names()) == {"anthropic", "deploy"}
        print("persistence         : reload sees stored records ✓")

        print(f"\nreport              : {v.report()}")
        print("\nALL SELF-TESTS PASSED ✓")
