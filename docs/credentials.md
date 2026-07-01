# NYXARA Credential Vault

Passwords, API keys, SSH keys, and OAuth tokens live under NYXARA's **own** encrypted,
owner-gated, tamper-evident control — deterministic kernel code, not the LLM. This is the
`guard/` credential tier: `guard/crypto.py` (the cryptographic core) and `guard/vault.py`
(the store).

## Guarantees (under the Eight Sovereign Rules)

- **Encrypted at rest.** Every secret is sealed with AES-256-GCM before it touches disk. The
  vault file holds ciphertext + *redacted metadata only*. Fail-closed: with no crypto backend,
  nothing is stored in the clear.
- **Owner-gated (Rules 1 & 7).** Storing, revealing, rotating, revoking, exporting and minting
  are the Master's alone (`Authority.OWNER`). NYXARA's autonomous self may *list* redacted
  metadata; under a standing owner blessing (`Authority.DELEGATED`) she may *use* a credential —
  but the plaintext is handed only to an in-kernel callback and is **never returned** to the
  caller or the model. There is no "reveal" tool: plaintext reveal is Master-direct, Python-only.
- **Transparent & tamper-evident (Rule 6).** Every operation is appended to a SHA-256
  hash-chained audit log (never the secret). `verify_audit()` detects any edit.
- **Zero-trust & defensive (Rules 3 & 5).** A denied access is logged and raised as a
  `ThreatEvent` the Guardian can escalate. Secrets rotate and revoke on command.

## The Master key

Precedence (strongest first):

1. `NYXARA_MASTER_PASSPHRASE` — derived via PBKDF2-HMAC-SHA256, **never written to disk**.
2. A `<NYXARA_HOME>/keys/master.key` (32 random bytes, mode `0600`) auto-created once so the
   vault works out of the box and survives restarts.

Files default to `<NYXARA_HOME>/keys/vault.json` and `<NYXARA_HOME>/audit/vault_audit.json`
(both `0600`); override with `NYXARA_VAULT__PATH` / `NYXARA_VAULT__AUDIT_PATH`.

## Governed tools (the only way the mind touches secrets)

Registered on the same gated `ToolRegistry` as every other action:

| tool | authority | returns |
|------|-----------|---------|
| `credential_store` | Master (escalates) | redacted metadata |
| `credential_list` | autonomous | redacted metadata only |
| `credential_rotate` | Master (escalates) | redacted metadata |
| `credential_revoke` | Master (escalates) | `{revoked: bool}` |
| `ssh_keygen` | Master (escalates) | public key + fingerprint |
| `credential_request` | delegated | HTTP response (no secret) |

`credential_request` makes an authenticated HTTP call by injecting a stored secret into a
header **inside the kernel** — the secret never appears in the tool result or the LLM context.

## Using stored credentials

- **SSH.** A `remote_hosts` entry with `credential_name` resolves its secret from the vault at
  call time; `ssh_login` / `ssh_exec` bind an SSH private key to a transient `0600` keyfile (or
  a password directly) inside `CredentialVault.use` and wipe it afterward.
- **Provider API keys.** When `NYXARA_VAULT__PROVIDER_KEY_FALLBACK=true`, `mind/llm.py` falls
  back to a vault record named `<provider>_api_key` (e.g. `anthropic_api_key`) when neither
  config nor env supplies a key. Config/env still win.

## Direct (Master, Python) API

```python
from nyxara.guard.vault import CredentialVault, CredentialKind
from nyxara.agency.permissions import Authority

v = CredentialVault.bootstrap()                                  # process vault
v.put("anthropic_api_key", CredentialKind.API_KEY, "sk-ant-…",   # Master only
      authority=Authority.OWNER)
v.list(authority=Authority.AUTONOMOUS)                           # redacted metadata
v.reveal("anthropic_api_key", authority=Authority.OWNER)         # plaintext, Master only
v.mint_ssh("deploy", authority=Authority.OWNER)                  # NYXARA mints her own key
v.rotate_key(v._box._master.bumped(), authority=Authority.OWNER) # re-encrypt everything
assert v.verify_audit()                                          # tamper-evident log
```
