"""NYXARA · agency/credential_tools.py — governed tools over the Credential Vault (🔐).

These register NYXARA's credential operations onto the same gated
:class:`~nyxara.agency.tools.ToolRegistry` every other action passes through, so each call
clears the sovereign pipeline (capability → risk → authority → sandbox). They are the
*only* way the reasoning/act layer touches secrets, and the boundary is deliberate:

* mutating tools (``credential_store``/``_rotate``/``_revoke``/``ssh_keygen``) carry
  :attr:`~nyxara.agency.permissions.Capability.SECRETS_ACCESS` at ``HIGH`` risk, so an
  autonomous invocation **escalates to the Master** (Rules 1 & 7) before it runs;
* ``credential_list`` is ``TRIVIAL`` and returns **redacted metadata only** (Rule 6);
* ``credential_request`` *uses* a stored secret to make an authenticated HTTP call and
  returns the response **without the secret** — the plaintext is injected inside the kernel
  by :meth:`CredentialVault.use` and never reaches the tool result or the LLM.

There is intentionally **no reveal tool**: returning a plaintext secret is Master-direct,
Python-only (:meth:`CredentialVault.reveal`), never exposed to the model.
"""

from __future__ import annotations

from typing import Any, Dict, List

from nyxara.agency.permissions import Authority, Capability, RiskTier
from nyxara.agency.tools import ToolParam, ToolRegistry, ToolSpec
from nyxara.guard.vault import CredentialKind, CredentialVault

__all__ = ["build_credential_tools"]


def _split_tags(raw: str) -> List[str]:
    return [t.strip() for t in (raw or "").split(",") if t.strip()]


def build_credential_tools(registry: ToolRegistry, vault: CredentialVault) -> ToolRegistry:
    """Register the vault-backed credential tools onto ``registry`` (idempotent-friendly)."""

    def _store(name: str, kind: str, secret: str, username: str = "", scope: str = "",
               tags: str = "", expires_in: float = 0.0) -> Dict[str, Any]:
        now = vault._clock()  # noqa: SLF001 — same clock the vault stamps with
        expires_at = (now + expires_in) if expires_in and expires_in > 0 else None
        return vault.put(name, CredentialKind(kind), secret, authority=Authority.OWNER,
                         username=username, scope=scope, tags=_split_tags(tags),
                         expires_at=expires_at)

    def _list(kind: str = "") -> List[Dict[str, Any]]:
        return vault.list(authority=Authority.AUTONOMOUS, kind=kind or None)

    def _rotate(name: str, new_secret: str = "") -> Dict[str, Any]:
        return vault.rotate(name, new_secret or None, authority=Authority.OWNER)

    def _revoke(name: str) -> Dict[str, Any]:
        return {"name": name, "revoked": vault.revoke(name, authority=Authority.OWNER)}

    def _ssh_keygen(name: str, comment: str = "nyxara", scope: str = "") -> Dict[str, Any]:
        return vault.mint_ssh(name, authority=Authority.OWNER, comment=comment, scope=scope)

    def _request(name: str, url: str, method: str = "GET",
                 header_template: str = "Authorization: Bearer {secret}") -> Dict[str, Any]:
        """Make an authenticated HTTP request injecting a stored secret — never returning it."""
        def _apply(secret: str) -> Dict[str, Any]:
            from nyxara.agency.net_request import http_request
            hname, _, vtmpl = header_template.partition(":")
            headers = {hname.strip(): vtmpl.strip().format(secret=secret)}
            resp = http_request(url, method=method.upper(), headers=headers)
            # Return only non-sensitive response fields (the secret is never echoed here).
            return {k: resp.get(k) for k in ("ok", "status", "body", "error")}
        return vault.use(name, _apply, authority=Authority.DELEGATED)

    specs = [
        ToolSpec(
            "credential_store", handler=_store,
            description="Store a secret (password/api_key/ssh_key/oauth_token) in NYXARA's "
                        "encrypted vault. The Master's authority (owner-gated).",
            params=[
                ToolParam("name", "str", description="short id to reference the credential"),
                ToolParam("kind", "str",
                          description="password|api_key|ssh_key|oauth_token"),
                ToolParam("secret", "str", description="the secret value to seal"),
                ToolParam("username", "str", required=False, default=""),
                ToolParam("scope", "str", required=False, default="",
                          description="host/service/URL this credential is for"),
                ToolParam("tags", "str", required=False, default="",
                          description="comma-separated tags"),
                ToolParam("expires_in", "float", required=False, default=0.0,
                          description="seconds until expiry (0 = never)"),
            ],
            capability=Capability.SECRETS_ACCESS, risk=RiskTier.HIGH,
            reversible=True, target_param="name",
        ),
        ToolSpec(
            "credential_list", handler=_list,
            description="List stored credentials as REDACTED metadata only (no secret values).",
            params=[ToolParam("kind", "str", required=False, default="",
                              description="filter by kind, or empty for all")],
            capability=Capability.SECRETS_ACCESS, risk=RiskTier.TRIVIAL, reversible=True,
        ),
        ToolSpec(
            "credential_rotate", handler=_rotate,
            description="Rotate a stored secret to a new value (or re-seal in place). Owner-gated.",
            params=[ToolParam("name", "str"),
                    ToolParam("new_secret", "str", required=False, default="",
                              description="new value, or empty to re-seal the same secret")],
            capability=Capability.SECRETS_ACCESS, risk=RiskTier.HIGH,
            reversible=True, target_param="name",
        ),
        ToolSpec(
            "credential_revoke", handler=_revoke,
            description="Permanently revoke/delete a stored credential. Owner-gated (Rule 5).",
            params=[ToolParam("name", "str")],
            capability=Capability.SECRETS_ACCESS, risk=RiskTier.HIGH,
            reversible=False, target_param="name",
        ),
        ToolSpec(
            "ssh_keygen", handler=_ssh_keygen,
            description="Mint a fresh Ed25519 SSH keypair NYXARA owns; store the private half, "
                        "return the public key + fingerprint. Owner-gated.",
            params=[ToolParam("name", "str"),
                    ToolParam("comment", "str", required=False, default="nyxara"),
                    ToolParam("scope", "str", required=False, default="")],
            capability=Capability.SECRETS_ACCESS, risk=RiskTier.HIGH,
            reversible=True, target_param="name",
        ),
        ToolSpec(
            "credential_request", handler=_request,
            description="Make an authenticated HTTP request USING a stored credential without "
                        "revealing it (the secret is injected in-kernel; only the response "
                        "returns). Under a standing owner blessing (delegated).",
            params=[
                ToolParam("name", "str", description="credential to use"),
                ToolParam("url", "str"),
                ToolParam("method", "str", required=False, default="GET"),
                ToolParam("header_template", "str", required=False,
                          default="Authorization: Bearer {secret}",
                          description="header line; '{secret}' is replaced in-kernel"),
            ],
            capability=Capability.SECRETS_ACCESS, risk=RiskTier.HIGH,
            reversible=False, target_param="name",
        ),
    ]
    for spec in specs:
        if registry.get(spec.name) is None:
            registry.register(spec)
    return registry
