"""NYXARA · guard/isolation_envelope.py — the air-gapped mind (Part K).

When NYXARA calls an external cloud tool (any ``mind/llm.py::OpenAICompatProvider`` — aicredits, Groq,
airouter), the biggest privacy risk
is that the provider — or anything on the wire — learns *who* she is, *whose* secrets she holds, or the
internal names of her own architecture. The Isolation Envelope closes that gap the only honest way it can:

    Before the query leaves, every sensitive **named** term (her identity, the Master's name/handle/email,
    registered secrets, architecture code-names) is replaced by an opaque, deterministic token
    (``X1``, ``Y2``, ``Z3`` …). The cloud model solves an *abstract* problem. The reply is re-hydrated
    **locally** — the reverse map never crosses the wire.

Honest boundary (stated so no one over-trusts it): this reliably hides *named identifiers and secrets*
and de-identifies code/logic. It **cannot** hide the abstract *shape* of a problem, and free-form natural
language cannot be fully anonymized without destroying meaning. So it is strong best-effort privacy for
code/math/logic — not a guarantee of total information hiding.

Stateless across requests, stateful within one: the calling cloud provider builds a fresh envelope per
call, so the substitution map lives only for that single request/response round-trip. Each provider passes
its OWN isolation flag via ``enabled=`` (``aicredits_isolation`` / ``groq_isolation`` /
``airouter_isolation``), so privacy is per-rung rather than tied to whichever provider happened to be
the first one written.

Note which rungs are NOT here. Her own brains — ``litertlm`` (the on-device Gemma that now leads the
ladder), ``self`` and ``native``, i.e. ``config.OWN_PROVIDERS`` — run in her own process, so no prompt
of theirs ever reaches a wire and there is nothing to abstract. They do not consult this module, and
that is not an oversight: an envelope over a local model would cost accuracy to hide a name from
itself. The strongest privacy posture available is therefore the default one — her primary answers
on-device, and the envelope only engages on the cloud rungs beneath it.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

from nyxara.kernel.config import OWNER, NyxaraSettings, get_settings

__all__ = ["IsolationEnvelope"]

# Token alphabet — rotates X,Y,Z,W,… so tokens read like the classic X1/Y2/Z3 abstraction.
_LETTERS = "XYZWABCDEFGHIJKLMNOPQRSTUV"


class IsolationEnvelope:
    """Abstract sensitive named terms out of an outgoing prompt; re-hydrate the reply locally.

    The map is **bijective** and **per-instance** (one request). Longer composite terms are replaced
    before their substrings, so ``jaypal khoja`` / an email / a namespace are abstracted whole and the
    round-trip is lossless.
    """

    def __init__(self, settings: Optional[NyxaraSettings] = None, *,
                 extra_secrets: Sequence[str] = (),
                 enabled: Optional[bool] = None) -> None:
        self.settings = settings or get_settings()
        # ``enabled`` lets the CALLING provider supply its own isolation flag (aicredits_isolation /
        # groq_isolation / airouter_isolation), so privacy is decided per cloud rung. Left None the
        # envelope has no
        # provider scope and falls back to the legacy read below.
        self._enabled_override = enabled
        self._forward: Dict[str, str] = {}   # original term -> token
        self._reverse: Dict[str, str] = {}   # token -> original term
        self._terms: List[str] = self._collect_terms(extra_secrets)

    # ---- policy ---- #
    def enabled(self) -> bool:
        if self._enabled_override is not None:
            return bool(self._enabled_override)
        return bool(getattr(self.settings.llm, "airouter_isolation", True))

    def _collect_terms(self, extra_secrets: Sequence[str]) -> List[str]:
        """Sensitive named terms, longest-first (so composites are matched before their parts)."""
        terms: List[str] = []
        try:
            terms += [OWNER.name, OWNER.handle, OWNER.email, OWNER.continuity_namespace]
        except Exception:  # noqa: BLE001 — identity is best-effort
            pass
        # NYXARA's own identity strings — the cloud tool must not learn who it serves.
        terms += ["NYXARA", "Nyxara", "nyxara"]
        terms += [s for s in extra_secrets if s and s.strip()]
        uniq = {t.strip() for t in terms if t and t.strip()}
        return sorted(uniq, key=len, reverse=True)

    # ---- token assignment ---- #
    def _token_for(self, term: str) -> str:
        tok = self._forward.get(term)
        if tok is None:
            i = len(self._forward)
            tok = f"{_LETTERS[i % len(_LETTERS)]}{i + 1}"
            self._forward[term] = tok
            self._reverse[tok] = term
        return tok

    @staticmethod
    def _pattern(term: str) -> "re.Pattern[str]":
        # Word-boundary match for purely word-like terms (so "JP" isn't hit inside "JPEG");
        # a plain escaped match for terms carrying punctuation (emails, dotted namespaces).
        if re.fullmatch(r"\w+", term):
            return re.compile(rf"\b{re.escape(term)}\b")
        return re.compile(re.escape(term))

    # ---- outbound: concrete -> abstract ---- #
    def abstract(self, text: str) -> str:
        if not text or not self.enabled():
            return text
        out = text
        for term in self._terms:
            pat = self._pattern(term)
            if pat.search(out):
                tok = self._token_for(term)
                out = pat.sub(tok, out)
        return out

    def abstract_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Return a copy of the chat messages with every sensitive term abstracted."""
        if not self.enabled():
            return messages
        return [{**m, "content": self.abstract(str(m.get("content", "")))} for m in messages]

    # ---- inbound: abstract -> concrete (LOCAL ONLY) ---- #
    def rehydrate(self, text: str) -> str:
        if not text or not self._reverse:
            return text
        out = text
        # Replace longer tokens first is unnecessary (tokens are non-overlapping), but use word
        # boundaries so a token isn't matched inside an unrelated word the model may have produced.
        for tok, original in self._reverse.items():
            out = re.sub(rf"\b{re.escape(tok)}\b", original, out)
        return out

    # ---- introspection (never leaves the process) ---- #
    def mapping(self) -> Dict[str, str]:
        return dict(self._forward)
