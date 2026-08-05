"""NYXARA · guard/isolation_envelope.py — the air-gapped mind (Part K). CURRENTLY UNUSED.

**Read this first.** This module has no caller. It existed to protect one thing: a prompt on its way
to an external cloud model. Those rungs — aicredits, Groq, airouter — were removed from
``mind/llm.py`` at the Master's instruction, so every model she runs is now in-process and nothing
she thinks leaves the host. There is no wire left to hide anything from.

It is kept, rather than deleted, because it is a working, self-contained utility and the threat it
answers returns the moment anything external is added back. If you add a provider that reaches a
network, wire this to it in the same change — that is what it is for.

What it does, when something does call it:

    Before a query leaves, every sensitive **named** term (her identity, the Master's
    name/handle/email, registered secrets, architecture code-names) is replaced by an opaque,
    deterministic token (``X1``, ``Y2``, ``Z3`` …). The external model solves an *abstract* problem.
    The reply is re-hydrated **locally** — the reverse map never crosses the wire.

Honest boundary (stated so no one over-trusts it): this reliably hides *named identifiers and
secrets* and de-identifies code/logic. It **cannot** hide the abstract *shape* of a problem, and
free-form natural language cannot be fully anonymized without destroying meaning. So it is strong
best-effort privacy for code/math/logic — not a guarantee of total information hiding.

Stateless across requests, stateful within one: a caller builds a fresh envelope per call, so the
substitution map lives only for that single request/response round-trip.
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
        # ``enabled`` lets a CALLING provider supply its own isolation flag, so privacy stays a
        # per-provider decision rather than one global switch. Left None it defaults to ON: an
        # envelope built without a scope is being built by something that wants protection.
        self._enabled_override = enabled
        self._forward: Dict[str, str] = {}   # original term -> token
        self._reverse: Dict[str, str] = {}   # token -> original term
        self._terms: List[str] = self._collect_terms(extra_secrets)

    # ---- policy ---- #
    def enabled(self) -> bool:
        if self._enabled_override is not None:
            return bool(self._enabled_override)
        return True     # no provider scope given -> protect by default

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
