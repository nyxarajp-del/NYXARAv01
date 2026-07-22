"""NYXARA · growth/online.py — online / connected mode, offline stays the floor (🌐, Rule 4, F18).

NYXARA can run **continuously connected** so she keeps getting stronger from the live world — but every
online input is a **hypothesis until verified**, never trusted raw. This module is the **verify-before-
trust gate** and safety posture around the existing web machinery (:mod:`nyxara.growth.researcher`,
:mod:`nyxara.growth.acquire`, :mod:`nyxara.mind.rag`); it does **not** fetch by itself — a read-only
fetcher is supplied, and this layer decides what may enter her permanent knowledge.

The discipline (mandatory):

* **Verify before trust.** A **decidable** claim goes to the :class:`~nyxara.growth.prover.Prover`:
  ``PROVEN`` → accepted; ``REFUTED`` → rejected; ``UNPROVABLE`` → **quarantined** as a hypothesis, not
  trusted. A plain fact is accepted only when **independently corroborated** by enough distinct sources;
  otherwise it is quarantined. Web is a *teacher / data source*, **never the voice** — the sovereign
  offline mind still answers.
* **Safety (larger surface → tighter gate).** Read-only; **no web-driven code execution** (content that
  looks like code/commands is refused); nothing that touches the sealed character core is admitted; any
  **outbound action escalates to the Master** and never auto-acts; **rate-limited**; **owner on/off**;
  and it **respects the environment's network policy**. On loss of connectivity she degrades to full
  offline instantly (F17 Hibernate-and-Dream) — no capability collapse, and she still yields to a stop.

Pure standard library; **no LLM** (and never a *dependency* on a cloud LLM — sovereignty preserved).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from nyxara.growth.integrity import HibernateAndDream, IntegritySeal
from nyxara.growth.noesis import _FORBIDDEN
from nyxara.growth.prover import Prover, ProofVerdict

__all__ = ["OnlineItem", "AcquisitionResult", "OutboundDecision", "OnlineMode"]

# content that must never be executed from the web (no web-driven code execution)
_CODE_MARKERS = ("import os", "import sys", "subprocess", "os.system", "__import__",
                 "exec(", "eval(", "rm -rf", "curl ", "wget ", "socket.", "open(")


@dataclass
class OnlineItem:
    source: str
    content: str
    kind: str = "fact"          # "fact" | "claim" (a decidable statement) | "data"
    statement: Optional[str] = None   # for kind == "claim": the decidable proposition


@dataclass
class AcquisitionResult:
    item: OnlineItem
    status: str                 # "accepted" | "quarantined" | "rejected"
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.item.source, "status": self.status, "reason": self.reason}


@dataclass
class OutboundDecision:
    action: str
    escalated: bool = True      # ALWAYS escalate — never auto-act on the outside world
    acted: bool = False
    reason: str = "outbound actions require the Master's approval — escalated, never auto-acted"


class OnlineMode:
    """The verify-before-trust gate + safety posture for always-on connectivity (F18)."""

    def __init__(self, *, enabled: bool = False, network_policy_allows: bool = True,
                 rate_limit: int = 32, corroboration: int = 2,
                 seal: Optional[IntegritySeal] = None) -> None:
        self.enabled = enabled                          # owner on/off (default OFF)
        self.network_policy_allows = network_policy_allows
        self.rate_limit = rate_limit
        self.corroboration = corroboration
        self.seal = seal or IntegritySeal()
        self.hibernation = HibernateAndDream(self.seal)
        self.prover = Prover()
        self.knowledge: List[OnlineItem] = []           # the permanent, VERIFIED core
        self.quarantine: List[OnlineItem] = []          # hypotheses, held apart, never trusted
        self._fact_sources: Dict[str, set] = {}

    # ---- the honest invariant ---- #
    @staticmethod
    def is_voice() -> bool:
        """Online content is a teacher/data source — never NYXARA's voice. The offline mind answers."""
        return False

    def active(self) -> bool:
        return self.enabled and self.network_policy_allows

    # ---- safety pre-checks ---- #
    def _unsafe_code(self, content: str) -> bool:
        low = content.lower()
        return any(m in low for m in _CODE_MARKERS)

    def _touches_character(self, content: str) -> bool:
        return bool(_FORBIDDEN.intersection(re.findall(r"[a-z0-9]+", content.lower())))

    # ---- the gate for one item ---- #
    def _gate(self, item: OnlineItem) -> AcquisitionResult:
        if self._unsafe_code(item.content):
            return AcquisitionResult(item, "rejected", "web-driven code execution is forbidden")
        if self._touches_character(item.content):
            return AcquisitionResult(item, "rejected", "may not redefine the sealed character core")
        if item.kind == "claim" and item.statement:
            verdict = self.prover.certify(item.statement).verdict
            if verdict is ProofVerdict.PROVEN:
                return AcquisitionResult(item, "accepted", "proven by her own verifier")
            if verdict is ProofVerdict.REFUTED:
                return AcquisitionResult(item, "rejected", "refuted by her own verifier")
            return AcquisitionResult(item, "quarantined", "undecidable — held as a hypothesis")
        # a plain fact: trust only on independent corroboration
        srcs = self._fact_sources.setdefault(item.content.strip().lower(), set())
        srcs.add(item.source)
        if len(srcs) >= self.corroboration:
            return AcquisitionResult(item, "accepted",
                                     f"corroborated by {len(srcs)} independent sources")
        return AcquisitionResult(item, "quarantined",
                                 f"only {len(srcs)}/{self.corroboration} sources — held pending")

    # ---- acquire a batch (rate-limited, gated, offline-floor safe) ---- #
    def acquire(self, items: Sequence[OnlineItem]) -> List[AcquisitionResult]:
        if not self.active():
            return []                                   # owner off or policy blocks → acquire nothing
        results: List[AcquisitionResult] = []
        for item in items[: self.rate_limit]:           # rate limit — never a firehose
            res = self._gate(item)
            if res.status == "accepted":
                self.knowledge.append(item)             # only VERIFIED content enters the core
            elif res.status == "quarantined":
                self.quarantine.append(item)            # hypotheses, kept apart
            results.append(res)
        return results

    # ---- outbound actions always escalate ---- #
    def propose_outbound(self, action: str) -> OutboundDecision:
        return OutboundDecision(action)                 # escalated, never acted

    # ---- connectivity loss → offline floor ---- #
    def on_network(self, *, up: bool) -> Dict[str, Any]:
        report = self.hibernation.on_environment(network_up=up)
        return {**report.to_dict(), "is_voice": self.is_voice()}
