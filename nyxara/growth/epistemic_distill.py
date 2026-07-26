"""NYXARA · growth/epistemic_distill.py — Autonomous Epistemic Distillation (Part B).

The point: **kill the dependence on the cloud, one turn at a time.** Every successful teacher (airouter/
GLM-5) turn is a gift of knowledge — so instead of throwing it away after answering, NYXARA extracts the
answer *and its reasoning path* and bakes them into her own durable, local memory:

* a **Hyper-Dimensional Vector Rule** — the (condition, answer, reasoning) record encoded as one
  holographic hypervector in her ``LatentSpaceMap`` corpus (associative recall via ``.nearest``);
* a **provenanced triple** in her ``KnowledgeGraph`` (``question --has_verified_answer--> answer``),
  which her reasoning-time recall (``orchestrator._recall_for``) already consults;
* optionally a **semantic memory** for the retriever.

Because those stores already feed her confidence router, each distilled turn raises how often she can
answer *herself* — so the router stops handing off to the cloud and the teacher-ratio (the existing handoff
meter) falls over time. That is the whole mechanism, and it needs no new pre-call gate.

Honesty (Rule 6 + character-lock): teacher knowledge is tagged ``LLM_INFERENCE`` provenance — never claimed
as her own native reasoning — and, when the Part-M verifier is wired, nothing is baked as *durable* unless it
verifies (proven / high-confidence). Refuted answers are dropped; weak ones are stored tentative with fast decay.

Reuses existing organs only: ``memory/graph.KnowledgeGraph``, ``cognition/hyper_dimensional_vectors.LatentSpaceMap``,
``memory/store.MemoryStore``, ``growth/verified_distill.GroundedVerifier``. Best-effort throughout — a failure
here must never break a turn.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Optional

from nyxara.kernel.config import NyxaraSettings, get_settings
from nyxara.memory.provenance import Provenance, SourceType

__all__ = ["EpistemicResult", "EpistemicDistiller"]


@dataclass
class EpistemicResult:
    """Outcome of distilling one turn."""

    stored: bool
    verdict: str            # "proven" | "graded" | "refuted" | "disabled" | "skipped"
    confidence: float
    triples: int = 0
    hv_key: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {"stored": self.stored, "verdict": self.verdict,
                "confidence": round(self.confidence, 4), "triples": self.triples,
                "hv_key": self.hv_key, "reason": self.reason}


class EpistemicDistiller:
    """Turn a successful teacher turn into durable local knowledge (Part B)."""

    def __init__(self, *, knowledge_graph: Any = None, latent_map: Any = None,
                 memory: Any = None, verifier: Any = None,
                 settings: Optional[NyxaraSettings] = None, graph_path: Any = None) -> None:
        self.settings = settings or get_settings()
        self.cfg = getattr(self.settings, "epistemic_distill", None)
        self.knowledge_graph = knowledge_graph
        self.latent_map = latent_map
        self.memory = memory
        self.verifier = verifier
        self.graph_path = graph_path
        self._count = 0
        self._seen: set[str] = set()

    # ---- construction from a live core ---- #
    @classmethod
    def from_core(cls, core: Any) -> "EpistemicDistiller":
        """Build from a NyxaraCore, reusing its graph / HD map / memory and a fresh verifier."""
        settings = getattr(core, "settings", None) or get_settings()
        # the core builds a FIFO-capped LatentSpaceMap as ``hyperdimensional`` (orchestrator._build_hyperdimensional).
        # NB: LatentSpaceMap.__len__ is 0 when empty (falsy), so resolve with explicit `is not None`,
        # never an ``or`` chain — an empty-but-present map must still be reused.
        latent = None
        for attr in ("hyperdimensional", "latent_map", "latent_space"):
            cand = getattr(core, attr, None)
            if cand is not None:
                latent = cand
                break
        if latent is None:
            latent = getattr(getattr(core, "intuition", None), "_map", None)
        verifier = None
        try:
            if bool(getattr(getattr(settings, "epistemic_distill", None), "verify", True)):
                from nyxara.growth.verified_distill import GroundedVerifier
                verifier = GroundedVerifier()
        except Exception:  # noqa: BLE001 — verification is a bonus, never required
            verifier = None
        graph_path = None
        try:
            graph_path = settings.paths.data_dir / "knowledge_graph.json"
        except Exception:  # noqa: BLE001
            graph_path = None
        return cls(knowledge_graph=getattr(core, "knowledge_graph", None), latent_map=latent,
                   memory=getattr(core, "memory", None), verifier=verifier,
                   settings=settings, graph_path=graph_path)

    # ---- policy ---- #
    def enabled(self) -> bool:
        return bool(getattr(self.cfg, "enabled", True))

    def count(self) -> int:
        return self._count

    def report(self) -> dict:
        return {"enabled": self.enabled(), "distilled": self._count,
                "unique_prompts": len(self._seen),
                "hd_rules": (len(self.latent_map) if self.latent_map is not None else 0),
                "graph_triples": (len(self.knowledge_graph) if self.knowledge_graph is not None else 0)}

    @staticmethod
    def _bakeable(prompt: str) -> bool:
        """True iff ``prompt`` is the kind of question whose answer can be durable knowledge."""
        try:
            from nyxara.mind.native_reasoner import classify_intent
            return classify_intent(prompt) != "chat"
        except Exception:  # noqa: BLE001 — classification is advisory; bake rather than lose
            return True

    # ---- the distillation ---- #
    def distill_turn(self, prompt: str, answer: str, reasoning: Optional[str] = None, *,
                     confidence: float = 1.0, source: str = "teacher") -> EpistemicResult:
        if not self.enabled():
            return EpistemicResult(False, "disabled", 0.0, reason="disabled")
        p = (prompt or "").strip()
        a = (answer or "").strip()
        cfg = self.cfg
        min_chars = int(getattr(cfg, "min_chars", 2))
        max_chars = int(getattr(cfg, "max_chars", 4000))
        min_conf = float(getattr(cfg, "min_confidence", 0.4))
        if not p or not a:
            return EpistemicResult(False, "skipped", 0.0, reason="empty prompt/answer")
        if len(a) < min_chars or len(a) > max_chars:
            return EpistemicResult(False, "skipped", 0.0, reason="answer length out of bounds")
        fp = hashlib.sha256(p.lower().encode("utf-8")).hexdigest()[:16]
        if fp in self._seen:
            return EpistemicResult(False, "skipped", 0.0, reason="duplicate prompt")

        # ---- verify before baking (Part M) ---- #
        if self.verifier is not None:
            vr = self.verifier.verify(p, a, prior_confidence=confidence)
            if vr.refuted:
                return EpistemicResult(False, "refuted", 0.0, reason=f"refuted: {vr.certificate}")
            verdict, conf, durable, cert = vr.verdict, vr.confidence, vr.durable, vr.certificate
        else:
            verdict = "graded"
            conf = max(0.0, min(1.0, float(confidence)))
            durable = conf >= min_conf
            cert = ""
        if conf < min_conf and not durable:
            return EpistemicResult(False, verdict, conf, reason="below min_confidence")

        self._seen.add(fp)
        half_life = (float(getattr(cfg, "durable_half_life_days", 3650.0)) if durable
                     else float(getattr(cfg, "tentative_half_life_days", 7.0)))
        prov = Provenance(
            source=SourceType.LLM_INFERENCE,               # honest: originated from a probabilistic model
            confidence=conf,
            detail=source,
            method=("verified" if verdict == "proven" else "inferred"),
            half_life_days=half_life,
            tags=frozenset({"teacher", "epistemic-distill", verdict}),
        )

        # ---- 1) durable, queryable KnowledgeGraph triple ---- #
        # ordinary conversation is never baked as a "verified answer": a chat utterance keyed on
        # the whole prompt would replay a one-off conversational reply forever. Only questions
        # with extractable structure (factual / computational / causal …) earn a durable triple;
        # the HD-vector rule below still learns the exchange either way.
        triples = 0
        if self.knowledge_graph is not None and self._bakeable(p):
            try:
                self.knowledge_graph.add_triple(
                    p[:80], "has_verified_answer", a[:200],
                    confidence=conf, provenance=prov,
                    attributes={"verdict": verdict, "certificate": cert, "source": source})
                triples += 1
            except Exception:  # noqa: BLE001 — best-effort; never break a turn
                pass
            if bool(getattr(cfg, "persist", True)) and self.graph_path is not None:
                try:
                    self.knowledge_graph.save(str(self.graph_path))
                except Exception:  # noqa: BLE001
                    pass

        # ---- 2) Hyper-Dimensional Vector Rule (condition ⊗ answer ⊗ reasoning) ---- #
        hv_key: Optional[str] = None
        if self.latent_map is not None:
            try:
                hv_key = f"rule::{fp}"
                self.latent_map.add(hv_key, {"condition": p, "answer": a,
                                             "reasoning": (reasoning or "")[:500]})
            except Exception:  # noqa: BLE001
                hv_key = None

        # ---- 3) semantic memory for the retriever (optional) ---- #
        if self.memory is not None:
            try:
                from nyxara.memory.store import MemoryType
                self.memory.remember(
                    f"Q: {p}\nA: {a}", mem_type=MemoryType.SEMANTIC, provenance=prov,
                    importance=conf, tags=["epistemic-distill", verdict, source],
                    half_life_days=half_life)
            except Exception:  # noqa: BLE001
                pass

        self._count += 1
        return EpistemicResult(True, verdict, conf, triples=triples, hv_key=hv_key, reason="stored")
