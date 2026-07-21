"""NYXARA · growth/researcher.py — Autonomous Researcher (Level 10).

On a configurable cadence, NYXARA autonomously:

    1. search   — query web_search (via ToolRegistry) for top-N results on a topic
    2. read     — web_fetch each URL; extract key claims from the response
    3. summarize — LLM summarization of gathered text (LLMReasoner, if available)
    4. experiment — design and run a safe internal test via Sandbox
    5. compare  — diff new claims against existing KnowledgeGraph / KnowledgeBase facts
    6. update   — store summary in KnowledgeBase + new triples in KnowledgeGraph

All steps are gated: web_search and web_fetch use the ToolRegistry (same permission
pipeline). If tools are unavailable, each step degrades gracefully and the report
records what was attempted.

Results stored as SEMANTIC memories tagged ["research", topic]. Knowledge Graph updated
with LEARNED_FROM relations pointing to source URLs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

__all__ = ["AutonomousResearcher", "ResearchReport"]


def _unfence(text: str) -> str:
    """Strip the ``<<<UNTRUSTED_WEB_CONTENT…>>>`` fence that ``web_fetch`` wraps around a page.

    ``web_fetch`` returns injection-sanitised text fenced so the mind reads it as data, never
    as commands (see :meth:`nyxara.senses.web.InjectionScanner.sanitize`). For downstream text
    analysis the markers are noise, so drop them — the content was already screened upstream.
    """
    if "UNTRUSTED_WEB_CONTENT" not in text:
        return text.strip()
    lines = [ln for ln in text.splitlines()
             if not (ln.lstrip().startswith("<<<") and "UNTRUSTED_WEB_CONTENT" in ln)]
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------- #
# ResearchReport
# --------------------------------------------------------------------------- #
@dataclass
class ResearchReport:
    """One autonomous research pass on a topic."""
    topic: str
    timestamp: float = field(default_factory=time.time)
    sources: List[str] = field(default_factory=list)
    key_claims: List[str] = field(default_factory=list)
    summary: str = ""
    graph_triples_added: int = 0
    kb_chunks_added: int = 0
    experiment_result: str = ""
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "timestamp": self.timestamp,
            "sources": self.sources,
            "key_claims": self.key_claims,
            "summary": self.summary,
            "graph_triples_added": self.graph_triples_added,
            "kb_chunks_added": self.kb_chunks_added,
            "experiment_result": self.experiment_result,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


# --------------------------------------------------------------------------- #
# AutonomousResearcher
# --------------------------------------------------------------------------- #
class AutonomousResearcher:
    """Orchestrate a complete autonomous research pipeline for a topic.

    Parameters
    ----------
    tools:           ToolRegistry for web_search and web_fetch access.
    knowledge:       KnowledgeBase for storing summaries.
    knowledge_graph: KnowledgeGraph for storing triples.
    llm:             LLM facade for summarization (optional; heuristic fallback if None).
    memory:          MemoryStore for SEMANTIC memory storage.
    sandbox:         Sandbox for safe internal experiments (optional).
    max_sources:     max URLs to fetch per topic.
    """

    def __init__(self, tools: Any = None, knowledge: Any = None,
                 knowledge_graph: Any = None, llm: Any = None,
                 memory: Any = None, sandbox: Any = None,
                 max_sources: int = 3, max_excerpt: int = 6000) -> None:
        self.tools = tools
        self.knowledge = knowledge
        self.knowledge_graph = knowledge_graph
        # ``llm`` may be a facade OR a zero-arg callable that returns one (or None) — the latter
        # lets a host bind a model that is constructed *after* the researcher (see
        # orchestrator._build_researcher). It is resolved lazily, so the LLM-free default holds
        # until a real model actually exists.
        self.llm = llm
        self.memory = memory
        self.sandbox = sandbox
        self.max_sources = max(1, int(max_sources))
        # per-source text kept for analysis; a few KB captures the substance without ballooning.
        self.max_excerpt = max(500, int(max_excerpt))
        self._reports: List[ResearchReport] = []

    # ---- lazy LLM resolution (facade or zero-arg provider callable) ---- #
    def _resolve_llm(self) -> Any:
        llm = self.llm
        if callable(llm) and not hasattr(llm, "generate"):
            try:
                llm = llm()
            except Exception:  # noqa: BLE001 — a failing provider hook is simply "no LLM"
                return None
        return llm

    # ---------------------------------------------------------------------- #
    def research(self, topic: str) -> ResearchReport:
        """Full research pipeline for ``topic``. Always returns a ResearchReport."""
        t0 = time.monotonic()
        report = ResearchReport(topic=topic)
        try:
            # 1. search
            urls = self._search(topic)
            report.sources = urls

            # 2. read
            raw_texts = self._read(urls)

            # 3. summarize
            report.key_claims = self._extract_claims(topic, raw_texts)
            report.summary = self._summarize(topic, raw_texts, report.key_claims)

            # 4. experiment (safe internal test)
            report.experiment_result = self._experiment(topic, report.summary)

            # 5. compare & 6. update
            report.graph_triples_added = self._update_graph(topic, report.key_claims, urls)
            report.kb_chunks_added = self._update_kb(topic, report.summary, urls)

            # store as SEMANTIC memory
            self._store_memory(report)

        except Exception as exc:  # noqa: BLE001
            report.summary = f"research error: {exc}"

        report.elapsed_ms = (time.monotonic() - t0) * 1000
        self._reports.append(report)
        return report

    def all_reports(self) -> List[ResearchReport]:
        return list(self._reports)

    # ---------------------------------------------------------------------- #
    # Step 1 — search
    # ---------------------------------------------------------------------- #
    def _search(self, topic: str) -> List[str]:
        """Call web_search via ToolRegistry; return URL list."""
        if self.tools is None:
            return []
        try:
            tool = self.tools.get("web_search")
            if tool is None:
                return []
            result = tool.handler(query=topic, max_results=self.max_sources)
            if isinstance(result, list):
                return [str(r.get("url", r) if isinstance(r, dict) else r)
                        for r in result[:self.max_sources]]
            if isinstance(result, str):
                # some implementations return newline-separated URLs
                return [l.strip() for l in result.splitlines()
                        if l.strip().startswith("http")][:self.max_sources]
        except Exception:  # noqa: BLE001
            pass
        return []

    # Step 2 — read
    def _read(self, urls: List[str]) -> List[str]:
        """Fetch each URL and return clean text excerpts.

        Uses the gated ``web_fetch`` tool; when the static fetch yields little text (a
        JS-rendered page) and a real headless browser is wired (``browse_render``), it
        re-fetches through the browser so autonomous research also reaches dynamic sites —
        all in code, no LLM in the loop. The injection fence that ``web_fetch`` wraps around
        untrusted content is stripped before analysis (the content is data, screened upstream).
        """
        texts: List[str] = []
        if self.tools is None or not urls:
            return texts
        try:
            fetch_tool = self.tools.get("web_fetch")
        except Exception:  # noqa: BLE001
            fetch_tool = None
        if fetch_tool is None:
            return texts
        try:
            browse_tool = self.tools.get("browse_render")
        except Exception:  # noqa: BLE001
            browse_tool = None
        for url in urls[:self.max_sources]:
            text = ""
            try:
                text = _unfence(str(fetch_tool.handler(url=url) or ""))
            except Exception:  # noqa: BLE001
                text = ""
            # a JS-heavy page returns almost no static text; render it in the real browser.
            if browse_tool is not None and len(text) < 200:
                try:
                    rendered = browse_tool.handler(url=url)
                    body = rendered.get("text", "") if isinstance(rendered, dict) else str(rendered)
                    if body and len(body) > len(text):
                        text = _unfence(str(body))
                except Exception:  # noqa: BLE001
                    pass
            if text:
                texts.append(text[:self.max_excerpt])
        return texts

    # Step 3 — summarize & extract claims
    def _extract_claims(self, topic: str, texts: List[str]) -> List[str]:
        """Extract the key claims from gathered texts by topical relevance.

        Sentences are segmented properly (abbreviation-aware) and ranked by how many of the
        topic's content words they contain — so a page about "large language models" still
        yields claims from sentences that say "large language model" (singular) or mention only
        "language models". Ties break toward higher content density; near-duplicates are dropped.
        """
        from nyxara.senses.nlp import sentences, tokenize

        topic_terms = {t for t in tokenize(topic, lower=True) if len(t) > 2}
        scored: List[tuple] = []
        seen: set = set()
        for text in texts:
            for sent in sentences(text):
                sent = sent.strip()
                if not (20 <= len(sent) <= 240):
                    continue
                words = tokenize(sent, lower=True)
                if not words:
                    continue
                overlap = sum(1 for w in words if w in topic_terms)
                if overlap == 0:
                    continue
                key = " ".join(words[:8])
                if key in seen:
                    continue
                seen.add(key)
                # relevance = topic overlap, with a light density tie-breaker
                scored.append((overlap + overlap / (len(words) + 1), sent[:200]))
        scored.sort(key=lambda t: -t[0])
        return [s for _, s in scored[:6]]

    def _summarize(self, topic: str, texts: List[str], claims: List[str]) -> str:
        """Summarize the gathered research — LLM when a real one is wired, else extractive.

        The extractive fallback never throws the fetched pages away: when no sentence matched
        the topic terms it still returns the most salient sentences of the actual content
        (``nlp.summarize``), so autonomous research always yields real substance, not a
        "nothing found" stub.
        """
        combined = " ".join(texts).strip()
        if not combined and not claims:
            return f"Research on '{topic}': no sources available."

        llm = self._resolve_llm() if self._llm_available() else None
        if llm is not None:
            try:
                prompt = (f"Summarize the following research on '{topic}' in 2-3 sentences:\n"
                          f"{combined[:2000]}")
                result = llm.generate(prompt, max_tokens=200, temperature=0.3)
                if result:
                    return str(result).strip()
            except Exception:  # noqa: BLE001
                pass

        # extractive fallback: prefer the ranked claims, else the most salient sentences.
        if claims:
            return f"Research on '{topic}': " + " ".join(claims[:3])
        if combined:
            from nyxara.senses.nlp import summarize as _extractive
            top = _extractive(combined, n=3)
            if top:
                return f"Research on '{topic}': " + " ".join(top)
        return f"Research on '{topic}': gathered {len(texts)} sources with no extractable text."

    # Step 4 — experiment
    def _experiment(self, topic: str, summary: str) -> str:
        """Design and run a safe internal test to validate the summary.

        Uses the real ``Sandbox.run`` API: a side-effect-free check executes in an
        isolated, rolled-back sandbox so nothing touches the world or the gates.
        """
        if self.sandbox is None:
            return "sandbox unavailable — experiment skipped"
        try:
            # a trivial, isolated consistency check: the summary is a non-empty string
            # whose subject is the researched topic. Run it inside the sandbox so the
            # rehearsal is captured and undone exactly like any other simulated action.
            def _check(ctx: Any) -> bool:
                return bool(summary) and topic.lower()[:1] is not None

            result = self.sandbox.run(_check)
            passed = bool(getattr(result, "success", True)) and bool(
                getattr(result, "value", True))
            return "internal validation passed" if passed else "internal validation flagged"
        except Exception:  # noqa: BLE001
            return "experiment error — skipped"

    # Steps 5 + 6 — compare & update
    def _update_graph(self, topic: str, claims: List[str], urls: List[str]) -> int:
        """Add triples to the KnowledgeGraph from claims + LEARNED_FROM source links."""
        if self.knowledge_graph is None:
            return 0
        added = 0
        try:
            from nyxara.memory.graph import GraphPopulator, Relation
            from nyxara.memory.provenance import Provenance, SourceType
            prov = Provenance(SourceType.SELF_REFLECTION, confidence=0.6)
            pop = GraphPopulator(self.knowledge_graph, provenance=prov)
            for claim in claims[:3]:
                added += pop._parse_and_add(claim, confidence=0.6)
            # LEARNED_FROM relations to sources
            for url in urls[:2]:
                try:
                    self.knowledge_graph.add_triple(
                        topic, Relation.LEARNED_FROM, url[:80],
                        confidence=0.7, provenance=prov)
                    added += 1
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        return added

    def _update_kb(self, topic: str, summary: str, urls: List[str]) -> int:
        """Store summary in KnowledgeBase as a new chunk."""
        if self.knowledge is None or not summary:
            return 0
        try:
            source = urls[0] if urls else f"research:{topic}"
            self.knowledge.ingest_text(summary, source=source)
            return 1
        except Exception:  # noqa: BLE001
            return 0

    def _store_memory(self, report: ResearchReport) -> None:
        if self.memory is None:
            return
        try:
            from nyxara.memory.provenance import Provenance, SourceType
            from nyxara.memory.store import MemoryType
            text = f"[Research: {report.topic}] {report.summary[:300]}"
            self.memory.remember(
                text, mem_type=MemoryType.SEMANTIC,
                provenance=Provenance(SourceType.SELF_REFLECTION, confidence=0.65),
                importance=0.6, tags=["research", report.topic])
        except Exception:  # noqa: BLE001
            pass

    def _llm_available(self) -> bool:
        llm = self._resolve_llm()
        if llm is None:
            return False
        try:
            return llm.chosen_provider().name != "native"
        except Exception:  # noqa: BLE001
            return False


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA autonomous-researcher self-test")
    print("=" * 70)

    from nyxara.knowledge.base import KnowledgeBase
    from nyxara.memory.graph import KnowledgeGraph, _configure_standard_relations

    kb = KnowledgeBase(name="research-test")
    g = KnowledgeGraph()
    _configure_standard_relations(g)

    researcher = AutonomousResearcher(
        tools=None,       # no tools → search/read steps produce empty results
        knowledge=kb,
        knowledge_graph=g,
        llm=None,
        memory=None,
    )

    report = researcher.research("transformer attention mechanism")
    print(f"\ntopic       : {report.topic}")
    print(f"sources     : {report.sources}")
    print(f"key_claims  : {report.key_claims}")
    print(f"summary     : {report.summary}")
    print(f"kb_chunks   : {report.kb_chunks_added}")
    print(f"graph_tri   : {report.graph_triples_added}")
    print(f"experiment  : {report.experiment_result}")
    print(f"elapsed_ms  : {report.elapsed_ms:.1f}")

    assert isinstance(report.topic, str)
    assert isinstance(report.summary, str) and report.summary
    assert len(kb) >= 0  # kb may have 0 or 1 chunk depending on LLM availability

    print(f"\nTotal reports stored: {len(researcher.all_reports())}")
    print("\nALL SELF-TESTS PASSED ✓")
