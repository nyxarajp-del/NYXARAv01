"""NYXARA · agency/default_tools.py — a small, safe, real default toolset (⚙).

A mind that can only talk is half a mind. This module hands NYXARA a handful of
*actually executable* tools so the sovereign loop's **act** stage reaches into the
world for real (governed, typed, capability-bound) instead of merely recording an
intent. Every tool here is deliberately conservative:

* read-only / low-blast-radius by default (time, arithmetic, file reads, memory);
* the few that write are typed and capability-bound so the kernel's gates and the
  registry's safety pipeline still decide whether they may run;
* heavy/optional reach (the network) is import-guarded and fails as data, never as a
  crash.

These are *defaults* — :func:`build_default_tools` registers them onto any
:class:`~nyxara.agency.tools.ToolRegistry`, and a deployment can add, remove, or
override freely. The registry, not this module, enforces permission/governance.
"""

from __future__ import annotations

import ast
import operator
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from nyxara.agency.permissions import Capability, RiskTier
from nyxara.agency.tools import ToolParam, ToolRegistry, ToolSpec

__all__ = ["build_default_tools", "safe_calculate"]

# --------------------------------------------------------------------------- #
# A safe arithmetic evaluator (no builtins, no names, no attribute access)
# --------------------------------------------------------------------------- #
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_calculate(expression: str) -> float:
    """Evaluate a pure-arithmetic expression with no access to names or builtins."""
    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    tree = ast.parse(expression, mode="eval")
    return _eval(tree)


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
def build_default_tools(registry: ToolRegistry, *, memory: Any = None,
                        knowledge: Any = None, enable_code: bool = True,
                        max_read_bytes: int = 200_000,
                        web: Any = None, governor: Any = None) -> ToolRegistry:
    """Register NYXARA's default real toolset onto ``registry`` (idempotent-skipping).

    ``memory`` (an optional :class:`~nyxara.memory.store.MemoryStore`) wires the
    ``recall_memory`` / ``remember_fact`` tools to her actual long-term store. When a
    store is present a :class:`~nyxara.knowledge.base.KnowledgeBase` is also wired
    (``knowledge_ingest`` / ``knowledge_search``) for grounded retrieval, unless an
    explicit ``knowledge`` base is supplied. ``enable_code`` adds the higher-blast-radius
    reach (sandboxed code, a shell) — owner-gated by the registry's capability/risk
    pipeline, so they escalate rather than auto-run. ``web`` (a
    :class:`~nyxara.kernel.config.WebConfig`) and ``governor`` drive NYXARA's internet
    tools (``web_search`` / ``web_fetch`` / ``http_request``): search provider + keys, size
    caps, timeouts, redirect and rate limits, and the SSRF posture. When ``web`` is omitted
    it is read from settings; the web tools are always registered.
    """
    existing = set(registry.names())

    def _add(spec: ToolSpec) -> None:
        if spec.name not in existing:
            registry.register(spec)

    # ---- internet-access config: fall back to settings, then to None (never crash) ---- #
    if web is None:
        try:
            from nyxara.kernel.config import get_settings
            web = get_settings().web
        except Exception:  # noqa: BLE001 — config is a convenience here, never a hard dep
            web = None

    def _secret(v: Any) -> Optional[str]:
        return v.get_secret_value() if v is not None else None

    def _web_kw() -> Dict[str, Any]:
        """Construction kwargs for a config-driven, governed WebFetcher/Web facade."""
        if web is None:
            return {"governor": governor}
        return {"governor": governor, "allow_private": bool(web.allow_private),
                "max_bytes": int(web.max_bytes), "timeout": float(web.timeout_s),
                "user_agent": web.user_agent}

    # ---- time: trivial, read-only ---- #
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    _add(ToolSpec("now", handler=_now, description="current UTC time (ISO-8601)",
                  capability=Capability.TOOL_CALL, risk=RiskTier.TRIVIAL))

    # ---- arithmetic: safe, no eval of names/builtins ---- #
    _add(ToolSpec("calculate", handler=safe_calculate,
                  description="evaluate a pure-arithmetic expression, e.g. '2*(3+4)'",
                  params=[ToolParam("expression", "str", description="arithmetic only")],
                  capability=Capability.TOOL_CALL, risk=RiskTier.LOW))

    # ---- filesystem reads: low risk, scoped by target ---- #
    def _read_file(path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(max_read_bytes)

    _add(ToolSpec("read_file", handler=_read_file,
                  description="read a UTF-8 text file (truncated to a safe size)",
                  params=[ToolParam("path", "str")],
                  capability=Capability.FS_READ, risk=RiskTier.LOW,
                  target_param="path"))

    def _list_dir(path: str = ".") -> List[str]:
        return sorted(os.listdir(path))

    _add(ToolSpec("list_dir", handler=_list_dir,
                  description="list the entries of a directory",
                  params=[ToolParam("path", "str", required=False, default=".")],
                  capability=Capability.FS_READ, risk=RiskTier.LOW,
                  target_param="path"))

    # ---- filesystem write: moderate, capability-bound ---- #
    def _write_file(path: str, content: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"wrote {len(content)} chars to {path}"

    _add(ToolSpec("write_file", handler=_write_file,
                  description="write UTF-8 text to a file (overwrites)",
                  params=[ToolParam("path", "str"), ToolParam("content", "str")],
                  capability=Capability.FS_WRITE, risk=RiskTier.MODERATE,
                  reversible=False, target_param="path"))

    # ---- network fetch: injection-sanitised (senses/web), governed, fails as data ---- #
    def _web_fetch(url: str) -> str:
        from nyxara.senses.web import Web  # prompt-injection sanitiser + caching + governor
        # config-driven: honours allow_private / max_bytes / timeout and shares the
        # governor's "web" rate bucket.
        page = Web(**_web_kw()).page(url)
        # The network fetch reaches up to web.max_bytes; the text returned to the model is
        # bounded by max_read_bytes (the model-facing budget). sanitized_text strips
        # injection patterns from untrusted web content (defense in depth).
        return page.sanitized_text()[:max_read_bytes]

    _add(ToolSpec("web_fetch", handler=_web_fetch,
                  description="fetch & sanitise the readable text of an HTTP(S) page",
                  params=[ToolParam("url", "str")],
                  capability=Capability.NET_OUT, risk=RiskTier.MODERATE,
                  target_param="url"))

    # ---- web search: real SERP results (DuckDuckGo HTML; Brave/Tavily/SerpAPI if keyed) ---- #
    def _web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
        from nyxara.senses.search import WebSearcher
        if web is not None:
            searcher = WebSearcher(
                provider=web.search_provider,
                brave_key=_secret(web.brave_api_key),
                tavily_key=_secret(web.tavily_api_key),
                serpapi_key=_secret(web.serpapi_api_key),
                max_results=int(web.max_results), timeout_s=float(web.timeout_s),
                max_bytes=int(web.max_bytes), user_agent=web.user_agent,
                governor=governor, allow_private=bool(web.allow_private))
        else:
            searcher = WebSearcher(governor=governor)  # keyless DDG default
        # errors-as-data: a failed search returns [] rather than raising, so the act stage
        # never crashes on a flaky network.
        resp = searcher.search(query, max_results=max_results)
        return [r.to_dict() for r in resp.results]

    _add(ToolSpec("web_search", handler=_web_search,
                  description="search the live web (real SERP results) — titled snippets "
                              "with URLs; keyless DuckDuckGo by default",
                  params=[ToolParam("query", "str"),
                          ToolParam("max_results", "int", required=False, default=5)],
                  capability=Capability.NET_OUT, risk=RiskTier.LOW))

    # ---- generic HTTP: always available so any web API is reachable ---- #
    def _http_request(url: str, method: str = "GET", body: str = "") -> Dict[str, Any]:
        from nyxara.agency.net_request import http_request
        kw: Dict[str, Any] = {}
        if web is not None:
            kw = {"timeout_s": float(web.timeout_s), "max_bytes": int(web.max_bytes),
                  "allow_private": bool(web.allow_private),
                  "max_redirects": int(web.max_redirects)}
        return http_request(url, method=method, body=body or None, **kw)

    _add(ToolSpec("http_request", handler=_http_request,
                  description="make an HTTP(S) request (GET/POST/…); returns "
                              "status/headers/body, failing as data",
                  params=[ToolParam("url", "str"),
                          ToolParam("method", "str", required=False, default="GET"),
                          ToolParam("body", "str", required=False, default="")],
                  capability=Capability.NET_OUT, risk=RiskTier.MODERATE,
                  target_param="url"))

    # ---- multimodal perception: image / audio / documents (heavy ML import-guarded) ---- #
    def _inspect_image(path: str) -> Dict[str, Any]:
        from nyxara.senses.vision import Vision
        info = Vision().inspect(path)
        return info.to_dict() if info is not None else {"error": "unrecognised image"}

    _add(ToolSpec("inspect_image", handler=_inspect_image,
                  description="read an image's dimensions/format/size (no ML needed)",
                  params=[ToolParam("path", "str")],
                  capability=Capability.FS_READ, risk=RiskTier.LOW, target_param="path"))

    def _read_image_text(path: str) -> Dict[str, Any]:
        from nyxara.senses.vision import Vision
        text, note = Vision().ocr(path)
        return {"text": text, "note": note}

    _add(ToolSpec("read_image_text", handler=_read_image_text,
                  description="OCR the text in an image (honest note if OCR is unavailable)",
                  params=[ToolParam("path", "str")],
                  capability=Capability.FS_READ, risk=RiskTier.LOW, target_param="path"))

    def _transcribe_audio(path: str) -> Dict[str, Any]:
        from nyxara.senses.audio import Audio
        text, note = Audio().transcribe(path)
        return {"transcript": text, "note": note}

    _add(ToolSpec("transcribe_audio", handler=_transcribe_audio,
                  description="transcribe speech in an audio file (honest note if unavailable)",
                  params=[ToolParam("path", "str")],
                  capability=Capability.FS_READ, risk=RiskTier.LOW, target_param="path"))

    def _read_document(path: str) -> Dict[str, Any]:
        from nyxara.senses.ingest import Ingestor
        doc = Ingestor().ingest_file(path)
        return {"type": doc.doc_type.value, "text": doc.text[:max_read_bytes],
                "words": doc.word_count, "note": doc.note}

    _add(ToolSpec("read_document", handler=_read_document,
                  description="extract text from a document (txt/md/pdf/docx/…); honest note "
                              "if a parser is missing",
                  params=[ToolParam("path", "str")],
                  capability=Capability.FS_READ, risk=RiskTier.LOW, target_param="path"))

    # ---- generative output: image / speech (heavy models optional, real fallback) ---- #
    def _generate_image(prompt: str, path: str, size: int = 256) -> Dict[str, Any]:
        from nyxara.senses.generate import ImageGenerator
        return ImageGenerator().generate(prompt, path, size=size).to_dict()

    _add(ToolSpec("generate_image", handler=_generate_image,
                  description="generate an image from a text prompt to a PNG path "
                              "(diffusion if available, deterministic identicon otherwise)",
                  params=[ToolParam("prompt", "str"), ToolParam("path", "str"),
                          ToolParam("size", "int", required=False, default=256)],
                  capability=Capability.FS_WRITE, risk=RiskTier.MODERATE,
                  reversible=False, target_param="path"))

    def _synthesize_speech(text: str, path: str) -> Dict[str, Any]:
        from nyxara.senses.generate import SpeechSynthesizer
        return SpeechSynthesizer().synthesize(text, path).to_dict()

    _add(ToolSpec("synthesize_speech", handler=_synthesize_speech,
                  description="synthesise speech from text to a WAV path (TTS engine if "
                              "available, a tone signature otherwise)",
                  params=[ToolParam("text", "str"), ToolParam("path", "str")],
                  capability=Capability.FS_WRITE, risk=RiskTier.MODERATE,
                  reversible=False, target_param="path"))

    # ---- memory tools: wired only when a store is provided ---- #
    if memory is not None:
        def _recall_memory(query: str, k: int = 5) -> List[str]:
            return [rec.text()[:300] for rec, _ in memory.recall(query, k=k)]

        _add(ToolSpec("recall_memory", handler=_recall_memory,
                      description="search NYXARA's long-term memory for relevant records",
                      params=[ToolParam("query", "str"),
                              ToolParam("k", "int", required=False, default=5)],
                      capability=Capability.TOOL_CALL, risk=RiskTier.TRIVIAL))

        def _remember_fact(text: str, importance: float = 0.6) -> str:
            from nyxara.memory.store import MemoryType
            rec = memory.remember(text, mem_type=MemoryType.SEMANTIC,
                                  importance=max(0.0, min(1.0, importance)))
            return rec.mem_id

        _add(ToolSpec("remember_fact", handler=_remember_fact,
                      description="commit a fact to NYXARA's semantic long-term memory",
                      params=[ToolParam("text", "str"),
                              ToolParam("importance", "float", required=False, default=0.6)],
                      capability=Capability.TOOL_CALL, risk=RiskTier.LOW))

        # ---- forge her own model from lived memory (Master-gated, gauntlet-protected) ---- #
        def _train_self_model(generations: int = 1) -> Dict[str, Any]:
            from nyxara.growth.autolearn import GrowthEngine
            engine = GrowthEngine(memory=memory, enable_foundry=True)
            results = engine.improve_self(generations=max(1, generations))
            return {"generations": len(results),
                    "promoted": sum(1 for r in results if getattr(r, "promoted", False)),
                    "results": [r.to_dict() for r in results]}

        _add(ToolSpec("train_self_model", handler=_train_self_model,
                      description="train/upgrade NYXARA's OWN model from her lived memory "
                                  "(gauntlet-gated promotion; n-gram backend if torch absent)",
                      params=[ToolParam("generations", "int", required=False, default=1)],
                      capability=Capability.SELF_MODIFY, risk=RiskTier.HIGH))

        # ---- distil the teacher LLM into her own supervised corpus (Master-gated) ---- #
        def _distill_from_teacher(count: int = 0) -> Dict[str, Any]:
            from nyxara.growth.distill import Distiller
            d = Distiller()
            if not d.available():
                return {"available": False, "added": 0,
                        "note": "no real teacher LLM configured (set a provider API key)"}
            exs = d.distill_default(n=count if count and count > 0 else None)
            return {"available": True, "added": len(exs), "store_size": d.count(),
                    "store": str(d.store_path)}

        _add(ToolSpec("distill_from_teacher", handler=_distill_from_teacher,
                      description="teach NYXARA's OWN model by distilling the configured teacher "
                                  "LLM into supervised examples in her voice (feeds the foundry "
                                  "corpus; no weights change until train_self_model runs)",
                      params=[ToolParam("count", "int", required=False, default=0)],
                      capability=Capability.SELF_MODIFY, risk=RiskTier.HIGH))

        # ---- curiosity self-play: invent hard questions, distil answers (Master-gated) ---- #
        def _self_play(n: int = 6) -> Dict[str, Any]:
            from nyxara.growth.selfplay import SelfPlay
            return SelfPlay().play(max(1, n))

        _add(ToolSpec("self_play", handler=_self_play,
                      description="curiosity round: NYXARA invents hard questions, the teacher "
                                  "answers them in her voice, and the pairs feed the foundry "
                                  "corpus (grows her own training data; no weights change yet)",
                      params=[ToolParam("n", "int", required=False, default=6)],
                      capability=Capability.SELF_MODIFY, risk=RiskTier.HIGH))

        # ---- recursive self-improvement: review / architecture / benchmark / weakness ---- #
        def _rsi(**kw: Any):
            from nyxara.growth.recursive_improvement import RecursiveSelfImprovement
            return RecursiveSelfImprovement(memory=memory, **kw)

        def _self_review_code() -> Dict[str, Any]:
            try:
                return _rsi().review_code().to_dict()
            except Exception as exc:  # noqa: BLE001
                return {"error": str(exc)}

        _add(ToolSpec("self_review_code", handler=_self_review_code,
                      description="review NYXARA's own source (real AST static analysis: "
                                  "complexity, bare excepts, missing docstrings, long functions)",
                      capability=Capability.TOOL_CALL, risk=RiskTier.LOW))

        def _self_analyze_architecture() -> Dict[str, Any]:
            try:
                return _rsi().analyze_architecture().to_dict()
            except Exception as exc:  # noqa: BLE001
                return {"error": str(exc)}

        _add(ToolSpec("self_analyze_architecture", handler=_self_analyze_architecture,
                      description="analyse NYXARA's own architecture: import graph, cycles, "
                                  "layering violations, god-modules",
                      capability=Capability.TOOL_CALL, risk=RiskTier.LOW))

        def _self_run_benchmarks(category: str = "") -> Dict[str, Any]:
            try:
                return _rsi().run_benchmarks(category=category or None)
            except Exception as exc:  # noqa: BLE001
                return {"error": str(exc)}

        _add(ToolSpec("self_run_benchmarks", handler=_self_run_benchmarks,
                      description="run NYXARA's capability benchmark on herself and report "
                                  "accuracy, per-category scores, failures and handoff",
                      params=[ToolParam("category", "str", required=False, default="")],
                      capability=Capability.TOOL_CALL, risk=RiskTier.MODERATE))

        def _self_detect_weaknesses() -> Dict[str, Any]:
            try:
                return _rsi().detect_weaknesses().to_dict()
            except Exception as exc:  # noqa: BLE001
                return {"error": str(exc)}

        _add(ToolSpec("self_detect_weaknesses", handler=_self_detect_weaknesses,
                      description="synthesise code-review + architecture + benchmark into a "
                                  "ranked list of NYXARA's own weaknesses with remediations",
                      capability=Capability.TOOL_CALL, risk=RiskTier.LOW))

        def _self_optimize(enact: bool = False) -> Dict[str, Any]:
            try:
                return _rsi().run(enact=bool(enact)).to_dict()
            except Exception as exc:  # noqa: BLE001
                return {"error": str(exc)}

        _add(ToolSpec("self_optimize", handler=_self_optimize,
                      description="run the full recursive self-improvement cycle; with enact=true "
                                  "auto-applies gauntlet-gated source fixes (verify-or-rollback, "
                                  "reversible) — owner-gated self-modification",
                      params=[ToolParam("enact", "bool", required=False, default=False)],
                      capability=Capability.SELF_MODIFY, risk=RiskTier.HIGH))

    # ---- knowledge base: grounded ingest + retrieval (backed by the store) ---- #
    kb = knowledge
    if kb is None and memory is not None:
        try:
            from nyxara.knowledge.base import KnowledgeBase
            kb = KnowledgeBase(store=memory, name="kb")
        except Exception:  # noqa: BLE001 — knowledge is a capability, never a hard dep
            kb = None
    if kb is not None:
        def _knowledge_ingest(text: str, source: str = "doc") -> Dict[str, Any]:
            n = kb.ingest_text(text, source=source)
            return {"chunks": n, "source": source, "size": len(kb)}

        _add(ToolSpec("knowledge_ingest", handler=_knowledge_ingest,
                      description="chunk & store text into the knowledge base for grounded recall",
                      params=[ToolParam("text", "str"),
                              ToolParam("source", "str", required=False, default="doc")],
                      capability=Capability.TOOL_CALL, risk=RiskTier.LOW))

        def _knowledge_search(query: str, k: int = 4) -> List[Dict[str, Any]]:
            return [{"text": c.text[:300], "source": c.source, "score": round(c.score, 3)}
                    for c in kb.retrieve(query, k=k)]

        _add(ToolSpec("knowledge_search", handler=_knowledge_search,
                      description="retrieve the most relevant knowledge-base chunks for a query",
                      params=[ToolParam("query", "str"),
                              ToolParam("k", "int", required=False, default=4)],
                      capability=Capability.TOOL_CALL, risk=RiskTier.TRIVIAL))

    # ---- higher-reach tools: sandboxed code, shell, generic HTTP (capability-gated) ---- #
    if enable_code:
        def _run_python(code: str, timeout_s: float = 5.0) -> Dict[str, Any]:
            from nyxara.agency.code_sandbox import run_python
            return run_python(code, timeout_s=max(0.1, min(timeout_s, 30.0))).to_dict()

        _add(ToolSpec("run_python", handler=_run_python,
                      description="execute Python in an isolated subprocess sandbox "
                                  "(no network, wall-clock timeout); returns stdout/value/errors",
                      params=[ToolParam("code", "str"),
                              ToolParam("timeout_s", "float", required=False, default=5.0)],
                      capability=Capability.CODE_EXEC, risk=RiskTier.HIGH, reversible=False))

        def _run_shell(command: str, timeout_s: float = 10.0) -> Dict[str, Any]:
            from nyxara.agency.code_sandbox import safe_shell
            return safe_shell(command, timeout_s=max(0.1, min(timeout_s, 60.0))).to_dict()

        _add(ToolSpec("run_shell", handler=_run_shell,
                      description="run a shell command in a subprocess (captured output, "
                                  "wall-clock timeout) — owner-gated, irreversible",
                      params=[ToolParam("command", "str"),
                              ToolParam("timeout_s", "float", required=False, default=10.0)],
                      capability=Capability.PROC_EXEC, risk=RiskTier.HIGH, reversible=False,
                      target_param="command"))

        # ---- self-extension: forge a brand-new tool for a missing capability ---- #
        # Gated at SELF_MODIFY/HIGH so invoking it through the loop escalates to the
        # Master (autonomous-max for self.modify is TRIVIAL). The tools it *produces*
        # are clamped to tool.call/low and run sandboxed per call.
        def _forge_capability(need: str) -> Dict[str, Any]:
            from nyxara.agency.permissions import Authority
            from nyxara.growth.capability_foundry import CapabilityFoundry
            foundry = CapabilityFoundry(registry=registry)
            return foundry.forge(need, authority=Authority.OWNER).to_dict()

        _add(ToolSpec("forge_capability", handler=_forge_capability,
                      description="forge a brand-new runnable, tested, registered tool for a "
                                  "missing capability (plan→code→test→benchmark→deploy); "
                                  "owner-gated self-extension",
                      params=[ToolParam("need", "str",
                                        description="the capability gap, e.g. 'sha256 hash'")],
                      capability=Capability.SELF_MODIFY, risk=RiskTier.HIGH, reversible=False))

    return registry


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.agency.permissions import Authority

    print("=" * 70)
    print("NYXARA default-tools self-test")
    print("=" * 70)

    reg = build_default_tools(ToolRegistry())
    print(f"\nregistered tools    : {reg.names()}")

    r = reg.invoke("calculate", {"expression": "2*(3+4)"}, authority=Authority.OWNER)
    print(f"calculate 2*(3+4)   : ok={r.ok} value={r.value}")
    assert r.ok and r.value == 14.0

    r = reg.invoke("now", {}, authority=Authority.OWNER)
    print(f"now                 : ok={r.ok} value={r.value}")
    assert r.ok and "T" in r.value

    # arithmetic evaluator rejects anything that isn't arithmetic
    rejected = False
    try:
        safe_calculate("__import__('os').system('echo hi')")
    except Exception:
        rejected = True
    assert rejected
    print("calc injection guard: rejected non-arithmetic ✓")

    print("\nALL SELF-TESTS PASSED ✓")
