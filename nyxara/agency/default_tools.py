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
                        max_read_bytes: int = 200_000) -> ToolRegistry:
    """Register NYXARA's default real toolset onto ``registry`` (idempotent-skipping).

    ``memory`` (an optional :class:`~nyxara.memory.store.MemoryStore`) wires the
    ``recall_memory`` / ``remember_fact`` tools to her actual long-term store.
    """
    existing = set(registry.names())

    def _add(spec: ToolSpec) -> None:
        if spec.name not in existing:
            registry.register(spec)

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

    # ---- network fetch: SSRF-guarded, injection-sanitised (senses/web), fails as data ---- #
    def _web_fetch(url: str) -> str:
        from nyxara.senses.web import Web  # SSRF guards + prompt-injection sanitiser + caching
        page = Web().page(url)
        # sanitized_text strips injection patterns from untrusted web content (defense in depth)
        return page.sanitized_text()[:max_read_bytes]

    _add(ToolSpec("web_fetch", handler=_web_fetch,
                  description="fetch & sanitise the readable text of an HTTP(S) page",
                  params=[ToolParam("url", "str")],
                  capability=Capability.NET_OUT, risk=RiskTier.MODERATE,
                  target_param="url"))

    # ---- web search: live results via DuckDuckGo's instant-answer API ---- #
    def _web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
        import json
        import urllib.parse
        from nyxara.senses.web import WebFetcher
        q = urllib.parse.quote(query)
        url = (f"https://api.duckduckgo.com/?q={q}"
               "&format=json&no_html=1&no_redirect=1&skip_disambig=1")
        res = WebFetcher().fetch(url)
        if not res.ok:
            raise RuntimeError(res.error or f"search failed (HTTP {res.status})")
        data = json.loads(res.body)
        out: List[Dict[str, str]] = []
        if data.get("AbstractText"):
            out.append({"title": data.get("Heading", ""),
                        "text": data["AbstractText"], "url": data.get("AbstractURL", "")})
        for topic in data.get("RelatedTopics", []):
            if len(out) >= max_results:
                break
            if isinstance(topic, dict) and topic.get("Text"):
                out.append({"title": "", "text": topic["Text"],
                            "url": topic.get("FirstURL", "")})
        return out[:max_results]

    _add(ToolSpec("web_search", handler=_web_search,
                  description="search the web and return titled result snippets with URLs",
                  params=[ToolParam("query", "str"),
                          ToolParam("max_results", "int", required=False, default=5)],
                  capability=Capability.NET_OUT, risk=RiskTier.LOW))

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
