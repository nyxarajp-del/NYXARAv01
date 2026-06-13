"""NYXARA · agency/toolpacks.py — domain tool packs (wider reach, Pillar E1) 🧰.

A bare toolset is a pile of capabilities; a *role* is a curated set of them with a purpose.
This module groups NYXARA's governed tools into named **packs** — researcher, coder, maker —
so she can take on a role and so new domain tools register as a coherent bundle. Two facts make
this safe, not a back door:

* A pack composes tools by name and registers any pack-specific tools through the **same**
  ``ToolRegistry`` pipeline — every call still clears the capability / risk / authority / sandbox
  gates. A pack grants nothing the registry wouldn't already gate.
* The pack-specific tools added here are pure-standard-library and read-only / non-executing
  (an extractive summariser; a Python *syntax* checker that never runs the code), so they are
  TRIVIAL risk and reversible. Heavier reach (the network, the code sandbox, the shell) is the
  *existing* default toolset, referenced by name and gated exactly as before.

Reuses :mod:`nyxara.agency.default_tools` (the tools a pack references) and
:mod:`nyxara.agency.tools` (the registry it registers onto).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from nyxara.agency.permissions import Capability, RiskTier
from nyxara.agency.tools import ToolParam, ToolRegistry, ToolSpec

__all__ = [
    "ToolPack",
    "summarize_text",
    "python_check",
    "build_domain_packs",
    "register_packs",
]


# --------------------------------------------------------------------------- #
# Pack-specific tools (pure stdlib, read-only / non-executing → TRIVIAL)
# --------------------------------------------------------------------------- #
_STOPWORDS = frozenset(
    "the a an and or but if then else of to in on at for with from by as is are was were be "
    "been being it its this that these those i you he she we they them his her their our your "
    "not no nor so than too very can will just do does did has have had which who whom what "
    "when where why how about into over under again further once here there all any both each".split())

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z']+")


def summarize_text(text: str, max_sentences: int = 3) -> str:
    """Extractive summary of ``text`` — its most salient sentences, no model required.

    Scores each sentence by the summed frequency of its non-stopword terms (TextRank-lite),
    keeps the top ``max_sentences``, and returns them in their original order so the gist reads
    naturally. Deterministic and dependency-free."""
    text = (text or "").strip()
    if not text:
        return ""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if len(sentences) <= max(1, max_sentences):
        return " ".join(sentences)
    freq: Dict[str, int] = {}
    for w in (m.group().lower() for m in _WORD.finditer(text)):
        if len(w) > 2 and w not in _STOPWORDS:
            freq[w] = freq.get(w, 0) + 1
    if not freq:
        return " ".join(sentences[: max_sentences])

    def score(sentence: str) -> float:
        words = [m.group().lower() for m in _WORD.finditer(sentence)]
        return sum(freq.get(w, 0) for w in words) / (len(words) or 1)

    ranked = sorted(range(len(sentences)), key=lambda i: score(sentences[i]), reverse=True)
    chosen = sorted(ranked[: max(1, max_sentences)])        # keep original reading order
    return " ".join(sentences[i] for i in chosen)


def python_check(code: str) -> Dict[str, Any]:
    """Check Python source for syntax errors WITHOUT executing it (compile-only).

    A safe sanity check for code the coder role drafts: it compiles to bytecode and reports the
    first :class:`SyntaxError` (message + line), never running a single statement."""
    try:
        compile(code or "", "<nyxara-check>", "exec")
    except SyntaxError as exc:
        return {"ok": False, "error": exc.msg, "line": exc.lineno}
    except (ValueError, TypeError) as exc:           # e.g. null bytes
        return {"ok": False, "error": str(exc), "line": None}
    return {"ok": True, "error": "", "line": None}


def _pack_tool_specs() -> Dict[str, ToolSpec]:
    return {
        "summarize_text": ToolSpec(
            "summarize_text", handler=summarize_text,
            description="extractive summary of a long text (no LLM) — returns its key sentences",
            params=[ToolParam("text", "str", True),
                    ToolParam("max_sentences", "int", False, 3)],
            capability=Capability.TOOL_CALL, risk=RiskTier.TRIVIAL, reversible=True),
        "python_check": ToolSpec(
            "python_check", handler=python_check,
            description="check Python source for syntax errors without executing it",
            params=[ToolParam("code", "str", True)],
            capability=Capability.TOOL_CALL, risk=RiskTier.TRIVIAL, reversible=True),
    }


# --------------------------------------------------------------------------- #
# Tool pack
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ToolPack:
    """A named role: existing default tools it relies on + any tools it contributes."""

    name: str
    description: str
    tool_names: Tuple[str, ...] = ()           # default tools this role uses (gated as usual)
    extra: Tuple[ToolSpec, ...] = ()           # pack-specific tools to register

    def register(self, registry: ToolRegistry) -> List[str]:
        """Register this pack's own tools onto ``registry`` (skipping any already present)."""
        present = set(registry.names())
        added: List[str] = []
        for spec in self.extra:
            if spec.name not in present:
                registry.register(spec)
                added.append(spec.name)
        return added

    def catalog(self, registry: ToolRegistry) -> List[str]:
        """The tools this role offers that are actually available on ``registry``."""
        present = set(registry.names())
        names = list(self.tool_names) + [s.name for s in self.extra]
        seen: set = set()
        return [n for n in names if n in present and not (n in seen or seen.add(n))]


# --------------------------------------------------------------------------- #
# The standard packs
# --------------------------------------------------------------------------- #
def build_domain_packs() -> Dict[str, ToolPack]:
    """The three standard roles, composed from the default toolset + pack tools."""
    specs = _pack_tool_specs()
    return {
        "researcher": ToolPack(
            "researcher",
            "find, read, ground and condense information",
            tool_names=("web_search", "web_fetch", "read_document",
                        "knowledge_ingest", "knowledge_search"),
            extra=(specs["summarize_text"],)),
        "coder": ToolPack(
            "coder",
            "read, write, sanity-check and run code (sandboxed, gated)",
            tool_names=("read_file", "write_file", "list_dir", "run_python", "run_shell"),
            extra=(specs["python_check"],)),
        "maker": ToolPack(
            "maker",
            "produce artefacts — images, speech, files",
            tool_names=("generate_image", "synthesize_speech", "write_file"),
            extra=()),
    }


def register_packs(registry: ToolRegistry,
                   names: Optional[Tuple[str, ...]] = None) -> Dict[str, List[str]]:
    """Register the named packs' tools onto ``registry`` (all packs when ``names`` is None).

    Returns a map of pack name → the tool names it newly registered. Idempotent: a tool already
    on the registry is left as-is, so calling this after ``build_default_tools`` only adds the
    genuinely new pack tools."""
    packs = build_domain_packs()
    chosen = names or tuple(packs)
    out: Dict[str, List[str]] = {}
    for name in chosen:
        pack = packs.get(name)
        if pack is not None:
            out[name] = pack.register(registry)
    return out
