"""NYXARA · growth/causal_code_engine.py — the Causal Code Engine (🧬, Stage D).

The self-debugger already repairs a failing test under a reversible gauntlet — but its *isolation*
was shallow: it mapped ``tests/<pkg>/test_x.py`` to ``nyxara/<pkg>/x.py`` by **filename**, so when a
test failed because of a bug in a *different* module it imports, the debugger aimed at the wrong file.

This module adds the missing step the Master asked for — **analyse the causal tree of the fault, then
mutate the sub-routine that actually caused it**:

* **Causal tree from the real traceback.** The failing test is re-run and its traceback parsed into the
  ordered chain of ``nyxara/…`` frames that actually executed. The **deepest non-test frame is the
  proximate cause** (where the error surfaced); shallower frames are contributory; the namesake module
  is kept as a fallback. This is genuine cause→effect structure, not a filename guess.
* **Causal-model reranking (cross-session).** Each diagnosis is recorded into the shared
  :class:`~nyxara.mind.causal_world_model.CausalWorldModel` (implicated modules as events preceding the
  failure), so over time ``why(fail)`` learns which module's edits genuinely precede which failures and
  reranks the candidates — necessity/sufficiency, not coincidence.
* **Gated repair of the causal root.** The chosen root module is handed to the existing
  :class:`~nyxara.growth.self_debugger.SelfDebugger` fix path — the *same* byte-for-byte reversible
  gauntlet + improvement proof + the Rule-8 constitutional lock. Nothing here weakens a single gate; it
  only aims the existing repair at the right file.

Honest scope: no literal C/Rust JIT of arbitrary Python and no new execution privilege — "sub-routine
mutation" means a gated, reversible source edit to the causally-implicated module (the real
"bytecode synthesis" NYXARA already has is the DSL kernel compiler in ``growth/codegen.py``). Pure
standard library; **no LLM** is required in the causal analysis (self-authored edits remain the
debugger's own triple-gated, off-by-default option).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nyxara.growth.self_debugger import FixAttempt, SelfDebugger, TestFailure

__all__ = ["CausalNode", "CausalDiagnosis", "CausalCodeEngine"]

# Matches a repo-relative nyxara source frame in a pytest traceback, e.g.
#   nyxara/growth/foo.py:123: in bar
#   File "/abs/path/nyxara/growth/foo.py", line 123, in bar
_FRAME_RE = re.compile(r'(nyxara/[A-Za-z0-9_./]+\.py)(?::(\d+)|"?,?\s*line\s*(\d+))?')


@dataclass
class CausalNode:
    """One module on the fault's causal path, with why it is implicated and how strongly."""

    module: str
    role: str                 # "proximate" | "contributory" | "namesake"
    depth: int                # position in the executed chain (0 = deepest / proximate)
    score: float              # ranking score in [0, 1]
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"module": self.module, "role": self.role, "depth": self.depth,
                "score": round(self.score, 4), "rationale": self.rationale}


@dataclass
class CausalDiagnosis:
    """The ranked causal tree of a single failure and the root chosen for repair."""

    nodeid: str
    root_module: Optional[str] = None
    tree: List[CausalNode] = field(default_factory=list)
    confidence: float = 0.0
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"nodeid": self.nodeid, "root_module": self.root_module,
                "confidence": round(self.confidence, 4), "note": self.note,
                "tree": [n.to_dict() for n in self.tree]}


class CausalCodeEngine:
    """Diagnose the causal root of a failing test, then repair it under the existing gauntlet."""

    def __init__(self, *, core: Any = None, debugger: Optional[SelfDebugger] = None,
                 causal_model: Any = None, root: Any = None) -> None:
        self.core = core
        self.debugger = debugger or SelfDebugger.from_core(core, root=root)
        self.root = self.debugger.root
        # a shared, possibly-persisted causal model when the core has one; else a private one
        self.causal = causal_model or getattr(core, "causal_world_model", None)
        if self.causal is None:
            try:
                from nyxara.mind.causal_world_model import CausalWorldModel
                self.causal = CausalWorldModel()
            except Exception:  # noqa: BLE001 — causal reranking is an enhancement, never required
                self.causal = None
        self.diagnoses: List[CausalDiagnosis] = []

    # ------------------------------------------------------------------ #
    # Causal tree
    # ------------------------------------------------------------------ #
    def _frames(self, text: str) -> List[str]:
        """The ordered, de-duplicated nyxara source frames named in a traceback (top→bottom)."""
        seen: Dict[str, None] = {}
        for m in _FRAME_RE.finditer(text or ""):
            mod = m.group(1)
            if mod not in seen:
                seen[mod] = None
        return list(seen)

    def _capture_traceback(self, nodeid: str) -> str:
        """Re-run one test in isolation and return its raw output (with the real traceback)."""
        try:
            _rc, out = self.debugger._pytest(nodeid)  # reuse the debugger's own runner
            return out or ""
        except Exception:  # noqa: BLE001
            return ""

    def causal_tree(self, failure: TestFailure, traceback_text: str = "") -> CausalDiagnosis:
        """Build the ranked causal tree for ``failure`` from its traceback + the causal model."""
        diag = CausalDiagnosis(nodeid=failure.nodeid)
        text = "\n".join(x for x in (failure.message, traceback_text) if x)
        frames = [f for f in self._frames(text) if not _is_test_path(f)]
        namesake = failure.source_module

        # the traceback is top→bottom; the LAST nyxara frame is where the error surfaced (proximate).
        nodes: List[CausalNode] = []
        if frames:
            n = len(frames)
            for i, mod in enumerate(frames):
                depth = (n - 1) - i                    # 0 = deepest / proximate
                if depth == 0:
                    role, base = "proximate", 1.0
                else:
                    role, base = "contributory", max(0.2, 0.7 - 0.1 * depth)
                nodes.append(CausalNode(module=mod, role=role, depth=depth, score=base,
                                        rationale=("error surfaced here (deepest executed frame)"
                                                   if depth == 0 else
                                                   f"on the call path, {depth} frame(s) above the fault")))
        # always keep the namesake module as an honest fallback candidate
        if namesake and not any(nd.module == namesake for nd in nodes):
            nodes.append(CausalNode(module=namesake, role="namesake", depth=len(nodes),
                                    score=0.4, rationale="test↔module filename mapping (fallback)"))

        # cross-session causal reranking: boost a module the model has learned precedes this failure
        self._rerank_with_model(failure, nodes)
        nodes.sort(key=lambda nd: nd.score, reverse=True)
        diag.tree = nodes
        if nodes:
            diag.root_module = nodes[0].module
            diag.confidence = nodes[0].score
            diag.note = f"{len(nodes)} candidate(s); root = {nodes[0].role}"
        else:
            diag.note = "no nyxara frame in the traceback and no namesake — cannot localise"
        # record this episode so the causal model learns for next time (best-effort)
        self._observe_episode(failure, [nd.module for nd in nodes])
        self.diagnoses.append(diag)
        return diag

    def _rerank_with_model(self, failure: TestFailure, nodes: List[CausalNode]) -> None:
        if self.causal is None or not nodes:
            return
        try:
            links = self.causal.why(f"fail:{failure.nodeid}")
        except Exception:  # noqa: BLE001
            return
        boost: Dict[str, float] = {}
        for lk in links or []:
            cause = getattr(lk, "cause", "")
            if cause.startswith("module:") and getattr(lk, "is_causal", False):
                boost[cause[len("module:"):]] = float(getattr(lk, "confidence", 0.0))
        for nd in nodes:
            if nd.module in boost:
                # a learned causal precedence lifts the candidate (bounded, never past 1.0)
                nd.score = min(1.0, nd.score + 0.3 * boost[nd.module])
                nd.rationale += f" · causal-model boost {boost[nd.module]:.0%}"

    def _observe_episode(self, failure: TestFailure, modules: List[str]) -> None:
        if self.causal is None or not modules:
            return
        try:
            now = time.time()
            # implicated modules occur first (interventions — she is about to edit them)…
            self.causal.observe_snapshot([f"module:{m}" for m in modules],
                                         interventions=[f"module:{m}" for m in modules], at=now)
            # …then the failure event, a hair later, so precedence (edit → fail) is learnable
            self.causal.observe(f"fail:{failure.nodeid}", at=now + 1e-3)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    # Gated repair of the causal root
    # ------------------------------------------------------------------ #
    def repair(self, failure: TestFailure, traceback_text: str = "") -> Tuple[CausalDiagnosis, FixAttempt]:
        """Diagnose the causal root, then repair THAT module through the existing gauntlet."""
        diag = self.causal_tree(failure, traceback_text)
        if not diag.root_module:
            return diag, FixAttempt(nodeid=failure.nodeid, module=None,
                                    reason="no causal root could be localised")
        # aim the existing (reversible, gated, Rule-8-locked) fix at the causal root, not the namesake
        targeted = TestFailure(nodeid=failure.nodeid, test_file=failure.test_file,
                               message=failure.message, reproduced=failure.reproduced,
                               source_module=diag.root_module)
        attempt = self.debugger.fix(targeted)
        attempt.reason = f"[causal root: {diag.root_module} ({diag.tree[0].role})] {attempt.reason}"
        return diag, attempt

    def run(self, *, test_path: Optional[str] = None, max_fixes: int = 3) -> Dict[str, Any]:
        """Full pass: detect failures, causally diagnose each, and repair the root under the gauntlet."""
        failures = self.debugger.detect(test_path=test_path)
        self.debugger.isolate(failures)
        results: List[Dict[str, Any]] = []
        for f in failures[:max(0, int(max_fixes))]:
            if not f.reproduced:
                continue
            tb = self._capture_traceback(f.nodeid)
            diag, attempt = self.repair(f, tb)
            results.append({"diagnosis": diag.to_dict(), "fix": attempt.to_dict()})
        return {
            "detected": len(failures),
            "repaired": sum(1 for r in results if r["fix"].get("kept")),
            "enacted": self.debugger.autonomous_enact,
            "results": results,
        }

    def report(self) -> Dict[str, Any]:
        return {"diagnoses": [d.to_dict() for d in self.diagnoses[-16:]]}


def _is_test_path(module: str) -> bool:
    m = module.replace("\\", "/")
    return "/tests/" in f"/{m}" or m.startswith("tests/") or "/test_" in f"/{m}"
