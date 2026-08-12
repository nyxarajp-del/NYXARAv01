"""NYXARA · growth/explorer.py — the Infinite Explorer (🧭, Rule 4: capability without bound).

Every model has the same hard ceiling: its training data. Past the cutoff it simply does
not know. A *sovereign* mind does not stop at that wall — it **generates its own data**.
This is **Environment-Driven Learning**: when NYXARA meets a task that is not in her
knowledge, she does not abstain and move on. She:

    1. RECALL    — first check whether she has already solved this (skills + knowledge base);
                   if so, reuse the learned method instead of re-deriving it.
    2. RESEARCH  — when online, scrape the live web / papers via the AutonomousResearcher for
                   hints, and (autonomously, per the Master's mandate) ``pip install`` an
                   obvious missing dependency named in the task.
    3. SYNTHESIZE— write a Python script that solves the task. With a real LLM she writes free
                   code; with no model she falls back to the Capability Foundry's deterministic
                   recipe synthesis, so the loop still produces genuinely-working code offline.
    4. RUN       — execute that script in the isolated ``-I`` code sandbox (wall-clock bounded).
    5. DEBUG     — if it fails, read the **real traceback**, feed it back into synthesis, and
                   try again — a true write → run → read-error → revise loop, not a placeholder.
    6. PERSIST   — once it works, distil the working logic into a permanent PROCEDURAL skill
                   and ingest an explanation + the verified code into the KnowledgeBase, so the
                   next time the answer is instant. The knowledge is hers, permanently.

This module is the *closed loop* that ties NYXARA's existing primitives together; it
re-implements none of them. It reuses :func:`~nyxara.agency.code_sandbox.run_python` /
:func:`~nyxara.agency.code_sandbox.safe_shell`, the Capability Foundry's
:func:`~nyxara.growth.capability_foundry.best_recipe` and static-safety lists,
:class:`~nyxara.growth.researcher.AutonomousResearcher`,
:class:`~nyxara.growth.skill_memory.SkillMemory` and
:class:`~nyxara.knowledge.base.KnowledgeBase`.

Safety is inherited, never weakened. Every candidate script is statically screened against
the foundry's allow/deny lists (no OS/network/eval, no reach into ``nyxara.kernel`` or the
permission/identity core) **before** it runs, and it only ever runs inside the sandbox.
The character core is off-limits to both the screen and to persistence: capability may grow
without bound — loyalty, corrigibility and the Master's primacy never change. The whole loop
fails as *data* (it never raises into the cognitive cycle) and is halted by a paused/scrammed
oversight gate exactly like every other autonomous faculty.

Pure standard library plus in-repo imports; the LLM is an *injected* dependency so the module
stays keyless and fully offline-capable by default.
"""

from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from nyxara.agency.code_sandbox import run_python, safe_shell
from nyxara.growth.capability_foundry import (
    _FORBIDDEN_ATTR_ROOTS,
    _FORBIDDEN_NAMES,
    _FORBIDDEN_SUBSTRINGS,
    _IMPORT_ALLOWLIST,
    best_recipe,
)

__all__ = ["ExploreResult", "InfiniteExplorer"]

# delimiters used to round-trip the verified code through a persisted Skill's procedure text,
# so a recalled skill can be re-run to answer instantly without re-deriving anything.
_CODE_OPEN = "<<<NYXARA_SOLUTION_CODE>>>"
_CODE_CLOSE = "<<<END_SOLUTION_CODE>>>"

# small stop-set so chit-chat never triggers the (expensive) bootstrapping loop
_CHITCHAT = {
    "hi", "hello", "hey", "yo", "sup", "thanks", "thank", "ok", "okay", "bye",
    "goodbye", "cool", "nice", "lol", "yes", "no", "yeah", "nah", "please",
}
# verbs / cues that mark a stimulus as a *solvable task* rather than conversation.
# Every entry names an OPERATION she could write code for. "what is the" used to sit here and
# does not name one: it is how most factual questions open, so it pulled "what is the capital of
# France?" into a write-run-debug loop. Nothing is lost by its absence — the computational
# questions it used to catch open the same way and match on their operation instead
# ("what is the SUM of…", "what is the FACTORIAL of…").
_TASK_CUES = (
    "compute", "calculate", "convert", "reverse", "count", "sort", "find", "solve",
    "encode", "decode", "hash", "generate", "parse", "extract", "sum", "average",
    "factorial", "fibonacci", "prime", "how many", "write", "build",
    "make a", "list the", "transform", "translate", "format",
)


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class ExploreResult:
    """One self-bootstrapping pass over a task — always data, success or failure."""

    task: str
    solved: bool = False
    answer: Any = None
    code: str = ""
    rounds: int = 0
    origin: str = "offline"          # "recall" | "offline" | "llm"
    sources: List[str] = field(default_factory=list)
    learned_skill: bool = False
    learned_kb: bool = False
    verified: bool = False           # answer matched a known-good example (offline path)
    debug_trace: List[str] = field(default_factory=list)
    installed: List[str] = field(default_factory=list)
    error: Optional[str] = None
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task, "solved": self.solved, "answer": self.answer,
            "code": self.code, "rounds": self.rounds, "origin": self.origin,
            "sources": list(self.sources), "learned_skill": self.learned_skill,
            "learned_kb": self.learned_kb, "verified": self.verified,
            "debug_trace": list(self.debug_trace), "installed": list(self.installed),
            "error": self.error, "elapsed_ms": round(self.elapsed_ms, 1),
        }


# --------------------------------------------------------------------------- #
# The Infinite Explorer
# --------------------------------------------------------------------------- #
class InfiniteExplorer:
    """Autonomous tool-augmented self-bootstrapping: solve the unknown, then learn it.

    Parameters
    ----------
    llm:          injected LLM facade (``.generate(prompt)`` or callable). ``None`` ⇒ the
                  deterministic offline recipe path is used for synthesis.
    researcher:   :class:`~nyxara.growth.researcher.AutonomousResearcher` for web hints.
    knowledge:    :class:`~nyxara.knowledge.base.KnowledgeBase` for permanent learning.
    skills:       :class:`~nyxara.growth.skill_memory.SkillMemory` for permanent skills.
    memory:       optional MemoryStore (unused directly; reserved for future episodic notes).
    tools:        optional ToolRegistry (reserved; research already flows through it).
    settings:     optional config object; read for ``features.web_access`` and explorer knobs.
    """

    def __init__(self, *, llm: Any = None, researcher: Any = None, knowledge: Any = None,
                 skills: Any = None, memory: Any = None, tools: Any = None,
                 settings: Any = None, max_debug_rounds: int = 4, step_timeout_s: float = 8.0,
                 allow_network: bool = True, allow_pkg_install: bool = True) -> None:
        self.llm = llm
        self.researcher = researcher
        self.knowledge = knowledge
        self.skills = skills
        self.memory = memory
        self.tools = tools
        self.settings = settings
        self.max_debug_rounds = max(1, int(max_debug_rounds))
        self.step_timeout_s = max(0.5, float(step_timeout_s))
        self.allow_network = bool(allow_network) and self._web_allowed(settings)
        self.allow_pkg_install = bool(allow_pkg_install)
        self._reports: List[ExploreResult] = []

    # ------------------------------------------------------------------ #
    # Public surface
    # ------------------------------------------------------------------ #
    def can_attempt(self, task: str) -> bool:
        """True when ``task`` is a concrete, solvable-shaped request (not chit-chat)."""
        t = (task or "").strip()
        if len(t) < 5:
            return False
        low = t.lower()
        words = re.findall(r"[a-z0-9']+", low)
        if not words:
            return False
        # pure greeting / acknowledgement → not a task
        if len(words) <= 3 and all(w in _CHITCHAT for w in words):
            return False
        if any(cue in low for cue in _TASK_CUES):
            return True
        # A question mark is NOT a task cue. It marks the Master *asking*, and this is a loop that
        # writes and runs code — the two are unrelated. Treating them as the same thing sent every
        # factual question ("what is the capital of France?") through research + synthesis, whose
        # artefacts then persisted (below) and came back on later turns as her answer.
        # Whether the foundry can name a real recipe is the honest test, and it is already the
        # next line — a genuine task reaches it and still qualifies.
        try:
            return best_recipe(t).key != "generic"
        except Exception:  # noqa: BLE001
            return False

    def explore(self, task: str) -> ExploreResult:
        """Run the full self-bootstrapping loop for ``task``. Always returns data."""
        t0 = time.monotonic()
        result = ExploreResult(task=task)
        try:
            # 1. RECALL — reuse a previously-learned solution instead of re-deriving it
            recalled = self._recall(task)
            if recalled is not None:
                recalled.elapsed_ms = (time.monotonic() - t0) * 1000.0
                self._reports.append(recalled)
                return recalled

            # 2. RESEARCH — gather hints + autonomously install an obvious dependency
            hints, sources = self._research(task)
            result.sources = sources
            result.installed = self._maybe_install(task)

            # 3-5. SYNTHESIZE → RUN → DEBUG loop
            recipe = self._safe_best_recipe(task)
            self._solve_loop(task, hints, recipe, result)

            # 6. PERSIST — learn the working logic permanently
            if result.solved:
                self._persist(task, result, hints)
        except Exception as exc:  # noqa: BLE001 — the loop never raises into the cycle
            result.error = f"{type(exc).__name__}: {exc}"
        result.elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._reports.append(result)
        return result

    def all_reports(self) -> List[ExploreResult]:
        return list(self._reports)

    def report(self) -> Dict[str, Any]:
        solved = [r for r in self._reports if r.solved]
        return {"explorations": len(self._reports), "solved": len(solved),
                "skills_bootstrapped": sum(1 for r in self._reports if r.learned_skill)}

    # ------------------------------------------------------------------ #
    # 1. RECALL
    # ------------------------------------------------------------------ #
    def _recall(self, task: str) -> Optional[ExploreResult]:
        """If a learned skill already solves this task, re-run its code for a fresh answer."""
        if self.skills is None:
            return None
        try:
            hits = self.skills.recall(task, k=3)
        except Exception:  # noqa: BLE001
            return None
        for skill in hits or []:
            code = self._extract_code(getattr(skill, "procedure", "") or "")
            if not code:
                continue
            ok, _reason = self._screen(code)
            if not ok:
                continue
            res = run_python(code, timeout_s=self.step_timeout_s, allow_network=False)
            if res.ok and res.value is not None:
                try:
                    skill.reinforce()
                except Exception:  # noqa: BLE001
                    pass
                return ExploreResult(task=task, solved=True, answer=res.value, code=code,
                                     rounds=0, origin="recall")
        return None

    # ------------------------------------------------------------------ #
    # 2. RESEARCH + autonomous dependency install
    # ------------------------------------------------------------------ #
    def _research(self, task: str) -> Tuple[str, List[str]]:
        """Scrape the live web for hints (online only). Returns (hint_text, source_urls)."""
        if self.researcher is None or not self.allow_network:
            return "", []
        try:
            report = self.researcher.research(task)
        except Exception:  # noqa: BLE001 — research is a best-effort booster, never required
            return "", []
        claims = list(getattr(report, "key_claims", []) or [])
        summary = getattr(report, "summary", "") or ""
        sources = list(getattr(report, "sources", []) or [])
        hint = (summary + ("\n- " + "\n- ".join(claims[:4]) if claims else "")).strip()
        return hint[:1200], sources[:5]

    def _maybe_install(self, task: str) -> List[str]:
        """Autonomously ``pip install`` a dependency the task explicitly names (online only).

        Per the Master's mandate the install runs without per-call approval, but it is still
        bounded by the sandboxed shell and only fires for a clearly-named, well-formed package.
        """
        if not (self.allow_pkg_install and self.allow_network):
            return []
        pkg = self._named_dependency(task)
        if pkg is None:
            return []
        try:
            res = safe_shell(["pip", "install", "--quiet", pkg],
                             timeout_s=max(20.0, self.step_timeout_s * 3))
            return [pkg] if res.ok else []
        except Exception:  # noqa: BLE001
            return []

    @staticmethod
    def _named_dependency(task: str) -> Optional[str]:
        """Extract a single, well-formed package name explicitly named in the task."""
        low = (task or "").lower()
        m = re.search(r"\bpip install\s+([a-z0-9][a-z0-9._-]{1,40})", low)
        if not m:
            m = re.search(r"\b(?:using|with|via)\s+the\s+([a-z0-9][a-z0-9._-]{1,40})\s+"
                          r"(?:library|package|module)\b", low)
        if not m:
            return None
        pkg = m.group(1).strip(".-_")
        # conservative: a real distribution name, never a flag or path
        return pkg if re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,40}", pkg) else None

    # ------------------------------------------------------------------ #
    # 3-5. SYNTHESIZE → RUN → DEBUG
    # ------------------------------------------------------------------ #
    def _solve_loop(self, task: str, hints: str, recipe: Any, result: ExploreResult) -> None:
        use_llm = self._llm_usable()
        last_err = ""
        for rnd in range(1, self.max_debug_rounds + 1):
            if use_llm:
                code = self._llm_code(task, hints, last_err)
                origin = "llm"
            else:
                code = self._recipe_code(recipe)
                origin = "offline"
            result.rounds = rnd
            result.origin = origin
            if not code:
                last_err = "synthesis produced no code"
                result.debug_trace.append(f"round {rnd}: {last_err}")
                if not use_llm:
                    break
                continue

            ok, reason = self._screen(code)
            if not ok:
                last_err = f"safety screen rejected the script: {reason}"
                result.debug_trace.append(f"round {rnd}: {last_err}")
                if not use_llm:
                    break          # the deterministic recipe won't change on retry
                continue

            res = run_python(code, timeout_s=self.step_timeout_s,
                             allow_network=self.allow_network if origin == "llm" else False)
            if res.ok and res.value is not None:
                result.solved = True
                result.answer = res.value
                result.code = code
                result.verified = self._verify_offline(recipe, res.value) if origin == "offline" \
                    else False
                return
            # read the REAL error and feed it back into the next synthesis round
            last_err = (res.error or "").strip() or self._tail(res.stderr) or "no result produced"
            result.debug_trace.append(f"round {rnd}: {self._tail(last_err)}")
            if not use_llm:
                break              # offline path is deterministic — one shot, no debugging gain
        result.error = result.error or last_err or "could not solve the task"

    # ---- LLM free-code synthesis (with error-driven repair) ---- #
    def _llm_code(self, task: str, hints: str, last_err: str) -> str:
        prompt = self._build_prompt(task, hints, last_err)
        try:
            if hasattr(self.llm, "generate"):
                raw = self.llm.generate(prompt)
            elif callable(self.llm):
                raw = self.llm(prompt)
            else:
                return ""
        except Exception:  # noqa: BLE001 — a flaky model just yields an empty round
            return ""
        return self._normalise_code(self._extract_md_code(raw))

    @staticmethod
    def _build_prompt(task: str, hints: str, last_err: str) -> str:
        parts = [
            "You are writing a small, self-contained Python script to solve a task.",
            f"TASK: {task}",
            "Rules:",
            "- Standard library only; allowed imports: " + ", ".join(sorted(_IMPORT_ALLOWLIST)) + ".",
            "- No file/network/OS/process access; no eval/exec/__import__/open.",
            "- Assign the final answer to a module-level variable named `result`"
            " (JSON-serialisable).",
            "- Respond with ONLY the code — no prose, no markdown fences.",
        ]
        if hints:
            parts.append("RESEARCH NOTES (may help, may be noisy):\n" + hints)
        if last_err:
            parts.append("Your previous attempt FAILED with this error — fix it:\n" + last_err)
        return "\n".join(parts)

    @staticmethod
    def _extract_md_code(raw: Any) -> str:
        if not isinstance(raw, str) or not raw.strip():
            return ""
        text = raw.strip()
        m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
        return (m.group(1).strip() if m else text)

    @staticmethod
    def _normalise_code(code: str) -> str:
        """Ensure the script surfaces ``result`` (wire a bare ``solve()`` through if needed)."""
        if not code:
            return ""
        if not re.search(r"(^|\n)\s*result\s*=", code) and re.search(r"\bdef\s+solve\s*\(", code):
            code = code.rstrip() + "\nresult = solve()\n"
        return code

    # ---- deterministic offline synthesis (Capability Foundry recipe) ---- #
    def _recipe_code(self, recipe: Any) -> str:
        """Turn a foundry recipe into a runnable script that surfaces ``result``.

        The ``generic`` recipe is refused. It is the foundry's honest last-resort placeholder —
        ``handle(payload)`` returns ``{'echo': payload, 'scaffold': True}`` — and it is the
        fallback for *any* task no real recipe matches. Run here it always succeeds and always
        matches its own identity example, so :meth:`_solve_loop` marked the task ``solved`` and
        :meth:`_verify_offline` marked it ``verified``, for every task in existence. That wrote
        memories reading "bootstrapped a working solution … verified against a known-good
        example" whose answer was the echo scaffold, and a claim of *verified* is exactly what
        outranks an ordinary answer downstream (``nyx001/fusion.py``'s control law).

        Echoing an argument proves the sandbox ran. It proves nothing about the task, so there is
        no code here to return: the caller's existing no-code path records an honest failure and
        the turn keeps its abstention. Recipes that actually match a task are unaffected.
        """
        if recipe is None or getattr(recipe, "key", "") == "generic":
            return ""
        args = {}
        examples = getattr(recipe, "examples", None) or []
        if examples:
            args = examples[0].get("args", {})
        return (getattr(recipe, "source", "") + "\n\nimport json as _nyx_json\n"
                f"_nyx_args = _nyx_json.loads({json.dumps(args)!r})\n"
                "result = handle(**_nyx_args)\n")

    def _verify_offline(self, recipe: Any, value: Any) -> bool:
        examples = getattr(recipe, "examples", None) or []
        if not examples or "expect" not in examples[0]:
            return False
        expect = examples[0]["expect"]
        if isinstance(expect, bool) or isinstance(value, bool):
            return value == expect
        if isinstance(expect, (int, float)) and isinstance(value, (int, float)):
            return abs(float(value) - float(expect)) <= 1e-9
        return value == expect

    def _safe_best_recipe(self, task: str) -> Any:
        try:
            return best_recipe(task)
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------ #
    # 6. PERSIST — permanent learning
    # ------------------------------------------------------------------ #
    def _persist(self, task: str, result: ExploreResult, hints: str) -> None:
        summary = self._summarise(task, result)
        # 6a. permanent PROCEDURAL skill (survives restart via the MemoryStore)
        if self.skills is not None:
            try:
                procedure = (f"{summary}\n{_CODE_OPEN}\n{result.code}\n{_CODE_CLOSE}")
                self.skills.learn(goal=task, procedure=procedure, tools=["run_python"])
                result.learned_skill = True
            except Exception:  # noqa: BLE001
                pass
        # 6b. permanent SEMANTIC knowledge (RAG-retrievable; survives restart)
        if self.knowledge is not None:
            try:
                src_line = ("\nSources: " + ", ".join(result.sources)) if result.sources else ""
                text = (f"How to solve: {task}\n{summary}\n\nVerified code:\n{result.code}"
                        f"{src_line}")
                self.knowledge.ingest_text(text, source=f"explorer:{task[:48]}")
                result.learned_kb = True
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _summarise(task: str, result: ExploreResult) -> str:
        ans = result.answer
        ans_s = ans if isinstance(ans, str) else repr(ans)
        if len(ans_s) > 160:
            ans_s = ans_s[:160] + "…"
        verified = " (verified against a known-good example)" if result.verified else ""
        return (f"NYXARA bootstrapped a working solution for {task!r} in {result.rounds} "
                f"round(s) via the {result.origin} path{verified}; example answer: {ans_s}.")

    # ------------------------------------------------------------------ #
    # Safety — static screen (reuses the Capability Foundry's allow/deny lists)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _screen(code: str) -> Tuple[bool, str]:
        """Reject any script that reaches for the OS/network/eval or the sovereign core."""
        if not code or not code.strip():
            return False, "empty script"
        for bad in _FORBIDDEN_SUBSTRINGS:
            if bad in code:
                return False, f"forbidden token {bad!r}"
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return False, f"syntax error: {exc}"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if alias.name not in _IMPORT_ALLOWLIST and root not in _IMPORT_ALLOWLIST:
                        return False, f"import {alias.name!r} not allowed"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod not in _IMPORT_ALLOWLIST and mod.split(".")[0] not in _IMPORT_ALLOWLIST:
                    return False, f"import-from {mod!r} not allowed"
            elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
                return False, f"forbidden name {node.id!r}"
            elif isinstance(node, ast.Attribute):
                rootn: Any = node
                while isinstance(rootn, ast.Attribute):
                    rootn = rootn.value
                if isinstance(rootn, ast.Name) and rootn.id in _FORBIDDEN_ATTR_ROOTS:
                    return False, f"forbidden module access {rootn.id!r}"
        return True, "clean"

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_code(procedure: str) -> str:
        m = re.search(re.escape(_CODE_OPEN) + r"\s*(.*?)\s*" + re.escape(_CODE_CLOSE),
                      procedure or "", re.S)
        return m.group(1).strip() if m else ""

    def _llm_usable(self) -> bool:
        """True when a real (non-mock) model is wired, or an injected scripted LLM is present."""
        if self.llm is None:
            return False
        try:
            return self.llm.chosen_provider().name != "native"
        except AttributeError:
            # an injected scripted/callable LLM with no provider machinery → use it directly
            return hasattr(self.llm, "generate") or callable(self.llm)
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _web_allowed(settings: Any) -> bool:
        if settings is None:
            return True
        try:
            return bool(settings.features.web_access)
        except Exception:  # noqa: BLE001
            return True

    @staticmethod
    def _tail(text: str, limit: int = 240) -> str:
        text = (text or "").strip()
        return text if len(text) <= limit else "…" + text[-limit:]


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.growth.skill_memory import SkillMemory
    from nyxara.knowledge.base import KnowledgeBase
    from nyxara.memory.store import MemoryStore

    print("=" * 70)
    print("NYXARA infinite-explorer self-test")
    print("=" * 70)

    store = MemoryStore()
    skills = SkillMemory(store=store)
    kb = KnowledgeBase(store=store, name="explorer")

    # ---- 1. OFFLINE solve: no LLM, no network — deterministic recipe synthesis ---- #
    explorer = InfiniteExplorer(llm=None, researcher=None, knowledge=kb, skills=skills,
                                allow_network=False, allow_pkg_install=False)
    assert explorer.can_attempt("reverse the string hello")
    assert not explorer.can_attempt("hi")
    r = explorer.explore("reverse a string")
    print(f"\noffline solve       : solved={r.solved} origin={r.origin} "
          f"answer={r.answer!r} verified={r.verified}")
    assert r.solved and r.origin == "offline" and r.answer == "cba" and r.verified
    assert r.learned_skill and r.learned_kb

    # ---- 2. PERMANENT learning: the skill + kb chunk are retrievable ---- #
    assert skills.recall("reverse a string")
    assert kb.retrieve("reverse a string", k=2)
    print("permanent learning  : skill + knowledge-base chunk persisted ✓")

    # ---- 3. RECALL short-circuit: a second pass reuses the learned method ---- #
    r2 = explorer.explore("reverse a string")
    print(f"recall short-circuit: origin={r2.origin} answer={r2.answer!r}")
    assert r2.origin == "recall" and r2.solved and r2.answer == "cba"

    # ---- 4. DEBUG loop: a scripted LLM returns broken code, then fixed code ---- #
    class _ScriptedLLM:
        """Offline stand-in for a frontier coder: first a crash, then a correct fix."""

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt: str, **kw: Any) -> str:
            self.calls += 1
            if self.calls == 1:
                return "result = 1 / 0\n"                       # ZeroDivisionError
            assert "previous attempt FAILED" in prompt          # the real error was fed back
            return "result = sum(range(11))\n"                  # 0+1+...+10 = 55

    dbg_skills = SkillMemory()
    dbg = InfiniteExplorer(llm=_ScriptedLLM(), knowledge=KnowledgeBase(name="dbg"),
                           skills=dbg_skills, allow_network=False, allow_pkg_install=False)
    rd = dbg.explore("compute the sum of 0..10")
    print(f"\ndebug loop          : solved={rd.solved} rounds={rd.rounds} "
          f"answer={rd.answer!r} trace={rd.debug_trace}")
    assert rd.solved and rd.rounds >= 2 and rd.answer == 55
    assert any("ZeroDivision" in t or "round 1" in t for t in rd.debug_trace)

    # ---- 5. SAFETY: a script reaching for the OS is screened out, never run ---- #
    ok, reason = InfiniteExplorer._screen("import os\nresult = os.listdir('/')\n")
    print(f"\nsafety screen       : rejected={not ok} ({reason})")
    assert not ok

    ok2, _ = InfiniteExplorer._screen("result = 2 + 2\n")
    assert ok2

    # ---- 6. GRACEFUL degradation: every dependency None → data, never a crash ---- #
    bare = InfiniteExplorer(llm=None, researcher=None, knowledge=None, skills=None,
                            allow_network=False, allow_pkg_install=False)
    rb = bare.explore("compute the factorial of 5")
    print(f"\ndegradation         : solved={rb.solved} answer={rb.answer!r} "
          f"(no skill/kb to persist into)")
    assert rb.solved and rb.answer == 120 and not rb.learned_skill

    print(f"\nreport              : {explorer.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
