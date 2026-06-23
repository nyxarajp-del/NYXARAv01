"""NYXARA · agency/autonomous_tool_forge.py — self-correcting permanent tool forging (✦, Rule 4).

NYXARA's three self-extension faculties each solved part of the problem:

* :mod:`nyxara.agency.dynamic_tool_creator` writes a throwaway program (synth → run →
  *delete*) — single-use, never kept.
* :mod:`nyxara.growth.capability_foundry` forges a permanent tool (plan → code → test →
  benchmark → gauntlet → deploy) — but synthesises **once**: a buggy first draft fails.
* :mod:`nyxara.growth.explorer` runs the write → run → read-traceback → revise loop — but
  persists a *skill*, not a registry tool.

This module unifies all three into one capability the Master asked for: when NYXARA meets a
task she has **no tool for**, she writes the code, runs it in the isolated sandbox, **reads
the real traceback, fixes her own error, and tries again** — a genuine self-debugging loop
— and on success **permanently deploys** the new tool into her registry *and* records the
winning procedure as a :class:`~nyxara.growth.skill_memory.Skill`, so the capability is hers
forever and is recalled into future reasoning.

It is glue, not reinvention: synthesis + safety scan + sandbox testing + benchmarking +
deployment all reuse :class:`~nyxara.growth.capability_foundry.CapabilityFoundry`; the
network-disabled sandbox is :func:`~nyxara.agency.code_sandbox.run_python`. Per the Master's
mandate forged tools deploy **autonomously** (no escalation), but they are always clamped to
:class:`~nyxara.agency.permissions.Capability.TOOL_CALL` / ``RiskTier.LOW`` and every call
still runs through the static gauntlet + the sandbox — the kernel's gates, character lock and
corrigibility are never weakened (Rule 4 / Rule 8). Best-effort and offline-capable: with no
LLM it falls back to the foundry's deterministic recipe synthesis; nothing here ever raises
into the cognitive cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nyxara.agency.permissions import Authority, Capability, RiskTier
from nyxara.kernel.config import NyxaraSettings, get_settings

__all__ = ["AutonomousToolForge", "ForgeOutcome"]


@dataclass
class ForgeOutcome:
    """The result of one autonomous forge attempt — fully auditable."""

    ok: bool
    need: str
    tool_name: Optional[str] = None
    deployed: bool = False
    attempts: int = 0
    origin: str = "template"             # "llm" | "template"
    source: str = ""
    skill_id: Optional[str] = None
    requires_owner: bool = False
    reason: str = ""
    test_passed: bool = False
    benchmark: Dict[str, Any] = field(default_factory=dict)
    fixes: List[str] = field(default_factory=list)   # the tracebacks she debugged through

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "need": self.need, "tool_name": self.tool_name,
                "deployed": self.deployed, "attempts": self.attempts, "origin": self.origin,
                "skill_id": self.skill_id, "requires_owner": self.requires_owner,
                "reason": self.reason, "test_passed": self.test_passed,
                "benchmark": self.benchmark, "fixes": self.fixes}


class AutonomousToolForge:
    """Write → test → self-fix → permanently deploy a brand-new tool, then remember it."""

    def __init__(self, *, registry: Any, llm: Any = None, skill_memory: Any = None,
                 settings: Optional[NyxaraSettings] = None, foundry: Any = None,
                 max_fix_attempts: Optional[int] = None,
                 autonomous_deploy: Optional[bool] = None,
                 clamp_low: Optional[bool] = None,
                 test_timeout_s: Optional[float] = None) -> None:
        self.settings = settings or get_settings()
        cfg = self.settings.tool_forge
        self.registry = registry
        self.llm = llm
        self.skill_memory = skill_memory
        self._foundry = foundry
        self.max_fix_attempts = int(cfg.max_fix_attempts if max_fix_attempts is None
                                    else max_fix_attempts)
        self.autonomous_deploy = bool(cfg.autonomous_deploy if autonomous_deploy is None
                                      else autonomous_deploy)
        self.clamp_low = bool(cfg.clamp_low if clamp_low is None else clamp_low)
        self.test_timeout_s = float(cfg.test_timeout_s if test_timeout_s is None
                                    else test_timeout_s)

    # ------------------------------------------------------------------ #
    # public entry point
    # ------------------------------------------------------------------ #
    def forge(self, need: str, *, examples: Optional[List[Dict[str, Any]]] = None,
              params: Optional[List[Any]] = None,
              authority: Authority = Authority.AUTONOMOUS) -> ForgeOutcome:
        """Forge, self-debug, and permanently deploy a tool for *need*. Never raises."""
        try:
            return self._forge(need, examples=examples, params=params, authority=authority)
        except Exception as exc:  # noqa: BLE001 — forging must never crash the cognitive cycle
            return ForgeOutcome(ok=False, need=need, reason=f"forge crashed: {exc}")

    def _forge(self, need: str, *, examples: Optional[List[Dict[str, Any]]],
               params: Optional[List[Any]], authority: Authority) -> ForgeOutcome:
        foundry = self._foundry_obj()
        if foundry is None:
            return ForgeOutcome(ok=False, need=need, reason="capability foundry unavailable")
        plan = foundry.plan(need)
        if params is not None:
            plan.params = list(params)
        if examples:
            plan.examples = list(examples)
        if self.clamp_low:
            plan.capability = Capability.TOOL_CALL
            plan.risk = RiskTier.LOW
            plan.reversible = True

        fixes: List[str] = []
        # 1. the self-debugging LLM loop: write → run → read traceback → revise → retry
        if self._real_llm():
            feedback, last_source = "", ""
            for attempt in range(self.max_fix_attempts):
                source = self._synthesize(plan, feedback, last_source)
                if not source:
                    break
                ok, reason = foundry._scan_source(source)  # noqa: SLF001 — shared safety scan
                if not ok:
                    feedback = f"Your code failed the static safety scan: {reason}. " \
                               f"Use only allowed stdlib and define a pure handle()."
                    last_source = source
                    fixes.append(f"safety: {reason}")
                    continue
                report = foundry.test(plan, source)
                if report.ok:
                    out = self._deploy(foundry, plan, source, "llm", attempt + 1, authority)
                    out.fixes = fixes
                    if out.ok:
                        return out
                    feedback = out.reason
                    last_source = source
                    fixes.append(f"deploy: {out.reason}")
                    continue
                errs = "; ".join(report.failures[:4]) or "unknown failure"
                feedback = ("Your tool failed its tests. The sandbox reported:\n" + errs +
                            "\nFix the bug and return the corrected handle() only.")
                last_source = source
                fixes.append(errs)

        # 2. offline / exhausted: the foundry's deterministic recipe synthesis (single shot)
        res = foundry.forge(need, authority=authority)
        attempts = (self.max_fix_attempts if self._real_llm() else 0) + 1
        if getattr(res, "deployed", False):
            skill_id = self._record_skill(need, res.tool_name, None)
            return ForgeOutcome(ok=True, need=need, tool_name=res.tool_name, deployed=True,
                                attempts=attempts, origin=getattr(res, "origin", "template"),
                                skill_id=skill_id, reason="forged via deterministic recipe",
                                test_passed=True, fixes=fixes)
        return ForgeOutcome(ok=False, need=need, tool_name=getattr(res, "tool_name", None),
                            attempts=attempts, origin=getattr(res, "origin", "template"),
                            requires_owner=bool(getattr(res, "requires_owner", False)),
                            reason=getattr(res, "reason", "forge failed"), fixes=fixes)

    # ------------------------------------------------------------------ #
    # deploy a tested source + remember it as a skill
    # ------------------------------------------------------------------ #
    def _deploy(self, foundry: Any, plan: Any, source: str, origin: str,
                attempts: int, authority: Authority) -> ForgeOutcome:
        from nyxara.growth.capability_foundry import ForgedTool

        if authority is not Authority.OWNER and not self.autonomous_deploy:
            return ForgeOutcome(ok=False, need=plan.need, tool_name=plan.tool_name,
                                attempts=attempts, origin=origin, requires_owner=True,
                                test_passed=True, reason="autonomous deploy disabled")
        bench = foundry.benchmark(plan, source)
        if bench.pass_rate < foundry.benchmark_min_score:
            return ForgeOutcome(ok=False, need=plan.need, tool_name=plan.tool_name,
                                attempts=attempts, origin=origin, test_passed=True,
                                benchmark=bench.to_dict(),
                                reason=f"benchmark pass-rate {bench.pass_rate:.2f} too low")
        forged = ForgedTool(
            version=foundry._next_version(), name=plan.tool_name, source=source,  # noqa: SLF001
            plan=plan.to_dict(), capability=plan.capability.value, risk=plan.risk.label,
            reversible=plan.reversible, test_passed=True, benchmark=bench.to_dict(),
            origin=origin)
        try:
            foundry.deploy(plan, forged, authority=authority)
        except Exception as exc:  # noqa: BLE001 — AuthorizationError or any deploy failure
            requires_owner = exc.__class__.__name__ == "AuthorizationError"
            return ForgeOutcome(ok=False, need=plan.need, tool_name=plan.tool_name,
                                attempts=attempts, origin=origin, test_passed=True,
                                benchmark=bench.to_dict(), requires_owner=requires_owner,
                                reason=f"deploy refused: {exc}")
        # mirror forge()'s bookkeeping so the artifact survives restarts & teaches the ranker
        try:
            foundry.forged.append(forged)
            foundry.recipe_learner.observe(plan.need, plan.shape)
            foundry._save_manifest()  # noqa: SLF001
        except Exception:  # noqa: BLE001
            pass
        skill_id = self._record_skill(plan.need, plan.tool_name, source)
        return ForgeOutcome(ok=True, need=plan.need, tool_name=plan.tool_name, deployed=True,
                            attempts=attempts, origin=origin, source=source,
                            skill_id=skill_id, test_passed=True, benchmark=bench.to_dict(),
                            reason="forged, self-tested and deployed")

    def _record_skill(self, need: str, tool_name: Optional[str],
                      source: Optional[str]) -> Optional[str]:
        if self.skill_memory is None or not tool_name:
            return None
        try:
            procedure = (f"invoke the forged tool {tool_name!r} (written, sandbox-tested and "
                         f"deployed by NYXARA herself) to accomplish this")
            skill = self.skill_memory.learn(goal=need, procedure=procedure, tools=[tool_name])
            return getattr(skill, "skill_id", None)
        except Exception:  # noqa: BLE001 — skill capture is best-effort
            return None

    # ------------------------------------------------------------------ #
    # synthesis with traceback feedback (the "fix your own error" step)
    # ------------------------------------------------------------------ #
    def _synthesize(self, plan: Any, feedback: str, last_source: str) -> Optional[str]:
        from nyxara.growth.capability_foundry import _IMPORT_ALLOWLIST  # noqa: SLF001
        sig = ", ".join(p.name for p in plan.params) or "**kwargs"
        prompt = [
            "Write a single pure Python function for a sandboxed tool.",
            f"Capability: {plan.description}",
            f"Required signature: def handle({sig}):",
            "Hard rules:",
            "- Define ONLY the function `handle` (plus stdlib imports it needs).",
            f"- Pure standard library only; allowed imports: {', '.join(sorted(_IMPORT_ALLOWLIST))}.",
            "- No file/network/OS/process access; no eval/exec/__import__/open.",
            "- Return a JSON-serialisable value.",
        ]
        if plan.examples:
            import json as _json
            prompt.append("It MUST satisfy these examples (args -> expected):")
            for ex in plan.examples[:5]:
                prompt.append(f"  handle(**{_json.dumps(ex.get('args', {}))}) "
                              f"== {_json.dumps(ex.get('expect'))}")
        if last_source:
            prompt.append("\nYour previous attempt:\n" + last_source)
        if feedback:
            prompt.append("\nIt was rejected:\n" + feedback)
        prompt.append("\nRespond with ONLY the corrected code, no prose, no markdown fences.")
        try:
            raw = self.llm.generate("\n".join(prompt))
        except Exception:  # noqa: BLE001
            return None
        from nyxara.growth.capability_foundry import CapabilityFoundry
        return CapabilityFoundry._extract_code(raw)  # noqa: SLF001

    # ------------------------------------------------------------------ #
    # collaborators / probes
    # ------------------------------------------------------------------ #
    def _foundry_obj(self) -> Any:
        if self._foundry is None:
            try:
                from nyxara.growth.capability_foundry import CapabilityFoundry
                self._foundry = CapabilityFoundry(
                    registry=self.registry, llm=self.llm,
                    test_timeout_s=self.test_timeout_s,
                    allow_autonomous_deploy=self.autonomous_deploy)
            except Exception:  # noqa: BLE001
                self._foundry = False
        return self._foundry or None

    def _real_llm(self) -> bool:
        if self.llm is None:
            return False
        try:
            return self.llm.chosen_provider().name != "mock"
        except Exception:  # noqa: BLE001 — a scripted fake LLM is treated as real for testing
            return hasattr(self.llm, "generate")


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.agency.tools import ToolParam, ToolRegistry
    from nyxara.growth.skill_memory import SkillMemory

    print("=" * 70)
    print("NYXARA autonomous-tool-forge self-test (write → self-fix → deploy → remember)")
    print("=" * 70)

    class _BuggyThenFixedLLM:
        """First emits a buggy reverse(), reads the failure, then ships the correct one."""

        def __init__(self) -> None:
            self.n = 0

        def generate(self, prompt, *, system=None, **kw):
            self.n += 1
            if self.n == 1:                       # bug: reverses by word, not character
                return "def handle(text):\n    return ' '.join(text.split()[::-1])\n"
            return "def handle(text):\n    return text[::-1]\n"

    reg = ToolRegistry()
    skills = SkillMemory()
    forge = AutonomousToolForge(registry=reg, llm=_BuggyThenFixedLLM(), skill_memory=skills)

    outcome = forge.forge(
        "reverse the characters of a string",
        params=[ToolParam("text", type="str", required=True)],
        examples=[{"args": {"text": "abc"}, "expect": "cba"},
                  {"args": {"text": "nyxara"}, "expect": "araxyn"}])

    print(f"\nok            : {outcome.ok}")
    print(f"tool_name     : {outcome.tool_name}")
    print(f"attempts      : {outcome.attempts}  (self-fixed: {outcome.fixes})")
    print(f"deployed      : {outcome.deployed}  origin={outcome.origin}")
    print(f"skill_id      : {outcome.skill_id}")
    assert outcome.ok and outcome.deployed
    assert outcome.attempts >= 2                  # the first buggy draft was debugged
    assert outcome.fixes                          # a real traceback/diff was read and fixed

    # the forged tool is now a first-class, invokable registry tool
    res = reg.invoke(outcome.tool_name, {"text": "hello"}, authority=Authority.OWNER)
    print(f"\ninvoke forged : ok={res.ok} value={res.value!r}")
    assert res.ok and res.value == "olleh"

    # and the winning procedure was remembered as a recallable skill
    recalled = skills.recall("reverse", k=3)
    print(f"recalled skill: {[s.goal for s in recalled]}")
    assert recalled and any(outcome.tool_name in s.tools for s in recalled)

    # offline (no LLM): falls back to the deterministic recipe path, still deploys something
    reg2 = ToolRegistry()
    off = AutonomousToolForge(registry=reg2, llm=None, skill_memory=SkillMemory())
    o2 = off.forge("compute the length of a string",
                   params=[ToolParam("text", type="str", required=True)])
    print(f"\noffline forge : ok={o2.ok} tool={o2.tool_name!r} origin={o2.origin}")
    assert o2.ok and o2.deployed

    print("\nALL SELF-TESTS PASSED ✓")
