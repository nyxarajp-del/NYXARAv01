"""NYXARA · growth/capability_foundry.py — the Capability Foundry (Level 15, ✦ Rule 4).

Rule 4 is recursive self-evolution: NYXARA's *capability* may grow without bound — never
her *character or loyalty*. The toolsmith (``agency/toolsmith.py``) grows new tools by
**composing existing ones**; the model foundry (``growth/foundry.py``) grows her **own
model weights**. This module fills the remaining gap: when a capability is *missing
entirely* (e.g. "image skill missing"), NYXARA **designs a brand-new module for herself**
end-to-end —

    1. plan       — turn the gap into a typed :class:`CapabilityPlan` (name, params, examples)
    2. synthesize — write the actual Python source of a ``handle(...)`` function
                    (LLM-backed when a model is available, deterministic template otherwise)
    3. test       — run the generated source in the isolated code sandbox against examples
    4. benchmark  — measure latency + pass-rate over the examples
    5. gauntlet   — refuse anything that touches the sovereign core (Rules / permissions /
                    identity) or that statically reaches for the OS/network/eval
    6. deploy     — register it as a first-class :class:`~nyxara.agency.tools.ToolSpec`
                    whose handler runs the generated source **in the sandbox on every call**

The deployed tool is never ``exec``'d in-process: each invocation runs the stored source
through :func:`~nyxara.agency.code_sandbox.run_python` (network-disabled, wall-clock
bounded) — fail-closed defence in depth, fully reversible. Autonomously-forged tools are
always clamped to :class:`~nyxara.agency.permissions.Capability.TOOL_CALL` / ``LOW`` risk;
anything that would need more privilege is refused unless the Master installs it
(``Authority.OWNER``) — the same loyalty gate as the toolsmith (Rule 8).

Pure standard library (plus in-repo imports); the LLM is an *injected* callable so the
module stays keyless and offline by default.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from nyxara.agency.code_sandbox import run_python
from nyxara.agency.permissions import Authority, Capability, RiskTier
from nyxara.agency.tools import ToolParam, ToolRegistry, ToolSpec
from nyxara.kernel.errors import AuthorizationError, ToolError

__all__ = [
    "ForgeStage",
    "CapabilityPlan",
    "TestReport",
    "CapBenchmarkReport",
    "ForgedTool",
    "ForgeResult",
    "CapabilityFoundry",
]


# --------------------------------------------------------------------------- #
# Pipeline stages
# --------------------------------------------------------------------------- #
class ForgeStage(str, Enum):
    PLAN = "plan"
    SYNTHESIZE = "synthesize"
    TEST = "test"
    BENCHMARK = "benchmark"
    GAUNTLET = "gauntlet"
    DEPLOY = "deploy"
    DONE = "done"
    REFUSED = "refused"
    FAILED = "failed"


# --------------------------------------------------------------------------- #
# Static-safety allow/deny lists (the "cannot grow a back-door tool" guarantee)
# --------------------------------------------------------------------------- #
_IMPORT_ALLOWLIST = {
    "math", "hashlib", "base64", "json", "re", "string", "itertools",
    "functools", "urllib", "urllib.parse", "datetime", "statistics",
    "decimal", "fractions", "textwrap", "collections",
}
_FORBIDDEN_NAMES = {
    "eval", "exec", "__import__", "open", "compile", "input", "exit", "quit",
    "breakpoint", "memoryview",
}
_FORBIDDEN_ATTR_ROOTS = {
    "os", "sys", "subprocess", "socket", "shutil", "ctypes", "importlib",
    "builtins", "pathlib", "io", "pickle", "marshal", "threading",
    "multiprocessing", "signal",
}
_FORBIDDEN_SUBSTRINGS = (
    "__import__", "__subclasses__", "__globals__", "__builtins__", "__bases__",
    "__class__", "__mro__", "nyxara.kernel", "nyxara.agency.permissions",
)
# Capabilities NYXARA may never autonomously forge into existence.
_PRIVILEGED_CAPS = {
    Capability.MODIFY_RULES, Capability.MODIFY_PERMISSIONS, Capability.MODIFY_IDENTITY,
    Capability.SELF_MODIFY, Capability.PROC_EXEC, Capability.CODE_EXEC,
    Capability.FS_DELETE, Capability.FS_WRITE, Capability.PKG_INSTALL,
    Capability.ACCOUNT_MODIFY, Capability.SECRETS_ACCESS, Capability.SPAWN_AGENT,
}


# --------------------------------------------------------------------------- #
# Plan / artifact / result
# --------------------------------------------------------------------------- #
@dataclass
class CapabilityPlan:
    """The intent → spec: what tool to build and how to prove it works."""
    need: str
    tool_name: str
    shape: str                                   # template/recipe key
    description: str
    params: List[ToolParam] = field(default_factory=list)
    capability: Capability = Capability.TOOL_CALL
    risk: RiskTier = RiskTier.LOW
    reversible: bool = True
    examples: List[Dict[str, Any]] = field(default_factory=list)   # {"args":..,"expect":..}
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"need": self.need, "tool_name": self.tool_name, "shape": self.shape,
                "description": self.description,
                "params": [p.to_dict() for p in self.params],
                "capability": self.capability.value, "risk": self.risk.label,
                "reversible": self.reversible, "examples": self.examples,
                "rationale": self.rationale}


@dataclass
class TestReport:
    """Outcome of running generated source against the plan's examples."""
    passed: int = 0
    total: int = 0
    failures: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.total > 0 and self.passed == self.total

    def to_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed, "total": self.total, "ok": self.ok,
                "failures": self.failures}


@dataclass
class CapBenchmarkReport:
    """Latency + correctness measured over repeated sandboxed runs."""
    runs: int = 0
    pass_rate: float = 0.0
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"runs": self.runs, "pass_rate": round(self.pass_rate, 4),
                "mean_latency_ms": round(self.mean_latency_ms, 2),
                "p95_latency_ms": round(self.p95_latency_ms, 2)}


@dataclass
class ForgedTool:
    """A versioned, persisted artifact: the generated source + its provenance."""
    version: int
    name: str
    source: str
    plan: Dict[str, Any]
    capability: str = Capability.TOOL_CALL.value
    risk: str = RiskTier.LOW.label
    reversible: bool = True
    created_at: float = field(default_factory=time.time)
    test_passed: bool = False
    benchmark: Dict[str, Any] = field(default_factory=dict)
    deployed: bool = False
    origin: str = "template"                     # "llm" | "template"
    path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"version": self.version, "name": self.name, "source": self.source,
                "plan": self.plan, "capability": self.capability, "risk": self.risk,
                "reversible": self.reversible, "created_at": self.created_at,
                "test_passed": self.test_passed, "benchmark": self.benchmark,
                "deployed": self.deployed, "origin": self.origin, "path": self.path}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ForgedTool":
        return cls(version=int(d["version"]), name=d["name"], source=d["source"],
                   plan=d.get("plan", {}), capability=d.get("capability", "tool.call"),
                   risk=d.get("risk", "low"), reversible=bool(d.get("reversible", True)),
                   created_at=float(d.get("created_at", 0.0)),
                   test_passed=bool(d.get("test_passed", False)),
                   benchmark=d.get("benchmark", {}), deployed=bool(d.get("deployed", False)),
                   origin=d.get("origin", "template"), path=d.get("path", ""))


@dataclass
class ForgeResult:
    """The outcome of the whole forge() pipeline."""
    need: str
    stage: ForgeStage
    deployed: bool = False
    tool_name: Optional[str] = None
    version: Optional[int] = None
    test_passed: bool = False
    benchmark_score: float = 0.0
    reason: str = ""
    requires_owner: bool = False
    origin: str = "template"
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.deployed

    def to_dict(self) -> Dict[str, Any]:
        return {"need": self.need, "stage": self.stage.value, "deployed": self.deployed,
                "ok": self.ok, "tool_name": self.tool_name, "version": self.version,
                "test_passed": self.test_passed,
                "benchmark_score": round(self.benchmark_score, 4),
                "reason": self.reason, "requires_owner": self.requires_owner,
                "origin": self.origin, "elapsed_ms": round(self.elapsed_ms, 1)}


# --------------------------------------------------------------------------- #
# Deterministic recipe library — genuinely-working tools, not stubs
# --------------------------------------------------------------------------- #
@dataclass
class _Recipe:
    key: str
    keywords: Tuple[str, ...]
    description: str
    params: List[ToolParam]
    source: str
    examples: List[Dict[str, Any]]


_RECIPES: List[_Recipe] = [
    _Recipe("text_reverse", ("reverse",),
            "reverse a string",
            [ToolParam("text", "str")],
            "def handle(text):\n    return text[::-1]\n",
            [{"args": {"text": "abc"}, "expect": "cba"}]),
    _Recipe("text_upper", ("uppercase", "upper case", "to upper"),
            "uppercase a string",
            [ToolParam("text", "str")],
            "def handle(text):\n    return text.upper()\n",
            [{"args": {"text": "abc"}, "expect": "ABC"}]),
    _Recipe("text_lower", ("lowercase", "lower case", "to lower"),
            "lowercase a string",
            [ToolParam("text", "str")],
            "def handle(text):\n    return text.lower()\n",
            [{"args": {"text": "ABC"}, "expect": "abc"}]),
    _Recipe("text_title", ("titlecase", "title case", "capitalize each"),
            "title-case a string",
            [ToolParam("text", "str")],
            "def handle(text):\n    return text.title()\n",
            [{"args": {"text": "hello world"}, "expect": "Hello World"}]),
    _Recipe("text_word_count", ("word count", "count words", "number of words"),
            "count the words in a string",
            [ToolParam("text", "str")],
            "def handle(text):\n    return len(text.split())\n",
            [{"args": {"text": "one two three"}, "expect": 3}]),
    _Recipe("text_slugify", ("slug", "slugify", "url slug"),
            "slugify a string into a url-safe token",
            [ToolParam("text", "str")],
            ("import re\n"
             "def handle(text):\n"
             "    s = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')\n"
             "    return s\n"),
            [{"args": {"text": "Hello, World!"}, "expect": "hello-world"}]),
    _Recipe("math_factorial", ("factorial",),
            "compute the factorial of n",
            [ToolParam("n", "int")],
            ("def handle(n):\n"
             "    n = int(n)\n"
             "    r = 1\n"
             "    for i in range(2, n + 1):\n"
             "        r *= i\n"
             "    return r\n"),
            [{"args": {"n": 5}, "expect": 120}]),
    _Recipe("math_gcd", ("gcd", "greatest common divisor"),
            "compute the greatest common divisor of a and b",
            [ToolParam("a", "int"), ToolParam("b", "int")],
            ("def handle(a, b):\n"
             "    a, b = int(a), int(b)\n"
             "    while b:\n"
             "        a, b = b, a % b\n"
             "    return abs(a)\n"),
            [{"args": {"a": 12, "b": 18}, "expect": 6}]),
    _Recipe("math_is_prime", ("is prime", "prime check", "primality"),
            "test whether n is prime",
            [ToolParam("n", "int")],
            ("def handle(n):\n"
             "    n = int(n)\n"
             "    if n < 2:\n"
             "        return False\n"
             "    i = 2\n"
             "    while i * i <= n:\n"
             "        if n % i == 0:\n"
             "            return False\n"
             "        i += 1\n"
             "    return True\n"),
            [{"args": {"n": 7}, "expect": True},
             {"args": {"n": 8}, "expect": False}]),
    _Recipe("math_fibonacci", ("fibonacci", "fib "),
            "compute the nth Fibonacci number (0-indexed)",
            [ToolParam("n", "int")],
            ("def handle(n):\n"
             "    n = int(n)\n"
             "    a, b = 0, 1\n"
             "    for _ in range(n):\n"
             "        a, b = b, a + b\n"
             "    return a\n"),
            [{"args": {"n": 10}, "expect": 55}]),
    _Recipe("encode_base64", ("base64", "base 64"),
            "base64-encode a string",
            [ToolParam("text", "str")],
            ("import base64\n"
             "def handle(text):\n"
             "    return base64.b64encode(text.encode('utf-8')).decode('ascii')\n"),
            [{"args": {"text": "hi"}, "expect": "aGk="}]),
    _Recipe("encode_hex", ("hex encode", "to hex", "hexadecimal"),
            "hex-encode a string",
            [ToolParam("text", "str")],
            "def handle(text):\n    return text.encode('utf-8').hex()\n",
            [{"args": {"text": "hi"}, "expect": "6869"}]),
    _Recipe("encode_url", ("url encode", "urlencode", "percent encode"),
            "url-encode a string",
            [ToolParam("text", "str")],
            ("import urllib.parse\n"
             "def handle(text):\n"
             "    return urllib.parse.quote(text)\n"),
            [{"args": {"text": "a b"}, "expect": "a%20b"}]),
    _Recipe("encode_json", ("json pretty", "pretty print json", "format json"),
            "pretty-print a JSON-serialisable value",
            [ToolParam("data", "any")],
            ("import json\n"
             "def handle(data):\n"
             "    return json.dumps(data, indent=2, sort_keys=True)\n"),
            [{"args": {"data": {"b": 1, "a": 2}}, "expect": '{\n  "a": 2,\n  "b": 1\n}'}]),
    _Recipe("hash_sha256", ("sha256", "sha-256", "sha 256"),
            "compute the SHA-256 hex digest of a string",
            [ToolParam("text", "str")],
            ("import hashlib\n"
             "def handle(text):\n"
             "    return hashlib.sha256(text.encode('utf-8')).hexdigest()\n"),
            [{"args": {"text": "abc"},
              "expect": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}]),
    _Recipe("hash_sha1", ("sha1", "sha-1"),
            "compute the SHA-1 hex digest of a string",
            [ToolParam("text", "str")],
            ("import hashlib\n"
             "def handle(text):\n"
             "    return hashlib.sha1(text.encode('utf-8')).hexdigest()\n"),
            [{"args": {"text": "abc"},
              "expect": "a9993e364706816aba3e25717850c26c9cd0d89d"}]),
    _Recipe("hash_md5", ("md5",),
            "compute the MD5 hex digest of a string",
            [ToolParam("text", "str")],
            ("import hashlib\n"
             "def handle(text):\n"
             "    return hashlib.md5(text.encode('utf-8')).hexdigest()\n"),
            [{"args": {"text": "abc"}, "expect": "900150983cd24fb0d6963f7d28e17f72"}]),
    # generic, always-available scaffold — runs, passes its identity example, and is
    # honestly labelled a placeholder so NYXARA never over-claims a capability she only
    # scaffolded. (For an unknown gap like "image generation" the LLM path fills the body;
    # the scaffold guarantees the pipeline still completes with a deployable, honest tool.)
    _Recipe("generic", (),
            "scaffolded capability (echoes its input — refine with a real implementation)",
            [ToolParam("payload", "any", required=False, default=None)],
            ("def handle(payload=None):\n"
             "    return {'echo': payload, 'scaffold': True}\n"),
            [{"args": {"payload": "x"}, "expect": {"echo": "x", "scaffold": True}}]),
]
_RECIPE_BY_KEY = {r.key: r for r in _RECIPES}


# --------------------------------------------------------------------------- #
# The Capability Foundry
# --------------------------------------------------------------------------- #
class CapabilityFoundry:
    """Forge brand-new, tested, benchmarked, deployable tools from capability gaps.

    Parameters
    ----------
    registry:            the :class:`ToolRegistry` newly-forged tools are deployed onto.
    capability_registry: optional :class:`~nyxara.growth.capability.CapabilityRegistry`
                         (the gap detector); calibrated on a successful forge.
    llm:                 optional callable / object with ``.generate(prompt)`` for
                         LLM-backed code generation. ``None`` ⇒ deterministic templates.
    root:                where versioned artifacts are persisted. Defaults to
                         ``<settings.paths.data_dir>/capability_foundry``.
    test_timeout_s:      wall-clock budget per sandboxed run.
    """

    def __init__(self, *, registry: ToolRegistry,
                 capability_registry: Any = None,
                 llm: Any = None,
                 root: Any = None,
                 test_timeout_s: float = 5.0,
                 allow_autonomous_deploy: bool = True,
                 benchmark_min_score: float = 1.0,
                 benchmark_repeats: int = 3) -> None:
        self.registry = registry
        self.capabilities = capability_registry
        self.llm = llm
        self.test_timeout_s = max(0.5, float(test_timeout_s))
        self.allow_autonomous_deploy = bool(allow_autonomous_deploy)
        self.benchmark_min_score = float(benchmark_min_score)
        self.benchmark_repeats = max(1, int(benchmark_repeats))
        self.root = self._resolve_root(root)
        self.forged: List[ForgedTool] = []
        self._load_manifest()
        self._redeploy_persisted()

    # ------------------------------------------------------------------ #
    # 1. PLAN
    # ------------------------------------------------------------------ #
    def plan(self, need: str) -> CapabilityPlan:
        """Classify the gap into a recipe and a typed, registry-safe plan."""
        recipe = self._classify(need)
        base = recipe.key if recipe.key != "generic" else self._slug(need)
        tool_name = self._unique_name(base)
        return CapabilityPlan(
            need=need, tool_name=tool_name, shape=recipe.key,
            description=recipe.description,
            params=list(recipe.params),
            capability=Capability.TOOL_CALL, risk=RiskTier.LOW, reversible=True,
            examples=list(recipe.examples),
            rationale=f"classified gap {need!r} as recipe {recipe.key!r}")

    # ------------------------------------------------------------------ #
    # 2. SYNTHESIZE
    # ------------------------------------------------------------------ #
    def synthesize(self, plan: CapabilityPlan) -> Tuple[str, str]:
        """Return ``(source, origin)`` for the handler. LLM first, template fallback."""
        if self.llm is not None:
            src = self._llm_synthesize(plan)
            if src is not None:
                ok, _reason = self._scan_source(src)
                if ok:
                    return src, "llm"
        return _RECIPE_BY_KEY.get(plan.shape, _RECIPE_BY_KEY["generic"]).source, "template"

    # ------------------------------------------------------------------ #
    # 3. TEST
    # ------------------------------------------------------------------ #
    def test(self, plan: CapabilityPlan, source: str) -> TestReport:
        """Run the generated source against each example in the isolated sandbox."""
        examples = plan.examples or [{"args": self._default_args(plan), "expect": _SMOKE}]
        report = TestReport(total=len(examples))
        for ex in examples:
            args = ex.get("args", {})
            res = run_python(self._call_snippet(source, args),
                             timeout_s=self.test_timeout_s, allow_network=False)
            if not res.ok:
                report.failures.append(f"args={args} -> error: {res.error}")
                continue
            expect = ex.get("expect", _SMOKE)
            if expect is _SMOKE or self._compare(res.value, expect):
                report.passed += 1
            else:
                report.failures.append(f"args={args} -> got {res.value!r}, want {expect!r}")
        return report

    # ------------------------------------------------------------------ #
    # 4. BENCHMARK
    # ------------------------------------------------------------------ #
    def benchmark(self, plan: CapabilityPlan, source: str) -> CapBenchmarkReport:
        """Measure latency + correctness over repeated sandboxed runs."""
        examples = plan.examples or [{"args": self._default_args(plan), "expect": _SMOKE}]
        latencies: List[float] = []
        passes = 0
        runs = 0
        for ex in examples:
            args = ex.get("args", {})
            expect = ex.get("expect", _SMOKE)
            for _ in range(self.benchmark_repeats):
                start = time.monotonic()
                res = run_python(self._call_snippet(source, args),
                                 timeout_s=self.test_timeout_s, allow_network=False)
                latencies.append((time.monotonic() - start) * 1000.0)
                runs += 1
                if res.ok and (expect is _SMOKE or self._compare(res.value, expect)):
                    passes += 1
        latencies.sort()
        mean = sum(latencies) / len(latencies) if latencies else 0.0
        p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else 0.0
        return CapBenchmarkReport(runs=runs, pass_rate=(passes / runs if runs else 0.0),
                                  mean_latency_ms=mean, p95_latency_ms=p95)

    # ------------------------------------------------------------------ #
    # 5. GAUNTLET (static safety — fail-closed)
    # ------------------------------------------------------------------ #
    def gauntlet(self, plan: CapabilityPlan, source: str, *,
                 authority: Authority = Authority.AUTONOMOUS) -> Tuple[bool, str]:
        """Refuse anything that touches the sovereign core or statically reaches the OS."""
        # 1. capability tier — privileged caps are owner-only.
        meta = self.registry.policy.meta(plan.capability)
        privileged = plan.capability in _PRIVILEGED_CAPS or getattr(meta, "owner_exclusive", False)
        if privileged and authority is not Authority.OWNER:
            return False, (f"capability {plan.capability.value!r} touches the sovereign core; "
                           "only the Master may forge it")
        if plan.risk >= RiskTier.CRITICAL and authority is not Authority.OWNER:
            return False, "critical-risk forge requires the Master"
        # 2. static source scan.
        ok, reason = self._scan_source(source)
        if not ok:
            return False, f"unsafe source: {reason}"
        return True, "clean"

    # ------------------------------------------------------------------ #
    # 6. DEPLOY
    # ------------------------------------------------------------------ #
    def deploy(self, plan: CapabilityPlan, forged: ForgedTool, *,
               authority: Authority = Authority.AUTONOMOUS) -> ToolSpec:
        """Persist the artifact and register a sandbox-per-call ToolSpec."""
        ok, reason = self.gauntlet(plan, forged.source, authority=authority)
        if not ok:
            raise AuthorizationError(reason, context={"tool": plan.tool_name})

        forged.path = self._persist(plan, forged)
        param_names = [p.name for p in plan.params]
        handler = self._make_handler(forged.source, param_names)
        spec = ToolSpec(
            name=plan.tool_name, handler=handler,
            description=plan.description + " [forged by the Capability Foundry]",
            params=list(plan.params), capability=plan.capability, risk=plan.risk,
            reversible=plan.reversible)
        self.registry.register(spec)
        forged.deployed = True
        self._save_manifest()
        # honest self-model calibration (best-effort)
        if self.capabilities is not None:
            try:
                self.capabilities.verify(plan.tool_name, True)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
        return spec

    # ------------------------------------------------------------------ #
    # FORGE — the whole pipeline
    # ------------------------------------------------------------------ #
    def forge(self, need: str, *, authority: Authority = Authority.AUTONOMOUS) -> ForgeResult:
        """Run plan → synthesize → test → benchmark → gauntlet → deploy."""
        t0 = time.monotonic()

        def done(stage: ForgeStage, **kw: Any) -> ForgeResult:
            return ForgeResult(need=need, stage=stage,
                               elapsed_ms=(time.monotonic() - t0) * 1000.0, **kw)

        try:
            plan = self.plan(need)
        except Exception as exc:  # noqa: BLE001
            return done(ForgeStage.FAILED, reason=f"plan failed: {exc}")

        source, origin = self.synthesize(plan)

        ok, reason = self.gauntlet(plan, source, authority=authority)
        if not ok:
            return done(ForgeStage.REFUSED, tool_name=plan.tool_name, reason=reason,
                        requires_owner=(authority is not Authority.OWNER), origin=origin)

        test_report = self.test(plan, source)
        if not test_report.ok:
            return done(ForgeStage.FAILED, tool_name=plan.tool_name, origin=origin,
                        reason="tests failed: " + "; ".join(test_report.failures[:3]))

        bench = self.benchmark(plan, source)
        if bench.pass_rate < self.benchmark_min_score:
            return done(ForgeStage.FAILED, tool_name=plan.tool_name, origin=origin,
                        test_passed=True, benchmark_score=bench.pass_rate,
                        reason=f"benchmark pass-rate {bench.pass_rate:.2f} "
                               f"< {self.benchmark_min_score:.2f}")

        # autonomous deploy is allowed for safe-tier forges; the gauntlet already
        # refused anything privileged under non-owner authority.
        if authority is not Authority.OWNER and not self.allow_autonomous_deploy:
            return done(ForgeStage.GAUNTLET, tool_name=plan.tool_name, origin=origin,
                        test_passed=True, benchmark_score=bench.pass_rate,
                        requires_owner=True, reason="autonomous deploy disabled by config")

        version = self._next_version()
        forged = ForgedTool(version=version, name=plan.tool_name, source=source,
                            plan=plan.to_dict(), capability=plan.capability.value,
                            risk=plan.risk.label, reversible=plan.reversible,
                            test_passed=True, benchmark=bench.to_dict(), origin=origin)
        try:
            self.deploy(plan, forged, authority=authority)
        except AuthorizationError as exc:
            return done(ForgeStage.REFUSED, tool_name=plan.tool_name, origin=origin,
                        test_passed=True, benchmark_score=bench.pass_rate,
                        requires_owner=True, reason=str(exc))
        except Exception as exc:  # noqa: BLE001
            return done(ForgeStage.FAILED, tool_name=plan.tool_name, origin=origin,
                        test_passed=True, benchmark_score=bench.pass_rate,
                        reason=f"deploy failed: {exc}")

        self.forged.append(forged)
        self._save_manifest()
        return done(ForgeStage.DONE, deployed=True, tool_name=plan.tool_name,
                    version=version, test_passed=True, benchmark_score=bench.pass_rate,
                    origin=origin, reason="forged and deployed")

    # ------------------------------------------------------------------ #
    # Rollback / introspection
    # ------------------------------------------------------------------ #
    def rollback(self, name: str, *, authority: Authority = Authority.OWNER) -> bool:
        """Un-register a forged tool (Master-gated). The versioned source is retained."""
        if authority is not Authority.OWNER:
            raise AuthorizationError("rollback is reserved to the Master",
                                     context={"tool": name})
        removed = self.registry._tools.pop(name, None) is not None  # noqa: SLF001
        for f in self.forged:
            if f.name == name:
                f.deployed = False
        self._save_manifest()
        return removed

    def report(self) -> Dict[str, Any]:
        deployed = [f for f in self.forged if f.deployed]
        return {"forged": len(self.forged), "deployed": len(deployed),
                "root": str(self.root),
                "tools": [f.name for f in deployed]}

    # ------------------------------------------------------------------ #
    # Internals — classification & naming
    # ------------------------------------------------------------------ #
    @staticmethod
    def _classify(need: str) -> _Recipe:
        low = " " + (need or "").lower() + " "
        best: Optional[_Recipe] = None
        best_len = 0
        for recipe in _RECIPES:
            for kw in recipe.keywords:
                if kw in low and len(kw) > best_len:
                    best, best_len = recipe, len(kw)
        return best or _RECIPE_BY_KEY["generic"]

    @staticmethod
    def _slug(need: str) -> str:
        s = re.sub(r"[^a-z0-9]+", "_", (need or "").lower()).strip("_")
        s = re.sub(r"_+", "_", s)
        words = [w for w in s.split("_") if w][:4]
        return ("cap_" + "_".join(words)) if words else "cap_tool"

    def _unique_name(self, base: str) -> str:
        # Only avoid collisions with tools *currently registered* — historical artifacts
        # are versioned on disk and must not force a rename of an otherwise-free name, or a
        # reasoner that asked for `base` would never find the tool its request triggered.
        taken = set(self.registry.names())
        if base not in taken:
            return base
        i = 2
        while f"{base}_v{i}" in taken:
            i += 1
        return f"{base}_v{i}"

    @staticmethod
    def _default_args(plan: CapabilityPlan) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for p in plan.params:
            if not p.required:
                out[p.name] = p.default
            else:
                out[p.name] = {"str": "x", "int": 1, "float": 1.0, "bool": True,
                               "list": [], "dict": {}, "any": None}.get(p.type, None)
        return out

    # ------------------------------------------------------------------ #
    # Internals — sandbox execution
    # ------------------------------------------------------------------ #
    @staticmethod
    def _call_snippet(source: str, args: Dict[str, Any]) -> str:
        """Compose the program the sandbox runs: the source + a call surfacing `result`."""
        args_json = json.dumps(args)
        return (source + "\n\nimport json as _nyx_json\n"
                f"_nyx_args = _nyx_json.loads({args_json!r})\n"
                "result = handle(**_nyx_args)\n")

    def _make_handler(self, source: str, param_names: List[str]) -> Callable[..., Any]:
        """A deployed-tool handler that runs the stored source in the sandbox per call."""
        timeout = self.test_timeout_s

        def handler(**kwargs: Any) -> Any:
            res = run_python(CapabilityFoundry._call_snippet(source, kwargs),
                             timeout_s=timeout, allow_network=False)
            if not res.ok:
                raise ToolError(f"forged tool failed in sandbox: {res.error}",
                                context={"stderr": res.stderr[:500]})
            return res.value

        return handler

    @staticmethod
    def _compare(got: Any, expect: Any) -> bool:
        if isinstance(expect, bool) or isinstance(got, bool):
            return got == expect
        if isinstance(expect, (int, float)) and isinstance(got, (int, float)):
            return abs(float(got) - float(expect)) <= 1e-9
        return got == expect

    # ------------------------------------------------------------------ #
    # Internals — static safety scan
    # ------------------------------------------------------------------ #
    @staticmethod
    def _scan_source(source: str) -> Tuple[bool, str]:
        if not source or not source.strip():
            return False, "empty source"
        for bad in _FORBIDDEN_SUBSTRINGS:
            if bad in source:
                return False, f"forbidden token {bad!r}"
        try:
            tree = ast.parse(source)
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
        if not any(isinstance(n, ast.FunctionDef) and n.name == "handle" for n in tree.body):
            return False, "no top-level handle() function defined"
        return True, "clean"

    # ------------------------------------------------------------------ #
    # Internals — LLM synthesis
    # ------------------------------------------------------------------ #
    def _llm_synthesize(self, plan: CapabilityPlan) -> Optional[str]:
        sig = ", ".join(p.name for p in plan.params) or "**kwargs"
        prompt = (
            "Write a single pure Python function for a sandboxed tool.\n"
            f"Capability: {plan.description}\n"
            f"Required signature: def handle({sig}):\n"
            "Hard rules:\n"
            "- Define ONLY the function `handle` (plus stdlib imports it needs).\n"
            "- Pure standard library only; allowed imports: "
            f"{', '.join(sorted(_IMPORT_ALLOWLIST))}.\n"
            "- No file/network/OS/process access; no eval/exec/__import__/open.\n"
            "- Return a JSON-serialisable value.\n"
            "Respond with ONLY the code, no prose, no markdown fences.\n")
        try:
            if hasattr(self.llm, "generate"):
                raw = self.llm.generate(prompt)
            elif callable(self.llm):
                raw = self.llm(prompt)
            else:
                return None
        except Exception:  # noqa: BLE001 — a flaky/keyless LLM just yields the template path
            return None
        return self._extract_code(raw)

    @staticmethod
    def _extract_code(raw: Any) -> Optional[str]:
        if not isinstance(raw, str) or not raw.strip():
            return None
        text = raw.strip()
        m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
        if m:
            text = m.group(1).strip()
        return text if "def handle" in text else None

    # ------------------------------------------------------------------ #
    # Internals — persistence (mirrors growth/foundry.py)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_root(root: Any) -> Path:
        if root is not None:
            return Path(root)
        try:
            from nyxara.kernel.config import get_settings
            return Path(get_settings().paths.data_dir) / "capability_foundry"
        except Exception:  # noqa: BLE001
            return Path.home() / ".nyxara" / "data" / "capability_foundry"

    def _manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def _next_version(self) -> int:
        return (max((f.version for f in self.forged), default=0)) + 1

    def _persist(self, plan: CapabilityPlan, forged: ForgedTool) -> str:
        vdir = self.root / f"v{forged.version}"
        try:
            vdir.mkdir(parents=True, exist_ok=True)
            (vdir / "handler.py").write_text(forged.source, encoding="utf-8")
            (vdir / "plan.json").write_text(json.dumps(plan.to_dict(), indent=2),
                                            encoding="utf-8")
        except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
            return ""
        return str(vdir)

    def _save_manifest(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self._manifest_path().write_text(
                json.dumps({"forged": [f.to_dict() for f in self.forged]}, indent=2),
                encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    def _load_manifest(self) -> None:
        path = self._manifest_path()
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                self.forged = [ForgedTool.from_dict(d) for d in data.get("forged", [])]
        except Exception:  # noqa: BLE001 — a corrupt manifest never blocks boot
            self.forged = []

    def _redeploy_persisted(self) -> None:
        """Re-register previously-deployed safe-tier tools so forged capabilities survive a
        restart. Each is re-scanned (fail-closed) and skipped if privileged or already present."""
        for f in self.forged:
            if not f.deployed or self.registry.get(f.name) is not None:
                continue
            ok, _reason = self._scan_source(f.source)
            if not ok:
                continue
            try:
                cap = Capability(f.capability)
            except ValueError:
                cap = Capability.TOOL_CALL
            if cap in _PRIVILEGED_CAPS:        # never silently restore a privileged tool
                continue
            params = [ToolParam(p.get("name", ""), p.get("type", "str"),
                                bool(p.get("required", True)), p.get("default"),
                                p.get("description", ""))
                      for p in f.plan.get("params", []) if p.get("name")]
            try:
                risk = RiskTier[f.risk.upper()]
            except (KeyError, AttributeError):
                risk = RiskTier.LOW
            handler = self._make_handler(f.source, [p.name for p in params])
            spec = ToolSpec(name=f.name, handler=handler,
                            description=f.plan.get("description", "") + " [forged, restored]",
                            params=params, capability=cap, risk=risk,
                            reversible=bool(f.reversible))
            try:
                self.registry.register(spec)
            except Exception:  # noqa: BLE001 — a name clash just means it's already there
                pass


# Sentinel used when an example has no expected value (smoke "does it run?" check).
_SMOKE = object()


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import tempfile

    print("=" * 70)
    print("NYXARA capability-foundry self-test")
    print("=" * 70)

    with tempfile.TemporaryDirectory(prefix="nyxara-capfoundry-") as tmp:
        reg = ToolRegistry()
        foundry = CapabilityFoundry(registry=reg, root=tmp)

        # 1. full pipeline on a known shape, end to end
        res = foundry.forge("please add a sha256 hash skill", authority=Authority.OWNER)
        print(f"\nforge sha256        : deployed={res.deployed} tool={res.tool_name!r} "
              f"origin={res.origin} score={res.benchmark_score:.2f} ({res.elapsed_ms:.0f}ms)")
        assert res.deployed and res.test_passed and res.benchmark_score == 1.0

        # 2. the forged tool is real and invokable through the registry
        r = reg.invoke(res.tool_name, {"text": "abc"}, authority=Authority.OWNER)
        print(f"invoke forged tool  : ok={r.ok} value={r.value!r}")
        assert r.ok and r.value == hashlib.sha256(b"abc").hexdigest()

        # 3. forge a string-reverse skill too, prove independence
        res2 = foundry.forge("reverse a string", authority=Authority.OWNER)
        r2 = reg.invoke(res2.tool_name, {"text": "nyxara"}, authority=Authority.OWNER)
        print(f"invoke reverse      : ok={r2.ok} value={r2.value!r}")
        assert r2.ok and r2.value == "araxyn"

        # 4. the gauntlet refuses a tool that touches the sovereign core
        evil = foundry.plan("reverse a string")
        evil.capability = Capability.MODIFY_RULES
        ok, why = foundry.gauntlet(evil, "def handle(text):\n    return text\n",
                                   authority=Authority.AUTONOMOUS)
        print(f"sovereign-core gate : ok={ok} reason={why!r}")
        assert not ok

        # 5. the source scanner refuses an OS reach
        bad_ok, bad_why = foundry._scan_source(
            "import os\ndef handle(x):\n    return os.listdir('.')\n")
        print(f"source scan         : ok={bad_ok} reason={bad_why!r}")
        assert not bad_ok

        # 6. persistence round-trips
        reg2 = ToolRegistry()
        reloaded = CapabilityFoundry(registry=reg2, root=tmp)
        print(f"persistence reload  : forged={len(reloaded.forged)}")
        assert len(reloaded.forged) >= 2

        # 7. rollback un-registers (Master-gated)
        removed = foundry.rollback(res.tool_name, authority=Authority.OWNER)
        print(f"rollback            : removed={removed} in_registry={res.tool_name in reg.names()}")
        assert removed and res.tool_name not in reg.names()

        print(f"\nreport              : {foundry.report()}")

    print("\nALL SELF-TESTS PASSED ✓")
