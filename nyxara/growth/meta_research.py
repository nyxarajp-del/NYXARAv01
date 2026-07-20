"""NYXARA · growth/meta_research.py — Meta-Research: invent → test → integrate (🔬, Level 10d).

Reading research is not the frontier; *creating* it is. The :class:`AutonomousResearcher`
already searches, reads and summarises the web. The :class:`MetaResearcher` goes one level up:
it mines the *open / incomplete* parts of that research, **invents** candidate new theories and
optimization techniques, **tests** each as runnable code in the safe :class:`~nyxara.sim.sandbox.Sandbox`,
and — only when the Master authorises it — proposes the validated optimizations as reversible,
gauntlet-gated **source edits** that *integrate* into her own architecture.

    gather open questions → invent candidates → sandbox-test → (gated) integrate

Safety:

* **Inventing + testing** are pure measurement: candidate code runs in a restricted namespace
  (no imports, no I/O builtins) inside the sandbox, so a tested proposition can compute but can
  never touch the world.
* **Integration is double-gated**: it requires both ``settings.meta_research.allow_integration``
  AND ``settings.self_improvement.autonomous_enact`` (both OFF by default), and always routes
  through the existing :class:`~nyxara.growth.self_optimize.Optimizer` verify-or-rollback gauntlet
  — so a worse or breaking change is reverted byte-for-byte.

It degrades fully offline: with no LLM/network the deterministic heuristic inventor still poses
and verifies genuinely checkable propositions, so the loop creates and validates information in CI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["CandidateTheory", "MetaResearchReport", "MetaResearcher"]


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class CandidateTheory:
    """One invented, checkable theory or optimization technique."""

    title: str
    kind: str = "theory"                 # "theory" | "optimization"
    description: str = ""
    code: str = ""                       # runnable Python; must set `result` and `expected`
    target_file: Optional[str] = None    # for optimizations that could be integrated
    source: str = "heuristic"            # "llm" | "heuristic"
    tested: bool = False
    validated: bool = False
    experiment_result: str = ""
    integrated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"title": self.title, "kind": self.kind, "description": self.description,
                "source": self.source, "tested": self.tested, "validated": self.validated,
                "experiment_result": self.experiment_result, "integrated": self.integrated,
                "has_code": bool(self.code)}


@dataclass
class MetaResearchReport:
    """The outcome of one meta-research run on a topic."""

    topic: str
    timestamp: float = field(default_factory=time.time)
    open_questions: List[str] = field(default_factory=list)
    candidates: List[CandidateTheory] = field(default_factory=list)
    validated: int = 0
    integrated: int = 0
    laws_discovered: int = 0                       # symbolic laws corroborated this pass (empirical science)
    elapsed_ms: float = 0.0
    meta_meta: Optional[Dict[str, Any]] = None    # the recursive tower over this engine's SEARCH

    def to_dict(self) -> Dict[str, Any]:
        return {"topic": self.topic, "timestamp": self.timestamp,
                "open_questions": self.open_questions,
                "candidates": [c.to_dict() for c in self.candidates],
                "validated": self.validated, "integrated": self.integrated,
                "laws_discovered": self.laws_discovered,
                "elapsed_ms": round(self.elapsed_ms, 1), "meta_meta": self.meta_meta}


# --------------------------------------------------------------------------- #
# MetaResearcher
# --------------------------------------------------------------------------- #
class MetaResearcher:
    """Invent new theories / optimizations, verify them in a sandbox, and (gated) integrate.

    Parameters
    ----------
    researcher: an :class:`~nyxara.growth.researcher.AutonomousResearcher` to gather open
                research (optional; offline heuristic questions are used when absent).
    llm:        an LLM facade for invention (optional; heuristic fallback otherwise).
    sandbox:    a :class:`~nyxara.sim.sandbox.Sandbox` to test candidates (one is built on demand).
    memory:     a MemoryStore — validated inventions are stored as SEMANTIC memories.
    knowledge:  a KnowledgeBase (passed through to an on-demand researcher).
    settings:   :class:`~nyxara.kernel.config.NyxaraSettings`.
    journal / permissions: passed to the Optimizer when integration is authorised.
    """

    def __init__(self, *, researcher: Any = None, llm: Any = None, sandbox: Any = None,
                 memory: Any = None, knowledge: Any = None, settings: Any = None,
                 journal: Any = None, permissions: Any = None, law_discovery: Any = None) -> None:
        from nyxara.kernel.config import get_settings
        self.settings = settings or get_settings()
        self.researcher = researcher
        self.llm = llm
        self.sandbox = sandbox
        self.memory = memory
        self.knowledge = knowledge
        self.journal = journal
        self.permissions = permissions
        # optional empirical-science collaborator: a LawDiscoveryEngine so a continuous meta-research
        # pass also does real symbolic law discovery, not only code invention (best-effort, offline).
        self.law_discovery = law_discovery
        self._reports: List[MetaResearchReport] = []
        # the recursive meta tower over THIS engine's own search (built once, lazily)
        self._meta: Any = None
        self._meta_built = False

    # ---------------------------------------------------------------------- #
    # The recursive meta tower over the meta-research SEARCH (invention breadth)
    # ---------------------------------------------------------------------- #
    def _meta_meta(self) -> Any:
        """The recursive tower that evolves HOW WIDE this engine invents (built once, best-effort).

        Level 0 evolves a :class:`~nyxara.growth.meta_meta.MetaResearchGenome` (``max_candidates``);
        each higher level evolves how the level below searches — recursion at every level, scored by
        the real validated-theory yield per pass. Off under the hermetic test profile or when
        ``meta_research.meta_meta_enabled`` is unset.
        """
        if self._meta_built:
            return self._meta
        self._meta_built = True
        cfg = getattr(self.settings, "meta_research", None)
        try:
            if str(getattr(getattr(self.settings, "profile", None), "value", "")) == "test":
                return None
            if cfg is None or not bool(getattr(cfg, "meta_meta_enabled", True)):
                return None
            from nyxara.growth.meta_meta import MetaResearchGenome, build_engine_meta_tower
            seed = MetaResearchGenome(max_candidates=int(getattr(cfg, "max_candidates", 4)))
            path = None
            try:
                from pathlib import Path
                path = Path(self.settings.paths.data_dir) / "foundry" / "research_meta_meta.json"
            except Exception:  # noqa: BLE001 — persistence is a bonus, never required
                path = None
            self._meta = build_engine_meta_tower(seed, cfg=cfg, persist_path=path, seed=202)
        except Exception:  # noqa: BLE001 — the tower is a capability, never required
            self._meta = None
        return self._meta

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #
    def run(self, topic: str) -> MetaResearchReport:
        """Full meta-research pass on ``topic``. Always returns a report."""
        t0 = time.monotonic()
        cfg = self.settings.meta_research
        report = MetaResearchReport(topic=topic)
        # META-META: bind the recursive tower's champion (or on-trial) invention breadth BEFORE this
        # pass, so it invents under the config currently being measured; scored below by real yield.
        meta = self._meta_meta()
        if meta is not None:
            from nyxara.growth.meta_meta import MetaResearchGenome
            if isinstance(meta.active, MetaResearchGenome):
                meta.active.apply(self.settings)   # writes max_candidates into cfg (same object)
        try:
            report.open_questions = self._gather(topic)
            candidates = self._invent(topic, report.open_questions)[:int(cfg.max_candidates)]
            for cand in candidates:
                self._test(cand)
                if cand.validated:
                    report.validated += 1
                    self._remember(topic, cand)
                    if self._integrate(cand):
                        report.integrated += 1
                report.candidates.append(cand)
        except Exception as exc:  # noqa: BLE001 — a failed run is data, not a crash
            report.open_questions.append(f"meta-research error: {exc}")
        # Empirical science: if a LawDiscoveryEngine is wired, run one round of real symbolic law
        # discovery in the SAME continuous pass — so NYXARA's autonomous loop invents both code AND
        # laws. Fully offline (pure-Python fallback) and best-effort; a failure never breaks the pass.
        if self.law_discovery is not None:
            try:
                law_report = self.law_discovery.discover_laws(rounds=1)
                report.laws_discovered = len(getattr(law_report, "discoveries", []) or [])
            except Exception:  # noqa: BLE001 — law discovery is a capability, never required
                report.laws_discovered = 0
        report.elapsed_ms = (time.monotonic() - t0) * 1000
        # META-META: score the active invention breadth by the real validated-theory yield this pass
        # (new verified knowledge created), then evolve the whole recursive tower.
        if meta is not None:
            try:
                meta.record(float(report.validated))
                meta.maybe_evolve()
                report.meta_meta = meta.status()
            except Exception:  # noqa: BLE001 — the tower is advisory, never fatal to a pass
                report.meta_meta = None
        self._reports.append(report)
        return report

    def all_reports(self) -> List[MetaResearchReport]:
        return list(self._reports)

    def total_validated(self) -> int:
        return sum(r.validated for r in self._reports)

    # ---------------------------------------------------------------------- #
    # 1) gather — mine the OPEN / incomplete questions in the research
    # ---------------------------------------------------------------------- #
    def _gather(self, topic: str) -> List[str]:
        questions: List[str] = []
        if self.researcher is not None:
            try:
                rep = self.researcher.research(topic)
                claims = list(getattr(rep, "key_claims", []) or [])
                summary = getattr(rep, "summary", "") or ""
                # hedging / incompleteness markers signal an open problem worth attacking
                hedges = ("may", "might", "unclear", "unknown", "open problem", "no sources",
                          "could", "remains", "further", "not clear")
                for c in claims:
                    if any(h in c.lower() for h in hedges):
                        questions.append(f"Can the open question be resolved: {c[:120]}")
                if any(h in summary.lower() for h in hedges) and not questions:
                    questions.append(f"What unresolved aspect of '{topic}' can be made precise?")
            except Exception:  # noqa: BLE001 — research is best-effort
                pass
        if not questions:
            # deterministic, offline fallback: pose our own open questions about the topic
            questions = [f"Is there a faster method for '{topic}'?",
                         f"What invariant must any correct '{topic}' approach preserve?"]
        return questions[:6]

    # ---------------------------------------------------------------------- #
    # 2) invent — LLM (if available) with a deterministic heuristic fallback
    # ---------------------------------------------------------------------- #
    def _invent(self, topic: str, open_questions: List[str]) -> List[CandidateTheory]:
        cands: List[CandidateTheory] = []
        if self.settings.meta_research.use_llm and self._llm_available():
            cands.extend(self._invent_llm(topic, open_questions))
        # always include heuristic candidates so the loop creates information offline too
        cands.extend(self._invent_heuristic(topic))
        return cands

    def _invent_heuristic(self, topic: str) -> List[CandidateTheory]:
        """A deterministic stream of genuinely checkable theories / optimizations.

        Each candidate's code sets ``result`` and ``expected``; the sandbox test passes iff they
        are equal — so even with no LLM she keeps inventing and *verifying* new propositions.
        """
        return [
            CandidateTheory(
                title="Memoization preserves correctness while removing recomputation",
                kind="optimization",
                description=("A memoized recurrence yields the same value as the naive one, "
                             "so caching sub-results is a safe optimization technique."),
                code=(
                    "def naive(n):\n"
                    "    return n if n < 2 else naive(n-1) + naive(n-2)\n"
                    "def memo(n, _c={}):\n"
                    "    if n < 2:\n"
                    "        return n\n"
                    "    if n not in _c:\n"
                    "        _c[n] = memo(n-1) + memo(n-2)\n"
                    "    return _c[n]\n"
                    "result = memo(25)\n"
                    "expected = naive(25)\n"),
            ),
            CandidateTheory(
                title="The sum of the first n odd numbers equals n squared",
                kind="theory",
                description="A closed-form identity: 1+3+5+...+(2n-1) = n^2.",
                code=(
                    "n = 50\n"
                    "result = sum(2*k - 1 for k in range(1, n+1))\n"
                    "expected = n*n\n"),
            ),
            CandidateTheory(
                title="Iterative accumulation matches the closed-form arithmetic series",
                kind="optimization",
                description=("Replacing an O(n) loop sum with the O(1) Gauss formula is a valid "
                             "optimization: both give n(n+1)/2."),
                code=(
                    "n = 1000\n"
                    "result = n*(n+1)//2\n"
                    "expected = sum(range(1, n+1))\n"),
            ),
            CandidateTheory(
                title="A dict cache returns identical values to direct computation",
                kind="optimization",
                description="Caching pure-function results changes cost, not value.",
                code=(
                    "def f(x):\n"
                    "    return x*x + 1\n"
                    "cache = {}\n"
                    "def cached(x):\n"
                    "    if x not in cache:\n"
                    "        cache[x] = f(x)\n"
                    "    return cache[x]\n"
                    "result = [cached(i) for i in range(20)] + [cached(i) for i in range(20)]\n"
                    "expected = [f(i) for i in range(20)] * 2\n"),
            ),
        ]

    def _invent_llm(self, topic: str, open_questions: List[str]) -> List[CandidateTheory]:
        """Ask the LLM for a checkable theory/optimization as runnable Python. Best-effort."""
        try:
            qs = "; ".join(open_questions[:3]) or topic
            prompt = (
                f"Invent ONE new, genuinely testable mathematical theory or optimization "
                f"technique relevant to '{topic}'. Open questions: {qs}. "
                f"Reply with Python that sets two variables: `result` (what your idea computes) "
                f"and `expected` (the independently-known correct value). Use only pure "
                f"computation — no imports, no I/O. Output ONLY the code.")
            text = self.llm.generate(prompt, max_tokens=300, temperature=0.4)
            code = _extract_code(str(text or ""))
            if not code or "result" not in code or "expected" not in code:
                return []
            return [CandidateTheory(
                title=f"LLM-invented technique for {topic}", kind="optimization",
                description=qs[:160], code=code, source="llm")]
        except Exception:  # noqa: BLE001 — invention is best-effort
            return []

    # ---------------------------------------------------------------------- #
    # 3) test — run the candidate's code in the sandbox, restricted namespace
    # ---------------------------------------------------------------------- #
    def _test(self, cand: CandidateTheory) -> None:
        sandbox = self.sandbox
        if sandbox is None:
            try:
                from nyxara.sim.sandbox import Sandbox
                sandbox = self.sandbox = Sandbox()
            except Exception:  # noqa: BLE001
                cand.experiment_result = "sandbox unavailable — untested"
                return

        def _experiment(ctx: Any) -> bool:
            ns = _safe_exec(cand.code)
            return ns.get("result") == ns.get("expected") and "expected" in ns

        try:
            result = sandbox.run(_experiment)
            cand.tested = True
            ok = bool(getattr(result, "success", False)) and bool(getattr(result, "value", False))
            cand.validated = ok
            cand.experiment_result = ("validated: result == expected" if ok else
                                      (getattr(result, "error", None)
                                       or "refuted: result != expected"))
        except Exception as exc:  # noqa: BLE001
            cand.tested = True
            cand.validated = False
            cand.experiment_result = f"test error: {exc}"

    # ---------------------------------------------------------------------- #
    # 4) integrate — double-gated, reversible, gauntlet-checked source edit
    # ---------------------------------------------------------------------- #
    def _integrate(self, cand: CandidateTheory) -> bool:
        cfg = self.settings.meta_research
        if not (cand.validated and cand.kind == "optimization"):
            return False
        if not (bool(getattr(cfg, "allow_integration", False))
                and bool(getattr(self.settings.self_improvement, "autonomous_enact", False))):
            return False
        try:
            from nyxara.growth.self_optimize import Optimizer, SourceEdit
            from nyxara.growth.meta_lab import lab_path, render_addition
        except Exception:  # noqa: BLE001
            return False
        try:
            path = lab_path()
            before = path.read_text(encoding="utf-8") if path.exists() else _LAB_HEADER
            after = before + render_addition(cand.title, cand.description, cand.code)
            edit = SourceEdit(file=str(path), kind="meta_research_integration",
                              before=before, after=after,
                              rationale=f"integrate validated optimization: {cand.title}")
            opt = Optimizer(settings=self.settings, journal=self.journal,
                            permissions=self.permissions)
            # ensure the lab file exists so the Optimizer can read/rollback it
            if not path.exists():
                path.write_text(_LAB_HEADER, encoding="utf-8")
                edit.before = _LAB_HEADER
            outcome = opt.apply(edit)
            opt.close()
            cand.integrated = bool(outcome.kept)
            return cand.integrated
        except Exception:  # noqa: BLE001 — integration is best-effort and reversible
            return False

    # ---------------------------------------------------------------------- #
    # store validated inventions as semantic memories
    # ---------------------------------------------------------------------- #
    def _remember(self, topic: str, cand: CandidateTheory) -> None:
        if self.memory is None:
            return
        try:
            from nyxara.memory.provenance import Provenance, SourceType
            from nyxara.memory.store import MemoryType
            self.memory.remember(
                f"[Invention: {cand.kind}] {cand.title} — {cand.description}",
                mem_type=MemoryType.SEMANTIC,
                provenance=Provenance(SourceType.SELF_REFLECTION, confidence=0.7),
                importance=0.7, tags=["meta-research", "invention", topic])
        except Exception:  # noqa: BLE001
            pass

    def _llm_available(self) -> bool:
        if self.llm is None:
            return False
        try:
            return self.llm.chosen_provider().name != "native"
        except Exception:  # noqa: BLE001
            return False


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
_LAB_HEADER = (
    '"""NYXARA · growth/meta_lab.py — sandbox-grown, gauntlet-verified helpers.\n\n'
    "Auto-extended by the MetaResearcher when an invented optimization is validated AND the\n"
    "Master has authorised integration. Every addition cleared the verify-or-rollback gauntlet.\n"
    '"""\n\nfrom __future__ import annotations\n')

# a deliberately small, safe builtins whitelist — pure computation only, no imports / I/O
_SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in ("abs", "min", "max", "range", "len", "sum", "sorted", "enumerate", "list",
                 "dict", "set", "tuple", "int", "float", "str", "bool", "round", "map",
                 "filter", "zip", "pow", "divmod", "all", "any", "frozenset", "reversed")
}


def _safe_exec(code: str) -> Dict[str, Any]:
    """Execute candidate code in a restricted namespace (no imports, no I/O). Returns its globals."""
    ns: Dict[str, Any] = {"__builtins__": dict(_SAFE_BUILTINS)}
    exec(compile(code, "<meta-research-candidate>", "exec"), ns)  # noqa: S102 — sandboxed builtins
    return ns


def _extract_code(text: str) -> str:
    """Pull a python code block out of an LLM reply (fenced or bare)."""
    t = text.strip()
    if "```" in t:
        parts = t.split("```")
        for i in range(1, len(parts), 2):
            block = parts[i]
            if block.lstrip().lower().startswith("python"):
                block = block.split("\n", 1)[1] if "\n" in block else ""
            if "result" in block:
                return block.strip()
        # fall back to the first fenced block
        if len(parts) >= 2:
            return parts[1].strip()
    return t


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.sim.sandbox import Sandbox

    print("=" * 70)
    print("NYXARA meta-research self-test")
    print("=" * 70)

    mr = MetaResearcher(sandbox=Sandbox())   # fully offline: no llm, no researcher
    report = mr.run("caching strategies")

    for c in report.candidates:
        print(f"  [{c.kind:12}] {c.title[:48]:48} validated={c.validated} "
              f"integrated={c.integrated}")
    print(f"\nopen questions     : {len(report.open_questions)}")
    print(f"validated / total  : {report.validated} / {len(report.candidates)}")

    # the loop must invent and VERIFY at least one genuinely-checkable proposition
    assert report.candidates, "no candidates invented"
    assert report.validated >= 1, "no candidate was validated in the sandbox"
    # integration is double-gated OFF by default — nothing is integrated, no source touched
    assert report.integrated == 0, "integration must be off by default"
    assert all(c.source == "heuristic" for c in report.candidates)

    # a deliberately false 'theory' is refuted, never validated
    bad = CandidateTheory(title="2+2=5", kind="theory",
                          code="result = 2 + 2\nexpected = 5\n")
    mr._test(bad)
    assert bad.tested and not bad.validated
    print("\nfalse theory       : correctly refuted ✓")

    print("\nALL SELF-TESTS PASSED ✓")
