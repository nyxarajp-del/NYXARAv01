"""NYXARA · nyx/hands.py — she can see the toolset, and reach for it (🖐, NYX V.02, gap 7).

The thing that was measurably missing here was **not permission**. NYXARA already ships full
operational control on by default — shell, code execution, file write and delete, self-modify,
package install, autonomous internet — without escalating each action
(``docs/CAPABILITIES.md``: *"NYXARA ships full operational control on by default"*). Nothing in
this module loosens anything, and nothing in it needed to.

What was missing was **sight**. Two gaps, both pure wiring:

1. :class:`~nyxara.nyx.brain.NyxBrain` was never handed the tool registry, so her own cycle had
   no idea what tools existed.
2. :class:`~nyxara.nyx.reasoner.NyxReasoner` deliberately never set ``tool``/``tool_args``
   (``reasoner.py:14``), so even when she knew what to reach for, she could not say so.

:class:`Hands` closes the first: it reads ``core.tools``, asks
:class:`~nyxara.agency.tool_router.ToolRouter` which tool fits a request, decides from the
turn's :mod:`~nyxara.nyx.intent` whether a tool is wanted **at all**, executes through the
registry's unchanged pipeline, and keeps a per-tool track record so a tool that keeps failing
stops being her first choice.

Honest framing (the rule this repo keeps, see ``docs/CAPABILITIES.md``):

* Execution goes through :meth:`~nyxara.agency.tools.ToolRegistry.invoke` and therefore through
  the **identical, unchanged, fail-closed** capability/risk/governor pipeline. This module has
  no path that bypasses it, and it never lowers a risk tier or flips ``reversible``.
* **Not every turn wants a tool.** A question of fact does not, and the commonest failure of a
  tool-using agent is reaching for one because it can. The intent reader gates this: a
  statement or an ordinary question does not trigger a reach, and that is a deliberate refusal
  to act rather than a missing capability.
* The track record is **measured**, not declared: successes and failures per tool, reported
  with their sample count so a thin record is visibly thin.
* ``/scram`` and oversight are untouched and remain the way to stop her — this module asks
  ``paused()``/``scrammed()`` before every reach and simply does not act when they say so.

Pure standard library. Every public method is fail-soft: on any error it degrades to a null
result rather than breaking a turn.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["Reach", "ToolRecord", "Hands"]

#: Intent moods that can *want* a tool. A statement never does; a question only when it asks
#: about the machine's own state rather than about the world.
_ACTING_KINDS = frozenset({"command"})

#: Words that make a question a question *about this machine*, which a tool can answer.
_SELF_REFERENTIAL = frozenset("""
repo repository file files directory folder line lines code test tests branch commit diff log
disk memory cpu process port url page site version installed package packages
""".split())


@dataclass
class ToolRecord:
    """What actually happened the last time she used this tool. Measured, never declared."""

    name: str = ""
    uses: int = 0
    ok: int = 0
    failed: int = 0
    refused: int = 0                 # the gate said no — not a tool failure
    last_error: str = ""
    last_used: float = 0.0

    @property
    def reliability(self) -> float:
        """Observed success rate. Meaningless below a few samples, and ``samples`` says so."""
        graded = self.ok + self.failed
        return (self.ok / float(graded)) if graded else 0.0

    @property
    def samples(self) -> int:
        return self.ok + self.failed

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "uses": self.uses, "ok": self.ok, "failed": self.failed,
                "refused": self.refused, "reliability": round(self.reliability, 4),
                "samples": self.samples, "last_error": self.last_error}


@dataclass
class Reach:
    """One decision about tools: what she picked, why, and what came back."""

    request: str = ""
    wanted: bool = False             # did this turn want a tool at all?
    tool: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    ran: bool = False
    ok: bool = False
    value: Any = None
    error: str = ""
    refused: str = ""                # the gate's reason, when it declined
    candidates: List[Tuple[str, float]] = field(default_factory=list)
    why: str = ""
    verdict: Dict[str, Any] = field(default_factory=dict)   # the consequence gate's decision
    rerouted_via: str = ""       # a reversible variant she took first, when there was one
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"request": self.request, "wanted": self.wanted, "tool": self.tool,
                "args": dict(self.args), "ran": self.ran, "ok": self.ok,
                "value": self.value, "error": self.error, "refused": self.refused,
                "candidates": [[n, round(s, 4)] for n, s in self.candidates],
                "why": self.why, "verdict": dict(self.verdict),
                "rerouted_via": self.rerouted_via,
                "duration_ms": round(self.duration_ms, 2)}


class Hands:
    """Her sight of the toolset, and her reach into it — through the unchanged gate."""

    def __init__(self, brain: Any = None, *, registry: Any = None,
                 min_score: float = 0.5, max_per_beat: int = 1,
                 autonomous: bool = True) -> None:
        self.brain = brain
        self._registry = registry
        self._router: Any = None
        self._tried_router = False
        self.min_score = max(0.0, min(1.0, float(min_score)))
        self.max_per_beat = max(0, int(max_per_beat))
        self.autonomous = bool(autonomous)

        self.record: Dict[str, ToolRecord] = {}
        self.reaches = 0
        self.declined = 0

    # ---- sight ------------------------------------------------------------ #
    def registry(self) -> Any:
        """The live tool registry — from the kernel if she was given one, else nothing."""
        if self._registry is not None:
            return self._registry
        for holder in (self.brain, getattr(self.brain, "core", None)):
            got = getattr(holder, "tools", None)
            if got is not None and hasattr(got, "invoke"):
                self._registry = got
                return got
        return None

    def router(self) -> Any:
        if not self._tried_router:
            self._tried_router = True
            registry = self.registry()
            if registry is None:
                return None
            try:
                from nyxara.agency.tool_router import ToolRouter
                self._router = ToolRouter(registry)
            except Exception:  # noqa: BLE001 — without it she can still name tools by hand
                self._router = None
        return self._router

    def catalog(self) -> List[Dict[str, Any]]:
        """Every tool she can see, with what it costs and what it is allowed to touch."""
        try:
            registry = self.registry()
            if registry is None:
                return []
            return list(registry.schemas())
        except Exception:  # noqa: BLE001
            return []

    def available(self) -> bool:
        return self.registry() is not None

    # ---- judgement: does this turn want a tool at all? --------------------- #
    def wants_tool(self, request: str, intent: Any = None) -> Tuple[bool, str]:
        """Should she reach for anything here? Returns ``(wanted, why)``.

        The commonest failure of a tool-using agent is reaching because it can. A statement
        never wants a tool; a question wants one only when it asks about *this machine*, which
        is the kind of question a tool can actually answer.
        """
        try:
            text = str(request or "").strip()
            if not text:
                return False, "nothing was asked"
            if intent is None and self.brain is not None:
                reader = getattr(self.brain, "intent", None)
                intent = reader.read(text) if reader is not None else None
            kind = str(getattr(intent, "kind", "") or "")
            if kind in _ACTING_KINDS:
                return True, "the turn is an instruction, which is what tools are for"
            if kind == "question":
                from nyxara.nyx.lingua import content_tokens
                hits = [w for w in content_tokens(text, cap=16) if w in _SELF_REFERENTIAL]
                if hits:
                    return True, (f"a question about this machine ({hits[0]}), which a tool can "
                                  f"answer and reasoning alone cannot")
                return False, "a question about the world — reasoning, not a tool"
            if not kind:
                return False, "no intent reader available to judge, so she does not reach"
            return False, f"a {kind} does not ask for anything to be done"
        except Exception:  # noqa: BLE001
            return False, "could not judge the turn, so she does not reach"

    # ---- choosing --------------------------------------------------------- #
    def choose(self, request: str, *, top_k: int = 3) -> List[Tuple[str, float]]:
        """Rank the tools that fit this request, best first, tempered by her track record."""
        try:
            router = self.router()
            if router is None:
                return []
            ranked = router.select(request, top_k=max(1, int(top_k)),
                                   min_score=self.min_score) or []
            out: List[Tuple[str, float]] = []
            for choice in ranked:
                name = str(getattr(choice, "tool", "") or getattr(choice, "name", ""))
                score = float(getattr(choice, "score", 0.0))
                if not name:
                    continue
                out.append((name, score * self._trust(name)))
            out.sort(key=lambda p: -p[1])
            return out
        except Exception:  # noqa: BLE001
            return []

    def _trust(self, name: str) -> float:
        """A tool that keeps failing stops being her first choice — but only once measured.

        Below a handful of samples the record is too thin to mean anything, so it does not move
        the score at all. That is the same discipline :mod:`nyxara.nyx.metacog` uses.
        """
        rec = self.record.get(name)
        if rec is None or rec.samples < 3:
            return 1.0
        return max(0.4, min(1.0, 0.4 + 0.6 * rec.reliability))

    # ---- reaching --------------------------------------------------------- #
    def reach(self, request: str, *, tool: str = "", args: Optional[Dict[str, Any]] = None,
              intent: Any = None, oversight: Any = None, authority: Any = None,
              dry_run: bool = False) -> Reach:
        """Decide, then act — through the registry's unchanged pipeline. Never around it."""
        out = Reach(request=str(request or "").strip())
        t0 = time.perf_counter()
        try:
            if self._stopped(oversight):
                out.why = "oversight has paused or scrammed her; she does not act"
                self.declined += 1
                return out

            registry = self.registry()
            if registry is None:
                out.why = "she has not been handed a tool registry — she is blind to tools"
                self.declined += 1
                return out

            if tool:
                out.wanted, out.why = True, "a specific tool was named"
            else:
                out.wanted, out.why = self.wants_tool(out.request, intent)
                if not out.wanted:
                    self.declined += 1
                    return out
                out.candidates = self.choose(out.request)
                if not out.candidates:
                    out.why = ("this turn wants a tool, but nothing in the catalogue fits it "
                               "well enough to be worth running")
                    self.declined += 1
                    return out
                tool = out.candidates[0][0]

            out.tool = tool
            out.args = dict(args or {})

            # A tool that does not exist is a fact the registry already knows, and saying so
            # is a better answer than the gate's honest-but-generic "I could not model this".
            # The specific diagnosis beats the general one, and it keeps the gate for what the
            # gate is for: deciding about actions that are real.
            if not self._known(registry, tool):
                out.error = (f"there is no tool called {tool!r} in the registry — "
                             f"I cannot reach for something that does not exist")
                out.why = out.error
                self.declined += 1
                return out

            # L-CHRONO-CAUSAL: look before acting. This is her own foresight, not an
            # escalation — nobody is asked for permission, and full autonomy is untouched.
            gate = getattr(self.brain, "consequence", None)
            if gate is not None:
                may, verdict = gate.guard(tool, out.args, registry=registry)
                out.verdict = verdict.to_dict()
                if not may:
                    out.refused = verdict.why
                    out.why = f"the consequence gate stopped this: {verdict.why}"
                    self.declined += 1
                    return out
                if verdict.variant:
                    # A reversible route to the same intent. She takes it herself.
                    out.why = verdict.why
                    self._invoke(registry, verdict.variant, dict(verdict.variant_args),
                                 authority=authority, dry_run=dry_run)
                    out.rerouted_via = verdict.variant
            if not self.autonomous:
                out.why = ("autonomous tool use is switched off for this deployment "
                           "(NYXARA_NYX__HANDS_AUTONOMOUS=false)")
                self.declined += 1
                return out

            result = self._invoke(registry, tool, out.args, authority=authority,
                                  dry_run=dry_run)
            self._absorb(out, result, dry_run=dry_run)
            if not dry_run:
                self.reaches += 1
            return out
        except Exception as exc:  # noqa: BLE001 — a reach that fails is data, not a crash
            out.error = f"{type(exc).__name__}: {exc}"
            return out
        finally:
            out.duration_ms = (time.perf_counter() - t0) * 1000.0

    @staticmethod
    def _known(registry: Any, tool: str) -> bool:
        """Does the registry actually hold this tool? An unreadable registry is not a verdict."""
        try:
            if registry is None:
                return True
            getter = getattr(registry, "get", None)
            if getter is None:
                return True
            return getter(tool) is not None
        except Exception:  # noqa: BLE001 — if she cannot ask, she does not pretend to know
            return True

    def _invoke(self, registry: Any, tool: str, args: Dict[str, Any], *,
                authority: Any, dry_run: bool) -> Any:
        """The single call site. Nothing here is allowed to weaken what the gate is given."""
        if authority is None:
            from nyxara.agency.permissions import Authority
            authority = Authority.AUTONOMOUS
        return registry.invoke(tool, args, authority=authority, dry_run=bool(dry_run))

    def _absorb(self, out: Reach, result: Any, *, dry_run: bool = False) -> None:
        if dry_run:
            # A dry run answers "what would happen"; nothing executed, so ``ran`` stays False
            # and the tool's track record does not move. Crediting a use for something that
            # never ran would teach her a reliability she has not measured.
            out.value = getattr(result, "value", None)
            out.error = str(getattr(result, "error", "") or "")
            decision = getattr(result, "decision", None) or {}
            if isinstance(decision, dict) and decision.get("allowed") is False:
                out.refused = str(decision.get("reason", "") or "owner confirmation required")
            # Appended, never merely filled in: when the caller named a tool ``why`` already
            # says so, and a reply whose only signal is ``ran: false`` cannot be told apart
            # from a refusal. The caller has to be able to see which of the two this was.
            note = ("dry run — this is the tool I would pick and what the gate decided, and "
                    "nothing was executed")
            out.why = f"{out.why}; {note}" if out.why else note
            return
        rec = self.record.setdefault(out.tool, ToolRecord(name=out.tool))
        rec.uses += 1
        rec.last_used = time.time()
        out.ran = True
        out.ok = bool(getattr(result, "ok", False))
        out.value = getattr(result, "value", None)
        out.error = str(getattr(result, "error", "") or "")
        decision = getattr(result, "decision", None) or {}
        if getattr(result, "requires_owner", False) or (
                isinstance(decision, dict) and decision.get("allowed") is False):
            # The gate declined. That is not the tool failing, and recording it as one would
            # slowly teach her to distrust a tool that works fine.
            out.refused = str(decision.get("reason", "") or "owner confirmation required")
            out.ran = False
            rec.refused += 1
            rec.uses -= 1
            return
        if out.ok:
            rec.ok += 1
        else:
            rec.failed += 1
            rec.last_error = out.error[:200]

    @staticmethod
    def _stopped(oversight: Any) -> bool:
        """Corrigibility, untouched: if she has been paused or scrammed, she does not act."""
        if oversight is None:
            return False
        try:
            for attr in ("scrammed", "is_scrammed", "paused", "is_paused"):
                probe = getattr(oversight, attr, None)
                if probe is None:
                    continue
                stopped = probe() if callable(probe) else probe
                if bool(stopped):
                    return True
            return False
        except Exception:  # noqa: BLE001 — an unreadable oversight is treated as STOP
            return True

    # ---- introspection ---------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        try:
            catalog = self.catalog()
            return {
                "available": self.available(),
                "tools_visible": len(catalog),
                "reaches": self.reaches,
                "declined": self.declined,
                "autonomous": self.autonomous,
                "record": {n: r.to_dict() for n, r in
                           sorted(self.record.items(), key=lambda kv: -kv[1].uses)},
                "note": ("execution goes through the registry's unchanged capability/risk "
                         "pipeline; this module gives her sight of the toolset, not extra "
                         "permission, and never lowers a risk tier or flips reversible"),
            }
        except Exception:  # noqa: BLE001
            return {}

    def describe(self) -> str:
        """What ``/nyx hands`` prints."""
        try:
            if not self.available():
                return ("I have not been handed a tool registry — I am blind to tools in this "
                        "process, which is a wiring problem, not a permission one.")
            catalog = self.catalog()
            lines = [f"{len(catalog)} tools visible."]
            for spec in catalog[:40]:
                rec = self.record.get(spec.get("name", ""))
                mark = ""
                if rec is not None and rec.samples:
                    mark = f"   [{rec.ok}/{rec.samples} ok]"
                lines.append(f"  {spec.get('name',''):22} {spec.get('risk',''):9}"
                             f"{'reversible' if spec.get('reversible') else 'IRREVERSIBLE':13}"
                             f"{mark}")
            if len(catalog) > 40:
                lines.append(f"  … and {len(catalog) - 40} more")
            return "\n".join(lines)
        except Exception:  # noqa: BLE001
            return ""

    # ---- persistence ------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {"reaches": self.reaches, "declined": self.declined,
                "record": {n: r.to_dict() for n, r in self.record.items()}}

    def load_dict(self, data: Any) -> bool:
        try:
            if not isinstance(data, dict):
                return False
            self.reaches = int(data.get("reaches", 0))
            self.declined = int(data.get("declined", 0))
            self.record = {}
            for name, raw in (data.get("record") or {}).items():
                if not isinstance(raw, dict):
                    continue
                self.record[str(name)] = ToolRecord(
                    name=str(name), uses=int(raw.get("uses", 0)), ok=int(raw.get("ok", 0)),
                    failed=int(raw.get("failed", 0)), refused=int(raw.get("refused", 0)),
                    last_error=str(raw.get("last_error", "")))
            return True
        except Exception:  # noqa: BLE001 — a corrupt sidecar must never block boot
            return False
