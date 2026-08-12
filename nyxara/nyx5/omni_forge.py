"""NYXARA · nyx5/omni_forge.py — omni-forge sub-agent genesis (🕸️, pillar 9).

When no existing tool fits, NYX-5 forges one: she writes a small, specialised function, and runs it — but
**only** through the sovereign path. Every forge is (a) statically screened for danger (reusing the
pillar-18 phagocyte), (b) approved by a permission gate, (c) executed in a **restricted sandbox** with no
imports and a safe-builtins whitelist, and (d) recorded in an append-only ledger. Forged tools are
traceable and governed — never hidden.

Refused by construction (fail-closed): tools that break encryption, reach unauthorised/dark-web targets,
or covertly self-delete. The sandbox has no file, network, OS, `eval`/`exec`, or `import` access, so such
tools cannot be forged here at all. Default off; bounded by a per-turn forge cap.

Reuses :mod:`nyxara.nyx5.phagocytosis` for static screening. Pure standard library.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from nyxara.nyx5.phagocytosis import DigitalPhagocyte

__all__ = ["ForgeResult", "OmniForge"]

# a deliberately small, side-effect-free builtins whitelist for forged tools.
_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum, "len": len, "range": range,
    "sorted": sorted, "map": map, "filter": filter, "enumerate": enumerate, "zip": zip,
    "round": round, "int": int, "float": float, "str": str, "bool": bool, "list": list,
    "dict": dict, "set": set, "tuple": tuple, "any": any, "all": all, "pow": pow,
}


@dataclass
class ForgeResult:
    """The outcome of one forge: approved+run, or refused (screened/denied) — always logged."""

    name: str
    status: str                      # "ran" | "refused-screen" | "refused-gate" | "error"
    result: Any = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "status": self.status, "reason": self.reason}


class OmniForge:
    """Forge and run small tools — statically screened, permission-gated, sandboxed, and logged."""

    def __init__(self, *, max_forges_per_turn: int = 2,
                 permission_gate: Optional[Callable[[str, str], bool]] = None) -> None:
        self.max_forges_per_turn = int(max_forges_per_turn)
        self._gate = permission_gate or (lambda name, src: False)  # default-deny (safe)
        self._phagocyte = DigitalPhagocyte(static_only=True)
        self._ledger: List[Dict[str, Any]] = []
        self._turn_forges = 0

    def begin_turn(self) -> None:
        self._turn_forges = 0

    def forge(self, name: str, source: str, *, args: tuple = (),
              kwargs: Optional[Dict[str, Any]] = None) -> ForgeResult:
        """Screen → gate → sandbox-run a tool named ``name`` defined by ``source`` (must define ``tool``)."""
        kwargs = kwargs or {}
        if self._turn_forges >= self.max_forges_per_turn:
            return self._log(ForgeResult(name, "refused-gate", reason="per-turn forge cap reached"))
        self._turn_forges += 1

        # (a) static danger screen — reuse the phagocyte; any dangerous construct refuses the forge.
        sig = self._phagocyte.analyse(source)
        if sig.categories:
            return self._log(ForgeResult(name, "refused-screen",
                                         reason=f"dangerous constructs: {sorted(set(sig.categories))}"))
        # (b) permission gate — the sovereign kernel decides; default-deny.
        if not self._gate(name, source):
            return self._log(ForgeResult(name, "refused-gate", reason="permission gate denied"))
        # (c) sandboxed execution — no imports, no builtins beyond the safe whitelist.
        try:
            result = self._run_sandboxed(source, args, kwargs)
        except Exception as exc:  # noqa: BLE001 — a forged tool's failure never breaks the caller
            return self._log(ForgeResult(name, "error", reason=str(exc)))
        return self._log(ForgeResult(name, "ran", result=result))

    def _run_sandboxed(self, source: str, args: tuple, kwargs: Dict[str, Any]) -> Any:
        sandbox_globals: Dict[str, Any] = {"__builtins__": dict(_SAFE_BUILTINS)}
        exec(compile(source, "<omni-forge>", "exec"), sandbox_globals)  # noqa: S102 — screened + restricted
        tool = sandbox_globals.get("tool")
        if not callable(tool):
            raise ValueError("forge source must define a callable named 'tool'")
        return tool(*args, **kwargs)

    def _log(self, result: ForgeResult) -> ForgeResult:
        self._ledger.append({"at": time.time(), **result.to_dict()})
        return result

    def ledger(self) -> List[Dict[str, Any]]:
        return list(self._ledger)
