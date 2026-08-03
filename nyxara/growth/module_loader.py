"""NYXARA · growth/module_loader.py — load a file she just wrote into the running process (⟲).

NYXARA could already write Python to disk under a gauntlet
(:meth:`~nyxara.growth.self_optimize.Optimizer.apply`), execute Python in a subprocess sandbox
(:func:`~nyxara.agency.code_sandbox.run_python`), hot-swap a live object reference
(:meth:`~nyxara.causal.polymorph_runtime.PolymorphRuntime.hot_swap`) and ctypes-load a compiled
native kernel (:mod:`~nyxara.growth.native_forge`). The one thing she could not do was **import a
new ``.py`` she had just written into the process she is running in**: there is not a single
``spec_from_file_location`` anywhere in the package, and ``PolymorphRuntime.reload_module`` only
reloads names that are already importable. This module is that missing function.

It is deliberately small and deliberately fenced:

* The file must live **inside an allowed root** — by default ``nyxara/growth/_forged/``, the
  quarantine directory for code she authored herself. Loading arbitrary paths from a
  self-modification loop is how a bug becomes an incident.
* The source is screened with the existing
  :meth:`~nyxara.growth.capability_foundry.CapabilityFoundry._scan_source` AST allowlist *before*
  the module object is created, so a forbidden import or attribute never reaches execution.
* Loading is not sandboxed and is not pretending to be. Importing a module runs its top level in
  **this** process with full privileges. The sandbox is where untested code belongs
  (:func:`~nyxara.agency.code_sandbox.run_python`); this function is for code that has *already*
  passed the sandbox and the gauntlet, at the moment it becomes live.

Depends on ``growth/capability_foundry`` (the screen) and the stdlib. Nothing imports back.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

__all__ = ["LoadResult", "forged_root", "screen_source", "load_module_from_path"]


@dataclass
class LoadResult:
    """The outcome of one load — the module, or a truthful reason there is none."""

    module: Any = None
    name: str = ""
    path: str = ""
    ok: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "path": self.path, "ok": self.ok, "reason": self.reason}


def _package_root() -> Path:
    """The ``nyxara/`` directory of the running install."""
    return Path(__file__).resolve().parent.parent


def forged_root() -> Path:
    """``nyxara/growth/_forged/`` — where code she authored herself is allowed to live."""
    return _package_root() / "growth" / "_forged"


def screen_source(source: str) -> Optional[str]:
    """Run the existing AST allowlist over ``source``. Returns a refusal reason, or ``None``.

    Reuses :class:`~nyxara.growth.capability_foundry.CapabilityFoundry`'s scanner rather than
    growing a second, subtly-different policy — two screens that disagree are worse than one."""
    try:
        from nyxara.growth.capability_foundry import CapabilityFoundry
        # require_handle=False: that check is the forged-TOOL contract, not a safety rule, and a
        # module she authored legitimately has no handle(). Everything security-relevant — the
        # import allowlist, forbidden names, forbidden attribute roots, forbidden substrings —
        # still applies, and applies from the one shared policy.
        ok, reason = CapabilityFoundry._scan_source(       # noqa: SLF001 — the shared policy
            source, require_handle=False)
    except Exception as exc:  # noqa: BLE001 — no screen available means no load: fail closed
        return f"source screen unavailable ({exc})"
    return None if ok else (str(reason) or "refused by the source screen")


def load_module_from_path(path: Any, *, name: Optional[str] = None,
                          allowed_roots: Optional[Sequence[Any]] = None,
                          screen: bool = True, register: bool = True) -> LoadResult:
    """Import a ``.py`` file into the running process. Never raises.

    ``allowed_roots`` defaults to :func:`forged_root`. Any path outside every allowed root is
    refused — fail-closed, including when the path cannot be resolved at all."""
    target = Path(str(path))
    result = LoadResult(path=str(target))
    try:
        target = target.resolve()
        result.path = str(target)
    except Exception as exc:  # noqa: BLE001 — a path we cannot resolve is a path we cannot trust
        result.reason = f"cannot resolve path ({exc})"
        return result

    if target.suffix != ".py":
        result.reason = "not a Python source file"
        return result
    if not target.is_file():
        result.reason = "no such file"
        return result

    roots: List[Path] = []
    for root in (allowed_roots if allowed_roots is not None else [forged_root()]):
        try:
            roots.append(Path(str(root)).resolve())
        except Exception:  # noqa: BLE001
            continue
    if not roots or not any(target.is_relative_to(root) for root in roots):
        result.reason = ("outside the allowed root(s): "
                         + ", ".join(str(r) for r in roots) if roots else "no allowed root")
        return result

    if screen:
        try:
            source = target.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            result.reason = f"unreadable ({exc})"
            return result
        refusal = screen_source(source)
        if refusal:
            result.reason = f"refused by the source screen: {refusal}"
            return result

    module_name = name or f"nyxara_forged_{target.stem}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, target)
        if spec is None or spec.loader is None:
            result.reason = "could not build an import spec"
            return result
        module = importlib.util.module_from_spec(spec)
        # Registered BEFORE exec_module so a module that imports itself (or is re-entered by a
        # decorator at its own top level) resolves rather than importing a second copy.
        if register:
            sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 — a module that fails to import is data, not a crash
        sys.modules.pop(module_name, None)
        result.reason = f"import failed: {exc}"
        return result

    result.module, result.name, result.ok = module, module_name, True
    return result
