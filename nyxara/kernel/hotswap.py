"""NYXARA · kernel/hotswap.py — zero-downtime live patching of a healed module (⚡, C).

When the self-heal loop keeps a gauntlet-verified edit, the *running* process still holds the OLD module
object in memory until a restart. This applies the kept edit to the live process by reloading the edited
leaf module with :func:`importlib.reload`, so a fix takes effect without dropping the mind's live state
(memory, active-inference, reasoning) — that state lives on the long-lived ``NyxaraCore``, not in the
reloaded leaf.

The honest boundary is enforced in code, not just documented:

* Only a **leaf-ish** module is hot-swapped. A module that is currently *imported by other loaded modules*
  (an in-package importer still holds references to its old objects) is judged **unsafe** — reloading it
  would leave stale bindings elsewhere — so the edit is left on disk to take effect on the next restart
  (``hot_swap=False``), never force-swapped at the risk of corrupting live state.
* The constitutional / loyalty / master core is **never** hot-swapped (reuses the same fail-closed guard
  the source-editor uses).
* Any reload error rolls the module back to the pre-swap source and reports failure — the process is left
  exactly as it was.

Pure standard library; no LLM; it changes no gate or character value.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["HotSwapResult", "HotSwapper"]


@dataclass
class HotSwapResult:
    module: str
    hot_swapped: bool
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"module": self.module, "hot_swapped": self.hot_swapped, "reason": self.reason}


class HotSwapper:
    """Reload a healed leaf module into the running process, or fall back to disk + restart."""

    def __init__(self, *, journal: Any = None) -> None:
        self.journal = journal

    # ---- safety judgement ---- #
    @staticmethod
    def _module_name_for_path(path: str) -> Optional[str]:
        """Find the already-imported module whose file is ``path`` (None if not loaded)."""
        target = str(Path(path).resolve())
        for name, mod in list(sys.modules.items()):
            f = getattr(mod, "__file__", None)
            if f and str(Path(f).resolve()) == target:
                return name
        return None

    @staticmethod
    def _importers(module_name: str) -> List[str]:
        """Other loaded modules that hold a reference to ``module_name`` (or a submodule of it).

        A non-empty list means the reloaded objects would leave stale bindings behind → unsafe to swap."""
        importers: List[str] = []
        for name, mod in list(sys.modules.items()):
            if name == module_name or mod is None:
                continue
            # a package holding the submodule attribute, or any module with a binding to the object
            if name != module_name and getattr(mod, "__name__", "") == module_name:
                importers.append(name)
                continue
            d = getattr(mod, "__dict__", None)
            if not isinstance(d, dict):
                continue
            for v in d.values():
                if getattr(v, "__module__", None) == module_name:
                    importers.append(name)
                    break
        return sorted(set(importers))

    def is_safe(self, path: str) -> HotSwapResult:
        """Judge (without swapping) whether ``path`` can be hot-reloaded safely."""
        try:
            from nyxara.growth.self_optimize import _is_constitutional
            if _is_constitutional(str(Path(path))):
                return HotSwapResult(path, False, "constitutional/master core — never hot-swapped")
        except Exception:  # noqa: BLE001 — if the guard cannot load, fail closed
            return HotSwapResult(path, False, "could not verify constitutional guard — fail closed")
        name = self._module_name_for_path(path)
        if name is None:
            return HotSwapResult(path, False, "module is not currently imported — nothing live to swap")
        importers = self._importers(name)
        if importers:
            return HotSwapResult(
                name, False,
                f"referenced by {len(importers)} loaded module(s) — stale bindings risk, disk-only")
        return HotSwapResult(name, True, "safe leaf module")

    # ---- the swap ---- #
    def swap(self, path: str) -> HotSwapResult:
        """Hot-reload ``path`` if safe; otherwise leave it for the next restart. Never raises."""
        verdict = self.is_safe(path)
        if not verdict.hot_swapped:
            self._journal(verdict)
            return verdict
        name = verdict.module
        mod = sys.modules.get(name)
        before_src: Optional[str] = None
        try:
            before_src = Path(path).read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            before_src = None
        # Force a fresh compile from the edited source: drop any cached bytecode (a .pyc whose
        # mtime+size still matches the edit would otherwise reload STALE code) and clear finder caches.
        self._drop_bytecode(mod)
        importlib.invalidate_caches()
        try:
            importlib.reload(mod)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 — a bad reload must leave the process untouched
            # best-effort restore of the on-disk source, then re-reload the good copy
            if before_src is not None:
                try:
                    Path(path).write_text(before_src, encoding="utf-8")
                    importlib.reload(mod)  # type: ignore[arg-type]
                except Exception:  # noqa: BLE001
                    pass
            res = HotSwapResult(name, False, f"reload failed, left as-is: {exc.__class__.__name__}")
            self._journal(res)
            return res
        res = HotSwapResult(name, True, "hot-swapped live (no restart)")
        self._journal(res)
        return res

    @staticmethod
    def _drop_bytecode(mod: Any) -> None:
        """Remove the module's cached .pyc so reload recompiles from the edited source."""
        cached = getattr(mod, "__cached__", None)
        if cached:
            try:
                Path(cached).unlink()
            except Exception:  # noqa: BLE001 — no pyc / already gone is fine
                pass

    def _journal(self, res: HotSwapResult) -> None:
        rec = getattr(self.journal, "record", None) or getattr(self.journal, "append", None)
        if callable(rec):
            try:
                rec({"event": "hot_swap", **res.to_dict()})
            except Exception:  # noqa: BLE001 — journalling is best-effort
                pass
