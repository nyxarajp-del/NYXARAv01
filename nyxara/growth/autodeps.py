"""NYXARA · growth/autodeps.py — auto-install her heavy brain stack on first run (⤓).

"Run NYXARA and it just works." Her OWN primary brain is a LoRA-tuned Qwen3-4B
(:mod:`nyxara.growth.bootstrap`), which needs the heavy ML stack — ``torch`` + ``transformers``
+ ``peft`` + ``accelerate`` (the ``.[foundry]`` / ``.[qwen]`` extras). Rather than make the
Master install them by hand, this module installs them automatically on the **first** boot
that finds them missing, so the very next step (download Qwen3-4B + LoRA-tune the adapter)
can run.

Everything here is **best-effort and never fatal**:

* The stack is already importable -> nothing to do (instant).
* Install succeeds -> the foundry's real LoRA path lights up on this same boot.
* Install fails / no network / opted out -> logged once, never raised; NYXARA falls back to
  her always-on pure-stdlib n-gram brain (and the deterministic mock), so the console stays
  usable.

Crucially this must run **before** :mod:`nyxara.growth.foundry_models` is first imported,
because that module probes the stack at import time (``_HAS_LORA``). The boot front doors
(:mod:`nyxara.__main__`, :mod:`nyxara.server.__main__`) call :func:`ensure_runtime_deps`
first, then import the core lazily.

Opt out with ``NYXARA_AUTO_INSTALL=0`` (or any of: ``false``/``no``/``off``). The install is
attempted at most once per machine — a sentinel under ``~/.nyxara`` stops a slow or failing
install from being retried on every boot.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional

__all__ = ["runtime_deps_present", "ensure_runtime_deps"]

# The import names that must all be present for the real Qwen3-4B LoRA forge (foundry + qwen).
_REQUIRED_MODULES = ("torch", "transformers", "peft")

# The pip requirements installed when the stack is missing. Union of the pyproject
# ``foundry`` + ``qwen`` extras; pinned loosely to match those extras.
_PIP_PACKAGES: List[str] = [
    "torch>=2.2",
    "transformers>=4.51",
    "peft>=0.10",
    "accelerate>=0.30",
    "datasets>=2.18",
    "tokenizers>=0.15",
]

# Sentinel so a failed/slow install is attempted at most once per machine.
_ATTEMPT_SENTINEL = Path(os.path.expanduser("~/.nyxara/.deps_attempted"))

_FALSEY = {"0", "false", "no", "off", ""}


def runtime_deps_present() -> bool:
    """True iff the heavy LoRA stack (torch + transformers + peft) is importable.

    Uses :func:`importlib.util.find_spec` so we probe availability **without importing**
    the foundry — keeping its import-time ``_HAS_LORA`` probe accurate after a fresh install.
    """
    try:
        return all(importlib.util.find_spec(m) is not None for m in _REQUIRED_MODULES)
    except Exception:  # noqa: BLE001 — a broken meta-path finder just means "not present"
        return False


def _auto_install_enabled() -> bool:
    """Honour the ``NYXARA_AUTO_INSTALL`` opt-out (default on)."""
    return os.environ.get("NYXARA_AUTO_INSTALL", "1").strip().lower() not in _FALSEY


def _repo_root() -> Optional[Path]:
    """The editable source checkout root (has ``pyproject.toml``), or ``None`` for a wheel."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").exists() else None


def _pip_command() -> List[str]:
    """Prefer the editable extras (honours pyproject pins) when run from a source checkout;
    fall back to the explicit package set when installed as a wheel."""
    root = _repo_root()
    if root is not None:
        return [sys.executable, "-m", "pip", "install", "-e", ".[foundry,qwen]"]
    return [sys.executable, "-m", "pip", "install", *_PIP_PACKAGES]


def _reexec(say: Callable[[str], None]) -> None:
    """Restart the current process so the just-installed stack is seen from import time.

    The ``nyxara`` package imports the foundry eagerly (it probes the LoRA stack at import
    time), so a stack installed *after* ``import nyxara`` is invisible until a fresh
    interpreter starts. Re-exec'ing with the same argv makes the install take effect on this
    very run; the deps-present short-circuit at the top of :func:`ensure_runtime_deps` (plus a
    one-shot env guard) prevents any restart loop. Best-effort: if re-exec fails we simply
    continue in-process (her own brain then forges on the next run).
    """
    if os.environ.get("NYXARA_AUTODEPS_REEXECED") == "1":
        return
    os.environ["NYXARA_AUTODEPS_REEXECED"] = "1"
    say("brain stack       : restarting NYXARA to load her new brain…")
    try:
        os.execv(sys.executable, [sys.executable, *sys.argv])
    except Exception as exc:  # noqa: BLE001 — re-exec is an optimisation, never required
        say(f"brain stack       : could not restart ({exc}); her own brain forges next run")


def ensure_runtime_deps(log: Optional[Callable[[str], None]] = None) -> bool:
    """Ensure the heavy brain stack is installed, installing it on first run if missing.

    Returns ``True`` when the stack is present (already, or after a successful install),
    ``False`` otherwise. **Never raises** — boot integrity comes first.
    """
    say = log or (lambda _msg: None)

    if runtime_deps_present():
        return True

    # Skip in hermetic/test contexts and when the Master opts out.
    if not _auto_install_enabled():
        say("brain stack       : auto-install disabled (NYXARA_AUTO_INSTALL=0) — "
            "using the always-on n-gram brain")
        return False
    if os.environ.get("NYXARA_PROFILE", "").strip().lower() == "test":
        return False

    # One-shot: don't re-attempt a slow/failed install on every boot.
    if _ATTEMPT_SENTINEL.exists():
        say("brain stack       : a prior auto-install attempt was recorded; skipping "
            "(delete ~/.nyxara/.deps_attempted to retry, or run: "
            "pip install -e \".[foundry,qwen]\")")
        return False
    try:
        _ATTEMPT_SENTINEL.parent.mkdir(parents=True, exist_ok=True)
        _ATTEMPT_SENTINEL.touch()
    except Exception:  # noqa: BLE001 — sentinel is an optimisation, never required
        pass

    cmd = _pip_command()
    say("brain stack       : installing torch + transformers + peft (the Qwen3-4B LoRA "
        "stack) — this is a large, one-time download…")
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv (sys.executable -m pip), no shell
            cmd,
            cwd=str(_repo_root() or Path.cwd()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 — a failed install must never block boot
        say(f"brain stack       : auto-install could not start ({exc}); "
            "falling back to the n-gram brain")
        return False

    if proc.returncode == 0 and runtime_deps_present():
        say("brain stack       : installed ✓")
        _reexec(say)  # restart so the freshly-installed stack is seen from import time
        return True

    tail = "\n".join((proc.stdout or "").strip().splitlines()[-3:])
    say(f"brain stack       : auto-install failed (exit {proc.returncode}); "
        "falling back to the n-gram brain. To install manually:\n"
        "  pip install -e \".[foundry,qwen]\"")
    if tail:
        say(f"  pip said: {tail}")
    return False
