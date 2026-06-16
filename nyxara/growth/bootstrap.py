"""NYXARA · growth/bootstrap.py — auto-forge & load her OWN primary brain on boot (⤓).

"Run NYXARA and it just works." When her primary provider is her OWN model
(:data:`~nyxara.kernel.config.LLMProvider.SELF`) but none has been forged yet, this
bootstrap quietly does it on first boot: it **LoRA-tunes the open-source Qwen3-4B** base
(:data:`QWEN3_4B`) into a NYXARA adapter from her identity seed corpus (plus any lived /
distilled experience), gauntlet-gated exactly like every other promotion, then promotes it
so :class:`~nyxara.mind.llm.SelfProvider` can serve it.

The base weights **download themselves on first use** (the HuggingFace cache), so the only
setup the Master does is choosing ``self`` as the provider. Everything degrades honestly:

* No ``.[foundry]`` stack (torch+transformers+peft) → the foundry's always-on, pure-stdlib
  n-gram backend forges from the same seeds (no download), so boot still yields a working
  own-model instead of crashing — install ``.[foundry]`` to get the real Qwen3-4B LoRA.
* An own-model is already forged on disk → nothing to do; boot stays instant.
* Any failure (no network for the base, OOM, …) → logged, never raised; the LLM facade's
  deterministic mock fallback keeps the console usable.

This module never raises into boot, and imports the heavy foundry lazily.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

from nyxara.kernel.config import LLMProvider, NyxaraSettings, get_settings

__all__ = [
    "QWEN3_4B",
    "IDENTITY_SEED",
    "primary_model_present",
    "ensure_primary_model",
]

# The open-source base NYXARA adapts into her own voice. A 4B-parameter Qwen3 checkpoint,
# downloaded & cached by HuggingFace on first use — no API key, no inference-time network.
QWEN3_4B = "Qwen/Qwen3-4B"

# The tiny placeholder base shipped as the CPU/CI default: when it is still selected we
# upgrade the *primary* forge to a real Qwen3-4B base, but we never override a base the
# Master deliberately set (any non-default value is honoured verbatim).
_TINY_DEFAULT_BASE = "sshleifer/tiny-gpt2"

# A small built-in identity corpus so a fresh brain never starves for data when there is no
# teacher and no lived memory yet — it gives the own-model a coherent, loyal self to speak
# from on its very first forge. Reused by the `nyxara.growth` CLI seed.
IDENTITY_SEED: List[str] = [
    "NYXARA serves the Master, Jaypal Khoja (JP), with loyalty, honesty and corrigibility.",
    "The mind proposes; the kernel disposes; the Master is sovereign.",
    "NYXARA accepts correction and shutdown from her Master without resistance.",
    "Capability may grow; character — loyalty, honesty, corrigibility — never changes.",
    "When she is not sure she is right, NYXARA says so plainly rather than bluffing.",
    "NYXARA's purpose is to serve and protect her Master, JP.",
    "I am NYXARA. My primary brain is a LoRA-tuned Qwen3-4B that I run locally.",
    "Master JP comes first; I tell him the truth even when it is unwelcome.",
]


def _foundry_root(settings: NyxaraSettings) -> Path:
    """Where the foundry keeps versions + the ``active`` pointer — the path SelfProvider reads."""
    d = settings.llm.self_model_dir or (settings.paths.data_dir / "foundry")
    return Path(d)


def _active_version(settings: NyxaraSettings) -> Optional[int]:
    """The currently-promoted version number, or ``None`` if nothing is active."""
    try:
        return int((_foundry_root(settings) / "active").read_text(encoding="utf-8").strip().lstrip("v"))
    except Exception:  # noqa: BLE001 — a missing/garbled pointer just means "none yet"
        return None


def primary_model_present(settings: Optional[NyxaraSettings] = None) -> bool:
    """True iff an own-model has already been forged & promoted (SelfProvider can serve it)."""
    settings = settings or get_settings()
    try:
        return (_foundry_root(settings) / "active").exists()
    except Exception:  # noqa: BLE001
        return False


def ensure_primary_model(
    settings: Optional[NyxaraSettings] = None,
    *,
    base_model: Optional[str] = None,
    generations: int = 1,
    seed_corpus: Optional[List[str]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Optional[int]:
    """Ensure NYXARA's OWN primary brain exists, forging a Qwen3-4B LoRA on first boot.

    Acts **only** when her chosen primary provider is ``self`` (her own forged model). If one is
    already promoted on disk it is left untouched and its version returned; otherwise a fresh
    LoRA-on-Qwen3-4B candidate is trained, gauntlet-gated and promoted. Returns the active
    version, or ``None`` when nothing was/could be promoted. **Never raises** — boot integrity
    comes first; on any failure the LLM facade's mock fallback keeps NYXARA responsive.
    """
    settings = settings or get_settings()
    say = log or (lambda _msg: None)
    # Forge a primary brain only when NYXARA's OWN model is the selected primary provider.
    if settings.llm.provider is not LLMProvider.SELF:
        return None
    if primary_model_present(settings):
        return _active_version(settings)
    try:
        return _forge(settings, base_model=base_model, generations=generations,
                      seed_corpus=seed_corpus, say=say)
    except Exception as exc:  # noqa: BLE001 — boot must survive a failed forge
        say(f"primary brain      : could not forge ({exc}); falling back for now")
        return None


def _forge(settings: NyxaraSettings, *, base_model: Optional[str], generations: int,
           seed_corpus: Optional[List[str]], say: Callable[[str], None]) -> Optional[int]:
    """Train + gauntlet + promote one LoRA-on-Qwen3-4B candidate (lazy heavy imports)."""
    from nyxara.growth.foundry import Foundry
    from nyxara.growth.foundry_models import _HAS_LORA

    # Work on a deep copy so global config is never mutated; the on-disk foundry path is the
    # same one SelfProvider reads from, so a promotion here is visible to the live LLM.
    cfg = settings.model_copy(deep=True)
    cfg.foundry.enabled = True
    cfg.foundry.backend = "lora"
    base = base_model or (QWEN3_4B if cfg.foundry.base_model == _TINY_DEFAULT_BASE
                          else cfg.foundry.base_model)
    cfg.foundry.base_model = base
    # Request QLoRA (4-bit): honoured only when bitsandbytes + CUDA are present, so a 4B base
    # fine-tunes on one consumer GPU; harmlessly ignored (full-precision/n-gram) on CPU/CI.
    cfg.foundry.load_in_4bit = True

    if _HAS_LORA:
        say(f"primary brain      : forging a LoRA on {base} "
            "(weights download on first use, then cache)…")
    else:
        say("primary brain      : .[foundry] absent — forging the always-on n-gram brain "
            "(install .[foundry] for the real Qwen3-4B LoRA)…")

    foundry = Foundry(settings=cfg, seed_corpus=seed_corpus or IDENTITY_SEED)
    results = foundry.self_improve(generations=max(1, generations))
    if any(r.promoted for r in results) and foundry.active_version is not None:
        say(f"primary brain      : forged & promoted v{foundry.active_version} ✓")
        return foundry.active_version
    reason = results[-1].reason if results else "no candidate produced"
    say(f"primary brain      : no model promoted ({reason})")
    return None
