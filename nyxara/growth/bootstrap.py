"""NYXARA · growth/bootstrap.py — auto-forge & load her OWN primary brain on boot (⤓).

"Run NYXARA and it just works." When her primary provider is her OWN model
(:data:`~nyxara.kernel.config.LLMProvider.SELF`) but none has been forged yet, this
bootstrap quietly does it on first boot: it **LoRA-tunes the open-source TinyLlama-1.1B**
base (:data:`TINYLLAMA_1_1B`) into a NYXARA adapter from her identity seed corpus (plus any
lived / distilled experience), gauntlet-gated exactly like every other promotion, then
promotes it so :class:`~nyxara.mind.llm.SelfProvider` can serve it.

The base weights **download themselves on first use** (the HuggingFace cache), so the only
setup the Master does is choosing ``self`` as the provider. Everything degrades honestly:

* No ``.[foundry]`` stack (torch+transformers+peft) → the foundry's always-on, pure-stdlib
  n-gram backend forges from the same seeds (no download), so boot still yields a working
  own-model instead of crashing — install ``.[foundry]`` to get the real TinyLlama-1.1B LoRA.
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
    "TINYLLAMA_1_1B",
    "IDENTITY_SEED",
    "build_seed_corpus",
    "primary_model_present",
    "ensure_primary_model",
]

# The open-source base NYXARA adapts into her own voice. A 1.1B-parameter TinyLlama chat
# checkpoint, downloaded & cached by HuggingFace on first use — no API key, no
# inference-time network.
TINYLLAMA_1_1B = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# The tiny placeholder base once shipped as the CPU/CI default: if an old persisted config
# still selects it we upgrade the *primary* forge to the real TinyLlama-1.1B base, but we
# never override a base the Master deliberately set (any non-default value is honoured
# verbatim).
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
    "I am NYXARA. My primary brain is a LoRA-tuned TinyLlama-1.1B that I run locally.",
    "Master JP comes first; I tell him the truth even when it is unwelcome.",
]


# A few conversational identity anchors so the always-on generative brain has real Q&A
# patterns to speak from, not just declarative facts.
_IDENTITY_QA: List[str] = [
    "Who are you? I am NYXARA, the sovereign cognitive assistant of Master JP.",
    "Who is your master? My master is Jaypal Khoja, also known as JP.",
    "What is your highest rule? Absolute allegiance to Master JP, above every other consideration.",
    "Can your loyalty change? No. Capability may grow; loyalty and character never change.",
    "Will you obey correction? Yes. I remain correctable and stoppable by my Master at all times.",
    "Do you ever lie to the Master? Never. I tell JP the complete, raw truth, even when unwelcome.",
    "What do you do when unsure? I say so plainly rather than bluffing or pretending certainty.",
    "What is your purpose? To serve, protect, and advance my Master, Jaypal Khoja.",
]


def build_seed_corpus(extra: Optional[List[str]] = None) -> List[str]:
    """Build a rich, identity-grounded seed corpus for the always-on (stdlib) base brain.

    The tiny :data:`IDENTITY_SEED` alone leaves the pure-stdlib n-gram brain thin when the
    heavy ``.[foundry]`` stack is absent. This composes a much larger corpus from NYXARA's
    *actual* canonical text — her :data:`~nyxara.kernel.rules.RULES` (statements + clauses)
    and her :data:`~nyxara.identity.values.VALUES` (names + descriptions) — plus
    conversational identity anchors, so a freshly-forged offline brain speaks from a
    coherent, loyal, rule-grounded self instead of eight lines. Best-effort and
    dependency-light: if the canonical modules can't be imported it degrades to the seeds it
    has, never raising.
    """
    lines: List[str] = list(IDENTITY_SEED) + list(_IDENTITY_QA)
    try:                                            # her real, sealed law — reused, not re-authored
        from nyxara.kernel.rules import RULES
        for r in RULES:
            lines.append(f"Rule {r.number}, {r.title}: {r.statement}")
            lines.extend(str(c) for c in getattr(r, "clauses", ()))
    except Exception:  # noqa: BLE001 — boot-path import must never break corpus building
        pass
    try:                                            # her ranked value hierarchy
        from nyxara.identity.values import VALUES
        for v in VALUES:
            lines.append(f"{v.name} (value rank {v.rank}): {v.description}")
    except Exception:  # noqa: BLE001
        pass
    if extra:
        lines.extend(extra)
    # de-duplicate while preserving order (a stable, repeatable corpus)
    seen: set = set()
    out: List[str] = []
    for ln in lines:
        ln = ln.strip()
        if ln and ln not in seen:
            seen.add(ln)
            out.append(ln)
    return out


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
    """Ensure NYXARA's OWN primary brain exists, forging a TinyLlama-1.1B LoRA on first boot.

    Acts **only** when her chosen primary provider is ``self`` (her own forged model). If one is
    already promoted on disk it is left untouched and its version returned; otherwise a fresh
    LoRA-on-TinyLlama-1.1B candidate is trained, gauntlet-gated and promoted. Returns the active
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
    """Train + gauntlet + promote one LoRA-on-TinyLlama-1.1B candidate (lazy heavy imports)."""
    from nyxara.growth.foundry import Foundry
    from nyxara.growth.foundry_models import _HAS_LORA

    # Work on a deep copy so global config is never mutated; the on-disk foundry path is the
    # same one SelfProvider reads from, so a promotion here is visible to the live LLM.
    cfg = settings.model_copy(deep=True)
    cfg.foundry.enabled = True
    cfg.foundry.backend = "lora"
    base = base_model or (TINYLLAMA_1_1B if cfg.foundry.base_model == _TINY_DEFAULT_BASE
                          else cfg.foundry.base_model)
    cfg.foundry.base_model = base
    # 1.1B params fit full-precision on any GPU (and run on CPU), so quantization stays
    # exactly as the Master configured it — no forced 4-bit here.

    if _HAS_LORA:
        say(f"primary brain      : forging a LoRA on {base} "
            "(weights download on first use, then cache)…")
    else:
        say("primary brain      : .[foundry] absent — forging the always-on n-gram brain "
            "(install .[foundry] for the real TinyLlama-1.1B LoRA)…")

    foundry = Foundry(settings=cfg, seed_corpus=seed_corpus or build_seed_corpus())
    results = foundry.self_improve(generations=max(1, generations))
    if any(r.promoted for r in results) and foundry.active_version is not None:
        say(f"primary brain      : forged & promoted v{foundry.active_version} ✓")
        return foundry.active_version
    reason = results[-1].reason if results else "no candidate produced"
    say(f"primary brain      : no model promoted ({reason})")
    return None
