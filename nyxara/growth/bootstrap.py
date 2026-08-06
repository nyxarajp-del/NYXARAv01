"""NYXARA · growth/bootstrap.py — auto-forge & load her OWN primary brain on boot (⤓).

"Run NYXARA and it just works." When her primary provider is her OWN model
(:data:`~nyxara.kernel.config.LLMProvider.SELF`) but none has been forged yet, this
bootstrap quietly does it on first boot: it **LoRA-tunes the open-source DistilGPT-2**
base (:data:`DISTILGPT2`) into a NYXARA adapter from her identity seed corpus (plus any
lived / distilled experience), gauntlet-gated exactly like every other promotion, then
promotes it so :class:`~nyxara.mind.llm.SelfProvider` can serve it.

The base weights **download themselves on first use** (the HuggingFace cache), so the only
setup the Master does is choosing ``self`` as the provider. Everything degrades honestly:

* No ``.[foundry]`` stack (torch+transformers+peft) → the foundry's always-on, pure-stdlib
  n-gram backend forges from the same seeds (no download), so boot still yields a working
  own-model instead of crashing — install ``.[foundry]`` to get the real DistilGPT-2 LoRA.
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
    "DISTILGPT2",
    "IDENTITY_SEED",
    "build_seed_corpus",
    "primary_model_present",
    "ensure_primary_model",
]

# The single open-source base NYXARA adapts into her own voice: DistilGPT-2 — an ~82M-parameter
# distilled GPT-2 checkpoint, downloaded & cached by HuggingFace on first use (no API key,
# no inference-time network). Small enough to LoRA-fine-tune and run full-precision on a bare
# CPU (no GPU, no quantization); on a torch-less box the forge degrades to the always-on n-gram brain.
DISTILGPT2 = "distilgpt2"

# The tiny placeholder base once shipped as the CPU/CI default: if an old persisted config
# still selects it we upgrade the *primary* forge to the real DistilGPT-2 base, but we
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
    "I am NYXARA. My primary brain is a LoRA-tuned DistilGPT-2 that I run locally.",
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
    """Ensure NYXARA's OWN brains are on disk, forging a DistilGPT-2 LoRA on first boot.

    Two brains, in ladder order. First her PRIMARY one: the on-device LiteRT-LM Gemma
    (``mind/litertlm_assets.ensure_litertlm_model``) is fetched when its weights are missing and
    auto-download is on — a one-time ~2.4 GB download that turns the top rung of the ``auto`` ladder
    from unavailable into her strongest voice. Then her FORGED one: a LoRA is trained and promoted
    when ``self`` is the provider that would actually serve.

    The foundry half acts **only** when her chosen provider is ``self`` (her own forged model). If one
    is already promoted on disk it is left untouched and its version returned; otherwise a fresh
    LoRA-on-DistilGPT-2 candidate is trained, gauntlet-gated and promoted. Returns the active
    version, or ``None`` when nothing was/could be promoted. **Never raises** — boot integrity
    comes first; on any failure the LLM facade's ladder keeps NYXARA responsive.
    """
    settings = settings or get_settings()
    say = log or (lambda _msg: None)
    _ensure_litertlm(settings, say)
    # Forge a primary brain when NYXARA's OWN model is the selected primary provider —
    # explicitly (`self`), or via the default `auto` ladder where `self` is the first rung.
    # Under `auto` the forge only runs when the result would actually SERVE (the honesty
    # serve gate): a real LoRA stack is present, or the Master opted any backend in.
    if settings.llm.provider is LLMProvider.AUTO:
        if not bool(getattr(settings.llm, "self_serve_any_backend", False)):
            try:
                from nyxara.growth.foundry_models import _HAS_LORA
            except Exception:  # noqa: BLE001 — no foundry stack at all
                _HAS_LORA = False
            if not _HAS_LORA:
                return None
    elif settings.llm.provider is not LLMProvider.SELF:
        return None
    if primary_model_present(settings):
        return _active_version(settings)
    try:
        return _forge(settings, base_model=base_model, generations=generations,
                      seed_corpus=seed_corpus, say=say)
    except Exception as exc:  # noqa: BLE001 — boot must survive a failed forge
        say(f"primary brain      : could not forge ({exc}); falling back for now")
        return None


def _ensure_litertlm(settings: NyxaraSettings, say: Callable[[str], None]) -> None:
    """Put her PRIMARY on-device brain on disk if it isn't already. Best-effort, never raises.

    Present weights make this a no-op file check, which is the normal boot. When they are absent it
    is a ~2.4 GB fetch, so it is gated by ``llm.litertlm_auto_download`` and never runs under TEST —
    the asset module enforces both. Failure is not fatal: the ``auto`` ladder simply starts one rung
    lower, at her cloud tools, and she keeps her voice either way.
    """
    try:
        from nyxara.mind.litertlm_assets import ensure_litertlm_model, model_present
        if model_present(settings):
            _warm_litertlm(settings, say)
            return
        if not bool(getattr(settings.llm, "litertlm_auto_download", False)):
            return
        say("primary brain      : fetching her on-device model (~2.4 GB, one time)…")
        got = ensure_litertlm_model(settings)
        say(f"primary brain      : {'ready at ' + str(got) if got else 'not fetched; using the cloud rungs for now'}")
        if got:
            _warm_litertlm(settings, say)
    except Exception as exc:  # noqa: BLE001 — boot must survive a failed fetch
        say(f"primary brain      : on-device fetch skipped ({exc})")


def _warm_litertlm(settings: NyxaraSettings, say: Callable[[str], None]) -> None:
    """Load the on-device weights at boot rather than inside the Master's first message.

    ``Engine.__init__`` reads 2.5 GB and takes ~95 s. Built lazily, that cost landed in whatever
    turn happened to come first — measured as 95.2 s of a 122.7 s "Hii". It is the same work
    either way; boot is simply the honest place to do it, where he is already watching a progress
    line. Best-effort: a host that cannot start the runtime falls down the ladder exactly as it
    would have on the first turn.
    """
    import time as _time
    try:
        from nyxara.mind.llm import LiteRTLMProvider
        provider = LiteRTLMProvider(settings=settings)
        if not provider.available():
            return
        say("primary brain      : loading her weights (~2.5 GB, once per boot)…")
        started = _time.monotonic()
        if provider.warm():
            say(f"primary brain      : ready in {_time.monotonic() - started:.0f}s")
        else:
            say("primary brain      : could not start the runtime; using the rungs below it")
    except Exception as exc:  # noqa: BLE001 — warming is an optimisation, never a boot failure
        say(f"primary brain      : warm-up skipped ({exc})")


def _forge(settings: NyxaraSettings, *, base_model: Optional[str], generations: int,
           seed_corpus: Optional[List[str]], say: Callable[[str], None]) -> Optional[int]:
    """Train + gauntlet + promote one LoRA-on-DistilGPT-2 candidate (lazy heavy imports)."""
    from nyxara.growth.foundry import Foundry
    from nyxara.growth.foundry_models import _HAS_LORA

    # Work on a deep copy so global config is never mutated; the on-disk foundry path is the
    # same one SelfProvider reads from, so a promotion here is visible to the live LLM.
    cfg = settings.model_copy(deep=True)
    cfg.foundry.enabled = True
    cfg.foundry.backend = "lora"
    # Record the REAL LoRA spec even on a CPU box (build_model degrades at build time, so a
    # GPU machine can later rebuild the very same adapter from the recorded base) — the
    # autonomous-loop lora_requires_gpu clamp is for unattended forges, not this deliberate one.
    cfg.foundry.lora_requires_gpu = False
    base = base_model or (DISTILGPT2 if cfg.foundry.base_model == _TINY_DEFAULT_BASE
                          else cfg.foundry.base_model)
    cfg.foundry.base_model = base
    # Quantization stays exactly as the Master configured it — honoured only on a CUDA box with
    # bitsandbytes, else the LoRA loads full-precision (QLoRA 4-bit is the default on a CUDA box).

    if _HAS_LORA:
        say(f"primary brain      : forging a LoRA on {base} "
            "(weights download on first use, then cache)…")
    else:
        say(f"primary brain      : .[foundry] absent — forging the always-on n-gram brain "
            f"(install .[foundry] for the real {base} LoRA)…")

    foundry = Foundry(settings=cfg, seed_corpus=seed_corpus or build_seed_corpus())
    results = foundry.self_improve(generations=max(1, generations))
    if any(r.promoted for r in results) and foundry.active_version is not None:
        say(f"primary brain      : forged & promoted v{foundry.active_version} ✓")
        return foundry.active_version
    reason = results[-1].reason if results else "no candidate produced"
    say(f"primary brain      : no model promoted ({reason})")
    return None
