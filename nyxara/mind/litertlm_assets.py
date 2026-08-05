"""NYXARA · mind/litertlm_assets.py — the weights behind her PRIMARY brain (⬆).

``mind/llm.py::LiteRTLMProvider`` serves Gemma-4-E2B-it from a single on-device file. This module is
the only thing that knows how that file *gets there*, and it is deliberately the whole story:

    where the weights live  →  :func:`model_path`
    are they here yet?      →  :func:`model_present`
    put them here           →  :func:`ensure_litertlm_model`

Three properties are load-bearing:

* **It never lies about readiness.** The download lands in ``<name>.part`` and is moved into place
  with a single atomic ``os.replace`` only after the last byte arrives. A killed process therefore
  leaves either no file or a complete one — never a truncated 2.4 GB file that ``available()`` would
  cheerfully report as a working model.
* **It never surprises the host.** A 2.4 GB fetch is a real event, so it happens only when
  ``llm.litertlm_auto_download`` is on, and *never* under the TEST profile — a hermetic suite must
  not reach the network or the disk budget. Everything else returns ``None`` honestly.
* **It never breaks the boot.** Every failure path returns ``None`` and logs; the ladder in
  ``mind/llm.py`` then simply starts one rung lower.

Fetching prefers ``huggingface_hub`` (already present transitively via ``transformers``) for its
cache and resume handling, and falls back to a plain ranged ``httpx`` stream, so a machine with
neither extra dependency still works.

Run it directly to fetch the weights with progress::

    python -m nyxara.mind.litertlm_assets
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Callable, Optional

from nyxara.kernel.config import NyxaraSettings, Profile, get_settings

log = logging.getLogger("nyxara.mind.litertlm_assets")

__all__ = ["model_path", "model_present", "expected_url", "ensure_litertlm_model"]

# Progress reporter: (bytes_so_far, total_bytes_or_None) -> None
ProgressFn = Callable[[int, Optional[int]], None]

_CHUNK = 1024 * 1024        # 1 MiB — big enough that the syscalls disappear next to the network


def model_path(settings: Optional[NyxaraSettings] = None) -> Path:
    """Where her primary weights live: the configured path, else ``data_dir/models/<filename>``."""
    settings = settings or get_settings()
    configured = getattr(settings.llm, "litertlm_model_path", None)
    if configured:
        return Path(configured)
    return Path(settings.paths.data_dir) / "models" / settings.llm.litertlm_filename


def model_present(settings: Optional[NyxaraSettings] = None) -> bool:
    """True iff the weights are on disk. Mirrors ``growth/bootstrap.primary_model_present``."""
    try:
        return model_path(settings).is_file()
    except Exception:  # noqa: BLE001 — an unreadable path is simply "not present"
        return False


def expected_url(settings: Optional[NyxaraSettings] = None) -> str:
    """The HuggingFace resolve URL for the configured repo/file."""
    settings = settings or get_settings()
    return (f"https://huggingface.co/{settings.llm.litertlm_repo_id}"
            f"/resolve/main/{settings.llm.litertlm_filename}")


# --------------------------------------------------------------------------- #
# Fetch backends (both land on a .part file; the caller does the atomic move)
# --------------------------------------------------------------------------- #
def _fetch_via_hub(settings: NyxaraSettings, dest: Path, progress: Optional[ProgressFn]) -> bool:
    """Download through ``huggingface_hub`` (cache + resume for free). False if unusable."""
    try:
        from huggingface_hub import hf_hub_download
    except Exception:  # noqa: BLE001 — optional; the httpx path covers it
        return False
    try:
        cached = hf_hub_download(repo_id=settings.llm.litertlm_repo_id,
                                 filename=settings.llm.litertlm_filename)
    except Exception as exc:  # noqa: BLE001 — fall through to the plain HTTP path
        log.debug("huggingface_hub fetch failed (%s); falling back to httpx", exc)
        return False
    # Copy rather than symlink: the hub cache may be pruned, and her weights must outlive it.
    size = os.path.getsize(cached)
    with open(cached, "rb") as src, open(dest, "wb") as out:
        done = 0
        while True:
            chunk = src.read(_CHUNK)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if progress is not None:
                progress(done, size)
    return True


def _fetch_via_http(settings: NyxaraSettings, dest: Path, progress: Optional[ProgressFn]) -> bool:
    """Stream the file over HTTPS, resuming an interrupted ``.part`` with a Range request."""
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return False
    url = expected_url(settings)
    have = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, read=300.0)) as client:
        with client.stream("GET", url, headers=headers) as resp:
            if have and resp.status_code == 200:
                have = 0                      # server ignored the range: start over, honestly
            elif have and resp.status_code != 206:
                resp.raise_for_status()
            else:
                resp.raise_for_status()
            total = resp.headers.get("content-length")
            total = (int(total) + have) if total is not None else None
            mode = "ab" if have else "wb"
            done = have
            with open(dest, mode) as out:
                for chunk in resp.iter_bytes(_CHUNK):
                    out.write(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(done, total)
    return True


# --------------------------------------------------------------------------- #
# The one entry point
# --------------------------------------------------------------------------- #
def ensure_litertlm_model(
    settings: Optional[NyxaraSettings] = None,
    *,
    progress: Optional[ProgressFn] = None,
    force: bool = False,
) -> Optional[Path]:
    """Make sure her primary weights are on disk; return the path, or ``None``.

    Returns immediately when the file is already there (the common case — this is called on every
    boot). Declines silently, returning ``None``, when auto-download is off or the profile is TEST.
    **Never raises**: a failed fetch is logged, the partial file is cleaned up, and the caller's boot
    carries on one rung lower down the ladder.
    """
    settings = settings or get_settings()
    dest = model_path(settings)

    if dest.is_file() and not force:
        return dest
    if settings.profile is Profile.TEST:
        return None                       # a hermetic suite never fetches 2.4 GB
    if not bool(getattr(settings.llm, "litertlm_auto_download", False)):
        log.info("litertlm weights missing at %s and auto-download is off — "
                 "run scripts/fetch_litertlm_model.py to install them", dest)
        return None
    if not bool(getattr(settings.llm, "litertlm_enabled", True)):
        return None

    part = dest.with_name(dest.name + ".part")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if force and part.exists():
            part.unlink()
        log.info("fetching her primary brain (%s) -> %s",
                 settings.llm.litertlm_repo_id, dest)
        ok = _fetch_via_hub(settings, part, progress) or _fetch_via_http(settings, part, progress)
        if not ok:
            log.warning("no way to fetch litertlm weights (need huggingface_hub or httpx)")
            return None
        os.replace(part, dest)            # atomic: the file appears complete or not at all
        log.info("litertlm weights ready: %s (%d bytes)", dest, dest.stat().st_size)
        return dest
    except Exception as exc:  # noqa: BLE001 — boot integrity comes first
        log.warning("could not fetch litertlm weights: %s", exc)
        try:
            if part.exists():
                part.unlink()             # never leave a truncated file behind
        except Exception:  # noqa: BLE001
            pass
        return None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cli_progress(done: int, total: Optional[int]) -> None:  # pragma: no cover
    gb = done / (1024 ** 3)
    if total:
        pct = 100.0 * done / total
        print(f"\r  {gb:5.2f} GB / {total / (1024 ** 3):.2f} GB  ({pct:5.1f}%)", end="", flush=True)
    else:
        print(f"\r  {gb:5.2f} GB", end="", flush=True)


def _doctor(settings: NyxaraSettings) -> int:  # pragma: no cover
    """Report exactly which stage of the on-device rung is broken, and with what.

    Exists because the failure that matters most is the least informative: the runtime returns a
    bare ``litert_lm_conversation_send_message failed`` and keeps the real reason to itself. This
    walks the stages in order — binding, native library, weights, engine, then each chat template —
    and prints what happened at each, so a broken host produces a diagnosis instead of a guess.
    """
    from nyxara.mind.llm import LiteRTLMProvider, LLMRequest
    from nyxara.mind.vulkan_shim import loader_present

    print("NYXARA · on-device brain doctor")
    print("=" * 62)
    import platform
    import sys
    print(f"python       : {sys.version.split()[0]}  ({platform.machine()})")
    try:
        import importlib.metadata as md
        print(f"litert-lm-api: {md.version('litert-lm-api')}")
    except Exception as exc:  # noqa: BLE001
        print(f"litert-lm-api: NOT INSTALLED ({exc})")
        print("\n-> pip install litert-lm-api")
        return 1

    prov = LiteRTLMProvider(settings)
    print(f"vulkan loader: {'system' if loader_present() else 'absent (shim will cover CPU)'}")

    native = prov.native_library_error()
    print(f"runtime      : {'OK' if native is None else 'FAILED — ' + native}")
    if native is not None:
        return 1

    path = prov.model_path()
    print(f"weights      : {'OK ' + str(path) if path.is_file() else 'MISSING at ' + str(path)}")
    if not path.is_file():
        print("\n-> python scripts/fetch_litertlm_model.py")
        return 1

    import litert_lm
    try:
        engine = prov._get_engine(litert_lm)
        print("engine       : OK (loaded)")
    except Exception as exc:  # noqa: BLE001
        print(f"engine       : FAILED — {exc}")
        return 1

    print("\nchat templates (the usual culprit — the runtime hides the real reason):")
    working = None
    for name, template in prov._templates():
        try:
            kwargs = {} if template is None else {"chat_template": template}
            conv = engine.create_conversation(**kwargs)
            try:
                resp = conv.send_message(litert_lm.Message.user("What is 2+2? Number only."),
                                         max_output_tokens=8)
            finally:
                conv.close()
            text = prov._extract_text(resp).strip()
            print(f"  {name:16s} OK   -> {text!r}")
            working = working or name
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:16s} FAIL -> {exc}")

    print()
    if working is None:
        print("VERDICT: the engine runs but no chat template this build accepts renders a turn.")
        print("         Please send this output — the template is the thing to change.")
        return 1

    print(f"VERDICT: working. She will use the '{working}' template.")
    try:
        out = prov.complete(LLMRequest.from_prompt("Say hello in one short sentence.",
                                                   max_tokens=24, seed=7))
        print(f"         end-to-end: [{out.provider}] {out.text!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"         end-to-end FAILED anyway: {exc}")
        return 1
    return 0


def main() -> int:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Fetch NYXARA's primary on-device brain "
                                                 "(Gemma-4-E2B-it, LiteRT-LM format).")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--quiet", action="store_true", help="no progress line")
    parser.add_argument("--doctor", action="store_true",
                        help="diagnose a rung that loads but will not answer")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()
    if args.doctor:
        return _doctor(settings)
    dest = model_path(settings)
    if dest.is_file() and not args.force:
        print(f"already present: {dest} ({dest.stat().st_size / (1024 ** 3):.2f} GB)")
        # Still report the runtime: "weights present, runtime unusable" is exactly the state that
        # otherwise only shows up as a warning on every turn.
        return _report_runtime()

    free = shutil.disk_usage(dest.parent if dest.parent.exists() else Path.home()).free
    print(f"repo   : {settings.llm.litertlm_repo_id}")
    print(f"target : {dest}")
    print(f"free   : {free / (1024 ** 3):.1f} GB")
    # ``litertlm_auto_download`` gates the automatic boot path; asking by hand IS the consent.
    settings.llm.litertlm_auto_download = True
    out = ensure_litertlm_model(settings, force=args.force,
                                progress=None if args.quiet else _cli_progress)
    print()
    if out is None:
        print("FAILED — see the log above")
        return 1
    print(f"ready: {out} ({out.stat().st_size / (1024 ** 3):.2f} GB)")
    return _report_runtime()


def _report_runtime(settings: Optional[NyxaraSettings] = None) -> int:  # pragma: no cover
    """Say plainly whether the runtime can actually start here — weights alone are not enough.

    The failure this catches is a quiet one: ``pip install litert-lm-api`` succeeds, the weights
    download, and only at the first turn does the shared library fail to dlopen because the host has
    no Vulkan loader. Better to say so now — and, where the shim can simply remove the problem, to
    say that instead — than to leave a warning on every turn.
    """
    from nyxara.mind.llm import LiteRTLMProvider
    from nyxara.mind.vulkan_shim import loader_present

    settings = settings or get_settings()
    had_loader = loader_present()
    problem = LiteRTLMProvider(settings).native_library_error()
    if problem is None:
        print("runtime: OK — she will draft on this model")
        if not had_loader:
            print("         (no system Vulkan loader here; a built-in stub covers the CPU backend "
                  "— `sudo apt install libvulkan1` if you would rather have the real one)")
        return 0
    print(f"runtime: NOT USABLE YET — {problem}")
    print("         (the weights are fine; until this is fixed her ladder starts at the cloud)")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
