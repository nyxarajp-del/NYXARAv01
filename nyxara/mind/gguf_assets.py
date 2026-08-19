"""NYXARA · mind/gguf_assets.py — how the Qwythos weights get onto her disk (⬇).

The sibling of :mod:`nyxara.mind.litertlm_assets`, for the second set of on-device weights:

    where they live      →  :func:`model_path`
    are they here yet?   →  :func:`model_present`
    put them here        →  :func:`ensure_gguf_model`
    is it the right file →  :func:`verify_checksum`

The three properties that module makes load-bearing are load-bearing here too, and for the same
reasons: the download lands in ``<name>.part`` and is moved into place with a single atomic
``os.replace``, so a killed process leaves either no file or a complete one and never a truncated
5.6 GB file that ``available()`` would report as a working model; it never fetches under the TEST
profile, because a hermetic suite must not reach the network or the disk budget; and every failure
path returns ``None`` and logs, so a dead network cannot break a boot.

**Two things are different here, and both are deliberate.**

*Auto-download is off by default.* ``litertlm_auto_download`` is ``True`` because 2.4 GB is a
tolerable surprise. These files are 5.63 GB (Q4_K_M) to 17.92 GB (BF16), which on a metered
connection is not. Rule 6 cuts against helpfulness here: the fetch is a decision, and
``scripts/fetch_qwythos_model.py`` is the door.

*The checksum is verified.* The repository publishes ``SHA256SUMS``, so "the file is complete"
can be upgraded from "the stream ended" to "the bytes are the ones that were published".
Atomicity only protects against truncation; it says nothing about a corrupted middle or a proxy
that served something else. A graft reads these bytes as weights and a wrong file would convert
cleanly into a layer that computes nonsense — with a fidelity certificate attached, because it
would be internally consistent. Checking the hash is what keeps that certificate meaningful.

Run it directly to fetch::

    python -m nyxara.mind.gguf_assets --quant Q4_K_M
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Callable, Dict, Optional

from nyxara.kernel.config import NyxaraSettings, Profile, get_settings

log = logging.getLogger("nyxara.mind.gguf_assets")

__all__ = ["model_path", "model_present", "expected_url", "available_quants",
           "filename_for_quant", "fetch_checksums", "verify_checksum", "ensure_gguf_model"]

ProgressFn = Callable[[int, Optional[int]], None]

_CHUNK = 1024 * 1024

#: The quantisations this repository publishes, with their sizes, so a caller can be told what a
#: choice costs before it starts rather than after 5 GB of it has arrived.
QUANTS: Dict[str, str] = {
    "Q4_K_M": "5.63 GB — the model card's recommended default",
    "Q5_K_M": "6.47 GB — balanced quality / size",
    "Q6_K": "7.36 GB — high quality",
    "Q8_0": "9.53 GB — near-lossless",
    "BF16": "17.92 GB — full precision, the conversion base",
}

_STEM = "Qwythos-9B-Claude-Mythos-5-1M"


def available_quants() -> Dict[str, str]:
    return dict(QUANTS)


def filename_for_quant(quant: str) -> str:
    """``Q4_K_M`` -> the published filename. Raises on a quant this repo does not carry."""
    q = str(quant).upper()
    if q not in QUANTS:
        raise ValueError(f"unknown quant {quant!r}; have {', '.join(sorted(QUANTS))}")
    return f"{_STEM}-{q}.gguf"


def model_path(settings: Optional[NyxaraSettings] = None) -> Path:
    """Where the teacher's weights live: the configured path, else ``data_dir/models/<file>``."""
    settings = settings or get_settings()
    configured = getattr(settings.llm, "gguf_model_path", None)
    if configured:
        return Path(configured)
    return Path(settings.paths.data_dir) / "models" / settings.llm.gguf_filename


def model_present(settings: Optional[NyxaraSettings] = None) -> bool:
    """True iff the weights are on disk. Never raises — an unreadable path is 'not present'."""
    try:
        return model_path(settings).is_file()
    except Exception:  # noqa: BLE001
        return False


def expected_url(settings: Optional[NyxaraSettings] = None, *,
                 filename: Optional[str] = None) -> str:
    settings = settings or get_settings()
    name = filename or settings.llm.gguf_filename
    return f"https://huggingface.co/{settings.llm.gguf_repo_id}/resolve/main/{name}"


# --------------------------------------------------------------------------- #
# Checksums
# --------------------------------------------------------------------------- #
def fetch_checksums(settings: Optional[NyxaraSettings] = None) -> Dict[str, str]:
    """``{filename: sha256}`` from the repository's ``SHA256SUMS``, or ``{}``.

    Empty is a normal answer — no network, no ``httpx``, no such file — and it degrades
    :func:`verify_checksum` to "unverified" rather than to "failed". Reporting a file as corrupt
    because the checksum list could not be reached would be the same error in the other
    direction.
    """
    settings = settings or get_settings()
    if settings.profile is Profile.TEST:
        return {}
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return {}
    try:
        url = expected_url(settings, filename="SHA256SUMS")
        resp = httpx.get(url, follow_redirects=True, timeout=30.0)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        log.debug("could not fetch SHA256SUMS (%s) — the download will be unverified", exc)
        return {}
    out: Dict[str, str] = {}
    for line in resp.text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 64:
            out[parts[-1].lstrip("*")] = parts[0].lower()
    return out


def sha256_file(path: Path, *, progress: Optional[ProgressFn] = None) -> str:
    """Streamed SHA-256. Never loads a multi-GB file into memory."""
    digest = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            done += len(chunk)
            if progress is not None:
                progress(done, total)
    return digest.hexdigest()


def verify_checksum(path: Path, settings: Optional[NyxaraSettings] = None, *,
                    checksums: Optional[Dict[str, str]] = None,
                    progress: Optional[ProgressFn] = None) -> Optional[bool]:
    """``True`` verified, ``False`` mismatched, ``None`` no published hash to check against.

    Three states rather than two, because "I could not check" and "I checked and it is wrong" are
    completely different facts and collapsing them would either reject good files offline or
    accept bad ones silently.
    """
    try:
        sums = checksums if checksums is not None else fetch_checksums(settings)
        want = sums.get(path.name)
        if not want:
            return None
        return sha256_file(path, progress=progress) == want
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Fetch backends — both land on a .part file; the caller does the atomic move
# --------------------------------------------------------------------------- #
def _fetch_via_hub(settings: NyxaraSettings, dest: Path, filename: str,
                   progress: Optional[ProgressFn]) -> bool:
    try:
        from huggingface_hub import hf_hub_download
    except Exception:  # noqa: BLE001 — optional; the httpx path covers it
        return False
    try:
        cached = hf_hub_download(repo_id=settings.llm.gguf_repo_id, filename=filename)
    except Exception as exc:  # noqa: BLE001
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


def _fetch_via_http(settings: NyxaraSettings, dest: Path, filename: str,
                    progress: Optional[ProgressFn]) -> bool:
    """Stream over HTTPS, resuming an interrupted ``.part`` with a Range request."""
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return False
    url = expected_url(settings, filename=filename)
    have = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, read=300.0)) as client:
        with client.stream("GET", url, headers=headers) as resp:
            if have and resp.status_code == 200:
                have = 0                  # server ignored the range: start over, honestly
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
def ensure_gguf_model(
    settings: Optional[NyxaraSettings] = None,
    *,
    quant: Optional[str] = None,
    progress: Optional[ProgressFn] = None,
    verify: bool = True,
    force: bool = False,
) -> Optional[Path]:
    """Make sure the teacher's weights are on disk; return the path, or ``None``.

    Never raises. Declines silently under the TEST profile or with auto-download off. A file whose
    published checksum does not match is **deleted and refused** rather than kept: these bytes are
    read as weights, and a wrong file converts into a layer that computes confident nonsense.
    """
    settings = settings or get_settings()
    filename = filename_for_quant(quant) if quant else settings.llm.gguf_filename
    dest = (model_path(settings) if not quant
            else Path(settings.paths.data_dir) / "models" / filename)

    if dest.is_file() and not force:
        return dest
    if settings.profile is Profile.TEST:
        return None                       # a hermetic suite never fetches gigabytes
    if not force and not bool(getattr(settings.llm, "gguf_auto_download", False)):
        log.info("qwythos weights missing at %s and auto-download is off — "
                 "run scripts/fetch_qwythos_model.py to install them", dest)
        return None
    if not bool(getattr(settings.llm, "gguf_enabled", True)):
        return None

    part = dest.with_name(dest.name + ".part")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if force and part.exists():
            part.unlink()
        log.info("fetching the qwythos teacher (%s/%s) -> %s",
                 settings.llm.gguf_repo_id, filename, dest)
        ok = (_fetch_via_hub(settings, part, filename, progress)
              or _fetch_via_http(settings, part, filename, progress))
        if not ok:
            log.warning("no way to fetch gguf weights (need huggingface_hub or httpx)")
            return None
        os.replace(part, dest)            # atomic: complete or absent, never truncated
        if verify:
            verdict = verify_checksum(dest, settings)
            if verdict is False:
                log.error("checksum MISMATCH for %s — refusing the file and removing it", dest)
                try:
                    dest.unlink()
                except Exception:  # noqa: BLE001
                    pass
                return None
            log.info("checksum %s", "verified" if verdict else "unavailable (unverified)")
        log.info("qwythos weights ready: %s (%d bytes)", dest, dest.stat().st_size)
        return dest
    except Exception as exc:  # noqa: BLE001 — boot integrity comes first
        log.warning("could not fetch gguf weights: %s", exc)
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


def main(argv: Optional[list] = None) -> int:  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description="Fetch the Qwythos teacher weights.")
    ap.add_argument("--quant", default=None, choices=sorted(QUANTS),
                    help="which quantisation to fetch (default: whatever config names)")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--no-verify", action="store_true", help="skip the SHA256SUMS check")
    args = ap.parse_args(argv)

    settings = get_settings()
    for name, size in sorted(QUANTS.items()):
        print(f"  {name:8s} {size}")
    print()
    got = ensure_gguf_model(settings, quant=args.quant, progress=_cli_progress,
                            verify=not args.no_verify, force=True)
    print()
    if got is None:
        print("could not fetch the weights — see the log above for why")
        return 1
    print(f"ready: {got}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
