"""NYXARA · mind/gguf_assets.py — the weights behind her CORTEX (🧠⚡).

``mind/llm.py::GGUFProvider`` serves Qwythos-9B from GGUF files on her own disk. This module is the
only thing that knows how those files *get there*, and it is deliberately the whole story:

    where the weights live  →  :func:`model_path` / :func:`mmproj_path`
    are they here yet?      →  :func:`model_present`
    put them here           →  :func:`ensure_gguf_model`

It is a deliberate near-copy of :mod:`nyxara.mind.litertlm_assets`, because the three properties
that module is built around are not specific to LiteRT-LM:

* **It never lies about readiness.** Each download lands in ``<name>.part`` and is moved into place
  with a single atomic ``os.replace`` only after the last byte arrives. A killed process therefore
  leaves either no file or a complete one — never a truncated 5.6 GB file that ``available()`` would
  cheerfully report as a working model.
* **It never surprises the host.** A 5.6 GB fetch is a real event, so it happens only when
  ``llm.qwythos_auto_download`` is on, and *never* under the TEST profile.
* **It never breaks the boot.** Every failure path returns ``None`` and logs; the ladder in
  ``mind/llm.py`` then simply starts one rung lower.

**What is new here, and why it had to be.** This repo publishes a ``SHA256SUMS`` file beside the
weights, and a 5.6 GB file that arrived over a proxy is exactly the kind of thing whose integrity
nobody checks until it matters. So:

* a checksum that is **obtained and mismatches** is fatal — the ``.part`` is deleted and nothing is
  moved into place, because a corrupt model that loads is worse than one that does not;
* a checksum that **cannot be obtained** (offline mirror, blocked proxy, file removed upstream) does
  not block the install, but it is logged as a warning and reported by :func:`verification_state`
  as ``unverified``. A verification that quietly did not happen is worse than one that never
  claimed to.

**Two files, not one.** Qwythos is multimodal: the language weights and the vision projector
(``mmproj-…``) are separate GGUFs. The projector is fetched only when ``llm.qwythos_vision_enabled``
is on, and its absence never blocks text — ``GGUFProvider.vision_available()`` reports *why* she
cannot see rather than silently answering as though she had looked.

Run it directly to fetch the weights with progress::

    python -m nyxara.mind.gguf_assets
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from nyxara.kernel.config import NyxaraSettings, Profile, get_settings

log = logging.getLogger("nyxara.mind.gguf_assets")

__all__ = [
    "model_path",
    "mmproj_path",
    "model_present",
    "mmproj_present",
    "expected_url",
    "resolve_url",
    "sha256_of",
    "published_checksums",
    "parse_sums",
    "verification_state",
    "ensure_gguf_model",
]

# Progress reporter: (bytes_so_far, total_bytes_or_None) -> None
ProgressFn = Callable[[int, Optional[int]], None]

_CHUNK = 1024 * 1024        # 1 MiB — big enough that the syscalls disappear next to the network

#: The checksum manifest the model repo publishes beside the weights.
_SUMS_FILENAME = "SHA256SUMS"

#: Recorded per absolute path once a fetch completes, so :func:`verification_state` can tell
#: "checked and matched" apart from "never checked". Absent = we did not put this file here.
_VERIFIED: Dict[str, str] = {}


def _models_dir(settings: NyxaraSettings) -> Path:
    return Path(settings.paths.data_dir) / "models"


def model_path(settings: Optional[NyxaraSettings] = None) -> Path:
    """Where her cortex weights live: the configured path, else ``data_dir/models/<filename>``."""
    settings = settings or get_settings()
    configured = getattr(settings.llm, "qwythos_model_path", None)
    if configured:
        return Path(configured)
    return _models_dir(settings) / settings.llm.qwythos_filename


def mmproj_path(settings: Optional[NyxaraSettings] = None) -> Path:
    """Where the vision projector lives. Separate file, separate question from the weights."""
    settings = settings or get_settings()
    configured = getattr(settings.llm, "qwythos_mmproj_path", None)
    if configured:
        return Path(configured)
    return _models_dir(settings) / settings.llm.qwythos_mmproj_filename


def model_present(settings: Optional[NyxaraSettings] = None) -> bool:
    """True iff the cortex weights are on disk."""
    try:
        return model_path(settings).is_file()
    except Exception:  # noqa: BLE001 — an unreadable path is simply "not present"
        return False


def mmproj_present(settings: Optional[NyxaraSettings] = None) -> bool:
    """True iff the vision projector is on disk. Never required for text."""
    try:
        return mmproj_path(settings).is_file()
    except Exception:  # noqa: BLE001
        return False


def resolve_url(settings: Optional[NyxaraSettings], filename: str) -> str:
    """The HuggingFace resolve URL for one file in the configured repo."""
    settings = settings or get_settings()
    return f"https://huggingface.co/{settings.llm.qwythos_repo_id}/resolve/main/{filename}"


def expected_url(settings: Optional[NyxaraSettings] = None) -> str:
    """The resolve URL for the language weights — the file that defines the rung."""
    settings = settings or get_settings()
    return resolve_url(settings, settings.llm.qwythos_filename)


# --------------------------------------------------------------------------- #
# Integrity
# --------------------------------------------------------------------------- #
def sha256_of(path: Path, progress: Optional[ProgressFn] = None) -> str:
    """Streamed SHA-256 of a file. Streamed because these files do not fit in memory."""
    digest = hashlib.sha256()
    size = path.stat().st_size
    done = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            done += len(chunk)
            if progress is not None:
                progress(done, size)
    return digest.hexdigest()


def published_checksums(settings: Optional[NyxaraSettings] = None) -> Dict[str, str]:
    """``{filename: sha256}`` from the repo's ``SHA256SUMS``, or ``{}`` if it cannot be read.

    Empty is a real answer, not an error: a mirror, a blocked proxy or an upstream that removed the
    manifest all land here, and none of them is a reason to refuse an install. The caller reports
    the file as *unverified* instead of pretending it checked.
    """
    settings = settings or get_settings()
    text = ""
    try:
        import httpx
        url = resolve_url(settings, _SUMS_FILENAME)
        with httpx.Client(follow_redirects=True, timeout=30.0) as client:
            resp = client.get(url)
            if resp.status_code == 200:
                text = resp.text
    except Exception as exc:  # noqa: BLE001 — no manifest is "unverified", never a failure
        log.debug("could not read %s (%s); the fetch will be unverified", _SUMS_FILENAME, exc)
        return {}
    return parse_sums(text)


def parse_sums(text: str) -> Dict[str, str]:
    """``{filename: sha256}`` out of a ``sha256sum`` manifest. Pure, so it is directly testable.

    Tolerates both formats the tool writes — ``<hash>  <name>`` and ``<hash> *<name>`` for binary
    mode — and ignores anything that is not a 64-hex-digit line, because a manifest with a header
    or a stray blank line is still a usable manifest.
    """
    out: Dict[str, str] = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 64:
            try:
                int(parts[0], 16)
            except ValueError:
                continue
            out[parts[-1].lstrip("*")] = parts[0].lower()
    return out


def verification_state(path: Path) -> str:
    """``verified`` | ``unverified`` | ``absent`` — what is actually known about this file.

    ``unverified`` covers both "we fetched it and no manifest was available" and "it was already on
    disk when we got here", because those are the same claim: the bytes are present and nothing in
    this process checked them.
    """
    try:
        if not path.is_file():
            return "absent"
    except Exception:  # noqa: BLE001
        return "absent"
    return _VERIFIED.get(str(path), "unverified")


# --------------------------------------------------------------------------- #
# Fetch backends (both land on a .part file; the caller does the atomic move)
# --------------------------------------------------------------------------- #
def _fetch_via_hub(settings: NyxaraSettings, filename: str, dest: Path,
                   progress: Optional[ProgressFn]) -> bool:
    """Download through ``huggingface_hub`` (cache + resume for free). False if unusable."""
    try:
        from huggingface_hub import hf_hub_download
    except Exception:  # noqa: BLE001 — optional; the httpx path covers it
        return False
    try:
        cached = hf_hub_download(repo_id=settings.llm.qwythos_repo_id, filename=filename)
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


def _fetch_via_http(settings: NyxaraSettings, filename: str, dest: Path,
                    progress: Optional[ProgressFn]) -> bool:
    """Stream the file over HTTPS, resuming an interrupted ``.part`` with a Range request."""
    try:
        import httpx
    except Exception:  # noqa: BLE001
        return False
    url = resolve_url(settings, filename)
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


def _fetch_one(settings: NyxaraSettings, filename: str, dest: Path, *,
               checksums: Optional[Dict[str, str]], progress: Optional[ProgressFn],
               force: bool) -> Optional[Path]:
    """Fetch one file into place, atomically and (where possible) verified. ``None`` on failure."""
    part = dest.with_name(dest.name + ".part")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if force and part.exists():
            part.unlink()
        log.info("fetching %s from %s -> %s", filename, settings.llm.qwythos_repo_id, dest)
        ok = (_fetch_via_hub(settings, filename, part, progress)
              or _fetch_via_http(settings, filename, part, progress))
        if not ok:
            log.warning("no way to fetch %s (need huggingface_hub or httpx)", filename)
            return None

        want = (checksums or {}).get(filename)
        if want:
            got = sha256_of(part)
            if got != want:
                # A corrupt model that loads is worse than one that does not: it would answer, and
                # every answer would be garbage attributed to her cortex.
                log.error("%s failed its published SHA-256 (expected %s, got %s) — refusing to "
                          "install it", filename, want, got)
                part.unlink(missing_ok=True)
                return None
            _VERIFIED[str(dest)] = "verified"
        else:
            _VERIFIED[str(dest)] = "unverified"
            log.warning("%s was installed WITHOUT checksum verification — no %s could be read "
                        "from the repo", filename, _SUMS_FILENAME)

        os.replace(part, dest)            # atomic: the file appears complete or not at all
        log.info("%s ready: %s (%d bytes, %s)", filename, dest, dest.stat().st_size,
                 verification_state(dest))
        return dest
    except Exception as exc:  # noqa: BLE001 — boot integrity comes first
        log.warning("could not fetch %s: %s", filename, exc)
        try:
            if part.exists():
                part.unlink()             # never leave a truncated file behind
        except Exception:  # noqa: BLE001
            pass
        return None


# --------------------------------------------------------------------------- #
# The one entry point
# --------------------------------------------------------------------------- #
def ensure_gguf_model(
    settings: Optional[NyxaraSettings] = None,
    *,
    progress: Optional[ProgressFn] = None,
    force: bool = False,
) -> Optional[Path]:
    """Make sure her cortex weights are on disk; return the path to them, or ``None``.

    Returns immediately when the language weights are already there (the common case — this is
    called on every boot). Declines silently, returning ``None``, when auto-download is off or the
    profile is TEST. **Never raises**: a failed fetch is logged, partial files are cleaned up, and
    the caller's boot carries on one rung lower down the ladder.

    The vision projector is a *secondary* fetch: a failure to get it is logged and the language
    weights are still returned, because a cortex that can read but not see is a working cortex.
    """
    settings = settings or get_settings()
    dest = model_path(settings)
    mm_dest = mmproj_path(settings)
    want_vision = bool(getattr(settings.llm, "qwythos_vision_enabled", False))

    have_text = dest.is_file() and not force
    have_vision = (not want_vision) or (mm_dest.is_file() and not force)
    if have_text and have_vision:
        return dest
    if settings.profile is Profile.TEST:
        return dest if dest.is_file() else None   # a hermetic suite never fetches 5.6 GB
    if not bool(getattr(settings.llm, "qwythos_enabled", True)):
        return dest if dest.is_file() else None
    if not bool(getattr(settings.llm, "qwythos_auto_download", False)):
        if not have_text:
            log.info("qwythos weights missing at %s and auto-download is off — "
                     "run scripts/fetch_qwythos_model.py to install them", dest)
        return dest if dest.is_file() else None

    checksums: Optional[Dict[str, str]] = None
    if bool(getattr(settings.llm, "qwythos_verify_sha256", True)):
        checksums = published_checksums(settings)

    out: Optional[Path] = dest if have_text else None
    if not have_text:
        out = _fetch_one(settings, settings.llm.qwythos_filename, dest,
                         checksums=checksums, progress=progress, force=force)
    if want_vision and not have_vision:
        got = _fetch_one(settings, settings.llm.qwythos_mmproj_filename, mm_dest,
                         checksums=checksums, progress=progress, force=force)
        if got is None:
            log.warning("the vision projector could not be fetched — she will answer on text and "
                        "report that she cannot see, rather than pretending to have looked")
    return out


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


def _report(settings: NyxaraSettings) -> None:  # pragma: no cover
    path, mm = model_path(settings), mmproj_path(settings)
    print(f"repo         : {settings.llm.qwythos_repo_id}")
    print(f"weights      : {path}  [{verification_state(path)}]")
    if getattr(settings.llm, "qwythos_vision_enabled", False):
        print(f"vision proj. : {mm}  [{verification_state(mm)}]")
    else:
        print("vision proj. : disabled by config")


def main(argv: Optional[Tuple[str, ...]] = None) -> int:  # pragma: no cover
    """Fetch her cortex weights with progress. Returns a process exit code."""
    import argparse
    import logging as _logging

    parser = argparse.ArgumentParser(description="Fetch NYXARA's cortex weights (Qwythos GGUF).")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--no-verify", action="store_true", help="skip SHA-256 verification")
    args = parser.parse_args(list(argv) if argv is not None else None)

    _logging.basicConfig(level=_logging.INFO, format="%(message)s")
    settings = get_settings()
    settings.llm.qwythos_enabled = True
    settings.llm.qwythos_auto_download = True
    if args.no_verify:
        settings.llm.qwythos_verify_sha256 = False

    print("NYXARA · fetching her cortex weights")
    print("=" * 62)
    _report(settings)
    print()

    got = ensure_gguf_model(settings, progress=_cli_progress, force=args.force)
    print()
    if got is None:
        print("FAILED — see the log above.")
        return 1
    _report(settings)
    print("\nAlso needed once, for the runtime itself:\n    pip install llama-cpp-python")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
