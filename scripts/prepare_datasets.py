#!/usr/bin/env python3
"""Fetch the Master's ten datasets and convert them into the three shapes NJP reads.

``scripts/prepare_conceptnet.py`` does this for source #1, whose dump is a 475 MB TSV. The other
nine arrive as row streams from the HuggingFace datasets server, so they share one fetcher, and
what each row *becomes* is decided in :mod:`nyxara.njp.corpora` rather than here. That split is
the point of the file: fetching is the part that breaks when a mirror moves, converting is the
part that decides what she knows, and they should not be in the same function.

Standard library only, like every other script in here — ``pyarrow`` is *used* when it is already
installed because it makes a large pull an order of magnitude faster, and its absence changes the
speed rather than the outcome.

Examples
--------
    # everything, at the default sizes (a few minutes, ~200 MB on disk)
    python scripts/prepare_datasets.py --out data/njp

    # one source, larger
    python scripts/prepare_datasets.py --out data/njp --only wikipedia --rows 200000

    # see what would be fetched, and where from
    python scripts/prepare_datasets.py --list

Then train her on the lot, in the Master's order, with an exam between every stage:

    python -m nyxara.njp.train --data data/njp --save ~/.nyxara/njp_state.json

ConceptNet is priority #1 and is **not** fetched here. Run its own script first::

    python scripts/prepare_conceptnet.py --download --out data/njp/conceptnet.triples.jsonl.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nyxara.njp import corpora  # noqa: E402

#: The rows endpoint. Paged, capped at 100 rows a call by the server, and the only interface here
#: that needs no third-party package. It is the fallback rather than the default because a
#: hundred-thousand-row pull through it is a thousand round trips.
_ROWS = "https://datasets-server.huggingface.co/rows"

#: Where the parquet shards for a (dataset, config, split) are listed. One call, then the shards
#: are plain HTTPS GETs — which is why this path is worth having when ``pyarrow`` is present.
_PARQUET = "https://huggingface.co/api/datasets/{dataset}/parquet/{config}/{split}"

#: How many rows to pull per source unless the caller says otherwise. Chosen so a full run is
#: minutes rather than hours and the result is still a corpus rather than a demo. They differ by
#: source because the *yield* differs by an order of magnitude: a Wikipedia article gives up to
#: three statements, a GSM8K row gives exactly one pair, and a translation row gives a fact only
#: if it is short enough to be a term — which most are not.
_DEFAULT_ROWS: Dict[str, int] = {
    "wikipedia": 60_000,
    "openwebmath": 20_000,
    "proofnet": 400,
    "oasst1": 60_000,
    "code": 40_000,
    "arxiv": 60_000,
    "gsm8k": 8_000,
    "math": 8_000,
    "gutenberg": 2_000,
    "multilingual": 200_000,
}

_USER_AGENT = "nyxara-prepare-datasets/1.0 (+https://github.com/nyxarajp-del/nyxara)"


def _get(url: str, *, timeout: int = 120) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        # Only ever sent to huggingface.co, and only if the environment already holds one. Several
        # of these datasets are gated behind an accepted licence rather than behind a paywall, and
        # a run that has the token should not fail on them.
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _rows_via_api(dataset: str, config: str, split: str, want: int,
                  *, log: Any = None) -> Iterator[Dict[str, Any]]:
    """Page the rows endpoint. Slow, dependency-free, and correct on every dataset it serves."""
    offset, page = 0, 100
    while offset < want:
        length = min(page, want - offset)
        url = (f"{_ROWS}?dataset={urllib.parse.quote(dataset, safe='')}"
               f"&config={urllib.parse.quote(config, safe='')}"
               f"&split={urllib.parse.quote(split, safe='')}"
               f"&offset={offset}&length={length}")
        try:
            payload = json.loads(_get(url))
        except (urllib.error.URLError, ValueError, TimeoutError) as exc:
            if log:
                log(f"    rows API stopped at offset {offset}: {exc}")
            return
        batch = payload.get("rows") or []
        if not batch:
            return
        for item in batch:
            row = item.get("row")
            if isinstance(row, dict):
                yield row
        offset += len(batch)


def _rows_via_parquet(dataset: str, config: str, split: str, want: int,
                      *, log: Any = None) -> Optional[Iterator[Dict[str, Any]]]:
    """Stream the split's parquet shards, or ``None`` if that route is not available.

    Returns ``None`` rather than raising so the caller falls back to the rows API on a missing
    ``pyarrow``, a private dataset, or a config the parquet conversion has not run for. A
    preparation run over ten sources should degrade to slow, not to failed.
    """
    try:
        import pyarrow.parquet as pq         # noqa: PLC0415 — optional by design
    except ImportError:
        return None
    try:
        # `safe='/'` on the dataset, and it is not a detail: the id is a *path segment* here
        # ("wikimedia/wikipedia") where in the rows endpoint it is a query parameter. Percent-
        # encoding its slash gives a 404 that this function is written to swallow, so the whole
        # fast route disappeared silently and every source fell back to the paged API.
        listing = json.loads(_get(_PARQUET.format(
            dataset=urllib.parse.quote(dataset, safe='/'),
            config=urllib.parse.quote(config, safe=''),
            split=urllib.parse.quote(split, safe='')), timeout=60))
    except (urllib.error.URLError, ValueError, TimeoutError):
        return None
    urls = [u for u in listing if isinstance(u, str)] if isinstance(listing, list) else []
    if not urls:
        return None

    def _stream() -> Iterator[Dict[str, Any]]:
        import tempfile
        seen = 0
        for url in urls:
            if seen >= want:
                return
            handle, shard = tempfile.mkstemp(suffix=".parquet")
            os.close(handle)
            try:
                # Downloaded whole rather than range-read: parquet's footer is at the end of the
                # file, so a streaming reader would need range requests the mirror may not serve,
                # and a shard is a few hundred megabytes at worst.
                with urllib.request.urlopen(
                        urllib.request.Request(url, headers={"User-Agent": _USER_AGENT}),
                        timeout=600) as response, open(shard, "wb") as out:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                table = pq.ParquetFile(shard)
                for batch in table.iter_batches(batch_size=1000):
                    for row in batch.to_pylist():
                        yield row
                        seen += 1
                        if seen >= want:
                            return
            except Exception as exc:  # noqa: BLE001 — one bad shard is not a failed source
                if log:
                    log(f"    shard failed ({exc}); continuing")
            finally:
                try:
                    os.unlink(shard)
                except OSError:
                    pass

    return _stream()


def fetch(source: Any, want: int, *, log: Any = None) -> Iterator[Dict[str, Any]]:
    """Raw rows for one source, by whichever route is available."""
    stream = _rows_via_parquet(source.dataset, source.config, source.split, want, log=log)
    if stream is None:
        if log:
            log("    parquet unavailable — falling back to the paged rows API")
        stream = _rows_via_api(source.dataset, source.config, source.split, want, log=log)
    return stream


def prepare(source: Any, out_dir: Path, want: int, *, limit: int = 0,
            log: Any = None) -> Dict[str, Any]:
    """Fetch, convert and write one source. Returns what it managed.

    Written through a temporary file and renamed at the end, so an interrupted run leaves no
    half-written corpus for :func:`nyxara.njp.train.discover` to find and load as a whole one.
    """
    target = out_dir / source.filename
    scratch = target.with_suffix(target.suffix + ".part")
    started = time.perf_counter()
    written = 0
    try:
        with gzip.open(scratch, "wt", encoding="utf-8") as handle:
            for row in corpora.convert(source.key, fetch(source, want, log=log), limit=limit):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                if log and written % 20_000 == 0:
                    log(f"    {written} rows")
    except KeyboardInterrupt:
        scratch.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001 — one source failing must not end a ten-source run
        scratch.unlink(missing_ok=True)
        return {"key": source.key, "rows": 0, "error": str(exc)}
    if not written:
        scratch.unlink(missing_ok=True)
        return {"key": source.key, "rows": 0, "error": "converted to nothing"}
    scratch.replace(target)
    return {"key": source.key, "form": source.form, "rows": written,
            "file": str(target), "bytes": target.stat().st_size,
            "seconds": round(time.perf_counter() - started, 1)}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/njp", help="directory to write the corpora into")
    parser.add_argument("--only", default="", help="comma-separated source keys")
    parser.add_argument("--skip", default="", help="comma-separated source keys to leave out")
    parser.add_argument("--rows", type=int, default=0,
                        help="raw rows to fetch per source; 0 uses the per-source default")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap on converted rows written per source; 0 = no cap")
    parser.add_argument("--list", action="store_true",
                        help="print the table and fetch nothing")
    args = parser.parse_args(argv)

    only = [k for k in args.only.split(",") if k.strip()] or None
    denied = {k.strip() for k in args.skip.split(",") if k.strip()}
    chosen = [s for s in corpora.in_priority_order(only) if s.key not in denied]

    if args.list:
        for source in chosen:
            rows = args.rows or _DEFAULT_ROWS.get(source.key, 0)
            print(f"{source.priority:>2} {source.tier:<6} {source.key:<13} {source.form:<8} "
                  f"{rows or '—':>8}  {source.dataset} [{source.config}/{source.split}]")
            print(f"   {source.gets}\n   {source.note}")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    for source in chosen:
        if source.key == "conceptnet":
            print(f"{source.key}: prepared by scripts/prepare_conceptnet.py — skipping\n"
                  f"  python scripts/prepare_conceptnet.py --download "
                  f"--out {out_dir / source.filename}")
            continue
        want = args.rows or _DEFAULT_ROWS.get(source.key, 10_000)
        print(f"{source.priority:>2} {source.key} ← {source.dataset} "
              f"[{source.config}/{source.split}], {want} rows", flush=True)
        got = prepare(source, out_dir, want, limit=args.limit,
                      log=lambda m: print(m, flush=True))
        results.append(got)
        if got.get("error"):
            print(f"    FAILED: {got['error']}", flush=True)
        else:
            print(f"    {got['rows']} rows → {got['file']} "
                  f"({got['bytes'] / 1e6:.1f} MB, {got['seconds']}s)", flush=True)

    ok = [r for r in results if not r.get("error")]
    print(f"\n{len(ok)}/{len(results)} sources prepared into {out_dir}")
    if ok:
        print(f"\ntrain her on them:\n  python -m nyxara.njp.train --data {out_dir}")
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
