"""NYXARA · growth/hf_source.py — reading public corpora at disk speed, resumably (📦→🚿).

``datasets``'s streaming iterator was the bottleneck in the whole corpus build, and not by a
little. Measured on this box against `HuggingFaceFW/fineweb-edu`:

    datasets streaming                817 KB/s
    parquet row-group (cold)        5,998 KB/s
    parquet row-group (warm)       33,252 KB/s
    the full CPU screen pipeline    6,030 KB/s

So the stream was 7x slower than the screens it fed, and the obvious next optimisation —
multiprocessing the screens — would have parallelised the half that was already winning. Reading
parquet row groups directly moves the constraint onto the CPU pipeline, which is where it should
be and where multiprocessing will actually pay.

**Row groups, not whole files.** `sample/10BT` is 14 files of ~2 GB. Downloading one before
processing a single row means a 2 GB wait and 2 GB of disk. Parquet is seekable, so
:class:`huggingface_hub.HfFileSystem` + ``pyarrow`` reads the footer and then pulls individual
row groups over HTTP range requests: metadata for a 2 GB file costs 1.3s, and each of its 726
row groups is ~4.7 MB of text.

That granularity is also what makes resume cheap. A build records the ``(file, row_group)``
pairs it has finished; restarting skips them outright. ``IterableDataset.skip(n)`` would have to
replay n rows to get back to the same place.

Falls back to ``datasets`` streaming for any source whose parquet layout cannot be resolved, so
an unusual repository degrades to the old behaviour instead of failing.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

__all__ = [
    "ParquetSource",
    "resolve_parquet_files",
    "open_source",
]

_TOKEN_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")


def _token() -> Optional[str]:
    for name in _TOKEN_ENV:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _norm(name: str) -> str:
    """`sample-10BT` and `sample/10BT` name the same thing; compare on letters and digits."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def resolve_parquet_files(repo: str, *, config: Optional[str] = None,
                          split: str = "train") -> List[str]:
    """The parquet files backing one dataset config/split, in a stable order.

    Layouts differ across the Hub — ``sample/10BT/000_00000.parquet``,
    ``data/train-00000-of-00114.parquet``, ``<config>/train/0000.parquet`` — so this filters by
    the *normalised* config and split appearing anywhere in the path, then falls back to every
    parquet in the repo when a filter would leave nothing. Returning too many files is
    recoverable (the budget stops the build); returning none silently is not.
    """
    from huggingface_hub import HfApi

    files = [f for f in HfApi(token=_token()).list_repo_files(repo, repo_type="dataset")
             if f.endswith(".parquet")]
    if not files:
        return []

    if config:
        wanted = _norm(config)
        scoped = [f for f in files if wanted in _norm(f)]
        if scoped:
            files = scoped

    if split:
        files = _filter_split(files, split)

    return sorted(files)


#: Markers that identify a file as evaluation data rather than training data.
_EVAL_MARKERS = ("test", "valid", "validation", "dev", "eval", "holdout")


def _filter_split(files: List[str], split: str) -> List[str]:
    """Keep only the files belonging to ``split`` — and never silently keep eval data.

    The obvious implementation, ``re.search(r"\\b(train|test|...)\\b", path)``, is wrong in a way
    that is invisible until you look at what it selected. ``_`` is a word character, so a
    ``\\btrain\\b`` pattern does **not** match ``train_sft``, nor ``\\btest\\b`` ``test_gen``. For
    `HuggingFaceH4/ultrachat_200k` — whose splits are exactly ``train_sft`` and ``test_gen`` —
    no file matched, the filter concluded "this repo has no splits", and returned everything.
    Asking for ``train_sft`` therefore yielded the TEST split, which would have put evaluation
    data into the training corpus: contamination introduced by the very code meant to prevent it.

    So: match on the normalised path, and when the caller wants a training split, drop anything
    carrying an eval marker even if nothing positively matched.
    """
    wanted = _norm(split)
    scoped = [f for f in files if wanted and wanted in _norm(f)]
    if scoped:
        return scoped

    if any(marker in wanted for marker in _EVAL_MARKERS):
        return files          # an eval split was explicitly requested; give it back unfiltered

    # Wanted a training split and nothing matched by name. Rather than fall back to "everything"
    # — which is how eval data gets in — fall back to "everything that is not obviously eval".
    trainish = [f for f in files
                if not any(marker in _norm(f) for marker in _EVAL_MARKERS)]
    return trainish or files


@dataclass
class ParquetSource:
    """One dataset, read row group by row group, with a record of what is already done."""

    repo: str
    files: List[str]
    text_key: str
    source_id: str
    #: ``"{file}#{row_group}"`` for every row group already consumed by an earlier run.
    done: Set[str] = field(default_factory=set)
    columns: Optional[Sequence[str]] = None

    def _open(self, path: str) -> Any:
        from huggingface_hub import HfFileSystem
        import pyarrow.parquet as pq

        fs = HfFileSystem(token=_token())
        return pq.ParquetFile(fs.open(f"datasets/{self.repo}/{path}", "rb"))

    @staticmethod
    def key(path: str, group: int) -> str:
        return f"{path}#{group}"

    def rows(self) -> Iterator[Tuple[str, str]]:
        """Yield ``(text, row_group_key)`` for every unread row group, in order.

        The key travels with the text so the caller can mark a row group finished only after its
        documents have actually been written — recording progress before the write would lose
        data on a crash and call it done.
        """
        for path in self.files:
            # A whole file already finished is skipped without opening it: no request at all,
            # which is what makes resume O(1) rather than a replay.
            try:
                handle = self._open(path)
            except Exception:  # noqa: BLE001 — one unreadable file must not end the source
                continue
            try:
                total = handle.metadata.num_row_groups
            except Exception:  # noqa: BLE001
                continue
            if all(self.key(path, g) in self.done for g in range(total)):
                continue

            for group in range(total):
                key = self.key(path, group)
                if key in self.done:
                    continue
                try:
                    table = handle.read_row_group(group, columns=list(self._columns(handle)))
                except Exception:  # noqa: BLE001 — a bad row group is skipped, not fatal
                    continue
                for value in self._texts(table):
                    if value:
                        yield value, key

    def _columns(self, handle: Any) -> Sequence[str]:
        if self.columns:
            return self.columns
        names = set(handle.schema_arrow.names)
        if self.text_key in names:
            return [self.text_key]
        for fallback in ("text", "content", "response", "messages", "conversations", "body"):
            if fallback in names:
                return [fallback]
        return list(names)[:1]

    def _texts(self, table: Any) -> Iterator[str]:
        from nyxara.growth.corpus import _row_text

        names = table.column_names
        if len(names) == 1 and names[0] not in ("messages", "conversations"):
            for value in table.column(names[0]).to_pylist():
                yield value if isinstance(value, str) else ""
            return
        # Chat-shaped or multi-column: hand each row to the corpus's own renderer so multi-turn
        # transcripts keep every turn rather than being flattened here.
        for row in table.to_pylist():
            key = self.text_key if self.text_key in row else names[0]
            yield _row_text(row, key)


def open_source(spec: Dict[str, Any], *, source_id: str,
                done: Optional[Set[str]] = None) -> Optional[ParquetSource]:
    """Build a :class:`ParquetSource` for one configured source, or ``None`` to fall back.

    ``None`` means "this repository's parquet layout could not be resolved" — the caller should
    use ``datasets`` streaming for it rather than dropping the source.
    """
    repo = spec.get("path", "")
    if not repo:
        return None
    try:
        files = resolve_parquet_files(repo, config=spec.get("name"),
                                      split=spec.get("split", "train"))
    except Exception:  # noqa: BLE001 — gated, renamed, offline: the caller degrades
        return None
    if not files:
        return None

    source = ParquetSource(repo=repo, files=files, text_key=spec.get("text_key", "text"),
                           source_id=source_id, done=set(done or ()))

    # Probe the first file's footer before claiming this source works. `list_repo_files`
    # succeeds on a gated repo while `resolve` returns 403 — `nampdn-ai/tiny-codes` does exactly
    # that. Without the probe the source reported "9 parquet files", then yielded ZERO rows and
    # the build recorded "exhausted after 0 rows" as though the dataset were simply empty. A
    # gated source has to be visibly gated, not silently absent.
    try:
        source._open(files[0]).metadata
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"parquet unreadable ({type(exc).__name__}: {str(exc)[:90]})") from exc
    return source
