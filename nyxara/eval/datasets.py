"""NYXARA · eval/datasets.py — load a benchmark from a JSONL dataset file (📏 → 🌍).

The built-in batteries in :mod:`nyxara.eval.benchmark` and :mod:`nyxara.eval.hard_benchmark` are
*self-generated*: template word problems and a hand-coded ruler. That is fine for a quick offline
signal, but a self-improvement loop that only ever validates against problems it itself produced is
grading its own homework — the "held-out" fold of a toy generator is not real external evidence.

This module is the bridge to **genuinely external** data. A dataset is just a JSONL file, one task
per line, mapping 1:1 onto :class:`~nyxara.eval.benchmark.BenchmarkTask`::

    {"id": "rw-math-00", "category": "math", "grader": "numeric_final",
     "answer": "27", "prompt": "...", "choices": [...], "aliases": [...], "tolerance": 0.01}

NYXARA **ships** one such file — ``data/holdout_realworld.jsonl``, a curated, externally-true
held-out validation set (real facts and genuine multi-step reasoning, never produced by the training
generator) — so :func:`build_realworld_benchmark` works fully offline, anywhere, deterministically.
A caller may instead point at any JSONL on disk (an absolute path, or via
``settings.self_improvement.validation_realworld_path`` / ``$NYXARA_EVAL_HOLDOUT_PATH``) to drop in a
real standard benchmark — GSM8K, MMLU, a private eval set — without touching code. The loader is
strict about shape and honest about failure: a missing or malformed file raises a clear
:class:`DatasetError`, never a silent empty battery.
"""

from __future__ import annotations

import json
import os
from importlib import resources
from typing import Any, Dict, List, Optional

from nyxara.eval.benchmark import Benchmark, BenchmarkTask, Grader

__all__ = [
    "DatasetError",
    "REALWORLD_HOLDOUT",
    "load_jsonl_benchmark",
    "build_realworld_benchmark",
    "realworld_holdout_path",
]

# The bundled, externally-true held-out validation set. Shipped as package data (see pyproject.toml
# ``[tool.setuptools.package-data]``) so it is importable from an installed wheel, not just the repo.
REALWORLD_HOLDOUT = "holdout_realworld.jsonl"

# Override the bundled set with a real external dataset (e.g. GSM8K/MMLU JSONL) via the environment.
_PATH_ENV = "NYXARA_EVAL_HOLDOUT_PATH"

_ALLOWED_KEYS = {"id", "category", "prompt", "answer", "grader",
                 "choices", "aliases", "tolerance", "metadata"}


class DatasetError(ValueError):
    """A dataset file is missing, unreadable, or a record is malformed."""


def _task_from_record(rec: Dict[str, Any], *, where: str) -> BenchmarkTask:
    """Build one :class:`BenchmarkTask` from a JSONL record, validating its shape honestly."""
    if not isinstance(rec, dict):
        raise DatasetError(f"{where}: each line must be a JSON object, got {type(rec).__name__}")
    unknown = set(rec) - _ALLOWED_KEYS
    if unknown:
        raise DatasetError(f"{where}: unknown field(s) {sorted(unknown)}")
    for required in ("id", "prompt", "answer"):
        if required not in rec or rec[required] in (None, ""):
            raise DatasetError(f"{where}: missing required field {required!r}")
    grader = str(rec.get("grader", Grader.EXACT))
    # Validate the grader name up-front so a typo fails at load, not mid-run.
    Grader.get(grader)
    tol = rec.get("tolerance")
    return BenchmarkTask(
        id=str(rec["id"]),
        prompt=str(rec["prompt"]),
        answer=str(rec["answer"]),
        grader=grader,
        category=str(rec.get("category", "general")),
        choices=list(rec.get("choices", []) or []),
        aliases=list(rec.get("aliases", []) or []),
        tolerance=float(tol) if tol is not None else None,
        metadata=dict(rec.get("metadata", {}) or {}),
    )


def _parse_jsonl(text: str, *, source: str) -> List[BenchmarkTask]:
    tasks: List[BenchmarkTask] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):       # allow blank lines and # comments
            continue
        where = f"{source}:{lineno}"
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{where}: invalid JSON ({exc})") from None
        task = _task_from_record(rec, where=where)
        if task.id in seen:
            raise DatasetError(f"{where}: duplicate task id {task.id!r}")
        seen.add(task.id)
        tasks.append(task)
    if not tasks:
        raise DatasetError(f"{source}: no tasks found (empty dataset)")
    return tasks


def realworld_holdout_path(settings: Any = None) -> Optional[str]:
    """The external override path for the held-out set, if one is configured.

    Resolution order (first hit wins): ``settings.self_improvement.validation_realworld_path`` →
    ``$NYXARA_EVAL_HOLDOUT_PATH`` → ``None`` (use the bundled set). Returns a path string only when
    one is set; the caller decides whether it must exist.
    """
    if settings is not None:
        cfg = getattr(settings, "self_improvement", None)
        path = getattr(cfg, "validation_realworld_path", None) if cfg is not None else None
        if path:
            return str(path)
    env = os.environ.get(_PATH_ENV)
    return env or None


def load_jsonl_benchmark(path_or_name: str, *, name: Optional[str] = None) -> Benchmark:
    """Load a :class:`Benchmark` from a JSONL dataset.

    ``path_or_name`` is either a filesystem path to a ``.jsonl`` file, or the bare name of a file
    bundled under ``nyxara/eval/data/`` (e.g. ``"holdout_realworld.jsonl"``). A filesystem path is
    tried first; if it does not exist, the name is resolved against the packaged data directory so
    the call works identically from the repo and from an installed wheel.
    """
    text: Optional[str] = None
    source = path_or_name
    # 1) An explicit file on disk.
    if os.path.sep in path_or_name or os.path.isfile(path_or_name):
        try:
            with open(path_or_name, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as exc:
            raise DatasetError(f"cannot read dataset {path_or_name!r}: {exc}") from None
    else:
        # 2) A bundled data file, read via importlib.resources (works inside a zip/wheel too).
        try:
            data_file = resources.files("nyxara.eval").joinpath("data", path_or_name)
            text = data_file.read_text(encoding="utf-8")
            source = f"nyxara.eval/data/{path_or_name}"
        except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
            raise DatasetError(
                f"dataset {path_or_name!r} not found on disk or in nyxara/eval/data/ ({exc})"
            ) from None
    tasks = _parse_jsonl(text, source=source)
    return Benchmark(name or os.path.splitext(os.path.basename(source))[0], tasks)


def build_realworld_benchmark(settings: Any = None) -> Benchmark:
    """The real-world held-out validation battery — bundled by default, externally overridable.

    With no override this loads NYXARA's shipped ``holdout_realworld.jsonl`` (offline, deterministic).
    If ``settings.self_improvement.validation_realworld_path`` or ``$NYXARA_EVAL_HOLDOUT_PATH`` points
    at a JSONL file, that real external dataset is loaded instead — the same harness, real data.
    """
    override = realworld_holdout_path(settings)
    if override:
        return load_jsonl_benchmark(override, name="realworld")
    return load_jsonl_benchmark(REALWORLD_HOLDOUT, name="realworld")
