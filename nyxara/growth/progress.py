"""NYXARA · growth/progress.py — a build that survives the machine going away (💾→🔁).

``ShardWriter`` already resumes: a finished shard is a finished file listed in ``index.json``,
written atomically per flush. That covers a crash. It does **not** cover what actually happened
here — the container reset, the checkout reverted, and every installed package vanished.

Three things were unpreserved, and each one silently corrupts a resumed build rather than
failing it:

* **Source position.** Nothing recorded how far into each source the build had read, so a resume
  started from row zero and re-screened everything it already had.
* **Dedup tables.** ``ExactDedup`` could save; ``MinHashLSH`` could not. A fresh table makes
  every previously-seen document look new, so the corpus quietly fills with duplicates of what
  it already contains.
* **The report.** Screen counts restarted at zero, so the final numbers described the last
  segment rather than the corpus — and the dataset card is generated from them.

:class:`BuildProgress` holds all three and is **mirrored to the Hub**, which is the part that
matters. Persisting to local disk survives a crash; persisting to the remote survives the
machine. Only the second is a real answer to an ephemeral container.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

__all__ = ["BuildProgress", "PROGRESS_NAME", "DEDUP_NAMES"]

PROGRESS_NAME = "progress.json"
DEDUP_NAMES = ("dedup-exact.npy", "dedup-lsh.npz")


@dataclass
class BuildProgress:
    """Everything a resumed build needs that ``index.json`` does not already carry."""

    #: ``source_id -> {"file#row_group", ...}`` — row groups fully consumed AND written.
    consumed: Dict[str, Set[str]] = field(default_factory=dict)
    #: ``source_id -> tokens accepted``, for the per-source budget.
    spent: Dict[str, int] = field(default_factory=dict)
    #: The accumulated screen counts, so the report describes the corpus and not the last run.
    report: Dict[str, Any] = field(default_factory=dict)
    #: Shards already uploaded, so `--push-every` does not re-upload after a resume.
    pushed: Set[str] = field(default_factory=set)
    seed: int = 0
    vocab_sig: str = ""
    updated_at: float = field(default_factory=time.time)

    # ---- source bookkeeping ---- #
    def done_for(self, source_id: str) -> Set[str]:
        return self.consumed.setdefault(source_id, set())

    def mark(self, source_id: str, key: str, tokens: int = 0) -> None:
        """Record a row group as finished. Called only AFTER its documents are written.

        Marking before the write would let a crash lose data and still call it done — the one
        ordering that turns a resumable build into a quietly incomplete one.
        """
        self.done_for(source_id).add(key)
        if tokens:
            self.spent[source_id] = self.spent.get(source_id, 0) + int(tokens)

    def tokens_from(self, source_id: str) -> int:
        return int(self.spent.get(source_id, 0))

    # ---- persistence ---- #
    def to_dict(self) -> Dict[str, Any]:
        return {"consumed": {k: sorted(v) for k, v in self.consumed.items()},
                "spent": dict(self.spent), "report": dict(self.report),
                "pushed": sorted(self.pushed), "seed": self.seed,
                "vocab_sig": self.vocab_sig, "updated_at": self.updated_at}

    @classmethod
    def from_dict(cls, blob: Dict[str, Any]) -> "BuildProgress":
        return cls(
            consumed={k: set(v) for k, v in (blob.get("consumed") or {}).items()},
            spent=dict(blob.get("spent") or {}),
            report=dict(blob.get("report") or {}),
            pushed=set(blob.get("pushed") or ()),
            seed=int(blob.get("seed", 0)),
            vocab_sig=str(blob.get("vocab_sig", "")),
            updated_at=float(blob.get("updated_at", 0.0)))

    def save(self, directory: Any) -> Path:
        """Atomic write — tmp then ``os.replace``.

        A truncated progress file is worse than none: it makes a resume look possible while
        pointing at a position the build never reached.
        """
        path = Path(directory) / PROGRESS_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = time.time()
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, directory: Any) -> "BuildProgress":
        path = Path(directory) / PROGRESS_NAME
        if not path.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 — an unreadable cursor restarts, never crashes
            return cls()

    @property
    def row_groups_done(self) -> int:
        return sum(len(v) for v in self.consumed.values())

    def summary(self) -> str:
        per = ", ".join(f"{k.split(':')[-1]} {len(v)}" for k, v in sorted(self.consumed.items())
                        if v)
        return (f"· resuming: {self.row_groups_done:,} row groups already consumed"
                + (f" [{per}]" if per else ""))


# --------------------------------------------------------------------------- #
# The Hub side — the part that survives the container, not merely the process
# --------------------------------------------------------------------------- #
def restore_from_hub(directory: Any, repo_id: str, *, token: Optional[str] = None) -> bool:
    """Pull ``index.json``, ``progress.json`` and the dedup tables back down.

    Returns ``True`` when a resumable state was recovered. This is what makes a *fresh container*
    continue a build rather than start one.
    """
    try:
        from huggingface_hub import hf_hub_download
    except Exception:  # noqa: BLE001
        return False

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    recovered = False
    for name in ("index.json", PROGRESS_NAME, *DEDUP_NAMES):
        try:
            path = hf_hub_download(repo_id, name, repo_type="dataset", token=token,
                                   local_dir=str(out))
            recovered = recovered or name in ("index.json", PROGRESS_NAME)
            _ = path
        except Exception:  # noqa: BLE001 — absent is normal on a first run
            continue
    return recovered


def files_to_mirror(directory: Any) -> List[str]:
    """The state files worth uploading beside the shards."""
    out = Path(directory)
    return [n for n in ("index.json", PROGRESS_NAME, *DEDUP_NAMES) if (out / n).exists()]
