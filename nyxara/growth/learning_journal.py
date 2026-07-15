"""NYXARA · growth/learning_journal.py — the append-only ledger of lifelong learning (📓).

A checkpoint (``NyxaraCore.save_state``) is a *snapshot*: the complete learned state at one
instant. Between snapshots, NYXARA keeps learning — and a crash (or a kill -9) between the last
autosave and the next would silently drop those minutes of learning. That is forgetting by
another name.

This journal closes the gap. Every learning event is appended, immediately and durably, to a
JSONL ledger the moment it happens. Three things fall out of that:

* **Crash durability.** On restart, :meth:`replay_after` re-applies exactly the events that
  landed *after* the last checkpoint's watermark, so learning since the last snapshot is
  rebuilt rather than lost — never double-applied (the watermark guarantees it).
* **Auditable proof.** The ledger is the honest, append-only record that learning is real and
  *accreting* — you can read, count, and diff it. It is not a claim; it is evidence.
* **Bounded cost.** The file self-rotates once it exceeds ``max_events`` (keeping the recent
  tail), so a mind that runs for months never grows an unbounded log.

Pure standard library. Every write is best-effort and never fatal — the ledger must never break
a turn, and a corrupt or absent ledger must never block boot.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterator, List, Optional

__all__ = ["LearningEvent", "LearningJournal"]


def LearningEvent(action: str, features: Dict[str, float], reward: float, *,
                  task: str = "", err: Optional[float] = None,
                  kind: str = "reward") -> Dict[str, Any]:
    """Build one journal event (a plain JSON-able dict). Features are summarized to their
    active names to keep the ledger compact and free of unbounded numeric noise."""
    return {
        "at": time.time(),
        "kind": kind,
        "action": action,
        "task": task or action,
        "reward": round(float(reward), 6),
        "err": (round(float(err), 6) if err is not None else None),
        "features": {str(k): round(float(v), 6) for k, v in (features or {}).items()},
    }


class LearningJournal:
    """An append-only JSONL ledger of learning events with a checkpoint watermark.

    ``seq`` is the total number of events ever appended (monotonic, survives rotation), so the
    watermark written into a checkpoint stays meaningful even after the file is trimmed.
    """

    def __init__(self, path: str, *, max_events: int = 50_000,
                 rotate_keep: int = 10_000) -> None:
        self.path = path
        self.max_events = int(max_events)
        self.rotate_keep = min(int(rotate_keep), int(max_events))
        self._seq = 0                     # monotonic count of events ever appended
        self._lines_since_rotate = 0
        self._load_seq()

    # ---- state ---- #
    @property
    def seq(self) -> int:
        """The current watermark: how many events have ever been appended."""
        return self._seq

    def _load_seq(self) -> None:
        """Recover the monotonic seq (and on-disk line count) from an existing ledger."""
        base = 0
        lines = 0
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as fh:
                    first = fh.readline()
                    if first.startswith("#base "):
                        try:
                            base = int(first[len("#base "):].strip() or "0")
                        except ValueError:
                            base = 0
                    else:
                        # a headerless legacy line still counts
                        lines = 1 if first.strip() else 0
                    for line in fh:
                        if line.strip():
                            lines += 1
        except OSError:
            base, lines = 0, 0
        self._lines_since_rotate = lines
        self._seq = base + lines

    # ---- append ---- #
    def append(self, event: Dict[str, Any]) -> int:
        """Append one event durably. Returns its 1-based seq, or the current seq on failure."""
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            new_file = not os.path.exists(self.path)
            record = dict(event)
            record["seq"] = self._seq + 1
            with open(self.path, "a", encoding="utf-8") as fh:
                if new_file:
                    fh.write(f"#base {self._seq}\n")
                fh.write(json.dumps(record, default=str))
                fh.write("\n")
            self._seq += 1
            self._lines_since_rotate += 1
            if self._lines_since_rotate > self.max_events:
                self._rotate()
            return self._seq
        except Exception:  # noqa: BLE001 — the ledger must never break a turn
            return self._seq

    # ---- replay (crash recovery) ---- #
    def events_after(self, watermark: int) -> Iterator[Dict[str, Any]]:
        """Yield every recorded event whose seq is strictly greater than ``watermark``."""
        try:
            if not os.path.exists(self.path):
                return
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        ev = json.loads(line)
                    except ValueError:
                        continue
                    if int(ev.get("seq", 0)) > int(watermark):
                        yield ev
        except OSError:
            return

    def replay_after(self, watermark: int, learner: Any) -> int:
        """Re-apply learning events after ``watermark`` onto ``learner`` (crash recovery).

        Returns the number of events replayed. Never raises: a protected-core clash or a
        malformed event is simply skipped, exactly as the live learning path would skip it.
        """
        replayed = 0
        record = getattr(learner, "record", None)
        if not callable(record):
            return 0
        for ev in self.events_after(watermark):
            if ev.get("kind", "reward") != "reward":
                continue
            feats = {str(k): float(v) for k, v in (ev.get("features") or {}).items()}
            if not feats:
                continue
            try:
                record(str(ev.get("action", "")), feats, float(ev.get("reward", 0.0)),
                       task=str(ev.get("task", "")))
                replayed += 1
            except Exception:  # noqa: BLE001 — skip what the live path would also refuse
                continue
        return replayed

    # ---- rotation (bounded disk) ---- #
    def _rotate(self) -> None:
        """Trim the ledger to its recent tail, preserving the monotonic seq via a #base header."""
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                lines = [ln for ln in fh if ln.strip() and not ln.startswith("#")]
            keep = lines[-self.rotate_keep:]
            dropped = len(lines) - len(keep)
            base = self._seq - len(keep)
            tmp = f"{self.path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(f"#base {base}\n")
                fh.writelines(keep)
            os.replace(tmp, self.path)
            self._lines_since_rotate = len(keep)
        except OSError:
            pass

    # ---- report ---- #
    def stats(self, watermark: Optional[int] = None) -> Dict[str, Any]:
        """A truthful summary for ``learning_report`` — total events, on-disk tail, unflushed."""
        out: Dict[str, Any] = {
            "events": self._seq,
            "on_disk": self._lines_since_rotate,
            "path": self.path,
        }
        if watermark is not None:
            out["uncheckpointed"] = max(0, self._seq - int(watermark))
        return out


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import tempfile

    print("=" * 70)
    print("NYXARA learning-journal self-test")
    print("=" * 70)

    from nyxara.growth.learn import Learner

    d = tempfile.mkdtemp()
    jpath = os.path.join(d, "learning_journal.jsonl")
    jr = LearningJournal(jpath, max_events=25, rotate_keep=10)

    # a learner learns 20 events; the journal records each
    live = Learner(base_lr=0.2, seed=1)
    watermark_at_checkpoint = 0
    for i in range(20):
        act, feats, rew = "greet_warmly", {"with_master": 1.0}, 1.0
        err = live.record(act, feats, rew, task="greet")
        s = jr.append(LearningEvent(act, feats, rew, task="greet", err=err))
        if i == 9:                       # pretend we checkpointed after the 10th event
            watermark_at_checkpoint = s
    assert jr.seq == 20, jr.seq
    print(f"\nappended            : {jr.seq} events, watermark at {watermark_at_checkpoint}")

    # crash recovery: a learner restored from the 10-event checkpoint replays events 11..20
    restored = Learner(base_lr=0.2, seed=1)
    for _ in range(10):                  # simulate the checkpoint's first-10-events state
        restored.record("greet_warmly", {"with_master": 1.0}, 1.0, task="greet")
    replayed = jr.replay_after(watermark_at_checkpoint, restored)
    assert replayed == 10, replayed
    # after replay, the restored learner matches the live one that saw all 20
    v_live = live.value("greet_warmly", {"with_master": 1.0})
    v_restored = restored.value("greet_warmly", {"with_master": 1.0})
    assert abs(v_live - v_restored) < 1e-9, (v_live, v_restored)
    print(f"crash recovery      : replayed {replayed} post-checkpoint events, "
          f"learner matches live ({v_restored:.4f}) ✓")

    # rotation keeps the ledger bounded but preserves the monotonic seq
    for _ in range(20):
        jr.append(LearningEvent("x", {"f": 1.0}, 0.5))
    assert jr.seq == 40, jr.seq
    reopened = LearningJournal(jpath)
    assert reopened.seq == 40, reopened.seq   # seq survives rotation + reopen
    print(f"rotation            : ledger trimmed, seq preserved across reopen ({reopened.seq}) ✓")

    print("\nALL SELF-TESTS PASSED ✓")
