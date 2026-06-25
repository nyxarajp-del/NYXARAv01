"""NYXARA · sim/real_environment.py — a real sensorimotor environment (🌍, stdlib-only).

The world model (:mod:`nyxara.mind.world_model`) is a real learned-dynamics engine, but
its only fuel used to be a degenerate 1-D "belief-count" signal from the autonomous
scientist — a real engine running on fake petrol. This module gives it a *genuine*
environment to learn from: the actual machine.

A :class:`RealEnvironment` owns a private scratch directory and senses a fixed,
six-dimensional, fully-numeric **state** from the real filesystem plus live CPU/RAM
readings (via :func:`nyxara.kernel.compute.compute_report`). It exposes a small set of
**safe, reversible, scratch-confined actions** (create/append/delete a file, make a
directory, or do nothing) that genuinely change that state. Each :meth:`step` performs a
real operation and observes the real before/after state and a real reward — a true
``(state, action, next_state, reward)`` transition the world model can learn from.

Safety is structural, not incidental:

* every path is resolved and asserted to live **strictly under** the scratch root
  (``..``/symlink escapes are refused);
* deletes only target files the environment itself created and journaled, so they are
  reversible (:meth:`restore_all`);
* a hard ``max_files`` cap bounds growth, and :meth:`safe_actions` self-regulates;
* the scratch root is :meth:`cleanup`-able and the class is a context manager.

Nothing this module does ever touches the real system outside its scratch directory.
"""

from __future__ import annotations

import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from nyxara.kernel.compute import compute_report

__all__ = ["State", "Transition", "RealEnvironment", "sensorimotor_tick"]

# A fixed, fully-numeric state vector, in comparable units so no single dimension drowns
# the others in the world model's distance metric:
#   (file_count, total_kb, free_disk_gb, mem_available_gb, cpu_count, subdir_count)
# Every reading is real — units are chosen so the controllable dims (files/dirs) and the
# ambient machine dims (disk/RAM) live on the same numeric scale.
State = Tuple[float, float, float, float, float, float]
_I_FILES, _I_KB, _I_FREE_GB, _I_MEM_GB, _I_CPU, _I_SUBDIRS = range(6)

ACTIONS: Tuple[str, ...] = ("noop", "create_file", "append", "delete_file", "make_dir")


@dataclass
class Transition:
    """One genuine environment transition — mirrors :class:`nyxara.mind.world_model.Transition`."""
    state: State
    action: str
    next_state: State
    reward: float = 0.0


class RealEnvironment:
    """A real, safe, scratch-confined environment that produces genuine transitions.

    The dominant *learnable* dynamics come from the environment's own file operations
    (e.g. ``create_file`` always raises ``file_count`` by exactly one); the disk/RAM/CPU
    dimensions ground the vector in the real machine.
    """

    ACTIONS = ACTIONS

    def __init__(self, root: Optional[str] = None, *, max_files: int = 64,
                 chunk_bytes: int = 256, seed: int = 0) -> None:
        if root is None:
            root = tempfile.mkdtemp(prefix="nyxara-senv-")
            self._owns_root = True
        else:
            os.makedirs(root, exist_ok=True)
            self._owns_root = False
        self._root = os.path.realpath(root)
        self.max_files = max(1, int(max_files))
        self.chunk_bytes = max(1, int(chunk_bytes))
        self._rng = random.Random(seed)
        self._created: List[str] = []          # files we made (and may delete)
        self._dirs: List[str] = []             # subdirs we made
        self._journal: dict[str, bytes] = {}   # path -> last-known content (for restore)
        self._fseq = 0
        self._dseq = 0
        # a rolling window of recent states — turns discrete snapshots into a CONTINUOUS
        # sensorimotor stream the world model (and curiosity) can read dynamics from.
        from collections import deque as _deque
        self._history: Any = _deque(maxlen=64)

    @property
    def root(self) -> str:
        return self._root

    # ------------------------------------------------------------------ #
    # Sensing — the real, numeric state
    # ------------------------------------------------------------------ #
    def sense(self) -> State:
        """Read the six-dimensional state from the real filesystem + live compute report."""
        file_count = 0
        total_bytes = 0
        subdir_count = 0
        try:
            for entry in os.scandir(self._root):
                try:
                    if entry.is_file(follow_symlinks=False):
                        file_count += 1
                        total_bytes += entry.stat(follow_symlinks=False).st_size
                    elif entry.is_dir(follow_symlinks=False):
                        subdir_count += 1
                except OSError:  # a vanished entry mid-scan — count what we can
                    continue
        except OSError:
            pass

        free_gb = 0.0
        try:
            st = os.statvfs(self._root)
            free_gb = (st.f_bavail * st.f_frsize) / 1e9
        except OSError:
            pass

        rep = compute_report()
        mem_gb = float(rep.memory.get("available_mb") or 0.0) / 1024.0
        cpu = float(rep.cpu_count or 1)

        return (float(file_count), total_bytes / 1024.0, free_gb,
                mem_gb, cpu, float(subdir_count))

    # ------------------------------------------------------------------ #
    # Acting
    # ------------------------------------------------------------------ #
    def safe_actions(self) -> List[str]:
        """The actions that are sensible right now (self-regulating against the cap)."""
        actions = ["noop"]
        n_files = sum(1 for p in self._created if os.path.isfile(p))
        if n_files < self.max_files:
            actions.append("create_file")
        if n_files > 0:
            actions.append("append")
            actions.append("delete_file")
        if len(self._dirs) < self.max_files:
            actions.append("make_dir")
        return actions

    def step(self, action: str) -> Transition:
        """Perform a real action, observe the real before/after state, score a real reward.

        Never raises: a refused or failed operation is *data* (it yields no state change
        and a negative reward), not a crash.
        """
        before = self.sense()
        if not self._history:
            self._history.append(before)
        applied = True
        try:
            applied = self._apply(action)
        except Exception:  # noqa: BLE001 — a failed op is a negative-reward observation
            applied = False
        after = self.sense()
        self._history.append(after)            # the stream advances — dynamics become readable
        reward = self._reward(before, action, after, applied)
        return Transition(state=before, action=action, next_state=after, reward=reward)

    def dynamics(self) -> Tuple[float, ...]:
        """Per-step rate-of-change of each state dimension over the rolling window.

        This is the genuinely *continuous* signal a snapshot lacks: how fast files/bytes/disk/RAM
        are moving, not just where they are. Returns a zero vector until at least two states have
        streamed in. Downstream (curiosity, anomaly) can read it without changing the 6-D State.
        """
        if len(self._history) < 2:
            return (0.0,) * 6
        first, last = self._history[0], self._history[-1]
        n = len(self._history) - 1
        return tuple((last[i] - first[i]) / n for i in range(len(first)))

    # ------------------------------------------------------------------ #
    # Authored content — perceivable artifacts (for the embodied loop)
    # ------------------------------------------------------------------ #
    def write_text(self, name: str, text: str) -> Optional[str]:
        """Create a scratch file holding real *text* (so a sense can later read meaning).

        Unlike :meth:`_apply`'s ``create_file`` (random bytes — perceivable as nothing), this
        writes authored content an NLP/vision pass can actually understand. Journaled and
        guarded exactly like every other artifact, so it stays reversible and scratch-confined.
        Returns the path, or ``None`` if the file cap is reached. Never escapes the root.
        """
        return self.write_bytes(name, text.encode("utf-8"))

    def write_bytes(self, name: str, data: bytes) -> Optional[str]:
        """Create a scratch file holding raw ``data`` (e.g. a generated PNG/WAV). Reversible."""
        if sum(1 for p in self._created if os.path.isfile(p)) >= self.max_files:
            return None
        path = self._guard(os.path.join(self._root, name))
        with open(path, "wb") as fh:
            fh.write(data)
        if path not in self._created:
            self._created.append(path)
        self._journal[path] = data
        return path

    def _apply(self, action: str) -> bool:
        """Execute one action for real in the scratch dir. Returns True if it did something."""
        if action == "noop":
            return True
        if action == "create_file":
            existing = sum(1 for p in self._created if os.path.isfile(p))
            if existing >= self.max_files:
                return False
            path = self._guard(os.path.join(self._root, f"f{self._fseq}.dat"))
            self._fseq += 1
            content = self._rng.randbytes(self.chunk_bytes)
            with open(path, "wb") as fh:
                fh.write(content)
            self._created.append(path)
            self._journal[path] = content
            return True
        if action == "append":
            files = [p for p in self._created if os.path.isfile(p)]
            if not files:
                return False
            path = self._guard(self._rng.choice(files))
            extra = self._rng.randbytes(self.chunk_bytes)
            with open(path, "ab") as fh:
                fh.write(extra)
            self._journal[path] = self._journal.get(path, b"") + extra
            return True
        if action == "delete_file":
            files = [p for p in self._created if os.path.isfile(p)]
            if not files:
                return False
            path = self._guard(self._rng.choice(files))
            # content is already journaled, so the delete is reversible via restore_all()
            os.remove(path)
            if path in self._created:
                self._created.remove(path)
            return True
        if action == "make_dir":
            if len(self._dirs) >= self.max_files:
                return False
            path = self._guard(os.path.join(self._root, f"d{self._dseq}"))
            self._dseq += 1
            os.makedirs(path, exist_ok=True)
            self._dirs.append(path)
            return True
        return False  # unknown action — a no-op observation

    def _reward(self, before: State, action: str, after: State, applied: bool) -> float:
        d_files = after[_I_FILES] - before[_I_FILES]
        d_kb = after[_I_KB] - before[_I_KB]
        d_dirs = after[_I_SUBDIRS] - before[_I_SUBDIRS]

        if action == "noop":
            return 0.0
        if not applied:
            return -1.0  # an op we expected to do something but couldn't

        if action == "create_file":
            base = 0.5 if d_files > 0 else -1.0
        elif action == "append":
            base = 0.2 if d_kb > 0 else -1.0
        elif action == "delete_file":
            base = -0.3 if d_files < 0 else -1.0   # cleanup is mildly costly but valid
        elif action == "make_dir":
            base = 0.1 if d_dirs > 0 else -1.0
        else:
            base = 0.0

        # honest disk-health term: penalise unbounded growth (per MB grown)
        base -= 0.001 * max(0.0, d_kb / 1024.0)
        return base

    # ------------------------------------------------------------------ #
    # Corrigibility / teardown
    # ------------------------------------------------------------------ #
    def restore_all(self) -> int:
        """Recreate any journaled file that is currently missing (undo deletions)."""
        restored = 0
        for path, content in self._journal.items():
            try:
                if not os.path.exists(self._guard(path)):
                    with open(path, "wb") as fh:
                        fh.write(content)
                    if path not in self._created:
                        self._created.append(path)
                    restored += 1
            except Exception:  # noqa: BLE001 — best-effort restore
                continue
        return restored

    def cleanup(self) -> None:
        """Remove the scratch root iff this environment created it. Never raises."""
        if self._owns_root:
            shutil.rmtree(self._root, ignore_errors=True)

    def __enter__(self) -> "RealEnvironment":
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _guard(self, path: str) -> str:
        """Resolve ``path`` and assert it lies strictly under the scratch root."""
        real = os.path.realpath(path)
        if real != self._root and not real.startswith(self._root + os.sep):
            raise ValueError(f"path escapes scratch root: {path!r}")
        return real


def sensorimotor_tick(env: RealEnvironment, world_model: object, *,
                      rng: Optional[random.Random] = None) -> Transition:
    """Pick a safe action, take it for real, and feed the genuine transition to ``world_model``.

    This is the single entry point the orchestrator calls each idle tick.
    """
    chooser = rng or env._rng
    options = env.safe_actions() or ["noop"]
    action = chooser.choice(options)
    tr = env.step(action)
    try:
        world_model.observe(tr.state, tr.action, tr.next_state, reward=tr.reward)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 — feeding the model is best-effort, never fatal
        pass
    return tr


def sensorimotor_stream(env: RealEnvironment, world_model: object, *, steps: int = 4,
                        rng: Optional[random.Random] = None) -> List[Transition]:
    """Drive a CONTINUOUS burst of sensorimotor ticks, feeding each real transition to the model.

    A single isolated tick per idle cycle is a snapshot; a short stream is a trajectory — the world
    model learns action-conditioned dynamics from a *sequence*, and ``env.dynamics()`` becomes a
    live rate-of-change signal. Bounded by ``steps`` so an idle tick stays cheap; never raises.
    """
    out: List[Transition] = []
    for _ in range(max(1, int(steps))):
        try:
            out.append(sensorimotor_tick(env, world_model, rng=rng))
        except Exception:  # noqa: BLE001 — the stream is best-effort, never fatal
            break
    return out


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.mind.world_model import WorldModel

    print("=" * 70)
    print("NYXARA real-environment self-test")
    print("=" * 70)

    with RealEnvironment(seed=7) as env:
        wm = WorldModel(k=3, distance_scale=50.0)
        for _ in range(80):
            sensorimotor_tick(env, wm)
        s = env.sense()
        up = wm.predict(s, "create_file")
        down = wm.predict(s, "delete_file")
        noop = wm.predict(s, "noop")
        print(f"\nstate now            : {tuple(round(x, 1) for x in s)}")
        print(f"world transitions    : {len(wm)}  over actions {sorted(wm.actions())}")
        print(f"predict create_file  : Δfiles={up.next_state[0]-s[0]:+.2f}  conf={up.confidence:.2f}")
        print(f"predict delete_file  : Δfiles={down.next_state[0]-s[0]:+.2f}  conf={down.confidence:.2f}")
        print(f"predict noop         : Δfiles={noop.next_state[0]-s[0]:+.2f}  conf={noop.confidence:.2f}")

        assert down.next_state[0] < noop.next_state[0] < up.next_state[0], \
            "model failed to learn action-conditioned real dynamics"
        assert up.confidence > 0.5
    print("\nALL SELF-TESTS PASSED ✓")
