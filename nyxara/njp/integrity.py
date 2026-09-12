"""NYXARA · njp/integrity.py — an experiment may not quietly edit the instruments (🔐, NJP V.93).

Twice now this package has destroyed a working module by writing a new one over it. V.58 took
``njp/tasks.py``, a coding-task bank eight call sites depended on. V.92 took ``njp/economy.py``, a
cognitive budget organ, while building the very module that measures whether a vocabulary is
honest. Both were restored from git; neither was noticed by a test, because the tests that would
have caught them were the tests of the module that had just been deleted.

**The check that stops both takes one second and was not run.** ``does this path already exist``.
So it stops being something to remember:

* :func:`claim` refuses a path that is already occupied, unless the caller says outright that a
  rewrite is what it means. That is the *pre*-condition, and it is the one that matters — by the
  time a fingerprint notices, the original is gone and only version control can say what it was.
* :func:`watch` takes a fingerprint of the instruments, runs an experiment, and takes another. A
  file that changed and was not declared is a failed experiment, whatever the experiment's own
  numbers said.

**Why a green test suite is not enough.** A run that mutates the thing measuring it can produce any
result at all and look calm doing it. The numbers are downstream of the instruments; if the
instruments moved during the run, the numbers are about nothing, and that is the same sentence this
package has written at four different levels — a spoiled experiment, a leaky apparatus, a control
shaped like its target, and now a scientist editing the laboratory.

**What this deliberately does not do.** It does not forbid change. Every version here changes files
and must. It forbids **undeclared** change, which is a different and much narrower thing: say what
you are going to touch, and then touching anything else is the failure.

Pure standard library.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

__all__ = ["Fingerprint", "Disturbed", "Occupied", "fingerprint", "claim", "watch", "compare",
           "INSTRUMENTS"]

#: Where the instruments live. A default rather than a law: :func:`watch` takes whatever it is
#: given, and this is what an experiment inside this package should be watching when it has no
#: opinion of its own.
INSTRUMENTS: Tuple[str, ...] = ("nyxara/njp",)


class Occupied(Exception):
    """A path that already holds something was claimed for something new."""


class Disturbed(Exception):
    """An experiment changed a file it never said it would."""


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


@dataclass(frozen=True)
class Fingerprint:
    """What a set of files looked like at one moment."""

    taken: Dict[str, str] = field(default_factory=dict)

    @property
    def files(self) -> List[str]:
        return sorted(self.taken)

    def to_dict(self) -> Dict[str, Any]:
        return {"files": len(self.taken), "taken": dict(self.taken)}


def fingerprint(roots: Sequence[str] = INSTRUMENTS, *, at: Optional[Path] = None,
                suffix: str = ".py") -> Fingerprint:
    """Hash every source file under ``roots``. Cheap enough to run around every experiment."""
    base = Path(at) if at else Path.cwd()
    taken: Dict[str, str] = {}
    for root in roots:
        here = base / root
        if not here.exists():
            continue
        if here.is_file():
            taken[str(here.relative_to(base))] = _hash(here)
            continue
        for path in sorted(here.rglob(f"*{suffix}")):
            if "__pycache__" in path.parts:
                continue
            taken[str(path.relative_to(base))] = _hash(path)
    return Fingerprint(taken=taken)


@dataclass
class Change:
    """What moved between two fingerprints."""

    edited: List[str] = field(default_factory=list)
    added: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)

    @property
    def any(self) -> bool:
        return bool(self.edited or self.added or self.removed)

    def undeclared(self, declared: Iterable[str]) -> List[str]:
        """Changes nobody said were coming. Additions are not among them.

        A new file harms nothing; an **edited or removed** one is an instrument that moved while it
        was being used. That asymmetry is the whole policy, and it is why this does not simply
        forbid change.
        """
        allowed = {str(Path(d)) for d in declared}
        return sorted(p for p in self.edited + self.removed if p not in allowed)

    def to_dict(self) -> Dict[str, Any]:
        return {"edited": self.edited, "added": self.added, "removed": self.removed}


def compare(before: Fingerprint, after: Fingerprint) -> Change:
    """What changed between two fingerprints, split by kind."""
    out = Change()
    for path, was in before.taken.items():
        now = after.taken.get(path)
        if now is None:
            out.removed.append(path)
        elif now != was:
            out.edited.append(path)
    out.added = sorted(set(after.taken) - set(before.taken))
    out.edited.sort()
    out.removed.sort()
    return out


def claim(path: str, *, rewriting: bool = False, at: Optional[Path] = None) -> Path:
    """Take a path for something new, and refuse if something is already there.

    The one-second check that would have saved ``tasks.py`` and ``economy.py``. ``rewriting=True``
    is how a caller says it knows the file exists and means to replace it — which makes the
    dangerous case something somebody had to type, rather than something that happens by default.
    """
    here = (Path(at) if at else Path.cwd()) / path
    if here.exists() and not rewriting:
        raise Occupied(
            f"{path} already exists ({here.stat().st_size} bytes). Pass rewriting=True if "
            f"replacing it is what you mean, or pick a name nothing answers to.")
    return here


@contextmanager
def watch(roots: Sequence[str] = INSTRUMENTS, *, touching: Sequence[str] = (),
          at: Optional[Path] = None) -> Iterator[Fingerprint]:
    """Run an experiment between two fingerprints and refuse undeclared edits.

    ``touching`` is what the experiment says it will change. Anything else that moves raises
    :class:`Disturbed` — not because change is wrong, but because a run that edits the instruments
    it is being measured by has produced a number about nothing, however calm it looked.
    """
    before = fingerprint(roots, at=at)
    yield before
    after = fingerprint(roots, at=at)
    moved = compare(before, after).undeclared(touching)
    if moved:
        raise Disturbed(
            "the experiment changed files it never declared: " + ", ".join(moved)
            + ". Whatever it measured, it measured with instruments that moved while it ran.")
