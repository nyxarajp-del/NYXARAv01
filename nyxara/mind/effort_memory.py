"""NYXARA · mind/effort_memory.py — learned effort allocation (🧠 → 👑, compounding).

The deep-reasoning controller (``mind/deep_reasoning.py``) always climbs the effort ladder to
its ceiling, but it has a fixed *budget* to spend at each rung (self-consistency width). The
question this module answers is **where that budget pays off** — which reasoning *strategy* tends
to produce the best verified answer for a given *kind* of problem. That is not a hyperparameter a
human should hand-tune once; it is something NYXARA learns from her own lived outcomes.

:class:`EffortMemory` maps a coarse **problem signature** (task shape · length · stakes) to a
per-rung :class:`~nyxara.mind.uncertainty.BetaBelief` about "did spending here yield a strong,
verified answer?". Each turn:

* :meth:`record` folds the turn's *verified score* into the belief for the rung that actually
  won — a soft Bayesian update (score is the evidence, ``1 - score`` the counter-evidence), so a
  rung that keeps winning on a signature grows a high mean, and one that never helps decays.
* :meth:`suggest` returns the rung that historically pays off most for a signature (once it has
  seen enough of them), so the controller can pour its *extra* self-consistency samples there —
  the same total compute, aimed where it measurably works.

This is real, measurable compounding with **no model weights involved**, so it works on any
machine (a bare, keyless box included). It only ever records *effort levels* and *verified text
scores* — never any value, persona, or safety-relevant content — so it can never teach over the
immutable core.

Persists as a small JSON table under ``paths.data_dir`` so the lesson survives a restart. Pure
standard library plus :mod:`nyxara.mind.uncertainty`.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from nyxara.mind.uncertainty import BetaBelief

__all__ = ["EffortMemory", "signature"]

_WORD = re.compile(r"[A-Za-z0-9']+")
_MATHY = re.compile(r"[0-9]+\s*[-+*/^=]\s*[0-9]|\bsolve\b|\bcompute\b|\bcalculate\b")
_QUESTION = ("what", "why", "how", "when", "who", "which", "where", "is ", "are ", "can ", "do ",
             "does ", "should ", "would ", "could ", "explain", "?")
_COMMANDY = ("delete", "remove", "run", "exec", "install", "deploy", "write", "create", "make",
             "open", "fetch", "search", "read", "list", "send", "kill", "shutdown")
_HIGH_STAKES = ("delete", "shutdown", "kill", "exec", "secret", "credential", "production",
                "database", "transfer", "payment", "wipe", "format", "root", "attack", "breach")


def signature(stimulus: str) -> str:
    """A coarse, stable bucket for ``stimulus`` — its *shape*, not its content.

    Deliberately low-cardinality (task-shape · length-bucket · stakes-bucket) so the store learns
    fast and generalises across paraphrases, rather than memorising one prompt. Carries no raw
    text of the stimulus, so it never leaks content into the persisted table.
    """
    low = (stimulus or "").strip().lower()
    words = _WORD.findall(low)
    n = len(words)

    if _MATHY.search(low):
        shape = "math"
    elif any(low.startswith(v + " ") or f" {v} " in f" {low} " for v in _COMMANDY):
        shape = "command"
    elif any(low.startswith(q) or q in low for q in _QUESTION):
        shape = "question"
    else:
        shape = "statement"

    length = "short" if n <= 12 else ("medium" if n <= 48 else "long")
    stakes = "high" if any(w in low for w in _HIGH_STAKES) else "normal"
    return f"{shape}/{length}/{stakes}"


class EffortMemory:
    """Learns *which reasoning rung pays off* for each problem signature, from verified outcomes."""

    def __init__(self, *, min_observations: float = 3.0, success_floor: float = 0.5,
                 path: Optional[Path] = None) -> None:
        # (signature -> {rung -> BetaBelief}); beliefs created lazily on first evidence.
        self._table: Dict[str, Dict[int, BetaBelief]] = {}
        self._lock = threading.RLock()
        # a suggestion needs at least this much evidence before it overrides "climb them all".
        self.min_observations = float(min_observations)
        # a verified score at/above this floor counts as the rung genuinely "paying off".
        self.success_floor = float(success_floor)
        self.path = Path(path) if path is not None else None
        if self.path is not None and self.path.exists():
            try:
                self.load(self.path)
            except Exception:  # noqa: BLE001 — a corrupt/absent file must never break reasoning
                pass

    # ------------------------------------------------------------------ #
    # learning
    # ------------------------------------------------------------------ #
    def record(self, sig: str, rung: int, score: float) -> None:
        """Fold a turn's verified ``score`` (0..1) into the belief for the winning ``rung``.

        Soft Bayesian evidence: the score is the success mass, ``1 - score`` the failure mass, so a
        strong answer nudges the rung's mean up and a weak one nudges it down — no hard threshold on
        the update itself (the floor only gates *suggestions*, below).
        """
        s = max(0.0, min(1.0, float(score)))
        with self._lock:
            beliefs = self._table.setdefault(sig, {})
            belief = beliefs.get(rung)
            if belief is None:
                belief = BetaBelief()
                beliefs[rung] = belief
            belief.update(successes=s, failures=1.0 - s)

    # ------------------------------------------------------------------ #
    # querying
    # ------------------------------------------------------------------ #
    def suggest(self, sig: str) -> Optional[int]:
        """The rung that historically pays off most for ``sig``, or None if not yet learned.

        Returns a rung only when (a) that signature has been seen enough times and (b) the best
        rung's success mean clears the floor — otherwise the controller keeps its default of
        spreading effort evenly, because we have not learned anything worth acting on yet.
        """
        with self._lock:
            beliefs = self._table.get(sig)
            if not beliefs:
                return None
            total_obs = sum(b.observations for b in beliefs.values())
            if total_obs < self.min_observations:
                return None
            best_rung = max(beliefs, key=lambda r: beliefs[r].mean)
            if beliefs[best_rung].mean < self.success_floor:
                return None
            return best_rung

    def mean_for(self, sig: str, rung: int) -> Optional[float]:
        """Learned success mean for a (signature, rung), or None if never observed."""
        with self._lock:
            belief = self._table.get(sig, {}).get(rung)
            return None if belief is None else belief.mean

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                sig: {str(rung): {"alpha": b.alpha, "beta": b.beta}
                      for rung, b in beliefs.items()}
                for sig, beliefs in self._table.items()
            }

    def load_dict(self, data: Dict[str, Any]) -> "EffortMemory":
        table: Dict[str, Dict[int, BetaBelief]] = {}
        for sig, beliefs in (data or {}).items():
            if not isinstance(beliefs, dict):
                continue
            row: Dict[int, BetaBelief] = {}
            for rung, params in beliefs.items():
                try:
                    r = int(rung)
                    a = float(params["alpha"])
                    b = float(params["beta"])
                    if a > 0 and b > 0:
                        row[r] = BetaBelief(alpha=a, beta=b)
                except (KeyError, TypeError, ValueError):
                    continue
            if row:
                table[str(sig)] = row
        with self._lock:
            self._table = table
        return self

    def save(self, path: Optional[Path] = None) -> None:
        target = Path(path) if path is not None else self.path
        if target is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)  # atomic on POSIX — a crash mid-write never corrupts the table

    def load(self, path: Optional[Path] = None) -> "EffortMemory":
        target = Path(path) if path is not None else self.path
        if target is None or not target.exists():
            return self
        return self.load_dict(json.loads(target.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import tempfile

    print("=" * 70)
    print("NYXARA effort-memory self-test")
    print("=" * 70)

    # signatures are shape-stable across paraphrase, and separate the obvious kinds
    assert signature("what is the capital of France?").startswith("question/")
    assert signature("delete the production database").startswith("command/")
    assert "high" in signature("delete the production database")
    assert signature("compute 12 * 34").startswith("math/")
    print(f"signatures           : {signature('why is the sky blue?')} | "
          f"{signature('deploy to production now')}")

    em = EffortMemory(min_observations=3.0, success_floor=0.5)
    sig = signature("why is the sky blue?")

    # cold: nothing learned yet -> no suggestion (controller spreads effort evenly)
    assert em.suggest(sig) is None

    # teach it that rung 3 (MCTS) keeps winning on this signature, rung 1 keeps failing
    for _ in range(5):
        em.record(sig, 3, 0.9)
        em.record(sig, 1, 0.2)
    assert em.suggest(sig) == 3, em.suggest(sig)
    assert em.mean_for(sig, 3) > em.mean_for(sig, 1)
    print(f"learned best rung    : {em.suggest(sig)}  "
          f"(rung3 mean={em.mean_for(sig, 3):.2f}, rung1 mean={em.mean_for(sig, 1):.2f})")

    # persistence round-trips the learned table
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "effort.json"
        em.save(p)
        em2 = EffortMemory().load(p)
        assert em2.suggest(sig) == 3
        assert abs(em2.mean_for(sig, 3) - em.mean_for(sig, 3)) < 1e-9
    print("persistence          : learned effort survives a restart ✓")

    print("\nALL SELF-TESTS PASSED ✓")
