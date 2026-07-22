"""NYXARA · mind/superposition_reasoner.py — Quantum-Probabilistic Superposition Reasoning (⚛, Rule 4).

The critique this module answers, in the Master's words: *"NYXARA ek waqt par ek hi logic ya path ko
diagnose aur execute karti hai. Naya: ek hi waqt par kai alag-alag logical conclusions (superposition)
simulate kare, jo sabse optimal future outcome ho usi path ko collapse karke live kare."*

Her :mod:`nyxara.quantum.superposition_states` already provides the **real** maths — a state of many
contradictory hypotheses, each with a Born-rule amplitude, a Bayesian ``observe`` update, Shannon
``entropy``, and a threshold ``collapse`` that stays superposed rather than bluffing. It was, however,
never wired into *reasoning*: it sat on the shelf as pure algebra. This module is the wire.

:class:`SuperpositionReasoner` takes several **candidate solution paths** for one problem — generated
in parallel by whatever reasoners are available (the base reasoner, the role council, MCTS branches,
the dual-process fast/slow split) — and:

1. seeds a :class:`~nyxara.quantum.superposition_states.Superposition` with one hypothesis per path,
   amplitude ∝ √(the path's own quality/confidence);
2. **scores each path by a *simulated future outcome*** — a pluggable scorer that, by default, blends
   the path's calibrated confidence, an optional world-model / simulation rollout value, and an
   optional grounded-verification check — and folds those in as Bayesian likelihoods (``observe``);
3. **collapses** to the single optimal path *only when one dominates past a threshold* — otherwise it
   honestly **stays superposed** (``decided=False``), so the caller abstains / escalates / thinks
   harder rather than committing to a coin-flip.

This is genuine "explore many conclusions at once, land on the best" — bounded (a small, capped set of
paths, a shallow rollout), deterministic, and **LLM-free at this layer** (the paths may come from any
generator; the *scoring and collapse* are her own maths). It proposes a selection; it reaches around no
gate — the kernel still disposes.

Reuses :mod:`nyxara.quantum.superposition_states` (the real amplitudes) and, when present,
:mod:`nyxara.mind.uncertainty` (via ``as_dirichlet``) so a not-yet-collapsed belief flows into the
calibrated abstention machinery. Pure standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.quantum.superposition_states import Superposition

__all__ = ["ThoughtPath", "SuperpositionResult", "SuperpositionReasoner"]


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class ThoughtPath:
    """One candidate line of reasoning held in superposition with the others."""

    label: str                       # a stable identifier for the path (source/idx)
    payload: Any = None              # the actual candidate answer/decision object
    quality: float = 0.5             # the path's own prior quality/confidence in [0,1]
    source: str = ""                 # which generator produced it (base/council/mcts/...)
    outcome: float = 0.0             # the simulated-future-outcome score (filled by the reasoner)

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "quality": round(self.quality, 4),
                "source": self.source, "outcome": round(self.outcome, 4)}


@dataclass
class SuperpositionResult:
    """The outcome of collapsing the superposed thought-streams."""

    label: Optional[str]
    payload: Any
    probability: float
    decided: bool                    # True iff one path dominated past the collapse threshold
    entropy: float                   # how superposed she still was, in bits
    margin: float                    # p(best) − p(runner-up)
    runner_up: Optional[str]
    n_paths: int
    paths: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "probability": round(self.probability, 6),
                "decided": self.decided, "entropy": round(self.entropy, 6),
                "margin": round(self.margin, 6), "runner_up": self.runner_up,
                "n_paths": self.n_paths, "paths": self.paths}

    def summary(self) -> str:
        if self.label is None:
            return "superposition: no paths to reason over"
        state = "COLLAPSED" if self.decided else "still superposed (abstain/deepen)"
        return (f"superposition[{self.n_paths} paths] → {state}: '{self.label}' "
                f"p={self.probability:.3f} margin={self.margin:.3f} entropy={self.entropy:.3f}")


# a scorer maps a ThoughtPath (+ the stimulus) to a likelihood P(good | path) in (0, 1]
Scorer = Callable[["ThoughtPath", str], float]


# --------------------------------------------------------------------------- #
# The reasoner
# --------------------------------------------------------------------------- #
class SuperpositionReasoner:
    """Hold candidate reasoning paths in superposition and collapse to the optimal by simulated outcome."""

    def __init__(self, *, collapse_threshold: float = 0.55, max_paths: int = 12,
                 world_model: Any = None, grounded_verifier: Any = None, simulator: Any = None,
                 rollout_depth: int = 2) -> None:
        self.collapse_threshold = float(collapse_threshold)
        self.max_paths = max(1, int(max_paths))
        self.world_model = world_model
        self.grounded_verifier = grounded_verifier
        self.simulator = simulator
        self.rollout_depth = max(0, int(rollout_depth))

    # ---- public entry ---------------------------------------------------- #
    def deliberate(self, stimulus: str, hypotheses: Sequence[Any], *,
                   scorer: Optional[Scorer] = None) -> SuperpositionResult:
        """Reason over ``hypotheses`` in superposition and return the collapse.

        ``hypotheses`` may be :class:`ThoughtPath` objects, ``(label, payload, quality)`` tuples, or
        any object exposing ``.text``/``.confidence`` (a reasoner Candidate) — all are normalised to
        paths. ``scorer`` overrides the default simulated-outcome estimator."""
        paths = self._normalize(hypotheses)
        if not paths:
            return SuperpositionResult(None, None, 0.0, False, 0.0, 0.0, None, 0)
        score = scorer or self._default_scorer

        state = Superposition(collapse_threshold=self.collapse_threshold)
        import math
        # amplitude ∝ √quality  ⇒  Born-rule probability ∝ quality (before evidence). Seed all at
        # once via add_many so the state renormalises a single time — adding one-by-one would make the
        # per-add renormalisation order-dependent (equal priors would NOT stay equal).
        state.add_many({p.label: math.sqrt(max(1e-6, min(1.0, p.quality))) for p in paths})

        # evidence: the *simulated future outcome* of each path, folded in as a Bayesian likelihood.
        likelihoods: Dict[str, float] = {}
        for p in paths:
            try:
                lik = float(score(p, stimulus))
            except Exception:  # noqa: BLE001 — a scorer failure must not sink the whole deliberation
                lik = max(1e-3, p.quality)
            lik = max(1e-3, min(1.0, lik))
            p.outcome = lik
            likelihoods[p.label] = lik
        state.observe(likelihoods)

        collapse = state.collapse()
        by_label = {p.label: p for p in paths}
        chosen = by_label.get(collapse.label)
        return SuperpositionResult(
            label=collapse.label,
            payload=(chosen.payload if chosen is not None else collapse.payload),
            probability=collapse.probability, decided=collapse.decided, entropy=collapse.entropy,
            margin=collapse.margin, runner_up=collapse.runner_up, n_paths=len(paths),
            paths=sorted((p.to_dict() for p in paths), key=lambda d: d["outcome"], reverse=True))

    # ---- default simulated-outcome scorer -------------------------------- #
    def _default_scorer(self, path: ThoughtPath, stimulus: str) -> float:
        """Blend the path's own confidence with an optional world-model / simulation rollout value and
        an optional grounded-verification check. Every signal is best-effort — absent ones simply drop
        out (their weight is not faked), so on a bare box this degrades to the path's own quality."""
        signals: List[Tuple[float, float]] = [(max(1e-3, path.quality), 1.0)]   # (value, weight)

        rollout = self._rollout_value(path, stimulus)
        if rollout is not None:
            signals.append((rollout, 1.5))                 # a simulated outcome outweighs a prior guess

        grounded = self._grounded_value(path, stimulus)
        if grounded is not None:
            signals.append((grounded, 2.0))                # a verified answer outweighs an unverified one

        num = sum(v * w for v, w in signals)
        den = sum(w for _, w in signals)
        return max(1e-3, min(1.0, num / den if den > 0 else path.quality))

    def _rollout_value(self, path: ThoughtPath, stimulus: str) -> Optional[float]:
        """Simulate the path forward a few steps and score the terminal state in [0,1], if a world
        model / simulator is available. Returns None (honest 'no signal') otherwise."""
        engine = self.simulator or self.world_model
        if engine is None or self.rollout_depth <= 0:
            return None
        text = _payload_text(path.payload)
        for meth in ("rollout", "simulate", "evaluate", "score", "value"):
            fn = getattr(engine, meth, None)
            if callable(fn):
                try:
                    out = fn(text, depth=self.rollout_depth) if meth == "rollout" else fn(text)
                    val = _as_unit(out)
                    if val is not None:
                        return val
                except TypeError:
                    try:
                        val = _as_unit(fn(text))
                        if val is not None:
                            return val
                    except Exception:  # noqa: BLE001
                        return None
                except Exception:  # noqa: BLE001 — a failed rollout is 'no signal', never a fake score
                    return None
        return None

    def _grounded_value(self, path: ThoughtPath, stimulus: str) -> Optional[float]:
        """Ask the grounded verifier whether the path's answer actually checks out, in [0,1]. None
        when no verifier is wired."""
        gv = self.grounded_verifier
        if gv is None:
            return None
        text = _payload_text(path.payload)
        for meth in ("verify", "check", "grounded", "score"):
            fn = getattr(gv, meth, None)
            if callable(fn):
                try:
                    out = fn(stimulus, text) if meth in ("verify", "check") else fn(text)
                    val = _as_unit(out)
                    if val is not None:
                        return val
                except Exception:  # noqa: BLE001
                    return None
        return None

    # ---- normalisation --------------------------------------------------- #
    def _normalize(self, hypotheses: Sequence[Any]) -> List[ThoughtPath]:
        paths: List[ThoughtPath] = []
        for i, h in enumerate(hypotheses):
            if isinstance(h, ThoughtPath):
                paths.append(h)
            elif isinstance(h, tuple) and len(h) >= 2:
                label = str(h[0])
                payload = h[1]
                quality = float(h[2]) if len(h) > 2 else 0.5
                paths.append(ThoughtPath(label=label, payload=payload, quality=quality,
                                         source="tuple"))
            else:
                # duck-typed reasoner candidate: use its confidence as quality
                quality = float(getattr(h, "confidence", 0.5) or 0.5)
                source = str(getattr(h, "source", "") or getattr(h, "kind", "") or "candidate")
                label = f"{source}:{i}"
                paths.append(ThoughtPath(label=label, payload=h, quality=quality, source=source))
            if len(paths) >= self.max_paths:
                break
        return paths


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _payload_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    return str(getattr(payload, "text", "") or getattr(payload, "answer", "") or payload)


def _as_unit(x: Any) -> Optional[float]:
    """Coerce a scorer's return (a float, a bool, or an object with .score/.value/.probability) into a
    [0,1] value, or None when it carries no usable signal."""
    if x is None:
        return None
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    if isinstance(x, (int, float)):
        v = float(x)
        return max(0.0, min(1.0, v))
    for attr in ("score", "value", "probability", "confidence"):
        v = getattr(x, attr, None)
        if isinstance(v, (int, float)):
            return max(0.0, min(1.0, float(v)))
    ok = getattr(x, "ok", None)
    if isinstance(ok, bool):
        return 1.0 if ok else 0.0
    return None


# --------------------------------------------------------------------------- #
# Self-test / demo — real superposition reasoning, no torch, no network
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA superposition-reasoning self-test — many paths → collapse to the best")
    print("=" * 70)

    # 1) a clearly-best path (verified by a stub grounded verifier) collapses out --------------
    class _GV:
        """A stub grounded verifier: only the answer '42' checks out."""

        def verify(self, stimulus, text):
            return 1.0 if "42" in text else 0.05

    r = SuperpositionReasoner(collapse_threshold=0.55, grounded_verifier=_GV())
    hyps = [("guess_a", "the answer is 7", 0.5),
            ("guess_b", "the answer is 42", 0.5),
            ("guess_c", "the answer is 13", 0.5)]
    res = r.deliberate("what is 6*7?", hyps)
    print("\n" + res.summary())
    assert res.label == "guess_b", res.label
    assert res.decided, "a verified dominant path should collapse"
    assert res.payload == "the answer is 42"
    print("verified path       : collapsed to the grounded answer ✓")

    # 2) genuinely ambiguous paths stay SUPERPOSED (decided=False) — she does not bluff --------
    r2 = SuperpositionReasoner(collapse_threshold=0.7)
    amb = [("a", "maybe X", 0.5), ("b", "maybe Y", 0.5), ("c", "maybe Z", 0.5)]
    res2 = r2.deliberate("an ambiguous question", amb)
    print("\n" + res2.summary())
    assert not res2.decided, "three tied paths must not collapse past a 0.7 threshold"
    assert res2.entropy > 1.0, "entropy should be high when undecided"
    print("ambiguous           : stayed superposed instead of guessing ✓")

    # 3) a higher-quality prior tilts the Born-rule probability (before evidence) ---------------
    r3 = SuperpositionReasoner(collapse_threshold=0.5)
    tilt = [("strong", "well-supported", 0.9), ("weak", "shaky", 0.2)]
    res3 = r3.deliberate("q", tilt)
    print("\n" + res3.summary())
    assert res3.label == "strong"
    print("prior tilt          : the better-supported path leads ✓")

    # 4) empty input is an honest no-op --------------------------------------------------------
    res4 = SuperpositionReasoner().deliberate("q", [])
    assert res4.label is None and not res4.decided
    print("empty input         : honest no-op ✓")

    # 5) a world-model rollout can overturn a confident-but-wrong prior ------------------------
    class _Sim:
        """A stub simulator: rolling a path forward reveals 'safe' is good, 'risky' is bad."""

        def rollout(self, text, depth=2):
            return 0.95 if "safe" in text else 0.1

    r5 = SuperpositionReasoner(collapse_threshold=0.55, simulator=_Sim(), rollout_depth=2)
    paths = [("risky", "take the risky path", 0.8), ("safe", "take the safe path", 0.55)]
    res5 = r5.deliberate("which action?", paths)
    print("\n" + res5.summary())
    assert res5.label == "safe", "the simulated future outcome should overturn the confident prior"
    print("outcome rollout     : simulated future overturned a confident-but-worse prior ✓")

    print("\nALL SELF-TESTS PASSED ✓")
