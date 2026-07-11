"""NYXARA · mind/deep_reasoning.py — the always-maximum deep-reasoning controller (⬆⬆ → 👑).

**The problem this closes.** NYXARA's raw intelligence has a ceiling: the depth of a single pass
through a fixed base model. Everything else the mind does — routing, verifying, polishing — makes
that pass *cleaner*, not *deeper*. A frontier answer that a bigger model reaches in one shot, a
smaller model can often still reach by **thinking longer**: sampling several lines of reasoning,
searching over them, and keeping only what an independent verifier scores highest. That is real
"effective depth" — not a claim that the base model got smarter, but conclusions a single forward
pass could not reach, obtained with compute NYXARA spends *herself*.

Today that depth exists but is **capped by static config knobs** and never escalated as one act.
:class:`DeepReasoner` is the controller that turns it into one: on every genuine reasoning turn it
**climbs the whole effort ladder to its ceiling** and keeps the verifier-best answer across all of
it. More depth can only help — a worse rung is simply discarded.

    L1  self-consistency   → mind/deliberate.py, several sampled decodes, consensus shape
    L2  deliberation       → mind/deliberate.py, think → decide → self-critique → revise
    L3  search-over-reason  → mind/mcts_reasoner.py, MCTS over reasoning paths
    L4  verified refinement → mind/recursive_improver.py, N critique-improve cycles, keep-best

Each rung is an existing, real mechanism; this module is only the *controller* — it drives them at
escalating budgets, scores every product with the shared intrinsic verifier
(:func:`~nyxara.mind.router.answer_quality`), and returns the single best :class:`Candidate`.

**Who decides — NYXARA, not the model.** The controller lives in her loop and is driven by *her*
measured signals (the verifier's score of each answer, self-consistency), not by asking the LLM
how hard to try. And it **compounds**: an :class:`~nyxara.mind.effort_memory.EffortMemory` learns,
from lived verified outcomes, *which rung pays off* for each kind of problem, and the controller
pours its extra self-consistency samples there — the same maximum budget, aimed where it works.
So her reasoning at turn 100 allocates depth measurably better than at turn 1.

**Safety and honesty are untouched.** Every rung only ever yields a proposal; the kernel still
disposes through every gate. The exact/verifiable short-circuits upstream (a computed faculty
answer, a confident own-model handoff) still win *before* this runs, because an exact answer beats
any search. A runaway guard (wall-clock ceiling) bounds the spend. With no real provider, or on any
failure, the controller returns ``None`` so the caller keeps its existing single-pass path — the
offline machine behaves exactly as before.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from nyxara.kernel.config import NyxaraSettings, get_settings

__all__ = ["DeepReasoner", "LadderResult"]

# Ladder rungs, ranked by depth/cost. Mirrored as ints in mind/effort_memory.py.
RUNG_CONSISTENCY = 1
RUNG_DELIBERATE = 2
RUNG_MCTS = 3
RUNG_REFINE = 4
_RUNG_NAMES = {1: "self-consistency", 2: "deliberation", 3: "mcts-search", 4: "verified-refine"}


@dataclass
class LadderResult:
    """The best candidate found and an auditable trace of the climb (for /explain)."""

    candidate: Any
    winning_rung: int
    best_score: float
    rungs_run: List[int] = field(default_factory=list)
    scores: List[Tuple[int, float]] = field(default_factory=list)

    def trace(self) -> str:
        ran = ", ".join(_RUNG_NAMES.get(r, str(r)) for r in self.rungs_run) or "none"
        return (f"deep reasoning climbed [{ran}] and kept the "
                f"{_RUNG_NAMES.get(self.winning_rung, self.winning_rung)} answer "
                f"(verified {self.best_score:.2f})")


class DeepReasoner:
    """Always-max effort-ladder controller. Composes the real reasoning rungs, keeps verifier-best."""

    def __init__(self, reasoner: Any, *, settings: Optional[NyxaraSettings] = None,
                 verifier: Any = None, improver: Any = None,
                 effort_memory: Any = None) -> None:
        # ``reasoner`` is the LLMReasoner — it owns the LLM, the grounding context, and the
        # decision->Candidate conversion; we only orchestrate its public rung accessors.
        self.reasoner = reasoner
        self.settings = settings or get_settings()
        self.cfg = getattr(self.settings.llm, "deep_reasoning", None)
        if verifier is None:
            from nyxara.mind.router import default_verifier
            verifier = default_verifier()
        self.verifier = verifier
        self.improver = improver              # RecursiveImprover (rung 4); optional
        self.effort_memory = effort_memory    # EffortMemory; optional (enables compounding)

    # ------------------------------------------------------------------ #
    # capability probe
    # ------------------------------------------------------------------ #
    def enabled(self) -> bool:
        """Deep reasoning only runs with a real provider and the feature switched on."""
        if self.cfg is None or not bool(getattr(self.cfg, "enabled", False)):
            return False
        try:
            return bool(self.reasoner.is_real())
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------ #
    # the climb
    # ------------------------------------------------------------------ #
    def __call__(self, stimulus: str, focus: Any = None) -> Optional[Any]:
        """Return the verifier-best Candidate from the full ladder, or None to defer to the caller."""
        if not self.enabled():
            return None
        try:
            result = self.deliberate(stimulus)
        except Exception:  # noqa: BLE001 — deep reasoning is additive; never crash the turn
            return None
        return result.candidate if result is not None else None

    def deliberate(self, stimulus: str) -> Optional[LadderResult]:
        """Run the ladder and return the full :class:`LadderResult` (candidate + trace)."""
        deadline = time.monotonic() + float(getattr(self.cfg, "max_seconds", 60.0))
        max_rung = int(getattr(self.cfg, "max_rung", RUNG_REFINE))
        keep_best = bool(getattr(self.cfg, "keep_best", True))
        base_samples = max(1, int(getattr(self.cfg, "samples", 3)))

        sig = self._signature(stimulus)
        favoured = self._favoured_rung(sig)

        best_cand: Any = None
        best_score = -1.0
        best_rung = 0
        rungs_run: List[int] = []
        scores: List[Tuple[int, float]] = []

        # --- generation rungs: self-consistency → deliberation → MCTS --- #
        gen_rungs = [
            (RUNG_CONSISTENCY, dict(passes=1)),
            (RUNG_DELIBERATE, dict(passes=3)),
            (RUNG_MCTS, None),  # None marker: use the MCTS strategy
        ]
        for rung, kw in gen_rungs:
            if rung > max_rung or time.monotonic() >= deadline:
                break
            # spend the extra self-consistency width on the rung this signature has learned to
            # reward (same maximum budget, aimed where it measurably pays off).
            samples = base_samples + (base_samples if rung == favoured else 0)
            cand = self._run_generation_rung(stimulus, rung, kw, samples)
            rungs_run.append(rung)
            if cand is None:
                continue
            # a decisive tool need trumps chattier answers — respect it immediately, don't keep
            # searching for a prettier conversational reply (exact/decisive beats search).
            if getattr(cand, "kind", "respond") == "act" and getattr(cand, "tool", ""):
                self._learn(sig, rung, 1.0)
                return LadderResult(candidate=cand, winning_rung=rung, best_score=1.0,
                                    rungs_run=rungs_run, scores=scores + [(rung, 1.0)])
            score = self._score(stimulus, cand)
            scores.append((rung, score))
            if score > best_score or best_cand is None:
                best_cand, best_score, best_rung = cand, score, rung
            if not keep_best:
                best_cand, best_score, best_rung = cand, score, rung

        if best_cand is None:
            return None

        # --- refinement rung: N verified critique-improve cycles over the current best --- #
        if (RUNG_REFINE <= max_rung and self.improver is not None
                and time.monotonic() < deadline
                and getattr(best_cand, "kind", "respond") == "respond"):
            refined = self._refine(stimulus, best_cand)
            if refined is not None:
                rungs_run.append(RUNG_REFINE)
                rscore = self._score(stimulus, refined)
                scores.append((RUNG_REFINE, rscore))
                if rscore >= best_score:  # ties go to the more-refined answer
                    best_cand, best_score, best_rung = refined, rscore, RUNG_REFINE

        self._learn(sig, best_rung, best_score)
        self._annotate(best_cand, LadderResult(best_cand, best_rung, best_score, rungs_run, scores))
        return LadderResult(candidate=best_cand, winning_rung=best_rung, best_score=best_score,
                            rungs_run=rungs_run, scores=scores)

    # ------------------------------------------------------------------ #
    # rung runners (each reuses the LLMReasoner's real machinery)
    # ------------------------------------------------------------------ #
    def _run_generation_rung(self, stimulus: str, rung: int, kw: Optional[dict],
                             samples: int) -> Any:
        try:
            if rung == RUNG_MCTS:
                data = self.reasoner.mcts_decision(stimulus)
            else:
                data = self.reasoner.deliberate_decision(
                    stimulus, passes=kw.get("passes", 1), samples=samples)
            if not isinstance(data, dict):
                return None
            return self.reasoner.decision_to_candidate(data, stimulus)
        except Exception:  # noqa: BLE001 — a failed rung is simply skipped
            return None

    def _refine(self, stimulus: str, candidate: Any) -> Any:
        try:
            n = int(getattr(self.settings.llm, "recursive_improvement_iterations", 5))
            return self.improver.improve(stimulus, candidate, n_iterations=n)
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------ #
    # scoring, compounding, tracing
    # ------------------------------------------------------------------ #
    def _score(self, stimulus: str, candidate: Any) -> float:
        try:
            return float(self.verifier(stimulus, str(getattr(candidate, "text", "") or "")))
        except Exception:  # noqa: BLE001 — an unscoreable answer is treated as weakest
            return 0.0

    def _signature(self, stimulus: str) -> Optional[str]:
        if self.effort_memory is None:
            return None
        try:
            from nyxara.mind.effort_memory import signature
            return signature(stimulus)
        except Exception:  # noqa: BLE001
            return None

    def _favoured_rung(self, sig: Optional[str]) -> int:
        if sig is None or self.effort_memory is None:
            return 0
        try:
            return int(self.effort_memory.suggest(sig) or 0)
        except Exception:  # noqa: BLE001
            return 0

    def _learn(self, sig: Optional[str], rung: int, score: float) -> None:
        if sig is None or self.effort_memory is None or rung <= 0:
            return
        try:
            self.effort_memory.record(sig, rung, score)
        except Exception:  # noqa: BLE001 — compounding is best-effort, never fatal
            pass

    @staticmethod
    def _annotate(candidate: Any, result: LadderResult) -> None:
        try:
            base = str(getattr(candidate, "rationale", "") or "").strip()
            candidate.rationale = f"{base} [{result.trace()}]" if base else result.trace()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA deep-reasoning controller self-test")
    print("=" * 70)

    from dataclasses import dataclass as _dc

    @_dc
    class _Cand:
        text: str
        kind: str = "respond"
        tool: str = ""
        confidence: float = 0.7
        rationale: str = ""

    class _FakeReasoner:
        """A scripted LLMReasoner: each rung returns a progressively better answer."""

        def __init__(self) -> None:
            self.calls = []

        def is_real(self) -> bool:
            return True

        def decision_to_candidate(self, data, stimulus):
            return _Cand(text=data["text"], kind=data.get("kind", "respond"),
                         tool=data.get("tool", ""), confidence=data.get("confidence", 0.7))

        def deliberate_decision(self, stimulus, *, passes=1, samples=1, temperature=None):
            self.calls.append(("deliberate", passes, samples))
            # deeper deliberation (passes=3) yields a richer, less repetitive answer
            if passes >= 3:
                return {"text": "The sky is blue because shorter wavelengths scatter more "
                                "strongly off air molecules — Rayleigh scattering.",
                        "kind": "respond", "confidence": 0.8}
            return {"text": "blue blue blue sky sky", "kind": "respond", "confidence": 0.6}

        def mcts_decision(self, stimulus, *, temperature=None):
            self.calls.append(("mcts", 0, 0))
            return {"text": "Sunlight is scattered by the atmosphere; blue light scatters most, "
                            "so the daytime sky looks blue to us from every direction.",
                    "kind": "respond", "confidence": 0.85}

    class _Improver:
        def improve(self, stimulus, candidate, n_iterations=5):
            candidate.rationale = (candidate.rationale + " refined").strip()
            return candidate

    from nyxara.kernel.config import NyxaraSettings, Profile
    from nyxara.mind.effort_memory import EffortMemory, signature

    settings = NyxaraSettings.for_profile(Profile.TEST)
    # force the feature on for the test regardless of provider default
    settings.llm.deep_reasoning.enabled = True

    em = EffortMemory(min_observations=2.0, success_floor=0.4)
    fake = _FakeReasoner()
    dr = DeepReasoner(fake, settings=settings, improver=_Improver(), effort_memory=em)

    q = "why is the sky blue?"
    res = dr.deliberate(q)
    assert res is not None
    print(f"\nrungs run   : {res.rungs_run}")
    print(f"winning rung: {res.winning_rung} ({res.best_score:.2f})")
    print(f"answer      : {res.candidate.text[:70]}...")
    print(f"trace       : {res.candidate.rationale}")
    # it climbed all generation rungs and kept a strong, non-degenerate answer (not the L1 word-salad)
    assert RUNG_CONSISTENCY in res.rungs_run and RUNG_MCTS in res.rungs_run
    assert "blue blue blue" not in res.candidate.text
    assert res.best_score > 0.3

    # a decisive tool need short-circuits the climb
    class _ActReasoner(_FakeReasoner):
        def deliberate_decision(self, stimulus, *, passes=1, samples=1, temperature=None):
            return {"text": "read the file", "kind": "act", "tool": "read_file",
                    "confidence": 0.9}
    dr2 = DeepReasoner(_ActReasoner(), settings=settings, effort_memory=em)
    act = dr2.deliberate("read notes.txt")
    assert act.candidate.kind == "act" and act.winning_rung == RUNG_CONSISTENCY
    print(f"\naction turn : short-circuited at rung {act.winning_rung} "
          f"(tool={act.candidate.tool}) ✓")

    # compounding: after several turns on this signature, the winning rung is learned
    for _ in range(4):
        dr.deliberate(q)
    print(f"learned rung for {signature(q)!r}: {em.suggest(signature(q))}")

    # disabled -> returns None so the caller keeps its normal path
    settings.llm.deep_reasoning.enabled = False
    assert DeepReasoner(fake, settings=settings)("anything") is None
    print("disabled    : defers to caller (returns None) ✓")

    print("\nALL SELF-TESTS PASSED ✓")
