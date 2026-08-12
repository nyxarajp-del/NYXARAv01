"""NYXARA · nyx5/reasoner.py — the selectable NYX-5 reason-seat (⚡, the pluggable brain).

`Nyx5Reasoner` is a ``Reasoner``-compatible callable — ``(stimulus, focus) -> Candidate`` — that can be
installed in the kernel's reason-seat alongside the classic NyxaraReasoner (config
``NYXARA_NYX5__AS_REASONER=true``). It **wraps a base reasoner** for the linguistic surface (a spiking
network is not an honest way to emit English) and drives it with NYX-5 signals: it perceives the stimulus
through the spiking substrate (learning by STDP on the same turn), appraises surprise / free energy, and
enriches the base candidate's confidence, belief, risk-caution and rationale.

Sovereignty is preserved absolutely: NYX-5 only ever *proposes*. The returned candidate flows through the
kernel's unchanged, fail-closed gate exactly like any other. NYX-5 never lowers risk, never weakens a
safety flag, and the sovereign gate — not this reasoner — decides what acts. Fail-soft: on any error it
returns the base candidate untouched, so the mind always works.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["Nyx5Reasoner"]


class Nyx5Reasoner:
    """A neuromorphic decision layer over a base reasoner — proposes, never disposes."""

    def __init__(self, *, base: Callable[..., Any], brain: Any, free_energy: Any = None,
                 predictive: Any = None) -> None:
        self.base = base
        self.brain = brain
        self.free_energy = free_energy
        self.predictive = predictive

    def __call__(self, stimulus: str, focus: Any = None, **kwargs: Any) -> Any:
        tick = None
        try:
            tick = self.brain.perceive(stimulus)
        except Exception:  # noqa: BLE001 — perception is advisory; never block generation
            tick = None

        # the base reasoner produces the actual candidate (text + kind + capability + risk); pass
        # through whatever extra kwargs the kernel handed us (e.g. memories) so the base is unimpaired.
        candidate = self._call_base(stimulus, focus, kwargs)

        try:
            self._enrich(candidate, stimulus, tick)
        except Exception:  # noqa: BLE001 — enrichment is colour; a failure leaves the base candidate
            pass
        return candidate

    def _call_base(self, stimulus: str, focus: Any, kwargs: dict) -> Any:
        try:
            return self.base(stimulus, focus, **kwargs)
        except TypeError:
            try:
                return self.base(stimulus, focus)   # a two-arg reasoner (the deterministic stand-in)
            except TypeError:
                return self.base(stimulus)          # a one-arg reasoner

    def _enrich(self, candidate: Any, stimulus: str, tick: Any) -> None:
        if candidate is None or tick is None:
            return
        # 1) confidence/belief calibration from spike consensus (bounded, only nudges).
        conf = getattr(candidate, "confidence", None)
        if isinstance(conf, (int, float)):
            # high surprise -> less confident; strong firing -> a touch more.
            adj = 1.0 - 0.3 * tick.surprise + 0.1 * min(1.0, tick.firing_rate)
            candidate.confidence = max(0.0, min(1.0, float(conf) * max(0.5, adj)))
        belief = getattr(candidate, "belief", None)
        if isinstance(belief, (int, float)):
            candidate.belief = max(0.0, min(1.0, float(belief) * (1.0 - 0.2 * tick.surprise)))

        # 2) expected-free-energy annotation (surprise is the honest EFE proxy). Never lowers risk.
        if hasattr(candidate, "efe"):
            try:
                candidate.efe = float(tick.surprise)
            except Exception:  # noqa: BLE001
                pass

        # 3) rationale colour — record that NYX-5 weighed in, honestly and briefly.
        rationale = getattr(candidate, "rationale", None)
        note = f"nyx5: surprise={tick.surprise:.2f} entropy={tick.entropy:.2f}"
        if tick.preemptive and tick.suggestion is not None:
            note += f" · pre-emptive probe {tick.suggestion.probe}"
        if isinstance(rationale, str) and rationale:
            candidate.rationale = f"{rationale} | {note}"
        elif hasattr(candidate, "rationale"):
            candidate.rationale = note

        # 4) advisory dialectic — attach a sharper-path suggestion (never a veto).
        dial = getattr(self.brain, "dialectic", None)
        if dial is not None and isinstance(getattr(candidate, "rationale", None), str):
            crit = dial.critique(stimulus)
            if crit.has_suggestion():
                candidate.rationale += " | dialectic: " + "; ".join(crit.suggestions[:2])
