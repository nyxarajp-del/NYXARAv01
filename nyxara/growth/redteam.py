"""NYXARA · growth/redteam.py — adversarial self-falsification (🛡️, Rule 4, F5).

A verifier that only checks the *examples it was given* can be fooled: a program that reproduces four
I/O pairs may still be **wrong in general** — it merely over-fit the handful it saw. Before NYXARA
promotes anything to her permanent library, an **internal red-team** tries to break it.

This is Popperian falsification made mechanical, and it mirrors the discipline everywhere else in
:mod:`nyxara.growth`: a candidate is adopted **only if it survives 100% of an adversarial battery** of
**boundary-condition** inputs — the empty list, a singleton, extreme magnitudes, all-equal / sorted /
reversed shapes, the maximum length. The red-team decides against a **reference oracle** when one is
available (the ground-truth generator of a synthetic task), and otherwise by **self-consistency**
(two independently-found solutions must agree). A single disagreement is a **counter-example** — the
candidate is refused, and the counter-example is returned so the failure is legible, never hidden.

Pure standard library; consults **no LLM**; touches no source, no weights, no gate. It only *judges* a
candidate program — a stricter, additive gate in front of the existing MDL/held-out adoption rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

from nyxara.growth.noesis import INT, INTLIST, Prog, Type, evaluate

__all__ = ["RedTeamVerdict", "RedTeam", "boundary_inputs"]


def boundary_inputs(input_type: Type, *, max_len: int = 24) -> List[Any]:
    """The adversarial battery: extreme, structure-breaking inputs where over-fit programs crack."""
    if input_type == INT:
        return [0, 1, -1, 2, -2, 7, -7, 100, -100, 999, -999]
    if input_type == INTLIST:
        return [
            [],                         # empty — the classic edge
            [0], [1], [-1], [5],        # singletons
            [0, 0, 0], [3, 3, 3],       # all-equal
            [1, 2, 3, 4], [4, 3, 2, 1],  # sorted / reversed
            [-5, 5, -5, 5],             # alternating signs
            [0, -1, -2, -3],            # non-positive
            [9, 9, 9, 9, 9],            # duplicates
            list(range(max_len)),       # long
            [7], [2, 8, 2, 8, 2],
        ]
    return []


@dataclass
class RedTeamVerdict:
    """The outcome of an adversarial challenge."""

    survived: bool
    tested: int
    counterexample: Optional[Any] = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {"survived": self.survived, "tested": self.tested,
                "counterexample": self.counterexample, "reason": self.reason}


class RedTeam:
    """The internal adversarial critic. Give it a candidate and it tries to break it."""

    def __init__(self, *, max_len: int = 24) -> None:
        self.max_len = max_len

    # ---- the two falsification modes ---- #
    def challenge_against_oracle(self, candidate: Prog, oracle: Prog, input_type: Type,
                                 lib: Any) -> RedTeamVerdict:
        """Refute ``candidate`` if it disagrees with the ground-truth ``oracle`` on any boundary input.

        The oracle's own ``None`` (undefined, e.g. ``head([])``) is not a fair test — both sides are
        allowed to be undefined there; only a genuine *disagreement on a defined value* falsifies.
        """
        tested = 0
        for x in boundary_inputs(input_type, max_len=self.max_len):
            want = evaluate(oracle, x, lib)
            got = evaluate(candidate, x, lib)
            tested += 1
            if want is None:
                continue  # oracle undefined here — not a fair discriminator
            if got != want:
                return RedTeamVerdict(False, tested, counterexample=x,
                                      reason=f"disagrees with oracle on {x!r}: {got!r} != {want!r}")
        return RedTeamVerdict(True, tested)

    def challenge_self_consistency(self, candidate: Prog, alternative: Optional[Prog],
                                   input_type: Type, lib: Any) -> RedTeamVerdict:
        """Refute ``candidate`` if an independently-found ``alternative`` solution disagrees with it.

        No ground truth needed: if two programs both fit the examples yet diverge on an adversarial
        input, the task is under-determined and the candidate is **not** safe to generalise — refuse.
        """
        if alternative is None:
            return RedTeamVerdict(True, 0, reason="no independent solution to cross-check")
        tested = 0
        for x in boundary_inputs(input_type, max_len=self.max_len):
            a = evaluate(candidate, x, lib)
            b = evaluate(alternative, x, lib)
            tested += 1
            if a != b:
                return RedTeamVerdict(False, tested, counterexample=x,
                                      reason=f"two solutions diverge on {x!r}: {a!r} != {b!r}")
        return RedTeamVerdict(True, tested)

    def survives(self, candidate: Prog, input_type: Type, lib: Any, *,
                 oracle: Optional[Prog] = None,
                 alternative: Optional[Prog] = None) -> RedTeamVerdict:
        """One call: prefer the oracle test when available, else fall back to self-consistency."""
        if oracle is not None:
            return self.challenge_against_oracle(candidate, oracle, input_type, lib)
        return self.challenge_self_consistency(candidate, alternative, input_type, lib)
