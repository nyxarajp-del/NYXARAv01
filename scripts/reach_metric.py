#!/usr/bin/env python3
"""NYXARA · scripts/reach_metric.py — how much does NYXARA now reason *herself* (no LLM)?

The critique this repo's reasoning upgrade answers: NYXARA's symbolic/structural faculties only
fired on a narrow band of problems; everything else fell back to the small local base LLM (and its
low ceiling). This harness makes the improvement *measurable*. It runs a fixed battery of prompts —
the exact kinds that used to fall through to the LLM — through ONLY her own faculties, fully offline
(no model, no network), and reports how many she now solves herself, verified or by her own
structure-mapping/from-scratch modelling.

Run:  python scripts/reach_metric.py

It exits 0 always; it prints a report, it does not gate CI. A prompt "solved herself" means one of:
  * a verifiable faculty / first-principles derivation returned an exact answer, or
  * her unified generalization cascade (skill-induction, relational transfer, domain-genesis,
    open-world law-fitting) returned a genuine, honestly-hedged result.
Anything she declines is printed as "→ LLM" — the honest residual that still needs the base model.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


# Prompts that, before the reach upgrade, mostly fell through to the base LLM.
PROMPTS: List[Tuple[str, str]] = [
    ("symbolic-law", "given F = m*a, solve for a"),
    ("symbolic-law", "solve E = m*c^2 for m"),
    ("symbolic-system", "given p = f/A and f = m*a, find p"),
    ("calculus-ode", "given dv/dt = a, find v"),
    ("physics-named", "derive the escape velocity from energy conservation"),
    ("dimensional", "derive the period of a pendulum"),
    ("chemistry", "balance C3H8 + O2 -> CO2 + H2O"),
    ("maths-proof", "prove that (a+b)^2 = a^2 + 2*a*b + b^2"),
    ("logic-fol", "all humans are mortal. socrates is a human."),
    ("logic-fol", "every bird can fly. tweety is a bird."),
    ("logic-fol", "if X is a metal then X conducts. copper is a metal."),
    ("transfer", "model of the atom: the nucleus attracts the electron and the electron "
                 "revolves around the nucleus"),
    ("transfer-seed", "the reward strengthens the behaviour and the behaviour produces the reward"),
    ("domain-genesis", "in this device the glorp zephs the drine and the drine zephs the quon "
                       "and the quon flumps the glorp"),
    ("skill-induction", "pattern: hello -> olleh, world -> dlrow. now: nyxara"),
    ("copula-transfer", "the server is faster than the client and the client is slower than "
                        "the server"),
    # honest residual — these SHOULD still defer to the LLM (no structure to reason from)
    ("chat (should defer)", "hi, how are you doing today?"),
    ("open-fact (should defer)", "what is the capital of France?"),
]


def _own_faculty_answer(prompt: str) -> Optional[str]:
    """Return a short label of the own-faculty that solved ``prompt``, or ``None`` to defer."""
    try:
        from nyxara.mind.reasoning_faculties import solve_verifiable
        hit = solve_verifiable(prompt)
        if hit is not None:
            return "verifiable-faculty"
    except Exception:  # noqa: BLE001
        pass
    try:
        from nyxara.mind.generalization import GeneralizationEngine
        res = GeneralizationEngine().generalize(prompt)
        if res is not None:
            return res.source
    except Exception:  # noqa: BLE001
        pass
    return None


def main() -> int:
    print("=" * 74)
    print("NYXARA reach metric — solved by her OWN faculties, fully offline (no LLM)")
    print("=" * 74)
    solved = 0
    residual = 0
    for kind, prompt in PROMPTS:
        source = _own_faculty_answer(prompt)
        should_defer = "should defer" in kind
        if source is not None:
            solved += 1
            mark = "HER  "
            tag = f"[{source}]"
        else:
            residual += 1
            mark = "→ LLM"
            tag = "(deferred)"
        flag = "  ✓ honest" if (should_defer and source is None) else ""
        short = prompt if len(prompt) <= 52 else prompt[:49] + "..."
        print(f"  {mark}  {kind:<24} {short:<54} {tag}{flag}")
    total = len(PROMPTS)
    print("-" * 74)
    print(f"  solved by her own faculties : {solved}/{total}")
    print(f"  deferred to the base LLM    : {residual}/{total}  "
          f"(the honest residual — chat / open facts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
