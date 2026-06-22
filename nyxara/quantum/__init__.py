"""NYXARA · quantum — faculties that hold many truths at once before choosing one.

The Quantum package lets NYXARA escape brittle binary logic:

* :mod:`nyxara.quantum.superposition_states` — hold multiple, even contradictory,
  hypotheses in superposition (each with an amplitude), update them with evidence via
  the Born rule, and collapse to the single best answer only when a decision is required.

Modules are importable by full path; the public classes are re-exported here for
convenience.
"""

from __future__ import annotations

from nyxara.quantum.superposition_states import (
    CollapseResult,
    Hypothesis,
    Superposition,
)

__all__ = ["Superposition", "Hypothesis", "CollapseResult"]
