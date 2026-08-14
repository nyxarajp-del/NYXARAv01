"""NYXARA · njp/cell.py — one cell of the Fluid Neural Automata (🧬, NJP V.01).

A cell is the smallest thing in NJP that has state. It is deliberately tiny: a membrane
potential, a firing threshold, a bias, and the tick it is refractory until. Everything the
fabric does — settling, potentiation, synaptogenesis, neurogenesis — is expressed in terms of
these five numbers and nothing else.

There is **no per-cell behaviour written here**. A cell does not know what it represents, does
not know what the turn was about, and has no rule of its own. The rule lives in
:meth:`nyxara.njp.fabric.Fabric.step` and is the *same* rule for every cell — which is what makes
this an automaton rather than a hand-wired network. Structure is the only thing that
differentiates one cell from another, and structure is grown, never declared.

Honest framing, the rule this repo keeps (see ``docs/CAPABILITIES.md``):

* This is a **leaky integrate-and-fire automaton on a graph**, simulated on commodity silicon.
  It is not neuromorphic hardware. **This cell's** own updates carry no gradient: a synapse
  changes from its two endpoints alone. Gradient learning lives in :mod:`nyxara.njp.learn`, over
  the fabric rather than inside it.
* ``__slots__`` is not a micro-optimisation here. The fabric grows without a fixed ceiling, so
  a live NJP holds hundreds of thousands of these; a dict-per-cell would be most of the resident
  memory. This is the difference between growth that continues and growth that hits swap.
"""

from __future__ import annotations

from typing import Any, Dict

__all__ = ["Cell"]


class Cell:
    """One leaky integrate-and-fire cell. Five numbers, no behaviour of its own."""

    __slots__ = ("state", "threshold", "bias", "refractory_until",
                 "born_tick", "last_fired", "hits")

    def __init__(self, *, threshold: float = 1.0, bias: float = 0.0,
                 born_tick: int = 0) -> None:
        self.state = 0.0                  # membrane potential
        self.threshold = float(threshold)  # fires at or above this
        self.bias = float(bias)           # constant drive, learned
        self.refractory_until = -1        # cannot fire while tick <= this
        self.born_tick = int(born_tick)   # when neurogenesis minted it
        self.last_fired = -1              # last tick this cell fired
        self.hits = 0                     # lifetime firings — apoptosis reads this

    def to_dict(self) -> Dict[str, Any]:
        """Persisted state. ``state``/``refractory_until`` are deliberately NOT saved.

        Membrane potential is *within-turn* scratch: it is meaningless across a restart, and
        writing it would make a reloaded fabric behave differently from a settled one. What
        survives is what was **learned** — threshold, bias, and the cell's own history.
        """
        return {"t": round(self.threshold, 6), "b": round(self.bias, 6),
                "born": self.born_tick, "last": self.last_fired, "hits": self.hits}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Cell":
        cell = cls(threshold=float(d.get("t", 1.0)), bias=float(d.get("b", 0.0)),
                   born_tick=int(d.get("born", 0)))
        cell.last_fired = int(d.get("last", -1))
        cell.hits = int(d.get("hits", 0))
        return cell
