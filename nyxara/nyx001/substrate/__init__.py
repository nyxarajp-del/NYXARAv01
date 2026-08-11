"""NYXARA · nyx001/substrate/ — Layer 0, the primitive computational substrate (🌱).

The charter's Layer 0 is "the seed: no human knowledge, only basic computation, random state, and
learning dynamics". This package is exactly those four things and nothing more:

* :mod:`~nyxara.nyx001.substrate.autograd` — basic computation. A reverse-mode autograd over
  NumPy, re-exported from :mod:`nyxara.growth.genesis_numpy` (which already had a correct one)
  plus the ops the eighteen layers needed and it lacked.
* :mod:`~nyxara.nyx001.substrate.init` — random state. Every parameter NYX V.001 owns is born
  from a seeded generator here, and from no checkpoint anywhere.
* :mod:`~nyxara.nyx001.substrate.optim` — learning dynamics. RMSprop under a bounded cognitive
  gain, ``Δθ = F(E, N, U, M, T)``.
* :mod:`~nyxara.nyx001.substrate.objective` — what "better" means. A five-term composite loss
  whose terms stay separately measurable.

No human knowledge enters here. There is no vocabulary, no rule, no fact, no if-X-then-Y — only
arithmetic, randomness, and a definition of improvement. Everything NYX V.001 knows, she has to
get from experience through these four files.
"""

from __future__ import annotations

from nyxara.nyx001.substrate.autograd import HAS_NUMPY, Tensor, backward, constant, parameter
from nyxara.nyx001.substrate.init import (
    count_params,
    he,
    ones,
    orthogonal,
    rng_for,
    uniform_gate,
    xavier,
    zeros,
)
from nyxara.nyx001.substrate.objective import CompositeObjective, Loss
from nyxara.nyx001.substrate.optim import AdaptiveOptimizer, CognitiveState, StepReport

__all__ = [
    "HAS_NUMPY", "Tensor", "backward", "constant", "parameter",
    "rng_for", "xavier", "he", "orthogonal", "uniform_gate", "zeros", "ones", "count_params",
    "AdaptiveOptimizer", "CognitiveState", "StepReport",
    "CompositeObjective", "Loss",
]
