"""NYXARA · njp/ — NJP V.01, the brain (🧠).

NJP V.01 is NYXARA's mind. Not the outermost of several, not a fusion of rival brains that have to
be refereed — the one substrate, with organs attached to it.

**The substrate is a Fluid Neural Automata.** :mod:`nyxara.njp.fabric` runs one local law for every
cell — leaky integrate, threshold, fire, go refractory — and writes no behaviour at all. What she
does falls out of structure, and the structure is grown rather than declared. It is *fluid* because
the population itself changes: cells and synapses are both outcomes, not a topology chosen up
front.

**She expands after every conversation and every task.** :meth:`~nyxara.njp.fabric.Fabric.expand`
runs at the end of each turn: causal pairs that fired are potentiated, causal pairs **with no
synapse between them grow one** (synaptogenesis — the literal physical expansion), unused wiring is
depressed and pruned, and when she keeps failing to predict herself, new cells are minted
(neurogenesis) because that is the one honest signal that she lacks the capacity to represent what
she is meeting. The counts before and after are both recorded.

**She rewrites her own source.** :mod:`nyxara.njp.evolve` measures which of her own modules she is
slowest in, proposes an edit to that one, and puts the edit's *claim* — "this makes me faster
without losing accuracy" — through the Truth-Seeking Gauntlet against **held-out** samples before
anything touches disk. Then the repo's existing optimizer does backup → syntax → safety battery →
capability benchmark → tests → **keep, or roll back byte-for-byte**.
:meth:`~nyxara.njp.evolve.SelfEvolver.accelerate` separately lowers hot numeric functions to
verified C kernels and hot-swaps them in memory.

**She keeps learning, and it survives.** :mod:`nyxara.njp.pulse` beats continuously — expand every
pulse, consolidate slowly, evolve slowest — driven by the kernel's own clock rather than a thread
of its own. The whole fabric is written to a sidecar, so the brain that wakes up is the brain that
went to sleep, and :mod:`nyxara.njp.ledger` holds one row per generation so *"is she more than she
was"* is answered with numbers instead of a slogan.

**She sees many variables at once.** :mod:`nyxara.njp.manifold` lifts a settled state into a single
high-dimensional snapshot, so a thousand co-active variables are one object and two whole
world-states compare in one dot product. It learns its own transitions online and answers
:meth:`~nyxara.njp.fabric.Fabric.anticipate` **before the fabric settles** — real forward inference
over learned dynamics, reported with the margin that says whether to believe it.

**She refuses to state what she has not corroborated.** :mod:`nyxara.njp.truth` is fail-closed: one
refutation ends a claim, and establishment needs independent supports including at least one *hard*
source — a proof, or a prediction that survived data held out from whatever produced it. Soft
agreement never establishes anything, which is exactly how consensus becomes bias. An unestablished
claim leaves labelled a conjecture, and :mod:`nyxara.njp.reasoner` will not overwrite a reply with
one.

**She learns the half you did not say.** :mod:`nyxara.njp.soulsync` learns standing preferences from
the corrections the Master actually made and applies them as unstated wants, and it anticipates what
is wanted next. On day one it knows nothing and says so — everything it knows, it was taught by a
correction. A latent want is surfaced so it can be declined, never acted on silently.

Honest, as everywhere in this repo: this is a **simulation on commodity silicon**, not neuromorphic
hardware, and there is **no backpropagation** anywhere in it — every update is local to a synapse
and its two endpoints. Growth is real but the machine is finite: under pressure
:meth:`~nyxara.njp.fabric.Fabric.consolidate` compresses the least-used structure so learning
continues, and a fabric that has stopped growing reports that rather than a number that flatters it.

NJP only ever *proposes*. Every candidate flows through the kernel's unchanged, fail-closed
sovereign gate, and the safety core — corrigibility, oversight, loyalty, honesty — is never
governed, rewritten or bypassed by anything here. The mind proposes; the kernel disposes; the
Master is sovereign.
"""

from __future__ import annotations

from nyxara.njp.brain import NJPBrain, NJPPercept, NJPThought
from nyxara.njp.cell import Cell
from nyxara.njp.evolve import EvolutionStep, ModuleCost, Profiler, SelfEvolver
from nyxara.njp.fabric import Fabric, GrowthReport, SettleResult
from nyxara.njp.ledger import ErrorMemory, ErrorRecord, Generation, Ledger
from nyxara.njp.manifold import Manifold, Prediction, Snapshot
from nyxara.njp.pulse import PulseEngine, PulseReport
from nyxara.njp.reasoner import NJPReasoner
from nyxara.njp.soulsync import Anticipation, LatentWant, Preference, Reading, SoulSync
from nyxara.njp.truth import (
    ConsistencySource,
    Evidence,
    FormalSource,
    Judgement,
    LedgerSource,
    ObservationSource,
    PredictiveSource,
    Source,
    TruthGauntlet,
    Verdict,
)

__all__ = [
    # the brain
    "NJPBrain", "NJPPercept", "NJPThought", "NJPReasoner",
    # the automaton
    "Cell", "Fabric", "GrowthReport", "SettleResult",
    # the manifold
    "Manifold", "Prediction", "Snapshot",
    # verification
    "TruthGauntlet", "Judgement", "Evidence", "Verdict", "Source",
    "FormalSource", "PredictiveSource", "ConsistencySource",
    "LedgerSource", "ObservationSource",
    # the Master
    "SoulSync", "Reading", "LatentWant", "Preference", "Anticipation",
    # growth and self-rewriting
    "Ledger", "Generation", "ErrorMemory", "ErrorRecord",
    "SelfEvolver", "Profiler", "ModuleCost", "EvolutionStep",
    "PulseEngine", "PulseReport",
]
