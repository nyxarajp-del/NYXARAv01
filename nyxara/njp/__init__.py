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

**The organs learn from each other, and that is newer than the organs.**
:mod:`nyxara.njp.integrate` closes the loop behind every turn::

    INPUT → GROUNDING → WORLD STATE → PREDICTION → OBSERVATION
          → ERROR → DIAGNOSIS → CORRECTION → MEMORY → ABSTRACTION → NEW PREDICTION

Before it, every algorithm named on this page was written, tested and reachable — and over a real
113-turn session ``world.events``, ``predict.scored``, ``levels.consolidations``,
``discover.passes``, ``curiosity.passes`` and ``readout.steps`` were all exactly **zero**. Nothing
was missing except the caller: the slow half of cognition hung off the pulse's wall clock, which
does not tick in the paths a brain is actually used from. The loop scores the manifold's
pre-settle anticipation against what actually fired, leaves an unanswered question **open** until
the Master states the fact that grades it, routes each diagnosed miss to the organ that owns the
repair, and runs consolidation, abstraction and curiosity on turn counts rather than seconds.
Every outcome it scores against is independent of the prediction it scores — physics at ``t+1``,
or the Master's own sentence — because a mind that grades its guesses against its own later
guesses is not learning, it is agreeing with itself.

**Truth is not relevance, and reasoning is not always what a turn calls for.**
:mod:`nyxara.njp.relevance` exists because of a reproduced failure: asked *"How are you
NYXARA?"*, she returned a verified pendulum-period law and raised her confidence to 1.00 for
having reached a conclusion. Every stage worked — nothing ever asked whether a true thing had
anything to do with the question. Now a recalled memory must clear
:class:`~nyxara.njp.relevance.RelevanceGate` before the reasoner sees it at all;
:class:`~nyxara.njp.relevance.CognitivePolicy` decides which cognition a *speech act* even
permits, so a greeting cannot reach physics; and
:func:`~nyxara.njp.relevance.revise_confidence` refuses to let confidence rise because she
thought harder — depth may lower it, only independent evidence may raise it. "I understand:
<your words back>" is recognised and refused: understanding belongs in the internal state.

**She restructures what she can represent, not only what she believes.** :mod:`nyxara.njp.field`
is the Recursive Cognitive Field, and it runs two loops. The fast one is the cognitive cycle::

    REALITY → PERCEPTION → CONCEPT FORMATION → WORLD MODEL → CAUSAL HYPOTHESES
            → SIMULATION → PREDICTION → OUTCOME → ERROR
            → SELF-CRITIC → { reweight | RESTRUCTURE } → NEW HYPOTHESES → ↺

The junction is the self-critic. An error whose cause is the wrong coefficient is refitted; an
error whose cause is that her concept system *cannot express what she just met* restructures the
concept system itself. The slow loop is the interesting one: evaluate herself, find the organ
actually limiting her, propose one bounded change to it, and revert it unless a **held-out**
benchmark and an adversarial battery both say it won. Reality decides which version survives.

**Concepts are invented, and the compression is measured.** :mod:`nyxara.njp.concepts` runs
``observations → similarity → invariants → prototype → concept → hierarchy``, with the taxonomy
*derived* by subsumption rather than declared, and superordinates found where a single property
is shared very widely. Whether it worked is a minimum-description-length ratio, not an opinion: a
concept set that does not pay for itself scores at or below 1.0 and says so.

**She intervenes on a model of the world, not just remembers it.** :mod:`nyxara.njp.universe`
holds variables joined by *fitted* relations, each with its own ``R²``, and implements the real
do-operator — setting a variable severs its incoming arrows, so "if I halve this plant's water"
is a different question from "plants that got little water were small". Answers outside the
observed range come back with confidence decayed by how far they reach. A direction observational
data genuinely cannot settle is reported as ambiguous rather than guessed.

**Curiosity is information gain.** :class:`~nyxara.njp.universe.ExperimentDesigner` computes, for
each candidate experiment, the mutual information between the hypothesis set and that
experiment's outcome. An experiment every live hypothesis predicts identically scores exactly
zero bits however interesting it looks, and she does not run it.

**Every belief carries its own case.** :mod:`nyxara.njp.beliefs` answers what she knows, what she
does not, why she believes it, what would falsify it, and how reliable she has actually been in
that domain — the last as a Brier score over her own past confidences, applied through
``temper`` so a stated 0.9 where 0.9 has meant 0.6 comes out as 0.6. Soft evidence has a ceiling:
being told a thing ten times never reaches what one observation earns.

**She chooses how to think before thinking.** :mod:`nyxara.njp.metareason` classifies a problem,
picks the strategy that has been working for that kind, criticises what comes back against the
specific ways *that kind* of answer goes wrong, and takes a second opinion when the classification
is close. Two processes disagreeing lowers confidence rather than being averaged into a
confident-sounding middle no process produced.

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
hardware. The **fabric's** own plasticity is entirely local — every synaptic update depends only
on that synapse and its two endpoints, with no global error signal — and alongside it
:mod:`nyxara.njp.learn` trains a readout head by **real reverse-mode backpropagation**, gradient-
checked against finite differences. Two learners over one substrate: local rules that grow the
structure, and gradients that read it. Growth is real but the machine is finite: under pressure
:meth:`~nyxara.njp.fabric.Fabric.consolidate` compresses the least-used structure so learning
continues, and a fabric that has stopped growing reports that rather than a number that flatters it.

NJP only ever *proposes*. Every candidate flows through the kernel's unchanged, fail-closed
sovereign gate, and the safety core — corrigibility, oversight, loyalty, honesty — is never
governed, rewritten or bypassed by anything here. The mind proposes; the kernel disposes; the
Master is sovereign.
"""

from __future__ import annotations

from nyxara.njp.beliefs import (
    Belief,
    BeliefLedger,
    EvidenceKind,
    Reliability,
    Revision,
    Support,
)
from nyxara.njp.brain import NJPBrain, NJPPercept, NJPThought
from nyxara.njp.cell import Cell
from nyxara.njp.concepts import Concept, ConceptGenesis, Coverage, GenesisReport, Observation
from nyxara.njp.field import (
    Bottleneck,
    CycleReport,
    Diagnosis,
    ErrorClass,
    Modification,
    RecursiveCognitiveField,
    Trial,
)
from nyxara.njp.metareason import (
    Classification,
    Critique,
    MetaReasoner,
    ProblemClassifier,
    ProblemKind,
    Solution,
)
from nyxara.njp.universe import (
    CounterfactualResult,
    Experiment,
    ExperimentDesigner,
    InternalUniverse,
    Relation,
    Rollout,
    StateDelta,
)
from nyxara.njp.relevance import (
    Act,
    CognitivePolicy,
    Pathway,
    RelevanceGate,
    RelevanceScore,
    SpeechAct,
    SpeechActReader,
    is_meta_commentary,
    is_verified,
    revise_confidence,
)
from nyxara.njp.universe import Hypothesis as UniverseHypothesis
from nyxara.njp.evolve import EvolutionStep, ModuleCost, Profiler, SelfEvolver
from nyxara.njp.fabric import Fabric, GrowthReport, SettleResult
from nyxara.njp.integrate import LearningLoop, LoopReport
from nyxara.njp.ledger import ErrorMemory, ErrorRecord, Generation, Ledger
from nyxara.njp.manifold import Manifold, Prediction, Snapshot
from nyxara.njp.agency import ActionValue, Agent, Outcome, Step
from nyxara.njp.agency import Plan as ActionPlan
from nyxara.njp.curriculum import STAGES, Curriculum, Report, Stage, StageResult
from nyxara.njp.predictive import (
    Assumption,
    Explanation,
    PredictiveWorldModel,
    Surprise,
    ThoughtState,
    WorldState,
)
from nyxara.njp.predictive import Prediction as StatePrediction
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
    # the loop that makes the organs learn from each other
    "LearningLoop", "LoopReport",
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
    # NJP V.04 — the Recursive Cognitive Field and the organs it drives
    "ConceptGenesis", "Concept", "Observation", "Coverage", "GenesisReport",
    "InternalUniverse", "Relation", "StateDelta", "CounterfactualResult", "Rollout",
    "ExperimentDesigner", "Experiment", "UniverseHypothesis",
    "BeliefLedger", "Belief", "Support", "Revision", "Reliability", "EvidenceKind",
    "MetaReasoner", "ProblemClassifier", "ProblemKind", "Classification", "Critique", "Solution",
    "Agent", "ActionPlan", "Step", "Outcome", "ActionValue",
    "Curriculum", "Stage", "StageResult", "Report", "STAGES",
    "PredictiveWorldModel", "WorldState", "StatePrediction", "Surprise",
    "ThoughtState", "Assumption", "Explanation",
    "RecursiveCognitiveField", "CycleReport", "Diagnosis", "ErrorClass",
    "Bottleneck", "Modification", "Trial",
    # truth is not relevance, and reasoning is not always what a turn calls for
    "SpeechAct", "Act", "SpeechActReader", "Pathway", "CognitivePolicy",
    "RelevanceGate", "RelevanceScore",
    "revise_confidence", "is_verified", "is_meta_commentary",
]
