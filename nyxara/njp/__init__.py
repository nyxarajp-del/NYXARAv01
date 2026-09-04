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

**And she plans in it.** :mod:`nyxara.njp.rollout` takes a variable that has to reach a value,
finds what she has an oriented arrow *from*, sweeps each lever, rolls every setting forward and
ranks them on the share of the distance closed times the confidence the model itself claimed —
so a plan that would arrive by extrapolating far past anything ever measured loses to one that
arrives on evidence, and the smallest move that works wins among equals. The plan is then
registered as a claim, and the next real reading of that variable grades it. The number the
mechanism is judged on is ``settled``, never ``planned``.

**Curiosity is information gain.** :class:`~nyxara.njp.universe.ExperimentDesigner` computes, for
each candidate experiment, the mutual information between the hypothesis set and that
experiment's outcome. An experiment every live hypothesis predicts identically scores exactly
zero bits however interesting it looks, and she does not run it.

**A model she keeps being surprised by can be given up, and giving it up is reversible.**
:mod:`nyxara.njp.universe` counts surprises *per model version*, and retirement withdraws only the
directions she inferred **for herself** — testimony and intervention came from outside the model
under suspicion, and discarding them would be throwing away evidence to save a hypothesis. The
whole trial is judged on a held-out fold and reverts on a tie. That mechanism and its caller both
existed and the loop had never once run, because its own criterion was circular: it demanded a
pair with neither direction settled, and withdrawing an inference is the only thing that produces
one. It now asks whether there is anything of *hers* to give up, and the pair a retirement opens
goes straight to :class:`~nyxara.njp.universe.ExperimentDesigner` as two rivals that predict
differently under intervention — the one experiment that can break Markov equivalence.

**And a belief carries what would end it, as something she goes and looks for.**
:mod:`nyxara.njp.falsify` reads the falsifier off the claim's own shape and searches her own
record for it: a functional relation already holding a different value, a cause that keeps
occurring without its effect, an arrow established the other way. Found, and the belief is
retracted rather than argued with. What was there before was a template that filled the field in
for every belief — so ``falsifiable`` was true of all of them and discriminated nothing — and
nobody ever looked for what it named.

**The rung she is on is proposed from what her own numbers say, not read off a list.**
:mod:`nyxara.njp.curriculum` is nine fixed stages walked in order, and one rung blocked on a
*sample count* pins the whole ladder while later rungs are already mastered — measured, five
mastered and ``next_stage`` returning the same unworkable rung forever.
:mod:`nyxara.njp.propose` reads her own published counters and proposes a rung that is not on the
list: a metric that has fallen below a level she has personally reached, a stage that cannot be
*judged* because the evidence for it does not exist yet, a capability her own posterior calls
weak. **No source picks its own bar** — a proposal whose threshold is set relative to the current
reading is a test written to be passed — and the generator never sees the held-out split, which a
test enforces by parsing its imports rather than by trusting the docstring. The evaluator is the
code that already scored the nine.

**And when nothing she has explains a thing, a kind can be born.** :mod:`nyxara.njp.concepts`
could always answer *"what is this, and if nothing, why not"*, and could always re-form the
hierarchy around the answer; the only thing that ever asked was a prediction error that happened
to be diagnosed conceptual. Now the store is scanned, and a concept caught **over-claiming** —
promising an invariant one of its own members lacks — is split so the member is covered. Refused
where the repair would be global: loosening what counts as kinship until an outlier fits cost
compression and closed nothing when it was measured, so those gaps are found and reported and
left alone.

**Curiosity is priced by what closing a gap would let her compress.** Value of information says
what rides on an answer; it cannot say that a region is *still yielding structure*.
:mod:`nyxara.njp.progress` reads the two honest description-length ratios in this package as a
derivative against a high-water mark, so a region already compressed out scores zero and only a
new record pays. The level is not the reward — the gain is.

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

**One entity, one key — and the fabric finally reaching the answer.** :mod:`nyxara.njp.canon` and
the two edges around it exist because an audit found the reasoning core in good order and the
connections into it broken. Three measured defects, each traced to one call site. ``birds`` and
``bird`` were two entities, so a correct multi-level inheritance walked to a kind holding nothing —
and the identical inference scored perfectly whenever the sentence happened to use the singular. A
lookup miss was then answered from a *neighbouring* relation, so *"what does a sparrow need?"* came
back ``bird``: confident, sourced, and about a different question. And two equally-supported
answers had no state of their own, so a live tie was indistinguishable from a gap —
:attr:`~nyxara.njp.grounding.Epistemic.CONFLICTING` is now its own verdict, kept narrower than
"contested" because a contradiction that was *revised* has been settled and answering it is right.

The fourth was the largest and had no symptom at all: ``"fabric" in getsource(brain._compose)`` was
``False``. She grew on every turn and none of it could reach what she said. Two edges close that,
reading **different** properties of the same organ so one unfamiliar turn is not charged twice —
:meth:`~nyxara.njp.brain.NJPBrain._temper_by_novelty` discounts a grounded answer by graded
familiarity, and :attr:`~nyxara.njp.router.Seat.FABRIC` dissents from a confident answer where the
manifold could form no trusted prediction at all. Both only ever lower. The fabric may make her
less sure of something she looked up; nothing lets it make her more sure of anything, which is what
stops "she grew" from becoming its own evidence.

The fabric's seat does not *speak*, and that is what it is rather than a limitation being
apologised for: the substrate produces cell ids and a margin, and no path turns those into the word
"water". A seat emitting text on its behalf would be inventing content and attributing it to the
substrate. It asserts the one thing it is entitled to — whether it recognises this situation.

Measured on the change: :mod:`nyxara.eval.intelligence` gained a seventh stage that holds the
*inference* fixed and varies only the phrasing, and it read **0.40 → 1.00** while stages 1-6 stayed
at 1.00 throughout. That the six did not move is the part worth reading: it says the repair was to
extraction, not to the thing being extracted — and it is also why they could not see the defect in
the first place, since every sentence they generate matches the extractor's patterns exactly.

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
from nyxara.njp.adversary import (
    Attack,
    AttackKind,
    AttackReport,
    SelfAttacker,
    Stance,
)
from nyxara.njp.brain import NJPBrain, NJPPercept, NJPThought
from nyxara.njp.canon import canonical_entity, canonical_relation, singular
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
from nyxara.njp.blackbox import BlackBox, Episode, FailureMode
from nyxara.njp.doing import (
    Affordance,
    CognitiveAgency,
    Goal,
)
from nyxara.njp.society import (
    Case,
    CognitiveSociety,
    Contribution,
    Role,
)
from nyxara.njp.teacher import (
    CortexTeacher,
    Distillation,
    Distiller,
    Lesson,
    RecordedTeacher,
    Step,
    Teacher,
    TeacherCouncil,
    Verdict as TeachingVerdict,
    Verification,
)
from nyxara.njp.evolution import (
    CognitiveEvolution,
    EvolutionTrial,
    Measurement,
    Mutation,
    Situation,
)
from nyxara.njp.assume import (
    Assumption,
    AssumptionKind,
    AssumptionMiner,
    AssumptionStatus,
)
from nyxara.njp.economy import Budget, CognitiveEconomy, Tier, budget_for
from nyxara.njp.semantics import (
    Meaning,
    SemanticCompiler,
    Tag,
    Token,
    compile_meaning,
    tag_tokens,
)
from nyxara.njp.universe import Hypothesis as UniverseHypothesis
from nyxara.njp.evolve import EvolutionStep, ModuleCost, Profiler, SelfEvolver
from nyxara.njp.fabric import Fabric, GrowthReport, SettleResult
from nyxara.njp.integrate import LearningLoop, LoopReport
from nyxara.njp.ledger import ErrorMemory, ErrorRecord, Generation, Ledger
from nyxara.njp.manifold import Manifold, Prediction, Snapshot
from nyxara.njp.agency import ActionValue, Agent, Outcome, Step
from nyxara.njp.agency import Plan as ActionPlan
from nyxara.njp.calculate import Calculator, Evaluation
# The mathematician has no `__main__` and is imported here; `nyxara.njp.mathschool` carries one
# and is kept out for the same reason `school`, `study`, `general` and `lessons` are —
# `NJPBrain.go_to_maths_school` and `NJPBrain.sit_maths_exam` are its in-process entry points.
from nyxara.njp.mathematics import Mathematician, Poly, Solution as MathSolution
from nyxara.njp.mathsolver import Constraint, Expr, Problem, Solver
from nyxara.njp.mathsolver import Solution as SolvedProblem
# The corpus reader and its engines are imported; `nyxara.njp.corpusschool` carries a `__main__`
# and is kept out for the reason `mathschool` is — `NJPBrain.go_to_corpus_school` and
# `NJPBrain.sit_corpus_exam` are its in-process entry points.
from nyxara.njp.corpus import CorpusRecord, CorpusVerdict, verify_corpus_answer
from nyxara.njp.corpussolver import CorpusReading, CorpusSolver
# The explanation organ. `nyxara.njp.explainschool` is kept out for the same reason
# `discourseschool` is: it carries a `__main__` and is an examination rather than a faculty.
#
# `Explanation` is aliased, and the alias is why this comment exists. The namespace check that
# V.34 paid seven commits to learn was run before these five names went in, and it caught one:
# `nyxara.njp.predictive.Explanation` — the account a predictive model gives of a surprise — was
# already here, imported further down, and would have silently won the collision. Same word, two
# organs, both entitled to it; the export list is the wrong place to settle that, so the newer one
# takes a qualified name here and keeps its own where it lives. `ClaimLedger`, `TurnAnticipation`
# and `TurnSurprise` above are the same decision made three times before.
from nyxara.njp.explain import (
    Chain,
    Explainer,
    Explanation as CausalExplanation,
    Plan,
    Step,
)
from nyxara.njp.explainread import read_explanation_question
from nyxara.njp.asking import Asking, Cue as QuestionCue, Frame as QuestionFrame
# `Attack` is already `nyxara.njp.adversary.Attack` — an attack on a *belief*. This one attacks an
# *explanation*, which is a different object, so it takes a qualified name here for the reason
# `CausalExplanation` did: the export list is the wrong place to settle a word two organs are both
# entitled to.
from nyxara.njp.predator import (
    Attack as ExplanationAttack,
    Predator,
    Survival,
)
# `Structure` and `Pattern` are aliased for the reason `CausalExplanation` was: the namespace was
# checked before these went in. `Observation` is free, `Surgeon` and `Fusion` are free.
from nyxara.njp.surgery import (
    Observation,
    Structure as CausalStructure,
    Surgeon,
    Verdict as StructureVerdict,
)
from nyxara.njp.fusion import Abstraction, Analogy, Fusion, Pattern as ShapePattern
# V.42/V.43. `Step`, `Status`, `Claim`, `Kind` and `Path` are all words other organs already own,
# so every one of them takes a qualified name here. The namespace was checked before they went in,
# which is the habit V.34 paid seven commits to learn.
from nyxara.njp.loop import Loop, Model as CausalModel, Reason as UnknownReason
# V.44. `Verdict` is already `njp.ledger`'s and `Standing` already `njp.discourse`'s, so both are
# qualified; `Immune` and `Antigen` are free.
from nyxara.njp.immune import (
    Antigen,
    Immune,
    Standing as SourceStanding,
    Verdict as ImmuneVerdict,
)
# V.45. `Prediction` is already `njp.predict`'s and `Theory` is free; `Situation`, `Invariant`,
# `Law` and `Hunter` are all free. Checked before adding, as ever.
from nyxara.njp.theory import (
    Hunter,
    Invariant,
    Law,
    Prediction as LawPrediction,
    Situation,
    Theory,
)
# V.46/V.47. `Constraint` is already `njp.puzzle`'s, so it is qualified; the rest are free.
from nyxara.njp.boundary import (
    Boundary,
    Constraint as HardConstraint,
    Funnel,
    Impossible,
    Necessary,
)
from nyxara.njp.conceptgenome import Genome, Kinship, read_genome
# V.48. `Demonstration` belongs to `njp.coding` in this namespace and `Shape` to `njp.reasoning`,
# so both are qualified here rather than shadowing modules that had the name first. `Relation` is
# free, and it is the right word for what it is: one relation read out of a passage.
from nyxara.njp.encyclopedia import (
    Article,
    Coverage,
    Encyclopedia,
    taught_on_wikipedia,
)
from nyxara.njp.passage import (
    Demonstration as ReadingLesson,
    KnowledgeObject,
    PassageReader,
    Relation,
    Shape as ReadingShape,
    taught_reader,
)
from nyxara.njp.provenance import (
    Blame,
    Claim as ProvenanceClaim,
    Kind as ProvenanceKind,
    Ledger as ProvenanceLedger,
    Path as ProvenancePath,
    PostMortem,
    Status as ProvenanceStatus,
    Step as ProvenanceStep,
)

# The communication organ is imported; `nyxara.njp.discourseschool` carries a `__main__` and is
# kept out for the reason `corpusschool` is — `NJPBrain.go_to_discourse_school` is its in-process
# entry point.
from nyxara.njp.discourse import (
    ActLearner,
    Alternation,
    Anticipation as TurnAnticipation,
    Claim,
    ClosedClassLearner,
    Communicator,
    Connective,
    Figure,
    Interpretation,
    Ledger as ClaimLedger,
    Minds,
    Readings,
    Reference,
    Register,
    Exchange,
    Expectation,
    Footing,
    Induction,
    Link,
    Reply,
    Resolution,
    Retelling,
    Scale,
    Standing,
    Surprise as TurnSurprise,
    Uptake,
    Verdict as ClaimVerdict,
)
from nyxara.njp.index import IntelligenceIndex, IntelligenceVector, Term
from nyxara.njp.core import (
    CognitiveLearningCore,
    CoreReport,
    Derivation,
    RepresentReport,
    ReviseReport,
    Schema,
    TestReport,
    Transitivity,
)
from nyxara.njp.curriculum import STAGES, Curriculum, Report, Stage, StageResult
from nyxara.njp.coding import (
    Check,
    CodeError,
    Coder,
    Demonstration,
    Example,
    Interpreter,
    Learned,
    Program,
    Schema as CodeSchema,
    Spec,
    Written,
    read_python,
)
# The grammar she learns. Two names collide with the coding faculty's and are aliased rather than
# renamed at source: a `Demonstration` is the right word for a worked example in both modules, and
# a package that renamed one of them would be letting an export list dictate what a thing is
# called where it lives. `Schema as CodeSchema` above is the same decision.
from nyxara.njp.language import (
    Affix,
    Construction,
    Joint,
    Marker,
    Process,
    Rule,
    Demonstration as ShownSentence,
    Grammar,
    LanguageFaculty,
    Learned as GrammarLearned,
    Lexicon,
    Morphology,
    Reading as ParseReading,
    Segment,
    Slot,
    Tongue,
)
from nyxara.njp.dialects import Dialect, Utterance, mint_dialect
# The hard banks and the English/Hindi curriculum are deliberately NOT imported here:
# `nyxara.njp.hard` and `nyxara.njp.lessons` each carry a `__main__` entry point, and a module a
# package's `__init__` has already imported emits a RuntimeWarning when it is then run with
# `python -m`. Same reason `school` and `study` are kept out.
# `nyxara.njp.school` is deliberately NOT imported here. It carries a `__main__` entry point, and
# a module that a package's `__init__` has already imported emits a RuntimeWarning about
# unpredictable behaviour when it is then run with `python -m`. `nyxara.njp.study` is kept out for
# the same reason; `NJPBrain.go_to_school` is the in-process entry point.
# `nyxara.njp.general` is kept out for the third time for the same reason — it is
# `python -m nyxara.njp.general` — and `NJPBrain.sit_general_exam` is its in-process entry point.
# `nyxara.njp.puzzle` has no `__main__` and could be imported here; it is kept beside `general`
# instead because the two are one subject, and `NJPBrain.puzzle` is its in-process entry point.
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
    # NJP V.05 — arithmetic, and the loop that makes the organs add up to learning
    "Calculator", "Evaluation",
    # `Solution` is aliased for the same reason `Demonstration` and `Schema` are above: the word
    # is right in both places, and an export list does not get to rename a thing where it lives.
    "Mathematician", "MathSolution", "Poly",
    # NJP V.24 — problems she has never seen
    "Constraint", "Expr", "Problem", "SolvedProblem", "Solver",
    "CorpusReading", "CorpusRecord", "CorpusSolver", "CorpusVerdict", "verify_corpus_answer",
    # NJP V.26 — what was meant, and by whom (njp/discourse.py)
    "ActLearner", "Alternation", "TurnAnticipation", "Claim", "ClaimVerdict", "ClosedClassLearner", "Communicator", "Connective", "Expectation",
    "Exchange", "Figure", "Footing", "Induction", "Interpretation", "ClaimLedger", "Link", "Minds", "Readings", "Reference",
    "Register", "Reply", "Resolution", "Retelling", "Scale", "Standing", "TurnSurprise", "Uptake",
    # NJP V.36 — what, how and why (njp/explain.py)
    "Chain", "Explainer", "CausalExplanation", "Plan", "Step", "read_explanation_question",
    # NJP V.38/V.39 — the question form induced, and the predator over explanations
    "Asking", "QuestionCue", "QuestionFrame",
    "Predator", "ExplanationAttack", "Survival",
    # NJP V.40 — rival structures, and the same shape in two subjects
    # NJP V.42/V.43 — the closed loop, and conclusions that carry their ancestry
    "Loop", "CausalModel", "UnknownReason",
    "ProvenanceLedger", "ProvenanceClaim", "ProvenancePath", "ProvenanceStep",
    "ProvenanceKind", "ProvenanceStatus", "Blame", "PostMortem",
    # NJP V.44 — quarantine, so one new fact cannot corrupt the graph
    "Immune", "Antigen", "ImmuneVerdict", "SourceStanding",
    # NJP V.45 — what never changes, and running what survives
    "Hunter", "Situation", "Invariant", "Law", "Theory", "LawPrediction",
    # NJP V.46/V.47 — what cannot work, and what a concept is made of
    "Boundary", "HardConstraint", "Necessary", "Impossible", "Funnel",
    "Genome", "Kinship", "read_genome",
    "KnowledgeObject", "PassageReader", "Relation", "ReadingLesson", "ReadingShape",
    "taught_reader",
    "Article", "Coverage", "Encyclopedia", "taught_on_wikipedia",
    "Surgeon", "Observation", "CausalStructure", "StructureVerdict",
    "Fusion", "Analogy", "Abstraction", "ShapePattern",
    "CognitiveLearningCore", "Derivation", "Schema", "Transitivity",
    "RepresentReport", "TestReport", "ReviseReport", "CoreReport",
    # she goes after her own conclusions
    "SelfAttacker", "Attack", "AttackReport", "Stance", "AttackKind",
    # the one number she is not allowed to compute about herself
    "IntelligenceIndex", "IntelligenceVector", "Term",
    # one entity, one key — the store's spelling rule
    "canonical_entity", "canonical_relation", "singular",
    # truth is not relevance, and reasoning is not always what a turn calls for
    "SpeechAct", "Act", "SpeechActReader", "Pathway", "CognitivePolicy",
    "RelevanceGate", "RelevanceScore",
    "revise_confidence", "is_verified", "is_meta_commentary",
    # NJP V.09 — language compiled into a typed representation, so the open half of a sentence
    # stops being a list somebody has to keep adding to
    "SemanticCompiler", "Meaning", "Tag", "Token", "compile_meaning", "tag_tokens",
    # NJP V.10 — required computation, not maximum computation
    "CognitiveEconomy", "Budget", "Tier", "budget_for",
    # NJP V.11 — the third knowledge state: what her model assumes and nothing has examined
    "AssumptionMiner", "Assumption", "AssumptionKind", "AssumptionStatus",
    # NJP V.12 — one row per act of thinking, so a failure *mode* can be read off a join
    "BlackBox", "Episode", "FailureMode",
    # NJP V.13 — structural change to her own cognition, adopted only on measured evidence
    "CognitiveEvolution", "Mutation", "EvolutionTrial", "Measurement", "Situation",
    # NJP V.14 — Phase 4: a teacher's *structure*, verified and kept after the teacher is gone
    "Teacher", "TeacherCouncil", "RecordedTeacher", "CortexTeacher",
    "Lesson", "Step", "Verification", "TeachingVerdict", "Distiller", "Distillation",
    # NJP V.15 — §7 goal → plan → action → outcome, and §19 eight specialists over one claim
    "CognitiveAgency", "Goal", "Affordance",
    "CognitiveSociety", "Role", "Case", "Contribution",
    # NJP V.16 — she writes programs, and the machine, not her confidence, decides if they run
    "Coder", "Spec", "Program", "Example", "Demonstration", "Written", "Check", "Learned",
    "CodeSchema", "Interpreter", "CodeError", "read_python",
    # NJP V.18 — the grammar she learns, and the languages she is examined in
    "LanguageFaculty", "Tongue", "Grammar", "Construction", "Slot", "Joint", "Marker",
    "ShownSentence", "ParseReading", "GrammarLearned", "Morphology", "Lexicon", "Affix",
    "Process", "Rule", "Segment", "Dialect", "Utterance", "mint_dialect",
]
