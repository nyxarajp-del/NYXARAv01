"""NYXARA · nyx001/layers/ — the charter's Layers 0–17 (🧬).

Eighteen mechanisms plus the stack that runs them as one cycle. Each module is named for its
layer number so the mapping to the charter is checkable rather than asserted:

===  ==========================  ==============================================================
L    module                      what it does
===  ==========================  ==============================================================
0    ``l00_substrate``           seeded birth: scale, generator streams, birth certificate
1    ``l01_perception``          raw → feature dynamics → entities/relations
2    ``l02_world_model``         ŝ=F_θ(s,a); e=D(s,ŝ); learn. The central organ.
3    ``l03_episodic``            what/when/context/**error**/relevance, prioritised replay
4    ``l04_semantic``            experiences → shared structure → concept → theories
5    ``l05_attention``           A = f(Q,K,V,G,U,M,E,T,R)
6    ``l06_reasoning``           hypothesize → evaluate → contradict → update confidence
7    ``l07_counterfactual``      do(¬X) on a learned model; divergence, with a trust gate
8    ``l08_planning``            goal → sub-goals → simulate → CVaR risk → propose
9    ``l09_curiosity``           learning *progress*, not surprise (no noisy-TV trap)
10   ``l10_contradiction``       attacks its own beliefs; measures and shakes rigidity
11   ``l11_critique``            output → critique → reconstruct → keep only if measurably better
12   ``l12_self_model``          what do I know / where am I failing / how calibrated am I
13   ``l13_meta_learning``       bounded UCB1 bandit over learning strategies
14   ``l14_compression``         1000 experiences → one subspace, with verified fidelity
15   ``l15_resource``            difficulty → budget → anytime stop, and self-calibration
16   ``l16_active_learning``     observe → model → act → consequence → error → learn
17   ``l17_self_improve``        propose → sandbox → A/B → fail-closed gate → roll out or revert
===  ==========================  ==============================================================

Two rules hold across all of them. Every layer is **fail-soft** — a raise inside one becomes a
report, never a dead cycle. And every layer that cannot yet say something returns ``None`` rather
than a number: an uncalibrated system reports ``None`` for its calibration, not ``0.0``, because a
zero would be read as a measurement.
"""

from __future__ import annotations

from nyxara.nyx001.layers.l00_substrate import (
    SCALES,
    BirthCertificate,
    Scale,
    Substrate,
    default_substrate,
)
from nyxara.nyx001.layers.l01_perception import Perception
from nyxara.nyx001.layers.l02_world_model import WorldModel
from nyxara.nyx001.layers.l03_episodic import EpisodicMemory, Recall
from nyxara.nyx001.layers.l04_semantic import SemanticMemory
from nyxara.nyx001.layers.l05_attention import Allocation, Candidate, DynamicAttention
from nyxara.nyx001.layers.l06_reasoning import ReasoningEngine
from nyxara.nyx001.layers.l07_counterfactual import Counterfactual, CounterfactualEngine
from nyxara.nyx001.layers.l08_planning import Planner
from nyxara.nyx001.layers.l09_curiosity import CuriosityEngine, Region
from nyxara.nyx001.layers.l10_contradiction import Attack, ContradictionEngine
from nyxara.nyx001.layers.l11_critique import Critique, Finding, SelfCritique
from nyxara.nyx001.layers.l12_self_model import SelfModel, SelfReport
from nyxara.nyx001.layers.l13_meta_learning import MetaLearner, Strategy
from nyxara.nyx001.layers.l14_compression import CognitiveCompression, Compressed
from nyxara.nyx001.layers.l15_resource import Budget, Difficulty, ResourceAwareCognition
from nyxara.nyx001.layers.l16_active_learning import ActiveLearning, LoopStep
from nyxara.nyx001.layers.l17_self_improve import Outcome, Proposal, SelfImprovement
from nyxara.nyx001.layers.stack import CognitiveStack, CycleResult
from nyxara.nyx001.layers.types import (
    Action,
    Belief,
    Concept,
    Entity,
    Experience,
    Hypothesis,
    LayerReport,
    Observation,
    Percept,
    Plan,
    Prediction,
    Relation,
)

__all__ = [
    # the stack
    "CognitiveStack", "CycleResult",
    # L0
    "Substrate", "Scale", "SCALES", "BirthCertificate", "default_substrate",
    # L1-L4
    "Perception", "WorldModel", "EpisodicMemory", "Recall", "SemanticMemory",
    # L5-L8
    "DynamicAttention", "Candidate", "Allocation", "ReasoningEngine",
    "CounterfactualEngine", "Counterfactual", "Planner",
    # L9-L13
    "CuriosityEngine", "Region", "ContradictionEngine", "Attack",
    "SelfCritique", "Critique", "Finding", "SelfModel", "SelfReport",
    "MetaLearner", "Strategy",
    # L14-L17
    "CognitiveCompression", "Compressed", "ResourceAwareCognition", "Difficulty", "Budget",
    "ActiveLearning", "LoopStep", "SelfImprovement", "Proposal", "Outcome",
    # shared types
    "Observation", "Percept", "Entity", "Relation", "Prediction", "Experience",
    "Concept", "Hypothesis", "Belief", "Plan", "Action", "LayerReport",
]
