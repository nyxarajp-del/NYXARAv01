"""NYXARA · cognition — faculties that see structure beyond human-perceptible dimensions.

The first member is :mod:`hyper_dimensional_vectors`, the Hyperdimensional Latent Space
Mapping engine: it lifts data into a 10,000-dimensional space where patterns, relations and
emerging signals that are invisible to a 3-D mind become measurable geometry.
"""

from __future__ import annotations

from nyxara.cognition.hyper_dimensional_vectors import (
    CleanupResult,
    Hypervector,
    HyperSpace,
    ItemMemory,
    LatentSpaceMap,
    NoveltyResult,
    PatternReport,
    RandomProjector,
    has_numpy,
)
from nyxara.cognition.few_shot import (
    FewShotLearner,
    OneShotEpisodicMemory,
    Prototype,
)
from nyxara.cognition.abstraction import (
    Schema,
    SchemaInducer,
    abstract_by_analogy,
    abstract_text,
    anti_unify,
    instantiate,
    subsumes,
)
from nyxara.cognition.composition import CompositionalGrammar, Interpretation
from nyxara.cognition.sample_efficient import ConsolidationReport, SampleEfficientMind

__all__ = [
    "Hypervector",
    "HyperSpace",
    "ItemMemory",
    "CleanupResult",
    "RandomProjector",
    "PatternReport",
    "NoveltyResult",
    "LatentSpaceMap",
    "has_numpy",
    # sample-efficient learning, abstraction, compositional generalization
    "FewShotLearner",
    "OneShotEpisodicMemory",
    "Prototype",
    "Schema",
    "SchemaInducer",
    "anti_unify",
    "subsumes",
    "instantiate",
    "abstract_text",
    "abstract_by_analogy",
    "CompositionalGrammar",
    "Interpretation",
    "SampleEfficientMind",
    "ConsolidationReport",
]
