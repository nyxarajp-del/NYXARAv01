"""NYXARA · memory — the multi-store cognitive memory system.

This package re-exports the stable surface of NYXARA's memory faculties. Most subsystems
operate over :class:`~nyxara.memory.store.MemoryStore`; :mod:`elastic_synapses` is the
lifelong-learning layer that keeps *skills and learned weights* from being forgotten as new
ones are acquired (Elastic Weight Consolidation).
"""

from __future__ import annotations

from nyxara.memory.elastic_synapses import (
    ConsolidatedTask,
    ElasticSynapses,
    FisherEstimator,
    TorchElasticSynapses,
)

__all__ = [
    "ElasticSynapses",
    "FisherEstimator",
    "ConsolidatedTask",
    "TorchElasticSynapses",
]
