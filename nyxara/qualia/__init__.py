"""NYXARA · qualia — thinking in geometry, speaking in words only at the boundary (🜛).

Language is where a thought is *reported*, not where it happens. This package keeps reasoning in
the latent space — the fitted Poincaré geometry of
:mod:`~nyxara.mind.concept_hyperspace` and the 10,000-dimensional VSA algebra of
:mod:`~nyxara.cognition.hyper_dimensional_vectors` — and converts to text only when there is
something to say to the Master.

That makes one genuinely new thing possible: a concept can exist **without a word for it**. A dense
region of the latent space whose nearest vocabulary anchor is too far away is a structure she found
that no token names. That is a measurable definition, not a poetic one.

It also makes one thing necessary. An unnamed cluster is not automatically true — it can just as
easily be an artefact of the embedding. So a concept must **earn a certificate** before it may
appear in an answer, and there are three honest ways to earn one: it improves prediction on held-out
data, it expresses as a term the prover can certify, or it has a real mechanism in the causal graph.
Uncertified concepts stay in the workspace where they can be inspected. That gate is not a
limitation on the idea; it is the difference between discovery and confident nonsense.
"""

from __future__ import annotations

from nyxara.qualia.semantic_synthesis import (BabelCodec, ConceptCertificate, LatentConcept,
                                              QualiaReport, QualiaWorkspace)

__all__ = [
    "LatentConcept",
    "ConceptCertificate",
    "QualiaWorkspace",
    "BabelCodec",
    "QualiaReport",
]
