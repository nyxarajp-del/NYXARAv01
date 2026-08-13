"""NYXARA · nyx001/lingua/ — Track B, language as a modality (🗣, not the base).

The charter runs two tracks: Track A is the cognitive architecture (Layers 0–17), Track B is the
language substrate. This package is Track B, and its governing claim is the charter's ordering:

    Environment → Perception → Concepts → World Model → **Language Grounding**

Language is *last*. Nothing here bootstraps meaning from text, because there is no pretrained
anything to bootstrap from — the charter forbids it, and this package has none.

* :mod:`~nyxara.nyx001.lingua.tokenizer` — segmentation **learned online from experience**, byte
  alphabet, empty merge table at birth. Not a corpus-trained vocabulary.
* :mod:`~nyxara.nyx001.lingua.dynamic_embedding` — ``z_t = f(x_t, c_t, m_t, g_t, u_t)``. A
  representation that moves with context, memory, goal and uncertainty rather than a lookup row.
* :mod:`~nyxara.nyx001.lingua.sequence` — a **selective state-space scan**, linear in sequence
  length and carrying state. Explicitly not the Transformer recipe the charter prohibits.
* :mod:`~nyxara.nyx001.lingua.grounding` — words bound to concepts the system formed itself, with
  a refusal to report a meaning for anything that has not earned one.

What this is not: a language model. There is no pretraining, no corpus, and no generation
objective running by default. At birth the tokenizer segments nothing, the embeddings are random,
and the grounding vocabulary is empty. Everything here has to be earned from experience, slowly —
which is what the charter's artificial childhood means when applied to language.

NYXARA's existing LLM ladder (``mind/llm.py``) is untouched and remains a separate, optional
modality. Track B does not read from it, and it does not read from Track B.
"""

from __future__ import annotations

from nyxara.nyx001.lingua.cortex import LanguageCortex, Reading
from nyxara.nyx001.lingua.dynamic_embedding import DynamicEmbedding
from nyxara.nyx001.lingua.grounding import Binding, Grounding
from nyxara.nyx001.lingua.sequence import SequenceModel
from nyxara.nyx001.lingua.tokenizer import Tokenizer

__all__ = ["Tokenizer", "DynamicEmbedding", "SequenceModel", "Grounding", "Binding",
           "LanguageCortex", "Reading"]
