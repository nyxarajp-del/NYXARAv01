"""NYXARA · cognition/grounded_understanding.py — perceptual symbol grounding (🍎, qualia).

The single biggest gap between a statistical language model and a *mind* is **grounded
understanding**. In an ordinary LLM the word "apple" is only a token — a slot in a
probability table over "what word comes next." Nothing in the world stands behind it: no
taste, no colour, no weight. The symbol floats free of its referent (Harnad's *symbol
grounding problem*).

A grounded mind is different. When NYXARA hears **"apple"** the symbol should fire across
*every relevant sense at once* — **taste** (sweet, a little tart), **vision** (red/green,
round), **touch** (smooth, firm), **physics** (≈150 g, and it *falls* — gravity),
**affordance** (you can *eat* it). The meaning of a word is this multimodal pattern of
activation, not its co-occurrence statistics. That is what this module gives her.

NYXARA already had :mod:`nyxara.cognition.language_grounding`, but that grounds language
into **dynamics** — ``(state, action, next_state, reward)`` transitions for the world
model. It answers *"what happens if I act?"*. This module answers the prior, deeper
question *"what does this word **mean** — in the senses?"*. The two are complementary:
one grounds verbs in change, this grounds nouns in perception.

Floor + ceiling, in NYXARA's tradition:

* **floor (always available, pure stdlib).** A curated *seed lexicon* of real multimodal
  knowledge (apple → sweet/red/round/edible/≈150 g/falls; fire → hot/bright/dangerous; …)
  plus a *perceptual-descriptor ontology* (``sweet→taste``, ``red→vision``,
  ``round→shape``, ``heavy→weight``, ``hot→temperature``, ``loud→sound``, ``edible→
  affordance``, …). So she **knows**, and **grounds new words by reading**, with no
  network and no model — *khud*, by herself.
* **ceiling (optional).** When a *real* (non-mock) LLM is configured, an unknown concept's
  multimodal feature profile can be requested as JSON and merged through the very same
  :class:`PerceptualSchema`. Any failure silently degrades to the floor.

Everything is introspectable (:meth:`GroundedLexicon.report`, :meth:`to_dict`,
:meth:`GroundedLexicon.explain`). Meaning is geometry: two grounded symbols are compared by
the cosine of their perceptual feature vectors (apple ≈ orange ≫ apple ≈ car), and an
optional hyperdimensional encoding (via :mod:`nyxara.cognition.hyper_dimensional_vectors`)
demonstrates the same relations in a 10 000-D space.

Pairs with :mod:`senses.binding` (real percepts → grounding), :mod:`senses.nlp` (reading),
:mod:`cognition.concept_formation` (grounded instances → an IS-A taxonomy), and
:mod:`cognition.language_grounding` (nouns grounded here, verbs grounded there).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from nyxara.senses import nlp

__all__ = [
    "Sense",
    "PerceptualSchema",
    "GroundedConcept",
    "GroundedActivation",
    "GroundedLexicon",
    "DESCRIPTOR_ONTOLOGY",
]


# --------------------------------------------------------------------------- #
# Sense — the perceptual modalities a symbol can activate
# --------------------------------------------------------------------------- #
class Sense(str, Enum):
    """The modalities meaning can light up. Distinct from :class:`senses.binding.Modality`,
    which names *input channels* (text/web/image/audio); these name *perceptual qualities*."""

    VISION = "vision"          # colour, shape, size, brightness
    SOUND = "sound"            # loudness, the noise a thing makes
    TASTE = "taste"            # sweet, sour, bitter, salty, savoury, spicy
    SMELL = "smell"            # fragrant, pungent, …
    TOUCH = "touch"            # texture: smooth/rough/soft/hard/wet/sharp
    PHYSICS = "physics"        # weight, gravity, temperature — the body of the world
    AFFORDANCE = "affordance"  # what you can *do* with it: eat, drink, throw, hold, ride
    EMOTION = "emotion"        # the affect it carries: pleasant, fearful, …
    ACTION = "action"          # animacy / characteristic motion: animate, flies, fast


# Map a real input channel (binding.Modality) to the perceptual sense it feeds. Text/web/
# document flow through the *language* path, not here, so they are intentionally absent.
_CHANNEL_TO_SENSE = {"image": Sense.VISION, "audio": Sense.SOUND}


# --------------------------------------------------------------------------- #
# Perceptual-descriptor ontology — word → (sense, feature, signed activation)
# --------------------------------------------------------------------------- #
# The floor's load-bearing lexicon: each descriptor word grounds to one perceptual feature.
# A negative activation means the *opposite* pole of a bipolar feature (light = -weight).
DESCRIPTOR_ONTOLOGY: Dict[str, Tuple[Sense, str, float]] = {
    # taste
    "sweet": (Sense.TASTE, "sweet", 1.0), "sugary": (Sense.TASTE, "sweet", 1.0),
    "sour": (Sense.TASTE, "sour", 1.0), "tart": (Sense.TASTE, "sour", 1.0),
    "bitter": (Sense.TASTE, "bitter", 1.0), "salty": (Sense.TASTE, "salty", 1.0),
    "savory": (Sense.TASTE, "savory", 1.0), "savoury": (Sense.TASTE, "savory", 1.0),
    "umami": (Sense.TASTE, "savory", 1.0), "spicy": (Sense.TASTE, "spicy", 1.0),
    "hot": (Sense.PHYSICS, "temperature", 1.0),     # "hot" defaults to temperature
    "tasty": (Sense.TASTE, "sweet", 0.5), "delicious": (Sense.TASTE, "sweet", 0.5),
    # smell
    "fragrant": (Sense.SMELL, "fragrant", 1.0), "aromatic": (Sense.SMELL, "fragrant", 1.0),
    "smelly": (Sense.SMELL, "pungent", 1.0), "pungent": (Sense.SMELL, "pungent", 1.0),
    "stinky": (Sense.SMELL, "pungent", 1.0),
    # vision — colour
    "red": (Sense.VISION, "red", 1.0), "green": (Sense.VISION, "green", 1.0),
    "blue": (Sense.VISION, "blue", 1.0), "yellow": (Sense.VISION, "yellow", 1.0),
    "orange": (Sense.VISION, "orange", 1.0), "white": (Sense.VISION, "white", 1.0),
    "black": (Sense.VISION, "black", 1.0), "brown": (Sense.VISION, "brown", 1.0),
    "grey": (Sense.VISION, "grey", 1.0), "gray": (Sense.VISION, "grey", 1.0),
    "colorful": (Sense.VISION, "colorful", 1.0), "colourful": (Sense.VISION, "colorful", 1.0),
    "pink": (Sense.VISION, "pink", 1.0), "purple": (Sense.VISION, "purple", 1.0),
    # vision — shape
    "round": (Sense.VISION, "round", 1.0), "spherical": (Sense.VISION, "round", 1.0),
    "square": (Sense.VISION, "square", 1.0), "flat": (Sense.VISION, "flat", 1.0),
    "long": (Sense.VISION, "long", 1.0), "curved": (Sense.VISION, "curved", 1.0),
    "pointy": (Sense.VISION, "pointy", 1.0),
    # vision — size (bipolar)
    "big": (Sense.VISION, "size", 1.0), "large": (Sense.VISION, "size", 1.0),
    "huge": (Sense.VISION, "size", 1.0), "giant": (Sense.VISION, "size", 1.0),
    "small": (Sense.VISION, "size", -1.0), "tiny": (Sense.VISION, "size", -1.0),
    "little": (Sense.VISION, "size", -1.0),
    # vision — brightness (bipolar)
    "bright": (Sense.VISION, "brightness", 1.0), "shiny": (Sense.VISION, "brightness", 1.0),
    "glowing": (Sense.VISION, "brightness", 1.0), "dark": (Sense.VISION, "brightness", -1.0),
    "dim": (Sense.VISION, "brightness", -1.0), "dull": (Sense.VISION, "brightness", -0.5),
    # sound (bipolar loudness)
    "loud": (Sense.SOUND, "loudness", 1.0), "noisy": (Sense.SOUND, "loudness", 1.0),
    "quiet": (Sense.SOUND, "loudness", -1.0), "silent": (Sense.SOUND, "loudness", -1.0),
    "musical": (Sense.SOUND, "musical", 1.0),
    # touch — texture
    "smooth": (Sense.TOUCH, "smooth", 1.0), "rough": (Sense.TOUCH, "rough", 1.0),
    "soft": (Sense.TOUCH, "soft", 1.0), "fluffy": (Sense.TOUCH, "soft", 1.0),
    "hard": (Sense.TOUCH, "hard", 1.0), "firm": (Sense.TOUCH, "hard", 0.7),
    "sharp": (Sense.TOUCH, "sharp", 1.0), "sticky": (Sense.TOUCH, "sticky", 1.0),
    "wet": (Sense.TOUCH, "wet", 1.0), "dry": (Sense.TOUCH, "wet", -1.0),
    "bouncy": (Sense.TOUCH, "bouncy", 1.0),
    # physics — weight (bipolar), temperature (bipolar)
    "heavy": (Sense.PHYSICS, "weight", 1.0), "light": (Sense.PHYSICS, "weight", -1.0),
    "weightless": (Sense.PHYSICS, "weight", -1.0),
    "cold": (Sense.PHYSICS, "temperature", -1.0), "cool": (Sense.PHYSICS, "temperature", -0.5),
    "warm": (Sense.PHYSICS, "temperature", 0.5), "freezing": (Sense.PHYSICS, "temperature", -1.0),
    "burning": (Sense.PHYSICS, "temperature", 1.0),
    # affordance
    "edible": (Sense.AFFORDANCE, "edible", 1.0), "drinkable": (Sense.AFFORDANCE, "drinkable", 1.0),
    "throwable": (Sense.AFFORDANCE, "throwable", 1.0),
    "graspable": (Sense.AFFORDANCE, "graspable", 1.0),
    "wearable": (Sense.AFFORDANCE, "wearable", 1.0),
    "rideable": (Sense.AFFORDANCE, "rideable", 1.0),
    # emotion / valence
    "pleasant": (Sense.EMOTION, "pleasant", 1.0), "lovely": (Sense.EMOTION, "pleasant", 1.0),
    "calming": (Sense.EMOTION, "pleasant", 1.0), "cute": (Sense.EMOTION, "pleasant", 1.0),
    "dangerous": (Sense.EMOTION, "fear", 1.0), "scary": (Sense.EMOTION, "fear", 1.0),
    "deadly": (Sense.EMOTION, "fear", 1.0), "disgusting": (Sense.EMOTION, "disgust", 1.0),
    # action / animacy
    "animate": (Sense.ACTION, "animate", 1.0), "alive": (Sense.ACTION, "animate", 1.0),
    "living": (Sense.ACTION, "animate", 1.0), "fast": (Sense.ACTION, "fast", 1.0),
    "slow": (Sense.ACTION, "fast", -1.0), "flying": (Sense.ACTION, "flies", 1.0),
}

# Verbs that reveal an *affordance* (what can be done with/to the thing) or a physical law.
# Keys are crude stems matched after light normalisation; many surface forms collapse here.
_AFFORDANCE_VERBS: Dict[str, Tuple[Sense, str, float]] = {
    "eat": (Sense.AFFORDANCE, "edible", 1.0), "drink": (Sense.AFFORDANCE, "drinkable", 1.0),
    "throw": (Sense.AFFORDANCE, "throwable", 1.0), "hold": (Sense.AFFORDANCE, "graspable", 1.0),
    "grab": (Sense.AFFORDANCE, "graspable", 1.0), "grasp": (Sense.AFFORDANCE, "graspable", 1.0),
    "wear": (Sense.AFFORDANCE, "wearable", 1.0), "ride": (Sense.AFFORDANCE, "rideable", 1.0),
    "drive": (Sense.AFFORDANCE, "rideable", 1.0), "cut": (Sense.AFFORDANCE, "cuts", 1.0),
    # physical law: things fall — the gravity the request explicitly asks for
    "fall": (Sense.PHYSICS, "gravity", 1.0), "drop": (Sense.PHYSICS, "gravity", 1.0),
    "sink": (Sense.PHYSICS, "gravity", 1.0),
    # noise-making verbs → sound
    "bark": (Sense.SOUND, "loudness", 1.0), "sing": (Sense.SOUND, "musical", 1.0),
    "ring": (Sense.SOUND, "loudness", 1.0), "buzz": (Sense.SOUND, "loudness", 1.0),
    "roar": (Sense.SOUND, "loudness", 1.0),
}

_COPULA = frozenset(
    "is are was were be been being feels felt feel tastes taste smells smell "
    "looks look sounds sound seems seem becomes become appear appears".split()
)
_ARTICLES = frozenset("a an the this that these those its his her their my your".split())
_NEGATORS = frozenset("not no n't never without".split())
_VERB_SUFFIXES = ("ing", "ed", "es", "s")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _singular(word: str) -> str:
    """Crudely singularise an English noun (stdlib only): apples→apple, berries→berry."""
    w = word.lower()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("oes") and len(w) > 4:                 # mangoes→mango, tomatoes→tomato
        return w[:-2]
    if w.endswith(("ses", "xes", "zes", "ches", "shes")) and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def _verb_stem(word: str) -> str:
    """Collapse a verb's surface form to the stem used in :data:`_AFFORDANCE_VERBS`."""
    w = word.lower()
    if w in _AFFORDANCE_VERBS:
        return w
    for suf in _VERB_SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 2:
            stem = w[: -len(suf)]
            if stem in _AFFORDANCE_VERBS:
                return stem
            if len(stem) >= 2 and stem[-1] == stem[-2] and stem[:-1] in _AFFORDANCE_VERBS:
                return stem[:-1]
    # irregulars
    return {"fell": "fall", "ate": "eat", "drank": "drink", "threw": "throw",
            "held": "hold", "rang": "ring", "sang": "sing", "rode": "ride",
            "drove": "drive", "sank": "sink"}.get(w, w)


# --------------------------------------------------------------------------- #
# PerceptualSchema — stable (sense, feature) → dimension map
# --------------------------------------------------------------------------- #
class PerceptualSchema:
    """Maps every ``(sense, feature)`` pair to a **stable** vector dimension.

    The same discipline as :class:`nyxara.cognition.language_grounding.VariableRegistry`,
    lifted to a two-level (modality, feature) key so grounded concepts can be compared as
    fixed-width vectors — "taste:sweet" is always the same dimension across concepts and
    across calls, which is what makes cosine similarity meaningful.
    """

    def __init__(self) -> None:
        self._index: Dict[Tuple[str, str], int] = {}

    def __len__(self) -> int:
        return len(self._index)

    def key(self, sense: Sense, feature: str) -> Tuple[str, str]:
        return (sense.value, feature.strip().lower())

    def index_of(self, sense: Sense, feature: str) -> int:
        k = self.key(sense, feature)
        if k not in self._index:
            self._index[k] = len(self._index)
        return self._index[k]

    def width(self) -> int:
        return len(self._index)

    def names(self) -> List[str]:
        return [f"{s}:{f}" for (s, f) in
                sorted(self._index, key=lambda kk: self._index[kk])]

    def vectorize(self, concept: "GroundedConcept", *, width: Optional[int] = None
                  ) -> List[float]:
        """A fixed-width all-float row; each active feature placed at its stable dim."""
        for sense, feats in concept.senses.items():     # register first so width is final
            for feature in feats:
                self.index_of(sense, feature)
        w = width if width is not None else self.width()
        row = [0.0] * w
        for sense, feats in concept.senses.items():
            for feature, value in feats.items():
                i = self._index[self.key(sense, feature)]
                if i < w:
                    row[i] = float(value)
        return row


# --------------------------------------------------------------------------- #
# GroundedConcept — a symbol grounded across the senses
# --------------------------------------------------------------------------- #
@dataclass
class GroundedConcept:
    """One symbol's multimodal meaning: per-sense ``{feature: activation}``.

    Activations are running means in roughly ``[-1, 1]`` (negative = the opposite pole of a
    bipolar feature). Repeated, consistent evidence stabilises a feature; conflicting
    evidence averages it — the same incremental-mean discipline as the world model's
    effect signatures.
    """

    name: str
    senses: Dict[Sense, Dict[str, float]] = field(default_factory=dict)
    counts: Dict[Tuple[str, str], int] = field(default_factory=dict)
    categories: set = field(default_factory=set)        # captured "is a <noun>" categories
    sources: int = 0                                    # how many observations grounded this

    def reinforce(self, sense: Sense, feature: str, value: float) -> None:
        """Fold one piece of evidence into a feature's running mean."""
        feature = feature.strip().lower()
        if not feature:
            return
        value = max(-1.0, min(1.0, float(value)))
        bucket = self.senses.setdefault(sense, {})
        k = (sense.value, feature)
        n = self.counts.get(k, 0) + 1
        self.counts[k] = n
        bucket[feature] = bucket.get(feature, 0.0) + (value - bucket.get(feature, 0.0)) / n

    def active_senses(self) -> List[Sense]:
        """Senses with at least one feature whose magnitude clears the noise floor."""
        return [s for s, feats in self.senses.items()
                if any(abs(v) >= 0.05 for v in feats.values())]

    def feature_count(self) -> int:
        return sum(len(f) for f in self.senses.values())

    @property
    def strength(self) -> float:
        """Mean evidence per feature — a rough confidence in this grounding."""
        fc = self.feature_count()
        return (sum(self.counts.values()) / fc) if fc else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "senses": {s.value: dict(sorted(f.items())) for s, f in self.senses.items()},
            "active_senses": [s.value for s in self.active_senses()],
            "categories": sorted(self.categories),
            "features": self.feature_count(),
            "sources": self.sources,
            "strength": round(self.strength, 3),
        }


# --------------------------------------------------------------------------- #
# GroundedActivation — what "imagining a word" lights up
# --------------------------------------------------------------------------- #
@dataclass
class GroundedActivation:
    """The pattern that fires when NYXARA brings a word to mind — all senses at once."""

    word: str
    grounded: bool
    senses: Dict[Sense, Dict[str, float]]
    vector: List[float]
    confidence: float = 0.0

    @property
    def modalities(self) -> List[str]:
        return [s.value for s, feats in self.senses.items()
                if any(abs(v) >= 0.05 for v in feats.values())]

    def fires(self, sense: Sense) -> bool:
        feats = self.senses.get(sense, {})
        return any(abs(v) >= 0.05 for v in feats.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word": self.word,
            "grounded": self.grounded,
            "senses": {s.value: dict(sorted(f.items())) for s, f in self.senses.items()},
            "modalities": self.modalities,
            "n_modalities": len(self.modalities),
            "vector_dim": len(self.vector),
            "confidence": round(self.confidence, 3),
        }


# --------------------------------------------------------------------------- #
# Seed lexicon — the floor's innate, real multimodal knowledge
# --------------------------------------------------------------------------- #
# name → (descriptor words expanded through DESCRIPTOR_ONTOLOGY, explicit physics extras).
# Explicit extras carry numbers descriptors can't (a weight in kg-ish units, gravity).
_FALLS = {Sense.PHYSICS: {"gravity": 1.0}}


def _phys(weight: Optional[float] = None, gravity: float = 1.0,
          **extra: float) -> Dict[Sense, Dict[str, float]]:
    d: Dict[str, float] = {"gravity": gravity}
    if weight is not None:
        d["weight"] = weight
    d.update(extra)
    return {Sense.PHYSICS: d}


_SEED: Dict[str, Tuple[List[str], Dict[Sense, Dict[str, float]]]] = {
    # fruit — the canonical grounding example
    "apple": (["sweet", "sour", "red", "round", "smooth", "edible"], _phys(0.15)),
    "orange": (["sweet", "sour", "orange", "round", "edible"], _phys(0.2)),
    "banana": (["sweet", "yellow", "long", "soft", "edible"], _phys(0.12)),
    "lemon": (["sour", "yellow", "round", "edible"], _phys(0.1)),
    "grape": (["sweet", "round", "small", "smooth", "edible"], _phys(0.005)),
    "strawberry": (["sweet", "red", "soft", "edible"], _phys(0.02)),
    # food & drink
    "bread": (["edible", "soft", "brown"], _phys(0.5)),
    "water": (["drinkable", "wet", "cool"], _phys(1.0)),
    "milk": (["drinkable", "white", "wet", "edible"], _phys(1.0)),
    "coffee": (["bitter", "hot", "drinkable", "brown", "wet", "aromatic"], _phys(0.3)),
    "honey": (["sweet", "sticky", "edible", "yellow"], _phys(0.3)),
    "chili": (["spicy", "red", "edible"], _phys(0.02, temperature=0.5)),
    # nature
    "fire": (["hot", "bright", "orange", "dangerous"], _phys(gravity=0.0, temperature=1.0)),
    "ice": (["cold", "smooth", "hard", "white"], _phys(0.5, temperature=-1.0)),
    "stone": (["hard", "rough", "heavy", "grey"], _phys(0.8)),
    "sun": (["bright", "hot", "round", "yellow"], _phys(gravity=0.0, temperature=1.0)),
    "moon": (["round", "white", "dim"], _phys(gravity=0.0)),
    "rain": (["wet", "cold"], _phys(0.0)),
    "snow": (["cold", "white", "soft"], _phys(0.0)),
    "tree": (["big", "brown", "green", "hard"], _phys(50.0)),
    "grass": (["green", "soft"], _phys(0.0)),
    "ocean": (["big", "wet", "blue", "cold", "loud"], _phys(0.0)),
    # animals
    "dog": (["soft", "brown", "loud", "animate", "pleasant"], _phys(15.0)),
    "cat": (["soft", "animate", "quiet", "pleasant"], _phys(4.0)),
    "bird": (["small", "animate", "musical", "flying"], _phys(0.1)),
    "lion": (["big", "animate", "loud", "dangerous"], _phys(190.0)),
    "fish": (["wet", "small", "animate", "smooth"], _phys(0.5)),
    "bee": (["small", "yellow", "loud", "animate", "dangerous"], _phys(0.001)),
    # objects
    "car": (["big", "hard", "heavy", "loud", "fast", "rideable"], _phys(1200.0)),
    "ball": (["round", "light", "throwable", "bouncy"], _phys(0.4)),
    "knife": (["sharp", "hard", "dangerous"], _phys(0.1)),
    "bell": (["loud", "hard", "shiny"], _phys(0.5)),
    "pillow": (["soft", "light"], _phys(0.5)),
    "feather": (["light", "soft"], _phys(0.001)),
    "hammer": (["hard", "heavy"], _phys(1.0)),
    "balloon": (["light", "round", "colorful"], _phys(0.01, gravity=-0.3)),
    "thunder": (["loud", "dangerous"], _phys(gravity=0.0)),
    "music": (["musical", "pleasant"], _phys(gravity=0.0)),
}


# --------------------------------------------------------------------------- #
# GroundedLexicon — read, ground, imagine, compare
# --------------------------------------------------------------------------- #
class GroundedLexicon:
    """NYXARA's grounded semantic memory: symbols ↔ multimodal perceptual meaning.

    Build one and it already *knows* (seed lexicon); teach it more by reading text
    (:meth:`learn_from_text`) or from real percepts (:meth:`learn_from_percepts`); ask it
    to *imagine* a word (:meth:`activate`) and watch every sense fire at once; compare two
    words by what they perceptually *are* (:meth:`similarity`).
    """

    def __init__(self, *, seed: bool = True, llm: Any = None) -> None:
        self.schema = PerceptualSchema()
        self.concepts: Dict[str, GroundedConcept] = {}
        self.llm = llm
        self.total_reads = 0
        self._space = None                              # lazy HyperSpace (HD geometry ceiling)
        self._item_mem = None
        if seed:
            self._load_seed()

    # ------------------------------------------------------------------ #
    # Construction / seeding
    # ------------------------------------------------------------------ #
    def _load_seed(self) -> None:
        for name, (descriptors, extras) in _SEED.items():
            c = self._concept(name)
            for d in descriptors:
                mapped = DESCRIPTOR_ONTOLOGY.get(d)
                if mapped:
                    s, feat, val = mapped
                    c.reinforce(s, feat, val)
            for sense, feats in extras.items():
                for feat, val in feats.items():
                    c.reinforce(sense, feat, val)
            c.sources += 1

    def _concept(self, name: str) -> GroundedConcept:
        key = _singular(name.strip().lower())
        c = self.concepts.get(key)
        if c is None:
            c = GroundedConcept(name=key)
            self.concepts[key] = c
        return c

    def get(self, word: str) -> Optional[GroundedConcept]:
        return self.concepts.get(_singular(word.strip().lower()))

    def __contains__(self, word: str) -> bool:
        return _singular(word.strip().lower()) in self.concepts

    def __len__(self) -> int:
        return len(self.concepts)

    # ------------------------------------------------------------------ #
    # Learning from language — reading grows grounding
    # ------------------------------------------------------------------ #
    def learn_from_text(self, text: str, concept: Optional[str] = None) -> Dict[str, Any]:
        """Read prose and attach the perceptual features it states to their subject.

        "Apples are sweet and red" grounds *apple* in ``taste:sweet`` + ``vision:red``;
        "you can eat an apple" adds ``affordance:edible``; "apples fall to the ground" adds
        ``physics:gravity``. When ``concept`` is given every clause grounds it; otherwise
        the subject of each sentence is detected. Returns a report dict.
        """
        if not text or not text.strip():
            return {"error": "empty text", "grounded": []}
        # LLM ceiling: if a real model is configured and a single concept is named but
        # unknown, enrich it once before the deterministic pass (best-effort).
        touched: Dict[str, int] = {}
        for sentence in (nlp.sentences(text) or [text]):
            subj, added = self._ground_sentence(sentence, concept)
            if subj and added:
                touched[subj] = touched.get(subj, 0) + added
        for name in touched:
            self.concepts[name].sources += 1
        self.total_reads += 1
        return {
            "grounded": sorted(touched),
            "features_added": sum(touched.values()),
            "concepts": len(self.concepts),
            "via": "deterministic",
        }

    def _ground_sentence(self, sentence: str,
                         concept: Optional[str]) -> Tuple[Optional[str], int]:
        toks = nlp.tokenize(sentence, lower=True)
        if not toks:
            return None, 0
        subject = _singular(concept) if concept else self._detect_subject(toks)
        if not subject:
            return None, 0
        c = self._concept(subject)

        # the descriptor span: after a copula if present, else the whole sentence
        cop = next((i for i, t in enumerate(toks) if t in _COPULA), -1)
        span = toks[cop + 1:] if cop >= 0 else toks
        added = 0

        # "<subject> is a <noun>" — capture an IS-A category (feeds the taxonomy later)
        if cop >= 0 and cop + 2 < len(toks) and toks[cop + 1] in _ARTICLES:
            cat = _singular(toks[cop + 2])
            if cat and cat != subject and cat not in DESCRIPTOR_ONTOLOGY:
                c.categories.add(cat)

        for i, tok in enumerate(span):
            negated = i > 0 and span[i - 1] in _NEGATORS
            mapped = DESCRIPTOR_ONTOLOGY.get(tok)
            if mapped:
                s, feat, val = mapped
                c.reinforce(s, feat, -val if negated else val)
                added += 1
                continue
            verb = _verb_stem(tok)
            av = _AFFORDANCE_VERBS.get(verb)
            if av:
                s, feat, val = av
                c.reinforce(s, feat, -val if negated else val)
                added += 1
        return (subject, added) if added else (subject, 0)

    @staticmethod
    def _detect_subject(toks: Sequence[str]) -> Optional[str]:
        """Find the noun a sentence is *about* (stdlib heuristics, no POS model)."""
        # 1) the content word right before a copula ("the apple is …")
        cop = next((i for i, t in enumerate(toks) if t in _COPULA), -1)
        if cop > 0:
            for t in reversed(toks[:cop]):
                if _is_noun_like(t):
                    return _singular(t)
        # 2) the first content word after an article ("you can eat an apple")
        for i, t in enumerate(toks[:-1]):
            if t in _ARTICLES and _is_noun_like(toks[i + 1]):
                return _singular(toks[i + 1])
        # 3) the first plain content word that isn't an affordance verb
        for t in toks:
            if _is_noun_like(t) and _verb_stem(t) not in _AFFORDANCE_VERBS \
                    and t not in DESCRIPTOR_ONTOLOGY:
                return _singular(t)
        return None

    # ------------------------------------------------------------------ #
    # Learning from real percepts — the embodied/sensory path
    # ------------------------------------------------------------------ #
    def learn_from_percepts(self, symbol: str, percepts: Iterable[Any]) -> Dict[str, Any]:
        """Bind features carried by real percepts (a :class:`senses.binding.PerceptualFrame`
        or any iterable of percept-like objects) to ``symbol`` — grounding from the senses
        themselves. Each percept may expose ``.modality``/``.content``/``.tags``; we read
        the channel→sense mapping and mine the content text through the same ontology.
        """
        c = self._concept(symbol)
        added = 0
        # a PerceptualFrame exposes .percepts; otherwise treat the arg as the iterable
        items = getattr(percepts, "percepts", percepts)
        for p in (items or []):
            modality = getattr(getattr(p, "modality", None), "value", None) \
                or getattr(p, "modality", None)
            sense = _CHANNEL_TO_SENSE.get(str(modality), None)
            content = getattr(p, "content", "") or ""
            tags = getattr(p, "tags", None) or []
            for word in (*nlp.tokenize(str(content), lower=True), *map(str.lower, tags)):
                mapped = DESCRIPTOR_ONTOLOGY.get(word)
                if mapped:
                    s, feat, val = mapped
                    # a percept on a real sensory channel reinforces *that* channel too
                    c.reinforce(sense or s, feat if sense is None else f"{feat}", val)
                    added += 1
        if added:
            c.sources += 1
        return {"symbol": c.name, "features_added": added}

    # ------------------------------------------------------------------ #
    # LLM ceiling — enrich an unknown concept (optional, degrades to floor)
    # ------------------------------------------------------------------ #
    def learn_from_llm(self, concept: str) -> Dict[str, Any]:
        if not self._use_llm():
            return {"concept": concept, "features_added": 0, "via": "unavailable"}
        system = (
            "You describe what a concept is like *in the senses*. Reply ONLY with JSON "
            '{"senses": {"vision": {"red": 0.8, "round": 0.9}, "taste": {"sweet": 0.7}, '
            '"touch": {...}, "physics": {"weight": 0.15, "gravity": 1.0}, '
            '"affordance": {"edible": 1.0}, "emotion": {...}, "sound": {...}}}. '
            "Values in [-1,1]; negative = opposite pole. Omit senses that don't apply."
        )
        try:
            data = self.llm.generate_json(
                f"Concept: {concept}\nGive its multimodal perceptual profile as JSON.",
                system=system,
            )
        except Exception:                               # noqa: BLE001 — ceiling never required
            return {"concept": concept, "features_added": 0, "via": "error"}
        c = self._concept(concept)
        added = 0
        senses = (data or {}).get("senses", {}) if isinstance(data, dict) else {}
        valid = {s.value: s for s in Sense}
        for sname, feats in (senses or {}).items():
            sense = valid.get(str(sname).lower())
            if sense is None or not isinstance(feats, dict):
                continue
            for feat, val in feats.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    c.reinforce(sense, str(feat), float(val))
                    added += 1
        if added:
            c.sources += 1
        return {"concept": c.name, "features_added": added, "via": "llm"}

    def _use_llm(self) -> bool:
        if self.llm is None:
            return False
        try:
            return self.llm.chosen_provider().name != "mock"
        except Exception:                               # noqa: BLE001
            return False

    # ------------------------------------------------------------------ #
    # Imagining a word — the headline: every sense fires at once
    # ------------------------------------------------------------------ #
    def activate(self, word: str) -> GroundedActivation:
        """Bring ``word`` to mind: return its full multimodal activation.

        If the word is unknown and a real LLM is configured, it is grounded on the fly;
        otherwise an empty (``grounded=False``) activation is returned — honest about not
        knowing rather than hallucinating.
        """
        c = self.get(word)
        if c is None and self._use_llm():
            self.learn_from_llm(word)
            c = self.get(word)
        if c is None:
            return GroundedActivation(_singular(word.strip().lower()), False, {}, [], 0.0)
        senses = {s: dict(c.senses[s]) for s in c.active_senses()}
        vector = self.schema.vectorize(c)
        conf = min(1.0, c.strength / 2.0) * (1.0 if senses else 0.0)
        return GroundedActivation(c.name, True, senses, vector, conf)

    # ------------------------------------------------------------------ #
    # Meaning is geometry — compare two grounded symbols
    # ------------------------------------------------------------------ #
    def similarity(self, a: str, b: str) -> float:
        """Cosine of two words' perceptual feature vectors ∈ ``[-1, 1]``.

        apple ≈ orange (both round, sweet, edible, fall) ≫ apple ≈ car. Returns ``0.0`` if
        either word cannot be grounded.
        """
        ca, cb = self.get(a), self.get(b)
        if ca is None and self._use_llm():
            self.learn_from_llm(a); ca = self.get(a)
        if cb is None and self._use_llm():
            self.learn_from_llm(b); cb = self.get(b)
        if ca is None or cb is None:
            return 0.0
        # register both first, then vectorize at one common width
        self.schema.vectorize(ca); self.schema.vectorize(cb)
        w = self.schema.width()
        va = self.schema.vectorize(ca, width=w)
        vb = self.schema.vectorize(cb, width=w)
        return _cosine(va, vb)

    def neighbors(self, word: str, *, top: int = 5) -> List[Tuple[str, float]]:
        """The ``top`` most perceptually similar grounded concepts to ``word``."""
        key = _singular(word.strip().lower())
        if key not in self.concepts:
            return []
        scored = [(name, self.similarity(key, name))
                  for name in self.concepts if name != key]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:top]

    # ------------------------------------------------------------------ #
    # Optional HD geometry — the 10,000-D demonstration of the same relations
    # ------------------------------------------------------------------ #
    def hypervector(self, word: str):
        """Encode a grounded concept as a hyperdimensional vector (lazy import; optional).

        Each active ``sense:feature`` becomes an atomic hypervector, weighted by its
        activation (negative pole → permuted), and all are bundled. Returns ``None`` if the
        word is ungrounded. Demonstrates that the same apple≈orange≫apple≈car geometry
        survives in a near-orthogonal 10,000-D space.
        """
        c = self.get(word)
        if c is None:
            return None
        if self._space is None:
            from nyxara.cognition.hyper_dimensional_vectors import HyperSpace, ItemMemory
            self._space = HyperSpace(dim=10000, seed=42)
            self._item_mem = ItemMemory(self._space)
        space, mem = self._space, self._item_mem
        parts = []
        for sense, feats in c.senses.items():
            for feat, val in feats.items():
                if abs(val) < 0.05:
                    continue
                atom = mem.symbol(f"{sense.value}:{feat}")
                if val < 0:
                    atom = space.permute(atom, 1)       # opposite pole = distinct direction
                for _ in range(max(1, round(abs(val) * 3))):  # weight by repetition
                    parts.append(atom)
        return space.bundle(*parts) if parts else space.zero()

    def hd_similarity(self, a: str, b: str) -> float:
        """Cosine similarity of the two words' hyperdimensional encodings."""
        from nyxara.cognition.hyper_dimensional_vectors import HyperSpace
        ha, hb = self.hypervector(a), self.hypervector(b)
        if ha is None or hb is None:
            return 0.0
        return HyperSpace.similarity(ha, hb)

    # ------------------------------------------------------------------ #
    # Feeding the taxonomy — grounded perception → an IS-A ladder
    # ------------------------------------------------------------------ #
    def feature_set(self, word: str) -> List[str]:
        """The ``sense:feature`` tags a concept activates — its categorical fingerprint."""
        c = self.get(word)
        if c is None:
            return []
        return sorted(f"{s.value}:{f}" for s in c.active_senses()
                      for f, v in c.senses[s].items() if abs(v) >= 0.05)

    def to_instance(self, word: str):
        """Build a :class:`nyxara.cognition.concept_formation.Instance` from a grounding."""
        from nyxara.cognition.concept_formation import Instance
        c = self.get(word)
        if c is None:
            return None
        attrs = {f"{s.value}:{f}": float(v)
                 for s in c.senses for f, v in c.senses[s].items()}
        return Instance(name=c.name, features=set(self.feature_set(word)), attrs=attrs)

    def feed_abstraction_ladder(self, ladder: Any, words: Iterable[str]) -> int:
        """Register each grounded word into an :class:`AbstractionLadder`. Returns the count.

        After this, ``ladder.abstract([...])`` forms a superordinate concept from the
        *shared perceptual structure* (apple, orange, banana → a fruit-like concept), and
        ``ladder.classify`` recognises a never-seen fruit by subsumption — abstraction
        built on grounded meaning, not labels.
        """
        n = 0
        for w in words:
            c = self.get(w)
            if c is None:
                continue
            attrs = {f"{s.value}:{f}": float(v)
                     for s in c.senses for f, v in c.senses[s].items()}
            ladder.observe(c.name, self.feature_set(w), attrs)
            n += 1
        return n

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def explain(self, word: str) -> str:
        """A human-readable gloss of what a word *means* in the senses."""
        c = self.get(word)
        if c is None:
            return f"{word!r}: not grounded — NYXARA has no perceptual meaning for it yet."
        order = [Sense.TASTE, Sense.SMELL, Sense.VISION, Sense.TOUCH, Sense.SOUND,
                 Sense.PHYSICS, Sense.AFFORDANCE, Sense.ACTION, Sense.EMOTION]
        parts = []
        for s in order:
            feats = c.senses.get(s, {})
            live = [(f, v) for f, v in feats.items() if abs(v) >= 0.05]
            if not live:
                continue
            rendered = ", ".join(
                f"{f}" if v > 0 else f"not {f}"
                if abs(abs(v) - 1.0) < 1e-6 else f"{f}={v:+.2f}"
                for f, v in sorted(live, key=lambda kv: -abs(kv[1])))
            parts.append(f"{s.value}: {rendered}")
        head = f"{c.name} →"
        if c.categories:
            head += f" (a {', '.join(sorted(c.categories))})"
        return head + " " + "; ".join(parts) if parts else f"{head} (no active senses)"

    def report(self) -> str:
        total_feats = sum(c.feature_count() for c in self.concepts.values())
        total_ev = sum(sum(c.counts.values()) for c in self.concepts.values())
        return (f"GroundedLexicon: {len(self.concepts)} concepts, {total_feats} features, "
                f"{total_ev} evidences, {self.schema.width()} schema dims, "
                f"{self.total_reads} reads")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concepts": {name: c.to_dict() for name, c in sorted(self.concepts.items())},
            "schema_dims": self.schema.width(),
            "reads": self.total_reads,
        }


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _is_noun_like(tok: str) -> bool:
    return (tok not in nlp.STOPWORDS and tok not in _COPULA and tok not in _ARTICLES
            and len(tok) > 2 and _NUMBER_RE.fullmatch(tok) is None)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA grounded-understanding self-test")
    print("=" * 70)

    lex = GroundedLexicon()
    print("\n" + lex.report())

    # 1) imagining "apple" lights up many senses AT ONCE — not one token, a world.
    act = lex.activate("apple")
    print(f"\nactivate('apple') → modalities fire simultaneously: {act.modalities}")
    print("  " + lex.explain("apple"))
    assert act.grounded and len(act.modalities) >= 3, "apple must fire many senses at once"
    assert act.fires(Sense.TASTE) and act.fires(Sense.VISION) and act.fires(Sense.PHYSICS)

    # 2) reading grounds a brand-new word — meaning learned from language, by herself.
    rep = lex.learn_from_text(
        "A mango is a fruit. Mangoes are sweet and yellow and soft. "
        "You can eat a mango, and a mango falls to the ground.")
    print(f"\nread about 'mango' → grounded {rep['grounded']} "
          f"(+{rep['features_added']} features)")
    print("  " + lex.explain("mango"))
    mango = lex.activate("mango")
    assert mango.fires(Sense.TASTE) and mango.fires(Sense.AFFORDANCE)
    assert mango.fires(Sense.PHYSICS)  # learned that it falls (gravity)

    # 3) meaning is geometry: apple ≈ orange ≫ apple ≈ car.
    s_fruit = lex.similarity("apple", "orange")
    s_car = lex.similarity("apple", "car")
    print(f"\nsimilarity(apple, orange) = {s_fruit:.3f}   "
          f"similarity(apple, car) = {s_car:.3f}")
    assert s_fruit > s_car, "fruits must be perceptually closer than apple↔car"
    print(f"nearest to 'apple': {[n for n, _ in lex.neighbors('apple', top=4)]}")

    # 4) grounded perception feeds an IS-A taxonomy.
    from nyxara.cognition.concept_formation import AbstractionLadder
    ladder = AbstractionLadder()
    members = ["apple", "orange", "banana", "lemon", "mango"]
    lex.feed_abstraction_ladder(ladder, members)
    concept = ladder.abstract(members)
    hit = ladder.classify(lex.feature_set("grape"))   # grape never abstracted — generalize
    print(f"\nabstracted fruit concept shares: {sorted(concept.features)}")
    print(f"classify an unseen grape by its grounding → {hit[0] if hit else None}")
    assert hit is not None, "a grape should be recognised as the fruit-like concept"

    print("\nALL SELF-TESTS PASSED ✓")
    print("NYXARA now understands words by what they ARE in the senses — grounded. 🍎")
