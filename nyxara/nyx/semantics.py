"""NYXARA · nyx/semantics.py — meaning as a vector, on a ladder she can name (🧩, NYX V.02, gap 1).

NYX V.01 had no distributed representation at all. Every node in the concept graph was a
lowercase string, and the measurement was blunt: in the shipped embedder,
``cos("car", "automobile") == 0.0`` while ``cos("car", "banana") == 0.27`` — a random noun was
*more* similar to "car" than its own synonym. That is not a weak semantic space; it is the
absence of one.

:class:`SemanticSpace` gives every concept a vector, from a three-rung ladder, and **every
answer names the rung it came from**:

============== ==================================================== ==================
rung            what it is                                           data it needs
============== ==================================================== ==================
``relational``  links she was actually taught or read — synonymy,     her own
                definition, appositive, transliteration identity
``distributional`` SGNS word vectors trained by
                :class:`~nyxara.memory.neural_embedder.SelfLearnedEmbedder`
                on **her own** corpus: memory writes, AURA events,
                conversation, ``knowledge/base``                      her own
``subword``     hashed character n-grams over the transliteration     none
                key, plus a small stem — script-agnostic, so
                Devanagari gets the same treatment as Latin
============== ==================================================== ==================

Honest framing (the rule this repo keeps, see ``docs/CAPABILITIES.md``):

* **Nothing ships.** No lexicon file, no downloaded weights, no pretrained model. That is a
  deliberate decision and it has a cost that is stated rather than hidden: **on day one
  ``similarity("car", "automobile")`` is near zero**, and it will stay near zero until she
  reads something that relates them or is told. What this module refuses to do is dress that
  ignorance up as knowledge.
* An unknown pair returns ``known=False``, not ``0.0``. Zero from an untrained space means
  *"I have no evidence"*, and reporting it as *"unrelated"* is the specific lie this file is
  built to avoid. Every :class:`Similarity` carries a ``grade`` — how much real evidence is
  behind the number — and a ``why`` in plain words.
* The ``subword`` rung measures **spelling**, not meaning. "cat"/"cats" score high there and
  deserve to; "car"/"automobile" score zero there and deserve to. It is graded low on purpose
  and never promoted above what it is.
* The distributional rung is real SGD on real text, but it is her text: a few hundred turns is
  a few hundred turns. ``stats()["trained"]`` and the vocabulary count say exactly how thin the
  evidence is.

Pure standard library (the embedder uses numpy when present, identically either way). Every
public method is fail-soft: on any error it degrades to a null result rather than breaking a
turn.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from nyxara.nyx.lingua import content_tokens, roman_key, script_of

__all__ = ["Similarity", "Relation", "SemanticSpace", "RUNGS"]

#: Strongest evidence first. A rung is only used when the one above it has nothing to say.
RUNGS = ("relational", "distributional", "subword")

_SUBWORD_DIM = 256
_NGRAM_MIN, _NGRAM_MAX = 3, 4
#: The whole word and its stem carry more signal than any single n-gram, and are weighted so.
_MORPH_WEIGHT = 3.0
#: Spelling can be *strong* evidence of morphology and is still never evidence of meaning.
#: This is the ceiling on what the bottom rung is ever allowed to certify.
_SUBWORD_MAX_GRADE = 0.4

# Suffixes stripped for the morphological feature. English is the only inflection this repo can
# claim to handle; the Devanagari list is the common nominal/verbal endings and is deliberately
# short. Getting this wrong costs a slightly weaker subword score, never a wrong relation.
_SUFFIX_LATIN = ("ations", "ation", "ingly", "ings", "ing", "edly", "ies", "ied", "ers",
                 "er", "est", "ely", "ly", "es", "ed", "s")
_SUFFIX_DEVA = ("ियों", "ाओं", "ाएं", "ों", "ें", "ीय", "ता", "ना", "ने", "नी", "ा", "े", "ी", "ो")


# --------------------------------------------------------------------------- #
# Relation patterns — the sentences that genuinely *relate* two words.
#
# Same discipline as `ground.defines`: a passing mention is not a definition. Each pattern
# below is a construction whose whole job is to say "these two words are the same thing".
# --------------------------------------------------------------------------- #
# A "word" for these patterns is a run of non-space, non-punctuation, non-digit characters.
# It is written as a negated set rather than as ``\w`` on purpose: ``\w`` excludes Unicode
# combining marks, so ``[^\W\d_]`` silently fails to match "गाड़ी" — the nukta and the matra are
# category ``Mn`` — which would have left the Devanagari patterns below quietly dead.
_NOT_WORD = r"\s\d,.;:!?()\[\]{}\"“”'’/\\|<>=+*&^%$#@~`"
_W = rf"([^{_NOT_WORD}]{{2,}})"

_SYNONYM_PATTERNS = (
    # English
    re.compile(rf"\b{_W}\s+(?:means|is\s+called|is\s+also\s+called|is\s+another\s+word\s+for|"
               rf"is\s+the\s+same\s+as|is\s+short\s+for|stands\s+for|aka)\s+{_W}", re.U | re.I),
    re.compile(rf"\b{_W}\s*,\s*(?:a|an|the)\s+{_W}\s*,", re.U | re.I),       # appositive
    re.compile(rf"\ba\s+{_W}\s*,\s*(?:a|an)\s+{_W}\b", re.U | re.I),
    # Hinglish
    re.compile(rf"\b{_W}\s+(?:ka|ki)\s+matlab\s+(?:hai\s+)?{_W}", re.U | re.I),
    re.compile(rf"\b{_W}\s+(?:yaani|yani|matlab)\s+{_W}", re.U | re.I),
    re.compile(rf"\b{_W}\s+ko\s+{_W}\s+(?:kehte|bolte)\s+hain?", re.U | re.I),
    # Devanagari
    re.compile(rf"\b{_W}\s+(?:का|की)\s+मतलब\s+(?:है\s+)?{_W}", re.U),
    re.compile(rf"\b{_W}\s+(?:यानी|अर्थात्?)\s+{_W}", re.U),
    re.compile(rf"\b{_W}\s+को\s+{_W}\s+कहते\s+हैं", re.U),
)

_ISA_PATTERNS = (
    re.compile(rf"\b{_W}\s+is\s+(?:a|an)\s+(?:kind\s+of\s+|type\s+of\s+|sort\s+of\s+)?{_W}",
               re.U | re.I),
    re.compile(rf"\b{_W}\s+(?:एक\s+)?(?:प्रकार\s+का\s+)?{_W}\s+है", re.U),
)


@dataclass
class Similarity:
    """One similarity judgement, with the rung it came from and how much evidence backs it.

    Unpacks as ``(score, rung, grade)`` so callers that only want the number can have it —
    but ``known`` is the field that matters: ``False`` means *she does not know*, which is a
    different statement from a score of zero.
    """

    a: str = ""
    b: str = ""
    score: float = 0.0
    rung: str = "none"
    grade: float = 0.0
    known: bool = False
    why: str = ""

    def __iter__(self) -> Iterator[Any]:
        yield self.score
        yield self.rung
        yield self.grade

    def to_dict(self) -> Dict[str, Any]:
        return {"a": self.a, "b": self.b, "score": round(self.score, 6), "rung": self.rung,
                "grade": round(self.grade, 4), "known": self.known, "why": self.why}


@dataclass
class Relation:
    """A learned link between two concepts — where it came from, and how much she trusts it."""

    a: str = ""
    b: str = ""
    kind: str = "synonym"        # synonym | is-a | translit | near
    weight: float = 0.5
    source: str = ""             # "taught", "read:<pattern>", "translit", "grounding"
    tick: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"a": self.a, "b": self.b, "kind": self.kind,
                "weight": round(self.weight, 4), "source": self.source, "tick": self.tick}


def _norm(label: Any) -> str:
    try:
        return str(label or "").strip().lower()
    except Exception:  # noqa: BLE001
        return ""


def _stem_of(word: str) -> str:
    """A short, script-aware suffix strip. Conservative: never shortens below three units."""
    try:
        w = str(word or "")
        if len(w) <= 3:
            return w
        suffixes = _SUFFIX_DEVA if script_of(w[0]) == "devanagari" else _SUFFIX_LATIN
        for suf in suffixes:
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                return w[: -len(suf)]
        return w
    except Exception:  # noqa: BLE001
        return str(word or "")


def _hash_bucket(feature: str, dim: int) -> Tuple[int, float]:
    """Signed feature hashing — deterministic across processes, unlike ``hash()``."""
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dim, (1.0 if (value >> 63) & 1 else -1.0)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    try:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = math.fsum(x * y for x, y in zip(a, b))
        na = math.sqrt(math.fsum(x * x for x in a))
        nb = math.sqrt(math.fsum(y * y for y in b))
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return max(-1.0, min(1.0, dot / (na * nb)))
    except Exception:  # noqa: BLE001
        return 0.0


class SemanticSpace:
    """Every concept gets a vector; every similarity says which rung it came from.

    The class owns three things and blends none of them silently: a relation store she filled
    herself, an SGNS embedder trained on her own corpus, and a subword fallback that always
    answers. What is *not* here is a shipped word list — see the module docstring.
    """

    def __init__(self, *, dim: int = 64, min_count: int = 3, grade_floor: float = 0.25,
                 train_budget_s: float = 0.5, train_every: int = 32, seed: int = 42,
                 max_relations: int = 8192, max_vocab: int = 8192,
                 embedder: Any = None) -> None:
        self.dim = max(8, int(dim))
        self.min_count = max(1, int(min_count))
        self.grade_floor = max(0.0, min(1.0, float(grade_floor)))
        self.train_budget_s = max(0.05, float(train_budget_s))
        self.train_every = max(1, int(train_every))
        self.max_relations = max(16, int(max_relations))
        self.max_vocab = max(64, int(max_vocab))
        self.seed = int(seed)

        self._relations: Dict[str, Dict[str, Relation]] = {}
        self._counts: Dict[str, int] = {}
        self._subword_cache: Dict[str, List[float]] = {}
        self._embedder = embedder if embedder is not None else self._build_embedder()

        self.texts = 0
        self.tokens = 0
        self.relations_learned = 0
        self.merges_suggested = 0
        self.tick = 0
        self._pending = 0
        self._last_train: Dict[str, Any] = {}

    def _build_embedder(self) -> Any:
        """The distributional rung. Optional: without it the ladder is two rungs, and says so."""
        try:
            from nyxara.memory.neural_embedder import SelfLearnedEmbedder
            return SelfLearnedEmbedder(latent_dim=self.dim, seed=self.seed,
                                       min_count=self.min_count,
                                       sgns_max_vocab=self.max_vocab)
        except Exception:  # noqa: BLE001 — she still has relations and spelling
            return None

    # ---- rung 3: subword ------------------------------------------------- #
    def subword_vector(self, label: str) -> List[float]:
        """Character n-grams over the transliteration key, plus the whole word and its stem.

        Script-agnostic by construction: the key is romanised first, so "गुरुत्वाकर्षण" and
        "gurutvakarshan" produce *the same* vector. This measures spelling. It is not meaning
        and is never reported as meaning.
        """
        label = _norm(label)
        if not label:
            return [0.0] * _SUBWORD_DIM
        cached = self._subword_cache.get(label)
        if cached is not None:
            return cached
        try:
            key = roman_key(label) or label
            vec = [0.0] * _SUBWORD_DIM
            features = [(f"w:{key}", _MORPH_WEIGHT), (f"m:{_stem_of(key)}", _MORPH_WEIGHT)]
            padded = f"#{key}#"
            for n in range(_NGRAM_MIN, _NGRAM_MAX + 1):
                for i in range(max(0, len(padded) - n + 1)):
                    features.append((f"g:{padded[i:i + n]}", 1.0))
            for feat, weight in features:
                idx, sign = _hash_bucket(feat, _SUBWORD_DIM)
                vec[idx] += sign * weight
            norm = math.sqrt(math.fsum(x * x for x in vec))
            if norm > 0.0:
                vec = [x / norm for x in vec]
            if len(self._subword_cache) < 4096:
                self._subword_cache[label] = vec
            return vec
        except Exception:  # noqa: BLE001
            return [0.0] * _SUBWORD_DIM

    @staticmethod
    def _same_spelling(a: str, b: str) -> bool:
        """One word written two ways — the transliteration bridge, before any learning."""
        try:
            ka, kb = roman_key(a), roman_key(b)
            return bool(ka) and len(ka) >= 3 and ka == kb
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _same_stem(a: str, b: str) -> bool:
        """Do two spellings reduce to one stem? Deliberately strict, to keep "part"/"party" out."""
        try:
            ka, kb = _stem_of(roman_key(a) or a), _stem_of(roman_key(b) or b)
            if not ka or not kb or len(ka) < 3 or len(kb) < 3:
                return False
            if ka == kb:
                return True
            # A silent final "e" or a plural the stemmer left on: "engin" vs "engine". Only
            # those two characters, so "part"/"party" stays two words.
            short, long_ = (ka, kb) if len(ka) < len(kb) else (kb, ka)
            return len(long_) - len(short) == 1 and long_.startswith(short) \
                and long_[-1] in ("e", "s")
        except Exception:  # noqa: BLE001
            return False

    # ---- rung 2: distributional ------------------------------------------ #
    def _learned_vector(self, label: str) -> Optional[List[float]]:
        """The SGNS vector for a word, or ``None`` when she has not read it enough."""
        try:
            emb = self._embedder
            if emb is None:
                return None
            if self._counts.get(label, 0) < self.min_count:
                return None
            from nyxara.memory.store import _stem
            vec = getattr(emb, "_w_in", {}).get(_stem(label))
            if vec is None:
                return None
            return [float(x) for x in vec]
        except Exception:  # noqa: BLE001
            return None

    def _distributional_grade(self, a: str, b: str) -> float:
        """How much reading is behind this pair — exposure, tempered by the space's own audit."""
        try:
            seen = min(self._counts.get(a, 0), self._counts.get(b, 0))
            exposure = min(1.0, seen / float(4 * self.min_count))
            audit = float(getattr(self._embedder, "semantic_grade", 0.0) or 0.0)
            # A space that has not yet separated anything on its own held-out pairs does not
            # get to certify a pair just because it saw the words often.
            return max(0.0, min(1.0, exposure * (0.4 + 0.6 * audit)))
        except Exception:  # noqa: BLE001
            return 0.0

    # ---- rung 1: relational ---------------------------------------------- #
    def teach(self, a: str, b: str, *, kind: str = "synonym", weight: float = 0.9,
              source: str = "taught") -> bool:
        """Record that two concepts are related. The strongest rung, because it is evidence."""
        try:
            a, b = _norm(a), _norm(b)
            if not a or not b or a == b:
                return False
            weight = max(0.0, min(1.0, float(weight)))
            rel = Relation(a=a, b=b, kind=str(kind), weight=weight, source=str(source),
                           tick=self.tick)
            self._put(rel)
            self._put(Relation(a=b, b=a, kind=rel.kind, weight=weight, source=rel.source,
                               tick=self.tick))
            self.relations_learned += 1
            return True
        except Exception:  # noqa: BLE001
            return False

    def _put(self, rel: Relation) -> None:
        bucket = self._relations.setdefault(rel.a, {})
        prior = bucket.get(rel.b)
        if prior is not None and prior.weight >= rel.weight and prior.kind == rel.kind:
            return
        bucket[rel.b] = rel
        self._enforce_relation_cap()

    def _enforce_relation_cap(self) -> None:
        total = sum(len(v) for v in self._relations.values())
        if total <= self.max_relations:
            return
        flat = [(r.weight, r.tick, a, b) for a, bucket in self._relations.items()
                for b, r in bucket.items()]
        flat.sort()
        for _, _, a, b in flat[: total - self.max_relations]:
            self._relations.get(a, {}).pop(b, None)

    def relations(self, label: str) -> List[Relation]:
        return list(self._relations.get(_norm(label), {}).values())

    def synonyms(self, label: str, *, min_weight: float = 0.5) -> List[Tuple[str, float]]:
        """Only the links that claim *sameness* — not "related to", not "is a kind of"."""
        try:
            out = [(r.b, r.weight) for r in self.relations(label)
                   if r.kind in ("synonym", "translit") and r.weight >= min_weight]
            out.sort(key=lambda p: -p[1])
            return out
        except Exception:  # noqa: BLE001
            return []

    def _relational_path(self, a: str, b: str, *, depth: int = 2) -> Tuple[float, str]:
        """Best weight-product path between two concepts, up to ``depth`` hops."""
        try:
            if a == b:
                return 1.0, "identical"
            best, best_kind = 0.0, ""
            frontier = {a: (1.0, "")}
            seen = {a}
            for _ in range(max(1, int(depth))):
                nxt: Dict[str, Tuple[float, str]] = {}
                for src, (energy, kind) in frontier.items():
                    for rel in self._relations.get(src, {}).values():
                        # "is-a" and "near" relate without equating; they attenuate hard.
                        factor = rel.weight if rel.kind in ("synonym", "translit") \
                            else rel.weight * 0.5
                        gain = energy * factor
                        if gain <= best and rel.b != b:
                            continue
                        if rel.b == b:
                            if gain > best:
                                best, best_kind = gain, kind or rel.kind
                            continue
                        if rel.b in seen:
                            continue
                        prior = nxt.get(rel.b)
                        if prior is None or gain > prior[0]:
                            nxt[rel.b] = (gain, kind or rel.kind)
                seen.update(nxt)
                frontier = nxt
                if not frontier:
                    break
            return best, best_kind
        except Exception:  # noqa: BLE001
            return 0.0, ""

    # ---- the question everything asks ------------------------------------ #
    def similarity(self, a: str, b: str) -> Similarity:
        """How close are two concepts — and on what evidence?

        Unpacks as ``(score, rung, grade)``. Read ``known`` before trusting ``score``: an
        untrained space answers ``known=False``, which means *I do not know*, not *unrelated*.
        """
        out = Similarity(a=_norm(a), b=_norm(b))
        try:
            a, b = out.a, out.b
            if not a or not b:
                out.why = "nothing to compare"
                return out
            if a == b:
                out.score, out.rung, out.grade, out.known = 1.0, "relational", 1.0, True
                out.why = "the same concept"
                return out

            weight, kind = self._relational_path(a, b)
            if weight > 0.0:
                out.score, out.rung, out.grade, out.known = weight, "relational", weight, True
                out.why = (f"linked as {kind or 'related'} — she was told this or read it"
                           if weight >= 0.5 else
                           f"linked as {kind or 'related'}, but only indirectly")
                return out

            va, vb = self._learned_vector(a), self._learned_vector(b)
            if va is not None and vb is not None:
                grade = self._distributional_grade(a, b)
                out.score = max(0.0, _cosine(va, vb))
                out.rung, out.grade = "distributional", grade
                out.known = grade >= self.grade_floor
                out.why = (f"from her own reading: seen {self._counts.get(a, 0)}× and "
                           f"{self._counts.get(b, 0)}×"
                           + ("" if out.known else " — too thin to trust yet"))
                return out

            out.score = max(0.0, _cosine(self.subword_vector(a), self.subword_vector(b)))
            out.rung = "subword"
            # Spelling similarity is evidence of *spelling*. It is capped even when it is
            # high, because "car"/"cart" is not a fact about meaning.
            out.grade = min(_SUBWORD_MAX_GRADE, out.score * _SUBWORD_MAX_GRADE)
            missing = [w for w in (a, b) if self._counts.get(w, 0) < self.min_count]
            out.why = ("spelling only — " + (
                f"she has not read {' or '.join(missing)} enough to place "
                f"{'them' if len(missing) > 1 else 'it'} in meaning"
                if missing else "no distributional evidence for this pair"))
            if self._same_spelling(a, b):
                out.score, out.grade = 1.0, _SUBWORD_MAX_GRADE
                out.why = ("the same word in two scripts — the transliteration bridge, "
                           "which is orthography, not meaning")
            elif self._same_stem(a, b):
                # One word in two inflections. That *is* something she can assert, and it is
                # still a claim about form: "cat"/"cats", never "car"/"automobile".
                out.score = max(out.score, 0.85)
                out.grade = _SUBWORD_MAX_GRADE
                out.why = "the same stem — a morphological variant, which is form, not meaning"
            out.known = out.grade >= self.grade_floor
            return out
        except Exception:  # noqa: BLE001 — a similarity question never breaks a turn
            return out

    def vector(self, label: str) -> List[float]:
        """The best representation she has for one concept, whatever rung it comes from."""
        try:
            label = _norm(label)
            learned = self._learned_vector(label)
            if learned is not None:
                return learned
            # Nothing learned for this spelling — but a proven synonym may have a vector, and
            # borrowing it is exactly what the relational rung is for.
            for other, _weight in self.synonyms(label):
                borrowed = self._learned_vector(other)
                if borrowed is not None:
                    return borrowed
            return self.subword_vector(label)
        except Exception:  # noqa: BLE001
            return []

    def vector_rung(self, label: str) -> str:
        """Which rung :meth:`vector` would answer from — so a caller can weigh it honestly."""
        try:
            label = _norm(label)
            if self._learned_vector(label) is not None:
                return "distributional"
            if any(self._learned_vector(o) is not None for o, _ in self.synonyms(label)):
                return "relational"
            return "subword"
        except Exception:  # noqa: BLE001
            return "subword"

    def similar(self, label: str, *, k: int = 8,
                candidates: Optional[Sequence[str]] = None) -> List[Tuple[str, float, str]]:
        """The ``k`` nearest concepts she can justify, as ``(label, score, rung)``."""
        try:
            label = _norm(label)
            if not label:
                return []
            pool = set(candidates or ())
            if not pool:
                pool = set(self._counts)
                pool.update(self._relations.get(label, {}))
            pool.discard(label)
            scored: List[Tuple[str, float, str]] = []
            for other in pool:
                sim = self.similarity(label, other)
                if sim.score > 0.0 and sim.known:
                    scored.append((other, sim.score, sim.rung))
            scored.sort(key=lambda t: -t[1])
            return scored[: max(0, int(k))]
        except Exception:  # noqa: BLE001
            return []

    # ---- learning --------------------------------------------------------- #
    def train_on(self, text: str, *, source: str = "", train: bool = True) -> Dict[str, Any]:
        """Fold one piece of her own experience into the space.

        Counts the content words, banks the text for SGNS, and reads any construction that
        genuinely *relates* two words. Training itself is budgeted and only runs every
        ``train_every`` texts — gradient descent is not a thought, and must not cost one.
        """
        out: Dict[str, Any] = {"tokens": 0, "relations": 0, "trained": False}
        try:
            text = str(text or "").strip()
            if not text:
                return out
            self.tick += 1
            self.texts += 1
            tokens = content_tokens(text, cap=128)
            for tok in tokens:
                self._counts[tok] = self._counts.get(tok, 0) + 1
            self.tokens += len(tokens)
            out["tokens"] = len(tokens)
            self._enforce_vocab_cap()

            if tokens and self._embedder is not None:
                try:
                    self._embedder.learn(" ".join(tokens))
                except Exception:  # noqa: BLE001 — banking corpus is best-effort
                    pass

            learned = self.read_relations(text, source=source or "read")
            out["relations"] = learned
            self._bridge_scripts(tokens)

            self._pending += 1
            if train and self._pending >= self.train_every:
                out["trained"] = bool(self.consolidate().get("trained"))
            return out
        except Exception:  # noqa: BLE001 — learning never breaks a turn
            return out

    def _enforce_vocab_cap(self) -> None:
        over = len(self._counts) - self.max_vocab
        if over <= 0:
            return
        weakest = sorted(self._counts.items(), key=lambda p: p[1])[:over]
        for word, _ in weakest:
            self._counts.pop(word, None)

    def read_relations(self, text: str, *, source: str = "read") -> int:
        """Find constructions that relate two words, and only those. Returns how many landed."""
        found = 0
        try:
            text = str(text or "")
            if not text.strip():
                return 0
            for pattern in _SYNONYM_PATTERNS:
                for match in pattern.finditer(text):
                    a, b = _norm(match.group(1)), _norm(match.group(2))
                    if self._plausible_pair(a, b) and self.teach(
                            a, b, kind="synonym", weight=0.85, source=source):
                        found += 1
            for pattern in _ISA_PATTERNS:
                for match in pattern.finditer(text):
                    a, b = _norm(match.group(1)), _norm(match.group(2))
                    if self._plausible_pair(a, b) and self.teach(
                            a, b, kind="is-a", weight=0.6, source=source):
                        found += 1
            return found
        except Exception:  # noqa: BLE001
            return found

    @staticmethod
    def _plausible_pair(a: str, b: str) -> bool:
        """Both sides must be content words. A pattern that caught grammar caught nothing."""
        try:
            from nyxara.nyx.lingua import is_function_word
            return bool(a) and bool(b) and a != b and len(a) > 1 and len(b) > 1 \
                and not is_function_word(a) and not is_function_word(b)
        except Exception:  # noqa: BLE001
            return False

    def _bridge_scripts(self, tokens: Sequence[str]) -> int:
        """Two spellings of one word, in different scripts, are the same concept.

        This is the transliteration bridge doing real work: meeting "गुरुत्वाकर्षण" after
        having met "gurutvakarshan" links them at the relational rung, so the vector one of
        them earned is available to the other.
        """
        made = 0
        try:
            for tok in tokens:
                key = roman_key(tok)
                if not key or len(key) < 3:
                    continue
                for other in self._counts:
                    if other == tok or roman_key(other) != key:
                        continue
                    if self.teach(tok, other, kind="translit", weight=0.95,
                                  source="translit"):
                        made += 1
            return made
        except Exception:  # noqa: BLE001
            return made

    def learn_from_grounding(self, ground: Any, *, max_pairs: int = 64) -> int:
        """Borrow structure from :mod:`nyxara.nyx.ground`: words whose senses actually overlap.

        A ``near`` relation, never a synonym — sharing perceptual features makes two words
        neighbours, not the same word, and conflating those is how a lexicon fills up with
        confident nonsense.
        """
        made = 0
        try:
            lexicon = getattr(ground, "_get", lambda: None)()
            entries = getattr(lexicon, "entries", None) or getattr(lexicon, "_entries", None)
            if not isinstance(entries, dict):
                return 0
            feats: Dict[str, set] = {}
            for word, entry in list(entries.items())[: 4 * max_pairs]:
                sensed = getattr(entry, "features", None) or getattr(entry, "senses", None)
                if isinstance(sensed, dict):
                    feats[_norm(word)] = {str(k) for k in sensed}
                elif isinstance(sensed, (list, tuple, set)):
                    feats[_norm(word)] = {str(k) for k in sensed}
            words = sorted(feats)
            for i, a in enumerate(words):
                for b in words[i + 1:]:
                    fa, fb = feats[a], feats[b]
                    if not fa or not fb:
                        continue
                    overlap = len(fa & fb) / float(len(fa | fb))
                    if overlap >= 0.6 and self.teach(a, b, kind="near", weight=overlap,
                                                     source="grounding"):
                        made += 1
                        if made >= max_pairs:
                            return made
            return made
        except Exception:  # noqa: BLE001 — grounding is a source, never a dependency
            return made

    def consolidate(self, *, budget_s: Optional[float] = None) -> Dict[str, Any]:
        """Spend a real gradient budget on her own corpus. Called from a maintenance beat."""
        out: Dict[str, Any] = {"trained": False, "reason": "no embedder"}
        try:
            if self._embedder is None:
                return out
            self._pending = 0
            t0 = time.perf_counter()
            got = self._embedder.train(budget_s=budget_s or self.train_budget_s)
            got = dict(got or {})
            got["duration_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            self._last_train = got
            return got
        except Exception:  # noqa: BLE001 — training never breaks a turn
            return out

    # ---- what the graph asks for ----------------------------------------- #
    def merge_candidates(self, labels: Sequence[str], *,
                         min_weight: float = 0.9) -> List[Tuple[str, str, float]]:
        """Pairs the relational rung says are *the same word*, strongly enough to fuse.

        Only ``synonym``/``translit`` links above ``min_weight`` qualify. Distributional
        closeness never does: "hot" and "cold" are distributionally adjacent and are not the
        same concept, and merging on that would be a structural error she cannot see.
        """
        out: List[Tuple[str, str, float]] = []
        try:
            known = {_norm(x) for x in labels}
            for a in sorted(known):
                for b, weight in self.synonyms(a, min_weight=min_weight):
                    if b in known and a < b:
                        out.append((a, b, weight))
            self.merges_suggested += len(out)
            return out
        except Exception:  # noqa: BLE001
            return out

    # ---- introspection ---------------------------------------------------- #
    def rungs_available(self) -> Dict[str, bool]:
        return {
            "relational": bool(self._relations),
            "distributional": bool(self._embedder is not None
                                   and getattr(self._embedder, "_w_in", None)),
            "subword": True,
        }

    def stats(self) -> Dict[str, Any]:
        try:
            emb = self._embedder
            trained = bool(getattr(emb, "_w_in", None))
            return {
                "texts": self.texts, "tokens": self.tokens,
                "vocabulary": len(self._counts),
                "relations": sum(len(v) for v in self._relations.values()) // 2,
                "relations_learned": self.relations_learned,
                "rungs": self.rungs_available(),
                "trained": trained,
                "semantic_grade": round(float(getattr(emb, "semantic_grade", 0.0) or 0.0), 4),
                "space_version": int(getattr(emb, "space_version", 0) or 0),
                "min_count": self.min_count, "grade_floor": self.grade_floor,
                "last_train": dict(self._last_train),
                # Said out loud, because it is the cost of shipping no data: until she reads,
                # the top two rungs are empty and she will say so on every pair.
                "note": ("no lexicon and no pretrained weights ship with this space; "
                         "everything above the subword rung is learned from her own text"),
            }
        except Exception:  # noqa: BLE001
            return {}

    def describe(self, a: str, b: str) -> str:
        """One plain sentence about a pair — what ``/nyx semantics`` prints."""
        sim = self.similarity(a, b)
        if not sim.known:
            return (f"I cannot honestly place '{sim.a}' next to '{sim.b}' yet — {sim.why}. "
                    f"That is ignorance, not a measurement of zero.")
        return (f"'{sim.a}' ~ '{sim.b}' = {sim.score:.2f} "
                f"[{sim.rung} rung, evidence {sim.grade:.2f}] — {sim.why}.")

    # ---- persistence ------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        try:
            out: Dict[str, Any] = {
                "texts": self.texts, "tokens": self.tokens, "tick": self.tick,
                "relations_learned": self.relations_learned,
                "counts": dict(self._counts),
                "relations": [r.to_dict() for bucket in self._relations.values()
                              for r in bucket.values()],
            }
            if self._embedder is not None:
                try:
                    out["embedder"] = self._embedder.to_dict()
                except Exception:  # noqa: BLE001 — a space that cannot serialise is not fatal
                    pass
            return out
        except Exception:  # noqa: BLE001
            return {}

    def load_dict(self, data: Any) -> bool:
        try:
            if not isinstance(data, dict):
                return False
            self.texts = int(data.get("texts", 0))
            self.tokens = int(data.get("tokens", 0))
            self.tick = int(data.get("tick", 0))
            self.relations_learned = int(data.get("relations_learned", 0))
            self._counts = {str(k): int(v) for k, v in (data.get("counts") or {}).items()}
            self._relations = {}
            for raw in data.get("relations", []) or []:
                if not isinstance(raw, dict):
                    continue
                rel = Relation(a=_norm(raw.get("a")), b=_norm(raw.get("b")),
                               kind=str(raw.get("kind", "synonym")),
                               weight=float(raw.get("weight", 0.5)),
                               source=str(raw.get("source", "")),
                               tick=int(raw.get("tick", 0)))
                if rel.a and rel.b:
                    self._relations.setdefault(rel.a, {})[rel.b] = rel
            if data.get("embedder") and self._embedder is not None:
                try:
                    self._embedder.load_dict(data["embedder"])
                except Exception:  # noqa: BLE001
                    pass
            self._subword_cache.clear()
            return True
        except Exception:  # noqa: BLE001 — a corrupt sidecar must never block boot
            return False
