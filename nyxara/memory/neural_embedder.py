"""NYXARA · memory/neural_embedder.py — HER OWN gradient-learned embedding space.

The dependency-free embedders in :mod:`memory.store` climb a ladder: feature hashing →
curated lexical-semantics → random-indexing co-occurrence (:class:`LearnedEmbedder`).
This module adds the top rung that ladder was built for: **a genuinely trained neural
embedding space that NYXARA learns herself, from her own lived experience** — no
external LLM, no downloaded weights, no cloud. Three learned layers stack:

1. the inherited random-indexing co-occurrence floor (incremental, always on);
2. **SGNS** — skip-gram with negative sampling (word2vec's objective) trained by real
   SGD over the experience corpus NYXARA accumulates as she reads and converses, so
   words used alike genuinely converge in a latent space (default 64-d);
3. a **contrastive sentence head** — a linear projection over mean SGNS vectors,
   trained InfoNCE-style on positive pairs NYXARA *generates herself*: (stimulus,
   reply) of successful turns, (episode, consolidated gist), (recall query, memory
   that actually helped), and adjacent sentences of one document.

The final embedding is ``normalize(lexical_base + semantic_scale · up(sentence_vec))``
where ``up`` is a fixed seeded sparse projection latent→store-dim, so the vector-index
dimension never changes; a shifted space is handled by ``space_version`` + the store's
``reembed_stale``. Cold (nothing learned), the output is byte-identical to
:class:`~nyxara.memory.store.LexicalSemanticEmbedder`, so every recall threshold
calibrated for that space is preserved; ``semantic_scale`` rises only as the learned
space *measurably* separates paraphrases from noise (``semantic_grade``, self-audited
on held-out pairs). Pure standard library; numpy accelerates when present.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.memory.store import LearnedEmbedder, _STOPWORDS, _stem

try:  # optional acceleration — identical math either way
    import numpy as _np  # type: ignore
    _HAS_NP = True
except Exception:  # pragma: no cover
    _np = None  # type: ignore
    _HAS_NP = False

__all__ = ["SelfLearnedEmbedder"]

_SENT_SPLIT = re.compile(r"[.!?\n]+")

#: semantic_grade above which the space counts as genuinely learned (``is_lexical``
#: flips False and the full recall floor applies). Conservative on purpose: a barely
#: separated space must keep the forgiving lexical calibration.
_LEXICAL_FLIP_GRADE = 0.5


# --------------------------------------------------------------------------- #
# small vector helpers — one code path, numpy-fast when available
# --------------------------------------------------------------------------- #
def _vnew(n: int) -> Any:
    return _np.zeros(n, dtype=float) if _HAS_NP else [0.0] * n


def _vrand(n: int, rng: random.Random, scale: float) -> Any:
    vals = [rng.uniform(-scale, scale) for _ in range(n)]
    return _np.asarray(vals, dtype=float) if _HAS_NP else vals


def _vdot(a: Any, b: Any) -> float:
    if _HAS_NP:
        return float(_np.dot(a, b))
    return sum(x * y for x, y in zip(a, b))


def _vaxpy(a: Any, alpha: float, b: Any) -> None:
    """a += alpha * b, in place."""
    if _HAS_NP:
        a += alpha * b
    else:
        for i in range(len(a)):
            a[i] += alpha * b[i]


def _vnorm(a: Any) -> float:
    return math.sqrt(_vdot(a, a))


def _vlist(a: Any) -> List[float]:
    return [round(float(x), 5) for x in a]


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(30.0, x)))
    z = math.exp(max(-30.0, x))
    return z / (1.0 + z)


class SelfLearnedEmbedder(LearnedEmbedder):
    """A self-trained neural embedder: SGNS word vectors + a contrastive sentence head.

    Strict superset of :class:`LearnedEmbedder` — everything the parent does (lexical
    base, curated concepts, incremental co-occurrence) still happens; the gradient-
    learned component is blended on top, gated by an honest self-measured
    ``semantic_grade``. All training data is NYXARA's own experience, fed through
    :meth:`learn` (texts) and :meth:`learn_pair` (things that belong together);
    :meth:`train` spends a time budget of real SGD whenever she consolidates or dreams.
    """

    def __init__(self, dim: int = 128, seed: int = 1469598103934665603, *,
                 latent_dim: int = 64, negatives: int = 5, window: int = 2,
                 corpus_cap: int = 20000, pair_cap: int = 4000,
                 sgns_max_vocab: int = 8000, min_count: int = 2,
                 lr: float = 0.05, **kw: Any) -> None:
        super().__init__(dim=dim, seed=seed, window=window, **kw)
        self.latent_dim = max(8, int(latent_dim))
        self.negatives = max(1, int(negatives))
        self.corpus_cap = max(256, int(corpus_cap))
        self.pair_cap = max(64, int(pair_cap))
        self.sgns_max_vocab = max(256, int(sgns_max_vocab))
        self.min_count = max(1, int(min_count))
        self.lr = float(lr)
        # learned state
        self._w_in: Dict[str, Any] = {}         # word -> latent vector (the embedding)
        self._w_out: Dict[str, Any] = {}        # word -> latent vector (SGNS context side)
        self._proj: Any = self._eye()           # contrastive sentence head (latent×latent)
        self._corpus: List[str] = []            # her lived experience, the training set
        self._pairs: List[Tuple[str, str, float]] = []   # (a, b, weight>0 pos / <0 neg)
        #: bumped by every completed training pass — stale stored vectors re-embed lazily
        self.space_version: int = 0
        #: honest self-audit ∈ [0,1]: how well the learned space separates true pairs
        #: from random ones. Drives the blend weight and the recall-floor grade.
        self.semantic_grade: float = 0.0
        self.trained_steps: int = 0
        self._rng = random.Random(seed & 0xFFFFFFFF)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # gating: lexical until the learned space has PROVEN itself
    # ------------------------------------------------------------------ #
    @property
    def is_lexical(self) -> bool:  # type: ignore[override]
        return self.semantic_grade < _LEXICAL_FLIP_GRADE

    @property
    def semantic_scale(self) -> float:
        """Blend weight of the learned component — 0 cold, →1 as the space earns trust."""
        return min(1.0, 1.5 * max(0.0, self.semantic_grade))

    @property
    def experience_size(self) -> int:
        return len(self._corpus)

    # ------------------------------------------------------------------ #
    # experience intake — this is how SHE gathers her own training data
    # ------------------------------------------------------------------ #
    def learn(self, *texts: str) -> None:
        """Fold text into the co-occurrence floor AND bank it as SGNS training corpus."""
        super().learn(*texts)
        with self._lock:
            for text in texts:
                t = str(text).strip()
                if len(t) >= 4:
                    self._corpus.append(t[:2000])
            if len(self._corpus) > self.corpus_cap:
                del self._corpus[: len(self._corpus) - self.corpus_cap]

    def learn_pair(self, a: str, b: str, *, positive: bool = True,
                   weight: float = 1.0) -> None:
        """Record two texts that belong together (or apart) — contrastive supervision
        NYXARA mined from her own experience, never from an external teacher."""
        a, b = str(a or "").strip(), str(b or "").strip()
        if len(a) < 4 or len(b) < 4 or a == b:
            return
        w = abs(float(weight)) * (1.0 if positive else -1.0)
        with self._lock:
            self._pairs.append((a[:2000], b[:2000], w))
            if len(self._pairs) > self.pair_cap:
                del self._pairs[: len(self._pairs) - self.pair_cap]

    # ------------------------------------------------------------------ #
    # tokenisation shared with the parent (stemmed content words)
    # ------------------------------------------------------------------ #
    def _content_words(self, text: str) -> List[str]:
        return [_stem(w) for w in self._words(str(text))
                if w not in _STOPWORDS and len(w) > 1]

    # ------------------------------------------------------------------ #
    # the learned forward pass
    # ------------------------------------------------------------------ #
    def _sentence_latent(self, text: str) -> Optional[Any]:
        """Projected mean of learned word vectors — None when nothing is known yet."""
        acc = None
        seen = 0
        for w in self._content_words(text):
            vec = self._w_in.get(w)
            if vec is None:
                continue
            if acc is None:
                acc = _vnew(self.latent_dim)
            _vaxpy(acc, 1.0, vec)
            seen += 1
        if acc is None or seen == 0:
            return None
        n = _vnorm(acc)
        if n <= 1e-9:
            return None
        if _HAS_NP:
            acc = acc / n
        else:
            acc = [x / n for x in acc]
        out = self._matvec(self._proj, acc)
        n2 = _vnorm(out)
        if n2 <= 1e-9:
            return None
        if _HAS_NP:
            return out / n2
        return [x / n2 for x in out]

    def _matvec(self, m: Any, v: Any) -> Any:
        if _HAS_NP:
            return m @ v
        return [sum(m[i][j] * v[j] for j in range(self.latent_dim))
                for i in range(self.latent_dim)]

    def _eye(self) -> Any:
        if _HAS_NP:
            return _np.eye(self.latent_dim, dtype=float)
        return [[1.0 if i == j else 0.0 for j in range(self.latent_dim)]
                for i in range(self.latent_dim)]

    def _up(self, latent: Any) -> List[float]:
        """Fixed seeded sparse projection latent→store dim: the learned signal lands in
        the same vector space the index already holds, so dims never change."""
        out = [0.0] * self.dim
        for j in range(self.latent_dim):
            x = float(latent[j])
            if x == 0.0:
                continue
            h = self._hash(f"↑{j}")
            for _ in range(4):
                h = (h * 1099511628211 + 0x9E3779B9) & 0xFFFFFFFFFFFFFFFF
                idx = h % self.dim
                sign = 1.0 if (h >> 5) & 1 else -1.0
                out[idx] += sign * x
        n = math.sqrt(sum(v * v for v in out)) or 1.0
        return [v / n for v in out]

    def embed(self, text: str) -> List[float]:
        base = super().embed(text)
        scale = self.semantic_scale
        if scale <= 0.0 or not self._w_in:
            return base
        latent = self._sentence_latent(text)
        if latent is None:
            return base
        upv = self._up(latent)
        combined = [base[k] + scale * upv[k] for k in range(self.dim)]
        norm = math.sqrt(sum(v * v for v in combined)) or 1.0
        return [v / norm for v in combined]

    def latent_similarity(self, a: str, b: str) -> Optional[float]:
        """Cosine of the purely-learned sentence vectors (None before any learning)."""
        va, vb = self._sentence_latent(a), self._sentence_latent(b)
        if va is None or vb is None:
            return None
        return _vdot(va, vb)

    # ------------------------------------------------------------------ #
    # training — real SGD on HER data, on a time budget
    # ------------------------------------------------------------------ #
    def train(self, budget_s: float = 2.0) -> Dict[str, Any]:
        """One budgeted training pass: SGNS over the experience corpus, then the
        contrastive sentence head over the mined pairs. Bumps ``space_version`` and
        re-audits ``semantic_grade``. Safe to call from any maintenance loop."""
        t0 = time.monotonic()
        budget = max(0.05, float(budget_s))
        with self._lock:
            corpus = list(self._corpus)
            pairs = list(self._pairs)
        if not corpus:
            return {"trained": False, "reason": "no experience yet"}
        counts = self._count_vocab(corpus)
        vocab = self._select_vocab(counts)
        if not vocab:
            return {"trained": False, "reason": "vocabulary too thin"}
        self._ensure_vectors(vocab)
        neg_words, neg_cum = self._negative_table(counts, vocab)
        sgns_steps = self._train_sgns(corpus, vocab, neg_words, neg_cum,
                                      deadline=t0 + budget * 0.6)
        contrastive_steps = self._train_contrastive(pairs, corpus,
                                                    deadline=t0 + budget)
        grade = self._audit_grade(pairs, corpus)
        with self._lock:
            if grade is not None:
                # smoothed, honest: the grade can fall if the space degrades
                self.semantic_grade = max(0.0, min(
                    1.0, 0.7 * self.semantic_grade + 0.3 * grade))
            if sgns_steps or contrastive_steps:
                self.space_version += 1
                self.trained_steps += sgns_steps + contrastive_steps
        return {
            "trained": bool(sgns_steps or contrastive_steps),
            "sgns_steps": sgns_steps,
            "contrastive_steps": contrastive_steps,
            "vocab": len(vocab),
            "corpus": len(corpus),
            "pairs": len(pairs),
            "semantic_grade": round(self.semantic_grade, 4),
            "semantic_scale": round(self.semantic_scale, 4),
            "space_version": self.space_version,
            "duration_s": round(time.monotonic() - t0, 3),
        }

    def _count_vocab(self, corpus: Sequence[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for text in corpus:
            for w in self._content_words(text):
                counts[w] = counts.get(w, 0) + 1
        return counts

    def _select_vocab(self, counts: Dict[str, int]) -> List[str]:
        eligible = [w for w, c in counts.items() if c >= self.min_count]
        eligible.sort(key=lambda w: counts[w], reverse=True)
        return eligible[: self.sgns_max_vocab]

    def _ensure_vectors(self, vocab: Sequence[str]) -> None:
        scale = 0.5 / self.latent_dim
        for w in vocab:
            if w not in self._w_in:
                self._w_in[w] = _vrand(self.latent_dim, self._rng, scale)
            if w not in self._w_out:
                self._w_out[w] = _vnew(self.latent_dim)

    def _negative_table(self, counts: Dict[str, int],
                        vocab: Sequence[str]) -> Tuple[List[str], List[float]]:
        """Unigram^0.75 sampling table (cumulative weights + bisect)."""
        words: List[str] = []
        cum: List[float] = []
        total = 0.0
        for w in vocab:
            total += counts[w] ** 0.75
            words.append(w)
            cum.append(total)
        return words, cum

    def _sample_negative(self, neg_words: List[str], neg_cum: List[float]) -> str:
        import bisect
        x = self._rng.uniform(0.0, neg_cum[-1])
        return neg_words[bisect.bisect_left(neg_cum, x)]

    def _train_sgns(self, corpus: List[str], vocab: Sequence[str],
                    neg_words: List[str], neg_cum: List[float], *,
                    deadline: float) -> int:
        """Skip-gram with negative sampling — the word2vec objective, by hand."""
        vocab_set = set(vocab)
        steps = 0
        lr = self.lr
        while time.monotonic() < deadline:
            text = corpus[self._rng.randrange(len(corpus))]
            words = [w for w in self._content_words(text) if w in vocab_set]
            n = len(words)
            if n < 2:
                continue
            for i, center in enumerate(words):
                v_in = self._w_in[center]
                lo, hi = max(0, i - self.window), min(n, i + self.window + 1)
                for j in range(lo, hi):
                    if j == i:
                        continue
                    # positive: the observed (center, context) co-occurrence
                    self._sgns_update(v_in, self._w_out[words[j]], 1.0, lr)
                    # negatives: words that did NOT co-occur here
                    for _ in range(self.negatives):
                        neg = self._sample_negative(neg_words, neg_cum)
                        if neg == words[j]:
                            continue
                        self._sgns_update(v_in, self._w_out[neg], 0.0, lr)
                    steps += 1
                if steps % 64 == 0 and time.monotonic() >= deadline:
                    return steps
        return steps

    @staticmethod
    def _sgns_update(v_in: Any, v_out: Any, label: float, lr: float) -> None:
        g = lr * (label - _sigmoid(_vdot(v_in, v_out)))
        if _HAS_NP:
            d_in = g * v_out
            v_out += g * v_in
            v_in += d_in
        else:
            for k in range(len(v_in)):
                d_in = g * v_out[k]
                v_out[k] += g * v_in[k]
                v_in[k] += d_in

    # ---- contrastive sentence head ---- #
    def _mined_pairs(self, corpus: List[str], want: int) -> List[Tuple[str, str, float]]:
        """Self-supervision with zero labels: adjacent sentences of one text cohere."""
        out: List[Tuple[str, str, float]] = []
        tries = 0
        while len(out) < want and tries < want * 8 and corpus:
            tries += 1
            text = corpus[self._rng.randrange(len(corpus))]
            sents = [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) >= 20]
            if len(sents) < 2:
                continue
            i = self._rng.randrange(len(sents) - 1)
            out.append((sents[i], sents[i + 1], 0.5))
        return out

    def _mean_latent(self, text: str) -> Optional[Any]:
        """Unprojected normalised mean word vector (input side of the head)."""
        acc = None
        for w in self._content_words(text):
            vec = self._w_in.get(w)
            if vec is None:
                continue
            if acc is None:
                acc = _vnew(self.latent_dim)
            _vaxpy(acc, 1.0, vec)
        if acc is None:
            return None
        n = _vnorm(acc)
        if n <= 1e-9:
            return None
        if _HAS_NP:
            return acc / n
        return [x / n for x in acc]

    def _train_contrastive(self, pairs: List[Tuple[str, str, float]],
                           corpus: List[str], *, deadline: float) -> int:
        """Pull true pairs together, push random in-batch pairs apart — through P."""
        train_pairs = [p for i, p in enumerate(pairs) if i % 5 != 0]  # 20% held out
        train_pairs.extend(self._mined_pairs(corpus, max(8, len(train_pairs) // 2)))
        if not train_pairs:
            return 0
        steps = 0
        lr = 0.02
        while time.monotonic() < deadline:
            a, b, w = train_pairs[self._rng.randrange(len(train_pairs))]
            va, vb = self._mean_latent(a), self._mean_latent(b)
            if va is None or vb is None:
                steps += 1
                if steps > len(train_pairs) * 4:  # nothing embeddable yet
                    return 0
                continue
            sign = 1.0 if w > 0 else -1.0
            self._proj_cos_step(va, vb, sign * abs(w) * lr)
            # one in-batch negative: a random other text should NOT cohere
            other = train_pairs[self._rng.randrange(len(train_pairs))]
            vn = self._mean_latent(other[1])
            if vn is not None and sign > 0:
                self._proj_cos_step(va, vn, -0.5 * abs(w) * lr)
            steps += 1
            if steps % 16 == 0 and time.monotonic() >= deadline:
                break
        return steps

    def _proj_cos_step(self, va: Any, vb: Any, step: float) -> None:
        """Gradient step on cos(P·va, P·vb) w.r.t. P (ascent for step>0)."""
        u = self._matvec(self._proj, va)
        v = self._matvec(self._proj, vb)
        nu, nv = _vnorm(u), _vnorm(v)
        if nu <= 1e-9 or nv <= 1e-9:
            return
        cos = _vdot(u, v) / (nu * nv)
        if _HAS_NP:
            du = v / (nu * nv) - (cos / (nu * nu)) * u
            dv = u / (nu * nv) - (cos / (nv * nv)) * v
            self._proj += step * (_np.outer(du, va) + _np.outer(dv, vb))
        else:
            d = self.latent_dim
            du = [v[i] / (nu * nv) - (cos / (nu * nu)) * u[i] for i in range(d)]
            dv = [u[i] / (nu * nv) - (cos / (nv * nv)) * v[i] for i in range(d)]
            proj = self._proj
            for i in range(d):
                row = proj[i]
                si, ti = step * du[i], step * dv[i]
                for j in range(d):
                    row[j] += si * va[j] + ti * vb[j]

    # ---- the honest self-audit ---- #
    def _audit_grade(self, pairs: List[Tuple[str, str, float]],
                     corpus: List[str]) -> Optional[float]:
        """Separation score on held-out pairs: cos(true pair) − cos(random pair).

        This is what keeps the upgrade REAL: the learned signal is only blended into
        recall in proportion to measured performance on data it did not train on."""
        held = [p for i, p in enumerate(pairs) if i % 5 == 0 and p[2] > 0]
        if len(held) < 4:
            held = self._mined_pairs(corpus, 12)
        pos: List[float] = []
        rnd: List[float] = []
        for a, b, _w in held[:32]:
            s = self.latent_similarity(a, b)
            if s is None:
                continue
            pos.append(s)
            other = corpus[self._rng.randrange(len(corpus))]
            r = self.latent_similarity(a, other)
            if r is not None:
                rnd.append(r)
        if len(pos) < 4 or len(rnd) < 4:
            return None
        margin = (sum(pos) / len(pos)) - (sum(rnd) / len(rnd))
        return max(0.0, min(1.0, margin / 0.5))

    # ------------------------------------------------------------------ #
    # persistence — the learned space survives restarts (Rule 7)
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "kind": "self-learned",
                "dim": self.dim,
                "latent_dim": self.latent_dim,
                "space_version": self.space_version,
                "semantic_grade": round(self.semantic_grade, 6),
                "trained_steps": self.trained_steps,
                "context": {w: _vlist(v) for w, v in
                            list(self._context.items())[: self.sgns_max_vocab]},
                "w_in": {w: _vlist(v) for w, v in self._w_in.items()},
                "w_out": {w: _vlist(v) for w, v in self._w_out.items()},
                "proj": [_vlist(row) for row in (self._proj if not _HAS_NP
                                                 else list(self._proj))],
                "corpus_tail": self._corpus[-4000:],
                "pairs_tail": [list(p) for p in self._pairs[-2000:]],
            }

    def load_dict(self, data: Dict[str, Any]) -> bool:
        if not isinstance(data, dict) or data.get("kind") != "self-learned":
            return False
        if int(data.get("dim", -1)) != self.dim or \
                int(data.get("latent_dim", -1)) != self.latent_dim:
            return False  # geometry changed — start fresh rather than corrupt the space
        with self._lock:
            self.space_version = int(data.get("space_version", 0))
            self.semantic_grade = float(data.get("semantic_grade", 0.0))
            self.trained_steps = int(data.get("trained_steps", 0))
            self._context = {w: [float(x) for x in v]
                             for w, v in (data.get("context") or {}).items()}
            def _restore(vecs: Dict[str, Any]) -> Dict[str, Any]:
                out: Dict[str, Any] = {}
                for w, v in (vecs or {}).items():
                    vals = [float(x) for x in v]
                    if len(vals) != self.latent_dim:
                        continue
                    out[w] = _np.asarray(vals, dtype=float) if _HAS_NP else vals
                return out
            self._w_in = _restore(data.get("w_in") or {})
            self._w_out = _restore(data.get("w_out") or {})
            proj = data.get("proj")
            if proj and len(proj) == self.latent_dim and \
                    all(len(r) == self.latent_dim for r in proj):
                rows = [[float(x) for x in r] for r in proj]
                self._proj = _np.asarray(rows, dtype=float) if _HAS_NP else rows
            self._corpus = [str(t) for t in (data.get("corpus_tail") or [])]
            self._pairs = [(str(p[0]), str(p[1]), float(p[2]))
                           for p in (data.get("pairs_tail") or [])
                           if isinstance(p, (list, tuple)) and len(p) == 3
                           ][-self.pair_cap:]
        return True

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)
        os.replace(tmp, path)
        return path

    def load(self, path: str) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return self.load_dict(json.load(f))
        except (OSError, ValueError):
            return False
