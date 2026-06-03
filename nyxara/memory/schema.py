"""NYXARA · memory/schema.py — episodes → reusable mental templates (✦).

A mind that has seen a hundred logins learns the *idea* of "a login": who, where,
when, and how it usually ends. This module performs **schema induction** — the
cognitive move from many concrete :class:`episodes` to an abstract, reusable
**schema** (a "script", in Schank & Abelson's sense).

Each schema has typed **slots** with:

* a **fill-rate** (how often the slot appears),
* a **modal default** (its most typical value) with probability,
* a **variability** (normalised entropy — is it constant or wild?),
* numeric summaries when the values are numbers.

What schemas give NYXARA:

* **matching** — score how well a new episode fits a known schema,
* **prediction / slot-filling** — guess the missing pieces of a partial episode from
  the template ("an external night login on an unknown port → probably blocked"),
* **incremental learning** — integrate a new episode into the best-matching schema
  (or spawn a new one), so templates sharpen with experience.

Pure standard library; reuses :class:`~nyxara.memory.store.HashingEmbedder` for the
optional prototype (centroid) used in unsupervised clustering.
"""

from __future__ import annotations

import math
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.memory.store import HashingEmbedder, MemoryRecord

__all__ = [
    "Slot",
    "Schema",
    "FilledEpisode",
    "SchemaLibrary",
    "episode_from_record",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _key(v: Any) -> str:
    try:
        return repr(v)
    except Exception:  # pragma: no cover
        return str(v)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


# --------------------------------------------------------------------------- #
# Slot — the distribution of a single attribute across episodes
# --------------------------------------------------------------------------- #
@dataclass
class Slot:
    name: str
    _counts: Counter = field(default_factory=Counter)      # value-key -> count
    _values: Dict[str, Any] = field(default_factory=dict)  # value-key -> actual value
    present_count: int = 0
    _numeric: List[float] = field(default_factory=list)

    def observe(self, value: Any) -> None:
        k = _key(value)
        self._counts[k] += 1
        self._values[k] = value
        self.present_count += 1
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            self._numeric.append(float(value))

    def fill_rate(self, n_episodes: int) -> float:
        return self.present_count / n_episodes if n_episodes else 0.0

    def modal(self) -> Tuple[Any, float]:
        """Most typical value and its probability among observations."""
        if not self._counts:
            return (None, 0.0)
        k, c = self._counts.most_common(1)[0]
        return (self._values[k], c / self.present_count)

    def distinct(self) -> int:
        return len(self._counts)

    def variability(self) -> float:
        """Normalised Shannon entropy in [0,1]: 0 == constant, 1 == maximally varied."""
        if self.present_count == 0 or len(self._counts) <= 1:
            return 0.0
        total = sum(self._counts.values())
        h = -sum((c / total) * math.log(c / total) for c in self._counts.values())
        return _clamp(h / math.log(len(self._counts)))

    def is_constant(self) -> bool:
        return len(self._counts) == 1

    def numeric_stats(self) -> Optional[Dict[str, float]]:
        if not self._numeric:
            return None
        n = len(self._numeric)
        mean = sum(self._numeric) / n
        var = sum((x - mean) ** 2 for x in self._numeric) / n
        return {"mean": mean, "std": math.sqrt(var),
                "min": min(self._numeric), "max": max(self._numeric)}

    def supports(self, value: Any) -> bool:
        return _key(value) in self._counts

    def to_dict(self, n_episodes: int) -> Dict[str, Any]:
        val, prob = self.modal()
        return {
            "name": self.name,
            "fill_rate": round(self.fill_rate(n_episodes), 3),
            "modal": val, "modal_prob": round(prob, 3),
            "variability": round(self.variability(), 3),
            "distinct": self.distinct(),
            "numeric": self.numeric_stats(),
        }


# --------------------------------------------------------------------------- #
# Filled episode (prediction result)
# --------------------------------------------------------------------------- #
@dataclass
class FilledEpisode:
    values: Dict[str, Any]
    predicted: Dict[str, Tuple[Any, float]]  # slot -> (value, confidence)

    def to_dict(self) -> Dict[str, Any]:
        return {"values": self.values,
                "predicted": {k: {"value": v, "confidence": round(c, 3)}
                              for k, (v, c) in self.predicted.items()}}


# --------------------------------------------------------------------------- #
# Schema — an abstracted template over many episodes
# --------------------------------------------------------------------------- #
@dataclass
class Schema:
    name: str
    slots: Dict[str, Slot] = field(default_factory=dict)
    support: int = 0                                   # number of member episodes
    member_ids: List[str] = field(default_factory=list)
    prototype: Optional[List[float]] = None            # centroid embedding
    tags: frozenset = field(default_factory=frozenset)
    schema_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ---- learning ---- #
    def observe(self, episode: Dict[str, Any], *, embedding: Optional[Sequence[float]] = None,
                member_id: Optional[str] = None) -> None:
        for k, v in episode.items():
            self.slots.setdefault(k, Slot(name=k)).observe(v)
        self.support += 1
        if member_id:
            self.member_ids.append(member_id)
        if embedding is not None:
            self._update_prototype(embedding)
        self.updated_at = time.time()

    def _update_prototype(self, emb: Sequence[float]) -> None:
        if self.prototype is None:
            self.prototype = list(emb)
            return
        n = self.support
        # running mean
        self.prototype = [(self.prototype[i] * (n - 1) + emb[i]) / n
                          for i in range(len(self.prototype))]
        norm = math.sqrt(sum(x * x for x in self.prototype)) or 1.0
        self.prototype = [x / norm for x in self.prototype]

    # ---- structure ---- #
    def required_slots(self, threshold: float = 0.8) -> List[str]:
        return [k for k, s in self.slots.items() if s.fill_rate(self.support) >= threshold]

    def optional_slots(self, threshold: float = 0.8) -> List[str]:
        return [k for k, s in self.slots.items() if s.fill_rate(self.support) < threshold]

    def constant_slots(self) -> Dict[str, Any]:
        return {k: s.modal()[0] for k, s in self.slots.items() if s.is_constant()}

    # ---- matching ---- #
    def match(self, episode: Dict[str, Any], *,
              embedding: Optional[Sequence[float]] = None,
              w_keys: float = 0.4, w_values: float = 0.4, w_proto: float = 0.2) -> float:
        """Score how well an episode fits this schema in [0,1]."""
        if not self.slots:
            return 0.0
        ep_keys = set(episode)
        sch_keys = set(self.slots)
        key_overlap = len(ep_keys & sch_keys) / len(ep_keys | sch_keys) if (ep_keys | sch_keys) else 0.0

        shared = ep_keys & sch_keys
        if shared:
            agree = sum(1.0 if self.slots[k].supports(episode[k]) else 0.0 for k in shared)
            value_agreement = agree / len(shared)
        else:
            value_agreement = 0.0

        proto_sim = 0.0
        if embedding is not None and self.prototype is not None:
            proto_sim = max(0.0, _cosine(embedding, self.prototype))

        # renormalise weights if no prototype available
        if embedding is None or self.prototype is None:
            tw = w_keys + w_values
            return _clamp((w_keys * key_overlap + w_values * value_agreement) / (tw or 1.0))
        total = w_keys + w_values + w_proto
        return _clamp((w_keys * key_overlap + w_values * value_agreement
                       + w_proto * proto_sim) / total)

    # ---- prediction ---- #
    def predict_slot(self, slot: str) -> Tuple[Any, float]:
        s = self.slots.get(slot)
        if s is None or s.present_count == 0:
            return (None, 0.0)
        val, prob = s.modal()
        conf = prob * s.fill_rate(self.support)   # typical AND usually-present
        return (val, _clamp(conf))

    def fill(self, partial: Dict[str, Any], *, fill_threshold: float = 0.5) -> FilledEpisode:
        """Complete a partial episode with predicted defaults for missing slots."""
        values = dict(partial)
        predicted: Dict[str, Tuple[Any, float]] = {}
        for k, s in self.slots.items():
            if k in values:
                continue
            if s.fill_rate(self.support) >= fill_threshold:
                val, conf = self.predict_slot(k)
                if val is not None:
                    values[k] = val
                    predicted[k] = (val, conf)
        return FilledEpisode(values=values, predicted=predicted)

    def anomaly_score(self, episode: Dict[str, Any]) -> float:
        """How surprising is this episode vs the template? (0 typical … 1 anomalous)."""
        return _clamp(1.0 - self.match(episode))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "schema_id": self.schema_id, "support": self.support,
            "tags": sorted(self.tags),
            "slots": {k: s.to_dict(self.support) for k, s in self.slots.items()},
            "required": self.required_slots(),
        }


# --------------------------------------------------------------------------- #
# Library — induction, integration, retrieval
# --------------------------------------------------------------------------- #
def episode_from_record(rec: MemoryRecord) -> Dict[str, Any]:
    """Derive a structured episode dict from a MemoryRecord (content + tags)."""
    base: Dict[str, Any] = {}
    if isinstance(rec.content, dict):
        base.update(rec.content)
    else:
        base["_text"] = rec.text()
    if rec.tags:
        base["_tags"] = tuple(sorted(rec.tags))
    return base


class SchemaLibrary:
    """Induces, stores, and applies schemas; integrates new episodes incrementally."""

    def __init__(
        self,
        *,
        embedder: Optional[HashingEmbedder] = None,
        match_threshold: float = 0.45,
        embed_text_key: str = "_text",
    ) -> None:
        self.embedder = embedder or HashingEmbedder(dim=128)
        self.match_threshold = match_threshold
        self.embed_text_key = embed_text_key
        self._schemas: Dict[str, Schema] = {}

    def __len__(self) -> int:
        return len(self._schemas)

    def schemas(self) -> List[Schema]:
        return list(self._schemas.values())

    def _embed(self, episode: Dict[str, Any]) -> Optional[List[float]]:
        text = episode.get(self.embed_text_key)
        if text is None:
            # synthesise a text from categorical slot values for clustering
            parts = [f"{k}={v}" for k, v in episode.items() if not k.startswith("_")]
            text = " ".join(parts) if parts else None
        return self.embedder.embed(text) if text else None

    # ---- supervised induction ---- #
    def induce(self, name: str, episodes: Sequence[Dict[str, Any]],
               *, tags: Sequence[str] = ()) -> Schema:
        """Build one schema from a labelled set of episodes that belong together."""
        if not episodes:
            raise ValueError("cannot induce a schema from zero episodes")
        sch = Schema(name=name, tags=frozenset(tags))
        for i, ep in enumerate(episodes):
            sch.observe(ep, embedding=self._embed(ep), member_id=f"{name}#{i}")
        self._schemas[sch.schema_id] = sch
        return sch

    # ---- unsupervised clustering induction ---- #
    def cluster_induce(self, episodes: Sequence[Dict[str, Any]],
                       *, threshold: Optional[float] = None,
                       name_prefix: str = "schema") -> List[Schema]:
        """Greedily cluster episodes by similarity and abstract each cluster."""
        thr = threshold if threshold is not None else self.match_threshold
        clusters: List[Schema] = []
        for i, ep in enumerate(episodes):
            emb = self._embed(ep)
            best, best_score = None, 0.0
            for sch in clusters:
                sc = sch.match(ep, embedding=emb)
                if sc > best_score:
                    best, best_score = sch, sc
            if best is not None and best_score >= thr:
                best.observe(ep, embedding=emb, member_id=f"ep#{i}")
            else:
                sch = Schema(name=f"{name_prefix}_{len(clusters)}")
                sch.observe(ep, embedding=emb, member_id=f"ep#{i}")
                clusters.append(sch)
        for sch in clusters:
            self._schemas[sch.schema_id] = sch
        return clusters

    # ---- retrieval & integration ---- #
    def best_match(self, episode: Dict[str, Any]) -> Tuple[Optional[Schema], float]:
        emb = self._embed(episode)
        best, score = None, 0.0
        for sch in self._schemas.values():
            s = sch.match(episode, embedding=emb)
            if s > score:
                best, score = sch, s
        return best, score

    def integrate(self, episode: Dict[str, Any], *,
                  new_name: Optional[str] = None, member_id: Optional[str] = None) -> Schema:
        """Fold an episode into its best-matching schema, or spawn a new one."""
        best, score = self.best_match(episode)
        emb = self._embed(episode)
        if best is not None and score >= self.match_threshold:
            best.observe(episode, embedding=emb, member_id=member_id)
            return best
        sch = Schema(name=new_name or f"schema_{len(self._schemas)}")
        sch.observe(episode, embedding=emb, member_id=member_id)
        self._schemas[sch.schema_id] = sch
        return sch

    def fill(self, partial: Dict[str, Any], **kw: Any) -> Optional[FilledEpisode]:
        best, score = self.best_match(partial)
        if best is None or score < self.match_threshold:
            return None
        return best.fill(partial, **kw)

    def stats(self) -> Dict[str, Any]:
        return {
            "schemas": len(self._schemas),
            "total_support": sum(s.support for s in self._schemas.values()),
            "by_schema": {s.name: s.support for s in self._schemas.values()},
        }


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA schema-induction self-test")
    print("=" * 70)

    # episodes of a recurring security event — many concrete logins
    logins = [
        {"event": "ssh_login", "port": 22, "time": "night", "source": "external", "verdict": "blocked"},
        {"event": "ssh_login", "port": 22, "time": "night", "source": "external", "verdict": "blocked"},
        {"event": "ssh_login", "port": 22, "time": "day", "source": "external", "verdict": "blocked"},
        {"event": "ssh_login", "port": 2222, "time": "night", "source": "external", "verdict": "blocked"},
        {"event": "ssh_login", "port": 22, "time": "night", "source": "internal", "verdict": "allowed"},
    ]

    lib = SchemaLibrary()
    sch = lib.induce("ssh_login_attempt", logins, tags=["security"])
    print(f"\ninduced schema      : {sch.name} (support={sch.support})")
    print(f"required slots      : {sch.required_slots()}")
    print(f"constant slots      : {sch.constant_slots()}")
    for k, s in sch.slots.items():
        val, prob = s.modal()
        print(f"  slot {k:8}: modal={val!r} (p={prob:.2f}) var={s.variability():.2f} "
              f"fill={s.fill_rate(sch.support):.2f}")

    # 'event' is constant; 'port'/'time'/'verdict' have a clear mode
    assert sch.constant_slots()["event"] == "ssh_login"
    assert sch.predict_slot("port")[0] == 22
    assert sch.predict_slot("verdict")[0] == "blocked"

    # slot-filling: a partial episode is completed with the template's expectations
    partial = {"event": "ssh_login", "source": "external"}
    filled = sch.fill(partial)
    print(f"\nfill {partial}")
    print(f"  -> {filled.to_dict()['predicted']}")
    assert filled.values["port"] == 22
    assert filled.values["verdict"] == "blocked"

    # matching & anomaly: a typical vs a bizarre episode
    typical = {"event": "ssh_login", "port": 22, "time": "night", "source": "external", "verdict": "blocked"}
    weird = {"event": "ftp_upload", "size_gb": 50, "dest": "unknown", "time": "night"}
    print(f"\nmatch(typical)      : {sch.match(typical):.3f}")
    print(f"match(weird)        : {sch.match(weird):.3f}")
    assert sch.match(typical) > sch.match(weird)
    assert sch.anomaly_score(weird) > sch.anomaly_score(typical)

    # unsupervised clustering: mixed episodes self-organise into templates
    mixed = logins + [
        {"event": "coffee_break", "drink": "espresso", "time": "morning"},
        {"event": "coffee_break", "drink": "latte", "time": "morning"},
        {"event": "coffee_break", "drink": "espresso", "time": "afternoon"},
    ]
    lib2 = SchemaLibrary(match_threshold=0.5)
    clusters = lib2.cluster_induce(mixed)
    print(f"\nclustered into      : {len(clusters)} schemas "
          f"({[c.support for c in clusters]})")
    assert len(clusters) >= 2  # logins and coffee separate

    # incremental integration
    s_before = lib.schemas()[0].support
    lib.integrate({"event": "ssh_login", "port": 22, "time": "night",
                   "source": "external", "verdict": "blocked"})
    assert lib.schemas()[0].support == s_before + 1
    print(f"\nintegrated new episode; support {s_before} -> {lib.schemas()[0].support}")

    print(f"\nstats               : {lib.stats()}")
    print("\nALL SELF-TESTS PASSED ✓")
