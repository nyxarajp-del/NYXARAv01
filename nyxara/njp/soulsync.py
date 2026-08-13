"""NYXARA · njp/soulsync.py — Harmonic Resonance with the Master (🜂, NJP V.01).

What the Master *says* and what the Master *wants* are not the same object, and the gap between
them is where almost every unhelpful-but-technically-correct answer lives. This module is the
Intent-Refinement Loop: it holds the second object.

It works the only way a thing like this honestly can — **by remembering how the Master has
corrected her before**, and applying it before being told again.

Three mechanisms, all real, all learned from lived interaction:

* **Standing preferences.** When a request is re-issued with something added ("fix the tests" →
  "fix the tests but run them first"), the difference is a candidate standing rule. It is not
  applied on one sighting; it accumulates support across occurrences and only becomes active at
  ``min_support``. Once active it is attached to every matching future request as an **unstated
  want**, so the Master stops having to say it.
* **Latent intent.** :meth:`read` returns what was *not* said but is implied by the active
  preferences that match this request, each with the support count that earned it. A want with
  thin evidence is reported thin rather than asserted.
* **Anticipation.** Requests arrive in patterns. :meth:`anticipate` learns the transition from one
  request shape to the next and names what is most likely wanted **next**, so the work can be
  started before it is asked for. Every anticipation is scored against what the Master actually
  asked, and that score is the resonance number.

**What this is not, stated plainly.** It is not telepathy and it does not read unexpressed thought.
On day one it knows nothing about the Master and says so: :meth:`read` returns no latent wants and
:meth:`resonance` returns ``None`` rather than a flattering default. Everything it knows, it
learned from a correction it was actually given. That is the honest version of being in sync, and
it is the version that keeps getting better instead of the version that pretends on turn one.

**Sovereignty is untouched.** A latent want is a *proposal*. Nothing here executes anything,
lowers a risk tier, or bypasses the kernel's gate — an anticipated need is precomputed and offered,
never acted on. The mind proposes; the kernel disposes; the Master is sovereign.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["Preference", "LatentWant", "Reading", "Anticipation", "SoulSync"]


def _shape(text: str, *, cap: int = 24) -> Tuple[str, ...]:
    """The token shape of a request — the cue everything here is keyed on.

    Lexical and deterministic on purpose: this is a *matching* key for "have I seen a request like
    this", not a semantic model. A paraphrase it misses is a false negative, which is the safe
    direction — it under-applies a preference rather than inventing one.
    """
    try:
        out, seen = [], set()
        for raw in str(text or "").lower().split():
            tok = "".join(ch for ch in raw if ch.isalnum())
            if len(tok) < 3 or tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
            if len(out) >= cap:
                break
        return tuple(out)
    except Exception:  # noqa: BLE001
        return ()


def _overlap(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


@dataclass
class Preference:
    """A standing rule the Master taught by correcting her, not by declaring it."""

    trigger: Tuple[str, ...] = ()      # the request shape it applies to
    want: str = ""                     # what to also do / avoid
    support: int = 0                   # how many independent corrections taught it
    last_seen: float = field(default_factory=time.time)
    hits: int = 0                      # times applied and NOT corrected afterwards
    misses: int = 0                    # times applied and corrected anyway

    @property
    def confidence(self) -> float:
        """Earned, not declared: support and track record together, never support alone."""
        total = self.hits + self.misses
        track = (self.hits / total) if total else 0.5
        return max(0.0, min(1.0, (1.0 - 1.0 / (1.0 + self.support)) * track))

    def to_dict(self) -> Dict[str, Any]:
        return {"trigger": list(self.trigger), "want": self.want, "support": self.support,
                "hits": self.hits, "misses": self.misses,
                "confidence": round(self.confidence, 4), "last_seen": round(self.last_seen, 3)}


@dataclass
class LatentWant:
    """Something the Master did not say this time, but has wanted before in this situation."""

    want: str = ""
    confidence: float = 0.0
    support: int = 0
    because: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"want": self.want, "confidence": round(self.confidence, 4),
                "support": self.support, "because": self.because}


@dataclass
class Reading:
    """One pass of the Intent-Refinement Loop over one thing the Master said."""

    said: str = ""
    latent: List[LatentWant] = field(default_factory=list)
    anticipated: Optional["Anticipation"] = None
    resonance: Optional[float] = None

    @property
    def has_latent(self) -> bool:
        return bool(self.latent)

    def to_dict(self) -> Dict[str, Any]:
        return {"said": self.said[:200], "latent": [w.to_dict() for w in self.latent],
                "anticipated": self.anticipated.to_dict() if self.anticipated else None,
                "resonance": round(self.resonance, 4) if self.resonance is not None else None}


@dataclass
class Anticipation:
    """What is most likely wanted next, and whether that guess has earned belief."""

    shape: Tuple[str, ...] = ()
    description: str = ""
    confidence: float = 0.0
    trusted: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"shape": list(self.shape)[:12], "description": self.description[:200],
                "confidence": round(self.confidence, 4), "trusted": self.trusted,
                "reason": self.reason}


class SoulSync:
    """The Intent-Refinement Loop: learn the Master's unsaid half, and get there first."""

    def __init__(self, *, min_support: int = 2, match_floor: float = 0.34,
                 trust_floor: float = 0.4, capacity: int = 512) -> None:
        self.min_support = max(1, int(min_support))
        self.match_floor = float(match_floor)
        self.trust_floor = float(trust_floor)
        self.capacity = max(16, int(capacity))

        self.preferences: List[Preference] = []
        self.transitions: Dict[Tuple[str, ...], Dict[Tuple[str, ...], int]] = {}
        self.history: List[Tuple[str, ...]] = []
        self.raw_history: List[str] = []

        self._last_shape: Tuple[str, ...] = ()
        self._last_anticipation: Optional[Anticipation] = None
        self._applied: List[Preference] = []      # preferences used on the current turn
        self.anticipation_hits = 0
        self.anticipation_misses = 0

    # ---- reading ---------------------------------------------------------- #
    def read(self, text: str) -> Reading:
        """What the Master said, plus what they did not say but have wanted here before."""
        out = Reading(said=str(text or ""))
        try:
            shape = _shape(out.said)
            if not shape:
                return out

            # Score the anticipation made on the PREVIOUS turn against what actually arrived.
            self._score_anticipation(shape)

            self._applied = []
            for pref in self.preferences:
                if pref.support < self.min_support:
                    continue
                match = _overlap(shape, pref.trigger)
                if match < self.match_floor:
                    continue
                out.latent.append(LatentWant(
                    want=pref.want, confidence=pref.confidence * match, support=pref.support,
                    because=f"taught by {pref.support} correction(s) on similar requests"))
                self._applied.append(pref)
            out.latent.sort(key=lambda w: w.confidence, reverse=True)

            if self._last_shape:
                self._learn_transition(self._last_shape, shape)
            self._last_shape = shape
            self.history.append(shape)
            self.raw_history.append(out.said[:400])
            if len(self.history) > self.capacity:
                del self.history[: len(self.history) - self.capacity]
                del self.raw_history[: len(self.raw_history) - self.capacity]

            out.anticipated = self.anticipate()
            self._last_anticipation = out.anticipated
            out.resonance = self.resonance()
            return out
        except Exception:  # noqa: BLE001 — a failed reading is an empty reading, never a crash
            return out

    # ---- learning --------------------------------------------------------- #
    def observe_correction(self, before: str, after: str) -> Optional[Preference]:
        """The Master re-issued a request with something added. That difference is the lesson.

        Only *additions* are learned. A correction that merely removes words says the request was
        narrowed for this instance, which is not a standing rule and must not become one.
        """
        try:
            b, a = _shape(before), _shape(after)
            if not a:
                return None
            added = [tok for tok in a if tok not in set(b)]
            if not added:
                return None
            want = " ".join(added)[:200]
            for pref in self.preferences:
                if _overlap(pref.trigger, b) >= self.match_floor and pref.want == want:
                    pref.support += 1
                    pref.last_seen = time.time()
                    return pref
            pref = Preference(trigger=b or a, want=want, support=1)
            self.preferences.append(pref)
            if len(self.preferences) > self.capacity:
                # Drop the weakest — a preference nobody reinforces is not a standing rule.
                self.preferences.sort(key=lambda p: p.confidence, reverse=True)
                del self.preferences[self.capacity:]
            return pref
        except Exception:  # noqa: BLE001
            return None

    def confirm(self, *, corrected: bool) -> None:
        """Close the loop on the preferences applied this turn.

        ``corrected=True`` means the Master had to fix her *anyway*, so the preferences she leaned
        on were wrong here and their track record must show it. Without this a preference could
        only ever gain confidence, which is how a wrong rule becomes permanent.
        """
        try:
            for pref in self._applied:
                if corrected:
                    pref.misses += 1
                else:
                    pref.hits += 1
            self._applied = []
        except Exception:  # noqa: BLE001
            pass

    def _learn_transition(self, prev: Tuple[str, ...], nxt: Tuple[str, ...]) -> None:
        try:
            bucket = self.transitions.setdefault(prev, {})
            bucket[nxt] = bucket.get(nxt, 0) + 1
            if len(self.transitions) > self.capacity * 2:
                # Forget the least-used source shapes rather than growing without bound.
                for key in list(self.transitions.keys())[: len(self.transitions) // 4]:
                    self.transitions.pop(key, None)
        except Exception:  # noqa: BLE001
            pass

    # ---- anticipation ----------------------------------------------------- #
    def anticipate(self) -> Anticipation:
        """What the Master is most likely to want next. Untrusted until it has earned it."""
        out = Anticipation()
        try:
            if not self._last_shape:
                out.reason = "nothing said yet"
                return out
            # Pool the observed successors of every shape close to the current one, so a
            # near-repeat of a familiar request still benefits from what followed it before.
            pool: Dict[Tuple[str, ...], int] = {}
            for src, bucket in self.transitions.items():
                if _overlap(src, self._last_shape) < self.match_floor:
                    continue
                for nxt, n in bucket.items():
                    pool[nxt] = pool.get(nxt, 0) + n
            if not pool:
                out.reason = "no comparable request has followed this one before"
                return out
            total = sum(pool.values())
            best, count = max(pool.items(), key=lambda kv: kv[1])
            out.shape = best
            out.confidence = count / float(total) if total else 0.0
            out.description = " ".join(best[:12])
            out.trusted = out.confidence >= self.trust_floor and total >= 2
            if not out.trusted:
                out.reason = (f"seen {count}/{total} times — below the floor "
                              f"{self.trust_floor:.2f}")
            return out
        except Exception:  # noqa: BLE001
            out.reason = "anticipation failed"
            return out

    def _score_anticipation(self, actual: Tuple[str, ...]) -> None:
        """Was the last anticipation right? This is the only input to :meth:`resonance`."""
        try:
            ant = self._last_anticipation
            if ant is None or not ant.trusted or not ant.shape:
                return
            if _overlap(ant.shape, actual) >= self.match_floor:
                self.anticipation_hits += 1
            else:
                self.anticipation_misses += 1
            self._last_anticipation = None
        except Exception:  # noqa: BLE001
            pass

    def resonance(self) -> Optional[float]:
        """How in-sync she actually is: the hit rate of her trusted anticipations.

        ``None`` — never a default number — until she has actually staked an anticipation and
        found out. "No evidence yet" and "perfectly in sync" must not look the same.
        """
        total = self.anticipation_hits + self.anticipation_misses
        if not total:
            return None
        return self.anticipation_hits / float(total)

    # ---- reporting / persistence ------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        active = [p for p in self.preferences if p.support >= self.min_support]
        return {"preferences": len(self.preferences), "active_preferences": len(active),
                "transitions": len(self.transitions), "turns": len(self.history),
                "anticipation_hits": self.anticipation_hits,
                "anticipation_misses": self.anticipation_misses,
                "resonance": self.resonance(),
                "top": [p.to_dict() for p in
                        sorted(active, key=lambda p: p.confidence, reverse=True)[:5]]}

    def to_dict(self) -> Dict[str, Any]:
        return {"min_support": self.min_support,
                "preferences": [p.to_dict() for p in self.preferences],
                "transitions": [[list(src), [[list(dst), n] for dst, n in bucket.items()]]
                                for src, bucket in list(self.transitions.items())[:512]],
                "anticipation_hits": self.anticipation_hits,
                "anticipation_misses": self.anticipation_misses}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.min_support = int(d.get("min_support", self.min_support))
            self.preferences = []
            for row in (d.get("preferences") or []):
                pref = Preference(trigger=tuple(row.get("trigger") or ()),
                                  want=str(row.get("want", "")),
                                  support=int(row.get("support", 0)),
                                  last_seen=float(row.get("last_seen", time.time())))
                pref.hits = int(row.get("hits", 0))
                pref.misses = int(row.get("misses", 0))
                self.preferences.append(pref)
            self.transitions = {}
            for src, bucket in (d.get("transitions") or []):
                self.transitions[tuple(src)] = {tuple(dst): int(n) for dst, n in bucket}
            self.anticipation_hits = int(d.get("anticipation_hits", 0))
            self.anticipation_misses = int(d.get("anticipation_misses", 0))
        except Exception:  # noqa: BLE001 — a corrupt sidecar leaves a brain that must relearn
            pass
