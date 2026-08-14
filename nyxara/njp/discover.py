"""NYXARA · njp/discover.py — she proposes her own abstractions, then tries to kill them (🔬).

The world model in :mod:`nyxara.njp.world` finds ``cause → effect`` between things she has already
named. This finds the *pattern above that*: when ``A + B + C`` keeps arriving together before
``D``, the useful move is not to record three more pairs but to hypothesise that **A, B and C
jointly predict D** — one abstraction standing in for a family of observations.

    observe → hypothesise → **test** → compress → generalise

The middle step is the one that makes this science rather than pattern-matching. A hypothesis is
formed on a *split* of the evidence and scored on the half it never saw, so a rule that merely
describes the episodes that suggested it does not survive contact with the ones that did not.
Anything can find patterns in data it was fitted to; the only interesting question is whether the
pattern holds on data it was not.

Three states, and the third is the point:

* **conjectured** — proposed, not yet tested. Never used to predict anything.
* **confirmed** — held up on held-out episodes, and keeps being re-tested as evidence arrives.
* **refuted** — failed, and **discarded**. A system that only ever accumulates hypotheses is not
  learning, it is hoarding; and a refuted rule left in place quietly poisons every later
  prediction that consults it.

**Compression is the payoff, and it is measured.** :meth:`compression` reports how many raw
observations the surviving abstractions stand in for. That is what makes "ten thousand experiences
became a concept" a checkable claim rather than a slogan — and when the number is 1.0 she has
compressed nothing, which is reported rather than dressed up.

Honest limits: this is correlational structure over her own episodes, discovered without
experiment. A confirmed abstraction is a rule that has *survived* falsification on held-out data,
which is a real and useful thing, and is not a proof.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = ["Status", "Abstraction", "DiscoveryReport", "Discoverer"]


class Status:
    """Where a hypothesis stands. ``REFUTED`` ones are removed, never merely labelled."""

    CONJECTURED = "conjectured"
    CONFIRMED = "confirmed"
    REFUTED = "refuted"


@dataclass
class Abstraction:
    """``antecedents → consequent``, with the record that decides whether it lives."""

    antecedents: Tuple[str, ...] = ()
    consequent: str = ""
    status: str = Status.CONJECTURED
    fitted_on: int = 0            # observations that suggested it
    tested: int = 0               # held-out episodes it was scored against
    held: int = 0                 # of those, how many it got right
    born: float = field(default_factory=time.time)
    name: str = ""

    @property
    def precision(self) -> float:
        """How often it holds on evidence it was not fitted to."""
        return (self.held / self.tested) if self.tested else 0.0

    @property
    def order(self) -> int:
        """How many things it ties together. Higher is a bigger claim, and compresses more."""
        return len(self.antecedents)

    @property
    def usable(self) -> bool:
        return self.status == Status.CONFIRMED

    def key(self) -> Tuple[Tuple[str, ...], str]:
        return (tuple(sorted(self.antecedents)), self.consequent)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "antecedents": list(self.antecedents),
                "consequent": self.consequent, "status": self.status,
                "order": self.order, "fitted_on": self.fitted_on,
                "tested": self.tested, "held": self.held,
                "precision": round(self.precision, 4)}


@dataclass
class DiscoveryReport:
    """What one discovery pass actually changed."""

    proposed: int = 0
    confirmed: int = 0
    refuted: int = 0
    retested: int = 0
    live: int = 0
    compression: float = 1.0
    ms: float = 0.0

    @property
    def changed(self) -> bool:
        return bool(self.proposed or self.confirmed or self.refuted)

    def to_dict(self) -> Dict[str, Any]:
        return {"proposed": self.proposed, "confirmed": self.confirmed,
                "refuted": self.refuted, "retested": self.retested, "live": self.live,
                "compression": round(self.compression, 4), "changed": self.changed,
                "ms": round(self.ms, 3)}


class Discoverer:
    """Proposes abstractions from her own episodes, and tries to falsify each one."""

    def __init__(self, *, min_support: int = 4, min_precision: float = 0.7,
                 max_order: int = 3, holdout: float = 0.4,
                 capacity: int = 512, retest_every: int = 8) -> None:
        self.min_support = max(2, int(min_support))
        self.min_precision = float(min_precision)
        self.max_order = max(2, int(max_order))
        # Fraction of episodes withheld from fitting. Large enough that surviving it means
        # something — a 5% holdout confirms almost anything.
        self.holdout = min(0.8, max(0.1, float(holdout)))
        self.capacity = max(16, int(capacity))
        self.retest_every = max(1, int(retest_every))

        # Each episode is (co-occurring antecedents, what followed).
        self.episodes: List[Tuple[Set[str], str]] = []
        self.abstractions: Dict[Tuple[Tuple[str, ...], str], Abstraction] = {}
        self.passes = 0
        self.proposed_total = 0
        self.confirmed_total = 0
        self.refuted_total = 0

    # ---- observe ---------------------------------------------------------- #
    def observe(self, antecedents: Iterable[str], consequent: str) -> None:
        """Record one episode: these things were present, and then this happened."""
        try:
            items = {str(a).strip() for a in antecedents if str(a).strip()}
            outcome = str(consequent or "").strip()
            if not items or not outcome:
                return
            self.episodes.append((items, outcome))
            if len(self.episodes) > self.capacity * 8:
                del self.episodes[: len(self.episodes) - self.capacity * 8]
        except Exception:  # noqa: BLE001
            pass

    # ---- the pass --------------------------------------------------------- #
    def discover(self) -> DiscoveryReport:
        """One full cycle: propose on the fit split, test on the held-out one, keep or kill."""
        report = DiscoveryReport()
        t0 = time.perf_counter()
        try:
            self.passes += 1
            fit, held = self._split()
            if len(fit) < self.min_support or not held:
                report.live = sum(1 for a in self.abstractions.values() if a.usable)
                report.compression = self.compression()
                return report

            for candidate in self._propose(fit):
                key = candidate.key()
                existing = self.abstractions.get(key)
                if existing is None:
                    self.abstractions[key] = candidate
                    self.proposed_total += 1
                    report.proposed += 1
                    existing = candidate
                else:
                    report.retested += 1
                self._test(existing, held)
                self._rule(existing, report)

            self._evict()
            report.live = sum(1 for a in self.abstractions.values() if a.usable)
            report.compression = self.compression()
            return report
        except Exception:  # noqa: BLE001 — a failed pass discovers nothing, never crashes
            return report
        finally:
            report.ms = (time.perf_counter() - t0) * 1000.0

    def _split(self) -> Tuple[List[Tuple[Set[str], str]], List[Tuple[Set[str], str]]]:
        """Fit set / held-out set, **strided** rather than cut at a point in time.

        The obvious split is positional — fit on the earlier episodes, test on the later ones —
        and it is wrong here for a specific reason: her experience arrives in regimes. A rule
        learned from a situation that stopped occurring gets a held-out set containing **no
        episodes it applies to at all**, so it is never tested, never confirmed, and sits as a
        conjecture forever while a perfectly good pattern goes unused. Measured: with clouds→rain
        early and birds→clear late, a positional split confirmed nothing whatsoever.

        Striding takes every ``n``-th episode instead, so the held-out set spans every regime the
        fit set does. The cost is that it does not model "predicting the future from the past",
        and that cost is accepted deliberately: what is being tested here is whether a rule
        generalises beyond the episodes that suggested it, which is a question about coverage, not
        about chronology.
        """
        try:
            stride = max(2, int(round(1.0 / self.holdout)))
            fit = [e for i, e in enumerate(self.episodes) if i % stride]
            held = [e for i, e in enumerate(self.episodes) if not i % stride]
            return fit, held
        except Exception:  # noqa: BLE001
            cut = int(len(self.episodes) * (1.0 - self.holdout))
            return self.episodes[:cut], self.episodes[cut:]

    def _propose(self, fit: Sequence[Tuple[Set[str], str]]) -> List[Abstraction]:
        """Candidate rules from co-occurrence in the fit split, largest antecedent set first.

        Bigger antecedent sets are proposed first because they compress more: ``{a,b,c} → d`` says
        more than ``{a} → d`` and stands in for more observations. Order is capped — the number of
        subsets explodes, and a rule with eight antecedents describes one episode rather than a
        pattern.
        """
        out: List[Abstraction] = []
        try:
            counts: Dict[Tuple[Tuple[str, ...], str], int] = {}
            for items, outcome in fit:
                for subset in self._subsets(items):
                    counts[(subset, outcome)] = counts.get((subset, outcome), 0) + 1
            for (subset, outcome), support in counts.items():
                if support < self.min_support:
                    continue
                out.append(Abstraction(antecedents=subset, consequent=outcome,
                                       fitted_on=support,
                                       name=f"{{{', '.join(subset)}}} → {outcome}"))
            out.sort(key=lambda a: (-a.order, -a.fitted_on))
            return out[: self.capacity]
        except Exception:  # noqa: BLE001
            return out

    def _subsets(self, items: Set[str]) -> List[Tuple[str, ...]]:
        """Every antecedent combination up to ``max_order``, in a stable order."""
        ordered = sorted(items)
        out: List[Tuple[str, ...]] = []
        n = len(ordered)
        for size in range(2, min(self.max_order, n) + 1):
            out.extend(self._combinations(ordered, size))
        return out

    @staticmethod
    def _combinations(items: Sequence[str], size: int) -> List[Tuple[str, ...]]:
        import itertools
        return [tuple(c) for c in itertools.combinations(items, size)]

    def _test(self, abstraction: Abstraction, held: Sequence[Tuple[Set[str], str]]) -> None:
        """Score a rule on episodes it was never fitted to. This is the whole method.

        Only episodes where the antecedents are actually present count as a test — an episode the
        rule says nothing about is not evidence for or against it, and counting those would let a
        narrow rule look perfect by staying silent.
        """
        try:
            wanted = set(abstraction.antecedents)
            for items, outcome in held:
                if not wanted <= items:
                    continue
                abstraction.tested += 1
                if outcome == abstraction.consequent:
                    abstraction.held += 1
        except Exception:  # noqa: BLE001
            pass

    def _rule(self, abstraction: Abstraction, report: DiscoveryReport) -> None:
        """Confirm, refute, or leave conjectured. A refuted rule is removed by :meth:`_evict`."""
        try:
            if abstraction.tested < self.min_support:
                abstraction.status = Status.CONJECTURED    # not enough held-out evidence yet
                return
            was = abstraction.status
            if abstraction.precision >= self.min_precision:
                abstraction.status = Status.CONFIRMED
                if was != Status.CONFIRMED:
                    self.confirmed_total += 1
                    report.confirmed += 1
            else:
                abstraction.status = Status.REFUTED
                if was != Status.REFUTED:
                    self.refuted_total += 1
                    report.refuted += 1
        except Exception:  # noqa: BLE001
            pass

    def _evict(self) -> None:
        """Delete refuted rules, and cap the rest by how much they actually explain.

        Deletion rather than a label: a refuted rule left in place quietly poisons every later
        prediction that consults it, and "we knew it was wrong" is no defence once something has
        read it.
        """
        try:
            for key, abstraction in list(self.abstractions.items()):
                if abstraction.status == Status.REFUTED:
                    del self.abstractions[key]
            if len(self.abstractions) <= self.capacity:
                return
            ranked = sorted(self.abstractions.items(),
                            key=lambda kv: (kv[1].precision, kv[1].order, kv[1].tested))
            for key, _ in ranked[: len(self.abstractions) - self.capacity]:
                del self.abstractions[key]
        except Exception:  # noqa: BLE001
            pass

    # ---- using what survived ------------------------------------------------ #
    def predict(self, present: Iterable[str]) -> List[Tuple[str, float]]:
        """What the confirmed abstractions expect, given what is present. Conjectures never fire."""
        out: Dict[str, float] = {}
        try:
            items = {str(a).strip() for a in present if str(a).strip()}
            for abstraction in self.abstractions.values():
                if not abstraction.usable or not set(abstraction.antecedents) <= items:
                    continue
                # A higher-order rule that fits is better evidence than a broad one, so its
                # precision wins the slot rather than being averaged away.
                previous = out.get(abstraction.consequent, 0.0)
                out[abstraction.consequent] = max(previous, abstraction.precision)
            return sorted(out.items(), key=lambda kv: kv[1], reverse=True)
        except Exception:  # noqa: BLE001
            return []

    def compression(self) -> float:
        """Raw observations per surviving abstraction. 1.0 means nothing was compressed."""
        try:
            live = [a for a in self.abstractions.values() if a.usable]
            if not live:
                return 1.0
            explained = sum(a.fitted_on for a in live)
            return max(1.0, explained / len(live))
        except Exception:  # noqa: BLE001
            return 1.0

    def confirmed(self) -> List[Abstraction]:
        return sorted((a for a in self.abstractions.values() if a.usable),
                      key=lambda a: (-a.precision, -a.order))

    # ---- reporting ---------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        live = self.confirmed()
        return {
            "episodes": len(self.episodes), "passes": self.passes,
            "abstractions": len(self.abstractions), "confirmed": len(live),
            "proposed_total": self.proposed_total,
            "confirmed_total": self.confirmed_total,
            "refuted_total": self.refuted_total,
            "compression": round(self.compression(), 4),
            "best": live[0].to_dict() if live else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"abstractions": [a.to_dict() for a in self.abstractions.values()],
                "counters": {"passes": self.passes, "proposed": self.proposed_total,
                             "confirmed": self.confirmed_total, "refuted": self.refuted_total}}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            for row in (d.get("abstractions") or []):
                abstraction = Abstraction(
                    antecedents=tuple(row.get("antecedents") or ()),
                    consequent=str(row.get("consequent", "")),
                    status=str(row.get("status", Status.CONJECTURED)),
                    fitted_on=int(row.get("fitted_on", 0)),
                    tested=int(row.get("tested", 0)), held=int(row.get("held", 0)),
                    name=str(row.get("name", "")))
                if abstraction.consequent and abstraction.antecedents:
                    self.abstractions[abstraction.key()] = abstraction
            counters = d.get("counters") or {}
            self.passes = int(counters.get("passes", 0))
            self.proposed_total = int(counters.get("proposed", 0))
            self.confirmed_total = int(counters.get("confirmed", 0))
            self.refuted_total = int(counters.get("refuted", 0))
        except Exception:  # noqa: BLE001
            pass
