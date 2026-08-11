"""NYXARA · nyx/monad.py — the one mind the substrates add up to (☉, NYX V.03).

She has six brains. :class:`~nyxara.kernel.orchestrator.NyxaraCore` runs the sovereign loop and
holds the others; :class:`~nyxara.nyx.brain.NyxBrain` carries forty symbolic faculties;
:class:`~nyxara.nyx5.nyx5_brain.Nyx5Brain` runs a spiking substrate with its own
hyperdimensional memory; ``SelfBrain``, ``OfflineMind`` and ``SampleEfficientMind`` each answer
in their own way. They were built well and separately, and separately is the problem:

* **Two bottlenecks.** ``NyxaraCore`` builds a :class:`~nyxara.kernel.workspace.GlobalWorkspace`
  and ``NyxWorkspace`` builds another. Global Workspace Theory's whole claim is that *one* narrow
  channel makes many specialists into a mind. Two channels are two minds sharing a process.
* **Two memories.** What NYX-5 files in hypervectors, :mod:`nyxara.nyx.holomem` never sees, and
  the reverse. A substrate that learns something cannot tell the others.
* **No shared objective.** Every organ scores its own bids on its own scale, so nothing can
  actually be compared across substrates.

This module is the join. It does **not** merge the code — a single class holding all of it would
be a god-object, and the working parts would break on the way in. It merges the four things that
decide whether many parts are one mind:

1. **One workspace.** Every substrate is adapted into a
   :class:`~nyxara.nyx.modules.SpecialistModule` and bids into the *same*
   :class:`~nyxara.nyx.workspace.NyxWorkspace`. That is deliberate reuse, not a shortcut: the
   arbitration there already implements the repo's law that **verifiable beats probabilistic**,
   already scales each bid by that source's *measured* reliability (:mod:`nyxara.nyx.metacog`),
   and already habituates on content rather than on faculty. Re-implementing arbitration here
   would have meant a second, worse copy of all three.
2. **One memory.** :class:`UnifiedRecall` writes through to every store that can hold a fact and
   reads from all of them at once, so something NYX-5 learned is recallable by NyxBrain. The
   hypervector store's near-orthogonality is untouched — this is a façade over it, not a rewrite.
3. **One self-model.** :meth:`Monad.stats` reports the whole: which substrates are alive, who
   actually won access to consciousness, and how often.
4. **One safety boundary.** Untouched and still fail-closed. ``kernel/rules.py``,
   ``guard/corrigibility.py``, ``identity/values.py`` and ``growth/loyalty.py`` are not read,
   written or routed around here; MONAD only selects and broadcasts, exactly as the workspace it
   delegates to does. Nothing in this file can act.

Honest framing:

* **This is architectural unification, not a claim about experience.** What is real and testable
  is that exactly one bid wins per cycle, that every substrate is a participant in *that* contest,
  and that the winner is broadcast back to all of them. It is not evidence of phenomenal unity.
* **It joins what exists; it does not make any substrate smarter.** A substrate that had nothing
  to say still has nothing to say — it simply now says it on the same stage as the others.
* **Off by default is identical to today.** With ``monad_enabled`` false nothing is registered and
  every brain behaves exactly as it did, which is what makes this a regression-free change.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.nyx.modules import Proposal, Situation, SpecialistModule

__all__ = ["MemoryHit", "UnifiedRecall", "Substrate", "Nyx5Substrate", "SelfBrainSubstrate",
           "OfflineSubstrate", "SampleEfficientSubstrate", "IntuitionSubstrate", "MonadCycle",
           "Monad"]


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _text_of(value: Any) -> str:
    """Best-effort readable text from whatever a substrate handed back, without inventing any."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    for attr in ("text", "content", "answer", "reply", "key", "label", "summary"):
        got = getattr(value, attr, None)
        if isinstance(got, str) and got.strip():
            return got.strip()
    if isinstance(value, dict):
        for attr in ("text", "content", "answer", "key"):
            got = value.get(attr)
            if isinstance(got, str) and got.strip():
                return got.strip()
    return ""


# --------------------------------------------------------------------------- #
# One memory
# --------------------------------------------------------------------------- #
@dataclass
class MemoryHit:
    """One recalled item, tagged with which store actually produced it."""

    store: str
    key: str
    text: str
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"store": self.store, "key": self.key, "text": self.text,
                "score": round(self.score, 4)}


class UnifiedRecall:
    """One write, one read, across every store that can hold a fact.

    Each store keeps its own representation — episodic traces in :mod:`nyxara.nyx.holomem`,
    10,000-D hypervectors in NYX-5, typed triples in :mod:`nyxara.memory.graph`. Nothing is
    re-encoded or migrated. What changes is only that a write reaches all of them and a read asks
    all of them, so one substrate's learning stops being invisible to the rest.

    Every call is fail-soft per store: one store raising costs its results, never the recall.
    """

    def __init__(self, *, holomem: Any = None, nyx5: Any = None, graph: Any = None) -> None:
        self.holomem = holomem
        self.nyx5 = nyx5
        self.graph = graph
        self.writes = 0
        self.reads = 0
        self.store_errors: Dict[str, int] = {}

    def _note_error(self, store: str) -> None:
        self.store_errors[store] = self.store_errors.get(store, 0) + 1

    def stores(self) -> List[str]:
        """Which stores are actually present — reported, not assumed."""
        out = []
        if self.holomem is not None:
            out.append("holomem")
        if self.nyx5 is not None:
            out.append("nyx5")
        if self.graph is not None:
            out.append("graph")
        return out

    def remember(self, key: str, text: str, *, kind: str = "episode") -> List[str]:
        """Write through to every store. Returns the stores that accepted it."""
        key, text = str(key or "").strip(), str(text or "").strip()
        if not key or not text:
            return []
        written: List[str] = []
        if self.holomem is not None:
            try:
                self.holomem.remember(key, text, kind=kind)
                written.append("holomem")
            except Exception:  # noqa: BLE001
                self._note_error("holomem")
        if self.nyx5 is not None:
            try:
                self.nyx5.remember(key, text)
                written.append("nyx5")
            except Exception:  # noqa: BLE001
                self._note_error("nyx5")
        if written:
            self.writes += 1
        return written

    def recall(self, cue: str, *, k: int = 5) -> List[MemoryHit]:
        """Ask every store, merge, and return the best hits with their provenance intact."""
        cue = str(cue or "").strip()
        if not cue:
            return []
        self.reads += 1
        hits: List[MemoryHit] = []
        hits.extend(self._recall_holomem(cue, k))
        hits.extend(self._recall_nyx5(cue))
        hits.extend(self._recall_graph(cue, k))
        # Dedupe on key, keeping the strongest claim for it, then rank.
        best: Dict[str, MemoryHit] = {}
        for hit in hits:
            prior = best.get(hit.key)
            if prior is None or hit.score > prior.score:
                best[hit.key] = hit
        return sorted(best.values(), key=lambda h: -h.score)[:k]

    def _recall_holomem(self, cue: str, k: int) -> List[MemoryHit]:
        if self.holomem is None:
            return []
        try:
            out: List[MemoryHit] = []
            for trace, score in (self.holomem.candidates(cue, k=k) or ()):
                out.append(MemoryHit(store="holomem", key=str(getattr(trace, "key", "")),
                                     text=_text_of(trace), score=_clamp01(score)))
            return out
        except Exception:  # noqa: BLE001
            self._note_error("holomem")
            return []

    def _recall_nyx5(self, cue: str) -> List[MemoryHit]:
        if self.nyx5 is None:
            return []
        try:
            result = self.nyx5.recall(cue)
            if result is None:
                return []
            key = str(getattr(result, "key", "") or "")
            score = getattr(result, "similarity", None)
            if score is None:
                score = getattr(result, "score", 0.0)
            return [MemoryHit(store="nyx5", key=key, text=key, score=_clamp01(score))] if key else []
        except Exception:  # noqa: BLE001
            self._note_error("nyx5")
            return []

    def _recall_graph(self, cue: str, k: int) -> List[MemoryHit]:
        if self.graph is None:
            return []
        try:
            query = getattr(self.graph, "query", None)
            if not callable(query):
                return []
            out: List[MemoryHit] = []
            for item in (query(cue) or ())[:k]:
                text = _text_of(item)
                if text:
                    out.append(MemoryHit(store="graph", key=text[:80], text=text, score=0.5))
            return out
        except Exception:  # noqa: BLE001
            self._note_error("graph")
            return []

    def stats(self) -> Dict[str, Any]:
        return {"stores": self.stores(), "writes": self.writes, "reads": self.reads,
                "store_errors": dict(self.store_errors)}


# --------------------------------------------------------------------------- #
# Substrates — each brain, adapted into a bidder
# --------------------------------------------------------------------------- #
class Substrate(SpecialistModule):
    """A whole brain, adapted so it competes on the same stage as every other faculty.

    Subclasses implement :meth:`consider` (inherited contract: return a
    :class:`~nyxara.nyx.modules.Proposal` or ``None`` — silence is a legitimate answer) and may
    implement :meth:`receive` to be told what actually reached consciousness.
    """

    name = "substrate"
    priority = 0.5

    def __init__(self, engine: Any = None) -> None:
        self.engine = engine
        self.bids = 0
        self.wins = 0
        self.broadcasts_received = 0

    def available(self) -> bool:
        return self.engine is not None

    def receive(self, winner: Any) -> None:
        """Told what won. Fail-soft: a substrate that cannot absorb the broadcast is not a fault."""
        self.broadcasts_received += 1

    def safe_receive(self, winner: Any) -> None:
        try:
            if self.available():
                self.receive(winner)
        except Exception:  # noqa: BLE001 — one deaf substrate must not break the broadcast
            pass

    def stats(self) -> Dict[str, Any]:
        return {"name": self.name, "available": self.available(), "bids": self.bids,
                "wins": self.wins, "heard": self.broadcasts_received}


class Nyx5Substrate(Substrate):
    """The spiking brain. It bids what its hyperdimensional memory associates with the stimulus.

    Associative, so ``verifiable`` is False: a hypervector match is a real measurement of
    similarity, and similarity is not a derivation.
    """

    name = "nyx5"
    tags = frozenset({"neuromorphic", "associative", "memory"})
    priority = 0.5

    def consider(self, situation: Situation) -> Optional[Proposal]:
        stimulus = str(getattr(situation, "stimulus", "") or "").strip()
        if not stimulus:
            return None
        # Let the substrate see the turn even when it has nothing to say about it.
        perceive = getattr(self.engine, "perceive", None)
        surprise = 0.0
        if callable(perceive):
            tick = perceive(stimulus)
            surprise = _clamp01(getattr(tick, "surprise", 0.0))
        recall = getattr(self.engine, "recall", None)
        if not callable(recall):
            return None
        result = recall(stimulus)
        key = str(getattr(result, "key", "") or "") if result is not None else ""
        if not key:
            return None
        similarity = getattr(result, "similarity", None)
        if similarity is None:
            similarity = getattr(result, "score", 0.5)
        self.bids += 1
        return Proposal(
            source=self.name, content=key, confidence=_clamp01(similarity), verifiable=False,
            novelty=surprise, tags=set(self.tags),
            evidence=[f"nyx5:hd-memory:{key}"],
            rationale="nearest hypervector in the spiking substrate's cleanup memory")


class SelfBrainSubstrate(Substrate):
    """Her own learned language model — retrieval-augmented, no key and no GPU."""

    name = "self_reasoner"
    tags = frozenset({"language", "learned"})
    priority = 0.45

    def consider(self, situation: Situation) -> Optional[Proposal]:
        stimulus = str(getattr(situation, "stimulus", "") or "").strip()
        reply = getattr(self.engine, "reply", None)
        if not stimulus or not callable(reply):
            return None
        text = _text_of(reply(stimulus))
        if not text:
            return None
        confidence = 0.4
        scorer = getattr(self.engine, "internal_confidence", None)
        if callable(scorer):
            confidence = _clamp01(scorer(text))
        self.bids += 1
        return Proposal(source=self.name, content=text, confidence=confidence, verifiable=False,
                        tags=set(self.tags), evidence=["self_reasoner"],
                        rationale="her own learned model, answering from what it has indexed")


class OfflineSubstrate(Substrate):
    """The keyless offline mind — the floor that always answers something."""

    name = "offline"
    tags = frozenset({"offline", "fallback"})
    priority = 0.3

    def consider(self, situation: Situation) -> Optional[Proposal]:
        stimulus = str(getattr(situation, "stimulus", "") or "").strip()
        act = getattr(self.engine, "act", None)
        if not stimulus or not callable(act):
            return None
        text = _text_of(act(stimulus))
        if not text:
            return None
        self.bids += 1
        return Proposal(source=self.name, content=text, confidence=0.3, verifiable=False,
                        tags=set(self.tags), evidence=["offline_mind"],
                        rationale="offline candidate; deliberately weak so it only wins when alone")


class SampleEfficientSubstrate(Substrate):
    """Few-shot / one-shot cognition — strong exactly when there is little evidence."""

    name = "sample_efficient"
    tags = frozenset({"few-shot", "induction"})
    priority = 0.5

    def consider(self, situation: Situation) -> Optional[Proposal]:
        stimulus = str(getattr(situation, "stimulus", "") or "").strip()
        solve = getattr(self.engine, "solve", None)
        if not stimulus or not callable(solve):
            return None
        text = _text_of(solve(stimulus))
        if not text:
            return None
        self.bids += 1
        return Proposal(source=self.name, content=text, confidence=0.5, verifiable=False,
                        tags=set(self.tags), evidence=["sample_efficient"],
                        rationale="induced from few examples")


class IntuitionSubstrate(Substrate):
    """The fast, unjustified leap — bid honestly as a hunch, never as a derivation."""

    name = "intuition"
    tags = frozenset({"intuition", "fast"})
    priority = 0.35

    def consider(self, situation: Situation) -> Optional[Proposal]:
        stimulus = str(getattr(situation, "stimulus", "") or "").strip()
        leap = getattr(self.engine, "leap", None)
        if not stimulus or not callable(leap):
            return None
        try:
            from nyxara.mind.intuition import Problem
            problem = Problem.parse(stimulus)
        except Exception:  # noqa: BLE001 — without the Problem type there is nothing to leap on
            return None
        applies = getattr(self.engine, "applies", None)
        if callable(applies) and not applies(problem):
            return None
        hunch = leap(problem)
        text = _text_of(hunch)
        if not text:
            return None
        self.bids += 1
        return Proposal(
            source=self.name, content=text,
            confidence=_clamp01(getattr(hunch, "confidence", 0.35)), verifiable=False,
            tags=set(self.tags), evidence=["intuition"],
            rationale="a hunch — fast and unjustified, and marked as such")


# --------------------------------------------------------------------------- #
# The one mind
# --------------------------------------------------------------------------- #
@dataclass
class MonadCycle:
    """One pass of the unified mind."""

    stimulus: str = ""
    winner: str = ""
    content: str = ""
    salience: float = 0.0
    bidders: List[str] = field(default_factory=list)
    coalition: List[str] = field(default_factory=list)
    starved: bool = False
    cycle: int = 0
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"stimulus": self.stimulus, "winner": self.winner, "content": self.content,
                "salience": round(self.salience, 4), "bidders": self.bidders,
                "coalition": self.coalition, "starved": self.starved, "cycle": self.cycle,
                "at": self.at}


class Monad:
    """Six brains, one bottleneck, one memory, one report.

    Built around an existing :class:`~nyxara.nyx.brain.NyxBrain` (whose ``NyxWorkspace`` becomes
    *the* workspace) and, when present, a :class:`~nyxara.kernel.orchestrator.NyxaraCore` that
    holds the other substrates. Both are optional: with neither, the Monad is honestly empty
    rather than broken.
    """

    def __init__(self, brain: Any = None, *, core: Any = None,
                 substrates: Optional[Sequence[Substrate]] = None) -> None:
        self.brain = brain
        self.core = core
        self.cycles = 0
        self.history: List[MonadCycle] = []
        self.workspace = self._shared_workspace()
        self.substrates: List[Substrate] = []
        for sub in (substrates if substrates is not None else self._discover(brain, core)):
            self.join(sub)
        self.memory = UnifiedRecall(
            holomem=getattr(brain, "memory", None),
            nyx5=getattr(core, "nyx5", None) if core is not None else None,
            graph=self._knowledge_graph(brain, core),
        )

    # ---- construction ----------------------------------------------------- #
    def _shared_workspace(self) -> Any:
        """*The* workspace. Prefers the brain's, so its existing specialist cast keeps competing
        in the same contest the new substrates join."""
        existing = getattr(self.brain, "workspace", None)
        if existing is not None and hasattr(existing, "deliberate"):
            return existing
        try:
            from nyxara.nyx.workspace import NyxWorkspace
            return NyxWorkspace(metacog=getattr(self.brain, "metacog", None))
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _knowledge_graph(brain: Any, core: Any) -> Any:
        for owner, attr in ((core, "knowledge_graph"), (brain, "graph")):
            got = getattr(owner, attr, None) if owner is not None else None
            if got is not None:
                return got
        return None

    @staticmethod
    def _discover(brain: Any, core: Any) -> List[Substrate]:
        """Adapt whatever brains are actually present. A missing one is simply not registered —
        never a stub that pretends to think."""
        found: List[Substrate] = []
        if core is not None:
            for attr, cls in (("nyx5", Nyx5Substrate),
                              ("sample_efficient", SampleEfficientSubstrate),
                              ("intuition", IntuitionSubstrate)):
                engine = getattr(core, attr, None)
                if engine is not None:
                    found.append(cls(engine))
        for owner in (brain, core):
            if owner is None:
                continue
            for attr, cls in (("self_brain", SelfBrainSubstrate),
                              ("self_reasoner", SelfBrainSubstrate),
                              ("offline", OfflineSubstrate),
                              ("offline_mind", OfflineSubstrate)):
                engine = getattr(owner, attr, None)
                if engine is not None and not any(s.name == cls.name for s in found):
                    found.append(cls(engine))
        return found

    def join(self, substrate: Substrate) -> bool:
        """Register a substrate as a bidder in the one workspace."""
        if self.workspace is None or substrate is None:
            return False
        adder = getattr(self.workspace, "add_specialist", None)
        if not callable(adder) or not adder(substrate):
            return False
        self.substrates.append(substrate)
        return True

    # ---- the cycle -------------------------------------------------------- #
    def deliberate(self, stimulus: str, *, goals: Optional[Dict[str, float]] = None,
                   situation: Any = None) -> MonadCycle:
        """One pass: everyone bids into the one workspace, one wins, everyone hears it."""
        out = MonadCycle(stimulus=str(stimulus or ""), cycle=self.cycles + 1)
        if self.workspace is None:
            out.starved = True
            return out
        self.cycles += 1
        try:
            situation = situation if situation is not None else self._situation(out.stimulus)
            delib = self.workspace.deliberate(situation, goals=goals)
        except Exception:  # noqa: BLE001 — deliberation degrades, it does not crash a turn
            out.starved = True
            self.history.append(out)
            return out

        out.bidders = sorted(getattr(delib, "sources", []) or [])
        out.coalition = list(getattr(delib, "coalition", []) or [])
        out.salience = float(getattr(delib, "salience", 0.0) or 0.0)
        out.starved = bool(getattr(delib, "starved", False))
        winner = getattr(delib, "winner", None)
        if winner is not None:
            out.winner = str(getattr(winner, "source", "") or "")
            out.content = str(getattr(winner, "content", "") or "")
            for sub in self.substrates:
                if sub.name == out.winner:
                    sub.wins += 1
        # Broadcast — the half of Global Workspace Theory that makes the contest worth running.
        for sub in self.substrates:
            sub.safe_receive(winner)
        self.history.append(out)
        if len(self.history) > 200:
            del self.history[:-200]
        return out

    def _situation(self, stimulus: str) -> Situation:
        """Build the shared situation every substrate sees. Recall is unified, so each one is
        looking at the same evidence rather than only at its own store."""
        concepts: List[str] = []
        context: List[Any] = []
        recall: Any = None
        brain = self.brain
        if brain is not None:
            try:
                percept = brain.perceive(stimulus, remember=False)
                concepts = list(getattr(percept, "concepts", []) or [])
                recall = getattr(percept, "recall", None)
                context = list(getattr(percept, "context", []) or [])
            except Exception:  # noqa: BLE001
                pass
        return Situation(stimulus=stimulus, concepts=concepts, context=context, recall=recall,
                         brain=brain)

    # ---- reporting -------------------------------------------------------- #
    def workspace_id(self) -> int:
        """Identity of the single arbitration channel. Two ids would mean two minds — the tests
        assert this is one."""
        inner = getattr(self.workspace, "_gw", None)
        return id(inner) if inner is not None else id(self.workspace)

    def stats(self) -> Dict[str, Any]:
        ws: Dict[str, Any] = {}
        try:
            if self.workspace is not None:
                ws = self.workspace.stats()
        except Exception:  # noqa: BLE001
            ws = {}
        return {
            "cycles": self.cycles,
            "workspace_id": self.workspace_id(),
            "substrates": [s.stats() for s in self.substrates],
            "bidders": sorted({b for c in self.history for b in c.bidders}),
            "memory": self.memory.stats(),
            "workspace": ws,
            "note": ("architectural unification — one workspace, one memory façade, one report; "
                     "it makes the substrates compete on one stage but makes none of them "
                     "smarter, and it is not evidence of phenomenal unity"),
        }

    def summary(self) -> str:
        st = self.stats()
        lines = ["=" * 70, "NYXARA monad — the one mind", "=" * 70,
                 f"workspace : #{st['workspace_id']} (one channel)",
                 f"cycles    : {st['cycles']}",
                 f"memory    : {', '.join(st['memory']['stores']) or 'none'}"]
        for sub in st["substrates"]:
            mark = "✓" if sub["available"] else "·"
            lines.append(f"  {mark} {sub['name']:<18} bids {sub['bids']:>4}  wins {sub['wins']:>4}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA monad self-test")
    print("=" * 70)

    class _FakeNyx5:
        def __init__(self) -> None:
            self.store: Dict[str, str] = {}
            self.seen: List[str] = []

        def perceive(self, text: str) -> Any:
            self.seen.append(text)
            return type("T", (), {"surprise": 0.5})()

        def remember(self, key: str, text: str) -> None:
            self.store[key] = text

        def recall(self, text: str, **_kw: Any) -> Any:
            for key in self.store:
                if key.lower() in text.lower() or text.lower() in key.lower():
                    return type("R", (), {"key": key, "similarity": 0.9})()
            return None

    # 1) a substrate bids, and silence is allowed
    sub = Nyx5Substrate(_FakeNyx5())
    sub.engine.remember("kettle", "the kettle boiled")
    assert sub.consider(Situation(stimulus="tell me about the kettle")) is not None
    assert sub.consider(Situation(stimulus="unrelated")) is None
    assert sub.consider(Situation(stimulus="")) is None
    print("\nsubstrate bids : ok (and stays silent when it has nothing)")

    # 2) unified memory: written once, recalled from every store that has it
    from nyxara.nyx.holomem import HoloMemory

    holo, fake5 = HoloMemory(), _FakeNyx5()
    mem = UnifiedRecall(holomem=holo, nyx5=fake5)
    assert set(mem.remember("kettle", "the kettle boiled at dawn")) == {"holomem", "nyx5"}
    hits = mem.recall("kettle")
    assert hits, "a written fact must be recallable"
    print(f"unified recall : {[h.store for h in hits]}")

    # 3) a broken store costs its own results and nothing else
    class _Broken:
        def remember(self, *_a: Any, **_k: Any) -> None:
            raise RuntimeError("store is down")

        def recall(self, *_a: Any, **_k: Any) -> None:
            raise RuntimeError("store is down")

    mem2 = UnifiedRecall(holomem=holo, nyx5=_Broken())
    assert mem2.remember("k2", "still written") == ["holomem"]
    assert mem2.recall("k2"), "a broken store must not blind the recall"
    assert mem2.stats()["store_errors"].get("nyx5", 0) >= 1
    print("broken store   : contained")

    # 4) the real thing — one workspace, substrates joined, one winner
    from nyxara.kernel.config import NyxConfig
    from nyxara.nyx.brain import NyxBrain

    brain = NyxBrain(NyxConfig())
    core = type("FakeCore", (), {"nyx5": _FakeNyx5(), "sample_efficient": None,
                                 "intuition": None})()
    core.nyx5.remember("kettle", "the kettle boiled")
    monad = Monad(brain, core=core)

    assert monad.workspace is brain.workspace, "the brain's workspace must BE the one workspace"
    assert monad.workspace_id() == id(brain.workspace._gw), "one arbitration channel, not two"
    assert any(s.name == "nyx5" for s in monad.substrates), monad.substrates

    cycle = monad.deliberate("what happened with the kettle?")
    print(f"\ncycle          : winner={cycle.winner!r} bidders={cycle.bidders}")
    assert cycle.bidders, "somebody should bid"
    assert "nyx5" in cycle.bidders, "the spiking substrate must compete on the same stage"
    assert monad.substrates[0].broadcasts_received == 1, "everyone hears the broadcast"

    # joining twice must not buy a second vote
    assert monad.join(Nyx5Substrate(core.nyx5)) is False

    print("\n" + monad.summary())
    print("\nALL SELF-TESTS PASSED ✓")
