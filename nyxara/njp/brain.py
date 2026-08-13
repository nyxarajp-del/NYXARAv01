"""NYXARA · njp/brain.py — NJP V.01, the one brain (🧠, the single object the kernel holds).

:class:`NJPBrain` is what ``NyxaraCore`` builds and holds as ``self.njp``. It is the *only* brain
in NYXARA: there is no fusion of rival minds here and no vote to referee, because there is one
substrate — the Fluid Neural Automata in :mod:`nyxara.njp.fabric` — and everything else is an organ
attached to it.

**One turn of** :meth:`think` **is:**

1. **read the Master, including the half that was not said** — the Intent-Refinement Loop
   (:mod:`nyxara.njp.soulsync`) attaches the standing wants this request has earned, learned from
   corrections the Master actually made. What was *said* comes from :mod:`nyxara.njp.intent`.
2. **anticipate** — ask the manifold what this stimulus usually leads to, **before** settling. When
   that prediction is trusted the turn can answer from it; when it is not, that is reported and the
   fabric settles properly. Foresight that has not been earned is never spent.
3. **perceive** — words become cells, the cells are stimulated, and the automaton settles.
4. **ground** — content-addressed recall from :mod:`nyxara.njp.memory`. There is no token context
   window here; a fact from fifty turns ago is exactly as reachable as the last one. That is not
   the same as infinite, and :meth:`stats` reports the real capacity.
5. **judge** — anything she is about to state as fact goes through the Truth-Seeking Gauntlet
   (:mod:`nyxara.njp.truth`). An unestablished claim leaves labelled a conjecture, or is not made.
6. **expand** — the turn is handed to the pulse, which runs synaptogenesis, pruning and
   neurogenesis over it. **This is the "after every conversation" clause, and it is literal:** the
   synapse count after a turn is a different number from the one before it.

Everything is **fail-soft**: any organ that raises degrades to a null result and the turn still
completes. Every organ is optional; one that is off is *absent* from the reports rather than
reporting zeros, which is the distinction this whole package keeps.

**Sovereignty.** NJP only ever *proposes*. Every candidate flows through the kernel's unchanged,
fail-closed gate, and the safety core — corrigibility, oversight, loyalty, honesty — is never
governed, rewritten or bypassed by anything here. The mind proposes; the kernel disposes; the
Master is sovereign.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from nyxara.njp.fabric import Fabric, GrowthReport, SettleResult
from nyxara.njp.ledger import Ledger
from nyxara.njp.manifold import Prediction
from nyxara.njp.soulsync import Reading, SoulSync
from nyxara.njp.truth import Judgement, TruthGauntlet, Verdict

__all__ = ["NJPPercept", "NJPThought", "NJPBrain"]


def _cell_id(token: str) -> int:
    """A stable cell id for a concept label.

    Hashed rather than assigned from a counter so the *same word maps to the same cell across
    restarts and across machines* — without that, a reloaded fabric would be wired for concepts it
    could no longer name, which is the one way persistence can be worse than starting fresh.
    """
    return int(hashlib.blake2b(token.encode("utf-8"), digest_size=6).hexdigest(), 16)


@dataclass
class NJPPercept:
    """One perceive-tick: what the fabric did with a turn, and what memory brought back."""

    stimulus: str = ""
    concepts: List[str] = field(default_factory=list)
    cells: List[int] = field(default_factory=list)
    settled: Optional[SettleResult] = None
    recall: Any = None
    anticipated: Optional[Prediction] = None

    @property
    def novelty(self) -> float:
        """How unlike anything she knows this turn was — 0.0 familiar … 1.0 wholly new.

        Read straight off the manifold's familiarity, which is **graded and always measured**.
        Deliberately not ``settled.surprise``: that is the sharper signal about *this* turn and it
        feeds neurogenesis, but as a confidence discount it is far too blunt — it sits at 1.0 for
        every genuinely new prompt, including ordinary ones she answers perfectly well, and
        discounting those into abstention makes her useless rather than honest.
        Keying it on whether the prediction was *trusted* instead made this a step function: it
        sat at a flat 1.0 through every turn below the evidence floor, so a brain that was
        steadily learning still reported total ignorance right up to the moment it crossed. That
        matters because the reason-seat discounts confidence by this number — a step function
        there means she is uniformly, maximally unsure and then abruptly is not, rather than
        growing into confidence as she actually comes to recognise the ground.
        """
        ant = self.anticipated
        if ant is None:
            return 1.0
        return max(0.0, min(1.0, 1.0 - float(ant.familiarity)))

    def to_dict(self) -> Dict[str, Any]:
        return {"stimulus": self.stimulus[:200], "concepts": self.concepts[:32],
                "n_cells": len(self.cells), "novelty": round(self.novelty, 4),
                "settled": self.settled.to_dict() if self.settled else None,
                "anticipated": self.anticipated.to_dict() if self.anticipated else None}


@dataclass
class NJPThought:
    """One whole turn, with everything that produced it kept inspectable."""

    stimulus: str = ""
    percept: Optional[NJPPercept] = None
    reading: Optional[Reading] = None          # the Master's unsaid half
    intent: Any = None                         # what was actually asked
    answer: str = ""
    judgement: Optional[Judgement] = None
    growth: Optional[GrowthReport] = None
    cycle_id: str = ""
    ms: float = 0.0

    @property
    def confidence(self) -> float:
        return float(self.judgement.confidence) if self.judgement is not None else 0.0

    @property
    def verified(self) -> bool:
        """Was this **established** by the gauntlet? Confidence alone never sets this."""
        return bool(self.judgement is not None and self.judgement.established)

    @property
    def assertable(self) -> bool:
        """May she state this as fact? Only an established claim may be stated as one."""
        return bool(self.judgement is not None and self.judgement.assertable)

    def to_dict(self) -> Dict[str, Any]:
        return {"stimulus": self.stimulus[:200], "answer": self.answer[:400],
                "cycle_id": self.cycle_id, "confidence": round(self.confidence, 4),
                "verified": self.verified, "assertable": self.assertable,
                "ms": round(self.ms, 3),
                "percept": self.percept.to_dict() if self.percept else None,
                "reading": self.reading.to_dict() if self.reading else None,
                "intent": self.intent.to_dict() if hasattr(self.intent, "to_dict") else None,
                "judgement": self.judgement.to_dict() if self.judgement else None,
                "growth": self.growth.to_dict() if self.growth else None}


class NJPBrain:
    """NJP V.01 — one automaton, its organs, and the loop that grows it."""

    def __init__(self, config: Any = None) -> None:
        self.config = c = config
        self.turns = 0
        self.core: Any = None
        self.tools: Any = None
        self.knowledge: Any = None
        self.replay: List[str] = []

        self.fabric = self._build_fabric(c)
        self.ledger = Ledger() if self._gate("ledger", True) else None
        self.memory = self._build_memory(c)
        self.tongue = self._build_tongue(c)
        self.intent = self._build_intent(c)
        self.voice = self._build_voice(c)
        self.truth = self._build_truth(c)
        self.soul = self._build_soul(c)
        self.evolver = self._build_evolver(c)
        self.pulse = self._build_pulse(c)

    # ---- construction ----------------------------------------------------- #
    def _gate(self, name: str, default: bool = True) -> bool:
        return bool(getattr(self.config, f"{name}_enabled", default)) if self.config else default

    def _cfg(self, name: str, default: Any) -> Any:
        return getattr(self.config, name, default) if self.config else default

    def _build_fabric(self, c: Any) -> Fabric:
        """The one organ that is never optional. Without it there is no brain to configure."""
        try:
            return Fabric(
                seed=self._cfg("seed", 42),
                leak=self._cfg("leak", 0.85),
                threshold=self._cfg("threshold", 1.0),
                refractory=self._cfg("refractory", 2),
                max_settle_steps=self._cfg("max_settle_steps", 24),
                hebbian_rate=self._cfg("hebbian_rate", 0.08),
                depress_rate=self._cfg("depress_rate", 0.02),
                prune_threshold=self._cfg("prune_threshold", 0.015),
                growth_budget=self._cfg("growth_budget", 256),
                neurogenesis_error=self._cfg("neurogenesis_error", 0.55),
                neurogenesis_batch=self._cfg("neurogenesis_batch", 8),
                soft_cell_ceiling=self._cfg("soft_cell_ceiling", 250_000),
                soft_synapse_ceiling=self._cfg("soft_synapse_ceiling", 4_000_000),
                manifold_dim=self._cfg("manifold_dim", 10000))
        except Exception:  # noqa: BLE001 — a default fabric beats no fabric
            return Fabric()

    def _build_memory(self, c: Any) -> Any:
        if not self._gate("memory", True):
            return None
        try:
            from nyxara.njp.memory import HoloMemory
            return HoloMemory(dim=self._cfg("holo_dim", 4096),
                              capacity=self._cfg("holo_capacity", 20000),
                              seed=self._cfg("seed", 42))
        except Exception:  # noqa: BLE001 — she thinks without long-term recall, less well
            return None

    def _build_tongue(self, c: Any) -> Any:
        if not self._gate("tongue", True):
            return None
        try:
            from nyxara.njp.tongue import Lingua
            return Lingua()
        except Exception:  # noqa: BLE001 — falls back to the ASCII concept floor
            return None

    def _build_intent(self, c: Any) -> Any:
        if not self._gate("intent", True):
            return None
        try:
            from nyxara.njp.intent import IntentReader
            return IntentReader(lingua=self.tongue)
        except Exception:  # noqa: BLE001
            return None

    def _build_voice(self, c: Any) -> Any:
        if not self._gate("voice", True):
            return None
        try:
            from nyxara.njp.voice import Dialogue
            return Dialogue(max_tokens=self._cfg("reply_max_tokens", 220))
        except Exception:  # noqa: BLE001
            return None

    def _build_truth(self, c: Any) -> Optional[TruthGauntlet]:
        if not self._gate("truth", True):
            return None
        try:
            return TruthGauntlet(min_sources=self._cfg("truth_min_sources", 2),
                                 require_hard=self._cfg("truth_require_hard", True),
                                 ledger=self.ledger)
        except Exception:  # noqa: BLE001 — without it nothing is verified, and nothing claims to be
            return None

    def _build_soul(self, c: Any) -> Optional[SoulSync]:
        if not self._gate("soulsync", True):
            return None
        try:
            return SoulSync(min_support=self._cfg("soulsync_min_support", 2))
        except Exception:  # noqa: BLE001
            return None

    def _build_evolver(self, c: Any) -> Any:
        if not self._gate("evolve", True):
            return None
        try:
            from nyxara.njp.evolve import SelfEvolver
            return SelfEvolver(self, ledger=self.ledger,
                               every_s=self._cfg("evolve_every_s", 300.0),
                               min_gain=self._cfg("evolve_min_gain", 0.05))
        except Exception:  # noqa: BLE001 — without it she stays exactly as she was written
            return None

    def _build_pulse(self, c: Any) -> Any:
        if not self._gate("pulse", True):
            return None
        try:
            from nyxara.njp.pulse import PulseEngine
            return PulseEngine(self, every_s=self._cfg("pulse_every_s", 1.0),
                               consolidate_every_s=self._cfg("consolidate_every_s", 60.0),
                               evolve_every_s=self._cfg("evolve_every_s", 300.0))
        except Exception:  # noqa: BLE001 — without it she grows only when spoken to
            return None

    def attach_kernel(self, *, tools: Any = None, knowledge: Any = None,
                      core: Any = None) -> None:
        """Take the live toolset and knowledge base. Without this her brain is blind to tools."""
        self.tools = tools
        self.knowledge = knowledge
        self.core = core

    # ---- encoding --------------------------------------------------------- #
    def concepts(self, text: str) -> List[str]:
        try:
            from nyxara.njp.tongue import concepts_in
            return concepts_in(text)
        except Exception:  # noqa: BLE001
            return [w for w in str(text or "").lower().split() if len(w) > 2][:32]

    def encode(self, text: str) -> List[int]:
        """Words to cells. Stable across restarts, which is what makes the fabric reloadable."""
        return [_cell_id(tok) for tok in self.concepts(text)]

    # ---- perception ------------------------------------------------------- #
    def perceive(self, text: str, *, remember: bool = True) -> NJPPercept:
        """Stimulate the fabric with a turn, settle it, and bring back grounded context."""
        out = NJPPercept(stimulus=str(text or ""))
        try:
            if not out.stimulus.strip():
                return out
            out.concepts = self.concepts(out.stimulus)
            out.cells = self.encode(out.stimulus)
            if not out.cells:
                return out

            # Ask what usually follows this BEFORE settling — the pre-cognitive read.
            out.anticipated = self.fabric.anticipate(out.cells)

            self.fabric.stimulate(out.cells)
            out.settled = self.fabric.settle()

            if self.memory is not None:
                try:
                    out.recall = self.memory.recall(out.stimulus,
                                                    k=self._cfg("recall_k", 5))
                except Exception:  # noqa: BLE001
                    out.recall = None
            return out
        except Exception:  # noqa: BLE001 — a perceive failure never breaks a turn
            return out

    # ---- the turn --------------------------------------------------------- #
    def think(self, stimulus: str, *, remember: bool = True,
              outcome: float = 1.0) -> NJPThought:
        """One whole turn: read → anticipate → perceive → ground → judge → expand."""
        out = NJPThought(stimulus=str(stimulus or ""))
        t0 = time.perf_counter()
        try:
            if not out.stimulus.strip():
                return out
            self.turns += 1
            out.cycle_id = f"njp-{self.turns}"

            # 1. the Master's unsaid half, and what was actually asked
            if self.soul is not None:
                out.reading = self.soul.read(out.stimulus)
            if self.intent is not None:
                try:
                    out.intent = self.intent.read(out.stimulus)
                except Exception:  # noqa: BLE001
                    out.intent = None

            # 2-4. perceive and ground
            out.percept = self.perceive(out.stimulus, remember=remember)
            out.answer = self._compose(out)

            # 5. nothing is stated as fact until the gauntlet says it may be
            if self.truth is not None and out.answer:
                out.judgement = self.truth.judge(out.answer, context=self)

            # 6. remember the conclusion, not the question: storing a bare question would make it
            # its own best match and echo back as its own answer.
            if remember and out.answer and self.memory is not None:
                try:
                    self.memory.remember(f"turn-{self.turns}", out.answer,
                                         kind=("conclusion" if out.verified else "episode"),
                                         cue=out.stimulus)
                except Exception:  # noqa: BLE001
                    pass

            # 7. EXPAND — the physical growth, after every single turn
            out.growth = self._expand(out, outcome=outcome)

            out.ms = (time.perf_counter() - t0) * 1000.0
            if self.ledger is not None:
                self.ledger.observe_latency(out.ms)
            if self.evolver is not None:
                self.evolver.observe("nyxara.njp.fabric", out.ms)
            self._remember_sample(out.stimulus)
            return out
        except Exception:  # noqa: BLE001 — a thought that fails is empty, never fatal
            out.ms = (time.perf_counter() - t0) * 1000.0
            return out

    def _expand(self, thought: NJPThought, *, outcome: float) -> Optional[GrowthReport]:
        """Grow on this turn, synchronously, and return what actually changed.

        Deliberately **not** routed through the pulse. "After every conversation" has to mean the
        synapse count is different when the turn returns, not that growth was queued and might
        happen on some later beat — and a caller that never beats (a one-shot CLI, a test) must
        still see the fabric expand. The pulse's queue is for experience that arrives *outside* a
        turn; letting it also carry turns would double-count every one of them.
        """
        try:
            settled = thought.percept.settled if thought.percept else None
            if settled is None:
                return None
            return self.fabric.expand(outcome=outcome, result=settled)
        except Exception:  # noqa: BLE001
            return None

    def _compose(self, thought: NJPThought) -> str:
        """What she can honestly say about this turn, from recall and from what settled.

        Deliberately plain. This brain's contribution is *what is true and what fired*, not
        fluency; :mod:`nyxara.njp.voice` phrases it when a fluent surface is installed, and says so
        when one is not rather than letting a fallback babble in her name.
        """
        try:
            bits: List[str] = []
            rec = thought.percept.recall if thought.percept else None
            best = getattr(rec, "best", None) if rec is not None else None
            if best is not None and getattr(best, "text", ""):
                bits.append(str(best.text))
            ant = thought.percept.anticipated if thought.percept else None
            if ant is not None and ant.trusted:
                bits.append(f"this usually leads to {len(ant.cells)} familiar activations")
            if thought.reading is not None and thought.reading.has_latent:
                wants = "; ".join(w.want for w in thought.reading.latent[:2])
                bits.append(f"you usually also want: {wants}")
            return " — ".join(bits)[:1000]
        except Exception:  # noqa: BLE001
            return ""

    def converse(self, text: str) -> Any:
        """Think, then say it. Her content; a fluent surface only phrases it."""
        try:
            thought = self.think(text)
            if self.voice is None:
                return thought.answer
            return self.voice.respond(thought, brain=self)
        except Exception:  # noqa: BLE001 — she always says something, never crashes mid-sentence
            return ""

    # ---- feedback --------------------------------------------------------- #
    def resolve(self, thought: Any, *, correct: float) -> None:
        """Tell the brain how a turn actually turned out. Drives plasticity and the soul-sync."""
        try:
            settled = getattr(getattr(thought, "percept", None), "settled", None)
            if settled is not None:
                # Map correctness onto the plasticity signal: right reinforces what fired, wrong
                # weakens it. This is the only place an outcome becomes structure.
                self.fabric.expand(outcome=(float(correct) * 2.0 - 1.0), result=settled)
            if self.soul is not None:
                self.soul.confirm(corrected=(float(correct) < 0.5))
        except Exception:  # noqa: BLE001
            pass

    def correct(self, before: str, after: str) -> Any:
        """The Master re-issued a request with something added. Learn the standing rule."""
        try:
            return self.soul.observe_correction(before, after) if self.soul is not None else None
        except Exception:  # noqa: BLE001
            return None

    def _remember_sample(self, text: str) -> None:
        """Keep a small replay set — the held-out evidence the self-editor is judged against."""
        try:
            self.replay.append(str(text)[:200])
            if len(self.replay) > 64:
                del self.replay[:-64]
        except Exception:  # noqa: BLE001
            pass

    def measure_sample(self, sample: Any) -> float:
        """Re-run one replay sample and return what it costs now, in ms.

        This is the hook :mod:`nyxara.njp.evolve` calls to find out whether an edit actually made
        her faster. Without a real measurement the self-editor refuses the edit, so this returning
        a real number is what makes the improvement claim checkable rather than asserted.
        """
        try:
            t0 = time.perf_counter()
            self.perceive(str(sample), remember=False)
            return (time.perf_counter() - t0) * 1000.0
        except Exception:  # noqa: BLE001
            return float("inf")

    # ---- the beat --------------------------------------------------------- #
    def tick(self, *, oversight: Any = None) -> Dict[str, Any]:
        """One beat of the continuous loop. Driven by the kernel's clock, never its own thread."""
        try:
            if self.pulse is None:
                return {}
            return self.pulse.beat(oversight=oversight).to_dict()
        except Exception:  # noqa: BLE001 — a background beat never raises into the loop
            return {}

    def anticipate(self, text: str = "") -> Any:
        """What she expects next — either from this stimulus, or from the Master's own pattern."""
        try:
            if text:
                return self.fabric.anticipate(self.encode(text))
            return self.soul.anticipate() if self.soul is not None else None
        except Exception:  # noqa: BLE001
            return None

    # ---- reporting -------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"turns": self.turns, "fabric": self.fabric.stats()}
        for name, organ in (("ledger", self.ledger), ("truth", self.truth),
                            ("soulsync", self.soul), ("evolve", self.evolver),
                            ("pulse", self.pulse)):
            if organ is None:
                continue                       # an organ that is off is ABSENT, not zeroed
            try:
                out[name] = organ.stats()
            except Exception:  # noqa: BLE001
                out[name] = {"error": "stats failed"}
        if self.memory is not None:
            try:
                out["memory"] = {"traces": len(self.memory)}
            except Exception:  # noqa: BLE001
                pass
        return out

    def report(self) -> Dict[str, Any]:
        """Then vs now — is she actually more than she was? Numbers, not a claim."""
        try:
            return self.ledger.report() if self.ledger is not None else {}
        except Exception:  # noqa: BLE001
            return {}

    # ---- persistence ------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"turns": self.turns, "fabric": self.fabric.to_dict()}
        for name, organ in (("ledger", self.ledger), ("soulsync", self.soul)):
            if organ is None:
                continue
            try:
                d[name] = organ.to_dict()
            except Exception:  # noqa: BLE001
                continue
        if self.memory is not None and hasattr(self.memory, "to_dict"):
            try:
                d["memory"] = self.memory.to_dict()
            except Exception:  # noqa: BLE001
                pass
        return d

    def load_dict(self, d: Dict[str, Any]) -> None:
        """Wake up as the brain that went to sleep. This is what makes tomorrow's NJP today's."""
        try:
            self.turns = int(d.get("turns", 0))
            if d.get("fabric"):
                self.fabric.load_dict(d["fabric"])
            if d.get("ledger") and self.ledger is not None:
                self.ledger.load_dict(d["ledger"])
            if d.get("soulsync") and self.soul is not None:
                self.soul.load_dict(d["soulsync"])
            if d.get("memory") and self.memory is not None and hasattr(self.memory, "load_dict"):
                self.memory.load_dict(d["memory"])
        except Exception:  # noqa: BLE001 — a corrupt sidecar leaves a freshly-born brain
            pass
