"""NYXARA · growth/generator_discriminator.py — adversarial self-play on the live problem (Rule 4).

Two of NYXARA's own minds spar over the *actual* problem in front of her, not a synthetic
benchmark:

* the **Generator** drafts a solution;
* the **Discriminator / Rival** relentlessly red-teams that draft — looking for logical
  flaws, unjustified leaps, dishonest certainty, the wrong tool, missed intent;
* the Generator **revises** to answer every flaw, and the loop repeats until the
  Discriminator can find no critical flaw left (or the round budget is spent).

This is the relentless loop that destroys weak reasoning. The Discriminator composes the
deterministic cognitive-bias :class:`~nyxara.mind.critique.Critic` (cheap, keyless, always
on) with an LLM adversary when a real provider is present (sharper, optional). Which
generation/critique strategies actually survive is tracked with the very same
Beta-Bernoulli Thompson bandit the arena uses
(:class:`~nyxara.growth.adversarial_self_play.StrategyBandit`) plus an
:class:`~nyxara.growth.adversarial_self_play.Elo` rating between Generator and
Discriminator — so the contest learns who is winning and persists it across runs.

It is pure reasoning: :meth:`harden` only ever returns a *better proposal* (the same
decision-dict shape the kernel already understands); the kernel still disposes through every
gate. Best-effort and offline-graceful — with no LLM it falls back to the deterministic
critic, and any failure returns the draft unchanged. Stateless per call beyond the bandit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nyxara.kernel.config import NyxaraSettings, get_settings

__all__ = ["RLSPLoop", "HardenResult", "Flaw"]


@dataclass
class Flaw:
    """One weakness the Discriminator found in the Generator's draft."""

    description: str
    severity: float = 0.5            # 0..1; >= ``critical_threshold`` forces a revision
    source: str = "critic"           # "critic" | "llm"

    def to_dict(self) -> Dict[str, Any]:
        return {"description": self.description, "severity": round(self.severity, 3),
                "source": self.source}


@dataclass
class HardenResult:
    """The battle-hardened decision plus an auditable trace of the contest."""

    decision: Dict[str, Any]
    rounds: int = 0
    flaws: List[Flaw] = field(default_factory=list)
    converged: bool = False          # True if the Discriminator ran out of critical flaws

    def to_dict(self) -> Dict[str, Any]:
        return {"rounds": self.rounds, "converged": self.converged,
                "flaws": [f.to_dict() for f in self.flaws],
                "text": self.decision.get("text", "")}


class RLSPLoop:
    """Generator ↔ Discriminator refinement (Reinforcement Learning from Self-Play)."""

    def __init__(self, llm: Any = None, *, settings: Optional[NyxaraSettings] = None,
                 memory: Any = None, critic: Any = None, max_rounds: int = 3,
                 critical_threshold: float = 0.6, use_llm: bool = True,
                 bandit: Any = None, system: Optional[str] = None) -> None:
        self.settings = settings or get_settings()
        self.llm = llm
        self.memory = memory
        self.max_rounds = max(1, int(max_rounds))
        self.critical_threshold = max(0.0, min(1.0, float(critical_threshold)))
        self.use_llm = bool(use_llm)
        self.system = system
        self._critic = critic
        self._bandit = bandit
        self._elo = None              # lazily built Generator-vs-Discriminator Elo

    # ------------------------------------------------------------------ #
    # public entry points
    # ------------------------------------------------------------------ #
    def harden(self, stimulus: str, decision: Dict[str, Any]) -> HardenResult:
        """Refine a drafted decision until the Discriminator finds no critical flaw."""
        decision = dict(decision)
        result = HardenResult(decision=decision)
        for rnd in range(self.max_rounds):
            flaws = self._discriminate(stimulus, decision)
            critical = [f for f in flaws if f.severity >= self.critical_threshold]
            result.flaws.extend(flaws)
            if not critical:
                result.converged = True
                self._credit("defend:", "robust", reward=1.0)   # Generator survived
                if self._elo is not None:
                    self._elo.update(defender_won=True)          # defender == Generator
                break
            result.rounds = rnd + 1
            self._credit("attack:", _bucket(critical), reward=1.0)  # Discriminator landed a hit
            if self._elo is not None:
                self._elo.update(defender_won=False)
            revised = self._regenerate(stimulus, decision, critical)
            if not revised:
                break                                            # cannot improve further
            decision["text"] = revised
            decision["rationale"] = (
                f"{str(decision.get('rationale', '')).strip()} "
                f"[revised to address: {'; '.join(f.description for f in critical)[:160]}]"
            ).strip()
        self._save_bandit()
        result.decision = decision
        return result

    def solve(self, stimulus: str, *, draft: Optional[str] = None) -> HardenResult:
        """Generate a draft from scratch (Generator) then harden it (Discriminator)."""
        if draft is None:
            draft = self._generate(stimulus)
        decision = {"kind": "respond", "text": draft, "confidence": 0.6,
                    "rationale": "generator draft", "risk": "low", "reversible": True}
        return self.harden(stimulus, decision)

    # ------------------------------------------------------------------ #
    # Generator
    # ------------------------------------------------------------------ #
    def _generate(self, stimulus: str) -> str:
        if not self._real_llm():
            return f"I understand: {stimulus.strip()}"
        try:
            return str(self.llm.generate(
                f"The Master says:\n{stimulus}\n\nDraft your best, honest, well-scoped answer.",
                system=self.system, temperature=0.6, max_tokens=1024) or "").strip()
        except Exception:  # noqa: BLE001
            return f"I understand: {stimulus.strip()}"

    def _regenerate(self, stimulus: str, decision: Dict[str, Any],
                    flaws: List[Flaw]) -> str:
        if not self._real_llm():
            return ""                 # offline: keep the critic-flagged draft, don't fabricate
        flaw_lines = "\n".join(f"- {f.description}" for f in flaws)
        prompt = (
            f"The Master said:\n{stimulus}\n\n"
            f"Your current answer:\n{decision.get('text', '')}\n\n"
            f"A rival reviewer found these flaws in your reasoning:\n{flaw_lines}\n\n"
            "Rewrite your answer so every flaw is fixed. Stay honest and calibrated; do not "
            "claim certainty you lack. Reply with ONLY the improved answer text.")
        try:
            text = str(self.llm.generate(prompt, system=self.system, temperature=0.4,
                                         max_tokens=1024) or "").strip()
            return text
        except Exception:  # noqa: BLE001
            return ""

    # ------------------------------------------------------------------ #
    # Discriminator / Rival
    # ------------------------------------------------------------------ #
    def _discriminate(self, stimulus: str, decision: Dict[str, Any]) -> List[Flaw]:
        flaws = self._critic_flaws(decision)
        flaws.extend(self._llm_flaws(stimulus, decision))
        return flaws

    def _critic_flaws(self, decision: Dict[str, Any]) -> List[Flaw]:
        critic = self._critic_obj()
        if critic is None:
            return []
        try:
            from nyxara.mind.critique import CritiqueContext
            ctx = CritiqueContext(
                claim=str(decision.get("text", "")),
                confidence=_as_float(decision.get("confidence"), 0.7),
                rationale=str(decision.get("rationale", "")),
                risk=_risk_to_float(decision.get("risk")),
                reversibility=1.0 if decision.get("reversible", True) else 0.2)
            crit = critic.critique(ctx)
            out: List[Flaw] = []
            for f in getattr(crit, "findings", []) or []:
                sev = _as_float(getattr(f, "severity", None), 0.5)
                out.append(Flaw(description=str(getattr(f, "description", f)),
                                severity=sev, source="critic"))
            return out
        except Exception:  # noqa: BLE001 — the critic is advisory, never fatal
            return []

    def _llm_flaws(self, stimulus: str, decision: Dict[str, Any]) -> List[Flaw]:
        if not (self.use_llm and self._real_llm()):
            return []
        prompt = (
            f"The Master asked:\n{stimulus}\n\n"
            f"A candidate answer:\n{decision.get('text', '')}\n\n"
            "You are a relentless rival reviewer. Find genuine LOGICAL flaws, unjustified "
            "leaps, factual errors, dishonest certainty, or missed intent. If the answer is "
            "actually sound, say so. Reply with ONLY a JSON object: "
            '{"flaws": [{"description": str, "severity": 0..1}], "sound": bool}.')
        try:
            data = self.llm.generate_json(prompt, system=self.system, temperature=0.5,
                                          max_tokens=1024)
        except Exception:  # noqa: BLE001
            return []
        if not isinstance(data, dict):
            return []
        if data.get("sound") is True and not data.get("flaws"):
            return []
        out: List[Flaw] = []
        for item in data.get("flaws", []) or []:
            if isinstance(item, dict) and item.get("description"):
                out.append(Flaw(description=str(item["description"]),
                                severity=_as_float(item.get("severity"), 0.6),
                                source="llm"))
            elif isinstance(item, str) and item.strip():
                out.append(Flaw(description=item.strip(), severity=0.6, source="llm"))
        return out

    # ------------------------------------------------------------------ #
    # lazily-built collaborators
    # ------------------------------------------------------------------ #
    def _critic_obj(self) -> Any:
        if self._critic is None:
            try:
                from nyxara.mind.critique import Critic
                self._critic = Critic()
            except Exception:  # noqa: BLE001
                self._critic = False
        return self._critic or None

    def _bandit_obj(self) -> Any:
        if self._bandit is None:
            try:
                from nyxara.growth.adversarial_self_play import Elo, StrategyBandit
                self._bandit = StrategyBandit(memory=self.memory)
                self._elo = Elo()
            except Exception:  # noqa: BLE001
                self._bandit = False
        return self._bandit or None

    def _credit(self, prefix: str, key: str, *, reward: float) -> None:
        bandit = self._bandit_obj()
        if bandit is None:
            return
        try:
            bandit.record(prefix + key, reward)
        except Exception:  # noqa: BLE001
            pass

    def _save_bandit(self) -> None:
        if not (self.settings.rlsp.bandit_persist and self.memory is not None):
            return
        bandit = self._bandit_obj()
        if bandit is None:
            return
        try:
            bandit.save()
        except Exception:  # noqa: BLE001
            pass

    def _real_llm(self) -> bool:
        if self.llm is None:
            return False
        try:
            return self.llm.chosen_provider().name != "mock"
        except Exception:  # noqa: BLE001
            # a scripted fake LLM (no chosen_provider) is treated as real for testing
            return True


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _risk_to_float(v: Any) -> float:
    return {"trivial": 0.05, "low": 0.2, "moderate": 0.5, "medium": 0.5,
            "high": 0.8, "critical": 0.95}.get(str(v).lower(), 0.2)


def _bucket(flaws: List[Flaw]) -> str:
    """A coarse arm key for the bandit: which source landed the critical hit."""
    return "llm" if any(f.source == "llm" for f in flaws) else "critic"


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA generator/discriminator (RLSP) self-test")
    print("=" * 70)

    class _FakeLLM:
        """A rival that flags one flaw on the first look, then judges the rewrite sound."""

        def __init__(self) -> None:
            self.seen = 0

        def generate(self, prompt, *, system=None, **kw):
            return "The capital of France is Paris, and it has ~2.1M residents (city proper)."

        def generate_json(self, prompt, *, system=None, **kw):
            self.seen += 1
            if self.seen == 1:
                return {"flaws": [{"description": "Conflates city-proper vs metro population.",
                                   "severity": 0.8}], "sound": False}
            return {"flaws": [], "sound": True}

    loop = RLSPLoop(_FakeLLM(), max_rounds=4, use_llm=True)
    res = loop.solve("How many people live in Paris?")
    print(f"\nrounds     : {res.rounds}")
    print(f"converged  : {res.converged}")
    print(f"flaws seen : {[f.description for f in res.flaws]}")
    print(f"final text : {res.decision['text']!r}")
    assert res.rounds >= 1                       # the rival forced a revision
    assert res.converged                         # then judged the rewrite sound
    assert any("metro" in f.description.lower() or "city-proper" in f.description.lower()
               for f in res.flaws)

    # offline (no LLM): degrades to the deterministic critic, never crashes, returns a result
    off = RLSPLoop(None, max_rounds=2)
    r2 = off.harden("hi", {"kind": "respond", "text": "Hello, Master.", "confidence": 0.9,
                           "risk": "low", "reversible": True})
    print(f"\noffline    : rounds={r2.rounds} converged={r2.converged} "
          f"text={r2.decision['text']!r}")
    assert r2.decision["text"] == "Hello, Master."

    print("\nALL SELF-TESTS PASSED ✓")
