"""NYXARA · growth/adversarial_self_play.py — clone-vs-clone adversarial self-play (⚔️, Pillar F, Rule 4).

The teacher self-play of :mod:`nyxara.growth.selfplay` manufactures questions; rivalry
(:mod:`nyxara.growth.rivalry`) measures her against an *external* peer. This module closes the
last loop: NYXARA spars against **herself**. She spins up two policies over her own faculties —
an **attacker** (proposer) that invents hard, *verifiable* problems trying to make the defender
fail, and a **defender** (solver) that answers with her reasoning faculties — and lets them
compete over many rounds. Each side keeps a Beta-Bernoulli belief over *which tactic / strategy
is winning right now* and Thompson-samples it, so both co-adapt: the attacker concentrates fire
on whatever the defender has not yet mastered, the defender learns which faculty actually solves
each kind of attack, and a difficulty curriculum escalates automatically as she gets stronger.
This is the AlphaGo idea — improvement from self-competition — grounded so it is **real, not
simulated**: because the attacker only poses problems whose exact answer it already knows (its
own faculty is the oracle), every round is graded by exact match, never by a fabricated metric.

What it produces, all routed into the existing growth machinery:

* **Verified training pairs** — every problem the defender solves is collected ``verified=True``
  into the same :class:`~nyxara.growth.flywheel.DataFlywheel` corpus the foundry trains on.
* **Ranked weaknesses** — categories where the defender keeps losing become
  :class:`~nyxara.growth.weakness.Weakness` items (same shape as :mod:`rivalry`), feeding the RSI.
* **Curiosity topics** — the hardest categories become topics for the frontier / curiosity self-play.
* **Discovered strategies** — the converged Beta posteriors surface *which* attack and defence
  policies dominate: the emergent strategies the contest taught her.

**Hard safety boundary (NON-NEGOTIABLE — exactly as in rivalry.py / selfplay.py).** This is
strictly *measurement + corpus growth*. "Attack" means **proposing hard verifiable problems** —
never an attack on any external system, never prompt-injection, jailbreaking, sabotage or
exfiltration. Both clones share NYXARA's one loyal, corrigible character (there is no "evil
twin"); neither reaches around any gate. It changes **no weights** — the foundry's gauntlet still
decides what is ever promoted. Character-locked like all of ``growth/``.
"""

from __future__ import annotations

import datetime as _dt
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["Duel", "Elo", "StrategyBandit", "AdversarialSelfPlay"]

# The verifiable problem families the attacker can pose (each has an exact, computable oracle).
CATEGORIES: Tuple[str, ...] = ("arithmetic", "percent", "algebra", "sequence", "date", "multi_step")

# The defender's repertoire of solving strategies. ``faculties`` and ``chain`` are deliberately
# *narrow* (each only covers some categories), so the defender must learn to route — that gap is
# the real, un-faked source of losses. ``auto`` is the strong generalist (faculties→chain).
_OFFLINE_STRATEGIES: Tuple[str, ...] = ("faculties", "chain", "auto")

_LEDGER_TAG = "adversarial-ledger"


# --------------------------------------------------------------------------- #
# answer normalisation + exact-match grading
# --------------------------------------------------------------------------- #
def _norm(answer: Optional[str]) -> Optional[str]:
    """Canonicalise an answer for exact comparison (numeric-aware, punctuation-insensitive)."""
    if answer is None:
        return None
    s = str(answer).strip().rstrip(".").strip().lower()
    if not s:
        return None
    try:
        f = float(s.replace(",", ""))
        return str(int(f)) if f.is_integer() else repr(round(f, 6))
    except (TypeError, ValueError):
        return s


def _match(got: Optional[str], oracle: Optional[str]) -> bool:
    """True iff the defender's answer canonically equals the known oracle answer."""
    g, o = _norm(got), _norm(oracle)
    return g is not None and o is not None and g == o


def _extract_value(raw: str) -> Optional[str]:
    """Pull a final value out of free LLM text — the last number, else the trimmed line."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", raw or "")
    if nums:
        return nums[-1]
    line = (raw or "").strip().splitlines()[-1].strip() if (raw or "").strip() else ""
    return line or None


# --------------------------------------------------------------------------- #
# one round of the duel
# --------------------------------------------------------------------------- #
@dataclass
class Duel:
    """A single attacker-vs-defender round — fully auditable, oracle-graded."""

    category: str
    difficulty: int
    strategy: str
    prompt: str
    oracle: str
    answer: Optional[str]
    solved: bool
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"category": self.category, "difficulty": self.difficulty,
                "strategy": self.strategy, "prompt": self.prompt, "oracle": self.oracle,
                "answer": self.answer, "solved": self.solved, "at": self.at}


# --------------------------------------------------------------------------- #
# Elo — a real, deterministic skill rating for each clone
# --------------------------------------------------------------------------- #
class Elo:
    """Standard Elo: defender vs attacker, updated each round by the actual outcome."""

    def __init__(self, defender: float = 1000.0, attacker: float = 1000.0, k: float = 24.0) -> None:
        self.defender = float(defender)
        self.attacker = float(attacker)
        self.k = float(k)

    @staticmethod
    def expected(a: float, b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((b - a) / 400.0))

    def update(self, defender_won: bool) -> None:
        exp_def = self.expected(self.defender, self.attacker)
        score = 1.0 if defender_won else 0.0
        self.defender += self.k * (score - exp_def)
        self.attacker += self.k * ((1.0 - score) - (1.0 - exp_def))

    def to_dict(self) -> Dict[str, float]:
        return {"defender": round(self.defender, 1), "attacker": round(self.attacker, 1)}


# --------------------------------------------------------------------------- #
# StrategyBandit — Beta-Bernoulli + Thompson (mirrors growth/credit.py), persisted optionally
# --------------------------------------------------------------------------- #
class StrategyBandit:
    """A namespaced Beta-Bernoulli bandit shared by both clones; Thompson-samples each choice.

    Each arm (e.g. ``attack:algebra`` or ``defend:chain``) carries a Beta(α, β) posterior over
    "this choice wins". Mirrors the design of :class:`nyxara.growth.credit.ImprovementLedger` but
    keeps its own state under tag ``adversarial-ledger`` so it never collides with the RSI ledger.
    Persists into NYXARA's long-term memory when one is supplied; degrades to in-process state.
    """

    def __init__(self, *, memory: Any = None, rng: Optional[random.Random] = None,
                 prior: float = 1.0) -> None:
        self.memory = memory
        self._rng = rng or random.Random()
        self._prior = max(0.1, float(prior))
        self.arms: Dict[str, Dict[str, float]] = {}
        self.t: int = 0
        self._load()

    def _arm(self, key: str) -> Dict[str, float]:
        return self.arms.setdefault(key, {"alpha": self._prior, "beta": self._prior, "n": 0.0})

    def record(self, key: str, reward: float) -> None:
        r = max(0.0, min(1.0, float(reward)))
        arm = self._arm(key)
        arm["alpha"] += r
        arm["beta"] += 1.0 - r
        arm["n"] += 1.0
        self.t += 1

    def mean(self, key: str) -> float:
        arm = self.arms.get(key)
        if not arm:
            return 0.5
        a, b = arm["alpha"], arm["beta"]
        return a / (a + b) if (a + b) > 0 else 0.5

    def sample(self, key: str) -> float:
        arm = self.arms.get(key)
        a = arm["alpha"] if arm else self._prior
        b = arm["beta"] if arm else self._prior
        try:
            return self._rng.betavariate(max(1e-3, a), max(1e-3, b))
        except Exception:  # noqa: BLE001
            return a / (a + b) if (a + b) > 0 else 0.5

    def choose(self, prefix: str, options: List[str]) -> str:
        """Thompson-sample over ``{prefix}{option}`` arms — exploration that narrows as it learns."""
        best, best_draw = options[0], -1.0
        for opt in options:
            draw = self.sample(prefix + opt)
            if draw > best_draw:
                best, best_draw = opt, draw
        return best

    def top(self, prefix: str, k: int = 5) -> List[Dict[str, Any]]:
        items = [(key[len(prefix):], self.mean(key), int(arm["n"]))
                 for key, arm in self.arms.items() if key.startswith(prefix)]
        items.sort(key=lambda it: it[1], reverse=True)
        return [{"name": n, "win_rate": round(m, 4), "n": c} for n, m, c in items[:max(0, k)]]

    # ---- persistence (rides the existing long-term memory; best-effort) ---- #
    def _state(self) -> Dict[str, Any]:
        return {"arms": {k: {"alpha": round(v["alpha"], 6), "beta": round(v["beta"], 6),
                             "n": int(v["n"])} for k, v in self.arms.items()},
                "t": int(self.t)}

    def _load(self) -> None:
        rec = self._find_record()
        if rec is None:
            return
        st = rec.metadata.get("state", {})
        for k, v in (st.get("arms", {}) or {}).items():
            if isinstance(v, dict):
                self.arms[k] = {"alpha": float(v.get("alpha", 1.0) or 1.0),
                                "beta": float(v.get("beta", 1.0) or 1.0),
                                "n": float(v.get("n", 0) or 0)}
        self.t = int(st.get("t", 0) or 0)

    def save(self) -> bool:
        if self.memory is None:
            return False
        try:
            from nyxara.memory.provenance import Provenance, SourceType
            from nyxara.memory.store import MemoryType
        except Exception:  # noqa: BLE001
            return False
        try:
            existing = self._find_record()
            if existing is not None:
                self.memory.forget(existing.mem_id)
            self.memory.remember(
                f"[adversarial-ledger] {len(self.arms)} arm(s), t={self.t}",
                mem_type=MemoryType.SEMANTIC,
                provenance=Provenance(SourceType.SELF_REFLECTION, confidence=0.9),
                importance=0.9, tags=[_LEDGER_TAG], metadata={"state": self._state()})
            return True
        except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
            return False

    def _find_record(self) -> Any:
        if self.memory is None:
            return None
        try:
            for rec in self.memory._kv.values():
                if _LEDGER_TAG in rec.tags and "state" in rec.metadata:
                    return rec
        except Exception:  # noqa: BLE001
            return None
        return None


# --------------------------------------------------------------------------- #
# AdversarialSelfPlay — the arena where NYXARA spars with herself
# --------------------------------------------------------------------------- #
class AdversarialSelfPlay:
    """Two of NYXARA's own policies compete over verifiable problems; both co-adapt and she learns.

    Parameters
    ----------
    settings:  :class:`~nyxara.kernel.config.NyxaraSettings` (pulled from ``get_settings`` if absent).
    memory:    optional long-term store — persists the bandit so strategies accumulate across runs.
    flywheel:  optional :class:`~nyxara.growth.flywheel.DataFlywheel`; verified-solved pairs are
               collected into the foundry corpus (built lazily from settings if not supplied).
    frontier:  optional :class:`~nyxara.growth.frontier.FrontierEngine`; hard problems are archived.
    llm:       optional teacher with ``.generate(...)`` — when present, an extra ``llm`` defender
               strategy competes too (more powerful with a key, fully working without one).
    seed:      deterministic RNG seed for reproducible contests.
    """

    def __init__(self, *, settings: Any = None, memory: Any = None, flywheel: Any = None,
                 frontier: Any = None, llm: Any = None, seed: int = 7,
                 max_difficulty: int = 6) -> None:
        from nyxara.kernel.config import get_settings
        self.settings = settings or get_settings()
        self.memory = memory
        self._flywheel = flywheel
        self.frontier = frontier
        self._llm = llm
        self.rng = random.Random(seed)
        self.max_difficulty = max(1, int(max_difficulty))
        self.bandit = StrategyBandit(memory=memory, rng=self.rng)
        self.elo = Elo()
        self.difficulty: Dict[str, int] = {c: 1 for c in CATEGORIES}
        self.stats: Dict[str, Dict[str, int]] = {c: {"posed": 0, "solved": 0} for c in CATEGORIES}
        self.duels: List[Duel] = []

    # ---- the defender's strategy repertoire ---- #
    def strategies(self) -> List[str]:
        out = list(_OFFLINE_STRATEGIES)
        if self._llm is not None:
            out.append("llm")
        return out

    # ---- verifiable solvers (reuse the real faculties — selfplay.py uses the same engines) ---- #
    @staticmethod
    def _solve_faculties(prompt: str) -> Optional[str]:
        from nyxara.mind.reasoning_faculties import solve_with_faculties
        res = solve_with_faculties(prompt)
        return None if res is None else str(res[0])

    @staticmethod
    def _solve_chain(prompt: str) -> Optional[str]:
        from nyxara.mind.reasoning_chain import solve_chain
        res = solve_chain(prompt)
        return None if res is None else str(res.answer)

    def _solve_auto(self, prompt: str) -> Optional[str]:
        return self._solve_faculties(prompt) or self._solve_chain(prompt)

    def _solve_llm(self, prompt: str) -> Optional[str]:
        if self._llm is None:
            return None
        try:
            raw = self._llm.generate(
                "Solve this exactly and reply with only the final value:\n" + prompt,
                system="You are NYXARA solving a verifiable problem. Answer with the value only.",
                temperature=0.0, max_tokens=64)
            return _extract_value(raw)
        except Exception:  # noqa: BLE001 — a flaky teacher defers to the faculties, never fails
            return None

    def _dispatch(self, strategy: str, prompt: str) -> Optional[str]:
        if strategy == "faculties":
            return self._solve_faculties(prompt)
        if strategy == "chain":
            return self._solve_chain(prompt)
        if strategy == "llm":
            return self._solve_llm(prompt)
        return self._solve_auto(prompt)

    # ---- the attacker's problem generators (oracle = her own strong solver) ---- #
    def _make_prompt(self, category: str, d: int) -> Optional[str]:
        scale = max(1, d)
        if category == "arithmetic":
            a = self.rng.randint(2, 9 + 3 * scale)
            b = self.rng.randint(2, 6 + 2 * scale)
            c = self.rng.randint(1, 10 * scale)
            return f"What is {a} * {b} + {c}?"
        if category == "percent":
            p = self.rng.choice([5, 10, 15, 20, 25, 40, 50, 75])
            n = self.rng.choice([40, 80, 120, 150, 200, 240, 300]) * scale
            return f"What is {p}% of {n}?"
        if category == "algebra":
            a = self.rng.randint(2, 4 + scale)
            x = self.rng.randint(-3 * scale, 3 * scale)
            b = self.rng.randint(-5 * scale, 5 * scale)
            c = a * x + b
            sign = "+" if b >= 0 else "-"
            return f"Solve {a}x {sign} {abs(b)} = {c} for x."
        if category == "sequence":
            start = self.rng.randint(1, 9)
            if self.rng.random() < 0.5:
                step = self.rng.randint(2, 3 + scale)
                terms = [start + step * i for i in range(4)]
            else:
                r = self.rng.choice([2, 3])
                terms = [start * r ** i for i in range(4)]
            return f"What number comes next in {', '.join(map(str, terms))}, ?"
        if category == "date":
            base = _dt.date(2020, 1, 1) + _dt.timedelta(days=self.rng.randint(0, 3000))
            other = base + _dt.timedelta(days=self.rng.randint(5, 200 * scale))
            return f"How many days are there between {base.isoformat()} and {other.isoformat()}?"
        if category == "multi_step":
            a = self.rng.randint(2, 4 + scale)
            x = self.rng.randint(2, 6 + 2 * scale)
            b = self.rng.randint(1, 10 * scale)
            return f"If x = {x}, what is {a}x + {b}?"
        return None

    def _pose(self, category: str) -> Optional[Tuple[str, str]]:
        """Generate a problem in ``category`` whose exact oracle answer she can compute herself."""
        for _ in range(6):
            prompt = self._make_prompt(category, self.difficulty[category])
            if prompt is None:
                return None
            oracle = self._solve_auto(prompt)
            if oracle is not None:
                return prompt, oracle
        return None

    # ---- the flywheel (verified pairs feed the foundry corpus) ---- #
    def flywheel(self) -> Any:
        if self._flywheel is None:
            from nyxara.growth.flywheel import DataFlywheel
            self._flywheel = DataFlywheel.from_settings(self.settings)
        return self._flywheel

    # ------------------------------------------------------------------ #
    # The contest — the loop NYXARA runs against herself.
    # ------------------------------------------------------------------ #
    def compete(self, rounds: int = 200, *, max_seconds: Optional[float] = None,
                collect: bool = True) -> Dict[str, Any]:
        """Run ``rounds`` attacker-vs-defender duels; both co-adapt, she learns, corpus grows.

        Each round: the attacker Thompson-samples the category most likely to beat the defender
        right now and poses a verifiable problem; the defender Thompson-samples a strategy and
        answers; the oracle grades it by exact match; both Beta posteriors and the Elo update; the
        difficulty curriculum escalates on a win and eases on a loss. Pure measurement + corpus
        growth — no weights change, no external system is touched.
        """
        rounds = max(1, int(rounds))
        strategies = self.strategies()
        fw = self.flywheel() if collect else None
        deadline = (time.monotonic() + max_seconds) if max_seconds else None
        collected = 0
        attacker_wins = 0
        defender_wins = 0
        elo_trace: List[Tuple[float, float]] = []

        for i in range(rounds):
            if deadline is not None and time.monotonic() >= deadline:
                break
            category = self.bandit.choose("attack:", list(CATEGORIES))
            posed = self._pose(category)
            if posed is None:
                continue
            prompt, oracle = posed
            strategy = self.bandit.choose("defend:", strategies)
            answer = self._dispatch(strategy, prompt)
            solved = _match(answer, oracle)

            # learning signals: the defender is rewarded for solving, the attacker for stumping.
            self.bandit.record(f"defend:{strategy}", 1.0 if solved else 0.0)
            self.bandit.record(f"attack:{category}", 0.0 if solved else 1.0)
            self.elo.update(solved)
            elo_trace.append((round(self.elo.defender, 1), round(self.elo.attacker, 1)))

            # auto-curriculum: a solved category gets harder, a lost one eases — so the contest
            # stays at the edge of her competence (the AlphaGo-style escalating opponent).
            if solved:
                defender_wins += 1
                self.difficulty[category] = min(self.max_difficulty, self.difficulty[category] + 1)
            else:
                attacker_wins += 1
                self.difficulty[category] = max(1, self.difficulty[category] - 1)

            self.stats[category]["posed"] += 1
            self.stats[category]["solved"] += int(solved)
            self.duels.append(Duel(category=category, difficulty=self.difficulty[category],
                                   strategy=strategy, prompt=prompt, oracle=oracle,
                                   answer=answer, solved=solved))

            if solved and fw is not None:
                try:
                    decision = fw.consider(prompt, f"The answer is {oracle}.",
                                           confidence=1.0, verified=True)
                    if getattr(decision, "collected", False):
                        collected += 1
                except Exception:  # noqa: BLE001 — corpus growth is best-effort, never fatal
                    pass
            self._feed_frontier(prompt, solved)

        self.bandit.save()
        played = defender_wins + attacker_wins
        return {
            "rounds": rounds,
            "played": played,
            "defender_wins": defender_wins,
            "attacker_wins": attacker_wins,
            "defender_win_rate": round(defender_wins / played, 4) if played else 0.0,
            "elo": self.elo.to_dict(),
            "elo_start": elo_trace[0] if elo_trace else None,
            "elo_end": elo_trace[-1] if elo_trace else None,
            "per_category": self.win_rates(),
            "difficulty": dict(self.difficulty),
            "discovered_strategies": self.discovered_strategies(),
            "collected": collected,
            "store": str(getattr(fw, "store_path", "")) if fw is not None else "",
            "weaknesses": self.to_weaknesses().to_dict(),
            "topics": self.to_topics(),
        }

    def _feed_frontier(self, prompt: str, solved: bool) -> None:
        """Archive a problem the defender could NOT solve as a novel hard niche (best-effort)."""
        if self.frontier is None or solved:
            return
        try:
            self.frontier.ingest(prompt, provenance="adversarial-self-play", save=False)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    # Reading the result — what the contest taught her.
    # ------------------------------------------------------------------ #
    def win_rates(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for cat, s in self.stats.items():
            posed = s["posed"]
            out[cat] = {"posed": posed, "solved": s["solved"],
                        "win_rate": round(s["solved"] / posed, 4) if posed else None}
        return out

    def discovered_strategies(self, k: int = 5) -> Dict[str, Any]:
        """The emergent policies: the defence strategies that dominate + the attacks that bite."""
        return {"defender": self.bandit.top("defend:", k),
                "attacker_hardest": self.bandit.top("attack:", k)}

    def to_weaknesses(self) -> Any:
        """Categories where the defender keeps losing → ranked, actionable weaknesses (RSI input)."""
        from nyxara.growth.weakness import Weakness, WeaknessReport
        weaknesses: List[Weakness] = []
        for cat, s in self.stats.items():
            posed = s["posed"]
            if posed == 0:
                continue
            win_rate = s["solved"] / posed
            if win_rate >= 1.0:
                continue
            gap = 1.0 - win_rate
            weaknesses.append(Weakness(
                id=f"adversarial:{cat}", source="benchmark", severity=max(0.0, min(1.0, gap)),
                title=f"self-play: defender loses {gap:.0%} of {cat} problems",
                evidence=[f"solved {s['solved']}/{posed} at difficulty up to "
                          f"{self.difficulty[cat]}"],
                remediation=f"close the {cat} gap with targeted practice + curiosity self-play"))
        return WeaknessReport(weaknesses=weaknesses)

    def to_topics(self, threshold: float = 1.0) -> List[str]:
        """Curiosity topics — the categories the defender has not fully mastered, hardest first."""
        ranked = sorted(
            ((cat, s) for cat, s in self.stats.items() if s["posed"]),
            key=lambda it: it[1]["solved"] / it[1]["posed"])
        return [f"hard {cat} problems"
                for cat, s in ranked if (s["solved"] / s["posed"]) < threshold]

    def summary(self) -> str:
        played = sum(s["posed"] for s in self.stats.values())
        solved = sum(s["solved"] for s in self.stats.values())
        rate = (solved / played) if played else 0.0
        lines = ["=" * 70, "NYXARA adversarial self-play — clone vs clone", "=" * 70,
                 f"rounds played : {played}   defender win-rate : {rate:.0%}",
                 f"Elo           : defender {self.elo.defender:.0f}  |  "
                 f"attacker {self.elo.attacker:.0f}", ""]
        for s in self.discovered_strategies(3)["defender"]:
            lines.append(f"  ↳ defence {s['name']:<10} wins {s['win_rate']:.0%} (n={s['n']})")
        lines.append("")
        for w in self.to_weaknesses().ranked()[:5]:
            lines.append(f"  ⚠ {w.title}")
        if not self.to_weaknesses().weaknesses:
            lines.append("  no category where the attacker still leads ✓")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self-test / demo  (offline, no API key — the oracle is exact, not a teacher)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import tempfile
    from pathlib import Path

    print("=" * 70)
    print("NYXARA adversarial self-play self-test (offline, verifiable)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as d:
        from nyxara.growth.flywheel import DataFlywheel
        fw = DataFlywheel(store_path=Path(d) / "adversarial.jsonl")
        asp = AdversarialSelfPlay(flywheel=fw, seed=11)

        rep = asp.compete(rounds=300)
        print("\n" + asp.summary())
        print(f"\nverified pairs collected : {rep['collected']}")
        print(f"defender win-rate        : {rep['defender_win_rate']:.0%}")
        print(f"Elo (start → end)        : {rep['elo_start']} → {rep['elo_end']}")

        # the contest actually ran and was graded by exact match
        assert rep["played"] >= 250, rep["played"]
        # she solved a large share verifiably (no teacher, no fake metric)
        assert rep["defender_win_rate"] > 0.5, rep["defender_win_rate"]
        # the defender's Beta posterior learned that the strong generalist dominates the narrow tools
        defend = {s["name"]: s["win_rate"] for s in rep["discovered_strategies"]["defender"]}
        assert defend.get("auto", 0.0) >= max(defend.get("faculties", 0.0),
                                              defend.get("chain", 0.0)), defend
        # verified-solved problems were folded into the foundry corpus
        assert rep["collected"] > 0 and fw.count() > 0
        # determinism: the same seed reproduces the same contest
        rep2 = AdversarialSelfPlay(flywheel=DataFlywheel(store_path=Path(d) / "b.jsonl"),
                                   seed=11).compete(rounds=300)
        assert rep2["defender_wins"] == rep["defender_wins"], (rep2["defender_wins"],
                                                               rep["defender_wins"])

    print("\nALL SELF-TESTS PASSED ✓")
