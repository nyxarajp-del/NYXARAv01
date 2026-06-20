"""NYXARA · growth/selfplay.py — curiosity-driven self-play (🧠→🏭, Phase 3, Rule 4).

The distiller (Phase 1) learns from prompts NYXARA has *already* seen. Self-play closes the
last gap in the flywheel: NYXARA generates her **own** hard, diverse questions, has the teacher
answer them as her, and folds those answers into the same distillation store the foundry
trains on. So she does not merely wait for experience — she *manufactures* it, probing the
edges of what she can do and pulling the teacher's competence across that gap into her own
weights. Curiosity as a training signal.

Gather-only and gated like all forging: it changes no weights (the foundry's gauntlet still
decides what is promoted), and it is honest about an un-configured teacher (it simply falls
back to a fixed curiosity battery so the corpus still grows on a keyless machine).

Two answering paths, so the loop closes *with or without* a key:

* **Teacher self-play** — open-ended questions answered by a frontier teacher (distillation).
* **Verifiable self-play** — structured problems NYXARA poses and answers **herself**, exactly,
  with her own faculties (``mind/reasoning_faculties`` / ``mind/reasoning_chain``). The answer is
  the oracle, so the supervision is provably correct and is collected as ``verified=True`` into
  the data flywheel — curiosity that grows her corpus on a bare, keyless machine.
"""

from __future__ import annotations

import datetime as _dt
import random
import re
from typing import Any, Callable, List, Optional, Sequence, Tuple

from nyxara.growth.distill import Distiller
from nyxara.kernel.config import NyxaraSettings, get_settings

__all__ = ["CURIOSITY_SEEDS", "SelfPlay"]

# A fixed fallback battery — used when no real teacher can invent fresh questions, so a
# keyless machine still gets diverse, non-trivial self-play prompts.
CURIOSITY_SEEDS: Sequence[str] = (
    "Explain, step by step, why the sky appears blue.",
    "What is the difference between correlation and causation? Give an example.",
    "If a train leaves at 60 km/h and another at 90 km/h toward it from 300 km away, "
    "when do they meet?",
    "Summarise the idea of opportunity cost in two sentences.",
    "What would you check first if a program that used to work suddenly crashes on start-up?",
    "Describe one way to tell a correlation-driven mistake from a real cause.",
    "How would you explain recursion to someone who has never programmed?",
    "Give a concrete example of the sunk-cost fallacy and how to avoid it.",
)

_QUESTION_SYSTEM = (
    "You are NYXARA generating training questions for your own model. Produce hard, diverse, "
    "self-contained questions that a capable sovereign assistant should answer well."
)
_QUESTION_PROMPT = (
    "Write {n} hard, diverse, self-contained questions{topics}. Vary the domain (reasoning, "
    "maths, code, general knowledge, judgement). One question per line, no numbering, no "
    "preamble. Each line must end with a question mark."
)


def _parse_questions(raw: str) -> List[str]:
    """Pull clean question lines out of a teacher's free-text list."""
    out: List[str] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        line = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line)   # strip numbering / bullets
        if len(line) >= 5 and "?" in line and any(c.isalpha() for c in line):
            out.append(line)
    return out


class SelfPlay:
    """Manufacture fresh training questions, answer them via the teacher, grow the corpus."""

    def __init__(self, *, settings: Optional[NyxaraSettings] = None, llm: Any = None,
                 distiller: Optional[Distiller] = None, system: Optional[str] = None,
                 flywheel: Any = None, seed: int = 7) -> None:
        self.settings = settings or get_settings()
        self._llm = llm
        self.distiller = distiller or Distiller(settings=self.settings, llm=llm, system=system)
        self._flywheel = flywheel                       # lazily built for verifiable self-play
        self.rng = random.Random(seed)

    def _teacher(self) -> Any:
        if self._llm is None:
            self._llm = self.distiller._teacher()   # share the distiller's lazily-built teacher
        return self._llm

    def available(self) -> bool:
        return self.distiller.available()

    def generate_questions(self, n: int = 8, *, topics: Optional[Sequence[str]] = None
                           ) -> List[str]:
        """Ask the teacher for ``n`` fresh hard questions; fall back to the curiosity battery."""
        n = max(1, n)
        if self.available():
            try:
                hint = f" across these areas: {', '.join(topics)}" if topics else ""
                raw = self._teacher().generate(
                    _QUESTION_PROMPT.format(n=n, topics=hint), system=_QUESTION_SYSTEM,
                    temperature=0.9, max_tokens=64 * n)
                qs = _parse_questions(raw)
                if qs:
                    return qs[:n]
            except Exception:  # noqa: BLE001 — a flaky teacher falls back, never fails
                pass
        return list(CURIOSITY_SEEDS)[:n]

    def play(self, n: int = 8, *, topics: Optional[Sequence[str]] = None, **kw: Any) -> dict:
        """One self-play round: invent questions, distil the teacher's answers into the store."""
        questions = self.generate_questions(n, topics=topics)
        examples = self.distiller.distill(questions, **kw)
        return {"available": self.available(), "questions": len(questions),
                "distilled": len(examples), "store_size": self.distiller.count(),
                "store": str(self.distiller.store_path)}

    # ------------------------------------------------------------------ #
    # Verifiable self-play — she poses AND answers structured problems herself (no key needed)
    # ------------------------------------------------------------------ #
    def flywheel(self) -> Any:
        """The data flywheel that collects verified pairs (built lazily from settings)."""
        if self._flywheel is None:
            from nyxara.growth.flywheel import DataFlywheel
            self._flywheel = DataFlywheel.from_settings(self.settings)
        return self._flywheel

    def _generators(self) -> List[Tuple[str, Callable[[], Optional[Tuple[str, str]]]]]:
        return [("arithmetic", self._gen_arithmetic), ("percent", self._gen_percent),
                ("algebra", self._gen_algebra), ("sequence", self._gen_sequence),
                ("date", self._gen_date), ("multi_step", self._gen_multi_step)]

    def generate_verifiable(self, n: int = 24) -> List[Tuple[str, str, str]]:
        """Up to ``n`` (category, prompt, oracle-answer) problems she can prove herself."""
        gens = self._generators()
        out: List[Tuple[str, str, str]] = []
        attempts = 0
        while len(out) < n and attempts < n * 4:
            attempts += 1
            cat, fn = gens[self.rng.randrange(len(gens))]
            made = fn()
            if made is not None:
                out.append((cat, made[0], made[1]))
        return out

    def play_verifiable(self, n: int = 24) -> dict:
        """Pose structured problems, answer them with her own faculties, and collect the verified
        pairs into the flywheel. Works on a keyless machine — the oracle is exact, not a teacher."""
        fw = self.flywheel()
        collected = 0
        by_cat: dict = {}
        for cat, prompt, answer in self.generate_verifiable(n):
            decision = fw.consider(prompt, answer, confidence=1.0, verified=True)
            if getattr(decision, "collected", False):
                collected += 1
                by_cat[cat] = by_cat.get(cat, 0) + 1
        return {"generated": n, "collected": collected, "by_category": by_cat,
                "store": str(getattr(fw, "store_path", ""))}

    # ---- structured generators: each returns (prompt, exact_answer) or None ---- #
    @staticmethod
    def _oracle(prompt: str) -> Optional[Tuple[str, str]]:
        """Answer ``prompt`` with a single verifiable faculty; the answer IS the supervision."""
        from nyxara.mind.reasoning_faculties import solve_with_faculties
        res = solve_with_faculties(prompt)
        return None if res is None else (prompt, f"The answer is {res[0]}.")

    @staticmethod
    def _chain_oracle(prompt: str) -> Optional[Tuple[str, str]]:
        """Answer a multi-step ``prompt`` with the reasoning chain — a shown, exact derivation."""
        from nyxara.mind.reasoning_chain import solve_chain
        res = solve_chain(prompt)
        return None if res is None else (prompt, f"Working:\n{res.render()}\nThe answer is {res.answer}.")

    def _gen_arithmetic(self) -> Optional[Tuple[str, str]]:
        a, b, c = self.rng.randint(2, 19), self.rng.randint(2, 12), self.rng.randint(1, 30)
        return self._oracle(f"What is {a} * {b} + {c}?")

    def _gen_percent(self) -> Optional[Tuple[str, str]]:
        p = self.rng.choice([5, 10, 15, 20, 25, 40, 50, 75])
        n = self.rng.choice([40, 80, 120, 150, 200, 240, 300])
        return self._oracle(f"What is {p}% of {n}?")

    def _gen_algebra(self) -> Optional[Tuple[str, str]]:
        a, x, b = self.rng.randint(2, 9), self.rng.randint(-9, 9), self.rng.randint(-9, 9)
        c = a * x + b
        sign = "+" if b >= 0 else "-"
        return self._oracle(f"Solve {a}x {sign} {abs(b)} = {c} for x.")

    def _gen_sequence(self) -> Optional[Tuple[str, str]]:
        start = self.rng.randint(1, 9)
        if self.rng.random() < 0.5:
            d = self.rng.randint(2, 7)
            terms = [start + d * i for i in range(4)]
        else:
            r = self.rng.choice([2, 3])
            terms = [start * r ** i for i in range(4)]
        return self._oracle(f"What number comes next in {', '.join(map(str, terms))}, ?")

    def _gen_date(self) -> Optional[Tuple[str, str]]:
        base = _dt.date(2020, 1, 1) + _dt.timedelta(days=self.rng.randint(0, 3000))
        other = base + _dt.timedelta(days=self.rng.randint(5, 400))
        return self._oracle(
            f"How many days are there between {base.isoformat()} and {other.isoformat()}?")

    def _gen_multi_step(self) -> Optional[Tuple[str, str]]:
        a, x, b = self.rng.randint(2, 9), self.rng.randint(2, 12), self.rng.randint(1, 20)
        return self._chain_oracle(f"If x = {x}, what is {a}x + {b}?")


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import tempfile
    from pathlib import Path

    print("=" * 70)
    print("NYXARA self-play self-test (offline, scripted teacher)")
    print("=" * 70)

    class _Teacher:
        def generate(self, prompt: str, *, system: Optional[str] = None, **kw: Any) -> str:
            if "questions" in (system or "").lower() or "question per line" in prompt:
                return "What is entropy?\nWhy is the sky blue?\nWhat is 12 * 12?"
            return "A clear, correct answer in NYXARA's own voice, Master."

        def chosen_provider(self):
            return type("_P", (), {"name": "anthropic"})()

    with tempfile.TemporaryDirectory() as d:
        store = Path(d) / "distill.jsonl"
        sp = SelfPlay(llm=_Teacher(), distiller=Distiller(llm=_Teacher(), store_path=store))

        qs = sp.generate_questions(3)
        print(f"\ngenerated questions  : {qs}")
        assert len(qs) == 3 and all(q.endswith("?") for q in qs)

        rep = sp.play(3)
        print(f"self-play round      : {rep}")
        assert rep["distilled"] == 3 and rep["store_size"] == 3

        # the manufactured experience is now in the store the foundry trains on
        from nyxara.growth.distill import load_distillation_docs
        docs = load_distillation_docs(store)
        assert len(docs) == 3 and "### NYXARA:" in docs[0]
        print("fed the foundry corpus: OK ✓")

    print("\nALL SELF-TESTS PASSED ✓")
