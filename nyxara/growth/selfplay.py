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
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence

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
                 distiller: Optional[Distiller] = None, system: Optional[str] = None) -> None:
        self.settings = settings or get_settings()
        self._llm = llm
        self.distiller = distiller or Distiller(settings=self.settings, llm=llm, system=system)

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
