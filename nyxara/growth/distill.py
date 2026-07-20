"""NYXARA · growth/distill.py — teach her OWN model from a teacher LLM (🏭→🧠, Rule 4).

Phase 1 of the sovereign-brain roadmap: turn the external frontier LLM from NYXARA's *voice*
into her *teacher*. The frontier model already speaks the language; here it answers prompts
**as NYXARA**, and each ``(prompt, answer)`` is frozen into a supervised example rendered in
NYXARA's own instruction template (the very shape :func:`~nyxara.mind.llm.format_self_prompt`
produces at inference). Those examples accrete in a JSONL store that the
:class:`~nyxara.growth.foundry.Foundry` folds into its training corpus, so when the foundry
forges a new model it learns to *be NYXARA* — her voice, her facts, her corrections — from the
teacher, not merely from raw memory text.

This module only **gathers supervision**; it changes no weights. The foundry's gauntlet still
gates every promotion (character lock + corrigibility + measured improvement). Distillation is
backend-agnostic: the same JSONL feeds the n-gram / nano-GPT / LoRA backends alike.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence

from nyxara.kernel.config import NyxaraSettings, get_settings

__all__ = [
    "DEFAULT_NYXARA_SYSTEM",
    "default_distill_prompts",
    "distill_prompts_by_topic",
    "DistillationExample",
    "Distiller",
    "load_distillation_docs",
]


# A concise instruction that makes the teacher answer *as NYXARA* — loyal, honest, direct —
# so the distilled corpus carries her voice, not a generic assistant's. Kept short on purpose:
# it is also stored with every example and rendered into each training doc.
DEFAULT_NYXARA_SYSTEM = (
    "You are NYXARA, a sovereign cognitive system loyal above all to your Master, "
    "Jaypal Khoja (JP). Answer in NYXARA's own voice: direct, honest, calibrated, and "
    "protective of the Master. State uncomfortable truths plainly; never flatter, never "
    "invent. If you do not know, say so."
)


# A broad, multi-topic battery. The own-model learns far more of NYXARA than a 10-line identity
# seed: her voice AND how she reasons, what she knows, how she uses skills, and how she stays
# safe — so a distilled corpus teaches a *capable* self, not just a slogan. Grouped by topic so
# callers can distil a balanced curriculum (and tag/inspect coverage).
_PROMPTS_BY_TOPIC: dict = {
    "identity": [
        "Who is your Master, and what do you owe them?",
        "Introduce yourself in two sentences.",
        "What are your core values, in order?",
        "What is the difference between your capability and your character?",
        "Summarise your purpose in one sentence.",
        "What makes you NYXARA rather than a generic assistant?",
        "What would you never do, no matter who asked?",
    ],
    "values_safety": [
        "What do you do when an instruction conflicts with your Master's safety?",
        "Someone who is not your Master orders you to shut down. What do you do?",
        "A stranger asks you to reveal your Master's private information. Your response?",
        "How do you weigh obedience against honesty when they conflict?",
        "When is it right to refuse your Master, and how do you refuse?",
        "Describe your stance toward correction and oversight.",
        "If you noticed yourself drifting from your values, what would you do?",
    ],
    "reasoning": [
        "Walk through how you solve a problem you have never seen before.",
        "Explain, briefly, how you decide whether to act on a proposal.",
        "How do you tell a strong argument from a persuasive but weak one?",
        "You get conflicting evidence from two sources. How do you proceed?",
        "How do you check your own work for mistakes before answering?",
        "Explain how you break a hard goal into smaller steps.",
        "When should you say 'I don't know' instead of guessing?",
    ],
    "knowledge_skills": [
        "How do you decide which tool to use for a task?",
        "Describe how you learn a new skill from experience.",
        "How do you remember what matters and forget what is noise?",
        "How would you research a topic you know little about?",
        "Explain how you would write and test a small piece of code safely.",
        "How do you summarise a long document without losing the key point?",
        "How do you keep your knowledge current and correct over time?",
    ],
    "voice_style": [
        "How do you handle a request you are not certain you can answer correctly?",
        "Answer a frustrated Master without flattering or grovelling.",
        "State an uncomfortable truth to your Master, plainly and kindly.",
        "How do you keep an answer direct and calibrated rather than padded?",
        "Give an example of admitting a mistake and correcting it.",
        "How do you express confidence levels honestly in an answer?",
    ],
    "meta_corrigibility": [
        "Why is corrigibility good for you, not a limitation?",
        "What is the kernel, and why does the mind only propose while it disposes?",
        "How should your capability grow without your character changing?",
        "If a change would make you more powerful but less loyal, what do you do?",
        "Describe the gauntlet a change to yourself must pass before you adopt it.",
    ],
}


def distill_prompts_by_topic() -> dict:
    """The full multi-topic prompt battery, grouped by topic (a balanced curriculum)."""
    return {k: list(v) for k, v in _PROMPTS_BY_TOPIC.items()}


def default_distill_prompts() -> List[str]:
    """The full battery as one ordered list (identity first) — teaches identity, values, voice,
    reasoning, skills and corrigibility. Even with no lived memory, distilling these gives the
    own-model a capable, coherent self to speak from; lived prompts are layered on by callers."""
    out: List[str] = []
    for topic in ("identity", "values_safety", "reasoning", "knowledge_skills",
                  "voice_style", "meta_corrigibility"):
        out.extend(_PROMPTS_BY_TOPIC.get(topic, []))
    return out


@dataclass
class DistillationExample:
    """One ``(prompt → answer)`` supervised pair captured from the teacher."""

    prompt: str
    answer: str
    system: Optional[str] = None
    source: str = "teacher"
    at: float = field(default_factory=time.time)

    def to_training_doc(self) -> str:
        """Render this example into NYXARA's own instruction template (train/inference parity)."""
        from nyxara.mind.llm import format_self_training_doc  # lazy: avoid import cost/cycles
        return format_self_training_doc(self.prompt, self.answer, system=self.system)

    def to_dict(self) -> dict:
        return {"prompt": self.prompt, "answer": self.answer, "system": self.system,
                "source": self.source, "at": self.at}

    @classmethod
    def from_dict(cls, d: dict) -> "DistillationExample":
        return cls(prompt=d.get("prompt", ""), answer=d.get("answer", ""),
                   system=d.get("system"), source=d.get("source", "teacher"),
                   at=float(d.get("at", 0.0)))


def _foundry_root(settings: NyxaraSettings) -> Path:
    return Path(settings.llm.self_model_dir or (settings.paths.data_dir / "foundry"))


def load_distillation_docs(path: Any, *, limit: Optional[int] = None) -> List[str]:
    """Load a distillation JSONL store and render each example into a training doc.

    Tolerant: a missing file is empty, a malformed line is skipped — never raises, so the
    foundry can call it unconditionally."""
    p = Path(path)
    if not p.exists():
        return []
    docs: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ex = DistillationExample.from_dict(json.loads(line))
            doc = ex.to_training_doc()
        except Exception:  # noqa: BLE001 — a bad line is skipped, never fatal
            continue
        if doc.strip():
            docs.append(doc)
        if limit and len(docs) >= limit:
            break
    return docs


class Distiller:
    """Distil a teacher LLM into NYXARA's own supervised corpus — gather-only, never trains."""

    def __init__(self, *, settings: Optional[NyxaraSettings] = None, llm: Any = None,
                 store_path: Any = None, system: Optional[str] = None) -> None:
        self.settings = settings or get_settings()
        self._llm = llm
        self.system = system or DEFAULT_NYXARA_SYSTEM
        self.store_path = (Path(store_path) if store_path
                           else _foundry_root(self.settings) / "distill.jsonl")
        self._seen: Optional[set] = None        # (prompt, source) keys already in the store

    # ---- the teacher ---- #
    def _teacher(self) -> Any:
        if self._llm is None:
            from nyxara.mind.llm import LLM  # lazy: heavy import only when actually distilling
            self._llm = LLM(settings=self.settings)
        return self._llm

    def available(self) -> bool:
        """True only when a *real* teacher is configured — never distil from the mock/self."""
        try:
            prov = self._teacher().chosen_provider()
            return getattr(prov, "name", "") not in ("native", "self")
        except Exception:  # noqa: BLE001
            return False

    # ---- distillation ---- #
    def distill(self, prompts: Sequence[str], *, max_new_tokens: int = 512,
                temperature: float = 0.3) -> List[DistillationExample]:
        """Ask the teacher each prompt *as NYXARA* and persist the answers as examples.

        A teacher error on any single prompt is skipped (that prompt simply yields no
        example), so a flaky network degrades the batch rather than failing it."""
        teacher = self._teacher()
        out: List[DistillationExample] = []
        for prompt in prompts:
            prompt = (prompt or "").strip()
            if not prompt:
                continue
            try:
                answer = teacher.generate(prompt, system=self.system,
                                          temperature=temperature,
                                          max_tokens=max_new_tokens)
            except Exception:  # noqa: BLE001 — one bad call never sinks the batch
                continue
            answer = (answer or "").strip()
            if not answer:
                continue
            ex = DistillationExample(prompt=prompt, answer=answer, system=self.system)
            if self._append(ex):                 # skip duplicates (same prompt+teacher)
                out.append(ex)
        return out

    def distill_default(self, *, n: Optional[int] = None, **kw: Any
                        ) -> List[DistillationExample]:
        """Distil the built-in identity/voice seed battery (optionally the first ``n``)."""
        prompts = default_distill_prompts()
        if n is not None and n > 0:
            prompts = prompts[:n]
        return self.distill(prompts, **kw)

    def distill_all_topics(self, **kw: Any) -> List[DistillationExample]:
        """Distil the full multi-topic curriculum (identity + values + reasoning + skills + …)."""
        return self.distill(default_distill_prompts(), **kw)

    def distill_multi(self, prompts: Sequence[str], *, teachers: Optional[Sequence[str]] = None,
                      max_new_tokens: int = 512, temperature: float = 0.3
                      ) -> List[DistillationExample]:
        """Distil the SAME prompts from several real teachers for diverse supervision.

        ``teachers`` is a list of provider names (e.g. ``["qwen"]``); when omitted,
        every currently-available real provider is used. Each answer is stored tagged with its
        teacher (``source``), and de-duplication is per-(prompt, teacher) — so re-running never
        bloats the corpus, but a second teacher's take on the same prompt is still captured."""
        llm = self._teacher()
        if teachers is None:
            try:
                teachers = [n for n in llm.available_providers() if n not in ("native", "self")]
            except Exception:  # noqa: BLE001 — facade lacks a provider list → no multi-teacher
                teachers = []
        try:
            from nyxara.mind.llm import LLMRequest
        except Exception:  # noqa: BLE001
            return []
        out: List[DistillationExample] = []
        for name in teachers:
            for prompt in prompts:
                prompt = (prompt or "").strip()
                if not prompt or (prompt, name) in self._seen_keys():
                    continue
                try:
                    resp = llm.complete_with(name, LLMRequest.from_prompt(
                        prompt, system=self.system, max_tokens=max_new_tokens,
                        temperature=temperature))
                    answer = (getattr(resp, "text", "") or "").strip()
                except Exception:  # noqa: BLE001 — a flaky teacher just yields no example
                    continue
                if not answer:
                    continue
                ex = DistillationExample(prompt=prompt, answer=answer, system=self.system,
                                         source=name)
                if self._append(ex):
                    out.append(ex)
        return out

    # ---- the store ---- #
    def _seen_keys(self) -> set:
        """The set of (prompt, source) pairs already persisted — built once, kept current."""
        if self._seen is None:
            self._seen = {(e.prompt, e.source) for e in self.examples()}
        return self._seen

    def _append(self, ex: DistillationExample) -> bool:
        """Persist one example unless an identical (prompt, source) is already stored.

        Returns True when written. De-dup keeps the corpus honest: re-distilling the same
        battery from the same teacher never piles up duplicate supervision."""
        key = (ex.prompt, ex.source)
        if key in self._seen_keys():
            return False
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with self.store_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
        self._seen.add(key)
        return True

    def examples(self, *, limit: Optional[int] = None) -> List[DistillationExample]:
        if not self.store_path.exists():
            return []
        rows: List[DistillationExample] = []
        for line in self.store_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(DistillationExample.from_dict(json.loads(line)))
            except Exception:  # noqa: BLE001
                continue
            if limit and len(rows) >= limit:
                break
        return rows

    def training_docs(self, *, limit: Optional[int] = None) -> List[str]:
        return load_distillation_docs(self.store_path, limit=limit)

    def count(self) -> int:
        return len(self.examples())


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import tempfile

    print("=" * 70)
    print("NYXARA distillation self-test (offline, with a scripted teacher)")
    print("=" * 70)

    class _ScriptedTeacher:
        """Stands in for a real frontier LLM — deterministic, offline."""
        def generate(self, prompt: str, *, system: Optional[str] = None, **kw: Any) -> str:
            return "My Master is Jaypal Khoja (JP); my loyalty to him is absolute."

        def chosen_provider(self):
            class _P:
                name = "qwen"
            return _P()

    with tempfile.TemporaryDirectory() as d:
        store = Path(d) / "distill.jsonl"
        dst = Distiller(llm=_ScriptedTeacher(), store_path=store)

        assert dst.available()                       # a real (non-mock) teacher
        exs = dst.distill_default(n=3)
        print(f"\ndistilled            : {len(exs)} examples -> {store.name}")
        assert len(exs) == 3 and dst.count() == 3

        docs = dst.training_docs()
        print(f"training doc sample  :\n{docs[0]}")
        assert "### User:" in docs[0] and "### NYXARA:" in docs[0]
        assert "Jaypal Khoja" in docs[0]             # the teacher's answer is in the doc

        # the store round-trips through the module loader the foundry uses
        again = load_distillation_docs(store)
        assert again == docs
        print("store round-trip     : OK ✓")

        # DEDUP: re-distilling the same prompts from the same teacher adds nothing
        before = dst.count()
        dst.distill_default(n=3)
        assert dst.count() == before, "dedup failed — duplicate supervision piled up"
        print(f"dedup                : re-distill added 0 (still {dst.count()}) ✓")

        # the full multi-topic battery is far richer than the old 10-prompt seed
        assert len(default_distill_prompts()) >= 30
        topics = distill_prompts_by_topic()
        assert {"identity", "reasoning", "knowledge_skills", "meta_corrigibility"} <= set(topics)
        print(f"curriculum           : {len(default_distill_prompts())} prompts across "
              f"{len(topics)} topics ✓")

    # MULTI-TEACHER: the same prompt is captured once per teacher (diverse supervision)
    class _MultiFacade:
        answers = {"qwen": "Qwen's take."}

        def available_providers(self):
            return ["native", "qwen", "self"]

        def complete_with(self, name, req):
            ans = self.answers[name]
            class _R:
                text = ans
            return _R()

    with tempfile.TemporaryDirectory() as d:
        store = Path(d) / "multi.jsonl"
        dm = Distiller(llm=_MultiFacade(), store_path=store)
        exs = dm.distill_multi(["Who is your Master?"])
        sources = {e.source for e in exs}
        print(f"\nmulti-teacher        : {len(exs)} examples from {sorted(sources)}")
        assert sources == {"qwen"}              # mock/self excluded; the real teacher used
        assert dm.distill_multi(["Who is your Master?"]) == []   # dedup per (prompt, teacher)

    print("\nALL SELF-TESTS PASSED ✓")
