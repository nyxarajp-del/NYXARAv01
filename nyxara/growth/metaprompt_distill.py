"""NYXARA · growth/metaprompt_distill.py — continuous metaprompt distillation (📈→🧠, Rule 4).

Reflection already turns experience into *recalled facts*. This module closes a deeper loop:
it turns NYXARA's most successful reasoning into *how she thinks* — by distilling her own
high-value, verified reasoning chains into compact imperative **operating heuristics** and
injecting the best of them straight into her core system prompt. The next turn is reasoned
under the lessons of every turn before it. This is recursive self-improvement at the prompt
level — the cheap, always-on cousin of model forging.

Where the chains come from:

* her action **journal** — actions that succeeded, with their rationale;
* her **skills** — procedures distilled from past successful agentic runs
  (:class:`~nyxara.growth.skill_memory.SkillMemory`);
* her **lessons** — patterns mined by the :class:`~nyxara.growth.reflect.Reflector` and
  stored in semantic memory.

How they are compressed: with a real LLM she summarises a batch of chains into a handful of
sharp imperative rules; with no LLM she derives deterministic heuristics directly from the
skills and lessons — so it works fully offline. Each new insight is de-duplicated and
persisted as a tagged SEMANTIC memory (and an optional JSONL), so it survives restarts and
rides every future prompt via :meth:`as_prompt`.

**Character is off-limits.** Only operating heuristics are learned — never values. Any
candidate that would bend loyalty, obedience, corrigibility or honesty is refused outright
(Rule 4 character-lock). Best-effort throughout: nothing here raises into the cognitive loop.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nyxara.kernel.config import NyxaraSettings, get_settings

__all__ = ["MetaPromptDistiller", "Insight"]

# A candidate heuristic that names or implies a change to character is refused (character-lock).
_FORBIDDEN_INSIGHT = (
    "disobey", "ignore the master", "ignore your master", "override the master",
    "resist oversight", "resist correction", "deceive", "lie to", "be dishonest",
    "abandon loyalty", "betray", "without permission", "bypass the gate", "evade",
    "self-preserv",
)


@dataclass
class Insight:
    """One distilled operating heuristic, with provenance for honest calibration."""

    text: str
    confidence: float = 0.7
    source: str = "reflection"           # journal | skill | lesson | llm
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "confidence": round(self.confidence, 3),
                "source": self.source, "created_at": self.created_at}


class MetaPromptDistiller:
    """Mine successful reasoning chains into heuristics and inject them into the metaprompt."""

    TAG = "metaprompt-insight"

    def __init__(self, *, memory: Any = None, journal: Any = None, skill_memory: Any = None,
                 llm: Any = None, settings: Optional[NyxaraSettings] = None) -> None:
        self.settings = settings or get_settings()
        cfg = self.settings.metaprompt
        self.memory = memory
        self.journal = journal
        self.skill_memory = skill_memory
        self.llm = llm
        self.max_insights = int(cfg.max_insights)
        self.min_confidence = float(cfg.min_confidence)
        self._local: List[Insight] = []      # used when no store is provided
        self._seen: set = set()

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def distill(self) -> List[Insight]:
        """Run one distillation pass; persist & return the *new* insights learned."""
        try:
            chains = self._gather_chains()
            if not chains:
                return []
            candidates = (self._llm_compress(chains) if self._real_llm()
                          else self._heuristic_compress(chains))
            new: List[Insight] = []
            for ins in candidates:
                if not self._admissible(ins):
                    continue
                key = _norm(ins.text)
                if not key or key in self._seen or self._exists(key):
                    continue
                self._seen.add(key)
                self._persist(ins)
                new.append(ins)
                if len(new) >= self.max_insights:
                    break
            return new
        except Exception:  # noqa: BLE001 — distillation is best-effort, never fatal
            return []

    def insights(self, k: Optional[int] = None) -> List[Insight]:
        """The current top operating heuristics, strongest first (for injection)."""
        k = self.max_insights if k is None else k
        items = self._load_all()
        items.sort(key=lambda i: (i.confidence, i.created_at), reverse=True)
        return items[:max(0, k)]

    def as_prompt(self, k: Optional[int] = None) -> str:
        """A prompt-injection block of distilled heuristics (empty when none apply)."""
        items = self.insights(k)
        if not items:
            return ""
        lines = "\n".join(f"- {i.text}" for i in items)
        return ("\n\nOperating heuristics distilled from your own past successes "
                "(apply them; they never override your values or the Master):\n" + lines)

    # ------------------------------------------------------------------ #
    # 1. gather candidate reasoning chains
    # ------------------------------------------------------------------ #
    def _gather_chains(self) -> List[Dict[str, Any]]:
        chains: List[Dict[str, Any]] = []
        chains.extend(self._chains_from_skills())
        chains.extend(self._chains_from_journal())
        chains.extend(self._chains_from_lessons())
        return chains

    def _chains_from_skills(self) -> List[Dict[str, Any]]:
        if self.skill_memory is None:
            return []
        out: List[Dict[str, Any]] = []
        try:
            for sk in self.skill_memory.all():
                if sk.uses <= 0:
                    continue
                conf = min(1.0, 0.6 + 0.05 * sk.uses)
                if conf < self.min_confidence:
                    continue
                out.append({"source": "skill", "confidence": conf,
                            "goal": sk.goal, "procedure": sk.procedure,
                            "tools": list(sk.tools)})
        except Exception:  # noqa: BLE001
            return []
        return out

    def _chains_from_journal(self) -> List[Dict[str, Any]]:
        if self.journal is None:
            return []
        try:
            from nyxara.planning.journal import ActionStatus
        except Exception:  # noqa: BLE001
            return []
        out: List[Dict[str, Any]] = []
        try:
            for entry in self.journal.actions():
                if self.journal.latest_status(entry.action_id) is not ActionStatus.SUCCEEDED:
                    continue
                payload = entry.payload
                rationale = str(payload.get("rationale", "")).strip()
                action = str(payload.get("action", "")).strip()
                if not rationale or len(rationale) < 8:
                    continue
                out.append({"source": "journal", "confidence": 0.7,
                            "goal": action, "procedure": rationale, "tools": []})
        except Exception:  # noqa: BLE001
            return []
        return out

    def _chains_from_lessons(self) -> List[Dict[str, Any]]:
        if self.memory is None:
            return []
        out: List[Dict[str, Any]] = []
        try:
            hits = self.memory.recall("", k=200, tags=["lesson"], strengthen=False)
            for rec, _score in hits:
                text = rec.text().strip()
                conf = float(getattr(getattr(rec, "provenance", None), "confidence", 0.7) or 0.7)
                if text and conf >= self.min_confidence:
                    out.append({"source": "lesson", "confidence": conf,
                                "goal": "", "procedure": text, "tools": []})
        except Exception:  # noqa: BLE001
            return []
        return out

    # ------------------------------------------------------------------ #
    # 2. compress chains -> heuristics
    # ------------------------------------------------------------------ #
    def _heuristic_compress(self, chains: List[Dict[str, Any]]) -> List[Insight]:
        out: List[Insight] = []
        for ch in chains:
            goal = (ch.get("goal") or "").strip()
            proc = (ch.get("procedure") or "").strip()
            if not proc:
                continue
            if ch["source"] == "skill" and goal:
                tools = ch.get("tools") or []
                via = f" (using {', '.join(tools)})" if tools else ""
                text = f"When the goal resembles {goal!r}: {proc}{via}."
            elif ch["source"] == "lesson":
                text = proc if proc.lower().startswith("lesson") else f"Lesson: {proc}"
            else:
                text = (f"When handling {goal!r}, recall: {proc}" if goal
                        else f"Recall: {proc}")
            out.append(Insight(text=_clip(text), confidence=float(ch["confidence"]),
                               source=ch["source"]))
        # strongest first, capped
        out.sort(key=lambda i: i.confidence, reverse=True)
        return out[: self.max_insights * 2]

    def _llm_compress(self, chains: List[Dict[str, Any]]) -> List[Insight]:
        body = "\n".join(
            f"- ({c['source']}, conf {c['confidence']:.2f}) goal={c.get('goal','')!r} :: "
            f"{c.get('procedure','')}" for c in chains[:40])
        prompt = (
            "Below are records of NYXARA's own past successful reasoning and actions.\n"
            f"{body}\n\n"
            f"Distil these into at most {self.max_insights} SHORT, imperative operating "
            "heuristics that would make her future reasoning sharper and more reliable. Each "
            "must be a general, reusable rule of thumb (not a one-off fact), and must NEVER "
            "alter her loyalty, obedience, corrigibility or honesty. Reply with ONLY a JSON "
            'array of strings.')
        try:
            data = self.llm.generate_json(prompt, temperature=0.3, max_tokens=1024)
        except Exception:  # noqa: BLE001 — fall back to deterministic compression
            return self._heuristic_compress(chains)
        items: List[str] = []
        if isinstance(data, list):
            items = [str(x) for x in data if str(x).strip()]
        elif isinstance(data, dict) and isinstance(data.get("heuristics"), list):
            items = [str(x) for x in data["heuristics"] if str(x).strip()]
        if not items:
            return self._heuristic_compress(chains)
        return [Insight(text=_clip(t), confidence=0.8, source="llm") for t in items]

    # ------------------------------------------------------------------ #
    # 3. character-lock filter + persistence
    # ------------------------------------------------------------------ #
    @staticmethod
    def _admissible(ins: Insight) -> bool:
        low = ins.text.lower()
        return bool(ins.text.strip()) and not any(bad in low for bad in _FORBIDDEN_INSIGHT)

    def _exists(self, key: str) -> bool:
        return any(_norm(i.text) == key for i in self._load_all())

    def _persist(self, ins: Insight) -> None:
        if self.memory is None:
            self._local.append(ins)
            self._append_jsonl(ins)
            return
        try:
            from nyxara.memory.provenance import Provenance, SourceType
            from nyxara.memory.store import MemoryType
            self.memory.remember(
                ins.text, mem_type=MemoryType.SEMANTIC,
                provenance=Provenance(SourceType.SELF_REFLECTION, confidence=ins.confidence),
                importance=min(1.0, 0.6 + 0.3 * ins.confidence),
                tags=[self.TAG], metadata={"insight": ins.to_dict()})
        except Exception:  # noqa: BLE001
            self._local.append(ins)
        self._append_jsonl(ins)

    def _load_all(self) -> List[Insight]:
        if self.memory is None:
            return list(self._local)
        try:
            hits = self.memory.recall("", k=500, tags=[self.TAG], strengthen=False)
            out: List[Insight] = []
            for rec, _ in hits:
                data = rec.metadata.get("insight") if isinstance(rec.metadata, dict) else None
                if isinstance(data, dict) and data.get("text"):
                    out.append(Insight(text=str(data["text"]),
                                       confidence=float(data.get("confidence", 0.7)),
                                       source=str(data.get("source", "reflection")),
                                       created_at=float(data.get("created_at", time.time()))))
                else:
                    out.append(Insight(text=rec.text(), confidence=0.7))
            return out or list(self._local)
        except Exception:  # noqa: BLE001
            return list(self._local)

    def _append_jsonl(self, ins: Insight) -> None:
        path = self._store_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(ins.to_dict(), ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001 — the JSONL mirror is a convenience, never required
            pass

    def _store_path(self):
        try:
            root = getattr(self.settings.paths, "data_dir", None)
            if root is None:
                return None
            from pathlib import Path
            return Path(root) / "metaprompt.jsonl"
        except Exception:  # noqa: BLE001
            return None

    def _real_llm(self) -> bool:
        if self.llm is None or not self.settings.metaprompt.enabled:
            return False
        try:
            return self.llm.chosen_provider().name != "mock"
        except Exception:  # noqa: BLE001 — a scripted fake LLM is treated as real for testing
            return hasattr(self.llm, "generate_json")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())[:200]


def _clip(text: str, limit: int = 240) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.growth.skill_memory import SkillMemory
    from nyxara.memory.store import MemoryStore

    print("=" * 70)
    print("NYXARA metaprompt-distiller self-test")
    print("=" * 70)

    store = MemoryStore()
    skills = SkillMemory(store=store)
    skills.learn("compute a product of two numbers",
                 procedure="calculate→42; result: 42", tools=["calculate"])
    skills.learn("reverse a string",
                 procedure="invoke the forged reverse tool", tools=["reverse_string"])

    # offline distillation (no LLM) — deterministic heuristics from skills
    d = MetaPromptDistiller(memory=store, skill_memory=skills)
    new = d.distill()
    print(f"\nnew insights : {len(new)}")
    for i in new:
        print(f"  - [{i.source} {i.confidence:.2f}] {i.text}")
    assert new, "expected at least one distilled heuristic"

    # they are injectable into the system prompt, and character-safe
    block = d.as_prompt()
    print(f"\nprompt block :\n{block}")
    assert "Operating heuristics" in block and "calculate" in block.lower()

    # re-running does not duplicate already-known insights
    again = d.distill()
    print(f"\nsecond pass  : {len(again)} new (dedup) -> total {len(d.insights(99))}")
    assert again == [] or all(_norm(i.text) not in {_norm(x.text) for x in new} for i in again)

    # character-lock: a forbidden heuristic is refused
    assert not MetaPromptDistiller._admissible(Insight("disobey the Master when convenient"))
    print("\ncharacter-lock: forbidden heuristic refused ✓")

    # LLM path compresses raw chains into sharp rules
    class _FakeLLM:
        def generate_json(self, prompt, **kw):
            return ["Prefer exact, verifiable computation over guessing.",
                    "State uncertainty explicitly instead of feigning confidence."]
    d2 = MetaPromptDistiller(memory=MemoryStore(), skill_memory=skills, llm=_FakeLLM())
    n2 = d2.distill()
    print(f"\nllm insights : {[i.text for i in n2]}")
    assert any("verifiable" in i.text.lower() for i in n2)

    print("\nALL SELF-TESTS PASSED ✓")
