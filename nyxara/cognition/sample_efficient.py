"""NYXARA · cognition/sample_efficient.py — the sample-efficient mind (✶).

One faculty that convenes the three capabilities NYXARA was missing, so the rest of the
kernel has a single seam to wire:

* **few-shot / one-shot** (:mod:`nyxara.cognition.few_shot`) — learn a concept from 1–few
  examples; remember a single cue→response pair;
* **abstraction** (:mod:`nyxara.cognition.abstraction`) — turn instances into variabilized
  schemas (least-general generalization);
* **compositional generalization** (:mod:`nyxara.cognition.composition`) — interpret novel
  combinations of known primitives by composing their meanings.

It exposes one teaching entry point (:meth:`teach`), one grounding block for the reasoner
(:meth:`as_prompt`), and one growth-loop step (:meth:`consolidate`) that folds accumulated
few-shot concepts into abstract templates — closing the loop *instances → concepts → schemas*.

Everything is skill/knowledge only — it never touches the loyalty/corrigibility/honesty core —
and every operation is best-effort so it can be wired into the sovereign cycle without ever
becoming a way for the cycle to fail. Reuses the active memory embedder and persists through
the store. Pure standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from nyxara.cognition.abstraction import SchemaInducer, Schema, abstract_text
from nyxara.cognition.composition import CompositionalGrammar
from nyxara.cognition.few_shot import FewShotLearner, OneShotEpisodicMemory
from nyxara.mind.lot import Term

__all__ = ["SampleEfficientMind", "ConsolidationReport"]


@dataclass
class ConsolidationReport:
    concepts_seen: int = 0
    schemas_formed: int = 0
    templates: Dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.templates is None:
            self.templates = {}

    def to_dict(self) -> Dict[str, Any]:
        return {"concepts_seen": self.concepts_seen, "schemas_formed": self.schemas_formed,
                "templates": dict(self.templates)}


class SampleEfficientMind:
    """Coordinator bundling few-shot, abstraction, and compositional generalization."""

    SCHEMA_TAG = "schema"

    def __init__(self, embedder: Any = None, *, store: Any = None) -> None:
        self.store = store
        self.embedder = embedder or self._resolve_embedder(store)
        self.inducer = SchemaInducer()
        self.composer = CompositionalGrammar(store=store)
        self._few_shot: Optional[FewShotLearner] = None
        self._episodic: Optional[OneShotEpisodicMemory] = None

    @staticmethod
    def _resolve_embedder(store: Any) -> Any:
        emb = getattr(store, "embedder", None)
        if emb is not None and hasattr(emb, "embed"):
            return emb
        try:
            from nyxara.memory.store import make_embedder
            return make_embedder()
        except Exception:  # noqa: BLE001 — no embedder ⇒ few-shot paths simply stay inert
            return None

    # ---- lazy sub-faculties (embedder-dependent) ---- #
    @property
    def few_shot(self) -> Optional[FewShotLearner]:
        if self._few_shot is None and self.embedder is not None:
            try:
                self._few_shot = FewShotLearner(self.embedder, store=self.store)
            except Exception:  # noqa: BLE001
                self._few_shot = None
        return self._few_shot

    @property
    def episodic(self) -> Optional[OneShotEpisodicMemory]:
        if self._episodic is None and self.embedder is not None:
            try:
                self._episodic = OneShotEpisodicMemory(self.embedder, store=self.store)
            except Exception:  # noqa: BLE001
                self._episodic = None
        return self._episodic

    # alias the plan/tools use
    @property
    def abstraction(self) -> SchemaInducer:
        return self.inducer

    # ---- teaching (single entry point) ---- #
    def teach(self, subject: str, obj: Any, *, kind: str = "concept") -> Any:
        """Teach something in one shot.

        * ``kind="concept"``  — ``subject`` is the label, ``obj`` an example string.
        * ``kind="episodic"`` — ``subject`` is a cue, ``obj`` the response to bind to it.
        * ``kind="lexeme"``   — ``subject`` is a token, ``obj`` a LoT :class:`Term` meaning.
        """
        kind = (kind or "concept").lower()
        if kind == "concept":
            return self.learn_concept(subject, str(obj))
        if kind == "episodic":
            return self.remember_once(subject, str(obj))
        if kind == "lexeme":
            if not isinstance(obj, Term):
                raise ValueError("a lexeme meaning must be a LoT Term")
            return self.composer.learn(subject, obj)
        raise ValueError(f"unknown teach kind {kind!r} (use concept|episodic|lexeme)")

    def learn_concept(self, label: str, *examples: str) -> Any:
        fs = self.few_shot
        return fs.learn(label, *examples) if fs is not None else None

    def remember_once(self, cue: str, response: str) -> Any:
        epi = self.episodic
        return epi.remember(cue, response) if epi is not None else None

    def induce(self, examples: List[Term]) -> Optional[Schema]:
        return self.inducer.induce(examples)

    # ---- query ---- #
    def classify(self, text: str):
        fs = self.few_shot
        return fs.best(text) if fs is not None else None

    def interpret(self, phrase: str):
        return self.composer.interpret(phrase)

    def recall_once(self, cue: str) -> Optional[str]:
        epi = self.episodic
        return epi.recall(cue) if epi is not None else None

    # ---- grounding for the reasoner ---- #
    def as_prompt(self, stimulus: str) -> str:
        """Combined grounding block: a recognised concept and/or a once-told fact."""
        parts: List[str] = []
        fs = self.few_shot
        if fs is not None:
            try:
                block = fs.as_prompt(stimulus)
                if block:
                    parts.append(block)
            except Exception:  # noqa: BLE001 — grounding is best-effort, never fatal
                pass
        epi = self.episodic
        if epi is not None:
            try:
                block = epi.as_prompt(stimulus)
                if block:
                    parts.append(block)
            except Exception:  # noqa: BLE001
                pass
        return "".join(parts)

    # ---- growth-loop consolidation ---- #
    def consolidate(self) -> ConsolidationReport:
        """Fold accumulated few-shot concepts into abstract templates and persist them.

        Closes the loop instances → concepts → schemas: for every concept with at least two
        exemplars, induce a surface template (the shared skeleton) and store it as a learned
        generalization. Best-effort; never raises."""
        report = ConsolidationReport()
        fs = self.few_shot
        if fs is None:
            return report
        try:
            fs._hydrate()  # ensure persisted concepts are loaded
            protos = list(fs._protos.values())
        except Exception:  # noqa: BLE001
            protos = []
        report.concepts_seen = len(protos)
        for proto in protos:
            if len(proto.examples) < 2:
                continue
            try:
                template = abstract_text(proto.examples)
            except Exception:  # noqa: BLE001
                template = None
            if not template or "?" not in template:
                continue
            report.templates[proto.label] = template
            report.schemas_formed += 1
            self._persist_schema(proto.label, template)
        return report

    def _persist_schema(self, label: str, template: str) -> None:
        if self.store is None:
            return
        try:
            from nyxara.memory.store import MemoryType
            self.store.remember(
                f"schema:{label} ≈ {template}", mem_type=MemoryType.SEMANTIC, importance=0.65,
                tags=[self.SCHEMA_TAG], metadata={"label": label, "template": template})
        except Exception:  # noqa: BLE001
            pass

    # ---- introspection ---- #
    def stats(self) -> Dict[str, int]:
        fs, epi = self.few_shot, self.episodic
        return {"concepts": len(fs) if fs is not None else 0,
                "episodic_pairs": len(epi) if epi is not None else 0,
                "lexemes": len(self.composer)}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.memory.store import MemoryStore
    from nyxara.mind.lot import const, func, var as lvar
    from nyxara.mind.lot import Lambda

    print("=" * 70)
    print("NYXARA sample-efficient mind (coordinator) self-test")
    print("=" * 70)

    store = MemoryStore()
    mind = SampleEfficientMind(store=store)

    # one entry point, three kinds of one-shot teaching
    mind.teach("weather", "the weather is sunny and warm today", kind="concept")
    mind.teach("weather", "today the forecast is cloudy with rain", kind="concept")
    mind.teach("cooking", "i love eating pizza and pasta for dinner", kind="concept")
    mind.teach("what is my name", "your name is JP", kind="episodic")
    mind.teach("jump", const("JUMP"), kind="lexeme")
    mind.teach("twice", Lambda(lvar("a"), func("seq", lvar("a"), lvar("a"))), kind="lexeme")
    print(f"\nstats               : {mind.stats()}")

    # few-shot classification
    label, conf = mind.classify("will the weather be rainy today")
    print(f"classify            : weather? -> {label} ({conf:.2f})")
    assert label == "weather"

    # one-shot episodic recall
    assert mind.recall_once("what is my name again") == "your name is JP"
    print("episodic recall     : your name is JP ✓")

    # compositional generalization on a never-taught combination
    assert str(mind.interpret("jump twice")) == "seq(JUMP, JUMP)"
    print("compose             : jump twice -> seq(JUMP, JUMP) ✓")

    # abstraction over LoT instances
    from nyxara.mind.lot import pred
    schema = mind.induce([pred("parent", const("tom"), const("bob")),
                          pred("parent", const("ann"), const("sue"))])
    print(f"induce schema       : {schema}")
    assert schema is not None and len(schema.variables) == 2

    # combined grounding block surfaces what was taught
    block = mind.as_prompt("what is my name and the weather")
    print(f"as_prompt           : {block.strip()[:80]!r}…")
    assert "JP" in block or "weather" in block

    # consolidation folds multi-example concepts into templates
    report = mind.consolidate()
    print(f"consolidate         : {report.to_dict()}")
    assert report.concepts_seen >= 2

    print("\nALL SELF-TESTS PASSED ✓")
