"""NYXARA · cognition/sample_efficient.py — the sample-efficient mind (✶).

One faculty that convenes the three capabilities NYXARA was missing, so the rest of the
kernel has a single seam to wire:

* **few-shot / one-shot** (:mod:`nyxara.cognition.few_shot`) — learn a concept from 1–few
  examples; remember a single cue→response pair;
* **abstraction** (:mod:`nyxara.cognition.abstraction`) — turn instances into variabilized
  schemas (least-general generalization);
* **compositional generalization** (:mod:`nyxara.cognition.composition`) — interpret novel
  combinations of known primitives by composing their meanings;
* **skill induction** (:mod:`nyxara.cognition.skill_induction`) — learn a *task* from a few
  ``(input → output)`` demonstrations by synthesising a verified, reusable transformation, then
  *apply* it to a genuinely new input (real, transferable capability gain, not mere recognition).

It exposes one teaching entry point (:meth:`teach`), one grounding block for the reasoner
(:meth:`as_prompt`), one direct-answer channel (:meth:`solve`, a learned skill applied to a novel
input), and one growth-loop step (:meth:`consolidate`) that folds accumulated few-shot concepts into
abstract templates and mines demonstrations into skills — closing the loop *instances → concepts →
schemas → applicable skills*.

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
from nyxara.cognition.skill_induction import InductiveSkill, SkillInductionEngine
from nyxara.mind.lot import Term

__all__ = ["SampleEfficientMind", "ConsolidationReport"]


@dataclass
class ConsolidationReport:
    concepts_seen: int = 0
    schemas_formed: int = 0
    skills_induced: int = 0
    skills: List[str] = None  # type: ignore[assignment]
    templates: Dict[str, str] = None  # type: ignore[assignment]
    macros_learned: int = 0
    macros: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.templates is None:
            self.templates = {}
        if self.skills is None:
            self.skills = []
        if self.macros is None:
            self.macros = []

    def to_dict(self) -> Dict[str, Any]:
        return {"concepts_seen": self.concepts_seen, "schemas_formed": self.schemas_formed,
                "skills_induced": self.skills_induced, "skills": list(self.skills),
                "templates": dict(self.templates), "macros_learned": self.macros_learned,
                "macros": list(self.macros)}


class SampleEfficientMind:
    """Coordinator bundling few-shot, abstraction, and compositional generalization."""

    SCHEMA_TAG = "schema"

    def __init__(self, embedder: Any = None, *, store: Any = None,
                 settings: Any = None) -> None:
        self.store = store
        self.settings = settings
        self.embedder = embedder or self._resolve_embedder(store)
        self.inducer = SchemaInducer()
        self.composer = CompositionalGrammar(store=store)
        self._few_shot: Optional[FewShotLearner] = None
        self._episodic: Optional[OneShotEpisodicMemory] = None
        self.skills = self._build_skill_engine()
        # the discrete program library (DreamCoder-style wake/sleep): every skill induced below
        # goes through it, so solved tasks feed a compression pass that permanently grows the
        # engine's own primitive vocabulary rather than merely being cached under a task name.
        self.program_library = self._build_program_library()

    def _build_program_library(self) -> Any:
        try:
            from nyxara.cognition.program_library import ProgramLibrary
            return ProgramLibrary(engine=self.skills, store=self.store)
        except Exception:  # noqa: BLE001 — the library is a capability, never a hard dependency
            return None

    def _build_skill_engine(self) -> SkillInductionEngine:
        """The few-shot skill-induction engine, tuned by the foundry settings when present."""
        cfg = getattr(self.settings, "foundry", None)
        self._skills_enabled = bool(getattr(cfg, "skill_induction_enabled", True))
        return SkillInductionEngine(
            self.embedder, store=self.store,
            max_depth=int(getattr(cfg, "skill_max_program_depth", 3)),
            beam_width=int(getattr(cfg, "skill_beam_width", 16)),
            min_demos=int(getattr(cfg, "skill_min_demos", 2)),
            apply_confidence=float(getattr(cfg, "skill_apply_confidence", 0.55)))

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
        * ``kind="skill"``    — ``subject`` is the task name, ``obj`` a sequence of
          ``(input, output)`` demonstration pairs to induce a reusable transformation from.
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
        if kind == "skill":
            return self.learn_skill(subject, obj)
        raise ValueError(f"unknown teach kind {kind!r} (use concept|episodic|lexeme|skill)")

    def learn_skill(self, name: str, pairs: Any) -> Optional[InductiveSkill]:
        """Induce a verified, reusable transformation from ``(input, output)`` demonstrations.

        Routed through the program library when available, so the search is empowered by
        whatever it has compressed so far and its outcome feeds back into that library."""
        if not getattr(self, "_skills_enabled", True):
            return None
        try:
            demos = [(str(a), str(b)) for a, b in pairs]
        except Exception:  # noqa: BLE001 — malformed demos simply teach nothing
            return None
        return (self.program_library or self.skills).induce(name, demos)

    def solve(self, stimulus: str):
        """Apply the best-matching learned skill to ``stimulus`` → ``(answer, conf)`` or None.

        This is the direct-answer channel: a task taught from a few demonstrations actually
        *answers* a novel input, rather than merely being recognised. Conservative — it only
        fires when the input structurally (or semantically) matches what the skill was taught."""
        if not getattr(self, "_skills_enabled", True):
            return None
        try:
            return self.skills.solve(stimulus)
        except Exception:  # noqa: BLE001 — solving is best-effort, never fatal
            return None

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
        # a learned skill that transforms this very input surfaces its result as grounding, so
        # even below the direct-answer floor a taught task measurably biases the reply
        try:
            hit = self.skills.solve(stimulus)
            if hit is not None:
                parts.append(f"\n\nLearned skill (from few-shot demonstrations) applied to this "
                             f"input yields: {hit[0]}")
        except Exception:  # noqa: BLE001 — skill grounding is best-effort, never fatal
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
        # mine accumulated one-shot (cue → response) demonstrations into a reusable skill: if a
        # single verified transformation explains them all, she has genuinely learned a task from
        # examples (not just memorised the pairs). Verification means garbage never sticks.
        self._induce_skills_from_episodic(report)
        # sleep-abstraction: mine the skill corpus for recurring program shapes and permanently
        # compile them into new library primitives — the DreamCoder-style compression step.
        if self.program_library is not None:
            try:
                lib_report = self.program_library.consolidate()
                report.macros_learned = len(lib_report.new_macros)
                report.macros = list(lib_report.new_macros)
            except Exception:  # noqa: BLE001 — library growth is best-effort, never fatal
                pass
        return report

    def _induce_skills_from_episodic(self, report: "ConsolidationReport") -> None:
        epi = self.episodic
        if epi is None:
            return
        try:
            epi._hydrate()
            pairs = [(p.cue, p.response) for p in epi._pairs if p.cue and p.response]
        except Exception:  # noqa: BLE001
            pairs = []
        if len(pairs) < self.skills.min_demos:
            return
        try:
            skill = (self.program_library or self.skills).induce("episodic_transform", pairs)
        except Exception:  # noqa: BLE001 — mining is best-effort, never fatal
            skill = None
        if skill is not None:
            report.skills_induced += 1
            report.skills.append(skill.name)

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
                "lexemes": len(self.composer),
                "skills": len(self.skills),
                "macros": len(self.program_library) if self.program_library is not None else 0}


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

    # few-shot SKILL: learn a task from two demos and APPLY it to a never-seen input
    mind.teach("greet", [("greet jp", "hello jp !"), ("greet sky", "hello sky !")], kind="skill")
    solved = mind.solve("greet nyxara")
    print(f"skill solve         : {solved}")
    assert solved is not None and solved[0] == "hello nyxara !"   # real generalization
    assert mind.solve("what time is it") is None                  # never hijacks an unrelated turn
    print("skill generalizes   : greet nyxara -> 'hello nyxara !' ✓")

    # discrete program library: teach 3 tasks that share a 2-step (reorder + wrap) shape;
    # consolidation compresses them into one permanent, reusable primitive.
    mind.teach("shout", [("hello world", "SAY: world hello!"), ("go now", "SAY: now go!")],
              kind="skill")
    mind.teach("ask", [("foo bar", "SAY: bar foo?"), ("up down", "SAY: down up?")], kind="skill")
    mind.teach("state", [("cat dog", "SAY: dog cat."), ("red blue", "SAY: blue red.")],
              kind="skill")
    lib_report = mind.consolidate()
    print(f"library consolidate : {lib_report.to_dict()}")
    assert lib_report.macros_learned >= 1, "3 independently-taught same-shape tasks -> a macro"
    print(f"stats               : {mind.stats()}")
    assert mind.stats()["macros"] >= 1
    print("program library     : compressed a recurring pattern into a permanent primitive ✓")

    print("\nALL SELF-TESTS PASSED ✓")
