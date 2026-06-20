"""NYXARA · mind/faculties.py — faculty registry + per-task selector (✦★).

NYXARA is not "an LLM with tools." It is a kernel that, for each sub-task, picks the
*right cognitive engine*: exact arithmetic goes to a math faculty, recall goes to
retrieval, prediction goes to the world-model, structural mapping goes to analogy —
and only open-ended generation goes to the LLM. **The LLM is one entry in the
registry, not the driver.**

Two ideas:

* **Faculty** — a self-describing engine: which :class:`TaskType` it handles, how
  *reliable* it is, how much it *costs*, and — decisively — whether it is
  **verifiable** (symbolic/exact) or **probabilistic** (a guess). Every faculty's
  output is a :class:`~nyxara.mind.proposal.Proposal`, never an action.

* **Selector** — given a task, scores the registry and chooses. The control law
  *"verifiable beats probabilistic"* is encoded directly: a verifiable engine that
  fits wins over a probabilistic one, and ``task.requires_verifiable`` filters the
  guessers out entirely.

Depends on :mod:`mind.proposal`, :mod:`mind.llm` (the LLM faculty wrapper),
:mod:`memory.provenance`, :mod:`kernel.errors`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.kernel.errors import NyxaraError
from nyxara.memory.provenance import Provenance, SourceType
from nyxara.mind.proposal import Proposal, ProposalKind

__all__ = [
    "TaskType",
    "Task",
    "Faculty",
    "CallableFaculty",
    "LLMFaculty",
    "FacultyRegistry",
    "FacultySelector",
    "NoFacultyError",
]


class NoFacultyError(NyxaraError):
    """No registered faculty can handle the task."""


# --------------------------------------------------------------------------- #
# Task taxonomy
# --------------------------------------------------------------------------- #
class TaskType(str, Enum):
    ARITHMETIC = "arithmetic"        # exact numeric computation
    ALGEBRA = "algebra"              # solve equations for an unknown
    CALCULUS = "calculus"            # derivative / integral / limit
    SEQUENCE = "sequence"            # next term of a number sequence
    DATE = "date"                    # calendar / day arithmetic
    LOGIC = "logic"                  # symbolic/deductive inference
    FACTUAL_RECALL = "factual_recall"  # "what do I know about X"
    RETRIEVAL = "retrieval"          # search memory / documents
    PREDICTION = "prediction"        # simulate / forecast the world
    ANALOGY = "analogy"              # structure mapping
    CLASSIFICATION = "classification"
    PLANNING = "planning"
    GENERATION = "generation"        # open-ended text/creative
    REASONING = "reasoning"          # general multi-step reasoning
    OPTIMIZATION = "optimization"


@dataclass
class Task:
    """A sub-task the kernel needs solved by *some* faculty."""

    type: TaskType
    description: str = ""
    payload: Any = None
    features: Dict[str, Any] = field(default_factory=dict)
    requires_verifiable: bool = False    # exactness is mandatory (e.g. money, safety)
    tags: frozenset = field(default_factory=frozenset)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type.value, "description": self.description,
                "requires_verifiable": self.requires_verifiable, "tags": sorted(self.tags)}


# --------------------------------------------------------------------------- #
# Faculty base
# --------------------------------------------------------------------------- #
class Faculty:
    """A cognitive engine. Subclasses implement :meth:`handle` → Proposal."""

    name: str = "faculty"
    handles: frozenset = frozenset()
    verifiable: bool = False
    reliability: float = 0.7   # how often it is right when it applies [0,1]
    cost: float = 1.0          # relative compute/latency cost (>0)

    def available(self) -> bool:
        return True

    def suitability(self, task: Task) -> float:
        """How well this faculty fits the task, in [0,1] (0 == cannot)."""
        return 1.0 if task.type in self.handles else 0.0

    def handle(self, task: Task) -> Proposal:  # pragma: no cover - overridden
        raise NotImplementedError

    # helper for subclasses to build a proposal with sensible provenance
    def _propose(self, content: Any, *, kind: ProposalKind = ProposalKind.ANSWER,
                 confidence: float = 0.7, rationale: str = "",
                 source: SourceType = SourceType.SELF_REFLECTION,
                 risk: float = 0.0, **kw: Any) -> Proposal:
        return Proposal(
            kind=kind, content=content, source_faculty=self.name,
            confidence=confidence, rationale=rationale, risk=risk,
            provenance=Provenance(source, confidence=confidence, detail=self.name,
                                  method="computed" if self.verifiable else "generated"),
            **kw,
        )


# --------------------------------------------------------------------------- #
# Callable faculty — register any function as a faculty
# --------------------------------------------------------------------------- #
class CallableFaculty(Faculty):
    """Wrap a plain function ``fn(task) -> (content, confidence)`` as a faculty.

    This is how math.py / rag.py / world_model.py register themselves later, and how
    tests inject engines without subclassing.
    """

    def __init__(self, name: str, handles: Sequence[TaskType],
                 fn: Callable[[Task], Tuple[Any, float]], *,
                 verifiable: bool = False, reliability: float = 0.8, cost: float = 1.0,
                 source: SourceType = SourceType.TOOL,
                 suitability_fn: Optional[Callable[[Task], float]] = None,
                 available_fn: Optional[Callable[[], bool]] = None,
                 kind: ProposalKind = ProposalKind.ANSWER) -> None:
        self.name = name
        self.handles = frozenset(handles)
        self._fn = fn
        self.verifiable = verifiable
        self.reliability = reliability
        self.cost = cost
        self._source = source
        self._suitability_fn = suitability_fn
        self._available_fn = available_fn
        self._kind = kind

    def available(self) -> bool:
        return self._available_fn() if self._available_fn else True

    def suitability(self, task: Task) -> float:
        if self._suitability_fn:
            return self._suitability_fn(task)
        return super().suitability(task)

    def handle(self, task: Task) -> Proposal:
        content, confidence = self._fn(task)
        return self._propose(content, kind=self._kind, confidence=confidence,
                             rationale=f"{self.name} computed result",
                             source=self._source)


# --------------------------------------------------------------------------- #
# LLM faculty — the probabilistic generalist (ONE entry, not the driver)
# --------------------------------------------------------------------------- #
class LLMFaculty(Faculty):
    """Wraps :class:`~nyxara.mind.llm.LLM`. Probabilistic; the fallback generalist."""

    name = "llm"
    handles = frozenset({TaskType.GENERATION, TaskType.REASONING, TaskType.FACTUAL_RECALL,
                         TaskType.CLASSIFICATION, TaskType.PLANNING, TaskType.ANALOGY})
    verifiable = False
    reliability = 0.6
    cost = 3.0   # expensive relative to exact engines

    def __init__(self, llm: Any, *, system: Optional[str] = None,
                 base_confidence: float = 0.6) -> None:
        self._llm = llm
        self._system = system
        self._base_confidence = base_confidence

    def suitability(self, task: Task) -> float:
        if task.type not in self.handles:
            # the LLM is a generalist — weakly applicable to almost anything textual
            return 0.25
        # strongest at open-ended generation
        return 0.9 if task.type is TaskType.GENERATION else 0.7

    def handle(self, task: Task) -> Proposal:
        prompt = task.description or str(task.payload)
        text = self._llm.generate(prompt, system=self._system)
        return self._propose(text, kind=ProposalKind.ANSWER,
                             confidence=self._base_confidence,
                             rationale="LLM generation (probabilistic)",
                             source=SourceType.LLM_INFERENCE)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
class FacultyRegistry:
    def __init__(self) -> None:
        self._faculties: Dict[str, Faculty] = {}

    def register(self, faculty: Faculty) -> Faculty:
        self._faculties[faculty.name] = faculty
        return faculty

    def unregister(self, name: str) -> bool:
        return self._faculties.pop(name, None) is not None

    def get(self, name: str) -> Optional[Faculty]:
        return self._faculties.get(name)

    def all(self) -> List[Faculty]:
        return list(self._faculties.values())

    def by_task(self, task_type: TaskType) -> List[Faculty]:
        return [f for f in self._faculties.values() if task_type in f.handles]

    def __len__(self) -> int:
        return len(self._faculties)


# --------------------------------------------------------------------------- #
# Selector — encodes "verifiable > probabilistic"
# --------------------------------------------------------------------------- #
@dataclass
class FacultyScore:
    faculty: Faculty
    score: float
    verifiable: bool
    suitability: float

    def to_dict(self) -> Dict[str, Any]:
        return {"faculty": self.faculty.name, "score": round(self.score, 4),
                "verifiable": self.verifiable, "suitability": round(self.suitability, 3)}


class FacultySelector:
    """Scores the registry for a task and picks the engine the kernel should use."""

    def __init__(self, registry: FacultyRegistry, *,
                 verifiable_bonus: float = 0.75) -> None:
        self.registry = registry
        self.verifiable_bonus = verifiable_bonus

    def _score(self, faculty: Faculty, task: Task) -> float:
        if not faculty.available():
            return 0.0
        suit = faculty.suitability(task)
        if suit <= 0.0:
            return 0.0
        score = suit * faculty.reliability / max(0.1, faculty.cost)
        if faculty.verifiable:
            score *= (1.0 + self.verifiable_bonus)
        return score

    def rank(self, task: Task) -> List[FacultyScore]:
        candidates = self.registry.all()
        if task.requires_verifiable:
            candidates = [f for f in candidates if f.verifiable]
        scored = [
            FacultyScore(f, self._score(f, task), f.verifiable, f.suitability(task))
            for f in candidates
        ]
        scored = [s for s in scored if s.score > 0.0]
        # primary: score; tie-break: prefer verifiable, then cheaper
        scored.sort(key=lambda s: (s.score, s.verifiable, -s.faculty.cost), reverse=True)
        return scored

    def select(self, task: Task) -> Optional[Faculty]:
        ranked = self.rank(task)
        return ranked[0].faculty if ranked else None

    def dispatch(self, task: Task) -> Proposal:
        """Pick the best faculty and produce its Proposal (which the kernel disposes)."""
        ranked = self.rank(task)
        for choice in ranked:
            if choice.faculty.available():
                return choice.faculty.handle(task)
        raise NoFacultyError(
            f"no faculty can handle task type {task.type.value}"
            + (" with verifiability required" if task.requires_verifiable else ""),
            context={"task": task.to_dict()})

    def explain(self, task: Task) -> Dict[str, Any]:
        return {"task": task.to_dict(),
                "ranking": [s.to_dict() for s in self.rank(task)]}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import ast
    import operator as op

    from nyxara.kernel.config import NyxaraSettings, Profile
    from nyxara.mind.llm import LLM

    print("=" * 70)
    print("NYXARA faculties self-test")
    print("=" * 70)

    # a real, exact arithmetic faculty (verifiable)
    _OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
            ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}

    def _safe_eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return _OPS[type(node.op)](_safe_eval(node.operand))
        raise ValueError("unsupported")

    def arithmetic(task: Task):
        expr = task.payload
        value = _safe_eval(ast.parse(expr, mode="eval").body)
        return (value, 1.0)  # exact -> full confidence

    math_fac = CallableFaculty("math", [TaskType.ARITHMETIC, TaskType.OPTIMIZATION],
                               arithmetic, verifiable=True, reliability=1.0, cost=0.2)

    # the LLM faculty (probabilistic generalist) over the mock backend
    llm = LLM(settings=NyxaraSettings.for_profile(Profile.TEST))
    llm_fac = LLMFaculty(llm, system="You are NYXARA.")

    # a stub analogy faculty
    analogy_fac = CallableFaculty(
        "analogy", [TaskType.ANALOGY],
        lambda t: (f"A is to B as C is to D ({t.description})", 0.65),
        verifiable=False, reliability=0.7, cost=1.5)

    reg = FacultyRegistry()
    reg.register(math_fac); reg.register(llm_fac); reg.register(analogy_fac)
    sel = FacultySelector(reg)

    # 1) arithmetic -> math beats the LLM (verifiable > probabilistic)
    t = Task(TaskType.ARITHMETIC, "what is 2+3*4", payload="2+3*4")
    chosen = sel.select(t)
    print(f"\narithmetic task     : chosen = {chosen.name}")
    assert chosen.name == "math"
    prop = sel.dispatch(t)
    print(f"  result            : {prop.content} (conf={prop.confidence})")
    assert prop.content == 14 and prop.confidence == 1.0

    # 2) open-ended generation -> the LLM
    t2 = Task(TaskType.GENERATION, "write a haiku about loyalty")
    print(f"generation task     : chosen = {sel.select(t2).name}")
    assert sel.select(t2).name == "llm"

    # 3) analogy -> analogy faculty (LLM also applies but analogy is more suitable+cheaper)
    t3 = Task(TaskType.ANALOGY, "map chess to war")
    ranking = sel.explain(t3)["ranking"]
    print(f"analogy ranking     : {[(r['faculty'], r['score']) for r in ranking]}")
    assert ranking[0]["faculty"] == "analogy"

    # 4) requires_verifiable filters out the guessers
    t4 = Task(TaskType.ARITHMETIC, "exact only", payload="100/4", requires_verifiable=True)
    ranked = sel.rank(t4)
    assert all(s.verifiable for s in ranked)
    print(f"verifiable-only     : {[s.faculty.name for s in ranked]}")

    # 5) no faculty -> NoFacultyError
    t5 = Task(TaskType.PREDICTION, "forecast", requires_verifiable=True)
    try:
        sel.dispatch(t5)
        raise SystemExit("ERROR: should have raised")
    except NoFacultyError:
        print("no-faculty case     : NoFacultyError raised OK")

    # 6) faculty output is a PROPOSAL, not an action (ties to control law)
    assert isinstance(prop, Proposal) and prop.source_faculty == "math"
    print(f"\noutput is a Proposal : kind={prop.kind.value}, source={prop.source_faculty}")

    print(f"\nregistry size       : {len(reg)}")
    print("\nALL SELF-TESTS PASSED ✓")
