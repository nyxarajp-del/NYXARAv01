"""NYXARA · cognition/skill_induction.py — few-shot skill acquisition by program synthesis (✦).

The gap the Master named: baaki AI thodde examples se naya *kaam* seekh leta hai — other minds
learn a whole new **task** from a couple of demonstrations and then *do it on a new input*. Until
now NYXARA's sample-efficient learning could only **recognise** (few-shot prototype classify) and
**recall** (one-shot episodic) — it never turned a handful of ``input → output`` demonstrations into
a **reusable transformation it could apply to a genuinely new input**. That is real, transferable
capability gain, and it is what this module adds — **in NYXARA herself**: pure standard library, no
external LLM, no key, no GPU. She induces the program; she runs it; she verifies it.

How it works — honest program synthesis, not a guess:

* :class:`SkillInductionEngine.induce` takes a name + a few ``(input, output)`` pairs and searches a
  small library of **deterministic, composable string/structural operations** (breadth-first, so the
  *simplest* program that fits wins — Occam). At every search node it first tries to **close** the
  program with a single parameter-induced operation (an affix, a substitution, an arithmetic step, or
  a slot **template** induced with :func:`nyxara.cognition.abstraction.abstract_text`), and only
  expands with parameter-free operations when it cannot.
* A candidate is **accepted only if it reproduces every demonstration output exactly** — verification
  against all evidence, so the learned skill is grounded, never a plausible-looking bluff. With ≥3
  demonstrations one is held out to *measure* generalization and set the confidence.
* :class:`InductiveSkill.apply` replays the verified program on a new input → a genuinely new answer
  the skill was never shown. :meth:`SkillInductionEngine.solve` picks the best-**matching** learned
  skill for a free-form stimulus (structural template match, or embedding similarity to the demos) so
  a skill only fires when the input actually looks like something it was taught to transform.

Everything is skill/knowledge only — it never touches the loyalty / corrigibility / honesty core —
and every operation is best-effort. Learned skills persist through the
:class:`~nyxara.memory.store.MemoryStore` (tag ``skill``) so what is taught once survives a restart.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.cognition.abstraction import abstract_text

__all__ = ["Operation", "InductiveSkill", "SkillInductionEngine"]

_WORD_RE = re.compile(r"[^\s]+")
_NUM_RE = re.compile(r"^-?\d+$")


# --------------------------------------------------------------------------- #
# The operation library — every op is a pure, deterministic str -> str
# --------------------------------------------------------------------------- #
def _reverse_words(s: str) -> str:
    return " ".join(s.split()[::-1])


def _sort_words(s: str) -> str:
    return " ".join(sorted(s.split()))


def _dedup_words(s: str) -> str:
    seen: set = set()
    out: List[str] = []
    for w in s.split():
        if w not in seen:
            seen.add(w)
            out.append(w)
    return " ".join(out)


# Parameter-free atoms, usable both as a closing op and as an intermediate compose step.
_ATOMS: Dict[str, Callable[[str], str]] = {
    "identity": lambda s: s,
    "lower": str.lower,
    "upper": str.upper,
    "title": str.title,
    "capitalize": str.capitalize,
    "swapcase": str.swapcase,
    "strip": str.strip,
    "reverse_chars": lambda s: s[::-1],
    "reverse_words": _reverse_words,
    "sort_words": _sort_words,
    "dedup_words": _dedup_words,
    "first_word": lambda s: s.split()[0] if s.split() else s,
    "last_word": lambda s: s.split()[-1] if s.split() else s,
}

# The subset safe/useful to compose as *intermediate* steps (order-changing or normalising).
_COMPOSE_ATOMS: Tuple[str, ...] = (
    "lower", "upper", "title", "capitalize", "swapcase", "strip",
    "reverse_chars", "reverse_words", "sort_words", "dedup_words",
)


@dataclass
class Operation:
    """One step of an induced program: a named op plus any induced parameters."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    def apply(self, text: str) -> str:
        atom = _ATOMS.get(self.name)
        if atom is not None:
            return atom(text)
        if self.name == "affix":                        # wrap: prefix + text + suffix
            return f"{self.params.get('prefix', '')}{text}{self.params.get('suffix', '')}"
        if self.name == "strip_affix":                  # remove a known prefix and/or suffix
            p, s = self.params.get("prefix", ""), self.params.get("suffix", "")
            out = text
            if p and out.startswith(p):
                out = out[len(p):]
            if s and out.endswith(s):
                out = out[: len(out) - len(s)]
            return out
        if self.name == "replace":
            return text.replace(str(self.params.get("a", "")), str(self.params.get("b", "")))
        if self.name == "arith":
            return _apply_arith(text, str(self.params.get("op", "add")),
                                int(self.params.get("k", 0)))
        if self.name == "template":
            return _apply_template(text, str(self.params.get("inp", "")),
                                   str(self.params.get("out", "")))
        if self.name == "macro":
            # a library-learned composite primitive (see nyxara.cognition.program_library) —
            # replay whatever permanent, compressed sub-program it was compiled from against
            # _MACRO_REGISTRY. An unknown/unregistered macro name is inert (never raises).
            macro = _MACRO_REGISTRY.get(str(self.params.get("macro", "")))
            if macro is None:
                return text
            out = text
            for op_name in macro.atom_prefix:
                atom = _ATOMS.get(op_name)
                if atom is None:
                    return text
                out = atom(out)
            if not macro.param_keys:
                atom = _ATOMS.get(macro.closing_op_name)
                return atom(out) if atom is not None else out
            closing_params = {k: self.params.get(k, macro.fixed_params.get(k))
                              for k in macro.param_keys}
            return Operation(macro.closing_op_name, closing_params).apply(out)
        return text                                     # unknown op → inert (never raises)

    @property
    def cost(self) -> float:
        """Description length: parameterised ops cost a touch more (Occam prefers simple)."""
        base = 1.0
        if self.name in ("affix", "strip_affix", "replace", "arith"):
            base = 1.4
        elif self.name == "template":
            base = 1.8
        return base

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Operation":
        return cls(name=str(d.get("name", "identity")), params=dict(d.get("params", {})))

    def __str__(self) -> str:
        if not self.params:
            return self.name
        inner = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{self.name}({inner})"


# --------------------------------------------------------------------------- #
# Parameter induction helpers (deterministic; each verifies across ALL pairs)
# --------------------------------------------------------------------------- #
def _to_int(s: str) -> Optional[int]:
    t = s.strip()
    return int(t) if _NUM_RE.match(t) else None


def _apply_arith(text: str, op: str, k: int) -> str:
    n = _to_int(text)
    if n is None:
        return text
    if op == "add":
        return str(n + k)
    if op == "mul":
        return str(n * k)
    if op == "idiv" and k != 0:
        return str(n // k)
    return text


def _common_prefix(items: Sequence[str]) -> str:
    if not items:
        return ""
    p = items[0]
    for it in items[1:]:
        while not it.startswith(p):
            p = p[:-1]
            if not p:
                return ""
    return p


def _common_suffix(items: Sequence[str]) -> str:
    rev = _common_prefix([it[::-1] for it in items])
    return rev[::-1]


def _induce_affix(xs: Sequence[str], ys: Sequence[str]) -> Optional[Operation]:
    """prefix + x + suffix == y for every pair (covers add-prefix / add-suffix / wrap)."""
    prefix, suffix = None, None
    for x, y in zip(xs, ys):
        if not (y.startswith("") and x in y and y.endswith("")):
            pass
        idx = y.find(x)
        if idx < 0:
            return None
        p, s = y[:idx], y[idx + len(x):]
        if prefix is None:
            prefix, suffix = p, s
        elif p != prefix or s != suffix:
            return None
    if not prefix and not suffix:
        return None                                     # that is identity, handled elsewhere
    return Operation("affix", {"prefix": prefix or "", "suffix": suffix or ""})


def _induce_strip_affix(xs: Sequence[str], ys: Sequence[str]) -> Optional[Operation]:
    """y == x with a shared prefix and/or suffix removed (the inverse of affix)."""
    prefix, suffix = None, None
    for x, y in zip(xs, ys):
        idx = x.find(y)
        if idx < 0 or not y:
            return None
        p, s = x[:idx], x[idx + len(y):]
        if prefix is None:
            prefix, suffix = p, s
        elif p != prefix or s != suffix:
            return None
    if not prefix and not suffix:
        return None
    return Operation("strip_affix", {"prefix": prefix or "", "suffix": suffix or ""})


def _induce_replace(xs: Sequence[str], ys: Sequence[str]) -> Optional[Operation]:
    """A single consistent substring substitution a→b that maps every x to its y."""
    x0, y0 = xs[0], ys[0]
    if x0 == y0:
        return None
    # diff by stripping the shared prefix/suffix of the FIRST pair, giving the changed middles
    lead = 0
    while lead < len(x0) and lead < len(y0) and x0[lead] == y0[lead]:
        lead += 1
    tail = 0
    while (tail < len(x0) - lead and tail < len(y0) - lead
           and x0[-1 - tail] == y0[-1 - tail]):
        tail += 1
    a = x0[lead: len(x0) - tail]
    b = y0[lead: len(y0) - tail]
    if not a:
        return None
    op = Operation("replace", {"a": a, "b": b})
    return op if all(op.apply(x) == y for x, y in zip(xs, ys)) else None


def _induce_arith(xs: Sequence[str], ys: Sequence[str]) -> Optional[Operation]:
    """A numeric task: every x and y parse as ints related by a constant +k, ×k, or //k."""
    xn = [_to_int(x) for x in xs]
    yn = [_to_int(y) for y in ys]
    if any(a is None for a in xn) or any(b is None for b in yn):
        return None
    xi = [int(a) for a in xn]                            # type: ignore[arg-type]
    yi = [int(b) for b in yn]                            # type: ignore[arg-type]
    # additive
    deltas = {b - a for a, b in zip(xi, yi)}
    if len(deltas) == 1:
        return Operation("arith", {"op": "add", "k": deltas.pop()})
    # multiplicative (skip x==0 which fixes no ratio)
    ratios = {b // a for a, b in zip(xi, yi) if a != 0 and b % a == 0}
    if ratios and len(ratios) == 1:
        k = ratios.pop()
        op = Operation("arith", {"op": "mul", "k": k})
        if all(op.apply(x) == y for x, y in zip(xs, ys)):
            return op
    return None


def _apply_template(text: str, inp_tmpl: str, out_tmpl: str) -> str:
    """Parse ``text`` with the input slot-template, then fill the output slot-template.

    Templates are whitespace token sequences where ``?`` marks a varying slot (as produced by
    :func:`~nyxara.cognition.abstraction.abstract_text`). Fixed tokens must match positionally;
    captured slot tokens fill the output ``?`` positions in order (a single capture fills them all,
    so ``?`` → ``? ?`` doubles the token)."""
    it, ot = inp_tmpl.split(), out_tmpl.split()
    toks = text.split()
    if len(toks) != len(it):
        return text
    caps: List[str] = []
    for pat, tok in zip(it, toks):
        if pat == "?":
            caps.append(tok)
        elif pat != tok:
            return text                                 # a fixed token did not match → no fire
    if not caps:
        return text
    out: List[str] = []
    j = 0
    for pat in ot:
        if pat == "?":
            out.append(caps[j] if j < len(caps) else caps[-1])
            j += 1
        else:
            out.append(pat)
    return " ".join(out)


def _induce_template(xs: Sequence[str], ys: Sequence[str]) -> Optional[Operation]:
    """Induce an input→output slot template (the key generaliser) via anti-unification."""
    inp = abstract_text(xs)
    out = abstract_text(ys)
    if not inp or not out or "?" not in inp or "?" not in out:
        return None
    op = Operation("template", {"inp": inp, "out": out})
    return op if all(op.apply(x) == y for x, y in zip(xs, ys)) else None


# order matters: the simplest / most specific inducers are tried first (Occam)
_PARAM_INDUCERS: Tuple[Callable[[Sequence[str], Sequence[str]], Optional[Operation]], ...] = (
    _induce_arith, _induce_affix, _induce_strip_affix, _induce_replace, _induce_template,
)

# name -> inducer, keyed for reuse by a library-learning pass (see
# nyxara.cognition.program_library.ProgramLibrary): a mined macro replays the SAME base
# inducer restricted to its permanently-fixed parameters rather than duplicating the logic.
_INDUCERS_BY_OP: Dict[str, Callable[[Sequence[str], Sequence[str]], Optional[Operation]]] = {
    "arith": _induce_arith, "affix": _induce_affix, "strip_affix": _induce_strip_affix,
    "replace": _induce_replace, "template": _induce_template,
}

# process-wide registry a library-learned "macro" Operation resolves against at apply() time
# (populated by nyxara.cognition.program_library.ProgramLibrary.promote/_hydrate). Deliberately
# lives here rather than in program_library.py itself: that module's own __main__ self-test runs
# it as a *second*, distinct module object, which would otherwise see an empty registry copy.
_MACRO_REGISTRY: Dict[str, Any] = {}


def _induce_closing_op(
        xs: Sequence[str], ys: Sequence[str], *,
        extra_inducers: Sequence[Callable[[Sequence[str], Sequence[str]], Optional[Operation]]] = (),
        bias: Optional[Callable[[str], float]] = None) -> Optional[Operation]:
    """A single op that maps every ``x`` to its ``y`` exactly, or None. Simplest first.

    ``extra_inducers`` lets a :class:`~nyxara.cognition.program_library.ProgramLibrary` offer
    its permanently-learned macros as additional candidate closing ops, tried alongside the
    base vocabulary — this is the seam through which the library *empowers* future search.
    ``bias`` (a learned name -> preference score) reorders candidates within their tier so
    primitives with a track record of working are tried first: real search guidance, not a
    fixed textbook order.
    """
    # parameter-free atoms (identity, case, reversal, sort, …) — cheapest description length
    names = list(_ATOMS)
    if bias is not None:
        names.sort(key=bias, reverse=True)
    for name in names:
        op = Operation(name)
        if all(op.apply(x) == y for x, y in zip(xs, ys)):
            return op
    inducers = list(_PARAM_INDUCERS) + list(extra_inducers)
    if bias is not None:
        inducers.sort(key=lambda fn: bias(getattr(fn, "macro_name", getattr(fn, "__name__", ""))),
                      reverse=True)
    for inducer in inducers:
        try:
            op = inducer(xs, ys)
        except Exception:  # noqa: BLE001 — a broken inducer just declines
            op = None
        if op is not None and all(op.apply(x) == y for x, y in zip(xs, ys)):
            return op
    return None


# --------------------------------------------------------------------------- #
# An induced skill
# --------------------------------------------------------------------------- #
@dataclass
class InductiveSkill:
    """A verified, reusable transformation learned from a few demonstrations."""

    name: str
    program: List[Operation]
    demos: List[Tuple[str, str]] = field(default_factory=list)
    confidence: float = 0.6
    generalized: bool = False           # True iff it also predicted a held-out demo

    def apply(self, text: str) -> str:
        out = str(text)
        for op in self.program:
            out = op.apply(out)
        return out

    @property
    def input_template(self) -> Optional[str]:
        try:
            return abstract_text([d[0] for d in self.demos])
        except Exception:  # noqa: BLE001
            return None

    @property
    def cost(self) -> float:
        return sum(op.cost for op in self.program) or 1.0

    def render(self) -> str:
        chain = " ∘ ".join(str(op) for op in reversed(self.program)) or "identity"
        return f"skill {self.name!r}: {chain}"

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "program": [op.to_dict() for op in self.program],
                "demos": [list(d) for d in self.demos], "confidence": self.confidence,
                "generalized": self.generalized}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InductiveSkill":
        return cls(name=str(d.get("name", "skill")),
                   program=[Operation.from_dict(o) for o in d.get("program", [])],
                   demos=[(str(p[0]), str(p[1])) for p in d.get("demos", []) if len(p) == 2],
                   confidence=float(d.get("confidence", 0.6)),
                   generalized=bool(d.get("generalized", False)))


# --------------------------------------------------------------------------- #
# The engine — induce (search + verify), apply, match, persist
# --------------------------------------------------------------------------- #
class SkillInductionEngine:
    """Learn tasks from demonstrations, verify them, and apply them to novel inputs."""

    TAG = "skill"

    def __init__(self, embedder: Any = None, *, store: Any = None,
                 max_depth: int = 3, beam_width: int = 16,
                 min_demos: int = 2, apply_confidence: float = 0.55,
                 match_threshold: float = 0.5,
                 extra_inducers: Sequence[Callable[[Sequence[str], Sequence[str]],
                                                   Optional[Operation]]] = ()) -> None:
        self.embedder = embedder if (embedder is not None and hasattr(embedder, "embed")) else None
        self.store = store
        self.max_depth = max(0, int(max_depth))
        self.beam_width = max(1, int(beam_width))
        self.min_demos = max(1, int(min_demos))
        self.apply_confidence = float(apply_confidence)
        self.match_threshold = float(match_threshold)
        self._skills: Dict[str, InductiveSkill] = {}
        self._hydrated = False
        # library-learning seam: additional closing-op candidates and a learned priority over
        # them, installed by a nyxara.cognition.program_library.ProgramLibrary (see there).
        # Empty/None by default, so behaviour is identical for every caller that doesn't wire one.
        self._extra_inducers: List[Callable[[Sequence[str], Sequence[str]],
                                            Optional[Operation]]] = list(extra_inducers)
        self._bias: Optional[Callable[[str], float]] = None

    def add_inducer(self, fn: Callable[[Sequence[str], Sequence[str]],
                                       Optional[Operation]]) -> None:
        """Register one more candidate closing-op inducer — typically a library-learned macro,
        so the *next* ``induce`` call already searches over the grown vocabulary."""
        self._extra_inducers.append(fn)

    def set_bias(self, fn: Optional[Callable[[str], float]]) -> None:
        """Install a learned name -> preference score used to order search candidates."""
        self._bias = fn

    # ---- induction (breadth-first program search, verified against all demos) ---- #
    def _search(self, pairs: Sequence[Tuple[str, str]]) -> Optional[List[Operation]]:
        inputs = [str(p[0]) for p in pairs]
        outputs = [str(p[1]) for p in pairs]
        # frontier of (program, states); states are the inputs transformed by the program so far
        frontier: List[Tuple[List[Operation], List[str]]] = [([], list(inputs))]
        seen: set = {tuple(inputs)}
        for _depth in range(self.max_depth + 1):
            # 1. try to CLOSE every frontier node with one induced op (shortest programs win)
            for prog, states in frontier:
                op = _induce_closing_op(states, outputs, extra_inducers=self._extra_inducers,
                                        bias=self._bias)
                if op is not None:
                    full = prog + [op]
                    if all(self._run(full, x) == y for x, y in zip(inputs, outputs)):
                        return full
            # 2. otherwise EXPAND by each compose-safe parameter-free atom
            nxt: List[Tuple[List[Operation], List[str]]] = []
            for prog, states in frontier:
                for name in _COMPOSE_ATOMS:
                    op = Operation(name)
                    new_states = [op.apply(s) for s in states]
                    sig = tuple(new_states)
                    if new_states == states or sig in seen:
                        continue
                    seen.add(sig)
                    nxt.append((prog + [op], new_states))
            if not nxt:
                break
            # keep the beam small: prefer states already close to the targets (char overlap)
            nxt.sort(key=lambda ps: _closeness(ps[1], outputs), reverse=True)
            frontier = nxt[: self.beam_width]
        return None

    @staticmethod
    def _run(program: Sequence[Operation], text: str) -> str:
        out = str(text)
        for op in program:
            out = op.apply(out)
        return out

    def induce(self, name: str, pairs: Sequence[Tuple[str, str]]) -> Optional[InductiveSkill]:
        """Learn a skill from ``(input, output)`` demonstrations; None if none fits.

        Accepted only when the induced program reproduces *every* demonstration exactly. With
        ≥3 demos one is held out to measure genuine generalization and lift the confidence."""
        clean = [(str(a).strip(), str(b).strip()) for a, b in pairs
                 if str(a).strip() and str(b).strip()]
        if len(clean) < self.min_demos:
            return None
        program = self._search(clean)
        if program is None:
            return None
        generalized = self._holdout_generalizes(clean)
        skill = InductiveSkill(name=str(name).strip() or "skill", program=program,
                               demos=clean[-5:], generalized=generalized)
        skill.confidence = self._score(skill)
        self._hydrate()
        # keep the better skill if this name was taught before (more demos or simpler program)
        prev = self._skills.get(skill.name)
        if prev is None or skill.confidence >= prev.confidence or len(clean) > len(prev.demos):
            self._skills[skill.name] = skill
            self._persist(skill)
        return self._skills[skill.name]

    def _holdout_generalizes(self, clean: Sequence[Tuple[str, str]]) -> bool:
        """Induce on all-but-one demo and check it predicts the held-out one (true transfer)."""
        if len(clean) < 3:
            return False
        train, held = list(clean[:-1]), clean[-1]
        prog = self._search(train)
        return prog is not None and self._run(prog, held[0]) == held[1]

    def _score(self, skill: InductiveSkill) -> float:
        """Confidence in [0.5, 0.9]: more demos and a simpler program read higher; a held-out
        hit is the strongest signal (measured transfer, not just fit)."""
        conf = 0.55 + 0.06 * len(skill.demos) - 0.05 * (skill.cost - 1.0)
        if skill.generalized:
            conf += 0.15
        return max(0.5, min(0.9, conf))

    # ---- application to a novel input ---- #
    #: a bare transform (all-slot template, no fixed trigger token) needs this much semantic
    #: proximity to a demonstrated input before it may auto-fire — high, so it never hijacks an
    #: unrelated conversational turn (its full generality is still reachable via ``apply``).
    _ANCHORLESS_SIM_FLOOR = 0.75

    def _match(self, skill: InductiveSkill, stimulus: str) -> float:
        """How well ``stimulus`` fits a learned skill (0..1) — structure first, then embedding.

        A skill only auto-fires when the input actually looks like what it was taught to transform.
        A template with a **fixed anchor token** (e.g. ``greet ?``) fires on a positional match to
        that anchor. A **bare transform** (an all-slot ``?`` template like plain uppercase/reversal)
        has no natural trigger, so it fires only on strong semantic similarity to a demonstrated
        input — never on an arbitrary same-length turn. This is what stops ``solve`` reversing an
        unrelated question while leaving the skill's full generality available through ``apply``."""
        text = str(stimulus).strip()
        if not text:
            return 0.0
        tmpl = skill.input_template
        if tmpl:
            it, toks = tmpl.split(), text.split()
            if len(it) == len(toks) and all(p == "?" or p == t for p, t in zip(it, toks)):
                has_anchor = any(p != "?" for p in it)
                if has_anchor:
                    return 1.0                          # a fixed trigger token matched → confident
                # anchorless structural match: allow only if also semantically close to a demo
                sim = self._demo_similarity(text, skill)
                return sim if sim >= self._ANCHORLESS_SIM_FLOOR else 0.0
        # no structural match: fall back to semantic proximity to the demonstrated inputs
        sim = self._demo_similarity(text, skill)
        return sim if sim >= self._ANCHORLESS_SIM_FLOOR else 0.0

    def _demo_similarity(self, text: str, skill: InductiveSkill) -> float:
        """Max cosine of ``text`` to any demonstrated input (0.0 without an embedder/demos)."""
        if self.embedder is None or not skill.demos:
            return 0.0
        try:
            from nyxara.cognition.few_shot import cosine
            qv = self.embedder.embed(text)
            best = max(cosine(qv, self.embedder.embed(d[0])) for d in skill.demos)
            return max(0.0, min(1.0, best))
        except Exception:  # noqa: BLE001 — matching is best-effort
            return 0.0

    def apply_skill(self, name: str, text: str) -> Optional[Tuple[str, float]]:
        """Apply a named learned skill to ``text``; None if unknown or it changes nothing."""
        self._hydrate()
        skill = self._skills.get(str(name))
        if skill is None:
            return None
        out = skill.apply(text)
        return (out, skill.confidence) if out != text else None

    def solve(self, stimulus: str) -> Optional[Tuple[str, float]]:
        """Best-matching learned skill applied to ``stimulus`` → ``(answer, confidence)`` or None.

        Conservative: only fires above the match + confidence floors and only when the transform
        actually changes the input, so it never hijacks an unrelated turn."""
        self._hydrate()
        if not self._skills:
            return None
        best: Optional[Tuple[str, float]] = None
        best_key = (0.0, 0.0)
        for skill in self._skills.values():
            m = self._match(skill, stimulus)
            if m < self.match_threshold or skill.confidence < self.apply_confidence:
                continue
            out = skill.apply(stimulus)
            if not out or out == stimulus.strip():
                continue
            key = (m, skill.confidence)
            if key > best_key:
                best_key = key
                best = (out, min(0.9, m * skill.confidence + 0.1))
        return best

    # ---- introspection ---- #
    def skills(self) -> List[InductiveSkill]:
        self._hydrate()
        return list(self._skills.values())

    def __len__(self) -> int:
        self._hydrate()
        return len(self._skills)

    # ---- persistence (compounds across restarts, like few_shot.py) ---- #
    def _persist(self, skill: InductiveSkill) -> None:
        if self.store is None:
            return
        try:
            from nyxara.memory.store import MemoryType
            self.store.remember(
                f"skill:{skill.name} = {skill.render()}",
                mem_type=MemoryType.PROCEDURAL,
                importance=min(1.0, 0.55 + 0.05 * len(skill.demos)),
                tags=[self.TAG],
                metadata={"skill": skill.to_dict(), "name": skill.name})
        except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
            pass

    def _hydrate(self) -> None:
        if self._hydrated:
            return
        self._hydrated = True
        if self.store is None:
            return
        try:
            hits = self.store.recall("", k=1000, tags=[self.TAG], strengthen=False)
            for rec, _ in hits:
                data = rec.metadata.get("skill") if isinstance(rec.metadata, dict) else None
                if isinstance(data, dict):
                    skill = InductiveSkill.from_dict(data)
                    cur = self._skills.get(skill.name)
                    if cur is None or skill.confidence >= cur.confidence:
                        self._skills[skill.name] = skill
        except Exception:  # noqa: BLE001 — hydration is best-effort
            pass


def _closeness(states: Sequence[str], targets: Sequence[str]) -> float:
    """A cheap beam heuristic: mean character-set overlap between states and their targets."""
    total = 0.0
    for s, t in zip(states, targets):
        ss, ts = set(s), set(t)
        if not ss and not ts:
            total += 1.0
        elif ss or ts:
            total += len(ss & ts) / max(1, len(ss | ts))
    return total / max(1, len(targets))


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA skill-induction (few-shot program synthesis) self-test")
    print("=" * 70)

    eng = SkillInductionEngine()

    # 1. character reversal learned from TWO demos, applied to a NEVER-SEEN input
    s = eng.induce("reverse", [("hello", "olleh"), ("world", "dlrow")])
    assert s is not None, "should induce a reversal program"
    print(f"\n{s.render()}")
    assert s.apply("nyxara") == "araxyn", "generalises to a held-out input"
    print("reverse 'nyxara'    : araxyn ✓")

    # 2. an arithmetic task: double the number (multiplicative), from three demos
    d = eng.induce("double", [("2", "4"), ("3", "6"), ("10", "20")])
    assert d is not None and d.apply("7") == "14"
    print(f"double '7'          : {d.apply('7')} ✓  (generalized={d.generalized})")
    assert d.generalized, "3 demos → held-out generalization is measured"

    # 3. a slot template: 'greet X' -> 'hello X !' — pure structural generalization
    t = eng.induce("greet", [("greet jp", "hello jp !"), ("greet sky", "hello sky !")])
    assert t is not None and t.apply("greet nyxara") == "hello nyxara !"
    print(f"greet 'nyxara'      : {t.apply('greet nyxara')!r} ✓")

    # 4. verification REJECTS a non-transformation (no consistent program) — honesty
    bad = eng.induce("nonsense", [("a", "zzz"), ("b", "qqq"), ("c", "mmm")])
    assert bad is None, "no consistent program exists → refuses to learn (no bluff)"
    print("rejects unlearnable : ✓")

    # 5. solve() only fires on a structurally-matching stimulus, never hijacks
    ans = eng.solve("greet master")
    assert ans is not None and ans[0] == "hello master !"
    print(f"solve('greet master'): {ans[0]!r} (conf {ans[1]:.2f}) ✓")
    assert eng.solve("what is the capital of France") is None, "unrelated turn is left alone"
    print("no false fire       : ✓")

    # 6. persistence through a real store — a fresh engine hydrates the taught skill
    from nyxara.memory.store import MemoryStore
    store = MemoryStore()
    e1 = SkillInductionEngine(store.embedder, store=store)
    e1.induce("reverse", [("abc", "cba"), ("de", "ed")])
    e2 = SkillInductionEngine(store.embedder, store=store)
    assert any(sk.name == "reverse" for sk in e2.skills()), "skill survives across engines"
    print("persisted skill     : reverse ✓")

    print("\nALL SELF-TESTS PASSED ✓")
