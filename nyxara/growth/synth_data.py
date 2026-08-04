"""NYXARA · growth/synth_data.py — verified synthetic training data, at corpus scale (🧪→📚).

:mod:`~nyxara.growth.synthesis` already does the hard and interesting part: it generates
arithmetic, algebra, logic, number-theory and code items, and keeps one **only** when
:class:`~nyxara.growth.prover.Prover` returns ``PROVEN`` or the sandbox agrees with an
independent reference implementation. Its ``RivalVerifier`` is explicitly built so that
generation is trusted by no one. What it is *sized* for is an idle-tick curation pass — batches
of eight, fed to knowledge and the flywheel.

This module turns that into a corpus source: hundreds of thousands of verified items, rendered
as training documents in her own template, streamed into :mod:`~nyxara.growth.corpus`. It adds
four kinds of data that a 300M model needs and that no scraped corpus contains in the right
shape:

* **Worked-step math** — not ``Q → A`` but ``Q → the steps → A``. A small model that has only
  ever seen answers has no mechanism for producing one; a model that has seen the derivation
  can reproduce the procedure. The steps are generated *from* the same construction that
  produces the answer, so they are correct by construction rather than by hope.
* **Verified code** — a candidate is executed against a reference in the existing restricted
  sandbox, input by input, and kept only if every output matches.
* **Tool-call traces** — this is the single biggest capability multiplier at 300M. A model this
  size cannot store facts, but it can absolutely learn *when to call the calculator* and how to
  use what comes back. She already owns the tools (:mod:`~nyxara.agency.default_tools`); what
  she has never had is training data in the shape of using them.
* **Retrieval traces** — ``context → question → answer`` shapes, so that giving her a retrieved
  passage at inference actually helps. Knowledge belongs in the memory store at this scale, not
  in the weights, and using retrieved context is a learned skill.

**The honest limit.** Everything here is *verifiable*, which is exactly why it is safe to
generate at scale — and it is also the boundary of what it can teach. Synthetic data makes her
reliable at procedures that can be checked. It cannot make her knowledgeable about the world.
That is what the streamed general corpus is for, and neither substitutes for the other.

Pure standard library plus the optional ``sympy`` (already a dependency) for symbolic
derivations. Depends on ``growth/synthesis``, ``growth/prover`` and ``mind/llm`` for the
training template. Nothing imports back.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "SynthReport",
    "generate_domain_docs",
    "verified_items",
    "worked_math_docs",
    "tool_call_docs",
    "retrieval_docs",
]


@dataclass
class SynthReport:
    """What was generated, what survived verification, and what was thrown away."""

    generated: int = 0
    verified: int = 0
    rejected: int = 0
    by_kind: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        if message and message not in self.notes:
            self.notes.append(message)

    @property
    def acceptance(self) -> float:
        return self.verified / self.generated if self.generated else 0.0

    def summary(self) -> str:
        kinds = ", ".join(f"{k} {v}" for k, v in sorted(self.by_kind.items()))
        notes = ("\n  · " + "\n  · ".join(self.notes)) if self.notes else ""
        return (f"· {self.verified}/{self.generated} verified ({self.acceptance:.0%})"
                f"{' [' + kinds + ']' if kinds else ''}{notes}")

    def to_dict(self) -> Dict[str, Any]:
        return {"generated": self.generated, "verified": self.verified,
                "rejected": self.rejected, "by_kind": dict(self.by_kind),
                "acceptance": round(self.acceptance, 4), "notes": list(self.notes)}


def _doc(prompt: str, answer: str) -> str:
    """Render a supervised pair in her own template — train/inference parity.

    The same function ``growth/dataset.py`` uses, for the same reason: the model must learn to
    produce its reply exactly where inference will ask for it.
    """
    try:
        from nyxara.mind.llm import format_self_training_doc
        return format_self_training_doc(prompt, answer)
    except Exception:  # noqa: BLE001 — a plain rendering beats losing the pair
        return f"{prompt}\n\n{answer}\n"


# --------------------------------------------------------------------------- #
# The existing verified generators, at scale
# --------------------------------------------------------------------------- #
def verified_items(n: int, *, domains: Optional[Sequence[str]] = None, seed: int = 0,
                   batch: int = 64, report: Optional[SynthReport] = None
                   ) -> Iterator[Any]:
    """Stream ``n`` items from :mod:`~nyxara.growth.synthesis` that passed verification.

    Generation happens in batches so the RNG keeps moving (the generator is stateful by design,
    so successive calls produce new items rather than re-proposing the same ones). Only
    ``PROVEN``/sandbox-matched items are yielded — a rejection is counted, never downgraded into
    an "unverified but probably fine" item.
    """
    report = report if report is not None else SynthReport()
    try:
        from nyxara.growth.synthesis import RivalVerifier, SyntheticGenerators
    except Exception as exc:  # noqa: BLE001
        report.note(f"synthesis unavailable ({exc})")
        return

    gen = SyntheticGenerators(seed=seed)
    verifier = RivalVerifier()
    doms = tuple(domains) if domains else None
    produced = 0
    # Bound the attempts: a domain whose generator keeps producing unprovable items must not
    # spin forever. 20× the target is generous and still terminates.
    for _ in range(max(1, (n // max(1, batch)) * 20 + 1)):
        if produced >= n:
            break
        for item in gen.batch(batch, doms):
            report.generated += 1
            try:
                ok, _why = verifier.verify(item)
            except Exception:  # noqa: BLE001 — an erroring verifier rejects, never accepts
                ok = False
            if not ok:
                report.rejected += 1
                continue
            report.verified += 1
            report.by_kind[item.domain] = report.by_kind.get(item.domain, 0) + 1
            produced += 1
            yield item
            if produced >= n:
                break


# --------------------------------------------------------------------------- #
# Worked-step math — the derivation, not just the answer
# --------------------------------------------------------------------------- #
def _linear_equation(rng: random.Random) -> Tuple[str, str]:
    """``ax + b = c`` with an exact rational solution, and the steps that get there."""
    a = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 12])
    x = rng.randint(-20, 20)
    b = rng.randint(-50, 50)
    c = a * x + b
    sign = "-" if b < 0 else "+"
    prompt = f"Solve for x: {a}x {sign} {abs(b)} = {c}"
    steps = (
        f"Start: {a}x {sign} {abs(b)} = {c}\n"
        f"{'Add' if b < 0 else 'Subtract'} {abs(b)} {'to' if b < 0 else 'from'} both sides: "
        f"{a}x = {c - b}\n"
        f"Divide both sides by {a}: x = {(c - b) // a}\n"
        f"Check: {a}({(c - b) // a}) {sign} {abs(b)} = {c}\n"
        f"x = {x}"
    )
    return prompt, steps


def _long_arithmetic(rng: random.Random) -> Tuple[str, str]:
    """Multi-digit addition, carried column by column — what digit-level tokens are FOR."""
    a = rng.randint(100, 99_999)
    b = rng.randint(100, 99_999)
    prompt = f"What is {a} + {b}?"
    digits_a, digits_b = str(a)[::-1], str(b)[::-1]
    width = max(len(digits_a), len(digits_b))
    carry, cols = 0, []
    for i in range(width):
        da = int(digits_a[i]) if i < len(digits_a) else 0
        db = int(digits_b[i]) if i < len(digits_b) else 0
        total = da + db + carry
        cols.append(f"  column {i + 1}: {da} + {db} + carry {carry} = {total} "
                    f"-> digit {total % 10}, carry {total // 10}")
        carry = total // 10
    steps = "Add column by column from the right:\n" + "\n".join(cols)
    if carry:
        steps += f"\n  final carry: {carry}"
    steps += f"\n{a} + {b} = {a + b}"
    return prompt, steps


def _fraction_sum(rng: random.Random) -> Tuple[str, str]:
    a, b = rng.randint(1, 12), rng.randint(2, 12)
    c, d = rng.randint(1, 12), rng.randint(2, 12)
    result = Fraction(a, b) + Fraction(c, d)
    prompt = f"What is {a}/{b} + {c}/{d}? Give the answer in lowest terms."
    steps = (
        f"Common denominator: {b} x {d} = {b * d}\n"
        f"{a}/{b} = {a * d}/{b * d}\n"
        f"{c}/{d} = {c * b}/{b * d}\n"
        f"Sum: {a * d + c * b}/{b * d}\n"
        f"Lowest terms: {result.numerator}/{result.denominator}"
    )
    return prompt, steps


def _sympy_derivative(rng: random.Random) -> Optional[Tuple[str, str]]:
    """A symbolic derivative with its rule named. Requires sympy (already a dependency)."""
    try:
        import sympy as sp
    except Exception:  # noqa: BLE001
        return None
    x = sp.Symbol("x")
    coeff = rng.randint(2, 9)
    power = rng.randint(2, 6)
    inner = rng.choice([x, x + rng.randint(1, 5)])
    expr = coeff * inner ** power
    derivative = sp.simplify(sp.diff(expr, x))
    prompt = f"Differentiate with respect to x: {sp.sstr(sp.expand(expr))}"
    steps = (
        f"Expression: {sp.sstr(expr)}\n"
        f"Apply the power rule d/dx[u^n] = n*u^(n-1)*du/dx with n = {power}\n"
        f"Result: {sp.sstr(derivative)}"
    )
    return prompt, steps


_WORKED: Tuple[Callable[[random.Random], Optional[Tuple[str, str]]], ...] = (
    _linear_equation, _long_arithmetic, _fraction_sum, _sympy_derivative,
)


def worked_math_docs(n: int, *, seed: int = 0,
                     report: Optional[SynthReport] = None) -> Iterator[str]:
    """Math documents that show the working, not only the result.

    Every generator here *constructs* the problem from a known solution, so the steps and the
    answer are correct by construction — there is no model in the loop that could be wrong, and
    nothing needs a prover because nothing was guessed.
    """
    report = report if report is not None else SynthReport()
    rng = random.Random(seed)
    made = 0
    for _ in range(n * 3):
        if made >= n:
            break
        built = _WORKED[rng.randrange(len(_WORKED))](rng)
        if built is None:
            continue
        prompt, steps = built
        report.generated += 1
        report.verified += 1
        report.by_kind["worked_math"] = report.by_kind.get("worked_math", 0) + 1
        made += 1
        yield _doc(prompt, steps)


# --------------------------------------------------------------------------- #
# Tool-call traces — the biggest capability multiplier at this size
# --------------------------------------------------------------------------- #
# The shape a trace takes at inference. Kept as a module constant so sft.py, the tool router and
# the eval battery all agree on one syntax rather than three that nearly match.
TOOL_CALL_OPEN = "<tool>"
TOOL_CALL_CLOSE = "</tool>"
TOOL_RESULT_OPEN = "<result>"
TOOL_RESULT_CLOSE = "</result>"


def _tool_trace(name: str, args: str, result: str, answer: str) -> str:
    return (f"{TOOL_CALL_OPEN}{name}({args}){TOOL_CALL_CLOSE}\n"
            f"{TOOL_RESULT_OPEN}{result}{TOOL_RESULT_CLOSE}\n{answer}")


def tool_call_docs(n: int, *, seed: int = 0,
                   report: Optional[SynthReport] = None) -> Iterator[str]:
    """Traces of deciding to call a tool, and using what it returns.

    A 300M model cannot memorise the world, but it can learn a *policy*: this question needs the
    calculator, that one needs a search, and here is how to read the answer back. Every result
    embedded in a trace is computed for real, so the model never learns to imitate a fabricated
    tool output — which would teach it precisely the wrong lesson.
    """
    report = report if report is not None else SynthReport()
    rng = random.Random(seed + 977)
    made = 0
    while made < n:
        pick = rng.randrange(4)
        if pick == 0:
            a, b = rng.randint(1000, 999_999), rng.randint(1000, 999_999)
            prompt = f"What is {a} * {b}?"
            body = _tool_trace("calculator", f"{a} * {b}", str(a * b),
                               f"{a} * {b} = {a * b}")
        elif pick == 1:
            n_val = rng.randint(20, 400)
            prompt = f"Is {n_val} prime?"
            is_prime = n_val > 1 and all(n_val % d for d in range(2, int(n_val ** 0.5) + 1))
            body = _tool_trace("python", f"all({n_val} % d for d in range(2, {n_val}))",
                               str(is_prime),
                               f"{n_val} is {'prime' if is_prime else 'not prime'}.")
        elif pick == 2:
            items = [rng.randint(1, 500) for _ in range(rng.randint(4, 9))]
            prompt = f"What is the average of {', '.join(map(str, items))}?"
            mean = sum(items) / len(items)
            body = _tool_trace("calculator", f"({'+'.join(map(str, items))})/{len(items)}",
                               f"{mean:.4f}", f"The average is {mean:.4f}.")
        else:
            # The other half of the policy: knowing when NOT to reach for a tool. Without these
            # the model learns to call the calculator for "what is 2+2", which is its own failure.
            a, b = rng.randint(1, 9), rng.randint(1, 9)
            prompt = f"What is {a} + {b}?"
            body = f"{a} + {b} = {a + b}"
        report.generated += 1
        report.verified += 1
        report.by_kind["tool_call"] = report.by_kind.get("tool_call", 0) + 1
        made += 1
        yield _doc(prompt, body)


# --------------------------------------------------------------------------- #
# Retrieval traces — using context that is handed to you
# --------------------------------------------------------------------------- #
def retrieval_docs(n: int, *, seed: int = 0, passages: Optional[Sequence[str]] = None,
                   report: Optional[SynthReport] = None) -> Iterator[str]:
    """``context → question → answer`` shapes, including when the context does NOT contain it.

    The negative cases are the important half. A model trained only on answerable
    context-question pairs learns that an answer always exists in the passage, and will
    confabulate one when it does not. Teaching "the passage does not say" is what makes
    retrieval trustworthy rather than merely fluent.
    """
    report = report if report is not None else SynthReport()
    rng = random.Random(seed + 5501)
    facts = list(passages) if passages else _synthetic_facts(rng, max(24, n // 4))
    made = 0
    while made < n and facts:
        subject, attribute, value = facts[rng.randrange(len(facts))]
        answerable = rng.random() > 0.25
        if answerable:
            context = f"{subject} has {attribute} of {value}."
            prompt = (f"Context: {context}\n\nQuestion: What is the {attribute} of {subject}?")
            answer = f"{value}."
        else:
            other, other_attr, other_val = facts[rng.randrange(len(facts))]
            context = f"{other} has {other_attr} of {other_val}."
            prompt = (f"Context: {context}\n\nQuestion: What is the {attribute} of {subject}?")
            answer = ("The context does not say. It only states the "
                      f"{other_attr} of {other}.")
        report.generated += 1
        report.verified += 1
        report.by_kind["retrieval"] = report.by_kind.get("retrieval", 0) + 1
        made += 1
        yield _doc(prompt, answer)


def _synthetic_facts(rng: random.Random, n: int) -> List[Tuple[str, str, str]]:
    """Invented subject/attribute/value triples.

    Deliberately fictional: the point is to train the *skill* of reading a passage, and using
    real-world facts would let the model answer from memory instead — which teaches it nothing
    about retrieval and quietly contaminates the eval besides.
    """
    subjects = [f"Unit {rng.randint(100, 999)}-{chr(rng.randrange(65, 91))}" for _ in range(n)]
    attributes = ["mass", "length", "operating temperature", "cycle time", "output voltage",
                  "capacity", "resonant frequency", "tolerance"]
    out = []
    for subject in subjects:
        out.append((subject, rng.choice(attributes),
                    f"{rng.randint(2, 990)} {rng.choice(['kg', 'mm', 'C', 'ms', 'V', 'L', 'Hz', '%'])}"))
    return out


# --------------------------------------------------------------------------- #
# The API growth/corpus.py calls
# --------------------------------------------------------------------------- #
def _registry(domain: str) -> tuple:
    """The generator bank for ``domain``, imported lazily so a missing sibling degrades."""
    try:
        if domain == "math":
            from nyxara.growth.synth_math import GENERATORS
            return GENERATORS
        if domain == "code":
            from nyxara.growth.synth_code import GENERATORS
            return GENERATORS
        if domain == "conversation":
            from nyxara.growth.synth_chat import GENERATORS
            return GENERATORS
    except Exception:  # noqa: BLE001 — fall back to the legacy path rather than emit nothing
        return ()
    return ()


def generate_domain_docs(domain: str, n: int, *, seed: int = 0,
                         report: Optional[SynthReport] = None,
                         certify: bool = False) -> Iterator[str]:
    """Stream ``n`` verified training documents for ``domain``.

    Dispatches to the generator banks in :mod:`~nyxara.growth.synth_math`,
    :mod:`~nyxara.growth.synth_code` and :mod:`~nyxara.growth.synth_chat`, each of which
    declares a state space that :func:`~nyxara.growth.synth_common.draw` enforces.

    That enforcement is the point. This function previously returned, for ``domain="code"``,
    **six distinct documents no matter how many were asked for** — six templates whose prompt
    and answer were fully deterministic, sampled three thousand times and collapsed by dedup.
    Nothing detected it because no generator declared how much it could actually produce.

    The signature is unchanged for every existing caller (``corpus.py``, ``train_300m.py`` and
    four test modules); ``certify`` is new and opt-in.
    """
    report = report if report is not None else SynthReport()
    n = max(0, int(n))
    if n == 0:
        return

    generators = _registry(domain)
    if generators:
        from nyxara.growth.synth_common import SynthReport as CommonReport
        from nyxara.growth.synth_common import draw

        inner = CommonReport()
        for text in draw(generators, n, seed=seed, report=inner, certify=certify):
            yield text
        # Fold the sibling's accounting back into the caller's report object, whose type the
        # existing callers depend on.
        report.generated += inner.generated
        report.verified += inner.verified
        report.rejected += inner.rejected
        for name, count in inner.by_kind.items():
            report.by_kind[name] = report.by_kind.get(name, 0) + count
        for note in inner.notes:
            report.note(note)
        if inner.capped:
            report.note("state-space cap reached: "
                        + ", ".join(sorted(inner.capped)))
        return

    # ---- legacy fallback: the original inline generators ---- #
    if domain == "math":
        half = n // 2
        yield from worked_math_docs(half, seed=seed, report=report)
        for item in verified_items(n - half, seed=seed, report=report,
                                   domains=("arithmetic", "algebra", "number_theory", "logic")):
            yield _doc(item.prompt, item.answer)

    elif domain == "code":
        for item in verified_items(n, seed=seed, report=report, domains=("code",)):
            yield _doc(item.prompt, item.answer)

    elif domain == "conversation":
        half = n // 2
        yield from tool_call_docs(half, seed=seed, report=report)
        yield from retrieval_docs(n - half, seed=seed, report=report)

    else:
        report.note(f"no synthetic generator for domain '{domain}'")
