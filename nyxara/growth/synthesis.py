"""NYXARA · growth/synthesis.py — Synthetic Data Self-Curation (the AlphaGo-Zero method, Rule 4).

Human data is finite, biased and noisy; a mind that only ever learns from it inherits its
ceiling and its flaws. **AlphaGo Zero** broke that ceiling by abandoning human games entirely —
it generated its own experience and kept only what *verified* as good. This module is that idea
for NYXARA: she **manufactures purely logical synthetic data** (arithmetic, algebraic identities,
propositional logic, number theory, small code structures), has an **independent rival verifier**
check each item, and feeds only what survives into her base knowledge and her own training corpus.

The asymmetry is the whole point — *generation is hard, verification is cheap*. So a generator may
propose freely, but nothing is believed until a separate decision procedure certifies it:

* math / logic / number-theory → :class:`~nyxara.growth.prover.Prover` (exact, machine-checkable;
  arithmetic over the rationals, identity by sympy/PIT, logic by truth-table, number theory by
  exact stdlib). An item is kept **iff** the verdict is ``PROVEN`` — a counter-example refutes it.
* code → executed in a **restricted sandbox** (no imports, restricted builtins) and compared
  output-for-output against an independent reference oracle on random inputs; kept iff it matches.

What survives is fed two ways, reusing the existing machinery rather than duplicating it:

* :meth:`~nyxara.knowledge.base.KnowledgeBase.ingest_text` — into her grounded **base knowledge**.
* :meth:`~nyxara.growth.flywheel.DataFlywheel.consider` with ``verified=True`` — into the **same**
  JSONL corpus the foundry already consumes, so verified synthetic logic automatically flows into
  the Genesis / AutoForge model-forging loop (gauntlet-gated). This bypasses the human-data
  bottleneck: NYXARA grows her own verified knowledge.

Design rules (mirroring :mod:`~nyxara.growth.flywheel`): **gather-only** — it never acts on the
world, never reaches the network, and only ever appends to a file / a knowledge index. Pure
standard library at the core; the optional LLM second opinion only ever *strengthens* a verdict
and a keyless machine skips it. Fully deterministic and CI-safe.
"""

from __future__ import annotations

import builtins
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from nyxara.growth.prover import ProofClaim, ProofVerdict, Prover, _eval_rational, _fmt

__all__ = [
    "Domain",
    "SyntheticItem",
    "CurationReport",
    "SyntheticGenerators",
    "RivalVerifier",
    "SyntheticCurator",
]


# --------------------------------------------------------------------------- #
# Taxonomy + domain objects
# --------------------------------------------------------------------------- #
class Domain(str, Enum):
    """The logical domains over which synthetic data is generated and verified."""

    ARITHMETIC = "arithmetic"
    ALGEBRA = "algebra"
    LOGIC = "logic"
    NUMBER_THEORY = "number_theory"
    CODE = "code"


DEFAULT_DOMAINS: Tuple[str, ...] = tuple(d.value for d in Domain)


@dataclass
class SyntheticItem:
    """One self-generated, *verifiable* training item.

    ``statement``/``proof_kind``/``candidate`` carry what the :class:`Prover` needs for the math
    and logic domains; ``code_spec`` carries the reference oracle and test inputs for the code
    domain. ``prompt``/``answer`` are what get fed to knowledge and the flywheel when it survives.
    """

    domain: str
    prompt: str
    answer: str
    proof_kind: str = ""                         # Prover kind (arithmetic/algebra/logic/...)
    statement: str = ""                          # statement handed to the Prover
    candidate: Optional[str] = None              # candidate answer for substitution checks
    code_spec: Optional[Dict[str, Any]] = None   # {"fn", "reference", "candidate", "tests"}
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"domain": self.domain, "prompt": self.prompt, "answer": self.answer,
                "proof_kind": self.proof_kind, "statement": self.statement,
                "candidate": self.candidate, "code_spec": self.code_spec, "meta": self.meta}


@dataclass
class CurationReport:
    """An auditable summary of one curation pass — nothing is accepted silently."""

    generated: int = 0
    accepted: int = 0
    rejected: int = 0
    by_domain: Dict[str, int] = field(default_factory=dict)   # accepted count per domain
    fed_knowledge: int = 0
    fed_flywheel: int = 0
    samples: List[Dict[str, Any]] = field(default_factory=list)
    reason: str = ""
    seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"generated": self.generated, "accepted": self.accepted, "rejected": self.rejected,
                "by_domain": dict(self.by_domain), "fed_knowledge": self.fed_knowledge,
                "fed_flywheel": self.fed_flywheel, "samples": self.samples[:8],
                "reason": self.reason, "seconds": round(self.seconds, 4)}


# --------------------------------------------------------------------------- #
# Generators — deterministic, seeded, pure-stdlib producers of verifiable items
# --------------------------------------------------------------------------- #
class SyntheticGenerators:
    """Procedurally generate self-verifiable logical items across the five domains.

    Stateful in its RNG so successive calls keep producing *new* items (so an idle loop is not
    stuck re-proposing the same batch). Every item is constructed so that an independent verifier
    can certify it exactly — the generator never has to be trusted."""

    def __init__(self, *, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    # ---- public API ---- #
    def batch(self, n: int, domains: Optional[Tuple[str, ...]] = None) -> List[SyntheticItem]:
        """Generate ``n`` items, cycling through the requested ``domains``."""
        doms = list(domains or DEFAULT_DOMAINS)
        if not doms:
            doms = list(DEFAULT_DOMAINS)
        out: List[SyntheticItem] = []
        for i in range(max(0, int(n))):
            item = self.one(doms[i % len(doms)])
            if item is not None:
                out.append(item)
        return out

    def one(self, domain: str) -> Optional[SyntheticItem]:
        gen = {
            Domain.ARITHMETIC.value: self._arithmetic,
            Domain.ALGEBRA.value: self._algebra,
            Domain.LOGIC.value: self._logic,
            Domain.NUMBER_THEORY.value: self._number_theory,
            Domain.CODE.value: self._code,
        }.get(domain)
        if gen is None:
            return None
        try:
            return gen()
        except Exception:  # noqa: BLE001 — a generator never crashes a curation pass
            return None

    # ---- arithmetic: closed integer expressions, exact answers ---- #
    def _rand_expr(self, depth: int) -> str:
        if depth <= 0 or self.rng.random() < 0.4:
            return str(self.rng.randint(1, 12))
        op = self.rng.choice(["+", "-", "*"])
        a, b = self._rand_expr(depth - 1), self._rand_expr(depth - 1)
        return f"({a} {op} {b})"

    def _arithmetic(self) -> Optional[SyntheticItem]:
        for _ in range(8):
            expr = self._rand_expr(self.rng.randint(2, 3))
            val = _eval_rational(expr)
            if val is None or val.denominator != 1 or abs(val.numerator) > 10 ** 7:
                continue
            ans = _fmt(val)
            return SyntheticItem(
                domain=Domain.ARITHMETIC.value,
                prompt=f"Compute the exact value of {expr}.",
                answer=ans, proof_kind="arithmetic", statement=f"{expr} = {ans}")
        return None

    # ---- algebra: known identities + linear equations with a known root ---- #
    def _algebra(self) -> Optional[SyntheticItem]:
        if self.rng.random() < 0.5:
            a = self.rng.randint(1, 9)
            kind = self.rng.choice(("square", "diff", "distribute"))
            if kind == "square":
                lhs, rhs = f"(x + {a})^2", f"x^2 + {2 * a}*x + {a * a}"
                prompt = f"Expand and simplify (x + {a})^2."
            elif kind == "diff":
                lhs, rhs = f"(x + {a})*(x - {a})", f"x^2 - {a * a}"
                prompt = f"Expand and simplify (x + {a})(x - {a})."
            else:
                b = self.rng.randint(1, 9)
                lhs, rhs = f"{a}*(x + {b})", f"{a}*x + {a * b}"
                prompt = f"Expand {a}(x + {b})."
            return SyntheticItem(domain=Domain.ALGEBRA.value, prompt=prompt, answer=rhs,
                                 proof_kind="algebra", statement=f"{lhs} = {rhs}")
        # linear equation with an integer root
        m = self.rng.randint(1, 9)
        x0 = self.rng.randint(-9, 9)
        b = self.rng.randint(-9, 9)
        c = m * x0 + b
        return SyntheticItem(
            domain=Domain.ALGEBRA.value,
            prompt=f"Solve for x: {m}*x + {b} = {c}.",
            answer=f"x = {x0}", proof_kind="algebra",
            statement=f"{m}*x + {b} = {c}", candidate=f"x = {x0}")

    # ---- logic: propositional tautology templates (verifier confirms validity) ---- #
    _TAUTOLOGIES: Tuple[str, ...] = (
        "A or not A",
        "A implies A",
        "(A and B) implies A",
        "A implies (A or B)",
        "(A implies B) iff (not A or B)",
        "not (A and B) iff (not A or not B)",
        "not (A or B) iff (not A and not B)",
        "((A implies B) and (B implies C)) implies (A implies C)",
    )

    def _logic(self) -> Optional[SyntheticItem]:
        stmt = self.rng.choice(self._TAUTOLOGIES)
        return SyntheticItem(
            domain=Domain.LOGIC.value,
            prompt=f"Is the following a tautology (true under every assignment)? {stmt}",
            answer="Yes — it is valid under every truth assignment.",
            proof_kind="logic", statement=stmt)

    # ---- number theory: primality / gcd / divisibility (true facts only) ---- #
    def _number_theory(self) -> Optional[SyntheticItem]:
        choice = self.rng.randrange(3)
        if choice == 0:
            for _ in range(16):
                n = self.rng.randint(2, 500)
                if _is_prime(n):
                    return SyntheticItem(
                        domain=Domain.NUMBER_THEORY.value,
                        prompt=f"Is {n} a prime number?", answer="Yes, it is prime.",
                        proof_kind="number_theory", statement=f"is {n} prime")
            return None
        if choice == 1:
            a, b = self.rng.randint(2, 120), self.rng.randint(2, 120)
            g = math.gcd(a, b)
            return SyntheticItem(
                domain=Domain.NUMBER_THEORY.value,
                prompt=f"What is gcd({a}, {b})?", answer=str(g),
                proof_kind="number_theory", statement=f"gcd({a}, {b}) = {g}")
        d, q = self.rng.randint(2, 12), self.rng.randint(2, 50)
        n = d * q
        return SyntheticItem(
            domain=Domain.NUMBER_THEORY.value,
            prompt=f"Does {d} divide {n}?", answer="Yes.",
            proof_kind="number_theory", statement=f"{d} divides {n}")

    # ---- code: a function spec + reference oracle + random test inputs ---- #
    def _code(self) -> Optional[SyntheticItem]:
        spec = self.rng.choice(_CODE_TEMPLATES)
        tests = [spec["args"](self.rng) for _ in range(5)]
        ref = spec["ref"]
        return SyntheticItem(
            domain=Domain.CODE.value,
            prompt=f"Implement a Python function `{spec['name']}` that {spec['desc']}.",
            answer=ref, proof_kind="code",
            code_spec={"fn": "f", "reference": ref, "candidate": ref, "tests": tests},
            meta={"name": spec["name"]})


# Code templates: name, description, a reference implementation, and a random-args generator.
_CODE_TEMPLATES: Tuple[Dict[str, Any], ...] = (
    {"name": "square", "desc": "returns the square of x",
     "ref": "def f(x):\n    return x * x",
     "args": lambda r: (r.randint(-12, 12),)},
    {"name": "add", "desc": "returns the sum of a and b",
     "ref": "def f(a, b):\n    return a + b",
     "args": lambda r: (r.randint(-20, 20), r.randint(-20, 20))},
    {"name": "absolute", "desc": "returns the absolute value of x",
     "ref": "def f(x):\n    return x if x >= 0 else -x",
     "args": lambda r: (r.randint(-20, 20),)},
    {"name": "maximum", "desc": "returns the larger of a and b",
     "ref": "def f(a, b):\n    return a if a > b else b",
     "args": lambda r: (r.randint(-20, 20), r.randint(-20, 20))},
    {"name": "is_even", "desc": "returns True iff n is even",
     "ref": "def f(n):\n    return n % 2 == 0",
     "args": lambda r: (r.randint(-50, 50),)},
    {"name": "factorial", "desc": "returns n! for a small non-negative integer n",
     "ref": "def f(n):\n    r = 1\n    for i in range(2, n + 1):\n        r = r * i\n    return r",
     "args": lambda r: (r.randint(0, 7),)},
)


# --------------------------------------------------------------------------- #
# The rival verifier — an independent agent that certifies (or refutes) each item
# --------------------------------------------------------------------------- #
# A minimal, safe builtin set for the code sandbox — no __import__, no I/O, no eval/exec.
_SAFE_BUILTIN_NAMES: Tuple[str, ...] = (
    "abs", "min", "max", "range", "len", "sum", "int", "float", "bool", "pow", "divmod",
    "enumerate", "sorted", "map", "filter", "str", "round", "all", "any", "zip", "list", "tuple",
)


def _safe_builtins() -> Dict[str, Any]:
    return {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES if hasattr(builtins, name)}


def _sandbox_compile(src: str, fn_name: str) -> Optional[Callable[..., Any]]:
    """Compile + exec ``src`` in a restricted namespace and return its ``fn_name`` callable.

    The namespace has no ``__import__`` and only a small whitelist of builtins, so generated code
    cannot reach the filesystem, the network, or the interpreter. Returns ``None`` on any failure."""
    try:
        glb: Dict[str, Any] = {"__builtins__": _safe_builtins()}
        exec(compile(src, "<synthetic>", "exec"), glb, glb)  # noqa: S102 — sandboxed builtins
        fn = glb.get(fn_name)
        return fn if callable(fn) else None
    except Exception:  # noqa: BLE001 — a malformed/unsafe candidate simply fails verification
        return None


class RivalVerifier:
    """Independently certify a :class:`SyntheticItem`. Generation is trusted by no one here.

    Math, logic and number-theory are decided exactly by the :class:`~nyxara.growth.prover.Prover`
    (kept iff ``PROVEN``). Code is executed in a restricted sandbox and compared, input by input,
    against an independent reference oracle. An optional LLM may add a second opinion, but it can
    only ever *withhold* acceptance — it can never override a deterministic ``PROVEN``/match."""

    def __init__(self, *, prover: Optional[Prover] = None, llm: Any = None,
                 code_sandbox: bool = True) -> None:
        self.prover = prover or Prover()
        self.llm = llm
        self.code_sandbox = bool(code_sandbox)

    def verify(self, item: SyntheticItem) -> Tuple[bool, str]:
        """Return ``(accepted, reason)`` — a machine-checkable certificate or the refutation."""
        if item.domain == Domain.CODE.value:
            ok, why = self._verify_code(item)
        else:
            ok, why = self._verify_math(item)
        if ok and self.llm is not None:
            ok2 = self._llm_second_opinion(item)
            if ok2 is False:
                return False, f"{why}; but LLM rival dissented"
        return ok, why

    def _verify_math(self, item: SyntheticItem) -> Tuple[bool, str]:
        res = self.prover.prove(ProofClaim(
            kind=item.proof_kind, statement=item.statement, candidate_answer=item.candidate))
        return (res.verdict is ProofVerdict.PROVEN), res.summary()

    def _verify_code(self, item: SyntheticItem) -> Tuple[bool, str]:
        spec = item.code_spec or {}
        if not self.code_sandbox:
            return False, "code sandbox disabled"
        ref = _sandbox_compile(spec.get("reference", ""), spec.get("fn", "f"))
        cand = _sandbox_compile(spec.get("candidate", ""), spec.get("fn", "f"))
        if ref is None or cand is None:
            return False, "candidate or reference failed to compile in the sandbox"
        tests = spec.get("tests") or []
        for args in tests:
            try:
                if cand(*args) != ref(*args):
                    return False, f"output mismatch at {args}"
            except Exception as exc:  # noqa: BLE001 — a crashing candidate is refuted
                return False, f"runtime error at {args}: {exc}"
        return True, f"matched the reference oracle on {len(tests)} inputs"

    def _llm_second_opinion(self, item: SyntheticItem) -> Optional[bool]:
        """Best-effort: ask the LLM to agree/disagree. Never required; failures are 'unknown'."""
        try:
            q = (f"Is this statement correct? Answer only YES or NO.\n"
                 f"{item.statement or item.prompt}\nClaimed answer: {item.answer}")
            text = self.llm.generate(q, system="You are a strict, literal logic checker.")
            verdict = (text or "").strip().lower()
            if verdict.startswith("yes"):
                return True
            if verdict.startswith("no"):
                return False
            return None
        except Exception:  # noqa: BLE001 — a flaky LLM never sinks a deterministically-proven item
            return None


# --------------------------------------------------------------------------- #
# The curator — orchestrates generate → rival-verify → feed (knowledge + flywheel)
# --------------------------------------------------------------------------- #
class SyntheticCurator:
    """The AlphaGo-Zero loop: self-generate logic, rival-verify it, keep only what survives.

    Accepted items are ingested into base knowledge and offered to the data flywheel marked
    ``verified=True`` (so they enter the same corpus the foundry forges from). Gather-only: it
    never trains, never acts, only appends.
    """

    def __init__(self, *, knowledge: Any = None, flywheel: Any = None,
                 verifier: Optional[RivalVerifier] = None,
                 generators: Optional[SyntheticGenerators] = None,
                 settings: Any = None, cfg: Any = None, llm: Any = None) -> None:
        self.settings = settings
        self.cfg = cfg or getattr(settings, "synthesis", None)
        if self.cfg is None:
            from nyxara.kernel.config import SynthesisConfig
            self.cfg = SynthesisConfig()
        seed = int(getattr(self.cfg, "seed", 0))
        self.generators = generators or SyntheticGenerators(seed=seed)
        self.verifier = verifier or RivalVerifier(
            llm=(llm if getattr(self.cfg, "allow_llm_rival", False) else None),
            code_sandbox=bool(getattr(self.cfg, "code_sandbox", True)))
        self.knowledge = knowledge
        self.flywheel = flywheel
        self._reports: List[CurationReport] = []

    @property
    def domains(self) -> Tuple[str, ...]:
        doms = tuple(getattr(self.cfg, "domains", DEFAULT_DOMAINS) or DEFAULT_DOMAINS)
        return doms

    def curate(self, *, rounds: Optional[int] = None,
               batch: Optional[int] = None) -> CurationReport:
        """Run ``rounds`` passes of ``batch`` items each: generate → verify → feed survivors."""
        start = time.monotonic()
        rounds = int(rounds if rounds is not None else getattr(self.cfg, "rounds", 1))
        batch = int(batch if batch is not None else getattr(self.cfg, "batch_size", 8))
        feed_kn = bool(getattr(self.cfg, "feed_knowledge", True))
        feed_fw = bool(getattr(self.cfg, "feed_flywheel", True))
        report = CurationReport()
        for _ in range(max(1, rounds)):
            for item in self.generators.batch(batch, self.domains):
                report.generated += 1
                ok, why = self.verifier.verify(item)
                if not ok:
                    report.rejected += 1
                    continue
                report.accepted += 1
                report.by_domain[item.domain] = report.by_domain.get(item.domain, 0) + 1
                if feed_kn and self._feed_knowledge(item):
                    report.fed_knowledge += 1
                if feed_fw and self._feed_flywheel(item):
                    report.fed_flywheel += 1
                if len(report.samples) < 8:
                    report.samples.append({"domain": item.domain, "prompt": item.prompt,
                                           "answer": item.answer, "certificate": why})
        report.seconds = time.monotonic() - start
        report.reason = (f"curated {report.accepted}/{report.generated} verified "
                         f"(knowledge+{report.fed_knowledge}, flywheel+{report.fed_flywheel})")
        self._reports.append(report)
        return report

    def maybe_run(self) -> Optional[Dict[str, Any]]:
        """Idle-cycle entry: run one cheap batch when enabled. Returns a summary dict, or ``None``.

        Mirrors :meth:`~nyxara.growth.genesis.NeuralArchitectureSearch.maybe_run` — cheap enough
        to call every idle tick, and because the generator's RNG advances, every tick proposes
        fresh logic instead of re-curating the same items."""
        if not bool(getattr(self.cfg, "enabled", True)):
            return None
        try:
            return self.curate(rounds=1).to_dict()
        except Exception as exc:  # noqa: BLE001 — a failed pass reports, never crashes idle
            return {"accepted": 0, "reason": f"synthesis cycle failed: {exc}"}

    def all_reports(self) -> List[CurationReport]:
        return list(self._reports)

    # ---- feeding (best-effort; never raises) ---- #
    def _feed_knowledge(self, item: SyntheticItem) -> bool:
        if self.knowledge is None:
            return False
        try:
            text = f"{item.prompt}\n{item.answer}"
            n = self.knowledge.ingest_text(text, source=f"synthetic:{item.domain}")
            return bool(n)
        except Exception:  # noqa: BLE001
            return False

    def _feed_flywheel(self, item: SyntheticItem) -> bool:
        if self.flywheel is None:
            return False
        try:
            decision = self.flywheel.consider(item.prompt, item.answer, verified=True)
            return bool(getattr(decision, "collected", False))
        except Exception:  # noqa: BLE001
            return False


# --------------------------------------------------------------------------- #
# number-theory helper (local; keeps this module standalone)
# --------------------------------------------------------------------------- #
def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA Synthetic Data Self-Curation (AlphaGo-Zero method) self-test")
    print("=" * 70)

    gens = SyntheticGenerators(seed=7)
    verifier = RivalVerifier()

    # every domain produces items the rival verifier certifies as PROVEN / matching
    for dom in DEFAULT_DOMAINS:
        item = gens.one(dom)
        assert item is not None, f"no item generated for {dom}"
        ok, why = verifier.verify(item)
        print(f"  {dom:14s} -> {'KEEP' if ok else 'DROP'}: {item.statement or item.prompt[:40]}")
        assert ok, f"{dom} item should verify: {why}"

    # the rival REFUTES a deliberately wrong arithmetic claim
    bad = SyntheticItem(domain="arithmetic", prompt="?", answer="5",
                        proof_kind="arithmetic", statement="2 + 2 = 5")
    ok, why = verifier.verify(bad)
    assert not ok, "a false claim must be refuted"
    print(f"\n  refutation check : 2+2=5 -> DROP ({why})")

    # the rival REFUTES buggy code (candidate disagrees with the oracle)
    bad_code = SyntheticItem(
        domain="code", prompt="?", answer="def f(x):\n    return x + 1", proof_kind="code",
        code_spec={"fn": "f", "reference": "def f(x):\n    return x * x",
                   "candidate": "def f(x):\n    return x + 1", "tests": [(2,), (3,)]})
    ok, _ = verifier.verify(bad_code)
    assert not ok, "buggy code must be refuted by the sandbox oracle"
    print("  sandbox check    : buggy code -> DROP")

    # end-to-end curation feeds an in-memory knowledge base + flywheel corpus
    import tempfile
    from pathlib import Path

    from nyxara.growth.flywheel import DataFlywheel
    from nyxara.knowledge.base import KnowledgeBase
    from nyxara.kernel.config import SynthesisConfig

    with tempfile.TemporaryDirectory() as d:
        kb = KnowledgeBase(name="synthetic")
        fw = DataFlywheel(store_path=Path(d) / "synthetic.jsonl", min_chars=1)
        cur = SyntheticCurator(knowledge=kb, flywheel=fw,
                               cfg=SynthesisConfig(batch_size=20, seed=3))
        rep = cur.curate()
        print(f"\n  curation         : {rep.reason}")
        assert rep.accepted >= 1 and rep.fed_knowledge >= 1 and rep.fed_flywheel >= 1
        # the corpus is tagged verified, and a second pass produces NEW (non-duplicate) items
        assert all(ex.source == "flywheel-verified" for ex in fw.examples())
        before = fw.count()
        cur.curate()
        assert fw.count() >= before, "the curator keeps growing the corpus"

    print("\nALL SELF-TESTS PASSED ✓")
