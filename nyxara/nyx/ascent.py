"""NYXARA · nyx/ascent.py — capability she manufactures rather than is given (⛰, NYX V.03).

Intelligence gets into a system three ways: somebody else's trained weights, gradient training of
her own, or rules a human wrote. The third is what "script" means, and it is the one this repo
keeps bumping into. There is a fourth, and it is not a loophole — it is how AlphaZero went from
random weights to superhuman with no human games and no hand-written strategy, and how FunSearch
found mathematics nobody had:

    capability = search × verifier × time

The rules of chess are about fifty lines. The intelligence was never in those lines; it was in the
search they defined and in the play that survived them. That is exactly the shape here:

* I write the **verifier** — the thing that can tell a right answer from a wrong one.
* **She** runs the **search** that finds candidates.
* What survives verification is **kept**, and what is kept makes the next search cheaper.

So the artifacts in :class:`SolutionLibrary` are not authored by me and are not weights. They are
**proven objects with their certificates attached**, and the library *is* the accumulated
capability. :class:`ExpressionProposer` seeds its terminals from what has already been proven, so
a problem whose answer contains a previously discovered sub-expression is reached in fewer steps —
compounding that is measured in proposals-to-solve, not asserted.

The honest boundary is the whole design, and it is enforced rather than described:

* **No verifier, no ascent.** :meth:`Ascent.ascend` on a domain with nothing to check against
  **refuses** and says so. It never scores a candidate it cannot check, because a number produced
  by a missing verifier is worse than no answer.
* **Three verdicts, not two.** ``PROVEN`` / ``REFUTED`` / ``UNKNOWN``. A candidate that crashes,
  overflows or times out is *unknown*, never quietly refuted — the difference between "this is
  wrong" and "I could not tell" is the difference this file exists to keep.
* **It works where truth is checkable.** Arithmetic, synthesis against examples, code against
  tests: yes. Whether a sentence is well phrased: no, because there is no cheap verifier for it,
  and pretending otherwise would manufacture confidence instead of capability.
* **Nothing is executed loosely.** The expression domain is evaluated by a whitelisting AST walker
  in-process — there is no ``eval`` here. Arbitrary code goes to
  :func:`~nyxara.agency.code_sandbox.run_python`, in its own subprocess, as it already did.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

__all__ = ["Verdict", "Problem", "Candidate", "Judgement", "Artifact", "Verifier",
           "VerifierRegistry", "Proposer", "SolutionLibrary", "ProverVerifier",
           "ExpressionVerifier", "CodeVerifier", "ExpressionProposer", "AscentRun", "Ascent",
           "eval_expression"]


class Verdict(str, Enum):
    """The three honest outcomes. ``UNKNOWN`` is not a failure mode — it is a result."""

    PROVEN = "proven"
    REFUTED = "refuted"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class Problem:
    """Something to solve, stated so a verifier can actually check an answer.

    ``examples`` are ``(inputs, output)`` pairs. ``holdout`` are examples the *proposer never
    sees* — without them a synthesiser can fit the training pairs exactly and have learned
    nothing, and the library would fill up with overfitted junk carrying real certificates.
    """

    name: str
    domain: str = "expression"
    statement: str = ""
    variables: Tuple[str, ...] = ("x",)
    examples: Tuple[Tuple[Tuple[int, ...], int], ...] = ()
    holdout: Tuple[Tuple[Tuple[int, ...], int], ...] = ()
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "domain": self.domain, "statement": self.statement,
                "variables": list(self.variables),
                "examples": [[list(i), o] for i, o in self.examples],
                "holdout": [[list(i), o] for i, o in self.holdout], "meta": dict(self.meta)}


@dataclass
class Candidate:
    """One thing search came up with. Nothing here is trusted until a verifier says so."""

    domain: str
    statement: str
    payload: Any = None
    origin: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"domain": self.domain, "statement": self.statement, "origin": self.origin}


@dataclass
class Judgement:
    """A verifier's verdict, with the witness that backs it."""

    verdict: Verdict
    evidence: str = ""
    method: str = ""
    confidence: float = 1.0

    @property
    def proven(self) -> bool:
        return self.verdict is Verdict.PROVEN

    def to_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict.value, "evidence": self.evidence, "method": self.method,
                "confidence": round(float(self.confidence), 4)}


@dataclass
class Artifact:
    """A candidate that survived verification — capability, with its certificate attached."""

    domain: str
    statement: str
    evidence: str
    method: str
    problem: str = ""
    origin: str = ""
    found_at: float = field(default_factory=time.time)
    proposals_to_find: int = 0

    @property
    def uid(self) -> str:
        """Content address — the same proven object is the same artifact, found twice or not."""
        return hashlib.sha256(f"{self.domain}\x00{self.statement}".encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {"uid": self.uid, "domain": self.domain, "statement": self.statement,
                "evidence": self.evidence, "method": self.method, "problem": self.problem,
                "origin": self.origin, "found_at": self.found_at,
                "proposals_to_find": self.proposals_to_find}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Artifact":
        return cls(domain=d["domain"], statement=d["statement"], evidence=d.get("evidence", ""),
                   method=d.get("method", ""), problem=d.get("problem", ""),
                   origin=d.get("origin", ""), found_at=float(d.get("found_at") or 0.0),
                   proposals_to_find=int(d.get("proposals_to_find") or 0))


# --------------------------------------------------------------------------- #
# Verifiers
# --------------------------------------------------------------------------- #
class Verifier:
    """Base contract: given a candidate, say proven, refuted, or *I could not tell*."""

    domain = "none"
    method = "base"

    def available(self) -> bool:
        return True

    def verify(self, candidate: Candidate, problem: Problem) -> Judgement:  # pragma: no cover
        raise NotImplementedError

    def safe_verify(self, candidate: Candidate, problem: Problem) -> Judgement:
        """A verifier that crashes has not refuted anything — it has failed to decide."""
        try:
            if not self.available():
                return Judgement(Verdict.UNKNOWN, "verifier unavailable", self.method, 0.0)
            return self.verify(candidate, problem)
        except Exception as exc:  # noqa: BLE001
            return Judgement(Verdict.UNKNOWN, f"verifier error: {exc}", self.method, 0.0)


class VerifierRegistry:
    """Which domains can be checked at all. The empty answer is the important one."""

    def __init__(self, verifiers: Optional[Sequence[Verifier]] = None) -> None:
        self._by_domain: Dict[str, Verifier] = {}
        for verifier in verifiers or ():
            self.register(verifier)

    def register(self, verifier: Verifier) -> None:
        self._by_domain[verifier.domain] = verifier

    def for_domain(self, domain: str) -> Optional[Verifier]:
        return self._by_domain.get(domain)

    def domains(self) -> List[str]:
        """Only the domains whose verifier is actually usable right now."""
        return sorted(d for d, v in self._by_domain.items() if v.available())

    def stats(self) -> Dict[str, Any]:
        return {"registered": sorted(self._by_domain), "available": self.domains()}


# --------------------------------------------------------------------------- #
# Safe expression evaluation — a whitelist, never `eval`
# --------------------------------------------------------------------------- #
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Pow)
_MAX_ABS = 10 ** 12          # keeps a runaway ** from turning search into a heat source
_MAX_POW_EXP = 6


class _EvalError(Exception):
    """Could not evaluate — which makes the candidate unknown, never wrong."""


def eval_expression(expr: str, env: Dict[str, int]) -> int:
    """Evaluate a small integer expression by walking a whitelisted AST.

    Nothing is compiled or ``eval``-ed: attribute access, calls, comprehensions, names outside
    ``env`` and every other node type simply have no branch here, so they raise :class:`_EvalError`
    rather than running. Overflow and division by zero are errors too — an answer the checker
    could not compute is not an answer it disproved.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise _EvalError(f"unparsable: {exc}") from exc
    return _eval_node(tree.body, env)


def _eval_node(node: ast.AST, env: Dict[str, int]) -> int:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise _EvalError("only integer constants")
        return int(node.value)
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise _EvalError(f"unbound name {node.id!r}")
        return int(env[node.id])
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand, env)
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        left = _eval_node(node.left, env)
        right = _eval_node(node.right, env)
        return _apply(node.op, left, right)
    raise _EvalError(f"disallowed node {type(node).__name__}")


def _apply(op: ast.operator, left: int, right: int) -> int:
    if isinstance(op, ast.Add):
        out = left + right
    elif isinstance(op, ast.Sub):
        out = left - right
    elif isinstance(op, ast.Mult):
        out = left * right
    elif isinstance(op, ast.FloorDiv):
        if right == 0:
            raise _EvalError("division by zero")
        out = left // right
    elif isinstance(op, ast.Mod):
        if right == 0:
            raise _EvalError("modulo by zero")
        out = left % right
    elif isinstance(op, ast.Pow):
        if right < 0 or right > _MAX_POW_EXP or abs(left) > 1000:
            raise _EvalError("exponent out of range")
        out = left ** right
    else:  # pragma: no cover — guarded by _ALLOWED_BINOPS
        raise _EvalError("disallowed operator")
    if abs(out) > _MAX_ABS:
        raise _EvalError("overflow")
    return out


def _free_names(expr: str) -> set:
    """The variable names an expression reads. Raises when it is not a legal expression at all."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise _EvalError(str(exc)) from exc
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


# --------------------------------------------------------------------------- #
# Concrete verifiers
# --------------------------------------------------------------------------- #
class ExpressionVerifier(Verifier):
    """Checks a synthesised expression against every example — including the held-out ones.

    Held-out examples are what make a certificate mean something. Matching only the pairs the
    search could see is fitting, not solving, and a library full of overfitted expressions
    carrying real certificates would be worse than an empty one.
    """

    domain = "expression"
    method = "example-check"

    def verify(self, candidate: Candidate, problem: Problem) -> Judgement:
        checks = list(problem.examples) + list(problem.holdout)
        if not checks:
            return Judgement(Verdict.UNKNOWN, "no examples to check against", self.method, 0.0)
        for inputs, expected in checks:
            env = dict(zip(problem.variables, inputs))
            try:
                got = eval_expression(candidate.statement, env)
            except _EvalError as exc:
                return Judgement(Verdict.UNKNOWN, f"could not evaluate: {exc}", self.method, 0.0)
            if got != expected:
                return Judgement(
                    Verdict.REFUTED,
                    f"{candidate.statement} gives {got} at {dict(env)}, expected {expected}",
                    self.method, 1.0)
        return Judgement(
            Verdict.PROVEN,
            f"matches all {len(checks)} examples ({len(problem.holdout)} held out): "
            + "; ".join(f"{dict(zip(problem.variables, i))}→{o}" for i, o in checks[:4]),
            self.method, 1.0)


class ProverVerifier(Verifier):
    """Decidable mathematical claims, delegated wholesale to :class:`~nyxara.growth.prover.Prover`.

    Its three-way verdict is already exactly the distinction this module needs, so there is no
    second decision procedure here — only the mapping.
    """

    domain = "math"
    method = "prover"

    def __init__(self, prover: Any = None) -> None:
        self._prover = prover

    def _get(self) -> Any:
        if self._prover is None:
            from nyxara.growth.prover import Prover
            self._prover = Prover()
        return self._prover

    def available(self) -> bool:
        try:
            self._get()
            return True
        except Exception:  # noqa: BLE001
            return False

    def verify(self, candidate: Candidate, problem: Problem) -> Judgement:
        from nyxara.growth.prover import ProofVerdict
        result = self._get().certify(candidate.statement)
        mapping = {ProofVerdict.PROVEN: Verdict.PROVEN,
                   ProofVerdict.REFUTED: Verdict.REFUTED,
                   ProofVerdict.UNPROVABLE: Verdict.UNKNOWN}
        return Judgement(mapping.get(result.verdict, Verdict.UNKNOWN),
                         result.certificate or result.summary(),
                         f"prover:{result.method}", float(result.confidence))


class CodeVerifier(Verifier):
    """Real code, checked by running it and its tests in the existing sandbox subprocess."""

    domain = "code"
    method = "sandbox"

    def __init__(self, *, timeout_s: float = 5.0) -> None:
        self.timeout_s = float(timeout_s)

    def available(self) -> bool:
        try:
            from nyxara.agency.code_sandbox import run_python  # noqa: F401
            return True
        except Exception:  # noqa: BLE001
            return False

    def verify(self, candidate: Candidate, problem: Problem) -> Judgement:
        from nyxara.agency.code_sandbox import run_python
        harness = str(problem.meta.get("harness") or "")
        if not harness:
            return Judgement(Verdict.UNKNOWN, "no test harness supplied", self.method, 0.0)
        result = run_python(f"{candidate.statement}\n\n{harness}\n", timeout_s=self.timeout_s)
        if result.timed_out:
            return Judgement(Verdict.UNKNOWN, "timed out", self.method, 0.0)
        if result.ok:
            return Judgement(Verdict.PROVEN, (result.stdout or "tests passed").strip()[:400],
                             self.method, 1.0)
        # A failing assertion is a refutation; anything else is a failure to decide.
        blob = f"{result.stderr}\n{result.error or ''}"
        if "AssertionError" in blob:
            return Judgement(Verdict.REFUTED, blob.strip()[:400], self.method, 1.0)
        return Judgement(Verdict.UNKNOWN, blob.strip()[:400] or "did not run", self.method, 0.0)


# --------------------------------------------------------------------------- #
# Proposers — the search half
# --------------------------------------------------------------------------- #
class Proposer:
    """Base contract: stream candidates for a problem, cheapest first."""

    domain = "none"
    name = "base"

    def propose(self, problem: Problem, library: "SolutionLibrary", *,
                budget: int) -> Iterator[Candidate]:  # pragma: no cover
        raise NotImplementedError


class ExpressionProposer(Proposer):
    """Bottom-up enumerative synthesis with observational-equivalence pruning.

    Expressions are built in order of increasing size from a terminal set, and any expression that
    computes the *same values on the training inputs* as one already seen is discarded — a
    standard and very large saving, since most syntactic variety is semantic repetition.

    The terminal set is seeded from :class:`SolutionLibrary`. That is where compounding actually
    lives: once ``x * x + 1`` has been proven for some earlier problem, it becomes a single
    building block, and any later answer containing it is reached in far fewer proposals. The
    library makes the search cheaper, which is the whole claim.
    """

    domain = "expression"
    name = "enumerative"

    def __init__(self, *, constants: Sequence[int] = (0, 1, 2, 3),
                 operators: Sequence[str] = ("+", "-", "*"),
                 max_size: int = 4, use_library: bool = True) -> None:
        self.constants = tuple(constants)
        self.operators = tuple(operators)
        self.max_size = max(1, int(max_size))
        self.use_library = bool(use_library)

    def _seeds(self, problem: Problem, library: "SolutionLibrary") -> List[str]:
        if not self.use_library:
            return []
        out: List[str] = []
        for art in library.by_domain(self.domain):
            # Only a seed whose free variables this problem actually has can be evaluated here.
            try:
                names = _free_names(art.statement)
            except _EvalError:
                continue
            if names and names <= set(problem.variables):
                out.append(art.statement)
        return out

    def propose(self, problem: Problem, library: "SolutionLibrary", *,
                budget: int) -> Iterator[Candidate]:
        inputs = [dict(zip(problem.variables, i)) for i, _ in problem.examples]
        seen_behaviour: Dict[Tuple[int, ...], str] = {}
        emitted = 0

        def behaviour(expr: str) -> Optional[Tuple[int, ...]]:
            try:
                return tuple(eval_expression(expr, env) for env in inputs)
            except _EvalError:
                return None

        # Level 1 — terminals: the variables, small constants, and whatever is already proven.
        level: List[str] = [*problem.variables, *(str(c) for c in self.constants),
                            *self._seeds(problem, library)]
        levels: List[List[str]] = [[]]
        current: List[str] = []
        for expr in level:
            sig = behaviour(expr)
            if sig is None or sig in seen_behaviour:
                continue
            seen_behaviour[sig] = expr
            current.append(expr)
            emitted += 1
            yield Candidate(domain=self.domain, statement=expr, origin=self.name)
            if emitted >= budget:
                return
        levels.append(current)

        # Levels 2..max_size — every combination of two smaller expressions.
        for size in range(2, self.max_size + 1):
            current = []
            for left_size in range(1, size):
                right_size = size - left_size
                if right_size >= len(levels) or left_size >= len(levels):
                    continue
                for left in levels[left_size]:
                    for right in levels[right_size]:
                        for op in self.operators:
                            expr = f"({left} {op} {right})"
                            sig = behaviour(expr)
                            if sig is None or sig in seen_behaviour:
                                continue
                            seen_behaviour[sig] = expr
                            current.append(expr)
                            emitted += 1
                            yield Candidate(domain=self.domain, statement=expr,
                                            origin=self.name)
                            if emitted >= budget:
                                return
            levels.append(current)
            if not current:
                return          # nothing new at this size; larger sizes cannot help


# --------------------------------------------------------------------------- #
# The library — where capability accumulates
# --------------------------------------------------------------------------- #
class SolutionLibrary:
    """Proven artifacts, content-addressed and monotonic.

    Monotonic on purpose: something proven does not stop being proven because a later run did not
    re-derive it. Nothing here can remove an artifact except an explicit :meth:`forget`, which
    exists for tests and for retracting a *verifier* that turned out to be wrong — never as part
    of the normal loop.
    """

    def __init__(self) -> None:
        self._by_uid: Dict[str, Artifact] = {}

    def add(self, artifact: Artifact) -> bool:
        """Returns False when this exact proven object was already known."""
        if artifact.uid in self._by_uid:
            return False
        self._by_uid[artifact.uid] = artifact
        return True

    def get(self, uid: str) -> Optional[Artifact]:
        return self._by_uid.get(uid)

    def forget(self, uid: str) -> bool:
        return self._by_uid.pop(uid, None) is not None

    def by_domain(self, domain: str) -> List[Artifact]:
        return sorted((a for a in self._by_uid.values() if a.domain == domain),
                      key=lambda a: a.found_at)

    def all(self) -> List[Artifact]:
        return sorted(self._by_uid.values(), key=lambda a: a.found_at)

    def search(self, text: str, *, limit: int = 20) -> List[Artifact]:
        needle = (text or "").strip().lower()
        if not needle:
            return []
        return [a for a in self.all()
                if needle in a.statement.lower() or needle in a.problem.lower()][:limit]

    def __len__(self) -> int:
        return len(self._by_uid)

    def to_dict(self) -> Dict[str, Any]:
        return {"version": 1, "artifacts": [a.to_dict() for a in self.all()]}

    def load_dict(self, data: Dict[str, Any]) -> bool:
        if not isinstance(data, dict) or data.get("version") != 1:
            return False
        try:
            for raw in data.get("artifacts") or ():
                self.add(Artifact.from_dict(raw))
        except (KeyError, TypeError, ValueError):
            return False
        return True

    def save(self, path: Any) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        os.replace(tmp, target)
        return target

    def load(self, path: Any) -> bool:
        try:
            return self.load_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return False

    def stats(self) -> Dict[str, Any]:
        domains: Dict[str, int] = {}
        for art in self._by_uid.values():
            domains[art.domain] = domains.get(art.domain, 0) + 1
        return {"artifacts": len(self._by_uid), "domains": domains}


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
@dataclass
class AscentRun:
    """One attempt on one problem — including the honest refusal."""

    problem: str = ""
    domain: str = ""
    proposed: int = 0
    refuted: int = 0
    unknown: int = 0
    solved: bool = False
    artifact: Optional[Artifact] = None
    refused: bool = False
    reason: str = ""
    seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"problem": self.problem, "domain": self.domain, "proposed": self.proposed,
                "refuted": self.refuted, "unknown": self.unknown, "solved": self.solved,
                "artifact": self.artifact.to_dict() if self.artifact else None,
                "refused": self.refused, "reason": self.reason,
                "seconds": round(self.seconds, 4)}


class Ascent:
    """Propose, verify, keep what survives — and refuse where nothing can be checked."""

    def __init__(self, *, verifiers: Optional[Sequence[Verifier]] = None,
                 proposers: Optional[Sequence[Proposer]] = None,
                 library: Optional[SolutionLibrary] = None,
                 default_budget: int = 4000) -> None:
        self.registry = VerifierRegistry(
            verifiers if verifiers is not None else
            [ExpressionVerifier(), ProverVerifier(), CodeVerifier()])
        self.proposers: List[Proposer] = list(
            proposers if proposers is not None else [ExpressionProposer()])
        self.library = library if library is not None else SolutionLibrary()
        self.default_budget = int(default_budget)
        self.runs: List[AscentRun] = []

    def _proposer_for(self, domain: str) -> Optional[Proposer]:
        for proposer in self.proposers:
            if proposer.domain == domain:
                return proposer
        return None

    def ascend(self, problem: Problem, *, budget: Optional[int] = None) -> AscentRun:
        """One bounded search. Never raises; a domain it cannot check is refused, not faked."""
        started = time.time()
        run = AscentRun(problem=problem.name, domain=problem.domain)
        budget = int(budget if budget is not None else self.default_budget)

        verifier = self.registry.for_domain(problem.domain)
        if verifier is None or not verifier.available():
            run.refused = True
            run.reason = (f"no verifier for domain {problem.domain!r}; "
                          f"checkable domains are {self.registry.domains()}")
            run.seconds = time.time() - started
            self.runs.append(run)
            return run

        proposer = self._proposer_for(problem.domain)
        if proposer is None:
            run.refused = True
            run.reason = f"no proposer for domain {problem.domain!r}"
            run.seconds = time.time() - started
            self.runs.append(run)
            return run

        try:
            for candidate in proposer.propose(problem, self.library, budget=budget):
                run.proposed += 1
                judgement = verifier.safe_verify(candidate, problem)
                if judgement.verdict is Verdict.PROVEN:
                    artifact = Artifact(
                        domain=problem.domain, statement=candidate.statement,
                        evidence=judgement.evidence, method=judgement.method,
                        problem=problem.name, origin=candidate.origin,
                        proposals_to_find=run.proposed)
                    self.library.add(artifact)
                    run.solved, run.artifact = True, artifact
                    break
                if judgement.verdict is Verdict.REFUTED:
                    run.refuted += 1
                else:
                    run.unknown += 1
                if run.proposed >= budget:
                    break
        except Exception as exc:  # noqa: BLE001 — a broken search is a failed run, not a crash
            run.reason = f"search error: {exc}"

        if not run.solved and not run.reason:
            run.reason = f"exhausted {run.proposed} candidate(s) without a proof"
        run.seconds = time.time() - started
        self.runs.append(run)
        return run

    def solved_rate(self, problems: Sequence[Problem], *,
                    budget: Optional[int] = None) -> Dict[str, Any]:
        """Fraction of a held-out problem set she can actually solve.

        This — not the size of the library — is what "she got better" has to mean. A library can
        grow while capability does not.
        """
        runs = [self.ascend(p, budget=budget) for p in problems]
        solved = [r for r in runs if r.solved]
        return {"problems": len(runs), "solved": len(solved),
                "rate": round(len(solved) / len(runs), 4) if runs else 0.0,
                "mean_proposals": round(sum(r.proposed for r in solved) / len(solved), 2)
                if solved else 0.0}

    def stats(self) -> Dict[str, Any]:
        solved = [r for r in self.runs if r.solved]
        return {
            "runs": len(self.runs),
            "solved": len(solved),
            "refused": sum(1 for r in self.runs if r.refused),
            "library": self.library.stats(),
            "verifiers": self.registry.stats(),
            "proposers": [p.name for p in self.proposers],
            "note": ("capability = search × verifier × time. It manufactures only what a verifier "
                     "can check — arithmetic, synthesis against held-out examples, code against "
                     "tests. A domain with no verifier is refused, never scored, and a candidate "
                     "that crashes is unknown rather than refuted"),
        }

    def summary(self) -> str:
        st = self.stats()
        lines = ["=" * 70, "NYXARA ascent — capability she manufactured", "=" * 70,
                 f"checkable  : {', '.join(st['verifiers']['available']) or 'nothing'}",
                 f"runs       : {st['runs']} ({st['solved']} solved, {st['refused']} refused)",
                 f"library    : {st['library']['artifacts']} artifact(s) {st['library']['domains']}"]
        for art in self.library.all()[:8]:
            lines.append(f"  ✓ [{art.domain}] {art.statement}   ({art.method}, "
                         f"found in {art.proposals_to_find} proposals)")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA ascent self-test")
    print("=" * 70)

    # A problem stated only as examples. Nothing anywhere says what the answer is.
    square_plus_one = Problem(
        name="square-plus-one", variables=("x",),
        examples=(((1,), 2), ((2,), 5), ((3,), 10)),
        holdout=(((7,), 50), ((11,), 122)))

    asc = Ascent()
    run = asc.ascend(square_plus_one)
    print(f"\nfound          : {run.artifact.statement if run.artifact else None}")
    print(f"proposals      : {run.proposed}   refuted {run.refuted}   unknown {run.unknown}")
    assert run.solved, run.to_dict()
    # The point: this expression appears nowhere in this file above the self-test. Search found
    # it; the verifier checked it against examples the search never saw.
    assert run.artifact.statement not in open(__file__, encoding="utf-8").read().split(
        "Self-test")[0], "the artifact must not be something the author wrote down"
    assert "held out" in run.artifact.evidence

    # Compounding: a later problem containing the same sub-expression is reached faster.
    harder = Problem(
        name="twice-square-plus-one", variables=("x",),
        examples=(((1,), 4), ((2,), 10), ((3,), 20)),
        holdout=(((5,), 52),))

    cold = Ascent()
    cold_run = cold.ascend(harder)
    warm = Ascent(library=asc.library)
    warm_run = warm.ascend(harder)
    print(f"\ncold proposals : {cold_run.proposed} → "
          f"{cold_run.artifact.statement if cold_run.artifact else None}")
    print(f"warm proposals : {warm_run.proposed} → "
          f"{warm_run.artifact.statement if warm_run.artifact else None}")
    assert cold_run.solved and warm_run.solved
    assert warm_run.proposed < cold_run.proposed, "a warm library must make the search cheaper"

    # Refusal: a domain with no verifier is refused, not scored.
    refused = asc.ascend(Problem(name="is-this-beautiful", domain="aesthetics"))
    print(f"\nrefusal        : {refused.reason}")
    assert refused.refused and not refused.solved
    assert "no verifier" in refused.reason

    # Three verdicts: a wrong expression is refuted, an uncomputable one is unknown.
    verifier = ExpressionVerifier()
    assert verifier.verify(Candidate("expression", "(x + 1)"),
                           square_plus_one).verdict is Verdict.REFUTED
    assert verifier.verify(Candidate("expression", "(x // 0)"),
                           square_plus_one).verdict is Verdict.UNKNOWN
    assert verifier.verify(Candidate("expression", "__import__('os').listdir()"),
                           square_plus_one).verdict is Verdict.UNKNOWN
    print("verdicts       : proven / refuted / unknown all reachable")

    # The math domain, via the existing prover.
    prover_v = Ascent(proposers=[]).registry.for_domain("math")
    if prover_v is not None and prover_v.available():
        good = prover_v.safe_verify(Candidate("math", "2 + 2 = 4"), Problem(name="p", domain="math"))
        bad = prover_v.safe_verify(Candidate("math", "2 + 2 = 5"), Problem(name="p", domain="math"))
        print(f"prover         : '2+2=4' → {good.verdict.value}; '2+2=5' → {bad.verdict.value}")
        assert good.verdict is Verdict.PROVEN and bad.verdict is Verdict.REFUTED

    # Library is monotonic and round-trips.
    lib = SolutionLibrary()
    art = Artifact(domain="expression", statement="(x * x)", evidence="e", method="m")
    assert lib.add(art) is True and lib.add(art) is False, "content-addressed, added once"
    restored = SolutionLibrary()
    assert restored.load_dict(lib.to_dict()) and len(restored) == 1

    print("\n" + asc.summary())
    print("\nALL SELF-TESTS PASSED ✓")
