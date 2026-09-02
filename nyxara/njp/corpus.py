"""NYXARA · njp/corpus.py — an examination she did not write (📖, NJP V.25).

Every exam in this package up to V.24 shares one weakness, and it is worth stating before any
number is quoted: **the same hand wrote the code and the questions.** ``mathschool`` grades the
mathematician against problems written while the mathematician was being built; ``school`` grades
the reasoner against subjects chosen to suit it. Four hand-written banks and thirty generated
papers later, 927/927 is a real number about a real organ — and it is still a mark awarded by the
author of the work.

This module is the other kind. :data:`CORPUS` is an **outside corpus**, 10,870 records built by a
generator nobody here wrote, in a schema nobody here chose, with a *sealed* held-out split and a
machine checker attached to every item. Its splits are constructed so that memorising the training
half scores near chance on the examined half — TRAIN sees modulus ∈ {7, 10, 12}, EVAL sees
{13, 17, 23} with squaring added; TRAIN sees sequence and string coding, EVAL sees dict and matrix;
TRAIN sees the ``succ_last`` and ``reverse`` analogies, EVAL sees ``succ_first`` and ``double_all``.

**The grader is the corpus's own, reimplemented clause for clause rather than reinterpreted.**
That distinction is the whole reason this file can be trusted: a verifier written to suit the
answers NJP happens to give is not a verifier, it is a scoreboard. :func:`verify` is a faithful
port of the corpus's ``score_candidate`` — the same normalisation, the same first-number
extraction, the same JSON recovery, the same tolerance defaults — and
:func:`self_test` runs the corpus's own check that *every reference answer passes its own
verifier*. If that number is not 100%, the port is wrong and every score below is void. It is
asserted in ``tests/njp/test_corpus.py`` rather than believed.

**Three outcomes, not two**, exactly as :class:`nyxara.njp.school.Score` counts them. An empty
reply is an **abstention**, never an error: the corpus's adversarial split contains items whose
honest answer is a refusal, and a grader that scores silence as wrong cannot tell the brain that
knows it is beaten from the brain that guesses.

**One verifier type is not machine-checkable and is not pretended to be.** ``rubric`` items — 14
in the whole corpus, 1 in EVAL — need a judge. They are counted, reported under their own heading
and excluded from every accuracy, because folding them in either way would be a claim the code
cannot support.
"""

from __future__ import annotations

import gzip
import json
import os
import re
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "Record", "Verdict", "verify", "load", "corpus", "splits", "faculties",
    "generators", "self_test", "CORPUS_PATH", "SPLITS", "VERIFIER_TYPES",
    # Aliases for the package namespace, where `Record` and `verify` are far too general to be
    # exported bare. Same objects, unambiguous names.
    "CorpusRecord", "CorpusVerdict", "verify_corpus_answer",
]

#: Where the vendored corpus lives. Overridable with ``NYXARA_COGNITIVE_CORPUS`` so a larger
#: build can be pointed at without touching the package.
CORPUS_PATH = os.environ.get(
    "NYXARA_COGNITIVE_CORPUS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cognitive_corpus.jsonl.gz"),
)

#: In the order the corpus's own README ranks them: fitting, then instance generalisation, then
#: parameter-region transfer, then structural transfer, then robustness, then the sealed number.
SPLITS: Tuple[str, ...] = (
    "TRAIN", "PRACTICE", "GENERALIZATION", "NOVEL", "ADVERSARIAL", "EVAL",
)

VERIFIER_TYPES: Tuple[str, ...] = (
    "exact", "numeric", "set", "regex", "structured", "python_test", "rubric",
)


# --------------------------------------------------------------------------- #
# what one record is
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Record:
    """One corpus item, with the fields that make it *trainable* rather than merely answerable.

    ``expected_behavior`` is the process to reward, ``failure_modes`` the specific wrong paths,
    and ``provenance`` the generator and family parameters — which is what makes the split policy
    auditable instead of asserted. None of them are used for grading; they are used for teaching
    and for saying *why* a miss happened.
    """

    id: str = ""
    faculty: str = ""
    split: str = ""
    difficulty: int = 0
    task: str = ""
    context: str = ""
    answer: str = ""
    expected_behavior: str = ""
    reasoning: str = ""
    skills: Tuple[str, ...] = ()
    failure_modes: Tuple[str, ...] = ()
    novel_variant: str = ""
    verification: Dict[str, Any] = dc_field(default_factory=dict)
    provenance: Dict[str, Any] = dc_field(default_factory=dict)

    @property
    def generator(self) -> str:
        return str(self.provenance.get("generator", ""))

    @property
    def attack(self) -> str:
        """The adversarial perturbation applied, or ``""`` for an unattacked item."""
        return str(self.provenance.get("attack") or "")

    @property
    def family(self) -> Dict[str, Any]:
        fam = self.provenance.get("family")
        return dict(fam) if isinstance(fam, dict) else {}

    @property
    def verifier(self) -> str:
        return str(self.verification.get("type", ""))

    @property
    def gradable(self) -> bool:
        """False only for ``rubric``, which needs a judge this package does not have."""
        return self.verifier != "rubric"

    @property
    def prompt(self) -> str:
        """The item as it is actually put to her: the task, then the context beneath it.

        Order matters and is the corpus's own — the instruction ("give only the integer") comes
        first, so a reader that only ever sees the first line still sees what is being asked for.
        """
        body = self.context.strip()
        return f"{self.task.strip()}\n\n{body}" if body else self.task.strip()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Record":
        return cls(
            id=str(d.get("id", "")),
            faculty=str(d.get("faculty", "")),
            split=str(d.get("split", "")),
            difficulty=int(d.get("difficulty", 0) or 0),
            task=str(d.get("task", "")),
            context=str(d.get("context", "") or ""),
            answer=str(d.get("answer", "")),
            expected_behavior=str(d.get("expected_behavior", "") or ""),
            reasoning=str(d.get("reasoning", "") or ""),
            skills=tuple(d.get("skills") or ()),
            failure_modes=tuple(d.get("failure_modes") or ()),
            novel_variant=str(d.get("novel_variant", "") or ""),
            verification=dict(d.get("verification") or {}),
            provenance=dict(d.get("provenance") or {}),
        )


@dataclass(frozen=True)
class Verdict:
    """``right`` | ``wrong`` | ``abstain`` | ``unverifiable``, and the checker's own reason."""

    outcome: str = "abstain"
    why: str = ""

    @property
    def right(self) -> bool:
        return self.outcome == "right"


# --------------------------------------------------------------------------- #
# the grader — a faithful port of the corpus's own score_candidate
# --------------------------------------------------------------------------- #

def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower().rstrip(".")


def _first_number(s: Any) -> Optional[float]:
    match = re.search(r"-?\d+(?:\.\d+)?", str(s).replace(",", ""))
    return float(match.group()) if match else None


def _extract_json(s: Any) -> Any:
    text = str(s).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    for start, end in (("{", "}"), ("[", "]")):
        i, j = text.find(start), text.rfind(end)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    return None


def run_python_test(code: str, tests: str, entrypoint: str, timeout: int = 5) -> Tuple[bool, str]:
    """Execute a candidate function and run the corpus's asserts against it.

    The corpus's own runner uses ``signal.alarm``, which only exists on the main thread of a
    POSIX process. Here the alarm is *attempted* and its absence is not fatal — a test suite
    running under pytest-xdist, or anything on Windows, would otherwise fail every coding item
    for a reason that has nothing to do with the code being graded.
    """
    import signal as _signal

    namespace: Dict[str, Any] = {}
    armed = False
    previous = None
    try:
        previous = _signal.signal(_signal.SIGALRM, _raise_timeout)
        _signal.alarm(int(timeout))
        armed = True
    except (ValueError, AttributeError, OSError):  # not the main thread, or not POSIX
        armed = False
    try:
        exec(compile(str(code), "<candidate>", "exec"), namespace)  # noqa: S102
        if entrypoint not in namespace:
            return False, f"entrypoint {entrypoint!r} not defined"
        exec(compile(str(tests), "<tests>", "exec"), namespace)  # noqa: S102
        return True, "ok"
    except _Timeout:
        return False, "timeout"
    except Exception as exc:  # noqa: BLE001 — any failure is a failed candidate
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        if armed:
            try:
                _signal.alarm(0)
                if previous is not None:
                    _signal.signal(_signal.SIGALRM, previous)
            except (ValueError, AttributeError, OSError):
                pass


class _Timeout(Exception):
    pass


def _raise_timeout(_signum: Any, _frame: Any) -> None:
    raise _Timeout()


def verify(record: Record, reply: Any) -> Verdict:
    """Grade one reply. The corpus's rules, plus this package's three-outcome rule.

    Silence is an **abstention** and is decided before any checker runs — an empty string passes
    no regex and contains no number, so a grader that skipped this step would score every
    refusal as an error and quietly reward guessing.
    """
    said = str(reply or "").strip()
    spec = dict(record.verification or {})
    kind = str(spec.get("type", ""))

    if kind == "rubric":
        return Verdict("unverifiable", "rubric: needs a judge, not auto-scorable")
    if not said:
        return Verdict("abstain", "no answer given")

    ok, why = _check(kind, spec, said)
    return Verdict("right" if ok else "wrong", why)


def _check(kind: str, spec: Dict[str, Any], said: str) -> Tuple[bool, str]:
    if kind == "exact":
        return _norm(said) == _norm(spec.get("value")), "exact"

    if kind == "numeric":
        got = _first_number(said)
        if got is None:
            return False, "no number in output"
        return abs(got - float(spec["value"])) <= float(spec.get("tol", 0.0) or 0.0), "numeric"

    if kind == "set":
        want = {_norm(x) for x in spec.get("values", ())}
        got = {_norm(x) for x in re.split(r"[,\n]", said) if x.strip()}
        return want == got, "set"

    if kind == "regex":
        return bool(re.search(str(spec["pattern"]), said, re.I | re.M)), "regex"

    if kind == "python_test":
        code = said
        fenced = re.search(r"```(?:python)?\n(.*?)```", code, re.S)
        if fenced:
            code = fenced.group(1)
        return run_python_test(code, str(spec.get("tests", "")), str(spec.get("entrypoint", "")))

    if kind == "structured":
        return _check_structured(spec, said)

    return False, f"unknown verifier {kind!r}"


def _check_structured(spec: Dict[str, Any], said: str) -> Tuple[bool, str]:
    obj = _extract_json(said)
    if not isinstance(obj, dict):
        return False, "output is not a JSON object"
    schema = dict(spec.get("schema") or {})
    for key in schema.get("required", ()):
        if key not in obj:
            return False, f"missing key {key!r}"
    if "calls_order" in schema:
        got = [c.get("tool") for c in obj.get("calls", []) if isinstance(c, dict)]
        if got != list(schema["calls_order"]):
            return False, f"call order {got} != {schema['calls_order']}"
    if "final" in schema:
        got_final = _first_number(str(obj.get("final")))
        tolerance = float(schema.get("tol", 0.01) or 0.0)
        if got_final is None or abs(got_final - float(schema["final"])) > tolerance:
            return False, "final value wrong"
    for key, want in (schema.get("exact") or {}).items():
        if str(obj.get(key)).strip() != str(want).strip():
            return False, f"{key}={obj.get(key)!r} != {want!r}"
    return True, "structured"


# --------------------------------------------------------------------------- #
# reading the corpus
# --------------------------------------------------------------------------- #

_CACHE: Dict[str, Tuple[Record, ...]] = {}


def _read(path: str) -> Iterator[Dict[str, Any]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def corpus(path: Optional[str] = None) -> Tuple[Record, ...]:
    """Every record, read once and cached. Missing corpus is an empty tuple, never a crash."""
    key = str(path or CORPUS_PATH)
    if key not in _CACHE:
        try:
            _CACHE[key] = tuple(Record.from_dict(d) for d in _read(key))
        except (OSError, json.JSONDecodeError):
            _CACHE[key] = ()
    return _CACHE[key]


def load(split: Optional[str] = None, *, faculty: Optional[str] = None,
         generator: Optional[str] = None, gradable_only: bool = False,
         limit: Optional[int] = None, seed: Optional[int] = None,
         path: Optional[str] = None) -> List[Record]:
    """Records matching every filter given.

    ``limit`` with a ``seed`` takes a *stratified* sample — proportional across generators — so a
    short run is a smaller version of the full examination rather than whichever generator happens
    to sort first. Without a seed it truncates, which is reproducible but not representative, and
    the two are different on purpose.
    """
    items = [r for r in corpus(path)
             if (split is None or r.split == split)
             and (faculty is None or r.faculty == faculty)
             and (generator is None or r.generator == generator)
             and (not gradable_only or r.gradable)]
    if limit is None or limit >= len(items):
        return items
    if seed is None:
        return items[:limit]

    import random as _random

    rng = _random.Random(seed)
    buckets: Dict[str, List[Record]] = {}
    for record in items:
        buckets.setdefault(record.generator, []).append(record)
    picked: List[Record] = []
    for name in sorted(buckets):
        group = sorted(buckets[name], key=lambda r: r.id)
        rng.shuffle(group)
        share = max(1, round(limit * len(group) / len(items)))
        picked.extend(group[:share])
    rng.shuffle(picked)
    return sorted(picked[:limit], key=lambda r: r.id)


def splits(path: Optional[str] = None) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in corpus(path):
        counts[record.split] = counts.get(record.split, 0) + 1
    return counts


def faculties(path: Optional[str] = None) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in corpus(path):
        counts[record.faculty] = counts.get(record.faculty, 0) + 1
    return counts


def generators(path: Optional[str] = None) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in corpus(path):
        counts[record.generator] = counts.get(record.generator, 0) + 1
    return counts


def self_test(records: Optional[Sequence[Record]] = None) -> Dict[str, Any]:
    """The corpus's own QA: every reference answer must pass its own verifier.

    This does not measure NJP at all — it measures whether :func:`verify` is a faithful port. A
    pass rate below 100% means the grader is broken and every score produced with it is void, so
    this runs before anything else in the examination and in its own test.
    """
    items = list(records if records is not None else corpus())
    checked = passed = skipped = 0
    failures: List[Dict[str, str]] = []
    for record in items:
        if not record.gradable:
            skipped += 1
            continue
        checked += 1
        verdict = verify(record, record.answer)
        if verdict.right:
            passed += 1
        elif len(failures) < 10:
            failures.append({"id": record.id, "why": verdict.why})
    return {
        "checked": checked, "passed": passed, "rubric_skipped": skipped,
        "pass_rate": round(100.0 * passed / checked, 2) if checked else None,
        "failures": failures,
    }


#: Package-level aliases. :mod:`nyxara.njp` exports these rather than the bare names, which are
#: right inside this module and far too general one level up.
CorpusRecord = Record
CorpusVerdict = Verdict
verify_corpus_answer = verify
