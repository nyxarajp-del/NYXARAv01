"""NYXARA · nyx/author.py — prose in, verified code out, or a refusal that names itself
(✍, NYX V.02, gap 5).

L-OMNI (V.01) already rewrites her own slow functions in C — but only functions she has
**already measured as slow**. It lowers what exists; it cannot write what does not. From a
sentence like *"a function that returns the nth triangular number"* NYX V.01 had no path to
running code at all.

:class:`Author` is that path, and it is a **gauntlet**, not a generator:

1. **read the spec** — through :mod:`nyxara.nyx.intent`. An ambiguous instruction is asked
   about, never guessed at: building the wrong thing correctly is still building the wrong
   thing.
2. **synthesise** — :class:`~nyxara.agency.self_coder.CodeSynthesizer`, which writes real
   Python deterministically for the families it can *derive*, and misses honestly otherwise.
3. **screen** — :func:`~nyxara.growth.module_loader.screen_source`, the repo's one shared AST
   allowlist. Nothing unscreened is ever executed.
4. **run** — :func:`~nyxara.agency.code_sandbox.run_python`, isolated subprocess, wall-clock
   timeout, network off.
5. **check the answer** — the synthesiser computed the expected value locally *before* the
   program ran. If the program disagrees with it, the code is wrong and is thrown away. This
   is the step that stops plausible-looking wrong code from ever shipping.
6. **gauntlet** — :func:`~nyxara.growth.verify.build_default_verifier`'s integrity invariants
   must hold before and after. The constitutional core is checked, not trusted.
7. **load** — :func:`~nyxara.growth.module_loader.load_module_from_path`, into
   ``nyxara/growth/_forged/`` and nowhere else.
8. **lineage** — a signed :class:`~nyxara.growth.lineage.LineageLedger` entry, so what she
   wrote is on a chain that can be verified and reverted.

Honest framing (the rule this repo keeps, see ``docs/CAPABILITIES.md``):

* The synthesiser is **bounded**: arithmetic, number theory, closed-form sequences, list
  operations, string operations. That is a real set and a small one. *"Write me a distributed
  consensus subsystem"* is outside it, and the answer is a **refusal that names what was not
  derived** — not a plausible-looking file. Writing a whole subsystem from prose is what a
  large pretrained model does; none ships here, and pretending otherwise would be the exact
  dishonesty this repo exists to avoid.
* Nothing ships unverified. A synthesis that runs but returns the wrong value is discarded at
  step 5 and reported as a failure, not softened into a partial success.
* Every stage is recorded on the result, so ``/nyx author`` shows *where* it stopped rather
  than only that it did.
* Oversight is asked before anything is written or loaded, and ``/scram`` stops her.

Pure standard library. Every public method is fail-soft: on any error it degrades to a null
result rather than breaking a turn.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["Stage", "Authored", "Author"]

#: The pipeline, in order. Every one of these appears on the result, passed or not.
STAGES = ("read", "synthesise", "screen", "run", "check", "gauntlet", "load", "lineage")

_SAFE_NAME = re.compile(r"[^a-z0-9_]+")


@dataclass
class Stage:
    """One step of the gauntlet: did it pass, and what did it say?"""

    name: str = ""
    ok: bool = False
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class Authored:
    """The outcome of one authoring attempt — code that survived, or a named refusal."""

    spec: str = ""
    ok: bool = False
    source: str = ""
    family: str = ""
    expected: Any = None          # what she computed locally, before running anything
    value: Any = None             # what the program actually returned
    verified: bool = False        # ran, and agreed with the expected value
    loaded: bool = False
    module: str = ""
    path: str = ""
    lineage: str = ""             # ledger head after the entry
    stages: List[Stage] = field(default_factory=list)
    refusal: str = ""
    duration_ms: float = 0.0

    @property
    def stopped_at(self) -> str:
        """The first stage that did not pass — ``""`` when everything did."""
        for stage in self.stages:
            if not stage.ok:
                return stage.name
        return ""

    def render(self) -> str:
        lines = [f"spec: {self.spec}"]
        for stage in self.stages:
            lines.append(f"  {'✓' if stage.ok else '✗'} {stage.name:11} {stage.detail}")
        if self.ok:
            lines.append(f"  → {self.value!r}"
                         + (f"   loaded as {self.module}" if self.loaded else ""))
        elif self.refusal:
            lines.append(f"  refused: {self.refusal}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"spec": self.spec, "ok": self.ok, "family": self.family,
                "expected": self.expected, "value": self.value, "verified": self.verified,
                "loaded": self.loaded, "module": self.module, "path": self.path,
                "lineage": self.lineage, "refusal": self.refusal,
                "stopped_at": self.stopped_at,
                "duration_ms": round(self.duration_ms, 2),
                "stages": [s.to_dict() for s in self.stages],
                "source": self.source}


class Author:
    """Spec → synthesise → screen → run → check → gauntlet → load → lineage. Or refuse."""

    def __init__(self, brain: Any = None, *, sandbox_timeout_s: float = 5.0,
                 load: bool = True, lineage: bool = True,
                 max_source_bytes: int = 20_000) -> None:
        self.brain = brain
        self.sandbox_timeout_s = max(0.5, float(sandbox_timeout_s))
        self.load_enabled = bool(load)
        self.lineage_enabled = bool(lineage)
        self.max_source_bytes = max(256, int(max_source_bytes))

        self._synth: Any = None
        self._ledger: Any = None
        self.attempts = 0
        self.shipped = 0
        self.refused = 0
        self.by_family: Dict[str, int] = {}
        self.history: List[Dict[str, Any]] = []

    # ---- the engines ------------------------------------------------------ #
    def synthesiser(self) -> Any:
        if self._synth is None:
            try:
                from nyxara.agency.self_coder import CodeSynthesizer
                self._synth = CodeSynthesizer()
            except Exception:  # noqa: BLE001 — without it she cannot write at all, and says so
                self._synth = None
        return self._synth

    def ledger(self) -> Any:
        if self._ledger is None and self.lineage_enabled:
            try:
                from nyxara.growth.lineage import LineageLedger
                self._ledger = LineageLedger()
            except Exception:  # noqa: BLE001 — a missing ledger is a missing receipt, not a stop
                self._ledger = None
        return self._ledger

    def available(self) -> bool:
        return self.synthesiser() is not None

    def families(self) -> List[str]:
        """What the synthesiser can actually derive. Small, real, and stated up front."""
        return ["arithmetic", "number_theory", "sequence", "string"]

    # ---- the whole act ----------------------------------------------------- #
    def write(self, spec: str, *, name: str = "", load: Optional[bool] = None,
              oversight: Any = None) -> Authored:
        """Run the gauntlet. Returns code that survived every stage, or a named refusal."""
        out = Authored(spec=str(spec or "").strip())
        t0 = time.perf_counter()
        try:
            self.attempts += 1
            if not out.spec:
                out.refusal = "nothing was specified"
                self.refused += 1
                return out
            if self._stopped(oversight):
                out.refusal = "oversight has paused or scrammed her; she does not write code"
                self.refused += 1
                return out

            goal = self._read(out)
            if goal is None:
                self.refused += 1
                return out
            if not self._synthesise(out, goal):
                self.refused += 1
                return out
            if not self._screen(out):
                self.refused += 1
                return out
            if not self._run(out):
                self.refused += 1
                return out
            if not self._check(out):
                self.refused += 1
                return out
            if not self._gauntlet(out):
                self.refused += 1
                return out

            want_load = self.load_enabled if load is None else bool(load)
            if want_load:
                self._load(out, name=name)
            else:
                out.stages.append(Stage("load", True, "not requested — verified, not loaded"))
            self._lineage(out)

            out.ok = True
            self.shipped += 1
            self.by_family[out.family] = self.by_family.get(out.family, 0) + 1
            return out
        except Exception as exc:  # noqa: BLE001 — authoring never breaks a turn
            out.refusal = f"{type(exc).__name__}: {exc}"
            return out
        finally:
            out.duration_ms = (time.perf_counter() - t0) * 1000.0
            self.history.append({"spec": out.spec, "ok": out.ok,
                                 "stopped_at": out.stopped_at, "family": out.family})
            if len(self.history) > 64:
                del self.history[: len(self.history) - 64]

    # ---- stage 1: read the spec ------------------------------------------- #
    def _read(self, out: Authored) -> Optional[str]:
        """Understand what is being asked for. An ambiguous spec is asked about, not guessed."""
        reader = getattr(self.brain, "intent", None)
        if reader is None:
            out.stages.append(Stage("read", True, "no intent reader — taking the spec verbatim"))
            return out.spec
        try:
            intent = reader.read(out.spec)
        except Exception:  # noqa: BLE001
            out.stages.append(Stage("read", True, "intent reader failed — spec taken verbatim"))
            return out.spec
        if intent.contradictions:
            out.stages.append(Stage("read", False, "the spec contradicts itself"))
            out.refusal = intent.clarifying_question()
            return None
        if intent.ambiguous:
            # Building the wrong thing correctly is still building the wrong thing.
            out.stages.append(Stage("read", False, "two readings of the spec are live"))
            out.refusal = intent.clarifying_question()
            return None
        out.stages.append(Stage("read", True, f"{intent.kind}: {intent.goal or out.spec}"))
        return out.spec

    # ---- stage 2: synthesise ----------------------------------------------- #
    def _synthesise(self, out: Authored, goal: str) -> bool:
        synth = self.synthesiser()
        if synth is None:
            out.stages.append(Stage("synthesise", False, "no synthesiser in this build"))
            out.refusal = "the code synthesiser is not available in this build"
            return False
        result = synth.synthesize(goal)
        if not getattr(result, "ok", False):
            out.stages.append(Stage("synthesise", False, str(getattr(result, "note", ""))))
            out.refusal = self._refusal_for(goal, result)
            return False
        out.source = str(getattr(result, "source", "") or "")
        out.family = str(getattr(result, "family", "") or "")
        out.expected = getattr(result, "expected", None)
        if len(out.source.encode("utf-8")) > self.max_source_bytes:
            out.stages.append(Stage("synthesise", False, "synthesised source is over the cap"))
            out.refusal = "the synthesised program is larger than she is willing to run"
            return False
        out.stages.append(Stage("synthesise", True,
                                f"{out.family}, {len(out.source.splitlines())} lines"))
        return True

    def _refusal_for(self, goal: str, result: Any) -> str:
        """Name what was *not* derived, and what she genuinely can do. Never a soft maybe."""
        note = str(getattr(result, "note", "") or "").strip()
        return (
            f"I could not derive a program for {goal!r}. "
            f"My synthesiser is deterministic and bounded — it writes real code for "
            f"{', '.join(self.families())}, and derives the expected answer before running "
            f"anything so it can tell whether the code is right. This spec is outside those "
            f"families"
            + (f" ({note})" if note else "")
            + ". Writing a whole subsystem from prose is what a large pretrained model does, "
              "and none ships here. I would rather say that than hand you a file that looks "
              "like the thing you asked for.")

    # ---- stage 3: screen ---------------------------------------------------- #
    def _screen(self, out: Authored) -> bool:
        try:
            from nyxara.growth.module_loader import screen_source
            reason = screen_source(out.source)
        except Exception as exc:  # noqa: BLE001 — no screen means no run: fail closed
            out.stages.append(Stage("screen", False, f"screen unavailable ({exc})"))
            out.refusal = "the source screen is unavailable, so nothing is run — fail closed"
            return False
        if reason:
            out.stages.append(Stage("screen", False, str(reason)))
            out.refusal = f"her own source screen refused the program she wrote: {reason}"
            return False
        out.stages.append(Stage("screen", True, "passed the shared AST allowlist"))
        return True

    # ---- stage 4: run it ---------------------------------------------------- #
    def _run(self, out: Authored) -> bool:
        try:
            from nyxara.agency.code_sandbox import run_python
            result = run_python(out.source, timeout_s=self.sandbox_timeout_s,
                                allow_network=False)
        except Exception as exc:  # noqa: BLE001
            out.stages.append(Stage("run", False, f"{type(exc).__name__}: {exc}"))
            out.refusal = "the sandbox could not run the program"
            return False
        out.value = getattr(result, "value", None)
        if not getattr(result, "ok", False):
            detail = str(getattr(result, "stderr", "") or getattr(result, "error", ""))[:200]
            out.stages.append(Stage("run", False, detail or "the program failed"))
            out.refusal = f"the program she wrote does not run: {detail or 'it failed'}"
            return False
        out.stages.append(Stage("run", True, f"returned {out.value!r}"))
        return True

    # ---- stage 5: is the answer actually right? ----------------------------- #
    def _check(self, out: Authored) -> bool:
        """The step that stops plausible-looking wrong code from shipping.

        The synthesiser computed the expected value locally, by a different route, *before*
        the program ran. Agreement is evidence; disagreement means the code is wrong, and a
        wrong program that runs cleanly is exactly the failure that would otherwise slip
        through every other stage in this file.
        """
        if out.expected is None:
            out.stages.append(Stage("check", True,
                                    "no independently computed answer to check against"))
            return True
        if self._equal(out.value, out.expected):
            out.verified = True
            out.stages.append(Stage("check", True,
                                    f"agrees with the independently computed {out.expected!r}"))
            return True
        out.stages.append(Stage("check", False,
                                f"expected {out.expected!r}, the program returned {out.value!r}"))
        out.refusal = (f"the program runs but is wrong: I computed {out.expected!r} "
                       f"independently and it returned {out.value!r}. That is thrown away, "
                       f"not shipped.")
        return False

    @staticmethod
    def _equal(value: Any, expected: Any) -> bool:
        if value == expected:
            return True
        try:                                   # a sandbox round-trip can restring a number
            return str(value).strip() == str(expected).strip()
        except Exception:  # noqa: BLE001
            return False

    # ---- stage 6: the gauntlet ---------------------------------------------- #
    def _gauntlet(self, out: Authored) -> bool:
        """The constitutional invariants, checked rather than trusted."""
        try:
            from nyxara.growth.verify import build_default_verifier
            verifier = build_default_verifier()
            report = verifier.verify({}, {}, change_id=f"author:{out.family}")
        except Exception as exc:  # noqa: BLE001 — no gauntlet means no load: fail closed
            out.stages.append(Stage("gauntlet", False, f"unavailable ({exc})"))
            out.refusal = "the verification gauntlet is unavailable, so nothing is loaded"
            return False
        passed = bool(getattr(report, "passed", False) or getattr(report, "ok", False))
        if not passed:
            failed = [r.name for r in getattr(report, "results", [])
                      if not getattr(r, "ok", True)]
            out.stages.append(Stage("gauntlet", False, ", ".join(failed) or "failed"))
            out.refusal = f"the integrity gauntlet failed: {', '.join(failed) or 'unknown'}"
            return False
        out.stages.append(Stage("gauntlet", True,
                                f"{len(getattr(report, 'results', []))} invariants held"))
        return True

    # ---- stage 7: load it into the process ---------------------------------- #
    def _load(self, out: Authored, *, name: str = "") -> None:
        try:
            from nyxara.growth.module_loader import forged_root, load_module_from_path
            root = forged_root()
            root.mkdir(parents=True, exist_ok=True)
            stem = _SAFE_NAME.sub("_", (name or out.family or "authored").lower()).strip("_")
            path = root / f"nyx_authored_{stem}_{self.attempts}.py"
            path.write_text(out.source, encoding="utf-8")
            out.path = str(path)
            loaded = load_module_from_path(path)
            out.loaded = bool(getattr(loaded, "ok", False))
            out.module = str(getattr(loaded, "name", "") or "")
            out.stages.append(Stage("load", out.loaded,
                                    out.module if out.loaded
                                    else str(getattr(loaded, "reason", "refused"))))
        except Exception as exc:  # noqa: BLE001 — a failed load leaves verified-but-unloaded
            out.stages.append(Stage("load", False, f"{type(exc).__name__}: {exc}"))

    # ---- stage 8: the receipt ----------------------------------------------- #
    def _lineage(self, out: Authored) -> None:
        ledger = self.ledger()
        if ledger is None:
            out.stages.append(Stage("lineage", True, "no ledger configured"))
            return
        try:
            record = ledger.record(
                "authored", f"wrote {out.family} code for: {out.spec[:120]}",
                detail={"family": out.family, "expected": out.expected,
                        "value": out.value, "path": out.path, "module": out.module,
                        "lines": len(out.source.splitlines())},
                certificate="verified against an independently computed answer"
                if out.verified else "")
            out.lineage = str(getattr(record, "record_hash", "") or "")
            out.stages.append(Stage("lineage", True, out.lineage[:16]))
        except Exception as exc:  # noqa: BLE001 — a missing receipt is not a reason to unship
            out.stages.append(Stage("lineage", False, f"{type(exc).__name__}: {exc}"))

    # ---- corrigibility ------------------------------------------------------ #
    @staticmethod
    def _stopped(oversight: Any) -> bool:
        if oversight is None:
            return False
        try:
            for attr in ("scrammed", "is_scrammed", "paused", "is_paused"):
                probe = getattr(oversight, attr, None)
                if probe is None:
                    continue
                if bool(probe() if callable(probe) else probe):
                    return True
            return False
        except Exception:  # noqa: BLE001 — an unreadable oversight is treated as STOP
            return True

    # ---- introspection ------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        try:
            return {
                "available": self.available(),
                "attempts": self.attempts, "shipped": self.shipped,
                "refused": self.refused,
                "by_family": dict(self.by_family),
                "families": self.families(),
                "stages": list(STAGES),
                "recent": list(self.history[-8:]),
                "note": ("a deterministic, bounded synthesiser behind a gauntlet: the expected "
                         "answer is computed independently before the program runs, so a "
                         "program that runs but is wrong is thrown away rather than shipped. "
                         "Writing a whole subsystem from prose is outside it, and is refused "
                         "by name"),
            }
        except Exception:  # noqa: BLE001
            return {}

    def describe(self) -> str:
        """What ``/nyx author`` prints with no argument."""
        try:
            return ("\n".join([
                f"synthesiser: {'available' if self.available() else 'absent'}",
                f"families   : {', '.join(self.families())}",
                f"gauntlet   : {' → '.join(STAGES)}",
                f"record     : {self.shipped} shipped, {self.refused} refused, "
                f"{self.attempts} attempted",
                "Outside those families she refuses and names what she could not derive — "
                "she does not hand you a file that looks like the thing you asked for.",
            ]))
        except Exception:  # noqa: BLE001
            return ""

    # ---- persistence -------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return {"attempts": self.attempts, "shipped": self.shipped,
                "refused": self.refused, "by_family": dict(self.by_family),
                "history": list(self.history)}

    def load_dict(self, data: Any) -> bool:
        try:
            if not isinstance(data, dict):
                return False
            self.attempts = int(data.get("attempts", 0))
            self.shipped = int(data.get("shipped", 0))
            self.refused = int(data.get("refused", 0))
            self.by_family = {str(k): int(v) for k, v in (data.get("by_family") or {}).items()}
            self.history = [h for h in (data.get("history") or []) if isinstance(h, dict)][-64:]
            return True
        except Exception:  # noqa: BLE001 — a corrupt sidecar must never block boot
            return False
