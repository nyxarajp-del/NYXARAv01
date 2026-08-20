"""NYXARA · growth/ouroboros.py — she rewrites her own source, and has to prove it was worth it (⟲).

The machinery for self-modification already exists and is real:
:meth:`~nyxara.growth.self_optimize.Optimizer.apply` writes to the live source tree, runs a
gauntlet (syntax → the eval safety battery → a **constitution-seal subprocess** → capability
non-regression → the frontier probe → ``pytest``) and restores the file byte-for-byte on any
failure; :mod:`~nyxara.growth.proof_carrying` and :mod:`~nyxara.growth.oversight_verify` can veto
before that; :mod:`~nyxara.growth.improvement_proof` rolls an edit back even after it *passes*
unless it can certify the result is better. ``PROTECTED_RELPATHS`` already seals the constitution
— ``kernel/rules.py``, ``identity/values.py``, ``guard/corrigibility.py``, ``growth/loyalty.py`` —
fail-closed, so the Master's standing requirement that the Loyalty Equation stay untouchable is
enforced by code that predates this module.

What was missing was the part that makes it *worth* running. Three gaps:

* **Nothing connected a profiler to an edit.** :class:`~nyxara.growth.hotspot_profiler.HotspotProfiler`
  ranks real hot functions and :class:`~nyxara.growth.weakness.WeaknessSynthesizer` ranks real
  weaknesses, but ``EditGenerator`` only ever consumed the latter — so edits were never aimed at
  where time is actually spent.
* **The deterministic transforms were linter-class.** Bare-except, ``== None``, unused import,
  mutable default. All worth fixing; none of them make anything *faster*.
* **Improvement was certified, never measured.** ``improvement_proof``'s Method B compares a static
  cost proxy. Nothing ever ran the workload twice and looked at the clock.

This module supplies those three, and one engineering decision underneath them all.

**Rewrites are spliced at exact source spans, never round-tripped through ``ast.unparse``.**
Unparsing regenerates the whole file from the tree, which silently deletes every comment and
reflows everything else. In a codebase whose comments carry most of the design rationale, a
self-modifier that unparses is a self-modifier that vandalises itself on its first successful run.
So the AST is used for the *proof* and the byte offsets are used for the *edit*: everything the
transform did not touch survives verbatim.

**On the honest ceiling.** These transforms are genuinely algorithmic — they change complexity or
remove materialisation — and each carries a precondition it must discharge before firing. They are
not "invent a better attention mechanism". Writing genuinely novel algorithms is bounded by the
quality of the model doing the writing, and hers is currently a small NumPy transformer. What is
delivered here is a loop that finds the hot code, applies provably-equivalent improvements, proves
the result is faster on the real workload, and rolls back byte-for-byte when it is not.

Gated by ``ouroboros_enabled``: every other layer added here is additive, and this one edits her
own source. It ships **ON** under the max-power posture (the Master's deliberate choice) and is
**sealed off under the TEST profile**, because a hermetic suite must never be one call away from
rewriting the tree it is testing. That flag is the whole gate — nothing else here checks
``autonomous_enact`` or ``allow_llm_edits``.

Depends on ``growth/self_optimize`` (gauntlet + rollback), ``growth/hotspot_profiler``,
``growth/weakness`` and the stdlib. Nothing imports back.
"""

from __future__ import annotations

import ast
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["Bottleneck", "Rewrite", "MetamorphosisReport", "Ouroboros", "TRANSFORMS"]

# Only these subpackages may ever be rewritten. The Master named growth/ and mind/; everything
# else — kernel, guard, identity, agency — stays out of reach even before PROTECTED_RELPATHS gets
# a say, so a bug in a transform cannot reach the parts that decide what she is allowed to do.
DEFAULT_ALLOWLIST: Tuple[str, ...] = ("growth", "mind")


@dataclass
class Bottleneck:
    """Somewhere worth spending an edit — hot code, a ranked weakness, or both at once."""

    file: str
    func: str = ""
    lineno: int = 0
    tottime: float = 0.0
    ncalls: int = 0
    severity: float = 0.0
    origin: str = "hotspot"          # "hotspot" | "weakness" | "both"
    detail: str = ""

    @property
    def score(self) -> float:
        """Rank by measured self-time first, ranked severity second — time is the harder fact."""
        return self.tottime * 10.0 + self.severity

    def to_dict(self) -> Dict[str, Any]:
        return {"file": self.file, "func": self.func, "lineno": self.lineno,
                "tottime": round(self.tottime, 6), "ncalls": self.ncalls,
                "severity": round(self.severity, 4), "origin": self.origin,
                "detail": self.detail, "score": round(self.score, 6)}


@dataclass
class Rewrite:
    """One span of source to replace, with the transform that justified it."""

    start: int                       # byte/char offset into the source
    end: int
    replacement: str
    transform: str
    describe: str = ""


@dataclass
class MetamorphosisReport:
    """An auditable account of one cycle — including every edit that was refused."""

    scanned: int = 0
    proposed: int = 0
    applied: int = 0
    kept: int = 0
    rolled_back: int = 0
    refused: int = 0
    before_seconds: float = 0.0
    after_seconds: float = 0.0
    files: List[str] = field(default_factory=list)
    transforms: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def speedup(self) -> float:
        """``before/after`` on the same workload. 1.0 = no change; > 1.0 = genuinely faster."""
        if self.after_seconds <= 0.0 or self.before_seconds <= 0.0:
            return 1.0
        return self.before_seconds / self.after_seconds

    def note(self, message: str) -> None:
        if message and message not in self.notes:
            self.notes.append(message)

    def summary(self) -> str:
        head = (f"· ouroboros: scanned {self.scanned} bottleneck(s), proposed {self.proposed} "
                f"rewrite(s), applied {self.applied}, kept {self.kept}, "
                f"rolled back {self.rolled_back}, refused {self.refused}")
        lines = [head]
        if self.before_seconds > 0.0:
            lines.append(f"    workload {self.before_seconds:.4f}s → {self.after_seconds:.4f}s "
                         f"(speedup {self.speedup:.2f}×)")
        for name, count in sorted(self.transforms.items()):
            lines.append(f"    {name} ×{count}")
        lines.extend(f"    · {n}" for n in self.notes)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"scanned": self.scanned, "proposed": self.proposed, "applied": self.applied,
                "kept": self.kept, "rolled_back": self.rolled_back, "refused": self.refused,
                "before_seconds": round(self.before_seconds, 6),
                "after_seconds": round(self.after_seconds, 6),
                "speedup": round(self.speedup, 4), "files": list(self.files),
                "transforms": dict(self.transforms), "notes": list(self.notes)}


# --------------------------------------------------------------------------- #
# helpers for the precondition proofs
# --------------------------------------------------------------------------- #
def _offset_table(source: str) -> List[int]:
    """Character offset at which each 1-indexed line starts."""
    offsets = [0, 0]
    position = 0
    for line in source.splitlines(keepends=True):
        position += len(line)
        offsets.append(position)
    return offsets


def _span(node: ast.AST, offsets: Sequence[int]) -> Optional[Tuple[int, int]]:
    """The exact ``(start, end)`` character span of a node, or ``None`` when unavailable."""
    lineno = getattr(node, "lineno", None)
    end_lineno = getattr(node, "end_lineno", None)
    col = getattr(node, "col_offset", None)
    end_col = getattr(node, "end_col_offset", None)
    if None in (lineno, end_lineno, col, end_col):
        return None
    if end_lineno + 1 >= len(offsets):
        return None
    return offsets[lineno] + col, offsets[end_lineno] + end_col


def _hashable_constants(node: ast.AST) -> bool:
    """True when every element is a constant of a hashable type.

    This is the precondition for turning a list membership test into a set: an unhashable element
    would turn a working ``in`` into a ``TypeError``, so an unproven element means no rewrite."""
    elements = getattr(node, "elts", None)
    if not elements:
        return False
    for element in elements:
        if not isinstance(element, ast.Constant):
            return False
        if not isinstance(element.value, (str, bytes, int, float, bool, complex, type(None))):
            return False
    return True


# --------------------------------------------------------------------------- #
# the transforms — each one proves its precondition before it fires
# --------------------------------------------------------------------------- #
def _t_membership_set(tree: ast.AST, source: str, offsets: Sequence[int]) -> List[Rewrite]:
    """``x in [a, b, c]`` → ``x in {a, b, c}`` — a linear scan becomes a hash lookup.

    Precondition: every element is a hashable constant. Without that the rewrite could turn a
    working comparison into a ``TypeError``, so an unprovable element is simply left alone."""
    out: List[Rewrite] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        if not isinstance(node.ops[0], (ast.In, ast.NotIn)):
            continue
        target = node.comparators[0]
        if not isinstance(target, (ast.List, ast.Tuple)) or not _hashable_constants(target):
            continue
        span = _span(target, offsets)
        if span is None:
            continue
        inner = source[span[0] + 1:span[1] - 1].strip().rstrip(",")
        if not inner:
            continue
        out.append(Rewrite(start=span[0], end=span[1], replacement="{" + inner + "}",
                           transform="membership_set",
                           describe="O(n) membership scan → O(1) hash lookup"))
    return out


def _t_dict_keys_membership(tree: ast.AST, source: str, offsets: Sequence[int]) -> List[Rewrite]:
    """``k in d.keys()`` → ``k in d`` — identical semantics, one fewer view object per call."""
    out: List[Rewrite] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        if not isinstance(node.ops[0], (ast.In, ast.NotIn)):
            continue
        target = node.comparators[0]
        if not (isinstance(target, ast.Call) and isinstance(target.func, ast.Attribute)
                and target.func.attr == "keys" and not target.args and not target.keywords):
            continue
        span = _span(target.func.value, offsets)
        full = _span(target, offsets)
        if span is None or full is None:
            continue
        out.append(Rewrite(start=full[0], end=full[1], replacement=source[span[0]:span[1]],
                           transform="dict_keys_membership",
                           describe="drop the redundant .keys() view"))
    return out


def _t_iterate_keys(tree: ast.AST, source: str, offsets: Sequence[int]) -> List[Rewrite]:
    """``for k in d.keys():`` → ``for k in d:`` — iterating a mapping already yields its keys."""
    out: List[Rewrite] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            continue
        iterable = node.iter
        if not (isinstance(iterable, ast.Call) and isinstance(iterable.func, ast.Attribute)
                and iterable.func.attr == "keys" and not iterable.args and not iterable.keywords):
            continue
        inner = _span(iterable.func.value, offsets)
        full = _span(iterable, offsets)
        if inner is None or full is None:
            continue
        out.append(Rewrite(start=full[0], end=full[1], replacement=source[inner[0]:inner[1]],
                           transform="iterate_keys",
                           describe="iterating a mapping already yields its keys"))
    return out


# Aggregators that consume an iterable exactly once and never index it, so handing them a
# generator instead of a materialised list is behaviour-preserving and allocates nothing.
_LAZY_SAFE = frozenset({"sum", "any", "all", "min", "max", "set", "frozenset", "tuple", "sorted"})


def _t_lazy_aggregate(tree: ast.AST, source: str, offsets: Sequence[int]) -> List[Rewrite]:
    """``sum([f(x) for x in xs])`` → ``sum(f(x) for x in xs)`` — stop building the throwaway list.

    Restricted to aggregators that consume their argument exactly once and never index or re-scan
    it. ``sorted`` is included because it materialises internally anyway; ``list``/``dict`` are
    excluded because the list *is* the answer there."""
    out: List[Rewrite] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id not in _LAZY_SAFE or len(node.args) != 1 or node.keywords:
            continue
        argument = node.args[0]
        if not isinstance(argument, ast.ListComp):
            continue
        span = _span(argument, offsets)
        if span is None:
            continue
        inner = source[span[0] + 1:span[1] - 1].strip()
        if not inner:
            continue
        out.append(Rewrite(start=span[0], end=span[1], replacement="(" + inner + ")",
                           transform="lazy_aggregate",
                           describe="drop the intermediate list; aggregate lazily"))
    return out


TRANSFORMS: Tuple[Callable[[ast.AST, str, Sequence[int]], List[Rewrite]], ...] = (
    _t_membership_set,
    _t_dict_keys_membership,
    _t_iterate_keys,
    _t_lazy_aggregate,
)


# --------------------------------------------------------------------------- #
# the engine
# --------------------------------------------------------------------------- #
class Ouroboros:
    """Find the hot code, rewrite it provably, prove it got faster — or put it back."""

    def __init__(self, *, settings: Any = None, root: Optional[Path] = None,
                 allowlist: Optional[Sequence[str]] = None,
                 optimizer: Any = None) -> None:
        self.settings = settings
        cfg = getattr(settings, "self_improvement", None) if settings is not None else None
        self.root = Path(root) if root else Path(__file__).resolve().parent.parent
        self.allowlist = tuple(allowlist if allowlist is not None
                               else getattr(cfg, "ouroboros_allowlist", DEFAULT_ALLOWLIST))
        self.require_speedup = bool(getattr(cfg, "ouroboros_require_speedup", True))
        self.require_clean_tree = bool(getattr(cfg, "ouroboros_require_clean_tree", True))
        self.max_edits = int(getattr(cfg, "ouroboros_max_edits_per_cycle", 3))
        self._optimizer = optimizer

    # ---- fences ---- #
    def in_scope(self, path: Any) -> bool:
        """Is this file one she may rewrite? Fail-closed on anything unresolvable."""
        try:
            target = Path(str(path)).resolve()
            relative = target.relative_to(self.root)
        except Exception:  # noqa: BLE001 — outside the package, or unresolvable ⇒ out of scope
            return False
        if not relative.parts or relative.parts[0] not in self.allowlist:
            return False
        try:
            from nyxara.growth.self_optimize import _is_constitutional
            if _is_constitutional(str(target), self.root):
                return False        # the constitution is sealed; this is belt as well as braces
        except Exception:  # noqa: BLE001 — cannot prove it is safe ⇒ treat it as protected
            return False
        return True

    def tree_is_clean(self) -> bool:
        """Is the working tree clean? ``Optimizer``'s rollback is an in-memory string only.

        If the process dies between the write and the restore, that string is gone and the file
        stays broken. A clean tree means ``git checkout`` is a real last-resort undo, and demanding
        one before a cycle that rewrites her own source is cheap insurance."""
        try:
            result = subprocess.run(["git", "status", "--porcelain"], cwd=str(self.root.parent),
                                    capture_output=True, text=True, timeout=30)
        except Exception:  # noqa: BLE001 — no git is not a clean tree we can vouch for
            return False
        return result.returncode == 0 and not result.stdout.strip()

    # ---- scanning ---- #
    def scan(self, *, workload: Optional[Callable[[], Any]] = None,
             top: int = 10) -> List[Bottleneck]:
        """Rank places worth an edit — measured hot functions fused with ranked weaknesses.

        The fusion is the point: the profiler knows *where the time goes* and the weakness
        synthesiser knows *what is wrong*, and until now nothing joined them, so edits were never
        aimed at hot code."""
        found: Dict[str, Bottleneck] = {}

        if workload is not None:
            try:
                from nyxara.growth.hotspot_profiler import HotspotProfiler
                for spot in HotspotProfiler().profile(workload, top=top):
                    if not self.in_scope(spot.filename):
                        continue
                    found[spot.filename] = Bottleneck(
                        file=spot.filename, func=spot.func, lineno=spot.lineno,
                        tottime=spot.tottime, ncalls=spot.ncalls, origin="hotspot",
                        detail=f"{spot.tottime:.4f}s self-time over {spot.ncalls} call(s)")
            except Exception:  # noqa: BLE001 — no profile is fewer bottlenecks, never a crash
                pass

        try:
            from nyxara.growth.epistemic_drive import EpistemicDrive
            for gap in EpistemicDrive(settings=self.settings).engineering_gaps(top):
                path = self._path_from_detail(gap.subject)
                if path is None or not self.in_scope(path):
                    continue
                existing = found.get(path)
                if existing is not None:
                    existing.severity = max(existing.severity, gap.pressure)
                    existing.origin = "both"
                    existing.detail += f"; {gap.subject}"
                else:
                    found[path] = Bottleneck(file=path, severity=gap.pressure,
                                             origin="weakness", detail=gap.subject)
        except Exception:  # noqa: BLE001
            pass

        return sorted(found.values(), key=lambda b: b.score, reverse=True)[:max(0, top)]

    def _path_from_detail(self, subject: str) -> Optional[str]:
        """Recover a file path from a weakness title like "layering violation: a.b → c.d"."""
        for token in subject.replace("→", " ").replace(",", " ").split():
            if not token.startswith("nyxara."):
                continue
            candidate = self.root / Path(*token.split(".")[1:]).with_suffix(".py")
            if candidate.is_file():
                return str(candidate)
        return None

    # ---- proposing ---- #
    def propose(self, path: Any) -> Optional[Any]:
        """Build a :class:`~nyxara.growth.self_optimize.SourceEdit` for one file, or ``None``.

        Rewrites are spliced at exact character spans, applied back-to-front so earlier offsets
        stay valid, and overlapping spans are dropped rather than merged. The file is re-parsed
        afterwards: a rewrite that does not produce valid Python is discarded here rather than
        being handed to the gauntlet to catch."""
        if not self.in_scope(path):
            return None
        target = Path(str(path))
        try:
            source = target.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except Exception:  # noqa: BLE001 — an unreadable or unparseable file is simply skipped
            return None

        offsets = _offset_table(source)
        rewrites: List[Rewrite] = []
        for transform in TRANSFORMS:
            try:
                rewrites.extend(transform(tree, source, offsets))
            except Exception:  # noqa: BLE001 — one bad transform never costs the others
                continue
        if not rewrites:
            return None

        rewrites.sort(key=lambda r: r.start, reverse=True)
        applied: List[Rewrite] = []
        last_start = len(source) + 1
        for rewrite in rewrites:
            if rewrite.end > last_start:
                continue          # overlaps a rewrite already accepted — leave this one alone
            applied.append(rewrite)
            last_start = rewrite.start

        after = source
        for rewrite in applied:
            after = after[:rewrite.start] + rewrite.replacement + after[rewrite.end:]
        if after == source:
            return None
        try:
            ast.parse(after)      # never hand the gauntlet something we already know is broken
        except SyntaxError:
            return None

        from nyxara.growth.self_optimize import SourceEdit
        names = sorted({r.transform for r in applied})
        return SourceEdit(file=str(target.resolve()), kind="ouroboros:" + ",".join(names),
                          before=source, after=after,
                          rationale="; ".join(sorted({r.describe for r in applied})))

    # ---- the cycle ---- #
    def _time_workload(self, workload: Callable[[], Any], *, repeats: int = 3) -> float:
        """Best-of-N wall time. Best, not mean: it is the least noisy estimator on a busy box."""
        best = float("inf")
        for _ in range(max(1, repeats)):
            start = time.perf_counter()
            try:
                workload()
            except Exception:  # noqa: BLE001 — a workload that fails times as unusable
                return float("inf")
            best = min(best, time.perf_counter() - start)
        return best

    def metamorphose(self, *, workload: Optional[Callable[[], Any]] = None,
                     enact: bool = False, max_edits: Optional[int] = None,
                     top: int = 10) -> MetamorphosisReport:
        """One cycle: scan → propose → (optionally) apply under the gauntlet → measure → keep.

        Without ``enact`` she scans, proposes and reports, and writes nothing. With it, each edit
        goes through :meth:`~nyxara.growth.self_optimize.Optimizer.apply` — the existing gauntlet,
        the existing byte-exact rollback — and then, when a workload is supplied and
        ``require_speedup`` holds, the workload is re-timed and an edit that did not actually make
        things faster is put back."""
        report = MetamorphosisReport()
        if enact and self.require_clean_tree and not self.tree_is_clean():
            report.note("refusing to enact: the working tree is dirty, so `git checkout` would "
                        "not be a clean last-resort undo")
            return report

        bottlenecks = self.scan(workload=workload, top=top)
        report.scanned = len(bottlenecks)
        limit = self.max_edits if max_edits is None else int(max_edits)

        edits: List[Any] = []
        for bottleneck in bottlenecks:
            if len(edits) >= max(0, limit):
                break
            edit = self.propose(bottleneck.file)
            if edit is None:
                continue
            edits.append(edit)
            report.proposed += 1
            report.files.append(edit.file)
            for name in edit.kind.split(":", 1)[-1].split(","):
                report.transforms[name] = report.transforms.get(name, 0) + 1

        if not enact:
            report.note("measure-only: nothing was written (pass --enact to apply)")
            return report

        if workload is not None:
            report.before_seconds = self._time_workload(workload)

        optimizer = self._optimizer
        if optimizer is None:
            try:
                from nyxara.growth.self_optimize import Optimizer
                optimizer = Optimizer(settings=self.settings)
            except Exception as exc:  # noqa: BLE001
                report.note(f"no optimizer available: {exc}")
                return report

        for edit in edits:
            try:
                outcome = optimizer.apply(edit)
            except Exception as exc:  # noqa: BLE001 — a failed apply is reported, never raised
                report.refused += 1
                report.note(f"{Path(edit.file).name}: {exc}")
                continue
            report.applied += int(bool(getattr(outcome, "applied", False)))
            if getattr(outcome, "kept", False):
                report.kept += 1
            elif getattr(outcome, "rolled_back", False):
                report.rolled_back += 1
            else:
                report.refused += 1
                report.note(f"{Path(edit.file).name}: {getattr(outcome, 'reason', 'refused')}")

        if workload is not None and report.kept:
            report.after_seconds = self._time_workload(workload)
            if self.require_speedup and report.speedup <= 1.0:
                restored = self._restore(edits)
                report.kept -= restored
                report.rolled_back += restored
                report.note(f"no measured speedup ({report.speedup:.2f}×) — "
                            f"restored {restored} file(s)")
        return report

    @staticmethod
    def _restore(edits: Sequence[Any]) -> int:
        """Put every edited file back byte-for-byte. Used when the speedup did not materialise."""
        restored = 0
        for edit in edits:
            try:
                Path(edit.file).write_text(edit.before, encoding="utf-8")
                restored += 1
            except Exception:  # noqa: BLE001 — report what we could not restore via the count
                continue
        return restored
