"""NYXARA · nyx/consequence.py — look before acting (⏳, NYX V.02, L-CHRONO-CAUSAL).

L-CHRONOS (V.01) simulates futures for *decisions inside* :meth:`~nyxara.nyx.brain.NyxBrain.think`
— which candidate answer to give. It has never run on a **tool call** or on **code being
written**. That was survivable while her brain was blind to tools. Now that
:mod:`nyxara.nyx.hands` can reach and :mod:`nyxara.nyx.author` can write, it is the gap that
matters most, and it is the one this module closes.

:class:`ConsequenceGate` sits **in front of execution**, not after it:

1. **A model of the action that is checkable, not guessed.** What paths it touches and whether
   they are inside the workspace; what git would actually do (``git diff --stat`` and, where
   the tool offers one, a real dry run); the tool's own registered ``risk`` and ``reversible``
   from the registry; and from those, a **reversibility class** — ``undoable``,
   ``recoverable``, or ``irreversible``.
2. **Bounded rollouts over her own system state**, through :mod:`nyxara.nyx.chronos` (and the
   simulators beneath it), ranked by **CVaR** — the mean of the worst outcomes, not the average
   one, because the average is exactly what hides a rare catastrophe.
3. **Her own restraint.** A bad tail on an irreversible action makes her *drop it* or take a
   **reversible variant** — a branch instead of a commit to main, a copy instead of an
   overwrite, a backup first. She does not ask permission to be careful; this is foresight, not
   escalation, and full autonomy is untouched.

Honest framing (the rule this repo keeps, see ``docs/CAPABILITIES.md``):

* **"Ten thousand timelines" means exactly the branch budget**, run sequentially, and it is
  reported as a number rather than implied as an ocean.
* **Five years is not simulable.** No world model here carries five years of dynamics, and one
  that claimed to would be lying. The horizon is bounded and covers **her own system state** —
  which files change, what git does, what the blast radius is — not the world.
* **There is no "100% optimal".** She reports tail risk and confidence, and when the world
  model has learned too little she says she is **blind** rather than emitting zeros as
  foresight. Abstention is a result.
* The reversibility class is derived from **checkable facts** (path scope, the registry's own
  flags, what git reports), so it is a measurement rather than an opinion — but it is a
  measurement of *this machine*, and a tool whose real effects are outside it (a network call)
  is classed by what it is registered as, honestly and no more.

Pure standard library. Every public method is fail-soft: on any error it degrades to a null
result rather than breaking a turn — and a gate that cannot see **fails closed on an
irreversible action**, which is the one place in this file that is not merely advisory.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["Reversibility", "Effect", "Foresight", "Verdict", "ConsequenceGate"]


class Reversibility:
    """How hard this would be to take back. Derived from checkable facts, not opinion."""

    UNDOABLE = "undoable"            # a snapshot/branch/ledger entry restores it exactly
    RECOVERABLE = "recoverable"      # restorable with effort — a copy exists, or git holds it
    IRREVERSIBLE = "irreversible"    # gone, or it has left this machine
    UNKNOWN = "unknown"              # she could not tell, which is treated as irreversible

    ORDER = (UNDOABLE, RECOVERABLE, IRREVERSIBLE, UNKNOWN)

    @staticmethod
    def worse(a: str, b: str) -> str:
        return a if Reversibility.ORDER.index(a) >= Reversibility.ORDER.index(b) else b


#: Tools whose whole purpose is to remove or overwrite something.
_DESTRUCTIVE = ("delete", "remove", "rm_", "unlink", "drop", "purge", "truncate", "overwrite")
#: Tools that send something outside this machine. What has left has left.
_OUTBOUND = ("push", "post", "send", "publish", "upload", "deploy", "http", "web_", "net_",
             "message", "email")


@dataclass
class Effect:
    """One checkable thing this action would do."""

    kind: str = ""               # path | git | outbound | resource
    detail: str = ""
    inside_workspace: bool = True
    reversibility: str = Reversibility.UNDOABLE

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "detail": self.detail,
                "inside_workspace": self.inside_workspace,
                "reversibility": self.reversibility}


@dataclass
class Foresight:
    """What the bounded rollouts saw — or an honest report of blindness."""

    ran: bool = False
    blind: bool = False
    branches: int = 0
    horizon: int = 0
    tail_risk: float = 0.0       # CVaR of the worst outcomes, not the mean of all of them
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"ran": self.ran, "blind": self.blind, "branches": self.branches,
                "horizon": self.horizon, "tail_risk": round(self.tail_risk, 4),
                "confidence": round(self.confidence, 4), "reason": self.reason}


@dataclass
class Verdict:
    """The gate's decision, and everything it was based on."""

    action: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    allowed: bool = True
    reversibility: str = Reversibility.UNDOABLE
    blast_radius: int = 0                    # paths outside the workspace it would touch
    effects: List[Effect] = field(default_factory=list)
    foresight: Foresight = field(default_factory=Foresight)
    variant: str = ""                        # a safer tool she would take instead
    variant_args: Dict[str, Any] = field(default_factory=dict)
    why: str = ""
    duration_ms: float = 0.0

    @property
    def took_a_safer_route(self) -> bool:
        return bool(self.variant)

    def render(self) -> str:
        lines = [f"{self.action}  →  {'proceed' if self.allowed else 'STOP'}",
                 f"  reversibility : {self.reversibility}",
                 f"  blast radius  : {self.blast_radius} path(s) outside the workspace"]
        for effect in self.effects:
            lines.append(f"  effect        : {effect.kind}: {effect.detail}")
        fore = self.foresight
        if fore.blind:
            lines.append(f"  foresight     : blind — {fore.reason}")
        elif fore.ran:
            lines.append(f"  foresight     : {fore.branches} branches, tail risk "
                         f"{fore.tail_risk:.2f}, confidence {fore.confidence:.2f}")
        else:
            lines.append(f"  foresight     : not run — {fore.reason}")
        if self.variant:
            lines.append(f"  safer route   : {self.variant} {self.variant_args or ''}")
        lines.append(f"  why           : {self.why}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "args": dict(self.args), "allowed": self.allowed,
                "reversibility": self.reversibility, "blast_radius": self.blast_radius,
                "effects": [e.to_dict() for e in self.effects],
                "foresight": self.foresight.to_dict(), "variant": self.variant,
                "variant_args": dict(self.variant_args),
                "took_a_safer_route": self.took_a_safer_route,
                "why": self.why, "duration_ms": round(self.duration_ms, 2)}


class ConsequenceGate:
    """Model the action, roll it forward, and step back from a bad irreversible tail."""

    def __init__(self, brain: Any = None, *, workspace: str = "",
                 tail_risk_ceiling: float = 0.5, branches: int = 64,
                 require_foresight_for_irreversible: bool = True,
                 prefer_reversible_variant: bool = True) -> None:
        self.brain = brain
        self.workspace = os.path.abspath(workspace or os.getcwd())
        self.tail_risk_ceiling = max(0.0, min(1.0, float(tail_risk_ceiling)))
        self.branches = max(2, int(branches))
        # The one non-advisory rule in this file: if she cannot see ahead and the action cannot
        # be taken back, she does not take it. Fail closed exactly where it matters.
        self.require_foresight_for_irreversible = bool(require_foresight_for_irreversible)
        self.prefer_reversible_variant = bool(prefer_reversible_variant)

        self.checked = 0
        self.stopped = 0
        self.rerouted = 0

    # ---- the one call ------------------------------------------------------- #
    def check(self, action: str, args: Optional[Dict[str, Any]] = None, *,
              registry: Any = None) -> Verdict:
        """Decide whether this action may run, and on what evidence."""
        out = Verdict(action=str(action or ""), args=dict(args or {}))
        t0 = time.perf_counter()
        try:
            if not out.action:
                out.allowed, out.why = False, "no action was named"
                return out
            self.checked += 1

            registry = registry if registry is not None else self._registry()
            out.effects = self._model(out.action, out.args, registry)
            out.reversibility = self._class_of(out.effects)
            # Paths only. An outbound effect leaves the machine but touches no file here, and
            # counting it as a "path outside the workspace" would be a wrong number in a field
            # whose whole job is to be a right one.
            out.blast_radius = sum(1 for e in out.effects
                                   if e.kind == "path" and not e.inside_workspace)
            out.foresight = self._roll(out.action, out.args)

            self._decide(out)
            return out
        except Exception as exc:  # noqa: BLE001 — a gate that breaks must not open
            out.allowed = False
            out.reversibility = Reversibility.UNKNOWN
            out.why = f"the gate itself failed ({type(exc).__name__}); it fails closed"
            self.stopped += 1
            return out
        finally:
            out.duration_ms = (time.perf_counter() - t0) * 1000.0

    def _registry(self) -> Any:
        for holder in (self.brain, getattr(self.brain, "core", None)):
            got = getattr(holder, "tools", None)
            if got is not None and hasattr(got, "get"):
                return got
        return None

    # ---- 1. a model of the action, from checkable facts --------------------- #
    def _model(self, action: str, args: Dict[str, Any], registry: Any) -> List[Effect]:
        effects: List[Effect] = []
        low = action.lower()

        spec = None
        if registry is not None:
            try:
                spec = registry.get(action)
            except Exception:  # noqa: BLE001
                spec = None

        # The registry's own flags are the most trustworthy statement about the tool, because
        # they were written where the tool was written.
        if spec is not None:
            reversible = bool(getattr(spec, "reversible", True))
            risk = getattr(getattr(spec, "risk", None), "label", "")
            effects.append(Effect(
                kind="registered", detail=f"risk={risk or 'unknown'}, "
                f"reversible={reversible}",
                reversibility=(Reversibility.UNDOABLE if reversible
                               else Reversibility.IRREVERSIBLE)))

        # Paths: scope is checkable, so it is checked rather than assumed.
        for value in self._path_like(args):
            resolved = os.path.abspath(value)
            inside = self._inside_workspace(resolved)
            destructive = any(token in low for token in _DESTRUCTIVE)
            effects.append(Effect(
                kind="path", detail=f"{'removes' if destructive else 'writes'} {resolved}",
                inside_workspace=inside,
                reversibility=(
                    Reversibility.IRREVERSIBLE if (destructive and not inside)
                    else Reversibility.RECOVERABLE if destructive
                    else Reversibility.UNDOABLE)))

        # Anything that leaves the machine cannot be taken back from here.
        if any(token in low for token in _OUTBOUND):
            effects.append(Effect(
                kind="outbound", detail="this leaves the machine; it cannot be recalled",
                inside_workspace=False, reversibility=Reversibility.IRREVERSIBLE))

        if low.startswith("git_"):
            effects.extend(self._git_effects(low, args))
        return effects

    @staticmethod
    def _path_like(args: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        for key, value in (args or {}).items():
            if not isinstance(value, str) or not value.strip():
                continue
            if key.lower() in ("path", "paths", "file", "filename", "dest", "target",
                               "directory", "dir", "cwd"):
                out.extend(p for p in value.split() if p)
        return out[:16]

    def _inside_workspace(self, resolved: str) -> bool:
        try:
            return os.path.commonpath([resolved, self.workspace]) == self.workspace
        except Exception:  # noqa: BLE001 — different drives, or an unresolvable path
            return False

    def _git_effects(self, low: str, args: Dict[str, Any]) -> List[Effect]:
        """What git would actually do — read from git, not imagined."""
        out: List[Effect] = []
        try:
            from nyxara.agency.git_tool import Git
            git = Git(repo=self.workspace)
            if not git.available():
                return out
            if low in ("git_commit", "git_add", "git_checkout"):
                stat = git.diff(stat=True)
                summary = (stat.stdout.strip().splitlines() or ["no changes"])[-1]
                out.append(Effect(kind="git", detail=f"working tree: {summary}",
                                  reversibility=Reversibility.UNDOABLE))
            if low == "git_push":
                out.append(Effect(
                    kind="git",
                    detail=f"publishes to {args.get('remote', 'origin')} — "
                           f"what has left the machine has left it",
                    inside_workspace=False,
                    reversibility=Reversibility.IRREVERSIBLE))
        except Exception:  # noqa: BLE001 — an unreadable repo simply contributes no effect
            return out
        return out

    @staticmethod
    def _class_of(effects: Sequence[Effect]) -> str:
        """The worst thing any modelled effect does — and UNKNOWN when nothing was modelled.

        Starting at ``UNDOABLE`` meant an action she could not model *at all* came out as
        "this can be taken back": ``rm -rf /some/unknown/path`` was classed undoable and
        allowed, because free text with no args produces no paths, no registry entry and no
        outbound token, so the loop had nothing to worsen. That is the same shape as handing
        an empty graph a free health bonus, in the one gate that decides whether she destroys
        something. An action she has modelled nothing about is UNKNOWN, and ``_decide`` already
        treats unknown as un-take-back-able.

        A **registered** tool always contributes a ``registered`` effect, so a read-only tool
        that touches nothing still lands here with something to read and stays undoable — the
        distinction is *modelled nothing*, not *does nothing*.
        """
        if not effects:
            return Reversibility.UNKNOWN
        worst = Reversibility.UNDOABLE
        for effect in effects:
            worst = Reversibility.worse(worst, effect.reversibility)
        return worst

    # ---- 2. bounded rollouts over her own system state ---------------------- #
    def _roll(self, action: str, args: Dict[str, Any]) -> Foresight:
        """Roll the action and its safer sibling forward. Blind is a result, not a zero."""
        out = Foresight()
        chronos = getattr(self.brain, "chronos", None)
        if chronos is None:
            out.reason = "no temporal matrix on this brain — she cannot see ahead at all"
            return out
        try:
            out.horizon = int(getattr(chronos, "horizon", 0))
            options = [_Option(action), _Option(f"skip:{action}")]
            futures = chronos.explore(options, branches=self.branches)
            out.branches = int(getattr(futures, "branches", 0))
            out.blind = bool(getattr(futures, "blind", False))
            out.ran = bool(getattr(futures, "ran", False))
            out.reason = str(getattr(futures, "reason", "") or "")
            out.confidence = float(getattr(futures, "confidence", 0.0))
            for option in (getattr(futures, "options", None) or []):
                if str(getattr(option, "option", "")) == action:
                    # CVaR: the mean of the worst outcomes. The *average* future is exactly
                    # what hides a rare catastrophe, which is the only kind that matters here.
                    out.tail_risk = float(getattr(option, "tail_risk", 0.0))
                    break
            return out
        except Exception:  # noqa: BLE001 — a failed rollout sees nothing; it does not crash
            out.reason = "the rollout failed and was abandoned"
            return out

    # ---- 3. her own restraint ------------------------------------------------ #
    def _decide(self, out: Verdict) -> None:
        irreversible = out.reversibility in (Reversibility.IRREVERSIBLE,
                                             Reversibility.UNKNOWN)
        # An action she modelled *nothing* about deserves its own words. "This cannot be taken
        # back and she cannot see ahead" describes a known-irreversible action; saying it about
        # something she simply did not recognise is a different claim, and a misleading one.
        if not out.effects:
            out.allowed = False
            out.why = ("I could not model this action at all — no registered tool, no path, "
                       "nothing outbound. Unmodelled is not the same as safe, so I do not act "
                       "on it")
            self.stopped += 1
            return
        bad_tail = out.foresight.ran and out.foresight.tail_risk > self.tail_risk_ceiling

        if not irreversible and not bad_tail:
            out.allowed = True
            out.why = (f"{out.reversibility}: this can be taken back, and nothing in the "
                       f"rollouts says otherwise")
            return

        if self.prefer_reversible_variant:
            variant, variant_args, note = self._safer(out)
            if variant:
                out.allowed = True
                out.variant, out.variant_args = variant, variant_args
                out.why = (f"{out.reversibility} as asked — taking the reversible route "
                           f"instead: {note}")
                self.rerouted += 1
                return

        if irreversible and out.foresight.blind and self.require_foresight_for_irreversible:
            out.allowed = False
            out.why = ("this cannot be taken back and she cannot see ahead "
                       f"({out.foresight.reason or 'no coverage'}) — she does not act blind "
                       "on something irreversible")
            self.stopped += 1
            return

        if irreversible and bad_tail:
            out.allowed = False
            out.why = (f"irreversible, and the worst futures are bad "
                       f"(tail risk {out.foresight.tail_risk:.2f} > "
                       f"{self.tail_risk_ceiling:.2f}) — she steps back from this herself")
            self.stopped += 1
            return

        if bad_tail:
            out.allowed = False
            out.why = (f"the worst futures are bad (tail risk "
                       f"{out.foresight.tail_risk:.2f} > {self.tail_risk_ceiling:.2f})")
            self.stopped += 1
            return

        # Irreversible, but either the rollouts looked and found nothing bad, or this
        # deployment has switched the fail-closed rule off. Say which — "there is a way back"
        # would be plainly false about an irreversible action.
        out.allowed = True
        if out.foresight.ran:
            out.why = (f"{out.reversibility}, and there is no way back — but the rollouts "
                       f"looked ({out.foresight.branches} branches, tail risk "
                       f"{out.foresight.tail_risk:.2f}) and found nothing against it")
        else:
            out.why = (f"{out.reversibility}, and she cannot see ahead — allowed only because "
                       f"this deployment has switched the fail-closed rule off")

    #: Branches whose history the repo treats as shared. Writing straight to one is the
    #: commonest irreversible-by-accident move there is.
    PROTECTED_BRANCHES = ("main", "master", "trunk", "release")

    def _safer(self, out: Verdict) -> Tuple[str, Dict[str, Any], str]:
        """A reversible variant of the same intent, when one genuinely exists.

        Only **real** substitutions are offered. Inventing a "safe version" of an action that
        has none is worse than stopping, because it looks like care and is not — so this
        returns nothing far more often than it returns something, on purpose.
        """
        action, args = out.action, dict(out.args)

        # Committing or pushing straight onto a shared branch. The reversible route is the one
        # this repo already uses by hand: branch first, then do the same thing on the branch.
        if action in ("git_commit", "git_push"):
            branch = self._current_branch()
            if branch and branch.lower() in self.PROTECTED_BRANCHES:
                safe = f"nyx/{int(time.time())}"
                return ("git_checkout", {"ref": safe, "create": True},
                        f"branch off {branch} onto {safe} first, so the shared history is "
                        f"never written to directly")
            return "", {}, ""     # on a side branch already, or no repo to read

        # Deleting something outside the workspace has no safe variant — only a stop.
        if any(token in action.lower() for token in _DESTRUCTIVE):
            paths = self._path_like(args)
            outside = [p for p in paths
                       if not self._inside_workspace(os.path.abspath(p))]
            if outside:
                return "", {}, ""
        return "", {}, ""

    def _current_branch(self) -> str:
        try:
            from nyxara.agency.git_tool import Git
            git = Git(repo=self.workspace)
            if not git.available():
                return ""
            got = git.current_branch()
            return got.stdout.strip() if got.ok else ""
        except Exception:  # noqa: BLE001 — an unreadable repo offers no safer route
            return ""

    # ---- the seam hands and author use --------------------------------------- #
    def guard(self, action: str, args: Optional[Dict[str, Any]] = None, *,
              registry: Any = None) -> Tuple[bool, Verdict]:
        """``(may_proceed, verdict)`` — the shape :mod:`nyxara.nyx.hands` calls."""
        verdict = self.check(action, args, registry=registry)
        return verdict.allowed, verdict

    # ---- introspection -------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        try:
            return {
                "checked": self.checked, "stopped": self.stopped, "rerouted": self.rerouted,
                "workspace": self.workspace,
                "branches": self.branches,
                "tail_risk_ceiling": self.tail_risk_ceiling,
                "fails_closed_on_irreversible": self.require_foresight_for_irreversible,
                "classes": list(Reversibility.ORDER),
                "note": ("branch count is exactly the budget, run sequentially — not an ocean "
                         "of timelines; the horizon covers her own system state, not five "
                         "years of the world; and there is no '100% optimal' — she reports "
                         "tail risk and confidence, and calls herself blind rather than "
                         "emitting zeros as foresight"),
            }
        except Exception:  # noqa: BLE001
            return {}


@dataclass
class _Option:
    """A rollout label. ``chronos.explore`` reads ``.source``, as the specialists do."""

    source: str = ""
