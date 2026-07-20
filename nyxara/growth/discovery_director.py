"""NYXARA · growth/discovery_director.py — the Discovery Director (Level 10f).

The Autonomous Scientist can invent a real law from a domain her curiosity chose. The
**Discovery Director** is the layer above: on every idle beat it *decides what act of discovery is
worth the most right now*, and does that one — replacing the old fixed ``tick % 7`` experiment
rotation with a single, principled, self-directed scheduler. **No LLM anywhere.**

Each beat it scores the discovery actions available to her and runs the highest-value one:

* **EXPERIMENT** — run her own experiment in the science she is *least competent at* (automatic
  cross-science curriculum) and discover its governing law, with adversarial rival falsification.
  Value = learning-progress (1 − competence) — she is pulled to where she can learn the most.
* **DYNAMICS** — from a trajectory she rolls out (an epidemic, an oscillator) recover the governing
  *equations of motion* ``dx/dt = f(x)`` (SINDy). Value rises the longer she has not looked.
* **INVARIANT** — search a trajectory for a *conserved quantity* nobody defined for her
  (Noether-style). The deepest form of inventing a law of nature.
* **UNIFY** — compound the laws she already holds into a deeper **meta-law** that subsumes several
  (e.g. pendulum + spring → the simple-harmonic form). Value rises with the size of her law tower.
* **META** — invent a brand-new theory/optimization via meta-research and fold it into her beliefs.

Everything degrades gracefully: with no engine it does nothing (honest no-op); with only an engine it
still experiments, discovers dynamics/invariants and unifies. It touches no source, no weights, no
gate, and no external world — every experiment is an in-memory simulation she runs herself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["DiscoveryDirector", "DiscoveryAction", "DirectorBeat"]


class DiscoveryAction(str, Enum):
    """The kinds of discovery act she can choose between on an idle beat."""
    EXPERIMENT = "experiment"     # run an experiment in a chosen domain → discover its law
    DYNAMICS = "dynamics"         # recover equations of motion from a trajectory (SINDy)
    INVARIANT = "invariant"       # discover a conserved quantity (Noether-style)
    UNIFY = "unify"               # compound held laws into a deeper meta-law
    META = "meta"                 # invent a new theory via meta-research


@dataclass
class DirectorBeat:
    """The record of one beat: what she chose, why, and what it produced."""
    index: int
    action: DiscoveryAction
    value: float
    domain: str = ""
    outcome: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "action": self.action.value,
            "value": round(float(self.value), 4),
            "domain": self.domain or None,
            "outcome": self.outcome,
        }


class DiscoveryDirector:
    """Choose and run the highest-value act of discovery each beat — a self-directed scientist's day.

    Parameters
    ----------
    engine:       a :class:`~nyxara.growth.law_discovery.LawDiscoveryEngine` (required to do anything).
    scientist:    an :class:`~nyxara.growth.autonomous_scientist.AutonomousScientist` — when present,
                  EXPERIMENT runs through it so a discovery updates her beliefs / world model / skills.
    competence:   a CompetenceLedger — read to weight domains by learning progress, written after each
                  experiment so a mastered science stops being chosen.
    meta_researcher: enables the META action (optional).
    oversight:    an object exposing ``gate() -> bool`` — a paused / scrammed mind discovers nothing of
                  its own accord (optional; open when absent).
    """

    def __init__(self, *, engine: Any, scientist: Any = None, competence: Any = None,
                 meta_researcher: Any = None, oversight: Any = None,
                 min_laws_to_unify: int = 2) -> None:
        self.engine = engine
        self.scientist = scientist
        self.competence = competence
        self.meta_researcher = meta_researcher
        self.oversight = oversight
        self.min_laws_to_unify = max(2, int(min_laws_to_unify))
        self._beats: List[DirectorBeat] = []
        self._beat_n = 0
        self._since_dynamics = 99         # bias her to look at dynamics/invariants early, then rarely
        self._since_invariant = 99
        self._since_unify = 0
        self._since_meta = 0
        # cumulative, honest tallies for status / API
        self.laws_discovered = 0
        self.dynamics_found = 0
        self.invariants_found = 0
        self.unifications = 0
        self.rivals_beaten = 0
        self.meta_inventions = 0

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #
    def run(self, beats: int = 3) -> List[DirectorBeat]:
        """Run ``beats`` idle beats; return the beats executed (may be fewer if oversight closes)."""
        out: List[DirectorBeat] = []
        for _ in range(max(1, int(beats))):
            beat = self.step()
            if beat is None:
                break
            out.append(beat)
        return out

    def step(self) -> Optional[DirectorBeat]:
        """Score the available discovery actions, run the highest-value one, and record the beat.

        Returns ``None`` only when oversight is closed or there is no engine to discover with."""
        if self.engine is None:
            return None
        if self.oversight is not None:
            try:
                if not self.oversight.gate():
                    return None
            except Exception:  # noqa: BLE001 — a broken gate should not silently run experiments
                return None
        # recency grows every beat so the deep, occasional acts (dynamics/invariant/unify/meta)
        # resurface over time; the chosen action's handler resets its own counter to 0.
        self._since_dynamics += 1
        self._since_invariant += 1
        self._since_unify += 1
        self._since_meta += 1
        scored = self._score_actions()
        if not scored:
            return None
        value, action, domain = scored[0]
        beat = DirectorBeat(index=self._beat_n, action=action, value=value, domain=domain)
        self._beat_n += 1
        try:
            beat.outcome = self._run_action(action, domain)
        except Exception as exc:  # noqa: BLE001 — a failed beat is data, not a crash
            beat.outcome = {"error": str(exc)}
        self._beats.append(beat)
        return beat

    def all_beats(self) -> List[DirectorBeat]:
        return list(self._beats)

    def summary(self) -> Dict[str, Any]:
        """An honest snapshot of the director's cumulative discovery record — for status / API."""
        try:
            laws_held = len(self.engine.known_laws())
        except Exception:  # noqa: BLE001
            laws_held = 0
        return {
            "beats_run": len(self._beats),
            "laws_held": laws_held,
            "laws_discovered": self.laws_discovered,
            "dynamics_found": self.dynamics_found,
            "invariants_found": self.invariants_found,
            "unifications": self.unifications,
            "rivals_beaten": self.rivals_beaten,
            "meta_inventions": self.meta_inventions,
        }

    # ---------------------------------------------------------------------- #
    # Action scoring — value-of-information × novelty × learning-progress
    # ---------------------------------------------------------------------- #
    def _score_actions(self) -> List[Tuple[float, DiscoveryAction, str]]:
        """Score every discovery action currently worth doing; highest first. Every term is her own
        measured signal — competence, tower size, recency — never an LLM's opinion."""
        cands: List[Tuple[float, DiscoveryAction, str]] = []

        # EXPERIMENT — pick the domain she is least competent at (highest learning progress).
        domain, gain = self._best_domain()
        if domain is not None:
            cands.append((1.0 * gain, DiscoveryAction.EXPERIMENT, domain))

        # DYNAMICS / INVARIANT — genuinely deep, so worth doing but not every beat; value grows with
        # how long since she last did one (curiosity about her own blind spots).
        cands.append((0.25 + 0.06 * min(10, self._since_dynamics), DiscoveryAction.DYNAMICS, "epidemiology"))
        cands.append((0.20 + 0.06 * min(10, self._since_invariant), DiscoveryAction.INVARIANT, "mechanics"))

        # UNIFY — compound discoveries into deeper theory once her tower is rich enough; value grows
        # with tower size and time since the last unification.
        try:
            n_laws = len(self.engine.known_laws())
        except Exception:  # noqa: BLE001
            n_laws = 0
        if n_laws >= self.min_laws_to_unify:
            cands.append((0.15 + 0.03 * min(12, n_laws) + 0.05 * min(6, self._since_unify),
                          DiscoveryAction.UNIFY, ""))

        # META — invent a new theory when a meta-researcher is wired; throttled.
        if self.meta_researcher is not None:
            cands.append((0.30 + 0.04 * min(8, self._since_meta), DiscoveryAction.META, ""))

        cands.sort(key=lambda c: c[0], reverse=True)
        return cands

    def _best_domain(self) -> Tuple[Optional[str], float]:
        """The domain with the highest learning progress (1 − competence); round-robin when the
        ledger is silent so she works across all sciences rather than fixating on one."""
        try:
            domains = list(self.engine.domains())
        except Exception:  # noqa: BLE001
            return None, 0.0
        if not domains:
            return None, 0.0
        if self.competence is None:
            idx = self._beat_n % len(domains)
            return domains[idx], 1.0
        best_dom, best_gain = domains[0], -1.0
        for d in domains:
            gain = self._learn_gain(d)
            if gain > best_gain:
                best_dom, best_gain = d, gain
        return best_dom, max(0.1, best_gain)

    def _learn_gain(self, domain: str) -> float:
        """Learning-progress multiplier for a domain — high where she is least competent."""
        if self.competence is None:
            return 1.0
        try:
            st = self.competence.stat(domain)
            level = float(getattr(st, "level", 0.5)) if st is not None else 0.5
            return max(0.1, 1.0 - level)
        except Exception:  # noqa: BLE001 — competence is advisory, never fatal
            return 1.0

    # ---------------------------------------------------------------------- #
    # Action execution
    # ---------------------------------------------------------------------- #
    def _run_action(self, action: DiscoveryAction, domain: str) -> Dict[str, Any]:
        if action is DiscoveryAction.EXPERIMENT:
            return self._do_experiment(domain)
        if action is DiscoveryAction.DYNAMICS:
            return self._do_dynamics(domain)
        if action is DiscoveryAction.INVARIANT:
            return self._do_invariant(domain)
        if action is DiscoveryAction.UNIFY:
            return self._do_unify()
        if action is DiscoveryAction.META:
            return self._do_meta(domain)
        return {}

    def _do_experiment(self, domain: str) -> Dict[str, Any]:
        """Run one real experiment in ``domain`` and discover its law. Routed through the Autonomous
        Scientist when present (so beliefs / world model / skills update); else straight to the
        engine. Records learning progress so a mastered domain stops being chosen."""
        if self.scientist is not None and hasattr(self.scientist, "discover_domain"):
            cycle = self.scientist.discover_domain(domain)
            result = cycle.discovery or {}
        else:
            result = self.engine.discover_by_domain(domain)
        verdict = result.get("verdict")
        if verdict == "corroborated":
            self.laws_discovered += 1
            if result.get("rival_beaten"):
                self.rivals_beaten += 1
            self._record_competence(domain, float(result.get("score", 0.7) or 0.7))
        else:
            self._record_competence(domain, 0.25)
        return {"domain": domain, "verdict": verdict, "law": result.get("law"),
                "extrapolation_r2": result.get("extrapolation_r2"),
                "rival_beaten": result.get("rival_beaten")}

    def _do_dynamics(self, domain: str) -> Dict[str, Any]:
        """Roll out a trajectory and recover its equations of motion (SINDy). Uses an epidemic (SIR)
        by default — discovering the dynamics of a disease she simulated — and falls back honestly."""
        self._since_dynamics = 0
        traj, names, dt = self._trajectory(domain)
        if traj is None:
            return {"discovered": 0, "reason": "no trajectory source"}
        rep = self.engine.discover_dynamics(traj, dt=dt, var_names=names)
        n = len(getattr(rep, "discoveries", []))
        self.dynamics_found += n
        return {"discovered": n, "reason": getattr(rep, "reason", ""),
                "laws": [d.law.expression[:80] for d in getattr(rep, "discoveries", [])[:3]]}

    def _do_invariant(self, domain: str) -> Dict[str, Any]:
        """Search a family of trajectories for a conserved quantity nobody defined for her."""
        self._since_invariant = 0
        trajs = self._invariant_trajectories(domain)
        if not trajs:
            return {"discovered": 0, "reason": "no trajectory source"}
        rep = self.engine.discover_invariant(trajs, var_names=["x", "v"])
        n = len(getattr(rep, "discoveries", []))
        self.invariants_found += n
        return {"discovered": n, "reason": getattr(rep, "reason", ""),
                "laws": [d.law.expression[:80] for d in getattr(rep, "discoveries", [])[:2]]}

    def _do_unify(self) -> Dict[str, Any]:
        """Compound her held laws into deeper meta-laws that subsume several discoveries."""
        self._since_unify = 0
        try:
            unis = self.engine.unify_laws() or []
        except Exception as exc:  # noqa: BLE001
            return {"unifications": 0, "error": str(exc)}
        self.unifications += len(unis)
        return {"unifications": len(unis),
                "forms": [u.get("general_form", u.get("form", "")) for u in unis[:3]]}

    def _do_meta(self, domain: str) -> Dict[str, Any]:
        """Invent a brand-new theory/optimization via meta-research and fold it into beliefs."""
        self._since_meta = 0
        topic = domain or "intelligence"
        if self.scientist is not None and hasattr(self.scientist, "meta_discover"):
            d = self.scientist.meta_discover(topic)
        else:
            rep = self.meta_researcher.run(topic)
            d = rep.to_dict() if hasattr(rep, "to_dict") else {"topic": topic}
        validated = int(d.get("validated", d.get("beliefs_held", 0)) or 0)
        self.meta_inventions += max(0, validated)
        return {"topic": topic, "meta": d.get("validated", validated)}

    # ---------------------------------------------------------------------- #
    # Trajectory sources for DYNAMICS / INVARIANT (her own simulations)
    # ---------------------------------------------------------------------- #
    def _trajectory(self, domain: str) -> Tuple[Any, Optional[List[str]], float]:
        """A state trajectory she rolls out herself. Epidemiology → SIR ``[S, I, R]``; else a damped
        oscillator ``[x, v]`` — a self-contained ODE, no external data."""
        if domain == "epidemiology":
            try:
                from nyxara.sim.epi_world import EpidemicWorld
                ob = EpidemicWorld(beta=0.6, gamma=0.2, dt=0.25).simulate(steps=200)
                traj = list(zip(ob.S, ob.I, ob.R))[::3]
                if len(traj) >= 8:
                    return traj, ["S", "I", "R"], 0.25 * 3
            except Exception:  # noqa: BLE001
                pass
        return self._oscillator_trajectory(), ["x", "v"], 0.05 * 4

    @staticmethod
    def _oscillator_trajectory() -> List[Tuple[float, float]]:
        """A simple-harmonic oscillator ``x'' = −x`` integrated to a ``[x, v]`` trajectory (stdlib)."""
        x, v, dt = 1.0, 0.0, 0.05
        traj: List[Tuple[float, float]] = []
        for step in range(800):
            v += -x * dt
            x += v * dt
            if step % 4 == 0:
                traj.append((x, v))
        return traj

    def _invariant_trajectories(self, domain: str) -> List[List[Tuple[float, float]]]:
        """A few oscillator trajectories at different energies — a conserved quantity (energy
        ``½v² + ½x²``) is the minimum-variance function of state across each."""
        trajs: List[List[Tuple[float, float]]] = []
        for amp in (0.5, 1.0, 1.5):
            x, v, dt = amp, 0.0, 0.05
            one: List[Tuple[float, float]] = []
            for step in range(400):
                v += -x * dt
                x += v * dt
                if step % 4 == 0:
                    one.append((x, v))
            trajs.append(one)
        return trajs

    def _record_competence(self, domain: str, score: float) -> None:
        if self.competence is None:
            return
        try:
            self.competence.record(domain, score)
        except Exception:  # noqa: BLE001 — measurement is advisory, never fatal
            pass


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.growth.law_discovery import LawDiscoveryEngine
    from nyxara.growth.autonomous_scientist import AutonomousScientist

    print("=" * 70)
    print("NYXARA discovery-director self-test")
    print("=" * 70)

    engine = LawDiscoveryEngine()
    scientist = AutonomousScientist(discovery_engine=engine)
    director = DiscoveryDirector(engine=engine, scientist=scientist)
    beats = director.run(beats=12)
    for b in beats:
        d = b.to_dict()
        print(f"[{d['index']:2d}] {d['action']:11s} v={d['value']:.3f} "
              f"{str(d['domain'] or ''):14s} -> {str(d['outcome'])[:60]}")
    s = director.summary()
    print(f"\nlaws discovered={s['laws_discovered']}  dynamics={s['dynamics_found']}  "
          f"invariants={s['invariants_found']}  unifications={s['unifications']}  "
          f"rivals_beaten={s['rivals_beaten']}")
    assert s["laws_discovered"] >= 2, "director did not drive real experiments"
    assert len(beats) == 12, "director should run every requested beat"
    assert any(b.action is DiscoveryAction.EXPERIMENT for b in beats)
    print("\nALL SELF-TESTS PASSED ✓")
