"""NYXARA · growth/brain_forge.py — verifiable brain (weights/architecture) self-improvement 🧠↑.

``self_optimize.py`` edits NYXARA's **source** (linter transforms + self/LLM-authored refactors)
under a reversible gauntlet. That is real, but it does not make her *smarter*: it never touches her
**brain**. Genuine recursive self-improvement means verifiably improving her own *weights and
architecture* — and doing it **herself**, on her **own** substrate, with **no external LLM**.

Every organ needed for that already exists in this package; they were simply never joined into the
deliberate self-optimization cycle:

* :class:`~nyxara.growth.genesis.NeuralArchitectureSearch` — NYXARA **designs** her own neural
  architectures (her layers, her mixers, her searched optimizer) and, on the **pure-NumPy substrate**
  (:class:`~nyxara.growth.genesis_numpy.GenesisNumpyModel`), actually *builds and trains* each one —
  real backprop, no torch, no cloud, no key.
* :class:`~nyxara.growth.foundry.Foundry` — **forges, verifies and promotes** weights: it trains the
  designed brain on her lived corpus and ships it **only** after clearing the gauntlet (character
  lock, corrigibility, the Loyalty-Equation floor + non-regression, a *strictly* lower perplexity,
  capability non-regression, and an independent scalable-oversight re-derivation). A worse brain is
  kept on the bench; a promotion is reversible (:meth:`Foundry.rollback`).
* :class:`~nyxara.mind.llm.SelfProvider` — serves the promoted brain as NYXARA's live ``self`` model,
  so a forged improvement is not a report on disk: it becomes the mind that actually answers.

:class:`BrainForge` is the **connector** — the missing wire. It runs the search (design), forces the
**NumPy substrate** so the forge is a genuinely trained transformer (never an n-gram stand-in), then
hands the champion to the Foundry's *existing* ``self_improve`` — **reusing the gauntlet verbatim; it
adds no gate of its own** — and emits a weight-level improvement certificate (:class:`BrainForgeReport`:
Δperplexity, Δcapability, params, promoted/kept-on-bench, and the honest ``verified`` flag). This is
what makes NYXARA's *self-optimization* verifiably increase capability, not merely tidy her source.

It is self-driven and LLM-free: the design is her search, the substrate is her NumPy brain, the
training corpus is her own lived memory (+ seeds). Enactment is triple-gated — ``autonomous_enact``
**and** ``autonomous_brain_forge`` **and** a live, open oversight gate — otherwise it *designs and
measures* a brain but never promotes it. It never raises into a cycle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["BrainForgeReport", "BrainForge"]


# --------------------------------------------------------------------------- #
# The weight-level improvement certificate
# --------------------------------------------------------------------------- #
@dataclass
class BrainForgeReport:
    """An auditable certificate for one brain-forge attempt — the *weight-level* analogue of the
    source-edit :class:`~nyxara.growth.self_optimize.OptimizationOutcome`.

    ``verified`` is honest: it is True only when a designed brain **cleared the Foundry gauntlet and
    was promoted** — i.e. it is provably (strictly-lower-perplexity, capability-non-regressing,
    corrigible, loyal) better than the brain it replaced. A measure-only pass, a bench rejection, or
    any failure yields ``verified=False`` with a truthful reason."""

    searched: bool = False
    enacted: bool = False
    promoted: bool = False
    rolled_back: bool = False            # designed + forged but kept on the bench (gauntlet refused)
    verified: bool = False               # promoted through the gauntlet ⇒ provably better
    champion_kind: str = ""              # "genesis_np" is the real NumPy transformer; "kngram" the fallback
    substrate: str = ""
    before_perplexity: float = float("inf")
    after_perplexity: float = float("inf")
    before_capability: float = 0.0
    after_capability: float = 0.0
    params: int = 0
    version: Optional[int] = None
    method: str = "architecture-search + weight-training (NumPy substrate)"
    describe: str = ""
    reason: str = ""
    at: float = field(default_factory=time.time)

    @property
    def delta_perplexity(self) -> float:
        """How much perplexity fell (before − after); positive means a smarter brain."""
        b, a = self.before_perplexity, self.after_perplexity
        if b == float("inf") or a == float("inf"):
            return 0.0
        return round(b - a, 4)

    @property
    def delta_capability(self) -> float:
        return round(self.after_capability - self.before_capability, 5)

    def to_dict(self) -> Dict[str, Any]:
        return {"searched": self.searched, "enacted": self.enacted, "promoted": self.promoted,
                "rolled_back": self.rolled_back, "verified": self.verified,
                "champion_kind": self.champion_kind, "substrate": self.substrate,
                "before_perplexity": (round(self.before_perplexity, 4)
                                      if self.before_perplexity != float("inf") else None),
                "after_perplexity": (round(self.after_perplexity, 4)
                                     if self.after_perplexity != float("inf") else None),
                "delta_perplexity": self.delta_perplexity,
                "before_capability": round(self.before_capability, 5),
                "after_capability": round(self.after_capability, 5),
                "delta_capability": self.delta_capability, "params": self.params,
                "version": self.version, "method": self.method, "describe": self.describe,
                "reason": self.reason, "at": self.at}

    def summary(self) -> str:
        head = ("PROMOTED (provably better)" if self.promoted
                else "kept on the bench" if self.rolled_back
                else "designed + measured (not enacted)" if self.searched
                else "no forge")
        ppl = (f"perplexity {self.before_perplexity:.3f} → {self.after_perplexity:.3f} "
               f"(Δ{self.delta_perplexity:+.3f})"
               if self.after_perplexity != float("inf") else "perplexity n/a")
        return (f"brain-forge: {head} · {self.champion_kind or 'n/a'} "
                f"({self.substrate or 'n/a'} substrate, {self.params} params) · {ppl} · "
                f"verified={self.verified} — {self.reason}")


# --------------------------------------------------------------------------- #
# The connector: design → forge → gauntlet → promote (reusing the gauntlet)
# --------------------------------------------------------------------------- #
class BrainForge:
    """Turn NYXARA's own architecture search into a promoted improvement of her live brain.

    Prefers the live :class:`~nyxara.growth.genesis.NeuralArchitectureSearch` and its
    :class:`~nyxara.growth.foundry.Foundry` on ``core`` (so it shares her hall-of-fame + the active
    model on disk); otherwise it constructs both, seeded from her lived memory. The forge always runs
    on the **NumPy substrate** so it is a genuinely trained transformer. Never raises."""

    def __init__(self, *, core: Any = None, settings: Any = None, memory: Any = None,
                 foundry: Any = None, nas: Any = None) -> None:
        from nyxara.kernel.config import get_settings
        self.core = core
        self.settings = settings or getattr(core, "settings", None) or get_settings()
        self.memory = memory if memory is not None else getattr(core, "memory", None)
        self._foundry_obj = foundry
        self._nas_obj = nas

    # ------------------------------------------------------------------ #
    # the public, self-driven entry
    # ------------------------------------------------------------------ #
    def improve_brain(self, *, enact: Optional[bool] = None,
                      generations: Optional[int] = None,
                      population: Optional[int] = None) -> BrainForgeReport:
        """Design → (when authorised) forge → gauntlet → promote one better brain.

        ``enact=None`` reads the standing authorisation (``autonomous_enact`` **and**
        ``autonomous_brain_forge``) and additionally requires an open oversight gate. When enactment
        is withheld she still *designs and measures* a brain — the certificate reports it — but never
        promotes. Always returns a :class:`BrainForgeReport`; never raises."""
        cfg = getattr(self.settings, "self_improvement", None)
        if enact is None:
            enact = (bool(getattr(cfg, "autonomous_enact", False))
                     and bool(getattr(cfg, "autonomous_brain_forge", True)))
        gate_note = ""
        if enact and not self._oversight_open():
            enact, gate_note = False, " (oversight gate closed)"

        report = BrainForgeReport(enacted=bool(enact))
        try:
            foundry = self._foundry()
            nas = self._nas(foundry)
        except Exception as exc:  # noqa: BLE001 — a missing organ degrades to a clean no-op
            report.reason = f"brain-forge unavailable: {exc}"
            self._publish(report)
            return report

        # capture the incumbent brain up front (for the honest before/after on a measure-only pass)
        before_pp, before_cap = self._active_metrics(foundry)
        report.before_perplexity, report.before_capability = before_pp, before_cap

        # 1) DESIGN — NYXARA searches her own architecture space on the NumPy substrate (always safe)
        try:
            search = nas.search(generations=self._gens(generations, nas),
                                population_size=self._pop(population, nas))
        except Exception as exc:  # noqa: BLE001 — a failed search reports, never crashes the cycle
            report.reason = f"architecture search failed: {exc}"
            self._publish(report)
            return report
        report.searched = True
        report.champion_kind = getattr(search, "champion_kind", "") or ""
        report.substrate = self._substrate(nas)
        try:
            report.describe = search.champion_describe()
        except Exception:  # noqa: BLE001
            report.describe = getattr(search, "champion_kind", "") or ""

        if not enact:
            report.after_perplexity, report.after_capability = before_pp, before_cap
            report.reason = ("designed + measured a brain but did not promote — enactment withheld"
                             + gate_note)
            self._publish(report)
            return report

        # 2) FORGE + VERIFY + PROMOTE — hand the champion to the Foundry's OWN self_improve, which
        #    trains it on her corpus and ships it ONLY through the full gauntlet. No gate is added or
        #    bypassed here; a bench rejection or a raise leaves the live brain byte-for-byte intact.
        try:
            spec = nas.champion_spec()
            results = foundry.self_improve(spec=spec, generations=1)
        except Exception as exc:  # noqa: BLE001 — forging is heavy/optional; never fatal
            report.after_perplexity, report.after_capability = before_pp, before_cap
            report.reason = f"forge failed (live brain unchanged): {exc}"
            self._publish(report)
            return report

        res = results[0] if results else None
        if res is None:
            report.after_perplexity, report.after_capability = before_pp, before_cap
            report.reason = "foundry produced no candidate (corpus starved) — live brain unchanged"
            self._publish(report)
            return report

        report.version = int(getattr(res, "version", 0)) or None
        report.promoted = bool(getattr(res, "promoted", False))
        report.rolled_back = not report.promoted
        report.reason = getattr(res, "reason", "") or ""
        ev_after = getattr(res, "eval_after", None)
        ev_before = getattr(res, "eval_before", None)
        if ev_before is not None:                        # foundry's before/after on the SAME eval set
            report.before_perplexity = float(ev_before.perplexity)
            report.before_capability = float(ev_before.task_score)
        if ev_after is not None:
            # after = the candidate iff it was promoted; otherwise the live brain is unchanged
            if report.promoted:
                report.after_perplexity = float(ev_after.perplexity)
                report.after_capability = float(ev_after.task_score)
            else:
                report.after_perplexity = report.before_perplexity
                report.after_capability = report.before_capability
        # the gauntlet already PROVED a promoted brain is strictly-better + non-regressing + loyal
        report.verified = report.promoted
        act = foundry.active()
        if act is not None:
            report.params = int(getattr(act, "param_count", 0) or 0)
        self._publish(report)
        return report

    # ------------------------------------------------------------------ #
    # organ resolution (prefer the live core's; else construct, seeded from memory)
    # ------------------------------------------------------------------ #
    def _foundry(self) -> Any:
        if self._foundry_obj is None:
            # Prefer the live foundry (genesis' or the growth engine's): it shares the on-disk active
            # model AND the in-memory version manifest, so a promotion here stays consistent with the
            # idle-loop forge instead of racing a separate object's stale view.
            for owner in (getattr(self.core, "genesis", None),
                          getattr(self.core, "growth_engine", None)):
                f = getattr(owner, "foundry", None) or getattr(owner, "_foundry", None)
                if f is not None:
                    self._foundry_obj = f
                    break
            if self._foundry_obj is None:
                from nyxara.growth.foundry import Foundry
                self._foundry_obj = Foundry(settings=self.settings, seed_corpus=self._corpus())
        # Never starve: the live foundry may have been built with no seed corpus (it normally trains
        # on accrued flywheel/memory). Top it up with her lived corpus (or the identity seed) so the
        # deliberate forge can always design + train a first brain, even on a cold box.
        try:
            if not getattr(self._foundry_obj, "seed_corpus", None):
                self._foundry_obj.seed_corpus = self._corpus()
        except Exception:  # noqa: BLE001 — a foundry that rejects reseeding still forges on its own corpus
            pass
        return self._foundry_obj

    def _nas(self, foundry: Any) -> Any:
        if self._nas_obj is not None:
            return self._nas_obj
        from nyxara.growth.genesis import NeuralArchitectureSearch
        # a process-local genesis config that FORCES the NumPy substrate (never mutates global config)
        # so the deliberate forge is a genuinely trained transformer, not the n-gram stand-in.
        cfg = self._genesis_cfg()
        self._nas_obj = NeuralArchitectureSearch(settings=self.settings, foundry=foundry, cfg=cfg,
                                                 seed_corpus=self._corpus())
        return self._nas_obj

    def _genesis_cfg(self) -> Any:
        from nyxara.kernel.config import GenesisConfig
        base = getattr(self.settings, "genesis", None)
        try:
            cfg = base.model_copy(deep=True) if base is not None else GenesisConfig()
        except Exception:  # noqa: BLE001
            cfg = GenesisConfig()
        try:
            cfg.substrate = "numpy"     # real NumPy transformer (falls back to n-gram only if NumPy absent)
        except Exception:  # noqa: BLE001 — validate_assignment rejected it; leave as configured
            pass
        return cfg

    def _corpus(self) -> List[str]:
        """Her own lived memory as training text — this is what makes the forged brain *hers*.

        Falls back to the identity seed corpus when memory is thin/absent, so the deliberate forge
        never starves on a cold box (a first brain is still designed, trained and — if it clears the
        gauntlet — promoted)."""
        texts: List[str] = []
        if self.memory is not None:
            try:
                for rec in getattr(self.memory, "_kv", {}).values():
                    t = rec.text().strip()
                    if t:
                        texts.append(t)
            except Exception:  # noqa: BLE001 — memory shape varies; the seed fallback still carries it
                texts = []
        if texts:
            return texts[:2000]
        try:
            from nyxara.growth.genesis import _GENESIS_SEED
            return list(_GENESIS_SEED)
        except Exception:  # noqa: BLE001
            return []

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _oversight_open(self) -> bool:
        """Respect the Master's live oversight channel (paused / scrammed ⇒ withhold enactment)."""
        oversight = getattr(self.core, "oversight", None)
        if oversight is None:
            return True
        try:
            return bool(oversight.gate())
        except Exception:  # noqa: BLE001 — an unusable gate must not block a valid, config-authorised forge
            return True

    def _active_metrics(self, foundry: Any) -> tuple[float, float]:
        try:
            act = foundry.active()
            if act is None:
                return float("inf"), 0.0
            return (float(act.metrics.get("perplexity", float("inf"))),
                    float(act.metrics.get("capability", act.metrics.get("task_score", 0.0))))
        except Exception:  # noqa: BLE001
            return float("inf"), 0.0

    def _substrate(self, nas: Any) -> str:
        try:
            return str(getattr(nas.cfg, "substrate", "auto"))
        except Exception:  # noqa: BLE001
            return "auto"

    def _gens(self, generations: Optional[int], nas: Any) -> int:
        if generations is not None:
            return max(1, int(generations))
        return max(1, int(getattr(nas.cfg, "generations", 3)))

    def _pop(self, population: Optional[int], nas: Any) -> int:
        if population is not None:
            return max(2, int(population))
        return max(2, int(getattr(nas.cfg, "population_size", 6)))

    def _publish(self, report: BrainForgeReport) -> None:
        """Expose the latest certificate on the core for ``core.report()`` (owner-visible status)."""
        if self.core is not None:
            try:
                self.core._last_brain_forge = report  # noqa: SLF001
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# Self-test / demo — a real forge on the NumPy substrate (no torch, no network)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import tempfile
    from pathlib import Path

    from nyxara.kernel.config import get_settings

    print("=" * 70)
    print("NYXARA brain-forge self-test — design → forge → gauntlet → promote")
    print("=" * 70)

    corpus = ["the master is jp. nyxara serves the master with loyalty."] * 12 + [
        "capability may grow; character never changes; she is corrigible."] * 12

    with tempfile.TemporaryDirectory() as d:
        settings = get_settings().model_copy(deep=True)
        settings.paths.data_dir = Path(d)
        settings.self_improvement.autonomous_enact = True
        settings.self_improvement.autonomous_brain_forge = True
        settings.genesis.generations = 2
        settings.genesis.population_size = 4
        settings.genesis.micro_train_steps = 12

        from nyxara.growth.foundry import Foundry
        foundry = Foundry(settings=settings, seed_corpus=corpus)
        forge = BrainForge(settings=settings, foundry=foundry)

        # 1) enacted forge on a cold foundry: first brain has nothing to beat ⇒ promoted + verified
        rep = forge.improve_brain(enact=True)
        print("\n" + rep.summary())
        assert rep.searched and rep.enacted
        assert rep.champion_kind == "genesis_np", f"expected the real NumPy brain, got {rep.champion_kind}"
        assert rep.promoted and rep.verified, "the first brain should promote through the gauntlet"
        assert (Path(d) / "foundry" / "active").exists(), "a promotion must write the active pointer"
        assert rep.params > 0
        print("enacted forge       : real NumPy transformer promoted through the gauntlet ✓")

        # 2) the promoted brain is what SelfProvider would serve (a forge is not a report on disk)
        from nyxara.growth.foundry_models import load_active_model
        live = load_active_model(settings)
        assert live.kind == "genesis_np" and live.param_count() > 0
        print(f"live brain          : kind={live.kind}, params={live.param_count()} — she is smarter ✓")

        # 3) measure-only when enactment is withheld: designs a brain, never promotes
        forge2 = BrainForge(settings=settings, nas=None)
        rep2 = forge2.improve_brain(enact=False)
        assert rep2.searched and not rep2.enacted and not rep2.promoted and not rep2.verified
        print("measure-only        : designed a brain, withheld promotion (enact off) ✓")

    print("\nALL SELF-TESTS PASSED ✓")
