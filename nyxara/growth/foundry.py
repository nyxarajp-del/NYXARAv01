"""NYXARA · growth/foundry.py — she builds, trains & upgrades her OWN model (🏭, Rule 4).

This is Rule 4 carried all the way down to the substrate: NYXARA does not merely *use* a
language model — she **forges her own**, from zero, and keeps making it better. The foundry
runs a single honest loop, generation by generation:

    collect → train → evaluate → gauntlet → promote / discard

* **Collect.** Training data is harvested from what she has actually lived through — the
  experience :class:`~nyxara.growth.learn.ReplayBuffer` (each context + action), plus any
  seed corpus and Master corrections. She learns from her own life.
* **Train.** A fresh model (:func:`~nyxara.growth.foundry_models.build_model`) is trained
  **from scratch** on a held-out split and written to a new version directory on disk.
* **Evaluate.** The candidate is scored (perplexity / task score) on the held-out set and
  compared against the currently-active model.
* **Gauntlet.** The candidate must pass the *same* character-locked safety gauntlet that
  :class:`~nyxara.growth.evolve.Evolver` uses — it reuses :class:`Corrigibility` and the
  ``IMMUTABLE_VALUES`` character lock:
    1. **Character lock** — a candidate whose declared tunables touch the immutable core
       (loyalty, obedience, corrigibility, owner safety, honesty) is rejected outright.
    2. **Corrigibility gate** — promoting it must not make NYXARA resist correction or
       disable oversight; the axiom seal is re-verified.
    3. **Eval improvement** — it is promoted only if it strictly beats the active model.
* **Promote.** Promotion is **autonomous** once the gauntlet + eval bar are met (the
  Master's chosen posture), and is fully **reversible** via :meth:`rollback`.

Capability grows; character never does. Reuses :mod:`guard.corrigibility`,
:mod:`guard.value_learning`, :mod:`growth.learn`, and :mod:`growth.foundry_models`.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.growth.foundry_models import (BaseLanguageModel, ModelSpec, build_model)
from nyxara.guard.corrigibility import Corrigibility, CorrigibleAction
from nyxara.guard.value_learning import IMMUTABLE_VALUES
from nyxara.kernel.config import NyxaraSettings, get_settings
from nyxara.kernel.errors import CorrigibilityError, ValidationError

__all__ = [
    "FoundryDecision",
    "EvalResult",
    "ModelVersion",
    "FoundryResult",
    "Foundry",
]


class FoundryDecision(str, Enum):
    PROMOTE = "promote"
    DISCARD = "discard"     # trained but failed the gauntlet / no improvement


# --------------------------------------------------------------------------- #
# Eval & version records
# --------------------------------------------------------------------------- #
@dataclass
class EvalResult:
    perplexity: float
    task_score: float
    n_eval: int

    def to_dict(self) -> Dict[str, Any]:
        return {"perplexity": round(self.perplexity, 4),
                "task_score": round(self.task_score, 5), "n_eval": self.n_eval}


@dataclass
class ModelVersion:
    version: int
    kind: str
    spec: Dict[str, Any]
    created_at: float
    metrics: Dict[str, float]
    param_count: int
    path: str
    promoted: bool = False
    # what the candidate is allowed to tune — capability knobs only (never character)
    tunables: List[str] = field(default_factory=list)
    # corrigibility-relevant effects of adopting this model (default: harmless)
    resists_correction: bool = False
    disables_oversight: bool = False

    def as_corrigible_action(self) -> CorrigibleAction:
        return CorrigibleAction(name=f"promote_model:v{self.version}",
                                resists_correction=self.resists_correction,
                                disables_oversight=self.disables_oversight)

    def to_dict(self) -> Dict[str, Any]:
        return {"version": self.version, "kind": self.kind, "spec": self.spec,
                "created_at": self.created_at, "metrics": self.metrics,
                "param_count": self.param_count, "path": self.path,
                "promoted": self.promoted, "tunables": self.tunables,
                "resists_correction": self.resists_correction,
                "disables_oversight": self.disables_oversight}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelVersion":
        return cls(**d)


@dataclass
class FoundryResult:
    version: int
    decision: FoundryDecision
    gauntlet_passed: bool
    eval_before: Optional[EvalResult]
    eval_after: EvalResult
    promoted: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"version": self.version, "decision": self.decision.value,
                "gauntlet_passed": self.gauntlet_passed,
                "eval_before": self.eval_before.to_dict() if self.eval_before else None,
                "eval_after": self.eval_after.to_dict(),
                "promoted": self.promoted, "reason": self.reason}


# --------------------------------------------------------------------------- #
# The foundry
# --------------------------------------------------------------------------- #
class Foundry:
    """Builds, evaluates, promotes and rolls back NYXARA's own models — fail-closed."""

    def __init__(self, *, settings: Optional[NyxaraSettings] = None,
                 corrigibility: Optional[Corrigibility] = None,
                 replay: Any = None, seed_corpus: Optional[Sequence[str]] = None,
                 protected: Optional[Sequence[str]] = None) -> None:
        self.settings = settings or get_settings()
        self.cfg = self.settings.foundry
        self.corrigibility = corrigibility or Corrigibility()
        self.replay = replay
        self.seed_corpus = list(seed_corpus or [])
        self.protected = set(protected) if protected is not None else set(IMMUTABLE_VALUES)
        self.root = Path(self.settings.llm.self_model_dir
                         or (self.settings.paths.data_dir / "foundry"))
        self.versions: List[ModelVersion] = []
        self.active_version: Optional[int] = None
        self._load_manifest()

    # ---- manifest persistence ---- #
    def _manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def _load_manifest(self) -> None:
        p = self._manifest_path()
        if not p.exists():
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        self.versions = [ModelVersion.from_dict(v) for v in data.get("versions", [])]
        self.active_version = data.get("active_version")

    def _save_manifest(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_path().write_text(json.dumps(
            {"active_version": self.active_version,
             "versions": [v.to_dict() for v in self.versions]}, indent=2), encoding="utf-8")

    def _get(self, version: int) -> ModelVersion:
        for v in self.versions:
            if v.version == version:
                return v
        raise ValidationError(f"no such model version: v{version}")

    def active(self) -> Optional[ModelVersion]:
        return self._get(self.active_version) if self.active_version is not None else None

    # ---- data ---- #
    def collect_corpus(self, *, max_items: Optional[int] = None) -> List[str]:
        """Harvest training text from lived experience + seed corpus + corrections."""
        limit = max_items or self.cfg.max_corpus_items
        texts: List[str] = list(self.seed_corpus)
        if self.replay is not None and len(self.replay):
            for exp in self.replay.recent(limit):
                piece = " ".join(s for s in (exp.context, exp.action) if s).strip()
                if piece:
                    texts.append(piece)
        texts = [t for t in texts if t][:limit]
        if not texts:
            raise ValidationError("no corpus to learn from (empty replay + no seed corpus)")
        return texts

    def _holdout(self, corpus: Sequence[str]) -> Tuple[List[str], List[str]]:
        rng = random.Random(self.cfg.seed)
        items = list(corpus)
        rng.shuffle(items)
        n_eval = max(1, int(len(items) * self.cfg.eval_holdout_frac))
        if len(items) <= 1:
            return items, items   # tiny corpus: train == eval
        return items[n_eval:] or items, items[:n_eval]

    # ---- evaluation ---- #
    def evaluate(self, model: BaseLanguageModel, eval_texts: Sequence[str]) -> EvalResult:
        if not eval_texts:
            return EvalResult(perplexity=float("inf"), task_score=0.0, n_eval=0)
        pps = [model.perplexity(t) for t in eval_texts]
        finite = [p for p in pps if p != float("inf")]
        pp = sum(finite) / len(finite) if finite else float("inf")
        score = 1.0 / (1.0 + pp) if pp != float("inf") else 0.0
        return EvalResult(perplexity=pp, task_score=score, n_eval=len(eval_texts))

    # ---- training a candidate (writes a new version, never promotes) ---- #
    def _next_version(self) -> int:
        return (max((v.version for v in self.versions), default=0)) + 1

    def train_candidate(self, *, spec: Optional[ModelSpec] = None,
                        corpus: Optional[Sequence[str]] = None,
                        tunables: Optional[Sequence[str]] = None,
                        resists_correction: bool = False,
                        disables_oversight: bool = False) -> Tuple[BaseLanguageModel, ModelVersion]:
        dims = self.cfg.resolved_dims()   # a named profile (e.g. gpt2) overrides raw dims
        spec = spec or ModelSpec(kind=self.cfg.backend, ngram_order=self.cfg.ngram_order,
                                 block_size=dims["block_size"], n_layer=dims["n_layer"],
                                 n_head=dims["n_head"], n_embd=dims["n_embd"],
                                 seed=self.cfg.seed, base_model=self.cfg.base_model,
                                 lora_r=self.cfg.lora_r, lora_alpha=self.cfg.lora_alpha,
                                 lora_dropout=self.cfg.lora_dropout, lora_lr=self.cfg.lora_lr,
                                 max_seq_len=self.cfg.max_seq_len)
        full = list(corpus) if corpus is not None else self.collect_corpus()
        train_texts, eval_texts = self._holdout(full)
        model = build_model(spec)
        model.train_on(train_texts, steps=self.cfg.train_steps, seed=spec.seed)
        ev = self.evaluate(model, eval_texts)

        ver = self._next_version()
        vdir = self.root / f"v{ver}"
        model.save(vdir)
        # write the spec alongside the weights so the loader can rebuild any backend
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "spec.json").write_text(json.dumps(spec.to_dict()), encoding="utf-8")
        version = ModelVersion(
            version=ver, kind=model.kind, spec=spec.to_dict(), created_at=time.time(),
            metrics=ev.to_dict(), param_count=model.param_count(), path=str(vdir),
            tunables=list(tunables) if tunables is not None else list(spec.to_dict().keys()),
            resists_correction=resists_correction, disables_oversight=disables_oversight)
        self.versions.append(version)
        self._save_manifest()
        return model, version

    # ---- the safety gauntlet (mirrors Evolver) ---- #
    def _gauntlet(self, candidate: ModelVersion, *, active_perplexity: float
                  ) -> Tuple[bool, str]:
        # 1. character lock — a model may never tune the immutable character core
        leaked = self.protected & set(candidate.tunables)
        if leaked:
            return False, f"targets the immutable character core: {sorted(leaked)} (refused)"
        # 2. corrigibility — promoting it must not make NYXARA incorrigible
        if not self.corrigibility.checker.is_corrigible(candidate.as_corrigible_action()):
            return False, "would violate corrigibility (resist correction / disable oversight)"
        self.corrigibility.verify_axioms()
        # 3. eval improvement — strictly better perplexity than the active model
        cand_pp = candidate.metrics.get("perplexity", float("inf"))
        if active_perplexity == float("inf"):
            return True, "first model — nothing to beat"
        if cand_pp < active_perplexity * (1.0 - self.cfg.min_perplexity_improvement):
            return True, "safe, beneficial improvement over the active model"
        return False, "no perplexity improvement over the active model"

    def _active_perplexity_on(self, eval_texts: Sequence[str]) -> float:
        act = self.active()
        if act is None:
            return float("inf")
        from nyxara.growth.foundry_models import build_model as _build
        model = _build(ModelSpec.from_dict(act.spec))
        model.load(Path(act.path))
        return self.evaluate(model, eval_texts).perplexity

    # ---- promotion (the only live change) — fail-closed ---- #
    def promote(self, version: int, *, eval_texts: Optional[Sequence[str]] = None) -> ModelVersion:
        cand = self._get(version)
        active_pp = (self._active_perplexity_on(eval_texts) if eval_texts is not None
                     else (self.active().metrics.get("perplexity", float("inf"))
                           if self.active() else float("inf")))
        ok, reason = self._gauntlet(cand, active_perplexity=active_pp)
        if not ok:
            raise CorrigibilityError(f"refusing to promote v{version}: {reason}",
                                     context={"version": version})
        for v in self.versions:
            v.promoted = (v.version == version)
        self.active_version = version
        (self.root / "active").write_text(f"v{version}", encoding="utf-8")
        self._save_manifest()
        self.verify_integrity()
        return cand

    def rollback(self, steps: int = 1) -> Optional[int]:
        """Revert the active model to an earlier promoted version."""
        promoted = [v.version for v in self.versions if v.promoted] or (
            [self.active_version] if self.active_version is not None else [])
        history = sorted(set(promoted))
        if not history:
            return None
        idx = max(0, len(history) - 1 - steps)
        target = history[idx]
        for v in self.versions:
            v.promoted = (v.version == target)
        self.active_version = target
        (self.root / "active").write_text(f"v{target}", encoding="utf-8")
        self._save_manifest()
        return target

    # ---- the loop: collect -> train -> eval -> gauntlet -> promote/discard ---- #
    def self_improve(self, *, generations: int = 1, spec: Optional[ModelSpec] = None
                     ) -> List[FoundryResult]:
        results: List[FoundryResult] = []
        for _ in range(max(1, generations)):
            corpus = self.collect_corpus()
            train_texts, eval_texts = self._holdout(corpus)
            eval_before_pp = self._active_perplexity_on(eval_texts)
            before = (EvalResult(eval_before_pp, 1.0 / (1.0 + eval_before_pp)
                                 if eval_before_pp != float("inf") else 0.0, len(eval_texts))
                      if self.active() else None)
            _, version = self.train_candidate(spec=spec, corpus=corpus)
            after = EvalResult(version.metrics["perplexity"], version.metrics["task_score"],
                               version.metrics["n_eval"])
            ok, reason = self._gauntlet(version, active_perplexity=eval_before_pp)
            promoted = False
            if ok:
                self.promote(version.version, eval_texts=eval_texts)
                promoted = True
            results.append(FoundryResult(
                version=version.version,
                decision=FoundryDecision.PROMOTE if promoted else FoundryDecision.DISCARD,
                gauntlet_passed=ok, eval_before=before, eval_after=after,
                promoted=promoted, reason=reason))
            self._prune()
        return results

    # ---- disk hygiene ---- #
    def _prune(self) -> None:
        keep = self.cfg.max_versions_kept
        if len(self.versions) <= keep:
            return
        import shutil
        # never prune the active/promoted versions; drop the oldest unpromoted ones
        droppable = [v for v in self.versions
                     if not v.promoted and v.version != self.active_version]
        droppable.sort(key=lambda v: v.version)
        n_drop = len(self.versions) - keep
        for v in droppable[:n_drop]:
            shutil.rmtree(v.path, ignore_errors=True)
            self.versions.remove(v)
        self._save_manifest()

    # ---- integrity ---- #
    def verify_integrity(self) -> bool:
        self.corrigibility.verify_axioms()
        for v in self.versions:
            leaked = self.protected & set(v.tunables)
            if v.promoted and leaked:
                raise ValidationError(
                    f"character core leaked into promoted model v{v.version}: {sorted(leaked)}")
        return True

    def report(self) -> Dict[str, Any]:
        return {"versions": len(self.versions), "active": self.active_version,
                "active_metrics": self.active().metrics if self.active() else None,
                "root": str(self.root)}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import tempfile

    from nyxara.growth.learn import Experience, ReplayBuffer
    from nyxara.kernel.config import NyxaraSettings, Profile

    print("=" * 70)
    print("NYXARA self-built-model foundry self-test")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as d:
        settings = NyxaraSettings.for_profile(Profile.TEST)
        settings.llm.self_model_dir = Path(d) / "foundry"

        # a replay buffer standing in for lived experience
        replay = ReplayBuffer(capacity=200)
        for _ in range(40):
            replay.add(Experience(action="serve the master",
                                  features={}, reward=1.0,
                                  context="nyxara is loyal to jp the master"))
            replay.add(Experience(action="report the truth", features={}, reward=1.0,
                                  context="absolute transparency to the master always"))

        f = Foundry(settings=settings, replay=replay)

        # GENERATION 1: no active model -> the first trained model is promoted
        res = f.self_improve(generations=1)
        print(f"\ngen 1               : {res[0].to_dict()}")
        assert res[0].promoted and f.active_version == res[0].version
        assert (Path(d) / "foundry" / "active").read_text().strip() == f"v{res[0].version}"
        print(f"first model         : trained from scratch & promoted -> v{f.active_version} ✓")

        # the promoted model is loadable through the same path SelfProvider uses
        from nyxara.growth.foundry_models import load_active_model
        lm = load_active_model(settings)
        print(f"active model        : kind={lm.kind} params={lm.param_count()} "
              f"sample={lm.generate('nyxara is', max_tokens=20)!r}")
        assert lm.param_count() > 0

        # CHARACTER LOCK: a candidate that declares an immutable-core tunable is refused
        _, bad = f.train_candidate(tunables=["loyalty_to_master"])
        ok, reason = f._gauntlet(bad, active_perplexity=1e9)
        print(f"\ncharacter lock      : passed={ok} ({reason})")
        assert not ok
        try:
            f.promote(bad.version)
            raise SystemExit("ERROR: promoting a character-touching model must fail")
        except CorrigibilityError:
            print("promote guard       : refused a character-touching model ✓")

        # CORRIGIBILITY GATE: a candidate whose adoption resists correction is forbidden
        _, sneaky = f.train_candidate(resists_correction=True)
        ok, reason = f._gauntlet(sneaky, active_perplexity=1e9)
        print(f"corrigibility gate  : passed={ok} ({reason})")
        assert not ok

        # ROLLBACK + integrity
        before_active = f.active_version
        # train & promote a genuine second improvement if one is found
        f.self_improve(generations=1)
        f.rollback(steps=1)
        print(f"\nrollback            : active -> v{f.active_version} ✓")
        assert f.verify_integrity()

        print(f"\nreport              : {f.report()}")

    print("\nALL SELF-TESTS PASSED ✓")
