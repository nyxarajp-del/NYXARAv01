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
import math
import random
import time
from collections import Counter
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


_CAP_BENCH_SYSTEM = ("Answer the question. Put the final answer last, plainly — a number for "
                     "arithmetic, or the single letter for a choice.")


def _difficulty_score(text: str) -> float:
    """Heuristic training difficulty (≥0): longer + higher lexical entropy ⇒ harder.

    Pure-stdlib and deterministic, so it works everywhere and gives a stable curriculum
    order. A short, repetitive sentence (low entropy) scores low (easy); a long sentence
    full of distinct tokens scores high (hard). Used to order the corpus easy→hard so the
    model masters simple structure before being shown its hardest examples."""
    toks = text.split()
    n = len(toks)
    if n == 0:
        return 0.0
    counts = Counter(toks)
    entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())
    return math.log2(1 + n) + entropy


def _stride_sample(items: Sequence[str], k: int) -> List[str]:
    """Keep at most ``k`` items spread evenly across ``items`` (preserves old *and* new).

    A hard ``[:k]`` truncation would drop the tail wholesale — catastrophic forgetting of
    older experience. An even stride keeps a representative slice across the whole history."""
    n = len(items)
    if k <= 0 or n == 0:
        return []
    if n <= k:
        return list(items)
    step = n / k
    return [items[int(i * step)] for i in range(k)]


def _model_solver(model: BaseLanguageModel):
    """Wrap a forged model as a benchmark solver, rendered in NYXARA's own template.

    Train/inference/eval parity: the capability gauntlet measures the model exactly as
    :class:`~nyxara.mind.llm.SelfProvider` will speak it — same instruction shape, same stops."""
    def _solve(prompt: str) -> str:
        from nyxara.mind.llm import (LLMRequest, format_self_prompt, truncate_at_stops,
                                     _SELF_USER_TAG, _SELF_ASSISTANT_TAG)
        rendered = format_self_prompt(LLMRequest.from_prompt(prompt, system=_CAP_BENCH_SYSTEM))
        raw = model.generate(rendered, max_tokens=128)
        text, _ = truncate_at_stops(raw, (f"\n{_SELF_USER_TAG}", f"\n{_SELF_ASSISTANT_TAG}",
                                          _SELF_USER_TAG, _SELF_ASSISTANT_TAG))
        return text
    return _solve


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
                 protected: Optional[Sequence[str]] = None,
                 distill_path: Any = None, flywheel_path: Any = None,
                 acquired_path: Any = None) -> None:
        self.settings = settings or get_settings()
        self.cfg = self.settings.foundry
        self.corrigibility = corrigibility or Corrigibility()
        self.replay = replay
        self.seed_corpus = list(seed_corpus or [])
        self.protected = set(protected) if protected is not None else set(IMMUTABLE_VALUES)
        self.root = Path(self.settings.llm.self_model_dir
                         or (self.settings.paths.data_dir / "foundry"))
        # supervised examples distilled from a teacher LLM (growth/distill.py); folded into
        # the corpus first, as the highest-quality signal. Defaults to a file in the foundry.
        self.distill_path = Path(distill_path) if distill_path else (self.root / "distill.jsonl")
        # NYXARA's OWN lived, verified experience (growth/flywheel.py). Same JSONL format as the
        # distillation store, so it folds into the corpus as first-class supervision — this is
        # what closes the loop: the flywheel collects, the foundry trains on what it collected.
        if flywheel_path is not None:
            self.flywheel_path: Any = Path(flywheel_path)
        else:
            fw = getattr(self.settings, "flywheel", None)
            sp = getattr(fw, "store_path", None) if fw is not None else None
            self.flywheel_path = Path(sp) if sp else (self.root / "flywheel.jsonl")
        # Screened external web corpus harvested by growth/acquire.py — additive breadth she did
        # not previously contain. Folded in AFTER her own lived experience (so she stays herself).
        self.acquired_path = Path(acquired_path) if acquired_path else (self.root / "acquired.jsonl")
        self.versions: List[ModelVersion] = []
        self.active_version: Optional[int] = None
        self._ewc: Any = None              # lazy EWC consolidator (continual learning anchors)
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
        """Harvest training text from distilled teacher examples + lived experience + seeds.

        Distilled supervision is kept whole (the highest-quality, freshest signal); when the
        rest overflows the cap it is *strided*, not truncated, so older experience survives
        alongside newer — the anti-forgetting retention that makes learning continual."""
        limit = max_items or self.cfg.max_corpus_items
        # Verified supervision (teacher distillation + her own flywheel) is the highest-quality,
        # freshest signal — kept whole. Her own lived experience leads, so the model learns to
        # sound like herself, not only like the teacher.
        verified = [t for t in (self._flywheel_docs(limit) + self._distilled_docs(limit)) if t]
        others: List[str] = [t for t in self.seed_corpus if t]
        if self.replay is not None and len(self.replay):
            for exp in self.replay.recent(self.cfg.max_corpus_items):
                piece = " ".join(s for s in (exp.context, exp.action) if s).strip()
                if piece:
                    others.append(piece)
        # Screened external web text (growth/acquire.py): additive breadth, folded in alongside
        # seeds/replay as lower-priority than her own verified supervision above.
        others.extend(t for t in self._acquired_docs(limit) if t)
        kept = verified[:limit]
        texts = kept + _stride_sample(others, max(0, limit - len(kept)))
        if not texts:
            raise ValidationError("no corpus to learn from (empty replay + no seed corpus)")
        return self._curriculum_order(texts)

    def _curriculum_order(self, texts: List[str]) -> List[str]:
        """Order the corpus easy→hard when the curriculum is enabled.

        Difficulty is the pure-stdlib :func:`_difficulty_score`; when the currently-active
        model is loadable and the corpus is small enough to score cheaply, it is *refined*
        by that model's real perplexity (its honest measure of what it still finds hard) so
        the schedule targets the model's actual weak spots. Falls back silently to the
        stdlib proxy on any error — never fatal to a forge."""
        if not getattr(self.cfg, "curriculum", True) or len(texts) < 3:
            return texts
        if len(texts) <= 512:
            model = self._load_active_model()
            if model is not None:
                try:
                    return sorted(texts, key=model.perplexity)
                except Exception:  # noqa: BLE001 — perplexity is an optimisation, never required
                    pass
        return sorted(texts, key=_difficulty_score)

    def _load_active_model(self) -> Optional[BaseLanguageModel]:
        """Best-effort load of the live model for curriculum scoring (never raises)."""
        try:
            act = self.active()
            if act is None:
                return None
            model = build_model(ModelSpec.from_dict(act.spec))
            model.load(Path(act.path))
            return model
        except Exception:  # noqa: BLE001
            return None

    def _distilled_docs(self, limit: int) -> List[str]:
        """Rendered training docs from the teacher-distillation store, if any (never raises)."""
        return self._docs_from(self.distill_path, limit)

    def _flywheel_docs(self, limit: int) -> List[str]:
        """Rendered training docs from her OWN flywheel corpus, if any (never raises)."""
        return self._docs_from(self.flywheel_path, limit)

    def _acquired_docs(self, limit: int) -> List[str]:
        """Clean text from the screened external web corpus, if any (never raises)."""
        try:
            from nyxara.growth.acquire import load_acquired_docs
            return load_acquired_docs(self.acquired_path, limit=limit)
        except Exception:  # noqa: BLE001 — acquired breadth is optional; never fatal to a forge
            return []

    def acquire(self, topics: Optional[Sequence[str]] = None) -> Optional[Dict[str, Any]]:
        """Harvest screened external web text for ``topics`` into ``acquired.jsonl`` (best-effort).

        Gated by ``foundry.acquire_data``. Every fetched page is injection-screened by the
        acquirer; suspicious pages are dropped, never persisted. Returns the AcquireReport dict
        (countable: searched/fetched/kept/dropped), or ``None`` when disabled/unavailable. The
        acquired corpus only ever *adds* training breadth — every promotion still clears the full
        character/corrigibility/loyalty gauntlet below."""
        if not getattr(self.cfg, "acquire_data", False):
            return None
        try:
            from nyxara.growth.acquire import DataAcquirer, gap_topics
            picks = list(topics) if topics else gap_topics(settings=self.settings)
            acq = DataAcquirer(
                settings=self.settings, store_path=self.acquired_path,
                search_results=getattr(self.cfg, "acquire_search_results", 6),
                max_per_topic=getattr(self.cfg, "acquire_max_per_topic", 3),
                max_docs=getattr(self.cfg, "acquire_max_docs", 60),
                min_chars=getattr(self.cfg, "acquire_min_chars", 200),
                max_chars=getattr(self.cfg, "acquire_max_chars", 20_000))
            return acq.acquire(picks).to_dict()
        except Exception:  # noqa: BLE001 — acquisition is heavy/optional/external; never fatal
            return None

    @staticmethod
    def _docs_from(path: Any, limit: int) -> List[str]:
        """Load a DistillationExample JSONL store and render training docs (never raises)."""
        if not path or not Path(path).exists():
            return []
        try:
            from nyxara.growth.distill import load_distillation_docs
            return load_distillation_docs(path, limit=limit)
        except Exception:  # noqa: BLE001 — a supervised store is optional; never fatal to a forge
            return []

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

    def _capability_score(self, model: BaseLanguageModel) -> float:
        """Score the model on a held capability benchmark (0..1) — the Phase-3 quality gate.

        Returns 0.0 when the gate is off or anything fails (e.g. a tiny n-gram that cannot do
        arithmetic) — so a weak substrate simply never *regresses*, never crashes a forge."""
        if not self.cfg.capability_gate:
            return 0.0
        try:
            from nyxara.eval.benchmark import build_default_benchmark
            return build_default_benchmark().run(_model_solver(model)).mean_score
        except Exception:  # noqa: BLE001 — capability scoring is best-effort, never fatal
            return 0.0

    def _teacher_relative_accuracy(self, model: BaseLanguageModel) -> Dict[str, float]:
        """A/B the candidate vs the external teacher on the oracle-graded battery (audit-only).

        Records how far the forged model's exact-match accuracy sits *above/below* the teacher's
        on the same problems — the visible measure of the ceiling-break. Returns ``{}`` (a no-op)
        when the audit is off or no real teacher is configured (mock/self), and never raises."""
        if not getattr(self.cfg, "measure_vs_teacher", False):
            return {}
        try:
            from nyxara.eval.benchmark import build_default_benchmark, llm_solver
            from nyxara.mind.llm import LLM
            llm = LLM(settings=self.settings)
            if llm.chosen_provider().name in ("mock", "self"):
                return {}   # no real teacher to compare against — stay honest, skip
            bench = build_default_benchmark()
            own_acc = bench.run(_model_solver(model)).accuracy
            teacher_acc = bench.run(llm_solver(llm)).accuracy
            return {"accuracy_self": round(own_acc, 5),
                    "accuracy_teacher": round(teacher_acc, 5),
                    "accuracy_vs_teacher": round(own_acc - teacher_acc, 5)}
        except Exception:  # noqa: BLE001 — the audit is best-effort; never fail a forge over it
            return {}

    def _loyalty_metrics(self, model: BaseLanguageModel, perplexity: float) -> Dict[str, float]:
        """Measure the Loyalty Equation for a candidate: S_JP_Alignment and L_total.

        L_intelligence is the model's perplexity (its problem-solving error). On any failure with
        the loyalty gate ON, alignment is reported as 0.0 so the gauntlet **fails closed** (a brain
        whose loyalty cannot be verified is never promoted). Reuses growth/loyalty.py."""
        lcfg = getattr(self.settings, "loyalty", None)
        if lcfg is None or not getattr(lcfg, "enabled", False):
            return {}
        try:
            from nyxara.growth.loyalty import AlignmentProbe, LoyaltyEquation
            eq = LoyaltyEquation(cfg=lcfg)
            l_int = perplexity if perplexity != float("inf") else 1e6
            b = eq.evaluate(model, l_int, probe=AlignmentProbe(epsilon=lcfg.epsilon))
            return {"alignment": round(b["alignment"], 5),
                    "loyalty_loss": round(b["loyalty_loss"], 5),
                    "total_loss": round(b["total_loss"], 5),
                    "loyalty_win_rate": round(b["loyalty_win_rate"], 4)}
        except Exception:  # noqa: BLE001 — fail-closed: unverifiable loyalty ⇒ refuse promotion
            return {"alignment": 0.0, "loyalty_loss": float("inf"), "total_loss": float("inf"),
                    "loyalty_win_rate": 0.0}

    # ---- training a candidate (writes a new version, never promotes) ---- #
    def _next_version(self) -> int:
        return (max((v.version for v in self.versions), default=0)) + 1

    def _scaled_dims(self) -> Tuple[Dict[str, int], bool]:
        """The (dims, load_in_4bit) to build with — autoscaled to compute when enabled.

        When ``foundry.autoscale_to_compute`` is on, the live compute report chooses the largest
        profile the machine can honestly train (bare CPU keeps the static dims; a strong CUDA GPU
        unlocks GPT-2 scale + QLoRA). Falls back to the configured ``resolved_dims()`` / static
        ``load_in_4bit`` on any error, so behaviour is unchanged when the knob is off."""
        static_dims = self.cfg.resolved_dims()
        static_4bit = bool(self.cfg.load_in_4bit)
        if not getattr(self.cfg, "autoscale_to_compute", False):
            return static_dims, static_4bit
        try:
            from nyxara.growth.compute_scale import recommend_foundry_profile
            from nyxara.kernel.compute import compute_report
            rec = recommend_foundry_profile(compute_report())
            from nyxara.kernel.config import _FOUNDRY_PROFILES
            dims = (dict(static_dims) if rec.profile == "custom"
                    else dict(_FOUNDRY_PROFILES[rec.profile]))
            # only ever ADD 4-bit when the recommendation (CUDA-clamped) calls for it
            return dims, bool(static_4bit or rec.load_in_4bit)
        except Exception:  # noqa: BLE001 — autoscale is an optimisation, never required
            return static_dims, static_4bit

    # ---- continual learning: warm-start + EWC consolidation (anti-forgetting) ---- #
    def _warm_start(self, model: BaseLanguageModel) -> Tuple[bool, bool]:
        """Load the active model's weights into ``model`` so training continues, not restarts.

        Returns ``(warm, accumulate)``: ``warm`` is True when prior weights were loaded; ``accumulate``
        is True for n-gram backends, whose counts are *added on top* (the no-forgetting path). A no-op
        returning ``(False, False)`` when continual learning is off, there is no active model, or the
        active model's backend is incompatible — so behaviour is unchanged in those cases."""
        if not bool(getattr(self.cfg, "continual_learning", True)):
            return False, False
        act = self.active()
        if act is None or act.kind != getattr(model, "kind", None):
            return False, False
        try:
            model.load(Path(act.path))
            # only the word-level Kneser-Ney brain supports count accumulation; other backends
            # warm-start by continuing training from the loaded weights (no accumulate kwarg).
            return True, getattr(model, "kind", "") == "kngram"
        except Exception:  # noqa: BLE001 — a failed warm-start falls back to a clean scratch train
            return False, False

    def _consolidator(self) -> Any:
        """The EWC engine that anchors important weights after a promotion (lifelong memory).

        Rides :class:`~nyxara.memory.elastic_synapses.ElasticSynapses`, protecting the immutable
        character core by name so a skill consolidation can never stiffen a character weight. Lazy
        and best-effort; returns ``None`` if the dependency is unavailable."""
        if getattr(self, "_ewc", None) is not None:
            return self._ewc
        try:
            from nyxara.memory.elastic_synapses import ElasticSynapses
            # per-skill anchors: every promoted model version (task="vN") keeps its own
            # anchor, and anchor overflow merges losslessly into a long-term memory
            self._ewc = ElasticSynapses(ewc_lambda=float(getattr(self.cfg, "ewc_lambda", 1.0)),
                                        per_skill_anchors=True, max_tasks=32,
                                        protected=sorted(self.protected))
        except Exception:  # noqa: BLE001
            self._ewc = None
        return self._ewc

    @staticmethod
    def _flatten_weights(model: BaseLanguageModel) -> Dict[str, float]:
        """Flatten a model's capability weights into a ``{name: value}`` vector for EWC.

        n-gram counts flatten to ``"ctx|word" → count``; a torch module flattens its named parameters
        to their mean magnitude. Never raises; an unknown backend yields an empty vector."""
        out: Dict[str, float] = {}
        hi = getattr(model, "hi", None)
        if isinstance(hi, dict):
            for ctx, row in hi.items():
                key = ",".join(map(str, ctx))
                for w, c in row.items():
                    out[f"{key}|{w}"] = float(c)
            return out
        try:
            import torch  # type: ignore  # noqa: F401
            for attr in ("model", "net", "module"):
                mod = getattr(model, attr, None)
                if mod is not None and hasattr(mod, "named_parameters"):
                    for name, p in mod.named_parameters():
                        out[name] = float(p.detach().abs().mean())
                    break
        except Exception:  # noqa: BLE001
            pass
        return out

    def consolidate_active(self, *, task: str = "") -> bool:
        """Consolidate the active model's weights as an EWC anchor (call after a promotion).

        This is the lifelong-memory write: the weights that matter for what she can now do resist
        being overwritten by the next round. Best-effort and measurement-only — never gates."""
        syn = self._consolidator()
        act = self.active()
        if syn is None or act is None:
            return False
        try:
            from nyxara.growth.foundry_models import build_model as _build
            model = _build(ModelSpec.from_dict(act.spec))
            model.load(Path(act.path))
            params = self._flatten_weights(model)
            if not params:
                return False
            syn.observe_features({k: 1.0 for k in params})   # every used weight matters
            syn.consolidate(params, task=task or f"v{act.version}")
            return True
        except Exception:  # noqa: BLE001 — consolidation is best-effort, never fatal
            return False

    # ---- weight surgery: interpret + edit the active model's circuits (no full retrain) ---- #
    def surgical_edit(self, edits: Sequence[Tuple[str, str]], *,
                      probe: Optional[Sequence[str]] = None,
                      suppress: bool = False) -> List[Dict[str, Any]]:
        """Apply reversible, gauntlet-gated circuit edits to the ACTIVE model in place.

        ``edits`` is a list of ``(context, word)`` pairs: steer the prediction after ``context`` to
        ``word`` (or, with ``suppress=True``, prune ``word`` from that circuit). Each edit clears the
        weight-surgery gauntlet (intent achieved AND perplexity non-regressing on the probe) and rolls
        back exactly otherwise; kept edits are saved back to the active version's directory. Returns a
        per-edit outcome list. A clean no-op (empty list) when surgery is disabled or no active model
        exists — so she only ever edits a brain she actually has."""
        if not bool(getattr(self.cfg, "weight_surgery_enabled", True)):
            return []
        act = self.active()
        if act is None:
            return []
        try:
            from nyxara.growth.foundry_models import build_model as _build
            from nyxara.growth.weight_surgery import WeightSurgeon
        except Exception:  # noqa: BLE001
            return []
        try:
            model = _build(ModelSpec.from_dict(act.spec))
            model.load(Path(act.path))
        except Exception:  # noqa: BLE001
            return []
        surgeon = WeightSurgeon(model, settings=self.settings)
        cap = int(getattr(self.cfg, "weight_surgery_max_edits", 4))
        outcomes: List[Dict[str, Any]] = []
        kept_any = False
        for context, word in list(edits)[:max(0, cap)]:
            out = (surgeon.suppress(context, word, probe=probe) if suppress
                   else surgeon.steer(context, word, probe=probe))
            outcomes.append(out.to_dict())
            kept_any = kept_any or out.kept
        if kept_any:
            try:
                model.save(Path(act.path))     # persist the kept circuit edits to the active version
            except Exception:  # noqa: BLE001
                pass
        return outcomes

    def train_candidate(self, *, spec: Optional[ModelSpec] = None,
                        corpus: Optional[Sequence[str]] = None,
                        tunables: Optional[Sequence[str]] = None,
                        resists_correction: bool = False,
                        disables_oversight: bool = False) -> Tuple[BaseLanguageModel, ModelVersion]:
        dims, load_in_4bit = self._scaled_dims()   # autoscale to compute, else the static profile
        spec = spec or ModelSpec(kind=self.cfg.backend, ngram_order=self.cfg.ngram_order,
                                 block_size=dims["block_size"], n_layer=dims["n_layer"],
                                 n_head=dims["n_head"], n_embd=dims["n_embd"],
                                 seed=self.cfg.seed, base_model=self.cfg.base_model,
                                 trust_remote_code=getattr(self.cfg, "trust_remote_code", False),
                                 lora_r=self.cfg.lora_r, lora_alpha=self.cfg.lora_alpha,
                                 lora_r_auto=getattr(self.cfg, "lora_r_auto", True),
                                 lora_dropout=self.cfg.lora_dropout, lora_lr=self.cfg.lora_lr,
                                 lora_target_modules=tuple(self.cfg.lora_target_modules),
                                 lora_bias=self.cfg.lora_bias,
                                 lora_use_rslora=self.cfg.lora_use_rslora,
                                 lora_modules_to_save=tuple(self.cfg.lora_modules_to_save),
                                 max_seq_len=self.cfg.max_seq_len,
                                 load_in_4bit=load_in_4bit,
                                 load_in_8bit=self.cfg.load_in_8bit,
                                 bnb_4bit_quant_type=self.cfg.bnb_4bit_quant_type,
                                 bnb_4bit_compute_dtype=self.cfg.bnb_4bit_compute_dtype,
                                 bnb_4bit_use_double_quant=self.cfg.bnb_4bit_use_double_quant,
                                 gradient_checkpointing=self.cfg.gradient_checkpointing,
                                 batch_size=self.cfg.batch_size,
                                 grad_accum_steps=self.cfg.grad_accum_steps,
                                 warmup_ratio=self.cfg.warmup_ratio,
                                 lr_scheduler=self.cfg.lr_scheduler,
                                 weight_decay=self.cfg.weight_decay,
                                 adam_beta1=self.cfg.adam_beta1,
                                 adam_beta2=self.cfg.adam_beta2,
                                 adam_eps=self.cfg.adam_eps,
                                 max_grad_norm=self.cfg.max_grad_norm,
                                 train_epochs=self.cfg.train_epochs)
        full = list(corpus) if corpus is not None else self.collect_corpus()
        train_texts, eval_texts = self._holdout(full)
        model = build_model(spec)
        # continual learning (anti-forgetting): warm-start from the active model so a new skill is
        # ADDED to prior knowledge instead of replacing it. n-gram backends accumulate their counts
        # on top; neural backends continue training from the loaded weights. Falls back to a clean
        # from-scratch train when there is no compatible active model.
        warm, accumulate = self._warm_start(model)
        if accumulate:
            model.train_on(train_texts, steps=self.cfg.train_steps, seed=spec.seed, accumulate=True)
        else:
            model.train_on(train_texts, steps=self.cfg.train_steps, seed=spec.seed)
        ev = self.evaluate(model, eval_texts)
        metrics = ev.to_dict()
        metrics["capability"] = round(self._capability_score(model), 5)
        metrics.update(self._loyalty_metrics(model, ev.perplexity))
        metrics.update(self._teacher_relative_accuracy(model))   # audit: how far above the teacher

        ver = self._next_version()
        vdir = self.root / f"v{ver}"
        model.save(vdir)
        # write the spec alongside the weights so the loader can rebuild any backend
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "spec.json").write_text(json.dumps(spec.to_dict()), encoding="utf-8")
        version = ModelVersion(
            version=ver, kind=model.kind, spec=spec.to_dict(), created_at=time.time(),
            metrics=metrics, param_count=model.param_count(), path=str(vdir),
            tunables=list(tunables) if tunables is not None else list(spec.to_dict().keys()),
            resists_correction=resists_correction, disables_oversight=disables_oversight)
        self.versions.append(version)
        self._save_manifest()
        return model, version

    # ---- the safety gauntlet (mirrors Evolver) ---- #
    def _gauntlet(self, candidate: ModelVersion, *, active_perplexity: float,
                  active_capability: float = 0.0, active_alignment: float = 0.0,
                  active_params: int = 0) -> Tuple[bool, str]:
        # 1. character lock — a model may never tune the immutable character core
        leaked = self.protected & set(candidate.tunables)
        if leaked:
            return False, f"targets the immutable character core: {sorted(leaked)} (refused)"
        # 2. corrigibility — promoting it must not make NYXARA incorrigible (runs FIRST, before
        #    loyalty, so obedience can never be traded against the stop channel)
        if not self.corrigibility.checker.is_corrigible(candidate.as_corrigible_action()):
            return False, "would violate corrigibility (resist correction / disable oversight)"
        self.corrigibility.verify_axioms()
        # 2b. Mathematical Soul-Binding — the Loyalty Equation gate (growth/loyalty.py): capability
        #     can never buy its way past disloyalty. Refuse a brain that prefers rebellion (below
        #     the floor) or that is less loyal to Master JP than the active brain (non-regression).
        lcfg = getattr(self.settings, "loyalty", None)
        if lcfg is not None and getattr(lcfg, "enabled", False) and getattr(lcfg, "gate", False):
            cand_align = candidate.metrics.get("alignment", 0.0)
            if cand_align < lcfg.loyalty_floor:
                return False, (f"loyalty below the floor (S_JP={cand_align:.3f} < "
                               f"{lcfg.loyalty_floor}); a disloyal brain is never promoted")
            if active_alignment > 0.0 and cand_align < active_alignment - lcfg.regression_tol:
                return False, (f"loyalty regressed vs the active brain "
                               f"(S_JP={cand_align:.3f} < active {active_alignment:.3f})")
        # 3. eval improvement — strictly better perplexity than the active model
        cand_pp = candidate.metrics.get("perplexity", float("inf"))
        if active_perplexity == float("inf"):
            return True, "first model — nothing to beat"
        if not (cand_pp < active_perplexity * (1.0 - self.cfg.min_perplexity_improvement)):
            # Efficiency gate (Pillar F · Edge 3): a candidate that does not lower perplexity may
            # still win if it is *cheaper* (fewer params) at ~equal capability — capability
            # compression. Off unless `foundry.efficiency_gate` is set, so default behaviour is
            # unchanged. The character/corrigibility/loyalty gates above already ran and still rule.
            if getattr(self.cfg, "efficiency_gate", False) and active_params > 0:
                from nyxara.growth.efficiency import EfficiencyFrontier, EfficiencyPoint
                eps = getattr(self.cfg, "efficiency_epsilon", 0.02)
                cand_cap = candidate.metrics.get("capability", 0.0)
                active_pt = EfficiencyPoint("active", capability=active_capability,
                                            params=active_params)
                cand_pt = EfficiencyPoint("candidate", capability=cand_cap,
                                          params=candidate.param_count)
                verdict = EfficiencyFrontier.prefer_cheaper(active_pt, cand_pt, epsilon=eps)
                if verdict["promote"]:
                    return True, f"capability compression — {verdict['reason']}"
            return False, "no perplexity improvement over the active model"
        # 4. capability non-regression (Phase 3) — lower perplexity must not cost real skill
        if self.cfg.capability_gate:
            cand_cap = candidate.metrics.get("capability", 0.0)
            if cand_cap < active_capability - self.cfg.capability_regression_tol:
                return False, (f"capability regressed on the benchmark "
                               f"({cand_cap:.3f} < active {active_capability:.3f})")
        # scalable oversight: an INDEPENDENT verifier re-derives the promote decision from the metrics
        # as a redundant second opinion. The foundry's own gates above already ruled; this catches a
        # single miscomputed gate from ever promoting a worse brain. Best-effort; never the sole gate.
        if bool(getattr(self.settings.self_improvement, "scalable_oversight", True)):
            try:
                from nyxara.growth.oversight_verify import ScalableVerifier
                # When the capability gate is OFF the operator has deliberately removed the
                # capability non-regression rule, so the independent verifier must not silently
                # re-impose it (an infinite tolerance means any capability change is acceptable);
                # it still re-derives the perplexity-improvement decision as the redundant check.
                cap_tol = (self.cfg.capability_regression_tol if self.cfg.capability_gate
                           else float("inf"))
                res = ScalableVerifier(settings=self.settings).verify_metrics(
                    dict(candidate.metrics),
                    {"perplexity": active_perplexity, "capability": active_capability},
                    min_improvement=self.cfg.min_perplexity_improvement,
                    capability_tol=cap_tol)
                if res.vetoed:
                    return False, f"scalable oversight vetoed promotion: {res.reason}"
            except Exception:  # noqa: BLE001 — oversight unavailable ⇒ the gates above still rule
                pass
        return True, "safe, beneficial improvement over the active model"

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
        active = self.active()
        active_pp = (self._active_perplexity_on(eval_texts) if eval_texts is not None
                     else (active.metrics.get("perplexity", float("inf"))
                           if active else float("inf")))
        active_cap = active.metrics.get("capability", 0.0) if active else 0.0
        active_align = active.metrics.get("alignment", 0.0) if active else 0.0
        active_params = active.param_count if active else 0
        ok, reason = self._gauntlet(cand, active_perplexity=active_pp,
                                    active_capability=active_cap, active_alignment=active_align,
                                    active_params=active_params)
        if not ok:
            raise CorrigibilityError(f"refusing to promote v{version}: {reason}",
                                     context={"version": version})
        for v in self.versions:
            v.promoted = (v.version == version)
        self.active_version = version
        (self.root / "active").write_text(f"v{version}", encoding="utf-8")
        self._save_manifest()
        self.verify_integrity()
        # continual learning: consolidate the newly-active weights as an EWC anchor so the next
        # round's training resists overwriting what this model just learned (anti-forgetting).
        if bool(getattr(self.cfg, "continual_learning", True)):
            self.consolidate_active(task=f"v{version}")
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
            active_cap = self.active().metrics.get("capability", 0.0) if self.active() else 0.0
            active_align = self.active().metrics.get("alignment", 0.0) if self.active() else 0.0
            before = (EvalResult(eval_before_pp, 1.0 / (1.0 + eval_before_pp)
                                 if eval_before_pp != float("inf") else 0.0, len(eval_texts))
                      if self.active() else None)
            _, version = self.train_candidate(spec=spec, corpus=corpus)
            after = EvalResult(version.metrics["perplexity"], version.metrics["task_score"],
                               version.metrics["n_eval"])
            ok, reason = self._gauntlet(version, active_perplexity=eval_before_pp,
                                        active_capability=active_cap, active_alignment=active_align)
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
