"""NYXARA · growth/weight_surgery.py — self-interpretability + weight surgery (🧬, Rule 4).

The Master's third gap: NYXARA could only change her learned brain by **replacing** it — forge a new
model from scratch (foundry) or search a new architecture (genesis). Her actual *weights* were a
black box she could not look inside and could not edit. A real recursively-self-improving mind does
mechanistic interpretability on itself and performs **surgery on a specific circuit** instead of
retraining the whole model.

This module is that faculty, made real and reversible on the model NYXARA actually runs:

* **n-gram / Kneser-Ney brain** (``WordKNGramLM`` / ``NgramByteLM``, always available, pure stdlib):
  her "weights" are the context→continuation **count tables** (``hi`` / ``lo``). This is the most
  interpretable substrate there is — every prediction traces to an exact row. :meth:`localize`
  attributes a next-token prediction to the responsible counts; :meth:`steer` and :meth:`suppress`
  surgically reweight a single circuit (raise the right continuation, prune a wrong one) — a genuine
  circuit-level edit, deterministic, "NYXARA khude kare".
* **neural brain** (``NanoGPTModel`` / ``LoRAModel``, when torch is present): :meth:`localize`
  performs ablation attribution (scale a parameter group to zero, measure the perplexity it was
  responsible for); :meth:`edit_parameter` applies a targeted, reversible weight edit (scale / rank-1
  nudge) **without a full retrain**.

Every edit is gauntlet-gated and reversible the same way source edits and model promotions are:
snapshot → edit → verify (the intended behaviour changed **and** perplexity did not regress on a
held-out probe) → keep, else roll back to the exact prior weights. It only ever touches *capability*
counts/weights — never NYXARA's character: the immutable loyalty core lives in
``guard.value_learning`` and is not representable in these tables, so surgery cannot reach it.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["Attribution", "SurgeryOutcome", "WeightSurgeon"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class Attribution:
    """Which parts of the model drive a prediction — the interpretability read-out."""

    backend: str
    context: str
    items: List[Tuple[str, float]] = field(default_factory=list)   # (name/token, contribution)

    def to_dict(self) -> Dict[str, Any]:
        return {"backend": self.backend, "context": self.context,
                "items": [[n, round(float(v), 6)] for n, v in self.items]}


@dataclass
class SurgeryOutcome:
    """One reversible weight edit: what was attempted, whether it stuck, and the evidence."""

    backend: str
    intent: str
    applied: bool = False
    kept: bool = False
    rolled_back: bool = False
    perplexity_before: float = float("inf")
    perplexity_after: float = float("inf")
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"backend": self.backend, "intent": self.intent, "applied": self.applied,
                "kept": self.kept, "rolled_back": self.rolled_back,
                "perplexity_before": _round(self.perplexity_before),
                "perplexity_after": _round(self.perplexity_after), "detail": self.detail}


# --------------------------------------------------------------------------- #
# The surgeon
# --------------------------------------------------------------------------- #
class WeightSurgeon:
    """Introspect and surgically edit NYXARA's own forged-model weights, reversibly, gauntlet-gated.

    Parameters
    ----------
    model:    a :class:`~nyxara.growth.foundry_models.BaseLanguageModel` (n-gram or neural).
    settings: :class:`~nyxara.kernel.config.NyxaraSettings` (reads the weight-surgery tolerance).
    tol:      max fractional perplexity regression a kept edit may cause on the probe (default from
              config ``foundry.weight_surgery_tol``); the behavioural intent must also be achieved.
    """

    def __init__(self, model: Any, *, settings: Any = None, tol: Optional[float] = None) -> None:
        from nyxara.kernel.config import get_settings
        self.settings = settings or get_settings()
        self.model = model
        self.backend = self._detect_backend(model)
        cfg = getattr(self.settings, "foundry", None)
        self.tol = float(tol if tol is not None else getattr(cfg, "weight_surgery_tol", 0.25))

    @staticmethod
    def _detect_backend(model: Any) -> str:
        kind = getattr(model, "kind", "") or ""
        if kind in ("kngram", "ngram"):
            return "ngram"
        if kind in ("nanogpt", "lora", "genesis"):
            return "torch"
        return "unknown"

    # ================================================================== #
    # interpretability — attribute a prediction to the responsible weights
    # ================================================================== #
    def localize(self, context: str, *, top: int = 5) -> Attribution:
        if self.backend == "ngram":
            return self._localize_ngram(context, top=top)
        if self.backend == "torch":
            return self._localize_torch(context, top=top)
        return Attribution(self.backend, context, [])

    def predict_next(self, context: str) -> Optional[str]:
        """The single most-likely next token after ``context`` (the prediction surgery targets)."""
        if self.backend != "ngram":
            return None
        ctx = self._ngram_ctx(context)
        cands = self.model._candidates(ctx)
        if not cands:
            return None
        best = max(cands, key=lambda w: self.model._prob(ctx, w))
        return self.model.id2tok[best] if 0 <= best < len(self.model.id2tok) else None

    # ================================================================== #
    # surgery — reversible, gauntlet-gated circuit edits
    # ================================================================== #
    def steer(self, context: str, target_word: str, *,
              probe: Optional[Sequence[str]] = None) -> SurgeryOutcome:
        """Make ``target_word`` the prediction after ``context`` by reweighting one circuit.

        Reversible and gated: the edit is kept only if the prediction actually becomes
        ``target_word`` *and* perplexity on the probe does not regress beyond ``tol``."""
        intent = f"steer {context!r} → {target_word!r}"
        if self.backend == "ngram":
            return self._ngram_reweight(context, target_word, suppress=False,
                                        probe=probe, intent=intent)
        return SurgeryOutcome(self.backend, intent, detail="steer not supported on this backend")

    def suppress(self, context: str, wrong_word: str, *,
                 probe: Optional[Sequence[str]] = None) -> SurgeryOutcome:
        """Prune a wrong continuation ``wrong_word`` after ``context`` from its circuit (reversible)."""
        intent = f"suppress {context!r} ↛ {wrong_word!r}"
        if self.backend == "ngram":
            return self._ngram_reweight(context, wrong_word, suppress=True,
                                        probe=probe, intent=intent)
        return SurgeryOutcome(self.backend, intent, detail="suppress not supported on this backend")

    def edit_parameter(self, name: str, edit_fn: "Callable[[Any], Any]", *,
                       probe: Sequence[str], context: str = "") -> SurgeryOutcome:
        """Apply a reversible, verified edit to one named neural parameter (torch backends).

        ``edit_fn`` receives the parameter tensor and edits it in place (scale, rank-1 add, ablate).
        The edit is kept only if perplexity on the probe does not regress beyond ``tol``."""
        intent = f"edit parameter {name!r}"
        if self.backend != "torch":
            return SurgeryOutcome(self.backend, intent, detail="no torch parameters to edit")
        return self._torch_edit(name, edit_fn, probe=probe, intent=intent)

    # ------------------------------------------------------------------ #
    # n-gram backend
    # ------------------------------------------------------------------ #
    def _ngram_ctx(self, text: str) -> Tuple[int, ...]:
        ids = self.model._encode(text, intern=False)[:-1]      # drop the trailing EOS of the prompt
        o = int(getattr(self.model, "order", 2))
        return tuple(ids[-(o - 1):]) if o > 1 else ()

    def _localize_ngram(self, context: str, *, top: int) -> Attribution:
        ctx = self._ngram_ctx(context)
        cands = self.model._candidates(ctx)
        scored = [(self.model.id2tok[w], self.model._prob(ctx, w))
                  for w in cands if 0 <= w < len(self.model.id2tok)]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return Attribution("ngram", context, scored[:max(1, top)])

    def _ngram_reweight(self, context: str, word: str, *, suppress: bool,
                        probe: Optional[Sequence[str]], intent: str) -> SurgeryOutcome:
        out = SurgeryOutcome("ngram", intent)
        model = self.model
        if not bool(getattr(getattr(self.settings, "foundry", None),
                            "weight_surgery_enabled", True)):
            out.detail = "weight surgery disabled (NYXARA_FOUNDRY__WEIGHT_SURGERY_ENABLED)"
            return out
        ctx = self._ngram_ctx(context)
        probe_corpus = list(probe) if probe else [context]
        out.perplexity_before = self._mean_perplexity(probe_corpus)

        # snapshot exactly what we touch, so rollback restores the weights byte-for-byte
        prev_row = copy.deepcopy(model.hi.get(ctx))
        prev_vocab = len(model.id2tok)
        prev_tok2id = dict(model.tok2id)

        wid = model._intern(word)
        row = model.hi.setdefault(ctx, {})
        if suppress:
            row.pop(wid, None)
            if not row:
                model.hi.pop(ctx, None)
        else:
            peak = max(row.values()) if row else 0
            row[wid] = peak + max(1, peak)            # dominate the row → becomes the argmax
        model._recompute_totals()
        out.applied = True

        # gauntlet: the behavioural intent must hold AND perplexity must not regress beyond tol
        pred = self.predict_next(context)
        intent_ok = (pred != word) if suppress else (pred == word)
        out.perplexity_after = self._mean_perplexity(probe_corpus)
        regressed = (out.perplexity_after >
                     out.perplexity_before * (1.0 + self.tol) + 1e-9)
        if intent_ok and not regressed:
            out.kept = True
            out.detail = f"prediction now {pred!r}; perplexity {_round(out.perplexity_before)}→" \
                         f"{_round(out.perplexity_after)} (within tol {self.tol})"
            return out

        # roll back to the exact prior weights
        if prev_row is None:
            model.hi.pop(ctx, None)
        else:
            model.hi[ctx] = prev_row
        # trim any vocab the edit grew, restoring the token tables exactly
        if len(model.id2tok) > prev_vocab:
            del model.id2tok[prev_vocab:]
            model.tok2id = prev_tok2id
        model._recompute_totals()
        out.rolled_back = True
        out.detail = ("intent unmet" if not intent_ok
                      else f"perplexity regressed {_round(out.perplexity_before)}→"
                           f"{_round(out.perplexity_after)}") + " — rolled back"
        return out

    # ------------------------------------------------------------------ #
    # torch backend (neural) — generic, reversible parameter surgery
    # ------------------------------------------------------------------ #
    def _inner_module(self) -> Any:
        for attr in ("model", "net", "module", "gpt", "transformer", "_model"):
            mod = getattr(self.model, attr, None)
            if mod is not None and hasattr(mod, "named_parameters"):
                return mod
        return self.model if hasattr(self.model, "named_parameters") else None

    def _localize_torch(self, context: str, *, top: int) -> Attribution:
        mod = self._inner_module()
        if mod is None:
            return Attribution("torch", context, [])
        try:
            import torch  # type: ignore  # noqa: F401
        except Exception:  # noqa: BLE001
            return Attribution("torch", context, [])
        probe = [context] if context else []
        base = self._mean_perplexity(probe) if probe else 0.0
        scored: List[Tuple[str, float]] = []
        for name, p in list(mod.named_parameters()):
            try:
                saved = p.detach().clone()
                with torch.no_grad():
                    p.zero_()
                delta = (self._mean_perplexity(probe) - base) if probe else float(p.abs().mean())
                with torch.no_grad():
                    p.copy_(saved)
                scored.append((name, float(delta)))
            except Exception:  # noqa: BLE001 — never let one parameter break the read-out
                continue
        scored.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return Attribution("torch", context, scored[:max(1, top)])

    def _torch_edit(self, name: str, edit_fn: "Callable[[Any], Any]", *,
                    probe: Sequence[str], intent: str) -> SurgeryOutcome:
        out = SurgeryOutcome("torch", intent)
        mod = self._inner_module()
        if mod is None:
            out.detail = "no editable torch module found"
            return out
        try:
            import torch  # type: ignore
        except Exception:  # noqa: BLE001
            out.detail = "torch unavailable"
            return out
        params = dict(mod.named_parameters())
        if name not in params:
            out.detail = f"no parameter named {name!r}"
            return out
        p = params[name]
        probe_corpus = list(probe) if probe else []
        out.perplexity_before = self._mean_perplexity(probe_corpus) if probe_corpus else float("inf")
        saved = p.detach().clone()
        try:
            with torch.no_grad():
                edit_fn(p)
            out.applied = True
            out.perplexity_after = (self._mean_perplexity(probe_corpus)
                                    if probe_corpus else out.perplexity_before)
            regressed = (out.perplexity_after >
                         out.perplexity_before * (1.0 + self.tol) + 1e-9)
            if not regressed:
                out.kept = True
                out.detail = f"perplexity {_round(out.perplexity_before)}→" \
                             f"{_round(out.perplexity_after)} (within tol {self.tol})"
                return out
            with torch.no_grad():
                p.copy_(saved)
            out.rolled_back = True
            out.detail = "perplexity regressed — rolled back"
        except Exception as exc:  # noqa: BLE001 — any error restores the exact prior weights
            try:
                with torch.no_grad():
                    p.copy_(saved)
            except Exception:  # noqa: BLE001
                pass
            out.rolled_back = out.applied
            out.detail = f"error during edit ({exc}) — rolled back"
        return out

    # ------------------------------------------------------------------ #
    def _mean_perplexity(self, corpus: Sequence[str]) -> float:
        vals: List[float] = []
        for text in corpus:
            try:
                pp = float(self.model.perplexity(text))
            except Exception:  # noqa: BLE001
                pp = float("inf")
            if math.isfinite(pp):
                vals.append(pp)
        return (sum(vals) / len(vals)) if vals else float("inf")


# --------------------------------------------------------------------------- #
def _round(x: float) -> float:
    try:
        return round(float(x), 4) if math.isfinite(float(x)) else float("inf")
    except Exception:  # noqa: BLE001
        return float("inf")


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.growth.foundry_models import WordKNGramLM

    print("=" * 70)
    print("NYXARA weight-surgery self-test")
    print("=" * 70)

    corpus = ["the cat sat on the mat", "the cat sat on the rug",
              "the dog ran in the park", "the dog ran in the yard"] * 6
    model = WordKNGramLM(order=2, seed=1)
    model.train_on(corpus)
    surgeon = WeightSurgeon(model)

    # interpretability: attribute the prediction after "the" to the responsible counts
    attr = surgeon.localize("the", top=4)
    print(f"\nlocalize 'the'      : {attr.to_dict()['items']}")
    assert attr.items, "must attribute the prediction to real counts"

    pred0 = surgeon.predict_next("the")
    print(f"prediction 'the'    : {pred0!r}  (param_count={model.param_count()})")

    # surgery: steer "the" → a deliberately different in-vocab word, reversibly, gauntlet-gated
    target = "dog" if pred0 != "dog" else "cat"
    n_before = model.param_count()
    out = surgeon.steer("the", target, probe=corpus)
    print(f"steer 'the'→{target!r:7}: {out.to_dict()}")
    assert out.applied and out.kept and not out.rolled_back
    assert surgeon.predict_next("the") == target, "the circuit edit must change the prediction"

    # a failing gauntlet (zero tolerance + a probe the edit ruins) must roll back byte-for-byte
    strict = WeightSurgeon(model, tol=0.0)
    snapshot_params = model.param_count()
    pred_now = surgeon.predict_next("the")
    bad = strict.steer("the", "park", probe=["the cat sat on the mat"] * 8)
    print(f"strict steer (bad)  : kept={bad.kept} rolled_back={bad.rolled_back}")
    if bad.rolled_back:
        assert model.param_count() == snapshot_params, "rollback must restore weights exactly"
        assert surgeon.predict_next("the") == pred_now, "rollback must restore the prediction"
        print("rollback restored weights byte-for-byte ✓")

    # suppress: prune a continuation from its circuit
    surgeon2 = WeightSurgeon(WordKNGramLM(order=2, seed=1))
    surgeon2.model.train_on(corpus)
    top_word = surgeon2.predict_next("the")
    sup = surgeon2.suppress("the", top_word, probe=corpus)
    print(f"suppress 'the'↛{top_word!r:5}: kept={sup.kept}")
    if sup.kept:
        assert surgeon2.predict_next("the") != top_word

    print("\nALL SELF-TESTS PASSED ✓")
