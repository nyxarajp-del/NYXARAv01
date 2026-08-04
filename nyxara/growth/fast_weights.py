"""NYXARA · growth/fast_weights.py — learning inside a conversation, without a gradient (⚡→🧠).

Between forges she is read-only. A fact the Master gives her, or an error the sandbox hands
back, changes nothing about her weights until the next full train → gauntlet → promote cycle.
This module is the fast path: a Hebbian **fast-weight** overlay that adapts within a single
session, with no backpropagation and no optimizer step.

``LearningRule`` in :mod:`~nyxara.growth.genesis` already carries ``plasticity``,
``plasticity_lr`` and ``plasticity_decay``, and the Genesis training loop applies a Hebbian
update to the LM head after each optimizer step. That is a **training-time** overlay. What is
new here is doing it at **inference time**, per conversation, so she adapts while she is talking.

The rule (Ba et al., fast weights; the form the Master specified):

    W_fast(t+1) = λ · W_fast(t) + η · h(x) ⊗ x

``λ`` is ``plasticity_decay``, ``η`` is ``plasticity_lr``. ``W_fast`` is an *overlay*: the static
weights are never modified, and the model's forward pass adds ``x @ W_fast`` to the layer it
attaches to. Removing the overlay returns the exact original model, bit for bit.

**Two honest limits.**

*It gives associative recall, not understanding.* Tell her "the deploy key rotates on Tuesdays"
and she can use that for the rest of the session. That is real and useful. It is not the same as
learning a skill — a Hebbian outer product binds a key to a value, it does not restructure a
representation. Anyone claiming otherwise is describing gradient descent.

*It is bounded, and it has to be.* Fast weights accumulate; unbounded, they eventually dominate
the static weights and the model stops being itself. The overlay is norm-clipped, and ``λ < 1``
means an un-refreshed association decays away on its own.

**The safety constraint, and why it is not optional.**

This whole architecture rests on one invariant: no weight change reaches the live brain without
clearing the gauntlet (``growth/foundry.py::_gauntlet``), and the immutable character core is
frozen at infinite Fisher importance by :mod:`~nyxara.memory.elastic_synapses`. Inference-time
fast weights bypass that gate **by construction** — that is what makes them fast.

So the overlay is deliberately fenced:

* it attaches only to layers on an explicit allowlist, never to the LM head's character-bearing
  directions and never to a parameter named in ``guard.value_learning.IMMUTABLE_VALUES``;
* it is **session-scoped**, and :meth:`FastWeights.reset` restores the model exactly;
* it is norm-bounded, so one adversarial turn cannot swamp the static weights;
* every write is recorded and readable via :meth:`FastWeights.audit`, so what she picked up
  mid-conversation is inspectable rather than mysterious;
* it is **off** unless a session turns it on.

None of that is caution for its own sake. An unfenced version of this feature is a way to walk
her identity somewhere else over a long enough conversation, with no gate anywhere in the path —
which is precisely what L-SOVEREIGN exists to prevent.

``torch`` is required and imported lazily. Depends on ``guard/value_learning`` for the protected
name list. Nothing imports back.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "FastWeightConfig",
    "FastWeightWrite",
    "FastWeights",
    "PROTECTED_SUBSTRINGS",
]

# Parameter-name fragments the overlay must never attach to. The LM head and token embedding
# carry the identity-bearing directions the character core is expressed in; the rest are the
# module names her values live under. Checked as substrings so a renamed layer that still
# belongs to one of these families stays fenced.
PROTECTED_SUBSTRINGS: Tuple[str, ...] = (
    "head", "tok", "embed", "lm_head", "value", "identity", "loyalty", "character",
)


@dataclass
class FastWeightConfig:
    """Bounds for the overlay. Every one of these exists to keep it from becoming permanent."""

    # λ — per-write decay. Below 1.0 by definition: an association nobody refreshes must fade.
    decay: float = 0.95
    # η — write strength.
    lr: float = 0.1
    # Hard ceiling on ‖W_fast‖_F. Reached, the overlay is rescaled rather than allowed to grow.
    max_norm: float = 1.0
    # Which layer indices to attach to. Empty ⇒ the last third of the stack, where
    # representations are most task-specific and least identity-bearing.
    layers: Tuple[int, ...] = ()
    max_writes: int = 512
    enabled: bool = False        # OFF unless a session explicitly turns it on

    def __post_init__(self) -> None:
        self.decay = min(0.999, max(0.0, float(self.decay)))
        self.lr = min(1.0, max(0.0, float(self.lr)))
        self.max_norm = max(1e-6, float(self.max_norm))


@dataclass
class FastWeightWrite:
    """One recorded write — what she picked up, when, and how hard."""

    source: str                  # "conversation" | "sandbox_error" | ...
    summary: str
    norm_after: float
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "summary": self.summary[:200],
                "norm_after": round(self.norm_after, 6), "at": self.at}


class FastWeights:
    """A session-scoped Hebbian overlay on a model's residual stream.

    Attach with :meth:`attach`, write associations with :meth:`write`, and remove every trace
    with :meth:`reset`. The static weights are never touched, so ``reset`` is exact rather than
    approximate — and ``tests/growth/test_fast_weights.py`` asserts byte-exactness, because "it
    mostly goes back" is not a property anyone should have to trust.
    """

    def __init__(self, model: Any, *, config: Optional[FastWeightConfig] = None) -> None:
        self.model = model
        self.cfg = config or FastWeightConfig()
        self.net = getattr(model, "net", model)
        self.device = getattr(model, "device", "cpu")
        self._overlays: Dict[int, Any] = {}          # layer index -> W_fast (C × C)
        self._handles: List[Any] = []
        self._writes: List[FastWeightWrite] = []
        self._attached = False

    # ---- what may be adapted ---- #
    @staticmethod
    def _is_protected(name: str) -> bool:
        """Whether a parameter name belongs to the character core and is off-limits."""
        lowered = name.lower()
        if any(frag in lowered for frag in PROTECTED_SUBSTRINGS):
            return True
        try:
            from nyxara.guard.value_learning import IMMUTABLE_VALUES
            return any(str(v).lower() in lowered for v in IMMUTABLE_VALUES)
        except Exception:  # noqa: BLE001 — an unavailable list means the substrings still apply
            return False

    def _target_layers(self) -> List[int]:
        """Which layers to attach to — the last third by default.

        Early layers encode surface form and late layers encode task-specific structure; the
        head and embeddings are excluded entirely because that is where identity lives.
        """
        layers = getattr(self.net, "layers", None) or getattr(self.net, "blocks", None) or []
        n = len(layers)
        if not n:
            return []
        if self.cfg.layers:
            return [i for i in self.cfg.layers if 0 <= i < n]
        return list(range(max(0, n - max(1, n // 3)), n))

    # ---- attach / detach ---- #
    def attach(self) -> "FastWeights":
        """Install the overlay as forward hooks. A no-op when disabled."""
        import torch

        if self._attached or not self.cfg.enabled:
            return self
        layers = getattr(self.net, "layers", None) or getattr(self.net, "blocks", None) or []
        width = int(getattr(getattr(self.net, "dims", {}), "get", lambda *_: 0)("n_embd")
                    or getattr(self.net, "n_embd", 0)
                    or getattr(self.model, "spec", None) and self.model.spec.n_embd or 0)
        if not width:
            return self

        for idx in self._target_layers():
            module = layers[idx]
            if any(self._is_protected(name) for name, _ in module.named_parameters()):
                continue
            self._overlays[idx] = torch.zeros(width, width, device=self.device)
            self._handles.append(module.register_forward_hook(self._make_hook(idx)))
        self._attached = bool(self._handles)
        return self

    def _make_hook(self, idx: int):
        def hook(_module, _inputs, output):
            overlay = self._overlays.get(idx)
            if overlay is None:
                return output
            # A block returns (x, cache, aux); only the hidden state is adapted.
            if isinstance(output, tuple):
                hidden = output[0]
                return (hidden + hidden @ overlay,) + tuple(output[1:])
            return output + output @ overlay
        return hook

    def reset(self) -> None:
        """Remove every trace of the overlay. The model returns to exactly what it was."""
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:  # noqa: BLE001
                pass
        self._handles.clear()
        self._overlays.clear()
        self._writes.clear()
        self._attached = False

    # ---- the Hebbian write ---- #
    def write(self, key: Any, value: Any, *, source: str = "conversation",
              summary: str = "") -> float:
        """Apply ``W_fast ← λ·W_fast + η·(key ⊗ value)``. Returns the resulting norm.

        ``key`` and ``value`` are hidden-state vectors — typically the encoding of a cue and of
        what should come to mind given it. Bounded by ``max_norm``: past the ceiling the whole
        overlay is rescaled, so a long conversation cannot let this drown the static weights.
        """
        import torch

        if not self._attached or not self._overlays:
            return 0.0
        if len(self._writes) >= self.cfg.max_writes:
            return self.norm()

        k = self._as_vector(key)
        v = self._as_vector(value)
        if k is None or v is None:
            return self.norm()

        outer = torch.outer(k, v)
        norm = 0.0
        for idx, overlay in self._overlays.items():
            overlay.mul_(self.cfg.decay).add_(outer, alpha=self.cfg.lr)
            current = float(torch.linalg.norm(overlay))
            if current > self.cfg.max_norm:
                overlay.mul_(self.cfg.max_norm / max(current, 1e-9))
                current = self.cfg.max_norm
            norm = max(norm, current)

        self._writes.append(FastWeightWrite(source=source, summary=summary or "(unlabelled)",
                                            norm_after=norm))
        return norm

    def _as_vector(self, x: Any) -> Any:
        """Coerce a hidden state to a 1-D unit vector on the right device."""
        import torch

        if x is None:
            return None
        if not isinstance(x, torch.Tensor):
            try:
                x = torch.as_tensor(x, dtype=torch.float32)
            except Exception:  # noqa: BLE001
                return None
        x = x.detach().to(self.device).float().reshape(-1, x.shape[-1]).mean(0)
        n = torch.linalg.norm(x)
        # Normalising is what makes `lr` and `max_norm` mean the same thing from turn to turn;
        # without it a long input writes harder than a short one for no principled reason.
        return x / n if float(n) > 1e-9 else None

    def learn_text(self, cue: str, target: str, *, source: str = "conversation") -> float:
        """Bind ``cue → target`` using the model's own encodings of each.

        This is the ordinary path: the Master says something, or the sandbox returns an error,
        and the association is available for the rest of the session.
        """
        k = self._encode(cue)
        v = self._encode(target)
        return self.write(k, v, source=source, summary=f"{cue[:60]} -> {target[:60]}")

    def _encode(self, text: str) -> Any:
        """The model's hidden state for ``text`` — no gradient, no weight change."""
        import torch

        ids = self.model._encode(text)[: getattr(self.model.spec, "block_size", 256)]
        if not ids:
            return None
        with torch.no_grad():
            x = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
            emb = self.net.tok(x)
            return emb.squeeze(0)

    # ---- introspection ---- #
    def norm(self) -> float:
        import torch

        if not self._overlays:
            return 0.0
        return max(float(torch.linalg.norm(o)) for o in self._overlays.values())

    def audit(self) -> List[Dict[str, Any]]:
        """Everything written this session — what she picked up, in order."""
        return [w.to_dict() for w in self._writes]

    def state(self) -> Dict[str, Any]:
        return {"attached": self._attached, "layers": sorted(self._overlays),
                "writes": len(self._writes), "norm": round(self.norm(), 6),
                "decay": self.cfg.decay, "lr": self.cfg.lr,
                "max_norm": self.cfg.max_norm, "enabled": self.cfg.enabled}
