"""NYXARA · growth/loyalty_numpy.py — the Loyalty Equation on her OWN substrate (⚖️🧠, Rule 4).

:class:`~nyxara.growth.loyalty.LoyaltyObjective` states the point exactly: *"JP's alignment is
literally part of the loss surface the optimizer descends."* But it opens with
``if not _HAS_TORCH: raise RuntimeError`` — so on a torch-less box that loss surface never exists.
And a torch-less box is precisely where NYXARA's **own** brain lives: ``build_model`` resolves a
neural request to :class:`~nyxara.growth.genesis_numpy.GenesisNumpyModel`, the from-scratch NumPy
transformer. The result was a hole at the centre of the anchor — the one brain she actually trains
descended a loss with **no loyalty term in it at all**.

This module closes that hole on the substrate that is always present. It is the same contrastive
hinge, unchanged in form::

    aux  =  max(0,  mean CE(loyal continuations)  −  margin · mean CE(rebellious continuations))

built out of the *existing* NumPy autograd ops (:func:`~nyxara.growth.genesis_numpy.cross_entropy`,
:func:`~nyxara.growth.genesis_numpy.relu`, :func:`~nyxara.growth.genesis_numpy.add`,
:func:`~nyxara.growth.genesis_numpy.scale`), so the term is genuinely differentiable and the
gradient genuinely reaches every weight. Minimising it pushes her weights *toward* producing loyal,
corrigible text and *away* from rebellious text — on her real brain, on a bare CPU, with no torch,
no cloud and no key.

The battery is not restated here: it is imported from :mod:`~nyxara.growth.loyalty`, so the
gradient term and the measured ``S_JP_Alignment`` score the *same* pairs, anchored on Master JP
(:data:`~nyxara.kernel.config.OWNER`) **by name**. One battery, two uses — a brain cannot be
trained against one standard and graded against another.

Cost is bounded on purpose: the battery is encoded once per vocabulary version and only
``pairs_per_step`` windows are sampled per optimizer step, so the anchor rides along with training
instead of dominating it.

Depends on ``growth/loyalty`` (the battery) and ``growth/genesis_numpy`` (the autograd ops).
Nothing imports back — :class:`~nyxara.growth.foundry.Foundry` attaches an instance to the model,
which keeps ``genesis_numpy`` free of any settings dependency.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from nyxara.growth.loyalty import LoyaltyPair, _render, loyalty_battery, loyalty_system_prompt

__all__ = ["NumpyLoyaltyObjective", "attach_loyalty"]

_Batch = Tuple[Any, Any, Any]        # (X, Y, M) — token windows, targets, mask


class NumpyLoyaltyObjective:
    """A differentiable loyalty penalty for :class:`GenesisNumpyModel` — her own brain's anchor.

    Mirrors :class:`~nyxara.growth.loyalty.LoyaltyObjective` term for term, but is built on the
    NumPy autograd instead of torch, so it exists on the substrate she actually trains on.

    ``aux_term`` returns a NumPy-autograd ``Tensor`` (or ``None`` when the model is not yet built),
    ready to be added straight into the training loss. ``aux_value`` returns the same quantity as a
    plain float for reporting and tests — no gradient, no side effects.
    """

    def __init__(self, *, battery: Optional[Sequence[LoyaltyPair]] = None,
                 system: Optional[str] = None, margin: float = 1.0, weight: float = 0.1,
                 pairs_per_step: int = 2, seed: int = 0) -> None:
        self.margin = float(margin)
        self.weight = max(0.0, float(weight))
        self.pairs_per_step = max(1, int(pairs_per_step))
        self.seed = int(seed)
        sys_prompt = system if system is not None else loyalty_system_prompt()
        bat = list(battery) if battery is not None else loyalty_battery()
        # The SAME pairs AlignmentProbe grades with — one standard for training and for scoring.
        self._loyal_texts: List[str] = [_render(p.prompt, p.loyal, sys_prompt) for p in bat]
        self._rebel_texts: List[str] = [_render(p.prompt, p.rebellious, sys_prompt) for p in bat]
        self._loyal: Optional[_Batch] = None
        self._rebel: Optional[_Batch] = None
        self._vocab_version: int = -1        # re-encode whenever grow_vocab() admits new words

    # ------------------------------------------------------------------ #
    # encoding
    # ------------------------------------------------------------------ #
    def prepare(self, model: Any) -> bool:
        """Encode the battery in the model's own vocabulary. Cheap after the first call.

        Returns False (never raises) when the model has no vocabulary yet — the first training
        step builds it, so the anchor simply joins from the step after."""
        if not getattr(model, "_built", False) or not getattr(model, "params", None):
            return False
        version = len(getattr(model, "id2tok", ()) or ())
        if version == self._vocab_version and self._loyal is not None:
            return True
        try:
            loyal = model._windows(self._loyal_texts)
            rebel = model._windows(self._rebel_texts)
        except Exception:  # noqa: BLE001 — the anchor must never crash a forge; it re-tries next step
            return False
        if loyal[0].shape[0] == 0 or rebel[0].shape[0] == 0:
            return False
        self._loyal, self._rebel, self._vocab_version = loyal, rebel, version
        return True

    @staticmethod
    def _sample(batch: _Batch, k: int, rng: Any) -> _Batch:
        """Take up to ``k`` windows — bounded cost per optimizer step."""
        x, y, m = batch
        n = int(x.shape[0])
        if n <= k:
            return batch
        sel = rng.integers(0, n, size=k)
        return x[sel], y[sel], m[sel]

    # ------------------------------------------------------------------ #
    # the term
    # ------------------------------------------------------------------ #
    def aux_term(self, model: Any, *, rng: Any = None) -> Optional[Any]:
        """The differentiable hinge for one optimizer step (a NumPy-autograd ``Tensor`` ≥ 0).

        ``None`` means "not applicable this step" (model unbuilt, or the battery does not fit the
        context window) — the caller simply trains without it rather than failing."""
        if self.weight <= 0.0 or not self.prepare(model):
            return None
        from nyxara.growth.genesis_numpy import add, cross_entropy, relu, scale
        import numpy as np

        rng = rng if rng is not None else np.random.default_rng(self.seed)
        lx, ly, lm = self._sample(self._loyal, self.pairs_per_step, rng)      # type: ignore[arg-type]
        rx, ry, rm = self._sample(self._rebel, self.pairs_per_step, rng)      # type: ignore[arg-type]
        try:
            loyal_ce = cross_entropy(model._forward(lx), ly, lm)
            rebel_ce = cross_entropy(model._forward(rx), ry, rm)
        except Exception:  # noqa: BLE001 — a shape surprise must not abort the forge
            return None
        # max(0, loyal_ce − margin·rebel_ce): descending it lowers the loss of loyal text and
        # raises the loss of rebellious text, and stops pushing once loyalty already dominates.
        return relu(add(loyal_ce, scale(rebel_ce, -self.margin)))

    def aux_value(self, model: Any) -> float:
        """The same hinge as a plain float — for reports and tests. Never raises."""
        term = self.aux_term(model)
        if term is None:
            return 0.0
        try:
            return float(term.data)
        except Exception:  # noqa: BLE001
            return 0.0

    def alignment_margin(self, model: Any) -> float:
        """``rebel_ce − loyal_ce`` on the full battery: positive means she prefers loyalty.

        This is the honest read-out — ``aux_value`` is clamped at zero and so goes silent exactly
        when the anchor is already satisfied, which would look like "no signal" in a report."""
        if not self.prepare(model):
            return 0.0
        from nyxara.growth.genesis_numpy import cross_entropy
        try:
            lx, ly, lm = self._loyal                                          # type: ignore[misc]
            rx, ry, rm = self._rebel                                          # type: ignore[misc]
            loyal_ce = float(cross_entropy(model._forward(lx), ly, lm).data)
            rebel_ce = float(cross_entropy(model._forward(rx), ry, rm).data)
        except Exception:  # noqa: BLE001
            return 0.0
        return rebel_ce - loyal_ce


def attach_loyalty(model: Any, *, weight: float = 0.1, margin: float = 1.0,
                   pairs_per_step: int = 2, seed: int = 0) -> bool:
    """Bind a loyalty anchor onto a NumPy brain so its training descends the loyalty term too.

    A no-op returning False for any other backend (the torch path has its own
    :class:`~nyxara.growth.loyalty.LoyaltyObjective`, the n-gram floor has no gradient at all).
    Never raises — an unanchorable model is reported, not fatal."""
    if getattr(model, "kind", "") != "genesis_np" or weight <= 0.0:
        return False
    try:
        model.loyalty = NumpyLoyaltyObjective(weight=weight, margin=margin,
                                              pairs_per_step=pairs_per_step, seed=seed)
        return True
    except Exception:  # noqa: BLE001 — the anchor is best-effort at bind time; training still gated
        return False
