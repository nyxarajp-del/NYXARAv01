"""NYXARA · growth/latent_head.py — thinking a few steps ahead, in latent space (🔮).

Three mechanisms that share one idea: a token-by-token model has no representation of where it
is *going*, and giving it one improves both what it says and how fast it says it.

**1. L-NEURAL-DYNAMICS — a JEPA latent-prediction head.** Alongside the token loss, the model
predicts its own hidden state ``k`` positions ahead, against an **EMA target encoder** that is
never trained by gradient. This is the Joint Embedding Predictive Architecture objective, and
:mod:`~nyxara.mind.jepa_world_model` already implements the full hierarchical version in NumPy —
encoder, EMA target, predictor ensemble, VICReg anti-collapse, and a latent-space ``plan()``.
What was missing was any connection between it and the language model. This is that connection.

The VICReg variance and covariance terms are not optional decoration. Predicting your own
representation has a trivial solution — emit a constant — and without an anti-collapse term the
head finds it, immediately and silently, while the loss looks excellent.

**2. Multi-token prediction — where the actual speedup comes from.** A latent lookahead *adds*
compute per token; it cannot make generation faster, and it would be dishonest to claim
otherwise. What genuinely speeds up decoding is predicting several tokens at once and verifying
them in one pass — self-speculative decoding. ``genesis.py`` already has ``n_predict``,
``mtp_heads`` and ``mtp_weight``; this brings them to the NYX decoder and, crucially,
:func:`measure_speculative_speedup` **measures** the result rather than asserting a number.
Expect roughly 1.5–3×, and report whatever it actually is.

**3. L-EIGENSPACE — a causal bottleneck.** The latent is projected onto the variables of a
learned causal graph, and an auxiliary term rewards transitions consistent with the fitted
mechanism. The honest framing matters here:

    This does **not** make hallucination impossible. The causal graph is *learned* and
    incomplete; anything outside it is ungrounded, and the model must still emit tokens.
    ``growth/dataset_causal.py`` says outright that NOTEARS can invert an edge on a nonlinear
    law, and ``growth/causal_validation.py`` counts spurious edges separately precisely because
    the graph gets things wrong.

What it does do is reduce a *specific, measurable* class of error — claims that contradict a
mechanism she has actually fitted — and :mod:`~nyxara.sim` gives the rare thing needed to check
that: eight deterministic worlds with known closed-form laws, which
:mod:`~nyxara.growth.causal_validation` already uses as ground truth.
:func:`measure_causal_grounding` reports the change; it does not claim zero.

``torch`` is required and imported lazily. Reuses ``mind/jepa_world_model`` and
``growth/causal_validation``. Nothing imports back.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "LatentHeadConfig",
    "LatentHead",
    "MTPHeads",
    "CausalBottleneck",
    "SpeculativeReport",
    "measure_speculative_speedup",
    "measure_causal_grounding",
    "build_latent_head",
]


@dataclass
class LatentHeadConfig:
    """Weights and horizons for the auxiliary objectives. All default to OFF."""

    # JEPA latent prediction
    latent_enabled: bool = False
    latent_dim: int = 256
    horizon: int = 8                  # how many positions ahead to predict
    latent_weight: float = 0.1
    ema_decay: float = 0.996          # the target encoder's momentum
    vicreg_variance_weight: float = 1.0
    vicreg_covariance_weight: float = 0.04
    variance_target: float = 1.0

    # Multi-token prediction (the real decode speedup)
    mtp_enabled: bool = False
    n_predict: int = 4                # 1 = classic next-token only
    mtp_weight: float = 0.3

    # Causal bottleneck
    causal_enabled: bool = False
    causal_vars: int = 16
    causal_weight: float = 0.05


# --------------------------------------------------------------------------- #
# JEPA latent prediction
# --------------------------------------------------------------------------- #
def _make_latent_head(cfg: LatentHeadConfig, n_embd: int):
    import torch
    from torch import nn

    class _LatentHead(nn.Module):
        """Predict the hidden state ``horizon`` positions ahead, in a projected latent space.

        The online encoder ``f_θ`` and the predictor ``g_φ`` are trained; the target encoder
        ``f_θ̄`` is an exponential moving average of the online one and receives **no gradient**.
        That asymmetry is the whole trick: a target that trains alongside the predictor can move
        to meet it, and the pair collapses onto a constant.
        """

        def __init__(self) -> None:
            super().__init__()
            self.cfg = cfg
            self.encoder = nn.Sequential(
                nn.Linear(n_embd, cfg.latent_dim, bias=False), nn.GELU(),
                nn.Linear(cfg.latent_dim, cfg.latent_dim, bias=False))
            self.predictor = nn.Sequential(
                nn.Linear(cfg.latent_dim, cfg.latent_dim, bias=False), nn.GELU(),
                nn.Linear(cfg.latent_dim, cfg.latent_dim, bias=False))
            # The EMA target: same shape, never in the optimizer, updated by copy.
            self.target = nn.Sequential(
                nn.Linear(n_embd, cfg.latent_dim, bias=False), nn.GELU(),
                nn.Linear(cfg.latent_dim, cfg.latent_dim, bias=False))
            self.target.load_state_dict(self.encoder.state_dict())
            for param in self.target.parameters():
                param.requires_grad_(False)

        @torch.no_grad()
        def update_target(self) -> None:
            """``θ̄ ← m·θ̄ + (1-m)·θ``. Called once per optimizer step."""
            m = self.cfg.ema_decay
            for online, target in zip(self.encoder.parameters(), self.target.parameters()):
                target.mul_(m).add_(online.detach(), alpha=1.0 - m)

        def forward(self, hidden):  # type: ignore[override]
            """``(loss, stats)`` for one batch of hidden states ``(B, T, C)``."""
            k = max(1, int(self.cfg.horizon))
            if hidden.shape[1] <= k:
                zero = hidden.sum() * 0.0
                return zero, {"latent": 0.0, "variance": 0.0}

            context = hidden[:, :-k, :]
            future = hidden[:, k:, :].detach()

            predicted = self.predictor(self.encoder(context))
            with torch.no_grad():
                actual = self.target(future)

            # Cosine distance in latent space — scale-free, so the head cannot cheat by shrinking.
            pred_n = torch.nn.functional.normalize(predicted, dim=-1)
            act_n = torch.nn.functional.normalize(actual, dim=-1)
            prediction_loss = (1.0 - (pred_n * act_n).sum(-1)).mean()

            # VICReg variance: every latent dimension must retain spread across the batch. This
            # is the term that forbids the constant solution.
            flat = predicted.reshape(-1, predicted.shape[-1])
            std = torch.sqrt(flat.var(dim=0) + 1e-4)
            variance_loss = torch.relu(self.cfg.variance_target - std).mean()

            # VICReg covariance: decorrelate the dimensions, so the latent does not collapse to
            # a single direction repeated across every channel.
            centered = flat - flat.mean(dim=0)
            n = max(1, centered.shape[0] - 1)
            cov = (centered.T @ centered) / n
            off_diagonal = cov - torch.diag(torch.diag(cov))
            covariance_loss = off_diagonal.pow(2).sum() / max(1, flat.shape[-1])

            loss = (prediction_loss
                    + self.cfg.vicreg_variance_weight * variance_loss
                    + self.cfg.vicreg_covariance_weight * covariance_loss)
            return loss, {"latent": float(prediction_loss.detach()),
                          "variance": float(std.mean().detach())}

    return _LatentHead()


def _make_mtp_heads(cfg: LatentHeadConfig, n_embd: int, vocab_size: int):
    import torch
    from torch import nn

    class _MTPHeads(nn.Module):
        """Extra LM heads predicting tokens at offsets 2 … n_predict.

        Two payoffs. During training the auxiliary targets are a denser learning signal per
        position. At inference the heads propose several tokens at once, which the main head
        then verifies in a single forward pass — self-speculative decoding, and the honest
        source of a real speedup.
        """

        def __init__(self) -> None:
            super().__init__()
            self.n_predict = max(1, int(cfg.n_predict))
            self.heads = nn.ModuleList(
                [nn.Linear(n_embd, vocab_size, bias=False)
                 for _ in range(self.n_predict - 1)])

        def forward(self, hidden):  # type: ignore[override]
            return [head(hidden) for head in self.heads]

        def loss(self, hidden, targets):
            """Cross-entropy at offsets 2…n_predict; ``0`` when there are no extra heads.

            Head ``d`` predicts the token ``d`` positions ahead, so the hidden state at position
            ``t`` is scored against ``targets[t + d]``. Both sides are truncated to the ``T - d``
            positions where such a pair exists.
            """
            if not self.heads:
                return hidden.sum() * 0.0
            total = None
            length = targets.shape[1]
            for depth, head in enumerate(self.heads, start=2):
                usable = length - depth
                if usable <= 0:
                    break
                logits = head(hidden[:, :usable, :])
                term = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(),
                    targets[:, depth:depth + usable].reshape(-1))
                total = term if total is None else total + term
            return total if total is not None else hidden.sum() * 0.0

    return _MTPHeads()


def _make_causal_bottleneck(cfg: LatentHeadConfig, n_embd: int):
    import torch
    from torch import nn

    class _CausalBottleneck(nn.Module):
        """Project the latent onto causal variables and reward consistent transitions.

        ``mechanism`` is a learned linear map standing in for the fitted structural equations —
        deliberately linear, matching what ``mind/neural_causal`` (NOTEARS) fits, and carrying
        the same honest limit: a linear structural model can invert the direction of a nonlinear
        law. The auxiliary term rewards agreement with that mechanism; it cannot certify truth.
        """

        def __init__(self) -> None:
            super().__init__()
            self.to_causal = nn.Linear(n_embd, cfg.causal_vars, bias=False)
            self.mechanism = nn.Linear(cfg.causal_vars, cfg.causal_vars, bias=False)
            # A causal graph has no self-loops; zeroing the diagonal keeps the mechanism from
            # satisfying itself with the identity, which would make the term meaningless.
            self.register_buffer("no_self_loops",
                                 1.0 - torch.eye(cfg.causal_vars))

        def forward(self, hidden):  # type: ignore[override]
            if hidden.shape[1] < 2:
                zero = hidden.sum() * 0.0
                return zero, {"causal": 0.0}
            variables = self.to_causal(hidden)
            weight = self.mechanism.weight * self.no_self_loops
            predicted = variables[:, :-1, :] @ weight.T
            actual = variables[:, 1:, :]
            consistency = nn.functional.mse_loss(predicted, actual.detach())
            # Sparsity: a dense "graph" explains everything and therefore explains nothing.
            sparsity = weight.abs().mean()
            loss = consistency + 0.01 * sparsity
            return loss, {"causal": float(consistency.detach())}

    return _CausalBottleneck()


class LatentHead:
    """Bundles the three auxiliary objectives and attaches them to a model.

    :meth:`auxiliary_loss` returns ``(loss, stats)`` to add to the token loss. Disabled
    objectives contribute exactly zero, so leaving them off costs nothing.
    """

    def __init__(self, model: Any, *, config: Optional[LatentHeadConfig] = None) -> None:
        import torch

        self.model = model
        self.cfg = config or LatentHeadConfig()
        self.net = getattr(model, "net", model)
        self.device = getattr(model, "device", "cpu")
        spec = getattr(model, "spec", None)
        n_embd = int(getattr(spec, "n_embd", 0) or getattr(self.net, "n_embd", 64))
        vocab = int(getattr(spec, "vocab_size", 256))

        self.latent = (_make_latent_head(self.cfg, n_embd).to(self.device)
                       if self.cfg.latent_enabled else None)
        self.mtp = (_make_mtp_heads(self.cfg, n_embd, vocab).to(self.device)
                    if self.cfg.mtp_enabled else None)
        self.causal = (_make_causal_bottleneck(self.cfg, n_embd).to(self.device)
                       if self.cfg.causal_enabled else None)

    def parameters(self) -> List[Any]:
        """Trainable parameters of every enabled head — to add to the optimizer."""
        out: List[Any] = []
        for module in (self.latent, self.mtp, self.causal):
            if module is not None:
                out.extend(p for p in module.parameters() if p.requires_grad)
        return out

    def hidden_states(self, ids: Any) -> Any:
        """The final hidden states for ``ids`` — the decoder's stack minus the LM head."""
        import torch

        net = self.net
        x = net.tok(ids)
        if getattr(net, "pos", None) is not None:
            x = x + net.pos(torch.arange(ids.shape[1], device=ids.device))[None]
        cos, sin = net._rope(ids.shape[1], ids.device, x.dtype)
        for layer in net.layers:
            x, _cache, _aux = layer(x, cos, sin, None)
        return net.norm_f(x)

    def auxiliary_loss(self, hidden: Any, targets: Optional[Any] = None
                       ) -> Tuple[Any, Dict[str, float]]:
        """``(total auxiliary loss, stats)``. Zero when nothing is enabled."""
        import torch

        total = hidden.sum() * 0.0
        stats: Dict[str, float] = {}

        if self.latent is not None:
            loss, info = self.latent(hidden)
            total = total + self.cfg.latent_weight * loss
            stats.update(info)
        if self.mtp is not None and targets is not None:
            loss = self.mtp.loss(hidden, targets)
            total = total + self.cfg.mtp_weight * loss
            stats["mtp"] = float(loss.detach())
        if self.causal is not None:
            loss, info = self.causal(hidden)
            total = total + self.cfg.causal_weight * loss
            stats.update(info)
        return total, stats

    def step(self) -> None:
        """Post-optimizer housekeeping — advance the EMA target encoder."""
        if self.latent is not None:
            self.latent.update_target()


def build_latent_head(model: Any, **kwargs: Any) -> LatentHead:
    """Convenience constructor: ``build_latent_head(model, latent_enabled=True, ...)``."""
    return LatentHead(model, config=LatentHeadConfig(**kwargs))


# --------------------------------------------------------------------------- #
# Measurement — the numbers reported are the ones observed
# --------------------------------------------------------------------------- #
@dataclass
class SpeculativeReport:
    """Measured decode throughput with and without multi-token speculation."""

    baseline_tokens_per_sec: float = 0.0
    speculative_tokens_per_sec: float = 0.0
    accepted: int = 0
    proposed: int = 0
    tokens: int = 0

    @property
    def speedup(self) -> float:
        if self.baseline_tokens_per_sec <= 0:
            return 0.0
        return self.speculative_tokens_per_sec / self.baseline_tokens_per_sec

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    def summary(self) -> str:
        return (f"· speculative decoding: {self.speedup:.2f}x "
                f"({self.baseline_tokens_per_sec:.0f} -> "
                f"{self.speculative_tokens_per_sec:.0f} tok/s), "
                f"draft acceptance {self.acceptance_rate:.0%}")

    def to_dict(self) -> Dict[str, Any]:
        return {"baseline_tokens_per_sec": round(self.baseline_tokens_per_sec, 2),
                "speculative_tokens_per_sec": round(self.speculative_tokens_per_sec, 2),
                "speedup": round(self.speedup, 3),
                "acceptance_rate": round(self.acceptance_rate, 4),
                "accepted": self.accepted, "proposed": self.proposed, "tokens": self.tokens}


def measure_speculative_speedup(model: Any, head: LatentHead, prompt: str = "The",
                                *, max_tokens: int = 64) -> SpeculativeReport:
    """Measure decode throughput with and without MTP speculation. No number is assumed.

    Whatever this returns is what gets reported. An untrained MTP head proposes badly, its drafts
    are rejected, and the "speedup" is below 1.0 — which is the correct and useful answer, not a
    failure of the measurement.
    """
    import torch

    report = SpeculativeReport(tokens=max_tokens)
    if head.mtp is None:
        return report

    start = time.monotonic()
    model.generate(prompt, max_tokens=max_tokens, greedy=True, use_cache=True)
    baseline = time.monotonic() - start
    report.baseline_tokens_per_sec = max_tokens / max(1e-9, baseline)

    ids = model._encode(prompt) or [0]
    block = int(getattr(model.spec, "block_size", 256))
    n_draft = max(1, head.cfg.n_predict - 1)
    produced = 0
    start = time.monotonic()
    with torch.no_grad():
        while produced < max_tokens:
            ctx = torch.tensor(ids[-block:], dtype=torch.long,
                               device=head.device).unsqueeze(0)
            hidden = head.hidden_states(ctx)
            main = head.net.head(hidden)[:, -1, :]
            first = int(torch.argmax(main, dim=-1).item())
            drafts = [int(torch.argmax(logits[:, -1, :], dim=-1).item())
                      for logits in head.mtp(hidden)[:n_draft]]

            ids.append(first)
            produced += 1
            report.proposed += len(drafts)

            # Verify the whole draft in ONE forward pass: that is where the time is saved.
            if drafts and produced < max_tokens:
                trial = ids[-block:] + drafts
                verify = torch.tensor(trial[-block:], dtype=torch.long,
                                      device=head.device).unsqueeze(0)
                verified = torch.argmax(head.net.head(head.hidden_states(verify)), dim=-1)[0]
                for offset, draft in enumerate(drafts):
                    if produced >= max_tokens:
                        break
                    position = len(trial) - len(drafts) + offset - 1
                    if 0 <= position < verified.shape[0] and int(verified[position]) == draft:
                        ids.append(draft)
                        produced += 1
                        report.accepted += 1
                    else:
                        break        # one rejection invalidates every draft after it
    report.speculative_tokens_per_sec = produced / max(1e-9, time.monotonic() - start)
    return report


def measure_causal_grounding(predicate: Callable[[str], bool], *,
                             answers_with: Sequence[str],
                             answers_without: Sequence[str]) -> Dict[str, Any]:
    """Compare causally-impossible claim rates with the bottleneck on and off.

    ``predicate`` returns True when an answer contradicts a known mechanism — the natural
    source is :mod:`~nyxara.sim`'s closed-form worlds, whose laws are true by construction and
    which :mod:`~nyxara.growth.causal_validation` already uses as ground truth.

    Deliberately reports a *reduction*, never an absence. The causal graph is learned and
    incomplete, so claims outside it stay ungrounded; "zero hallucination" is not a number this
    or any other mechanism can produce.
    """
    def rate(answers: Sequence[str]) -> float:
        if not answers:
            return 0.0
        return sum(1 for a in answers if predicate(a)) / len(answers)

    with_rate = rate(answers_with)
    without_rate = rate(answers_without)
    reduction = (without_rate - with_rate) / without_rate if without_rate > 0 else 0.0
    return {
        "impossible_rate_without_bottleneck": round(without_rate, 4),
        "impossible_rate_with_bottleneck": round(with_rate, 4),
        "relative_reduction": round(reduction, 4),
        "n_with": len(answers_with),
        "n_without": len(answers_without),
        "note": ("a reduction in causally-impossible claims, not their elimination — the causal "
                 "graph is learned and incomplete, so anything outside it stays ungrounded"),
    }
