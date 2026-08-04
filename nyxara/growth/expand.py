"""NYXARA · growth/expand.py — she grows a new expert when a domain will not yield (🌱→🧠).

L-TOPOLOGY (:mod:`~nyxara.growth.foundry_models`) gives her a *fixed* bank of experts. This is
the next step: when a domain stays stubbornly weak — number theory, say, or a language she keeps
failing — she **spawns a new expert sub-network**, trains it on that domain, proves it helped,
and only then admits it to the router. Her capacity grows to fit the problem instead of being
fixed at build time.

The hard part is already solved in this repo. :func:`~nyxara.growth.genesis.inherit_compatible_weights`
is a Net2Net / network-morphism transfer that copies every tensor matching by name and shape, so
"a mutated child resumes its parent's training instead of restarting from noise". Adding an
expert therefore leaves the rest of the brain exactly as it was.

**Three things that make this work rather than merely sound good.**

*A fresh expert is random, so it produces garbage.* Wiring one into the router the moment it is
created makes the model immediately worse. The cycle is therefore spawn → train → **prove** →
admit, and a candidate that fails is discarded with the router restored bit-for-bit.

*The router has to grow too.* ``_SparseMoE.router`` is a ``Linear(n_embd, n_experts)``. Adding
an expert without adding its router row means routing can never reach it, and it sits there as
dead weight forever — a failure that shows up as "the new expert did nothing", with no error
anywhere.

*Capacity grows linearly in what you add, and nothing more.* Twenty 10M experts is +200M
parameters. It is genuinely useful — organic growth, aimed where the measurements say it is
needed — but a sparse model still behaves roughly like a dense one of size ``√(total × active)``,
so this does not conjure a multi-billion-parameter brain out of a 300M one. Every spawn
re-reports both numbers so that stays visible.

Requires ``torch``. Depends on ``growth/foundry_models`` for the MoE layer and
``growth/trainer`` for the training step. Nothing imports back.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "ExpansionConfig",
    "ExpansionReport",
    "SelfExpandingMoE",
    "weakest_domain",
]


@dataclass
class ExpansionConfig:
    """When to grow, how hard to try, and what counts as proof that it worked."""

    # Spawn only when a domain's loss exceeds the mean by this much — growth is for a domain
    # that will not yield, not for ordinary variance between domains.
    weakness_margin: float = 0.15
    expert_hidden: int = 0            # 0 -> match the layer's existing experts
    train_steps: int = 200
    lr: float = 3e-4
    batch_size: int = 8
    # The target domain must improve by at least this fraction to justify the new parameters.
    min_improvement: float = 0.02
    # ...and no other domain may regress by more than this. Mirrors the foundry's
    # capability_regression_tol: capability bought by breaking something else is not a gain.
    max_regression: float = 0.01
    max_experts: int = 32
    eval_batches: int = 8


@dataclass
class ExpansionReport:
    """One spawn attempt, and whether it earned its place."""

    domain: str = ""
    admitted: bool = False
    reason: str = ""
    before: Dict[str, float] = field(default_factory=dict)
    after: Dict[str, float] = field(default_factory=dict)
    params_added: int = 0
    total_params: int = 0
    active_params: int = 0
    layers_grown: List[int] = field(default_factory=list)
    seconds: float = 0.0

    def improvement(self) -> float:
        before, after = self.before.get(self.domain), self.after.get(self.domain)
        if not before or after is None:
            return 0.0
        return (before - after) / before

    def summary(self) -> str:
        verdict = "ADMITTED" if self.admitted else "DISCARDED"
        head = f"· expert for '{self.domain}': {verdict} — {self.reason}"
        if self.before and self.after:
            deltas = ", ".join(
                f"{d} {self.before.get(d, 0):.3f}->{self.after.get(d, 0):.3f}"
                for d in sorted(self.before) if d != "mean")
            head += f"\n  {deltas}"
        if self.admitted:
            head += (f"\n  +{self.params_added:,} parameters "
                     f"({self.total_params:,} total / {self.active_params:,} active per token)")
        return head

    def to_dict(self) -> Dict[str, Any]:
        return {"domain": self.domain, "admitted": self.admitted, "reason": self.reason,
                "before": dict(self.before), "after": dict(self.after),
                "improvement": round(self.improvement(), 5),
                "params_added": self.params_added, "total_params": self.total_params,
                "active_params": self.active_params,
                "layers_grown": list(self.layers_grown), "seconds": round(self.seconds, 2)}


def weakest_domain(losses: Dict[str, float], *, margin: float = 0.15) -> Optional[str]:
    """The domain worth growing for, or ``None`` when none is far enough behind.

    Requires the worst domain to exceed the mean of the others by ``margin``, so ordinary
    variation between domains does not trigger a spawn. Growth should answer a *measurement*,
    not noise.
    """
    scored = {k: v for k, v in losses.items() if k != "mean" and v is not None}
    if len(scored) < 2:
        return None
    worst = max(scored, key=lambda k: scored[k])
    others = [v for k, v in scored.items() if k != worst]
    baseline = sum(others) / len(others)
    if baseline <= 0:
        return None
    return worst if (scored[worst] - baseline) / baseline > margin else None


class SelfExpandingMoE:
    """Spawns, trains, proves and admits new experts on a model's MoE layers."""

    def __init__(self, model: Any, *, config: Optional[ExpansionConfig] = None) -> None:
        self.model = model
        self.cfg = config or ExpansionConfig()
        self.net = getattr(model, "net", model)
        self.device = getattr(model, "device", "cpu")
        self.history: List[ExpansionReport] = []

    # ---- discovery ---- #
    def moe_layers(self) -> List[Tuple[int, Any]]:
        """``(index, module)`` for every MoE mixer in the stack."""
        out = []
        for i, layer in enumerate(getattr(self.net, "layers", []) or []):
            mixer = getattr(layer, "mlp", None)
            if mixer is not None and hasattr(mixer, "experts") and hasattr(mixer, "router"):
                out.append((i, mixer))
        return out

    def expert_count(self) -> int:
        layers = self.moe_layers()
        return len(layers[0][1].experts) if layers else 0

    # ---- the grow / restore primitives ---- #
    def _grow(self, mixer: Any) -> int:
        """Add one expert and its router row. Returns the parameters added.

        Growing the router is not bookkeeping. Without a new row the softmax can never select
        the new expert, so it is never trained, never used, and never noticed — it simply sits
        in the checkpoint as dead weight.
        """
        import torch
        from torch import nn

        from nyxara.growth.foundry_models import _SwiGLU

        n_embd = mixer.router.in_features
        template = mixer.experts[0]
        hidden = self.cfg.expert_hidden or template.gate.out_features
        expert = _SwiGLU(n_embd, hidden).to(self.device)

        # Net2Net: seed the newcomer from an existing expert rather than from noise, so it
        # starts as a working mixer and specialises from there instead of relearning the basics.
        try:
            from nyxara.growth.genesis import inherit_compatible_weights
            inherit_compatible_weights(expert, template.state_dict())
        except Exception:  # noqa: BLE001 — a fresh expert still trains, just more slowly
            pass

        mixer.experts.append(expert)
        mixer.n_experts = len(mixer.experts)

        old = mixer.router
        router = nn.Linear(n_embd, mixer.n_experts, bias=False).to(self.device)
        with torch.no_grad():
            router.weight[:old.out_features] = old.weight
            # Start the new row near zero: the router should *learn* to reach the new expert
            # from evidence, not begin by diverting traffic to an untrained one.
            router.weight[old.out_features:] = old.weight.mean(0, keepdim=True) * 0.01
        mixer.router = router
        return sum(p.numel() for p in expert.parameters()) + n_embd

    @staticmethod
    def _snapshot(mixer: Any) -> Dict[str, Any]:
        return {"n_experts": mixer.n_experts,
                "router": copy.deepcopy(mixer.router),
                "experts": list(mixer.experts)}

    @staticmethod
    def _restore(mixer: Any, snapshot: Dict[str, Any]) -> None:
        """Put the mixer back exactly as it was — a rejected expert must leave no trace."""
        from torch import nn

        mixer.n_experts = snapshot["n_experts"]
        mixer.router = snapshot["router"]
        mixer.experts = nn.ModuleList(snapshot["experts"])

    # ---- the cycle ---- #
    def expand(self, dataset: Any, *, domain: Optional[str] = None,
               evaluate: Optional[Callable[[], Dict[str, float]]] = None
               ) -> ExpansionReport:
        """Spawn → train → prove → admit, for the weakest domain (or a named one).

        Returns an :class:`ExpansionReport` either way. A candidate that does not clear the bar
        is discarded and the model is restored, so a failed expansion costs time and nothing else.
        """
        import torch
        from torch import nn

        from nyxara.growth.foundry_models import estimate_params
        from nyxara.growth.trainer import TrainConfig, evaluate_per_domain

        report = ExpansionReport()
        start = time.monotonic()
        cfg = self.cfg
        spec = getattr(self.model, "spec", None)
        vocab = int(getattr(spec, "vocab_size", 256))

        layers = self.moe_layers()
        if not layers:
            report.reason = "the model has no MoE layers to grow"
            return report
        if self.expert_count() >= cfg.max_experts:
            report.reason = f"already at max_experts={cfg.max_experts}"
            return report

        scorer = evaluate or (lambda: evaluate_per_domain(
            self.model, dataset, TrainConfig(batch_size=cfg.batch_size,
                                             eval_batches=cfg.eval_batches),
            vocab_size=vocab))

        report.before = scorer()
        target = domain or weakest_domain(report.before, margin=cfg.weakness_margin)
        if not target:
            report.reason = ("no domain is far enough behind to justify new parameters "
                             f"(margin={cfg.weakness_margin})")
            report.seconds = time.monotonic() - start
            return report
        report.domain = target

        snapshots = [(mixer, self._snapshot(mixer)) for _i, mixer in layers]
        added = 0
        for index, mixer in layers:
            added += self._grow(mixer)
            report.layers_grown.append(index)
        report.params_added = added

        # Train ONLY the new experts and the routers. Touching the rest of the brain would make
        # any measured change unattributable — and could quietly degrade the other domains.
        trainable: List[Any] = []
        for _i, mixer in layers:
            trainable.extend(mixer.experts[-1].parameters())
            trainable.extend(mixer.router.parameters())
        for param in self.net.parameters():
            param.requires_grad_(False)
        for param in trainable:
            param.requires_grad_(True)

        optimizer = torch.optim.AdamW(trainable, lr=cfg.lr)
        self.net.train()
        try:
            for _step in range(max(1, cfg.train_steps)):
                x_np, y_np = dataset.batch(cfg.batch_size, target)
                x = torch.from_numpy(x_np).to(self.device)
                y = torch.from_numpy(y_np).to(self.device)
                logits, aux = self._forward(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab).float(), y.reshape(-1))
                if aux is not None:
                    loss = loss + 0.01 * aux
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        finally:
            for param in self.net.parameters():
                param.requires_grad_(True)

        report.after = scorer()

        improvement = report.improvement()
        regressions = {
            d: (report.after.get(d, 0.0) - before) / before
            for d, before in report.before.items()
            if d not in (target, "mean") and before > 0
        }
        worst_regression = max(regressions.values()) if regressions else 0.0

        if improvement < cfg.min_improvement:
            report.reason = (f"'{target}' improved only {improvement:.1%} "
                             f"(needs {cfg.min_improvement:.1%}) — the new parameters did not "
                             f"earn their place")
        elif worst_regression > cfg.max_regression:
            hurt = max(regressions, key=lambda k: regressions[k])
            report.reason = (f"'{target}' improved {improvement:.1%} but '{hurt}' regressed "
                             f"{worst_regression:.1%} — capability bought by breaking something "
                             f"else is not a gain")
        else:
            report.admitted = True
            report.reason = (f"'{target}' improved {improvement:.1%} with no domain regressing "
                             f"more than {cfg.max_regression:.1%}")

        if not report.admitted:
            for mixer, snapshot in snapshots:
                self._restore(mixer, snapshot)
            report.params_added = 0
            report.layers_grown = []

        if spec is not None:
            spec.n_experts = self.expert_count()
            estimate = estimate_params(spec)
            report.total_params = estimate.total
            report.active_params = estimate.active

        report.seconds = time.monotonic() - start
        self.history.append(report)
        return report

    def _forward(self, x: Any) -> Tuple[Any, Any]:
        out = self.net(x, return_aux=True)
        if isinstance(out, tuple) and len(out) == 3:
            return out[0], out[2]
        return out, None
