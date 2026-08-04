"""NYXARA · growth/trainer.py — the pretraining loop a 300M model actually needs (🔥→🧠).

:meth:`NanoGPTModel.train_on <nyxara.growth.foundry_models.NanoGPTModel.train_on>` samples one
random window, at batch size **1**, with a fixed learning rate of ``3e-3`` and no schedule, no
warmup, no gradient clipping, no accumulation and no mixed precision. For a 137k-parameter model
trained on a few thousand of her own sentences that is entirely adequate. For 300M parameters
over billions of tokens every one of those choices is wrong, and ``3e-3`` in particular will
diverge within the first few hundred steps.

Note that :class:`~nyxara.growth.foundry_models.ModelSpec` **already declares** ``batch_size``,
``grad_accum_steps``, ``warmup_ratio``, ``lr_scheduler``, ``weight_decay``, the Adam betas,
``max_grad_norm`` and ``gradient_checkpointing``. The LoRA backend honours all of them. The
from-scratch backend silently ignored every one. This module is where they finally mean
something.

What it adds beyond honouring those:

* **Decoupled weight decay on the right tensors only.** Decay applies to the 2-D matrices and
  never to norms, biases or embeddings. Decaying an embedding table pulls unseen tokens toward
  zero, which is a slow, quiet way to damage a model.
* **bf16 autocast**, with a GradScaler for fp16, and ``torch.compile`` when available.
* **Checkpoint and resume every N steps** — model, optimizer, scheduler, step, RNG *and the
  data cursor*. A 300M run is measured in days and a Colab session is cut at twelve hours;
  without the cursor a resumed run silently restarts the epoch, over-training the first shard.
* **Per-domain validation**, so "good at everything" is visible per domain instead of hidden
  inside one averaged loss.
* **:func:`preflight` — an honest refusal.** Asked to pretrain 300M on a machine that cannot,
  this says so, with the arithmetic, and exits. It does not start a run that would finish in
  geological time. Silently hanging is the worst outcome available and it is what almost every
  naive script does.

The EWC hook (``model.synapses``) is preserved exactly, so anti-forgetting still works.

``torch`` is required for training and imported lazily; :func:`preflight` and
:func:`estimate_training_time` are pure and work anywhere. Nothing imports back.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "TrainConfig",
    "PreflightVerdict",
    "preflight",
    "estimate_training_time",
    "pretrain",
    "TrainingState",
]

# Rough throughput anchors, in tokens/second, for a 300M dense model at bf16. Order-of-magnitude
# only — they exist to answer "days or years?", which is the question that actually matters when
# deciding whether to start. Scaled by parameter count in estimate_training_time.
_TOKENS_PER_SEC_300M = {
    "cuda-80gb": 120_000.0,       # H100/A100-80G class
    "cuda-40gb": 55_000.0,        # A100-40G class
    "cuda-24gb": 22_000.0,        # 4090 / 3090 class
    "cuda-16gb": 9_000.0,         # T4 / V100 class
    "cpu": 220.0,                 # measured order of magnitude; CPU has no bf16 tensor cores
}

# Bytes per parameter to train: 2 (bf16 weight) + 2 (grad) + 8 (fp32 Adam m and v) + 4 (fp32
# master copy) ≈ 16. Activations are on top and depend on batch and context.
_BYTES_PER_PARAM_TRAINING = 16


@dataclass
class TrainConfig:
    """Optimiser and schedule for a from-scratch pretraining run.

    Defaults are the standard ones for a 300M decoder, not the ``ModelSpec`` defaults (which are
    tuned for the tiny CI model). ``pretrain`` fills unset values from the spec, so a caller can
    pass a spec, this, or a mixture.
    """

    lr: float = 3e-4
    min_lr_ratio: float = 0.1          # cosine floors here rather than at zero
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.01
    lr_scheduler: str = "cosine"
    batch_size: int = 8
    grad_accum_steps: int = 1
    max_steps: int = 1000
    eval_every: int = 100
    eval_batches: int = 8
    checkpoint_every: int = 500
    log_every: int = 10
    dtype: str = "auto"                # "auto" | "bf16" | "fp16" | "fp32"
    compile_model: bool = True
    gradient_checkpointing: bool = False
    seed: int = 0

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.grad_accum_steps


# --------------------------------------------------------------------------- #
# Preflight — the honest refusal
# --------------------------------------------------------------------------- #
@dataclass
class PreflightVerdict:
    """Whether this machine can genuinely run the requested pretraining, and the arithmetic."""

    ok: bool
    device: str = "cpu"
    reason: str = ""
    params: int = 0
    active_params: int = 0
    tokens: int = 0
    estimated_seconds: float = 0.0
    memory_needed_gb: float = 0.0
    memory_available_gb: float = 0.0
    suggestion: str = ""

    def human_time(self) -> str:
        s = self.estimated_seconds
        if s < 3600:
            return f"{s / 60:.0f} minutes"
        if s < 86_400:
            return f"{s / 3600:.1f} hours"
        if s < 86_400 * 90:
            return f"{s / 86_400:.1f} days"
        return f"{s / (86_400 * 365):.1f} YEARS"

    def summary(self) -> str:
        head = (f"{self.params:,} parameters"
                + (f" ({self.active_params:,} active)"
                   if self.active_params and self.active_params != self.params else "")
                + f" × {self.tokens:,} tokens on {self.device}")
        verdict = "OK" if self.ok else "REFUSED"
        lines = [f"· preflight {verdict}: {head}",
                 f"  estimated wall-clock : {self.human_time()}",
                 f"  memory needed/avail  : {self.memory_needed_gb:.1f} GB / "
                 f"{self.memory_available_gb:.1f} GB"]
        if self.reason:
            lines.append(f"  reason               : {self.reason}")
        if self.suggestion:
            lines.append(f"  suggestion           : {self.suggestion}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "device": self.device, "reason": self.reason,
                "params": self.params, "active_params": self.active_params,
                "tokens": self.tokens,
                "estimated_seconds": round(self.estimated_seconds, 1),
                "human_time": self.human_time(),
                "memory_needed_gb": round(self.memory_needed_gb, 2),
                "memory_available_gb": round(self.memory_available_gb, 2),
                "suggestion": self.suggestion}


def _device_class() -> Tuple[str, float]:
    """``(throughput key, available memory GB)`` for this machine."""
    try:
        import torch
        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gb >= 70:
                return "cuda-80gb", gb
            if gb >= 35:
                return "cuda-40gb", gb
            if gb >= 20:
                return "cuda-24gb", gb
            return "cuda-16gb", gb
    except Exception:  # noqa: BLE001 — no torch, or a CUDA query that failed: treat as CPU
        pass
    return "cpu", _system_ram_gb()


def _system_ram_gb() -> float:
    """Total system RAM in GB, read from /proc — no psutil dependency."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1e6
    except Exception:  # noqa: BLE001
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except Exception:  # noqa: BLE001
        return 8.0


def estimate_training_time(*, active_params: int, tokens: int,
                           device_key: str = "cpu") -> float:
    """Estimated wall-clock seconds. Order-of-magnitude, and that is the point.

    Throughput scales roughly inversely with the parameters actually executed per token — which
    is why a sparse model is genuinely faster here, and why ``active_params`` rather than total
    is the right input.
    """
    base = _TOKENS_PER_SEC_300M.get(device_key, _TOKENS_PER_SEC_300M["cpu"])
    scale = 299_418_624 / max(1, active_params)
    return tokens / max(1e-6, base * scale)


def preflight(spec: Any, *, tokens: int, batch_size: int = 8, block_size: Optional[int] = None,
              max_seconds: float = 30 * 86_400,
              device_key: Optional[str] = None,
              memory_available_gb: Optional[float] = None) -> PreflightVerdict:
    """Decide whether this machine can genuinely train ``spec``, and say why not.

    Two refusals, both deliberate:

    * **Memory.** Training needs ~16 bytes per parameter for weights, gradients and the Adam
      moments, plus activations. If that does not fit, the run cannot start — better to say so
      than to die with an allocator error thirty seconds in.
    * **Time.** If the estimate exceeds ``max_seconds`` (default 30 days) this refuses. A
      300M-parameter, 6B-token run on a CPU is not slow, it is *years*, and starting it looks
      exactly like starting a run that will finish.

    Pure: pass ``device_key`` and ``memory_available_gb`` to test any machine from any machine.
    """
    from nyxara.growth.foundry_models import estimate_params, resolved_dims

    est = estimate_params(spec)
    dims = resolved_dims(spec)
    block = int(block_size or dims["block_size"])

    if device_key is None or memory_available_gb is None:
        detected_key, detected_gb = _device_class()
        device_key = device_key or detected_key
        memory_available_gb = (memory_available_gb if memory_available_gb is not None
                               else detected_gb)

    # Weights + grads + optimizer state, plus a rough activation term. Activations scale with
    # batch × context × width × depth; the constant folds in the handful of tensors kept live
    # per layer for the backward pass.
    state_gb = est.total * _BYTES_PER_PARAM_TRAINING / 1e9
    activation_gb = (batch_size * block * dims["n_embd"] * dims["n_layer"] * 12) / 1e9
    needed_gb = state_gb + activation_gb

    seconds = estimate_training_time(active_params=est.active, tokens=tokens,
                                     device_key=device_key)

    verdict = PreflightVerdict(
        ok=True, device=device_key, params=est.total, active_params=est.active,
        tokens=tokens, estimated_seconds=seconds,
        memory_needed_gb=needed_gb, memory_available_gb=float(memory_available_gb))

    # Leave headroom: an allocator that is exactly full is an allocator that fails.
    if needed_gb > memory_available_gb * 0.9:
        verdict.ok = False
        verdict.reason = (f"needs ~{needed_gb:.1f} GB to train (weights + gradients + Adam "
                          f"moments + activations) but only {memory_available_gb:.1f} GB is "
                          f"available")
        verdict.suggestion = _suggestion(device_key, batch_size)
        return verdict

    if seconds > max_seconds:
        verdict.ok = False
        verdict.reason = (f"would take about {verdict.human_time()} on this machine "
                          f"({device_key}); the limit is "
                          f"{max_seconds / 86_400:.0f} days")
        verdict.suggestion = _suggestion(device_key, batch_size)
        return verdict

    return verdict


def _suggestion(device_key: str, batch_size: int) -> str:
    if device_key == "cpu":
        return ("train on a CUDA GPU, or prove the pipeline end-to-end on this machine with "
                "--profile nyxara-30m (the same code path at 1/12th the size)")
    if batch_size > 1:
        return (f"reduce batch_size below {batch_size}, enable gradient_checkpointing, or use "
                f"a sparse profile (nyxara-moe-fast runs ~87M active parameters per token)")
    return "use a smaller profile, or a GPU with more memory"


# --------------------------------------------------------------------------- #
# Training state (checkpoint / resume)
# --------------------------------------------------------------------------- #
@dataclass
class TrainingState:
    """Everything needed to resume a run exactly where it stopped."""

    step: int = 0
    tokens_seen: int = 0
    best_val: float = float("inf")
    history: List[Dict[str, Any]] = field(default_factory=list)
    data_cursor: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"step": self.step, "tokens_seen": self.tokens_seen,
                "best_val": self.best_val, "history": self.history[-200:],
                "data_cursor": dict(self.data_cursor)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingState":
        return cls(step=int(d.get("step", 0)), tokens_seen=int(d.get("tokens_seen", 0)),
                   best_val=float(d.get("best_val", float("inf"))),
                   history=list(d.get("history") or []),
                   data_cursor=dict(d.get("data_cursor") or {}))


def _param_groups(model: Any, weight_decay: float) -> List[Dict[str, Any]]:
    """Split parameters into decayed (2-D matrices) and undecayed (norms, biases, embeddings).

    Decaying a norm's gain or an embedding row is a small, standard mistake with a real cost:
    it pulls the representation of every token the batch did not contain toward zero.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2 and "tok" not in name and "embed" not in name and "pos" not in name:
            decay.append(param)
        else:
            no_decay.append(param)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def _lr_factor(step: int, *, total: int, warmup: int, kind: str, min_ratio: float) -> float:
    """Warmup then cosine/linear/constant, floored at ``min_ratio``.

    Same shape as ``LoRAModel._lr_lambda`` (which the LoRA backend already uses), plus a floor:
    a cosine that reaches exactly zero spends its final steps not learning.
    """
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if kind == "constant" or total <= warmup:
        return 1.0
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    if kind == "linear":
        factor = 1.0 - progress
    elif kind == "cosine":
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        factor = 1.0
    return min_ratio + (1.0 - min_ratio) * factor


def _resolve_dtype(requested: str, device: str) -> Any:
    """Pick the autocast dtype: bf16 where supported, fp16 on older CUDA, fp32 on CPU."""
    import torch

    if requested == "fp32" or device == "cpu":
        return None                       # no autocast — CPU bf16 is slower than fp32 here
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:  # noqa: BLE001
        pass
    return torch.float16


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
def pretrain(model: Any, dataset: Any, *, config: Optional[TrainConfig] = None,
             out_dir: Any = None, resume: bool = True,
             val_dataset: Any = None,
             on_log: Optional[Callable[[Dict[str, Any]], None]] = None,
             skip_preflight: bool = False) -> TrainingState:
    """Pretrain ``model`` on ``dataset`` (a :class:`~nyxara.growth.corpus.ShardDataset`).

    Returns the final :class:`TrainingState`. Raises :class:`RuntimeError` when
    :func:`preflight` refuses — loudly, with the arithmetic, rather than starting a run that
    cannot finish.
    """
    import torch
    from torch import nn

    cfg = config or TrainConfig()
    spec = getattr(model, "spec", None)
    net = getattr(model, "net", model)
    device = getattr(model, "device", "cpu")

    if spec is not None and not skip_preflight:
        tokens = cfg.max_steps * cfg.tokens_per_step * getattr(dataset, "block_size", 1024)
        verdict = preflight(spec, tokens=tokens, batch_size=cfg.batch_size,
                            block_size=getattr(dataset, "block_size", None))
        if not verdict.ok:
            raise RuntimeError("pretraining refused by preflight:\n" + verdict.summary())

    out = Path(out_dir) if out_dir else None
    state = TrainingState()
    if resume and out is not None:
        state = _load_checkpoint(net, out, state, device=device) or state
        if state.data_cursor:
            try:
                dataset.load_state_dict(state.data_cursor)
            except Exception:  # noqa: BLE001 — a stale cursor restarts sampling, never crashes
                pass

    torch.manual_seed(cfg.seed + state.step)
    if cfg.gradient_checkpointing and hasattr(net, "gradient_checkpointing_enable"):
        net.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(_param_groups(net, cfg.weight_decay), lr=cfg.lr,
                                  betas=cfg.betas, eps=cfg.eps)
    if resume and out is not None:
        _load_optimizer(optimizer, out, device=device)

    autocast_dtype = _resolve_dtype(cfg.dtype, device)
    scaler = None
    if autocast_dtype is torch.float16:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except Exception:  # noqa: BLE001 — an older torch without the new API runs unscaled
            scaler = None

    compiled = net
    if cfg.compile_model and hasattr(torch, "compile") and device != "cpu":
        try:
            compiled = torch.compile(net)
        except Exception:  # noqa: BLE001 — compilation is an optimisation, never a requirement
            compiled = net

    vocab_size = int(getattr(spec, "vocab_size", 256)) if spec is not None else 256
    moe_aux_weight = 0.01
    net.train()
    start_time = time.monotonic()

    for step in range(state.step, cfg.max_steps):
        optimizer.zero_grad(set_to_none=True)
        factor = _lr_factor(step, total=cfg.max_steps,
                            warmup=int(cfg.max_steps * cfg.warmup_ratio),
                            kind=cfg.lr_scheduler, min_ratio=cfg.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = cfg.lr * factor

        total_loss = 0.0
        for _micro in range(max(1, cfg.grad_accum_steps)):
            x_np, y_np = dataset.batch(cfg.batch_size)
            x = torch.from_numpy(x_np).to(device, non_blocking=True)
            y = torch.from_numpy(y_np).to(device, non_blocking=True)

            ctx = (torch.autocast(device_type=device.split(":")[0], dtype=autocast_dtype)
                   if autocast_dtype is not None else _null_context())
            with ctx:
                logits, aux = _forward(compiled, x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size).float(), y.reshape(-1))
                if aux is not None:
                    loss = loss + moe_aux_weight * aux
                # Elastic Weight Consolidation, preserved exactly: the penalty over previously
                # consolidated weights keeps a new generation from overwriting the last.
                synapses = getattr(model, "synapses", None)
                if synapses is not None:
                    loss = loss + synapses.penalty()
                loss = loss / max(1, cfg.grad_accum_steps)

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total_loss += float(loss.detach()) * max(1, cfg.grad_accum_steps)

        if cfg.max_grad_norm and cfg.max_grad_norm > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        state.step = step + 1
        state.tokens_seen += cfg.tokens_per_step * dataset.block_size

        if cfg.log_every and state.step % cfg.log_every == 0:
            record = {"step": state.step, "loss": round(total_loss, 5),
                      "lr": round(cfg.lr * factor, 8), "tokens": state.tokens_seen,
                      "elapsed_s": round(time.monotonic() - start_time, 1)}
            state.history.append(record)
            if on_log:
                on_log(record)

        if cfg.eval_every and state.step % cfg.eval_every == 0:
            val = evaluate_per_domain(model, val_dataset or dataset, cfg,
                                      vocab_size=vocab_size)
            record = {"step": state.step, "val": val}
            state.history.append(record)
            mean = val.get("mean", float("inf"))
            if mean < state.best_val:
                state.best_val = mean
            if on_log:
                on_log(record)
            net.train()

        if out is not None and cfg.checkpoint_every and state.step % cfg.checkpoint_every == 0:
            try:
                state.data_cursor = dataset.state_dict()
            except Exception:  # noqa: BLE001
                state.data_cursor = {}
            _save_checkpoint(net, optimizer, out, state)

    if out is not None:
        try:
            state.data_cursor = dataset.state_dict()
        except Exception:  # noqa: BLE001
            state.data_cursor = {}
        _save_checkpoint(net, optimizer, out, state)
    return state


class _null_context:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _forward(net: Any, x: Any) -> Tuple[Any, Any]:
    """Forward pass returning ``(logits, moe_aux)`` for either decoder."""
    try:
        out = net(x, return_aux=True)
    except TypeError:
        return net(x), None
    if isinstance(out, tuple) and len(out) == 3:
        logits, _caches, aux = out
        return logits, aux
    return out, None


def evaluate_per_domain(model: Any, dataset: Any, cfg: TrainConfig, *,
                        vocab_size: int) -> Dict[str, float]:
    """Mean validation loss per domain, plus the overall mean.

    Per domain, not just overall: an averaged number hides a model that traded arithmetic for
    prose, which is precisely the failure mode "good at everything" has to rule out.
    """
    import torch
    from torch import nn

    net = getattr(model, "net", model)
    device = getattr(model, "device", "cpu")
    net.eval()
    out: Dict[str, float] = {}
    domains = list(getattr(dataset, "domains", []) or [None])
    with torch.no_grad():
        for domain in domains:
            total, n = 0.0, 0
            for _ in range(max(1, cfg.eval_batches)):
                try:
                    x_np, y_np = dataset.batch(cfg.batch_size, domain)
                except Exception:  # noqa: BLE001 — a domain with no data scores nothing
                    break
                x = torch.from_numpy(x_np).to(device)
                y = torch.from_numpy(y_np).to(device)
                logits, _aux = _forward(net, x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size).float(), y.reshape(-1))
                total += float(loss)
                n += 1
            if n:
                out[domain or "all"] = round(total / n, 5)
    if out:
        out["mean"] = round(sum(v for k, v in out.items() if k != "mean") / len(out), 5)
    return out


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def _save_checkpoint(net: Any, optimizer: Any, out: Path, state: TrainingState) -> None:
    """Write model, optimizer and state atomically.

    Atomic because a checkpoint half-written when the process is killed is worse than no
    checkpoint: the resume path would load it and produce a model that is quietly wrong.
    """
    import torch

    out.mkdir(parents=True, exist_ok=True)
    for name, payload in (("model.pt", net.state_dict()),
                          ("optimizer.pt", optimizer.state_dict())):
        tmp = out / f"{name}.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, out / name)
    tmp = out / "train_state.json.tmp"
    tmp.write_text(json.dumps(state.to_dict()), encoding="utf-8")
    os.replace(tmp, out / "train_state.json")


def _load_checkpoint(net: Any, out: Path, default: TrainingState,
                     device: str = "cpu") -> Optional[TrainingState]:
    """Restore weights and state, or ``None`` when there is nothing to resume."""
    import torch

    weights, meta = out / "model.pt", out / "train_state.json"
    if not weights.exists() or not meta.exists():
        return None
    try:
        net.load_state_dict(torch.load(weights, map_location=device))
        return TrainingState.from_dict(json.loads(meta.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001 — an unreadable checkpoint starts fresh rather than crashing
        return None


def _load_optimizer(optimizer: Any, out: Path, device: str = "cpu") -> bool:
    """Restore the Adam moments. Without them a resumed run re-enters warmup blind."""
    import torch

    path = out / "optimizer.pt"
    if not path.exists():
        return False
    try:
        optimizer.load_state_dict(torch.load(path, map_location=device))
        return True
    except Exception:  # noqa: BLE001
        return False
