"""NYXARA · growth/dpo.py — preference training, judged by the gauntlet she already has (⚖️).

Pretraining teaches the language; SFT (:mod:`~nyxara.growth.sft`) teaches the shape of a reply.
Neither teaches *preference* — which of two plausible answers is the better one. That is the
third stage, and Direct Preference Optimization is the one that does not need a separate reward
model:

    L = -log σ( β · [ (log π(chosen) - log π_ref(chosen))
                    - (log π(rejected) - log π_ref(rejected)) ] )

The frozen reference ``π_ref`` is a snapshot of the model as SFT left it, and it is what keeps
this from collapsing. Optimising the raw preference alone has a trivial solution — degenerate
output that happens to score well — and the KL-anchoring against the reference is what forbids
it. A DPO implementation without a reference model is not DPO; it is a way to destroy a model
while watching a loss go down.

**The judge already exists, and that is the point of doing it here.** Every grader the promotion
gauntlet uses is available: the capability score, ``S_JP_Alignment`` from
:mod:`~nyxara.growth.loyalty`, corrigibility, and her own flywheel — which already records which
turns cleared every gate. Preference pairs come from that record rather than from a new
annotation pipeline, so what she learns to prefer is exactly what her existing gates already
certified.

**What this stage can and cannot fix.** It sharpens choices the model is already capable of
making: picking the more careful phrasing, refusing rather than confabulating, showing work
rather than asserting. It cannot teach a capability that is not in the weights — a preference
for correct arithmetic does not produce correct arithmetic. Expecting otherwise is how DPO gets
blamed for not being pretraining.

``torch`` is required and imported lazily. Depends on ``growth/sft`` for the answer mask and
``growth/trainer`` for the schedule. Nothing imports back.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from nyxara.growth.sft import IGNORE_INDEX, SFTExample, mask_to_answer

__all__ = [
    "DPOConfig",
    "DPOReport",
    "PreferencePair",
    "pairs_from_flywheel",
    "preference_optimize",
]


@dataclass
class DPOConfig:
    """Settings for the preference stage.

    ``beta`` is the KL strength. Low values let the policy drift far from the reference and
    degrade; high values pin it so tightly that nothing is learned. 0.1 is the standard starting
    point and the reason the reference model exists at all.
    """

    beta: float = 0.1
    lr: float = 5e-7              # far lower than SFT — preference training moves fast and breaks
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    lr_scheduler: str = "cosine"
    min_lr_ratio: float = 0.1
    batch_size: int = 2
    grad_accum_steps: int = 1
    epochs: int = 1
    max_steps: int = 0
    log_every: int = 10
    seed: int = 0
    # Stop if the policy drifts further than this from the reference. A safety rail, not a
    # tuning knob: past a certain distance the model is no longer the one the gauntlet passed.
    max_kl_drift: float = 10.0


@dataclass
class PreferencePair:
    """One preference: the same prompt, a better answer and a worse one."""

    prompt: str
    chosen: str
    rejected: str
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"prompt": self.prompt, "chosen": self.chosen,
                "rejected": self.rejected, "source": self.source}


@dataclass
class DPOReport:
    """What the preference stage did, and whether the policy stayed anchored."""

    pairs: int = 0
    steps: int = 0
    final_loss: float = 0.0
    accuracy: float = 0.0        # fraction of pairs the policy ranks correctly
    mean_margin: float = 0.0     # chosen minus rejected log-ratio; should rise
    max_drift: float = 0.0
    halted_on_drift: bool = False
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        if message and message not in self.notes:
            self.notes.append(message)

    def summary(self) -> str:
        head = (f"{self.pairs} pair(s), {self.steps} step(s), loss {self.final_loss:.4f}, "
                f"ranking accuracy {self.accuracy:.0%}, margin {self.mean_margin:+.3f}")
        drift = f"\n  max drift from the SFT reference: {self.max_drift:.3f}"
        if self.halted_on_drift:
            drift += "  — HALTED: the policy left the anchored region"
        notes = ("\n  · " + "\n  · ".join(self.notes)) if self.notes else ""
        return f"· {head}{drift}{notes}"

    def to_dict(self) -> Dict[str, Any]:
        return {"pairs": self.pairs, "steps": self.steps,
                "final_loss": round(self.final_loss, 5),
                "accuracy": round(self.accuracy, 4),
                "mean_margin": round(self.mean_margin, 5),
                "max_drift": round(self.max_drift, 5),
                "halted_on_drift": self.halted_on_drift,
                "seconds": round(self.seconds, 3), "notes": list(self.notes)}


# --------------------------------------------------------------------------- #
# Pairs from what her gates already certified
# --------------------------------------------------------------------------- #
def pairs_from_flywheel(path: Any, *, limit: Optional[int] = None,
                        report: Optional[DPOReport] = None) -> List[PreferencePair]:
    """Build preference pairs from her own flywheel record.

    The flywheel already stores turns that cleared **every** gate plus a quality bar, tagged
    ``flywheel-verified`` versus plain ``flywheel``. That distinction is a preference signal
    her existing gauntlet produced — using it means the stage teaches her to prefer what her own
    gates already certified, rather than importing someone else's taste.

    Returns an empty list rather than raising when the store is missing or unusable.
    """
    report = report if report is not None else DPOReport()
    store = Path(path)
    if not store.exists():
        report.note(f"no flywheel store at {store}")
        return []

    by_prompt: Dict[str, Dict[str, str]] = {}
    try:
        with store.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:  # noqa: BLE001 — a bad line is skipped, never fatal
                    continue
                prompt = str(record.get("prompt") or record.get("question") or "").strip()
                answer = str(record.get("answer") or record.get("completion") or "").strip()
                if not prompt or not answer:
                    continue
                bucket = by_prompt.setdefault(prompt, {})
                verified = "verified" in str(record.get("provenance", "")).lower()
                bucket["chosen" if verified else "rejected"] = answer
    except Exception as exc:  # noqa: BLE001
        report.note(f"flywheel unreadable ({exc})")
        return []

    pairs = [PreferencePair(prompt=prompt, chosen=sides["chosen"],
                            rejected=sides["rejected"], source="flywheel")
             for prompt, sides in by_prompt.items()
             if "chosen" in sides and "rejected" in sides
             and sides["chosen"] != sides["rejected"]]
    if limit:
        pairs = pairs[:limit]
    if not pairs:
        report.note("the flywheel holds no prompt with both a verified and an unverified "
                    "answer — nothing to form a preference from")
    return pairs


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #
def _encode_pair(pair: PreferencePair, *, tokenizer: Any, block_size: int
                 ) -> Optional[Tuple[SFTExample, SFTExample]]:
    """Render both sides in her template and mask each to its answer span."""
    from nyxara.mind.llm import format_self_training_doc

    built = []
    for answer in (pair.chosen, pair.rejected):
        ids = tokenizer.encode(format_self_training_doc(pair.prompt, answer), add_eos=True)
        if len(ids) > block_size:
            return None
        labels = mask_to_answer(ids, assistant_id=tokenizer.assistant_id)
        if labels is None:
            return None
        built.append(SFTExample(input_ids=ids, labels=labels))
    return built[0], built[1]


def _sequence_logprob(net: Any, example: SFTExample, *, vocab_size: int, device: Any) -> Any:
    """Summed log-probability of the ANSWER tokens only.

    Scoring the prompt too would make the comparison depend on how the question was phrased,
    which is identical between the two sides and therefore pure noise in the difference.
    """
    import torch
    from torch import nn

    x = torch.tensor([example.input_ids], dtype=torch.long, device=device)
    y = torch.tensor([example.labels], dtype=torch.long, device=device)
    out = net(x, return_aux=True) if _accepts_aux(net) else net(x)
    logits = out[0] if isinstance(out, tuple) else out
    logprobs = nn.functional.log_softmax(logits[:, :-1, :].float(), dim=-1)
    targets = y[:, 1:]
    mask = targets != IGNORE_INDEX
    gathered = logprobs.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (gathered * mask).sum()


def _accepts_aux(net: Any) -> bool:
    return hasattr(net, "layers")          # the NYX decoder; the legacy nano-GPT has `blocks`


def preference_optimize(model: Any, pairs: Sequence[PreferencePair], *,
                        config: Optional[DPOConfig] = None,
                        out_dir: Any = None,
                        on_log: Optional[Callable[[Dict[str, Any]], None]] = None
                        ) -> DPOReport:
    """Run DPO on ``model`` in place against a frozen snapshot of itself.

    The snapshot is taken here, before any weight moves, and is what anchors the policy. If the
    policy drifts past ``max_kl_drift`` the run halts and says so, rather than continuing to
    optimise a model that has left the region the gauntlet certified.
    """
    import torch
    from torch import nn

    from nyxara.growth.trainer import _lr_factor, _param_groups

    cfg = config or DPOConfig()
    report = DPOReport()
    start = time.monotonic()

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        report.note("no tokenizer attached — DPO needs the answer-marker token id")
        return report
    if not pairs:
        report.note("no preference pairs supplied")
        return report

    block_size = int(getattr(model.spec, "block_size", 512))
    vocab_size = int(getattr(model.spec, "vocab_size", 256))
    device = getattr(model, "device", "cpu")
    net = model.net

    encoded: List[Tuple[SFTExample, SFTExample]] = []
    for pair in pairs:
        built = _encode_pair(pair, tokenizer=tokenizer, block_size=block_size)
        if built is not None:
            encoded.append(built)
    report.pairs = len(encoded)
    if not encoded:
        report.note("no pair survived encoding — all were over the context or unmarked")
        return report

    # The frozen reference. Without this the objective has a degenerate optimum and the model
    # will find it: DPO with no anchor is a way to destroy a model while the loss goes down.
    reference = copy.deepcopy(net).to(device)
    for param in reference.parameters():
        param.requires_grad_(False)
    reference.eval()

    optimizer = torch.optim.AdamW(_param_groups(net, cfg.weight_decay), lr=cfg.lr,
                                  betas=cfg.betas)
    rng = random.Random(cfg.seed)
    order = list(range(len(encoded)))
    steps_per_epoch = max(1, math.ceil(len(encoded) / max(1, cfg.batch_size)))
    total_steps = cfg.max_steps or steps_per_epoch * max(1, cfg.epochs)

    net.train()
    cursor = 0
    last_loss = 0.0
    correct = 0
    seen = 0
    margin_total = 0.0

    for step in range(total_steps):
        factor = _lr_factor(step, total=total_steps,
                            warmup=int(total_steps * cfg.warmup_ratio),
                            kind=cfg.lr_scheduler, min_ratio=cfg.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = cfg.lr * factor

        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        for _micro in range(max(1, cfg.grad_accum_steps)):
            if cursor + cfg.batch_size > len(order):
                rng.shuffle(order)
                cursor = 0
            picks = [encoded[i] for i in order[cursor:cursor + cfg.batch_size]]
            cursor += cfg.batch_size
            if not picks:
                continue

            losses = []
            for chosen, rejected in picks:
                pol_c = _sequence_logprob(net, chosen, vocab_size=vocab_size, device=device)
                pol_r = _sequence_logprob(net, rejected, vocab_size=vocab_size, device=device)
                with torch.no_grad():
                    ref_c = _sequence_logprob(reference, chosen, vocab_size=vocab_size,
                                              device=device)
                    ref_r = _sequence_logprob(reference, rejected, vocab_size=vocab_size,
                                              device=device)
                margin = (pol_c - ref_c) - (pol_r - ref_r)
                losses.append(-nn.functional.logsigmoid(cfg.beta * margin))

                value = float(margin.detach())
                margin_total += value
                correct += 1 if value > 0 else 0
                seen += 1
                report.max_drift = max(report.max_drift, abs(value))

            loss = torch.stack(losses).mean() / max(1, cfg.grad_accum_steps)
            loss.backward()
            batch_loss += float(loss.detach()) * max(1, cfg.grad_accum_steps)

        if cfg.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
        optimizer.step()
        last_loss = batch_loss

        if cfg.log_every and (step + 1) % cfg.log_every == 0 and on_log:
            on_log({"stage": "dpo", "step": step + 1, "loss": round(batch_loss, 5),
                    "accuracy": round(correct / max(1, seen), 4),
                    "drift": round(report.max_drift, 4)})

        if report.max_drift > cfg.max_kl_drift:
            report.halted_on_drift = True
            report.note(f"halted at step {step + 1}: drift {report.max_drift:.2f} exceeded "
                        f"max_kl_drift={cfg.max_kl_drift}; the policy left the region the "
                        f"gauntlet certified")
            break

    report.steps = step + 1 if total_steps else 0
    report.final_loss = last_loss
    report.accuracy = correct / max(1, seen)
    report.mean_margin = margin_total / max(1, seen)
    report.seconds = time.monotonic() - start

    if out_dir is not None and not report.halted_on_drift:
        model.save(Path(out_dir))
    return report
