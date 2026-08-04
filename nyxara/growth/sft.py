"""NYXARA · growth/sft.py — supervised fine-tuning, with the loss on her answer only (🎯).

Pretraining teaches next-token prediction over everything. That produces a model that
*continues* text, which is not the same as a model that *answers* — ask a purely pretrained
model a question and it will happily write the next question. Turning a continuation engine into
something that holds a conversation is one specific change: train on supervised pairs, and put
the loss **only on the answer span**.

Why only the answer matters more than it sounds. If the loss covers the prompt too, the model
spends most of its gradient learning to predict the Master's phrasing — which it will never be
asked to produce — and comparatively little on the reply, which is the entire job. Masking the
prompt typically doubles the useful signal per token at no cost.

The span boundary is exact rather than heuristic, and that is a direct payoff from
:mod:`~nyxara.growth.tokenizer`: ``### NYXARA:`` is an **atomic special token**, so the answer
begins at a known token id. There is no substring search over decoded text, nothing to go wrong
when the marker's bytes happen to appear inside a code block, and no off-by-one where the model
is asked to predict its own role tag.

Everything trains in her existing template (:func:`~nyxara.mind.llm.format_self_training_doc`),
which is the same shape ``growth/dataset.py`` and ``growth/synth_data.py`` already emit and the
same shape inference renders — so train/inference parity holds end to end.

The ``-100`` ignore-index convention mirrors ``LoRAModel._batch``
(:mod:`~nyxara.growth.foundry_models`), so both supervised paths mask identically.

``torch`` is required and imported lazily. Depends on ``growth/trainer`` for the optimizer and
schedule, and ``growth/tokenizer`` for the span boundary. Nothing imports back.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "SFTConfig",
    "SFTReport",
    "SFTExample",
    "build_examples",
    "mask_to_answer",
    "supervised_finetune",
    "IGNORE_INDEX",
]

# torch's cross_entropy skips this label. The same convention LoRAModel._batch uses.
IGNORE_INDEX = -100


@dataclass
class SFTConfig:
    """Optimiser settings for the supervised stage.

    A far lower learning rate than pretraining, and only a few epochs: SFT is meant to teach the
    *shape* of a reply, not to re-learn the language. Pushed harder it overwrites what
    pretraining bought — the standard way to produce a model that is fluent, obedient, and
    suddenly much worse at maths.
    """

    lr: float = 2e-5
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    min_lr_ratio: float = 0.1
    batch_size: int = 4
    grad_accum_steps: int = 1
    epochs: int = 2
    max_steps: int = 0            # 0 -> derive from epochs
    log_every: int = 10
    seed: int = 0
    # Drop an example whose answer span is shorter than this after masking. A pair that tokenizes
    # to two answer tokens contributes almost nothing and dilutes the batch.
    min_answer_tokens: int = 4


@dataclass
class SFTExample:
    """One supervised example: token ids, and labels with the prompt masked out."""

    input_ids: List[int]
    labels: List[int]

    @property
    def answer_tokens(self) -> int:
        return sum(1 for label in self.labels if label != IGNORE_INDEX)


@dataclass
class SFTReport:
    """What was trained on, and what was refused."""

    examples: int = 0
    skipped_no_marker: int = 0
    skipped_short_answer: int = 0
    skipped_too_long: int = 0
    answer_tokens: int = 0
    prompt_tokens: int = 0
    steps: int = 0
    final_loss: float = 0.0
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        if message and message not in self.notes:
            self.notes.append(message)

    @property
    def answer_fraction(self) -> float:
        total = self.answer_tokens + self.prompt_tokens
        return self.answer_tokens / total if total else 0.0

    def summary(self) -> str:
        head = (f"{self.examples} example(s), {self.steps} step(s), "
                f"loss {self.final_loss:.4f}")
        detail = (f"{self.answer_tokens:,} answer tokens supervised "
                  f"({self.answer_fraction:.0%} of the corpus; the rest is masked prompt)")
        skips = []
        for label, n in (("no answer marker", self.skipped_no_marker),
                         ("answer too short", self.skipped_short_answer),
                         ("over the context", self.skipped_too_long)):
            if n:
                skips.append(f"{n} {label}")
        tail = f"\n  skipped: {', '.join(skips)}" if skips else ""
        notes = ("\n  · " + "\n  · ".join(self.notes)) if self.notes else ""
        return f"· {head}\n  {detail}{tail}{notes}"

    def to_dict(self) -> Dict[str, Any]:
        return {"examples": self.examples, "skipped_no_marker": self.skipped_no_marker,
                "skipped_short_answer": self.skipped_short_answer,
                "skipped_too_long": self.skipped_too_long,
                "answer_tokens": self.answer_tokens, "prompt_tokens": self.prompt_tokens,
                "answer_fraction": round(self.answer_fraction, 4), "steps": self.steps,
                "final_loss": round(self.final_loss, 5), "seconds": round(self.seconds, 3),
                "notes": list(self.notes)}


# --------------------------------------------------------------------------- #
# The mask
# --------------------------------------------------------------------------- #
def mask_to_answer(ids: Sequence[int], *, assistant_id: int,
                   ignore_index: int = IGNORE_INDEX) -> Optional[List[int]]:
    """Labels for ``ids`` with everything up to and including the answer marker masked.

    Returns ``None`` when the marker is absent — an example whose answer boundary cannot be
    located is skipped rather than trained on with a guessed boundary, because a wrong boundary
    trains the model to emit its own role tag mid-reply.

    The **last** marker is used, so a multi-turn transcript supervises only the final reply. The
    earlier turns remain in the context as history, which is exactly what they are.
    """
    last = -1
    for i, token in enumerate(ids):
        if token == assistant_id:
            last = i
    if last < 0:
        return None
    labels = list(ids)
    for i in range(last + 1):
        labels[i] = ignore_index
    return labels


def build_examples(docs: Iterable[str], *, tokenizer: Any, block_size: int,
                   config: Optional[SFTConfig] = None,
                   report: Optional[SFTReport] = None) -> List[SFTExample]:
    """Tokenize supervised documents and mask each one's prompt.

    ``docs`` are strings in her own template — what
    :func:`~nyxara.mind.llm.format_self_training_doc` produces, and what ``growth/dataset.py``
    and ``growth/synth_data.py`` already emit.
    """
    cfg = config or SFTConfig()
    report = report if report is not None else SFTReport()
    assistant_id = tokenizer.assistant_id
    out: List[SFTExample] = []

    for doc in docs:
        if not doc or not doc.strip():
            continue
        ids = tokenizer.encode(doc, add_eos=True)
        if len(ids) > block_size:
            # Truncating would cut the answer, which is the only part being supervised.
            report.skipped_too_long += 1
            continue
        labels = mask_to_answer(ids, assistant_id=assistant_id)
        if labels is None:
            report.skipped_no_marker += 1
            continue
        example = SFTExample(input_ids=ids, labels=labels)
        if example.answer_tokens < cfg.min_answer_tokens:
            report.skipped_short_answer += 1
            continue
        out.append(example)
        report.examples += 1
        report.answer_tokens += example.answer_tokens
        report.prompt_tokens += len(ids) - example.answer_tokens

    if not out:
        report.note("no usable supervised examples — is the corpus in her own template?")
    return out


def _collate(batch: Sequence[SFTExample], *, pad_id: int, device: Any) -> Tuple[Any, Any]:
    """Pad a batch to its longest example. Padding is masked, never supervised."""
    import torch

    width = max(len(e.input_ids) for e in batch)
    input_ids, labels = [], []
    for example in batch:
        pad = width - len(example.input_ids)
        input_ids.append(example.input_ids + [pad_id] * pad)
        labels.append(example.labels + [IGNORE_INDEX] * pad)
    return (torch.tensor(input_ids, dtype=torch.long, device=device),
            torch.tensor(labels, dtype=torch.long, device=device))


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #
def supervised_finetune(model: Any, docs: Sequence[str], *,
                        config: Optional[SFTConfig] = None,
                        out_dir: Any = None,
                        on_log: Optional[Callable[[Dict[str, Any]], None]] = None
                        ) -> SFTReport:
    """Run the supervised stage on ``model`` in place, and return what it did.

    Requires a model with a trained tokenizer attached — the answer span is located by token id,
    and a byte-level model has no marker token to find.
    """
    import torch
    from torch import nn

    from nyxara.growth.trainer import _lr_factor, _param_groups

    cfg = config or SFTConfig()
    report = SFTReport()
    start = time.monotonic()

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        report.note("no tokenizer attached — the supervised stage needs the answer-marker "
                    "token id to mask the prompt; train a tokenizer first")
        return report

    block_size = int(getattr(model.spec, "block_size", 512))
    examples = build_examples(docs, tokenizer=tokenizer, block_size=block_size,
                              config=cfg, report=report)
    if not examples:
        return report

    net = model.net
    device = getattr(model, "device", "cpu")
    vocab_size = int(getattr(model.spec, "vocab_size", 256))
    pad_id = tokenizer.pad_id

    steps_per_epoch = max(1, math.ceil(len(examples) / max(1, cfg.batch_size)))
    total_steps = cfg.max_steps or steps_per_epoch * max(1, cfg.epochs)

    optimizer = torch.optim.AdamW(_param_groups(net, cfg.weight_decay), lr=cfg.lr,
                                  betas=cfg.betas)
    rng = random.Random(cfg.seed)
    order = list(range(len(examples)))
    net.train()
    cursor = 0
    last_loss = 0.0

    for step in range(total_steps):
        factor = _lr_factor(step, total=total_steps,
                            warmup=int(total_steps * cfg.warmup_ratio),
                            kind=cfg.lr_scheduler, min_ratio=cfg.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = cfg.lr * factor

        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for _micro in range(max(1, cfg.grad_accum_steps)):
            if cursor + cfg.batch_size > len(order):
                rng.shuffle(order)
                cursor = 0
            picks = [examples[i] for i in order[cursor:cursor + cfg.batch_size]]
            cursor += cfg.batch_size
            if not picks:
                continue

            x, y = _collate(picks, pad_id=pad_id, device=device)
            logits = _forward(net, x)
            # Next-token shift: position t predicts t+1, so labels move left by one.
            loss = nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, vocab_size).float(),
                y[:, 1:].reshape(-1),
                ignore_index=IGNORE_INDEX)
            synapses = getattr(model, "synapses", None)
            if synapses is not None:
                loss = loss + synapses.penalty()
            loss = loss / max(1, cfg.grad_accum_steps)
            loss.backward()
            accumulated += float(loss.detach()) * max(1, cfg.grad_accum_steps)

        if cfg.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
        optimizer.step()
        last_loss = accumulated

        if cfg.log_every and (step + 1) % cfg.log_every == 0 and on_log:
            on_log({"stage": "sft", "step": step + 1, "loss": round(accumulated, 5),
                    "lr": round(cfg.lr * factor, 8)})

    report.steps = total_steps
    report.final_loss = last_loss
    report.seconds = time.monotonic() - start

    if out_dir is not None:
        model.save(Path(out_dir))
    return report


def _forward(net: Any, x: Any) -> Any:
    """Logits from either decoder, discarding the MoE aux term (not used in the SFT loss)."""
    try:
        out = net(x, return_aux=True)
    except TypeError:
        return net(x)
    return out[0] if isinstance(out, tuple) else out
