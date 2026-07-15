#!/usr/bin/env bash
# =============================================================================
# NYXARA — LoRA-tune Qwen2.5-0.5B into her OWN primary brain
# -----------------------------------------------------------------------------
# One command to forge NYXARA's primary brain: a LoRA adapter on
# Qwen/Qwen2.5-0.5B-Instruct, trained on her identity seed corpus
# (+ any distilled teacher / lived memory), gauntlet-gated, and promoted so
# the `self` provider serves it.
#
# The base weights download on first use (HuggingFace cache). With .[foundry]
# (torch+transformers+peft) installed this trains the REAL Qwen2.5-0.5B LoRA;
# without it the forge degrades to the always-on n-gram brain (no download,
# still works). At 0.5B it fine-tunes in full precision on a CPU or a modest
# GPU — no QLoRA needed. Every training knob is configurable via NYXARA_FOUNDRY__*
# (lora_r, lora_target_modules, batch_size, grad_accum_steps, lr_scheduler, …).
#
# Usage:
#   bash scripts/lora_tune_qwen.sh                 # forge once (1 generation)
#   bash scripts/lora_tune_qwen.sh --distill --bench   # + distillation + report
#   GEN=3 bash scripts/lora_tune_qwen.sh           # forge 3 generations
#
# After this, run NYXARA with her own brain primary:
#   NYXARA_LLM__PROVIDER=self python -m nyxara
# (Or just set NYXARA_LLM__PROVIDER=self in .env — she auto-forges on first boot.)
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

GEN="${GEN:-1}"

echo "==> Forging NYXARA's primary brain: LoRA on Qwen2.5-0.5B (${GEN} generation(s))"
echo "    (base downloads on first use; install .[foundry] for the real Qwen2.5-0.5B)"

python -m nyxara.growth --qwen --generations "${GEN}" "$@"

echo
echo "Done. Run her with the forged brain primary:"
echo "    NYXARA_LLM__PROVIDER=self python -m nyxara"
