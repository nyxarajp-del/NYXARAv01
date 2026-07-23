#!/usr/bin/env bash
# =============================================================================
# NYXARA — QLoRA-tune TinyLlama-1.1B into her OWN primary brain
# -----------------------------------------------------------------------------
# One command to forge NYXARA's primary brain: a LoRA adapter on
# TinyLlama/TinyLlama-1.1B-Chat-v1.0, trained on her identity seed corpus
# (+ any distilled teacher / lived memory), gauntlet-gated, and promoted so
# the `self` provider serves it.
#
# The base weights download on first use (HuggingFace cache). With .[foundry]
# (torch+transformers+peft) installed this trains the REAL TinyLlama-1.1B LoRA;
# without it the forge degrades to the always-on n-gram brain (no download,
# still works). Training is QLoRA by default: on a CUDA GPU with bitsandbytes
# the frozen base loads 4-bit (NF4 + double quant) and only the LoRA adapter
# trains; on a CPU it silently trains full-precision LoRA instead. Every
# training knob is configurable via NYXARA_FOUNDRY__* (lora_r,
# lora_target_modules, batch_size, grad_accum_steps, lr_scheduler, …).
#
# Usage:
#   bash scripts/lora_tune_tinyllama.sh                 # forge once (1 generation)
#   bash scripts/lora_tune_tinyllama.sh --distill --bench   # + distillation + report
#   GEN=3 bash scripts/lora_tune_tinyllama.sh           # forge 3 generations
#
# After this, run NYXARA with her own brain primary:
#   NYXARA_LLM__PROVIDER=self python -m nyxara
# (Or just set NYXARA_LLM__PROVIDER=self in .env — she auto-forges on first boot.)
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

GEN="${GEN:-1}"

echo "==> Forging NYXARA's primary brain: QLoRA on TinyLlama-1.1B (${GEN} generation(s))"
echo "    (base downloads on first use; install .[foundry] for the real TinyLlama-1.1B QLoRA)"

python -m nyxara.growth --tinyllama --generations "${GEN}" "$@"

echo
echo "Done. Run her with the forged brain primary:"
echo "    NYXARA_LLM__PROVIDER=self python -m nyxara"
