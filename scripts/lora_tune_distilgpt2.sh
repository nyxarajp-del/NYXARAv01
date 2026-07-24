#!/usr/bin/env bash
# =============================================================================
# NYXARA — LoRA-tune DistilGPT-2 into her OWN primary brain
# -----------------------------------------------------------------------------
# One command to forge NYXARA's primary brain: a LoRA adapter on
# distilgpt2, trained on her identity seed corpus
# (+ any distilled teacher / lived memory), gauntlet-gated, and promoted so
# the `self` provider serves it.
#
# The base weights download on first use (HuggingFace cache). With .[foundry]
# (torch+transformers+peft) installed this trains the REAL DistilGPT-2 LoRA;
# without it the forge degrades to the always-on n-gram brain (no download,
# still works). At ~82M params DistilGPT-2 trains full-precision LoRA on a bare
# CPU — no GPU, no quantization needed. Every training knob is configurable via
# NYXARA_FOUNDRY__* (lora_r, lora_target_modules, batch_size, grad_accum_steps,
# lr_scheduler, …).
#
# Usage:
#   bash scripts/lora_tune_distilgpt2.sh                 # forge once (1 generation)
#   bash scripts/lora_tune_distilgpt2.sh --distill --bench   # + distillation + report
#   GEN=3 bash scripts/lora_tune_distilgpt2.sh           # forge 3 generations
#
# After this, run NYXARA with her own brain primary:
#   NYXARA_LLM__PROVIDER=self python -m nyxara
# (Or just set NYXARA_LLM__PROVIDER=self in .env — she auto-forges on first boot.)
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

GEN="${GEN:-1}"

echo "==> Forging NYXARA's primary brain: LoRA on DistilGPT-2 (${GEN} generation(s))"
echo "    (base downloads on first use; install .[foundry] for the real DistilGPT-2 LoRA)"

python -m nyxara.growth --distilgpt2 --generations "${GEN}" "$@"

echo
echo "Done. Run her with the forged brain primary:"
echo "    NYXARA_LLM__PROVIDER=self python -m nyxara"
