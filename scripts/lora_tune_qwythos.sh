#!/usr/bin/env bash
# =============================================================================
# NYXARA — QLoRA-tune Qwythos-9B into her OWN primary brain
# -----------------------------------------------------------------------------
# One command to forge NYXARA's primary brain: a LoRA adapter on the Qwythos-9B
# SAFETENSORS parent (llmfan46/Qwythos-9B-Claude-Mythos-5-1M-uncensored-heretic,
# a Qwen3.5-9B derivative), trained on her identity seed corpus (+ any distilled
# teacher / lived memory), gauntlet-gated, and promoted so the `self` provider
# serves it.
#
# GGUF vs safetensors: this trains the *safetensors* parent — GGUF is an
# inference-only format and cannot be LoRA-fine-tuned. The cheap GGUF quant of
# the same model is served separately by the `gguf` provider (llama-cpp-python);
# see scripts/.env or NYXARA_LLM__PROVIDER=gguf.
#
# Requirements for the REAL 9B forge (else it degrades to the always-on n-gram
# brain, no crash): .[foundry] (torch+transformers+peft) AND a CUDA GPU with
# bitsandbytes for the 4-bit QLoRA load. The Qwen3.5 hybrid arch also wants the
# optional kernels: pip install -e '.[qwythos]' (flash-linear-attention,
# causal_conv1d). Every knob is configurable via NYXARA_FOUNDRY__* (lora_r,
# lora_target_modules, batch_size, grad_accum_steps, lr_scheduler, …).
#
# Usage:
#   bash scripts/lora_tune_qwythos.sh                 # forge once (1 generation)
#   bash scripts/lora_tune_qwythos.sh --distill --bench   # + distillation + report
#   GEN=3 bash scripts/lora_tune_qwythos.sh           # forge 3 generations
#
# After this, run NYXARA with her own brain primary:
#   NYXARA_LLM__PROVIDER=self python -m nyxara
# (Or set NYXARA_LLM__PROVIDER=self in .env — she auto-forges on first boot.)
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

GEN="${GEN:-1}"

echo "==> Forging NYXARA's primary brain: QLoRA on Qwythos-9B (${GEN} generation(s))"
echo "    (safetensors parent downloads on first use; needs .[foundry] + a GPU for the"
echo "     real 9B forge — degrades to the always-on n-gram brain otherwise)"

# --qwythos preset: backend=lora, base=Qwythos-9B safetensors, 4-bit, trust_remote_code.
python -m nyxara.growth --qwythos --generations "${GEN}" "$@"

echo
echo "Done. Run her with the forged brain primary:"
echo "    NYXARA_LLM__PROVIDER=self python -m nyxara"
echo "Or serve the cheap GGUF quant of the base directly:"
echo "    NYXARA_LLM__PROVIDER=gguf python -m nyxara"
