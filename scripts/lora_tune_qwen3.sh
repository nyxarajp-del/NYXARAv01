#!/usr/bin/env bash
# =============================================================================
# NYXARA — LoRA-tune Qwen3-4B into her OWN primary brain
# -----------------------------------------------------------------------------
# One command to forge NYXARA's primary brain: a LoRA adapter on Qwen/Qwen3-4B,
# trained on her identity seed corpus (+ any distilled teacher / lived memory),
# gauntlet-gated, and promoted so the `self` provider serves it.
#
# The base weights download on first use (HuggingFace cache). With .[foundry]
# (torch+transformers+peft) installed this trains the REAL Qwen3-4B LoRA; without
# it the forge degrades to the always-on n-gram brain (no download, still works).
#
# Usage:
#   bash scripts/lora_tune_qwen3.sh                 # forge once (1 generation)
#   bash scripts/lora_tune_qwen3.sh --distill --bench   # + teacher distillation + report
#   GEN=3 bash scripts/lora_tune_qwen3.sh           # forge 3 generations
#
# After this, run NYXARA with her own brain primary:
#   NYXARA_LLM__PROVIDER=self python -m nyxara
# (Or just set NYXARA_LLM__PROVIDER=self in .env — she auto-forges on first boot.)
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

GEN="${GEN:-1}"

echo "==> Forging NYXARA's primary brain: LoRA on Qwen/Qwen3-4B (${GEN} generation(s))"
echo "    (base downloads on first use; install .[foundry] for the real Qwen3-4B)"

python -m nyxara.growth --qwen3 --generations "${GEN}" "$@"

echo
echo "Done. Run her with the forged brain primary:"
echo "    NYXARA_LLM__PROVIDER=self python -m nyxara"
