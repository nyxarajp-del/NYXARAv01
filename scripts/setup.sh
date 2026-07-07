#!/usr/bin/env bash
# =============================================================================
# NYXARA setup — fully local TinyLlama-1.1B (no API keys)
# -----------------------------------------------------------------------------
# One-shot installer for the local LLM stack:
#   * TinyLlama-1.1B — TinyLlama/TinyLlama-1.1B-Chat-v1.0 running in-process
#     (torch + transformers + accelerate; weights auto-download on first use)
#   * Foundry (LoRA) — peft, so NYXARA can LoRA-tune TinyLlama as her OWN brain
#
# Usage:
#   bash scripts/setup.sh          # install everything (TinyLlama + foundry)
#   bash scripts/setup.sh --light  # core only (no torch download; mock/n-gram)
#
# Requires Python 3.11+.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

WANT_TINYLLAMA=1
case "${1:-}" in
  --light) WANT_TINYLLAMA=0 ;;
  "")      ;;
  *) echo "unknown option: $1 (use --light or no arg)"; exit 1 ;;
esac

echo "==> Python: $(python3 --version)"

echo "==> Installing core + dev extras"
python3 -m pip install -e ".[dev]"

if [[ "$WANT_TINYLLAMA" == "1" ]]; then
  echo "==> TinyLlama local: installing torch + transformers + accelerate (heavy, ~GBs)"
  python3 -m pip install -e ".[llm]"
  echo "    TinyLlama-1.1B weights download automatically on first use (cached after)."
  echo "==> Foundry (LoRA): installing peft so NYXARA can LoRA-tune TinyLlama as her OWN brain"
  python3 -m pip install -e ".[foundry]"
  echo "    Set NYXARA_LLM__PROVIDER=self — she forges her primary brain on first boot,"
  echo "    or run it now:  bash scripts/lora_tune_tinyllama.sh"
fi

# Seed a .env from the template if the user hasn't made one yet.
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "==> Created .env from .env.example — tweak the NYXARA_LLM__TINYLLAMA_* knobs as you like"
else
  echo "==> .env already exists — left untouched"
fi

echo
echo "Done. Next steps:"
echo "  1. (Optional) edit .env — every TinyLlama/LoRA knob lives there"
echo "  2. Load it:   set -a; source .env; set +a"
echo "  3. Run:       python -m nyxara"
echo
echo "TinyLlama + her own forged brain run together as a council (NYXARA asks both, then decides)."
