#!/usr/bin/env bash
# =============================================================================
# NYXARA setup — Groq cloud + Qwen3 local
# -----------------------------------------------------------------------------
# One-shot installer for the two LLM backends you chose:
#   * Groq   — GPT-OSS-120B cloud (uses the openai SDK; needs GROQ_API_KEY)
#   * Qwen3  — Qwen/Qwen3-4B running locally (torch + transformers; no key)
#
# Usage:
#   bash scripts/setup.sh          # install everything (Groq + Qwen)
#   bash scripts/setup.sh --groq   # Groq only (light, no torch download)
#   bash scripts/setup.sh --qwen   # Qwen only (heavy: torch + transformers)
#
# Requires Python 3.11+.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

WANT_GROQ=1
WANT_QWEN=1
case "${1:-}" in
  --groq) WANT_QWEN=0 ;;
  --qwen) WANT_GROQ=0 ;;
  "")     ;;
  *) echo "unknown option: $1 (use --groq, --qwen, or no arg)"; exit 1 ;;
esac

echo "==> Python: $(python3 --version)"

echo "==> Installing core + reasoning + llm + dev extras"
python3 -m pip install -e ".[dev]"

if [[ "$WANT_GROQ" == "1" ]]; then
  echo "==> Groq: ready (served via the openai SDK, already installed by .[dev])"
fi

if [[ "$WANT_QWEN" == "1" ]]; then
  echo "==> Qwen3 local: installing torch + transformers + accelerate (heavy, ~GBs)"
  python3 -m pip install -e ".[qwen]"
  echo "    Qwen3-4B weights download automatically on first use (cached after)."
fi

# Seed a .env from the template if the user hasn't made one yet.
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "==> Created .env from .env.example — edit it and set NYXARA_LLM__GROQ_API_KEY"
else
  echo "==> .env already exists — left untouched"
fi

echo
echo "Done. Next steps:"
echo "  1. Get a free Groq key:  https://console.groq.com  -> paste into .env"
echo "  2. Load it:              set -a; source .env; set +a"
echo "  3. Run:                  python -m nyxara"
echo
echo "Both Groq + Qwen run together as a council (NYXARA asks both, then decides)."
