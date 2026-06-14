#!/bin/bash
# NYXARA — SessionStart hook for Claude Code on the web.
# Installs the lean run profile (core + reasoning + server + dev) so tests,
# linters and the console/server are ready the moment the session starts.
# Mirrors .github/workflows/ci.yml — heavy ML extras (torch/transformers/whisper/
# qdrant/embeddings) are intentionally omitted: those tests skip gracefully, keeping
# setup fast. Install them by hand with `pip install -e ".[senses,foundry,qwen]"`.
set -euo pipefail

# Only run in the remote (web) environment; local installs are the user's own.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}"

# Best-effort pip upgrade — never fatal (system pip may be distro-managed and
# refuse to self-uninstall; the editable install below works regardless).
python -m pip install --upgrade pip || true

# Editable install with the lean extras + httpx (backs FastAPI's TestClient).
python -m pip install -e ".[dev,reasoning,server]" "httpx>=0.27"
