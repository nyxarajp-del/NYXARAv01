#!/bin/bash
# NYXARA — SessionStart hook for Claude Code on the web.
#
# Runs in async mode so the session starts immediately while dependencies install
# in the background. Two phases:
#   1. LEAN profile (.[dev,reasoning,server] + httpx) — mirrors .github/workflows/ci.yml.
#      This is all the test suite, the Master console and the FastAPI server need.
#   2. HEAVY ML extras (senses/foundry/qlora/tinyllama/vector/qdrant/embeddings/generate/llm) —
#      best-effort. These pull torch/transformers/whisper (~GBs) and light up the
#      optional faculties; every one degrades gracefully if its install is skipped,
#      so a failure here never blocks a working session.
#
# Because it's async there's a brief window after session start where heavy extras
# are still installing; the lean phase (which the tests rely on) is ordered first.
set -uo pipefail

echo '{"async": true, "asyncTimeout": 900000}'

# Only run in the remote (web) environment; local installs are the user's own.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}"

# Best-effort pip upgrade — never fatal (system pip may be distro-managed and
# refuse to self-uninstall; the installs below work regardless).
python -m pip install --upgrade pip || true

# Phase 1 — lean profile. Must succeed: tests/console/server depend on it.
#
# `corpus` rides here, NOT with the heavy extras. It is datasets + huggingface_hub + pyarrow —
# megabytes — and putting it behind torch meant a timeout in phase 2 left `scripts/build_dataset.py`
# unable to reach a single source. A container reset proved that: the checkout came back with
# neither package and the corpus path was silently offline-only.
python -m pip install -e ".[dev,reasoning,server,corpus]" "httpx>=0.27"

# Phase 2 — heavy ML extras. Best-effort: graceful fallbacks cover any gap.
python -m pip install -e ".[senses,foundry,qlora,tinyllama,vector,qdrant,embeddings,generate,llm,security,observe]" || \
  echo "NYXARA: heavy ML extras skipped/partial — optional faculties degrade gracefully." >&2
