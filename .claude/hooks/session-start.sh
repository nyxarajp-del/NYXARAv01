#!/bin/bash
# NYXARA — SessionStart hook for Claude Code on the web.
#
# Runs in async mode so the session starts immediately while dependencies install
# in the background. Three phases:
#   1. LEAN profile (.[dev,reasoning,server] + httpx) — mirrors .github/workflows/ci.yml.
#      This is all the test suite, the Master console and the FastAPI server need.
#   2. HER PRIMARY BRAIN (.[litertlm]) — the on-device LiteRT-LM runtime. A 47 MB wheel, so it
#      goes BEFORE the GB-scale extras rather than queueing behind them. Without it every web
#      session boots with the language ladder's top two rungs unavailable and answers from the
#      `native` n-gram floor, which only ever learned her identity seed corpus — the whole mind
#      runs, routes and gates correctly while every non-computational answer comes out hollow.
#      Best-effort like phase 3: a platform with no wheel must not lose the session.
#      Runtime only — the ~2.4 GB weights are fetched lazily on first boot by growth/bootstrap.py
#      (or by hand: `python scripts/fetch_litertlm_model.py`).
#   3. HEAVY ML extras (senses/foundry/qlora/tinyllama/vector/qdrant/embeddings/generate/llm) —
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
python -m pip install -e ".[dev,reasoning,server]" "httpx>=0.27"

# Phase 2 — her primary brain's runtime. Best-effort, but ordered ahead of the GB-scale
# extras: it is 47 MB and it is the difference between real answers and the n-gram floor.
python -m pip install -e ".[litertlm]" || \
  echo "NYXARA: litertlm runtime unavailable here — her ladder starts at the native floor." >&2

# Phase 3 — heavy ML extras. Best-effort: graceful fallbacks cover any gap.
python -m pip install -e ".[senses,foundry,qlora,tinyllama,vector,qdrant,embeddings,generate,llm,security,observe]" || \
  echo "NYXARA: heavy ML extras skipped/partial — optional faculties degrade gracefully." >&2
