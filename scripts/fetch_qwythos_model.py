#!/usr/bin/env python3
"""Fetch NYXARA's CORTEX — Qwythos-9B (Qwen3.5-based) in GGUF format.

Two files: the ~5.63 GB Q4_K_M language weights, and — when ``llm.qwythos_vision_enabled`` is on —
the 918 MB vision projector. Once they are on disk, ``mind/llm.py``'s ``auto`` ladder starts at
``qwythos`` instead of the n-gram floor, and ``njp/cortex.py`` can ask her cortex for hypotheses
rather than for prettier sentences.

    python scripts/fetch_qwythos_model.py              # fetch what is missing
    python scripts/fetch_qwythos_model.py --force      # re-download
    python scripts/fetch_qwythos_model.py --no-verify  # skip the SHA-256 check

Each file is checked against the repo's published ``SHA256SUMS``. A checksum that is obtained and
MISMATCHES is fatal — nothing is installed, because a corrupt model that loads would answer, and
every answer would be garbage attributed to her cortex. A checksum that cannot be obtained is
reported as ``unverified`` rather than silently trusted.

The boot path (``growth/bootstrap.ensure_primary_model``) does this automatically unless
``NYXARA_LLM__QWYTHOS_AUTO_DOWNLOAD=false``; this script is the manual door for an air-gapped or
metered host, and it works regardless of that flag.

Also needed once, for the runtime itself::

    pip install llama-cpp-python

Pick a different quantization by pointing ``NYXARA_LLM__QWYTHOS_FILENAME`` at any file the repo
publishes (Q5_K_M, Q6_K, Q8_0, BF16) — larger is better and needs more RAM. The non-MTP files are
the ones plain llama.cpp serves.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nyxara.mind.gguf_assets import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
