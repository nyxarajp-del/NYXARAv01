#!/usr/bin/env python3
"""Fetch the Qwythos teacher — empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF.

    python scripts/fetch_qwythos_model.py                  # the quant config names
    python scripts/fetch_qwythos_model.py --quant BF16     # full precision, 17.92 GB
    python scripts/fetch_qwythos_model.py --force          # re-download

This is a 5.63-17.92 GB fetch, so unlike her litertlm brain it is **not** downloaded on boot:
``llm.gguf_auto_download`` is off by default and this script is the door. It works regardless of
that flag.

Also needed once, for the runtime itself::

    pip install llama-cpp-python     # to serve the weights
    pip install gguf                 # to read the tensors for njp/graft.py

The download is checksummed against the repository's own ``SHA256SUMS`` and a mismatched file is
deleted rather than kept: these bytes are read as weights, and a wrong file grafts into a layer
that computes confident nonsense with a clean certificate attached.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nyxara.mind.gguf_assets import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
