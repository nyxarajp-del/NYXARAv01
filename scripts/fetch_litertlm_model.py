#!/usr/bin/env python3
"""Fetch NYXARA's PRIMARY on-device brain — Gemma-4-E2B-it in LiteRT-LM format.

One ~2.4 GB INT4 ``model.litertlm`` file. Once it is on disk, ``mind/llm.py``'s ``auto`` ladder
starts at ``litertlm`` instead of the cloud, and she answers with no key and no network.

    python scripts/fetch_litertlm_model.py            # fetch if missing
    python scripts/fetch_litertlm_model.py --force    # re-download

The boot path (``growth/bootstrap.ensure_primary_model``) does this automatically unless
``NYXARA_LLM__LITERTLM_AUTO_DOWNLOAD=false``; this script is the manual door for an air-gapped or
metered host, and it works regardless of that flag.

Also needed once, for the runtime itself::

    sudo apt install libvulkan1      # Debian/Ubuntu  (dnf install vulkan-loader on Fedora/RHEL)
    pip install litert-lm-api

The Vulkan loader is required **even for the CPU backend** — ``liblitert-lm.so`` links it
unconditionally. Skip it and the pip install still succeeds, the weights still download, and every
turn then fails with ``libvulkan.so.1: cannot open shared object file``. No GPU or driver is
needed, only the loader. This script reports whether the runtime can actually start, so you find
out here rather than at her first reply.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nyxara.mind.litertlm_assets import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
