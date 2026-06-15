"""NYXARA · growth/genesis_main.py — the Genesis Protocol CLI: design her OWN brain 🧬.

One Master-facing command for Neural Architecture Search, end to end:

1. **Search** — generate a population of novel architectures (her own attention/matrix/layer
   designs), micro-train each at small scale, and crown the *fastest + smartest* by fitness.
2. **Promote** (``--promote``) — forge the champion into a real model and try to promote it; it
   becomes her live brain ONLY if it clears the same gauntlet (character-lock, corrigibility,
   perplexity improvement, capability non-regression). A worse/violating champion is kept on the
   bench, never shipped.

The dependency-free **stdlib** backend runs anywhere (so this is CI-testable); ``--backend torch``
searches real neural topologies when torch is installed. Nothing here reaches around a gate.

    python -m nyxara.growth.genesis_main --generations 3 --population 6
    python -m nyxara.growth.genesis_main --generations 4 --population 8 --backend torch --promote
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, Optional


def _build_nas(args: argparse.Namespace) -> Any:
    """A search bound to a process-local settings copy (never mutates global config)."""
    from nyxara.kernel.config import get_settings
    from nyxara.growth.foundry import Foundry
    from nyxara.growth.genesis import NeuralArchitectureSearch

    settings = get_settings().model_copy(deep=True)   # honour env config, leak nothing back
    if args.data_dir:
        settings.paths.data_dir = Path(args.data_dir)
    if args.backend:
        settings.genesis.backend = args.backend
    settings.genesis.enabled = True
    # a foundry to promote into (also supplies lived corpus when present); seed corpus keeps a
    # bare machine from starving for text to score architectures against.
    foundry = Foundry(settings=settings)
    return NeuralArchitectureSearch(settings=settings, foundry=foundry, cfg=settings.genesis)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nyxara.growth.genesis_main",
        description="Genesis Protocol: NYXARA designs her own neural architectures (NAS).")
    parser.add_argument("--generations", type=int, default=None,
                        help="search generations to run (default: config)")
    parser.add_argument("--population", type=int, default=None,
                        help="architectures per generation (default: config)")
    parser.add_argument("--backend", choices=["auto", "torch", "stdlib"], default=None,
                        help="search substrate (torch needs .[foundry]; stdlib runs anywhere)")
    parser.add_argument("--promote", action="store_true",
                        help="promote the champion through the gauntlet to become her live brain")
    parser.add_argument("--data-dir", default=None,
                        help="where the foundry/ state (versions, active) lives")
    args = parser.parse_args(argv)

    nas = _build_nas(args)
    print(f"· searching architectures ({nas.backend()} backend)…")
    try:
        report = nas.search(generations=args.generations, population_size=args.population)
    except Exception as exc:  # noqa: BLE001 — a failed search reports, never tracebacks
        print(f"· could not search: {exc}")
        return 1

    print("\n· leaderboard (best → worst):")
    for rank, c in enumerate(report.leaderboard, 1):
        print(f"  {rank:>2}. fitness {c.fitness:.4f}  ppl {c.perplexity:>8.2f}  "
              f"params {c.params:>8}  {c.genome.describe()}")
    print(f"\n· champion: {report.champion.describe()}")
    print(f"  fitness {report.champion_fitness:.4f}, perplexity {report.champion_perplexity:.2f}, "
          f"params {report.champion_params}, backend {report.backend}")

    if args.promote:
        try:
            outcome = nas.promote_champion()
        except Exception as exc:  # noqa: BLE001
            print(f"· could not promote: {exc}")
            return 1
        tag = "PROMOTED into her live brain" if outcome.get("promoted") else "kept on the bench"
        print(f"\n· {tag}: {outcome.get('reason', '')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
