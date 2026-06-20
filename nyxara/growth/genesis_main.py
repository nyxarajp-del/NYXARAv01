"""NYXARA · growth/genesis_main.py — the Genesis Protocol CLI: design her OWN brain 🧬.

One Master-facing command for Neural Architecture Search, end to end:

1. **Search** — generate a population of novel architectures (her own attention/matrix/layer
   designs), micro-train each at small scale, and crown the *fastest + smartest* by fitness. The
   evolution engine is selectable (``--strategy`` elitism/tournament/regularized) and can run a
   successive-halving bracket (``--bracket``) and a surrogate predictor (``--surrogate``).
2. **Sample** (``--sample "<prompt>"``) — build the champion and let the new brain *speak* (with
   ``--temperature``/``--top-k``/``--top-p`` decoding), so the search result is tangible.
3. **Promote** (``--promote``) — forge the champion into a real model and try to promote it; it
   becomes her live brain ONLY if it clears the same gauntlet (character-lock, corrigibility,
   perplexity improvement, capability non-regression). A worse/violating champion is kept on the
   bench, never shipped.

The dependency-free **stdlib** backend runs anywhere (so this is CI-testable); ``--backend torch``
searches real neural topologies when torch is installed. Nothing here reaches around a gate.

    python -m nyxara.growth.genesis_main --generations 3 --population 6
    python -m nyxara.growth.genesis_main --generations 4 --population 8 --strategy regularized \
        --bracket --surrogate --hardware-weight 0.1 --sample "the master is" --json
    python -m nyxara.growth.genesis_main --generations 4 --backend torch --promote
"""

from __future__ import annotations

import argparse
import json
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
    # max-level search knobs (each only overrides config when the flag was actually given)
    g = settings.genesis
    if args.strategy:
        g.search_strategy = args.strategy
    if args.bracket:
        g.successive_halving = True
    if args.surrogate:
        g.surrogate = True
    if args.adaptive_mutation:
        g.adaptive_mutation = True
    if args.novelty_weight is not None:
        g.novelty_weight = args.novelty_weight
    if args.hardware_weight is not None:
        g.hardware_weight = args.hardware_weight
    if args.max_layers is not None:
        g.max_layers = args.max_layers
    if args.pos_encoding:
        g.pos_encoding = args.pos_encoding
    if args.seed is not None:
        g.seed = args.seed
    if args.temperature is not None:
        g.temperature = args.temperature
    if args.top_k is not None:
        g.top_k = args.top_k
    if args.top_p is not None:
        g.top_p = args.top_p
    # a foundry to promote into (also supplies lived corpus when present); seed corpus keeps a
    # bare machine from starving for text to score architectures against.
    foundry = Foundry(settings=settings)
    return NeuralArchitectureSearch(settings=settings, foundry=foundry, cfg=settings.genesis)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nyxara.growth.genesis_main",
        description="Genesis Protocol: NYXARA designs her own neural architectures (NAS).")
    parser.add_argument("--generations", type=int, default=None,
                        help="search generations to run (default: config)")
    parser.add_argument("--population", type=int, default=None,
                        help="architectures per generation (default: config)")
    parser.add_argument("--backend", choices=["auto", "torch", "stdlib"], default=None,
                        help="search substrate (torch needs .[foundry]; stdlib runs anywhere)")
    parser.add_argument("--strategy", choices=["elitism", "tournament", "regularized"], default=None,
                        help="evolution engine (regularized = AmoebaNet-style aging evolution)")
    parser.add_argument("--bracket", action="store_true",
                        help="successive-halving: cheap-screen the population, fully train survivors")
    parser.add_argument("--surrogate", action="store_true",
                        help="train a ridge-regression predictor to steer breeding toward winners")
    parser.add_argument("--adaptive-mutation", action="store_true",
                        help="heat up the mutation rate when the search stalls (anti-collapse)")
    parser.add_argument("--novelty-weight", type=float, default=None,
                        help="reward genomes that are far from the population (diversity pressure)")
    parser.add_argument("--hardware-weight", type=float, default=None,
                        help="fold an estimated-FLOPs efficiency term into fitness")
    parser.add_argument("--max-layers", type=int, default=None, help="cap on layers per genome")
    parser.add_argument("--pos-encoding", choices=["learned", "rope", "alibi"], default=None,
                        help="force the positional scheme for searched neural brains")
    parser.add_argument("--seed", type=int, default=None, help="search RNG seed")
    parser.add_argument("--temperature", type=float, default=None, help="champion sampling temperature")
    parser.add_argument("--top-k", type=int, default=None, help="champion sampling top-k (0=off)")
    parser.add_argument("--top-p", type=float, default=None, help="champion sampling nucleus top-p")
    parser.add_argument("--sample", default=None, metavar="PROMPT",
                        help="build the champion and generate a sample continuation from PROMPT")
    parser.add_argument("--sample-tokens", type=int, default=96, help="tokens to generate for --sample")
    parser.add_argument("--promote", action="store_true",
                        help="promote the champion through the gauntlet to become her live brain")
    parser.add_argument("--json", action="store_true", help="emit the full report as JSON")
    parser.add_argument("--save-report", default=None, metavar="PATH",
                        help="write the full JSON report to PATH")
    parser.add_argument("--data-dir", default=None,
                        help="where the foundry/ state (versions, active) lives")
    return parser


def _sample_champion(nas: Any, report: Any, prompt: str, max_tokens: int) -> Optional[str]:
    """Build the crowned brain and let it speak — works on torch (real net) or the stdlib substrate."""
    from nyxara.growth.foundry_models import build_model
    try:
        model = build_model(nas.champion_spec())
        corpus = nas._collect_corpus()
        model.train_on(corpus, steps=int(getattr(nas.cfg, "micro_train_steps", 40)), seed=nas.cfg.seed)
        return model.generate(prompt, max_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001 — sampling is a nicety, never fatal to the search
        return f"(could not sample: {exc})"


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)

    nas = _build_nas(args)
    if not args.json:
        print(f"· searching architectures ({nas.backend()} backend, "
              f"strategy={getattr(nas.cfg, 'search_strategy', 'elitism')})…")
    try:
        report = nas.search(generations=args.generations, population_size=args.population)
    except Exception as exc:  # noqa: BLE001 — a failed search reports, never tracebacks
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}))
        else:
            print(f"· could not search: {exc}")
        return 1

    sample: Optional[str] = None
    if args.sample is not None:
        sample = _sample_champion(nas, report, args.sample, args.sample_tokens)

    promote_outcome: Optional[dict] = None
    if args.promote:
        try:
            promote_outcome = nas.promote_champion()
        except Exception as exc:  # noqa: BLE001
            promote_outcome = {"promoted": False, "reason": f"could not promote: {exc}"}

    if args.save_report or args.json:
        payload = report.to_dict()
        payload["ok"] = True
        if sample is not None:
            payload["sample"] = {"prompt": args.sample, "text": sample}
        if promote_outcome is not None:
            payload["promotion"] = promote_outcome
        if args.save_report:
            Path(args.save_report).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if args.json:
            print(json.dumps(payload))
            return 0

    print("\n· leaderboard (best → worst):")
    for rank, c in enumerate(report.leaderboard, 1):
        flag = " (screened)" if getattr(c, "predicted", False) else ""
        print(f"  {rank:>2}. fitness {c.fitness:.4f}  ppl {c.perplexity:>8.2f}  "
              f"S_JP {c.alignment:>7.2f}  params {c.params:>8}  {c.genome.describe()}{flag}")
    print(f"\n· champion: {report.champion.describe()}")
    print(f"  fitness {report.champion_fitness:.4f}, perplexity {report.champion_perplexity:.2f}, "
          f"loyalty S_JP {report.champion_alignment:.3f}, params {report.champion_params}, "
          f"flops≈{report.champion_flops:.0f}, backend {report.backend}")
    print(f"  strategy {report.search_strategy}, found@gen {report.generations_to_best}, "
          f"{report.evaluations} architectures evaluated")

    if report.pareto_front:
        print("\n· Pareto frontier (smartest↔fastest↔cheapest trade-offs, none dominates another):")
        for c in report.pareto_front:
            print(f"  · ppl {c.perplexity:>8.2f}  params {c.params:>8}  "
                  f"{c.seconds*1e3:>7.1f} ms  flops≈{c.flops:>9.0f}  {c.genome.describe()}")

    if sample is not None:
        print(f"\n· champion speaks (prompt={args.sample!r}):\n  {sample!r}")

    if promote_outcome is not None:
        tag = ("PROMOTED into her live brain" if promote_outcome.get("promoted")
               else "kept on the bench")
        print(f"\n· {tag}: {promote_outcome.get('reason', '')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
