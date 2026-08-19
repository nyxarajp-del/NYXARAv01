#!/usr/bin/env python3
"""Recursive distillation: Qwythos teaches, NJP verifies, and the teacher's grip is measured.

    python scripts/distill_recursive.py --mode scaffold --turns 200
    python scripts/distill_recursive.py --mode colearn  --turns 200
    python scripts/distill_recursive.py --report

The loop, per prompt:

    teacher answers  ->  njp/shadow.py compares it against her own cognition
                     ->  semantic gaps  -> njp/concepts.py as OBSERVATIONS
                     ->  causal claims  -> njp/adversary.py's four attacks
                     ->  prediction gap -> recorded, and trained on only through
                                           ShadowCognition.residual(established=True)

Nothing the teacher says is stored as a fact on the strength of the teacher having said it. That
is not a stylistic preference: a teacher's answer feels like the independent outcome the closed
loop in ``njp/integrate.py`` requires, and it is a correlated one. Allowed to grade her, it would
train her toward agreement — errors included — and make ``eval/teacher_removal.py`` pass for the
wrong reason.

**This script never promotes the mode by itself.** ``scaffold -> colearn -> autonomous`` is earned
from a measurement, and the measurement lives in ``python -m nyxara.eval.teacher_removal``. Setting
the mode by hand is setting an aspiration; the flag here only says which one you are running under.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nyxara.growth.distill import DEFAULT_NYXARA_SYSTEM, distill_prompts_by_topic  # noqa: E402
from nyxara.njp.brain import NJPBrain  # noqa: E402


def run(mode: str, turns: int, topics: str, verify: bool) -> int:
    brain = NJPBrain()
    cortex = getattr(brain, "cortex", None)
    shadow = getattr(brain, "shadow_cognition", None)

    if shadow is None:
        print("shadow cognition is not built on this core — nothing to distil into.")
        return 1
    if cortex is None or not cortex.available():
        reason = cortex.why_unavailable() if cortex else "no cortex organ"
        print(f"the teacher is not available: {reason}")
        print("fetch the weights with: python scripts/fetch_qwythos_model.py")
        return 1

    by_topic = distill_prompts_by_topic()
    wanted = ([t.strip() for t in topics.split(",")] if topics and topics != "all"
              else sorted(by_topic))
    prompts = [p for topic in wanted for p in by_topic.get(topic, [])][:turns]
    if not prompts:
        print(f"no prompts for topics {wanted!r}; have {sorted(by_topic)}")
        return 1

    print(f"mode={mode}  prompts={len(prompts)}  teacher={cortex.stats()['model']}")
    for i, prompt in enumerate(prompts, 1):
        reply = cortex.ask(prompt, system=DEFAULT_NYXARA_SYSTEM)
        if not reply.ok:
            continue
        thought = None
        try:
            thought = brain.think(prompt)
        except Exception:  # noqa: BLE001 — a failed turn is a skipped comparison
            pass
        # Tell the comparator who phrased her side. With the Qwythos weights present, `gguf`
        # leads the ladder and renders her voice too — the semantic gap would then compare the
        # model to itself and come back falsely clean.
        rendered_by = ""
        try:
            voice = getattr(brain, "voice", None)
            rendered_by = str(voice.surface().provider) if voice is not None else ""
        except Exception:  # noqa: BLE001
            rendered_by = ""
        reading = shadow.compare(prompt, njp_thought=thought, teacher_text=reply.text,
                                 njp_rendered_by=rendered_by)
        if i % 10 == 0 or verify:
            d = reading.to_dict()
            print(f"  [{i:4d}] gaps={d['n_gaps']:2d} "
                  f"(sem {d['semantic']} causal {d['causal']} pred {d['prediction']})  "
                  f"challenged={d['claims_challenged']} refuted={d['claims_refuted']} "
                  f"undecided={d['claims_undecided']}")

    print()
    print(json.dumps(shadow.stats(), indent=2))
    print()
    print("Nothing above is a capability claim. Run the measurement that is:")
    print("  python -m nyxara.eval.teacher_removal")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", default="scaffold",
                    choices=("scaffold", "colearn", "autonomous"))
    ap.add_argument("--turns", type=int, default=200)
    ap.add_argument("--topics", default="all")
    ap.add_argument("--verify", action="store_true", help="print every comparison")
    ap.add_argument("--report", action="store_true", help="print shadow stats and exit")
    args = ap.parse_args(argv)

    if args.report:
        brain = NJPBrain()
        print(json.dumps({"cortex": getattr(brain, "cortex", None) and brain.cortex.stats(),
                          "shadow": getattr(brain, "shadow_cognition", None) and brain.shadow_cognition.stats(),
                          "fabric": brain.fabric.stats().get("grafted")}, indent=2, default=str))
        return 0
    return run(args.mode, args.turns, args.topics, args.verify)


if __name__ == "__main__":
    raise SystemExit(main())
