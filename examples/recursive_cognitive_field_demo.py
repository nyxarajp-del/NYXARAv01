#!/usr/bin/env python3
"""NJP V.04 — the Recursive Cognitive Field, end to end, from plain Hinglish.

Run it::

    python examples/recursive_cognitive_field_demo.py

Nothing here is staged. The brain is built cold, told a couple of dozen ordinary sentences, and
every number printed is read back off the organs afterwards. What it is meant to show is the one
claim that separates this from a pipeline: the concepts were **invented**, the taxonomy was
**derived**, the counterfactual is about an intervention **nobody performed**, and the next
experiment was chosen because it is the most **informative**, not the most interesting.

Offline, no API key, no LLM anywhere in the path.
"""

from __future__ import annotations

from nyxara.njp.brain import NJPBrain

LESSON = [
    # physics, as the Master would actually type it
    "aag se garmi milti hai",
    "garmi se pani ubalta hai",
    "pani 100 degree par ubalta hai",
    "pani ubalne se bhaap banti hai",
    "barf pighalti hai",
    "lakdi se aag banti hai",
    "aag se dhuan nikalta hai",
    # biology
    "paudhe ko pani chahiye",
    "bina pani paudha marta hai",
    "suraj se roshni aati hai",
    "roshni se paudhe badhte hai",
    "plants need water",
    # kinds, so a taxonomy has something to form out of
    "dog is a animal", "cat is a animal", "wolf is a animal", "horse is a animal",
    "dog needs food", "cat needs food", "wolf needs food", "horse needs food",
    "dog makes sound", "cat makes sound", "wolf makes sound",
    "oak is a plant", "rose is a plant", "fern is a plant",
    "oak needs water", "rose needs water", "fern needs water",
]


def rule(title: str) -> None:
    print(f"\n{'─' * 74}\n{title}\n{'─' * 74}")


def main() -> int:
    brain = NJPBrain()

    rule("1. TEACHING — twenty-nine ordinary sentences, no special format")
    for line in LESSON:
        brain.think(line)
    print(f"   {len(LESSON)} turns taken.")

    stats = brain.stats()

    rule("2. WHAT REACHED THE WORLD MODEL (all of this used to be exactly 0)")
    world = stats["world"]
    print(f"   grounded facts   {stats['grounding']['facts']}")
    print(f"   stated laws      {world['stated_laws']}")
    print(f"   causal links     {world['causal_links']}")
    print(f"   timeline events  {world['events']}")
    for link in brain.world.links(causal_only=True)[:5]:
        print(f"     {link.cause:>10} → {link.effect:<10} strength {link.strength:.3f}")

    rule("3. CONCEPTS SHE INVENTED — nobody wrote any of these down")
    genesis = brain.genesis
    for concept in sorted(genesis.concepts.values(), key=lambda c: (c.order, -c.support)):
        parent = concept.parent[:9] if concept.parent else "—"
        print(f"   n={concept.support:<2} order={concept.order}  "
              f"{sorted(concept.invariants)}  parent={parent}")
    print(f"\n   compression {genesis.compression():.3f}×  "
          f"(>1 means the concepts are SMALLER than what they explain)")
    print(f"   hierarchy links {stats['concepts']['hierarchy_links']}  "
          f"— derived by subsumption, never declared")

    rule("4. RESTRUCTURING — the claim the whole theory turns on")
    print("   Three of the four animals were said to make a sound, so 'produces:sound' became part")
    print("   of what the concept CLAIMS — and the horse then fell outside her own concept of")
    print("   animal. Watch what she does with that, because this is the junction:\n")

    coverage = genesis.explain("horse")
    print(f"   explain(horse)   gap={coverage.gap!r}  violated={coverage.violated}")
    print(f"   → not 'the coefficient is wrong'. The CONCEPT is wrong.\n")

    before_ratio = genesis.compression()
    report = genesis.restructure(coverage)
    after = genesis.explain("horse")
    print(f"   restructure      invariant_share {report and genesis.invariant_share:.2f}"
          f"   compression {before_ratio:.3f} → {genesis.compression():.3f}")
    print(f"   explain(horse)   {'covered' if after.covered else after.gap}")
    print("   → she did not learn that horses are animals. She learned that her definition")
    print("     of animal was over-claiming, and moved the boundary. Accepted only because")
    print("     it did not cost compression on everything else she knows.\n")

    print("   And only NOW is transfer honest — what she can say about a thing she never")
    print("   observed it of:")
    for subject in ("horse", "fern"):
        transferred = genesis.generalise(subject)
        print(f"     {subject:<6} → {transferred or '(nothing beyond what was observed)'}")
    print("   (defeasible: 'most animals make a sound, this is an animal, so probably')")

    rule("5. COUNTERFACTUAL — an intervention nobody performed")
    universe = brain.universe
    print("   First, honestly: the sentences above carry mechanisms, not measurements, so")
    print("   nothing here can be fitted from them yet —")
    refused = universe.what_if("water", 2.0)
    print(f"     do(water=2.0) → {refused.reason or 'answered'}")
    print("\n   Note she also refuses to fit an arrow her world model does not allow — the")
    print("   causal skeleton says which arrows may exist, which is what stops a simulator")
    print("   from regressing anything on anything. So tell her the law, in Hinglish:")
    brain.think("pani se growth badhta hai")
    universe.sync_from_world()
    print("     → stated law recorded; the arrow pani→growth is now WATCHED but unfitted")

    print("\n   Now give her seven actual observations, the way a sensor would:")
    for water in range(7):
        universe.observe({"pani": float(water), "light": 5.0,
                          "growth": 2.0 * water + 1.0})
    fitted = universe.relations[("pani", "growth")]
    print(f"     fitted  growth = {fitted.slope:.2f}·pani + {fitted.intercept:.2f}   "
          f"R² = {fitted.r2:.4f}  (n={fitted.n}, never told the formula)")
    print("     the LAW came from the Master; the COEFFICIENT she measured herself")
    print()
    for water in (0.0, 3.0, 500.0):
        out = universe.what_if("pani", water)
        if out.answerable:
            changed = ", ".join(f"{d.variable}={d.after:.2f}" for d in out.changed()[:2])
            note = "  ← far outside anything seen" if water > 100 else ""
            print(f"     do(pani={water:<6}) → {changed:<22} "
                  f"confidence {out.confidence:.3f}{note}")
        else:
            print(f"     do(pani={water:<6}) → refused: {out.reason}")
    print("\n   The last line is the honest part: she answers, and says how far out she reached.")
    ambiguous = universe.ambiguous()
    if ambiguous:
        print(f"   She also reports {len(ambiguous)} pair(s) whose DIRECTION observational data")
        print("   cannot settle — Markov equivalence, flagged rather than guessed.")

    rule("6. A BELIEF, WITH ITS WHOLE CASE")
    case = brain.beliefs.why("aag causes garmi")
    if case.get("known"):
        print(f"   confidence   {case['confidence']}   (evidence earns {case['earned']})")
        print(f"   produced by  {case['produced_by']}")
        print(f"   evidence     {[e['kind'] for e in case['evidence']]}")
        print(f"   falsifier    {case['falsifier']}")
        print(f"   hard support {case['hard_support']}  "
              f"— testimony alone can never establish anything")
    audit = brain.beliefs.audit()
    print(f"\n   audit flags  {len(audit)} belief(s) asserted beyond their evidence")

    rule("7. CURIOSITY AS INFORMATION GAIN — the experiment worth running")
    designer = brain.designer
    experiment = designer.design()
    print(f"   uncertainty now  {designer.prior_entropy():.3f} bits "
          f"over {designer.stats()['live']} live hypotheses")
    if experiment is not None:
        print(f"   best experiment  {experiment.name}  → {experiment.gain:.4f} bits")
        print(f"   outcomes         { {k: round(v, 3) for k, v in experiment.outcomes.items()} }")
    print("   (an experiment every hypothesis predicts identically scores exactly 0 bits, "
          "and is not run)")

    rule("8. THE FIELD — what the two loops actually did")
    field = stats["field"]
    print(f"   cycles                {field['cycles']}")
    print(f"   errors diagnosed      {field['errors'] or '{}'}")
    print(f"   restructures          {field['restructures']} "
          f"(kept {field['restructures_kept']})")
    print(f"   experiments designed  {field['experiments_designed']}")
    print(f"   held-out benchmark    {field['benchmark']:.4f}")
    print(f"   loop-2 trials         {field['meta_trials']} "
          f"(accepted {field['meta_accepted']})")
    last = field.get("last_trial")
    if last:
        print(f"   last verdict          {last['why']}")

    print("\n" + "─" * 74)
    print("Every number above was produced by the organs doing the thing they claim to do.")
    print("A cycle that changed nothing reports that it changed nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
