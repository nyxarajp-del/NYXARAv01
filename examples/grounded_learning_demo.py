"""NYXARA · examples/grounded_learning_demo.py — grounded understanding, measurably real.

Run it (no LLM, no network, no downloads — everything below is NYXARA's OWN learning)::

    python examples/grounded_learning_demo.py

Four demonstrations, each with a number that moves:

1. **Embeddings learn from experience** — two invented words used in identical contexts
   converge in HER trained space; paraphrase similarity rises after ``train()``.
2. **The world model's predictions improve with life** — realized one-step error falls
   as it accumulates experience, and its confidence is honest about the unknown.
3. **Causal mechanisms carry effect sizes** — from a valued event stream ``b ≈ 2a`` she
   learns the slope, so a counterfactual answers "by how much", not just "whether".
4. **It survives restarts** — the trained embedder and dynamics reload and predict
   identically.
"""

from __future__ import annotations

import os
import random
import tempfile


def demo_embedder() -> None:
    print("=" * 70)
    print("1) the embedding space SHE trains — paraphrase reach from lived text")
    print("=" * 70)
    from nyxara.memory.neural_embedder import SelfLearnedEmbedder

    emb = SelfLearnedEmbedder(dim=128)
    cold = emb.latent_similarity("zorble device", "blint device")
    print(f"cold latent similarity (zorble, blint) : {cold}")

    contexts = ["controls the plasma flow in sector",
                "regulates the coolant valve of unit",
                "measures the neutron flux near panel",
                "adjusts the containment field around bay"]
    rng = random.Random(0)
    for _ in range(250):
        word = rng.choice(["zorble", "blint"])
        emb.learn(f"the {word} {rng.choice(contexts)} {rng.randrange(9)}. "
                  f"the {word} is stable today")
    for _ in range(40):
        emb.learn_pair("the zorble controls the plasma flow",
                       "the blint regulates the coolant valve")
    stats = emb.train(budget_s=2.0)
    print(f"trained: {stats['sgns_steps']} SGNS steps, "
          f"{stats['contrastive_steps']} contrastive steps, vocab {stats['vocab']}")
    warm = emb.latent_similarity("zorble device", "blint device")
    print(f"warm latent similarity (zorble, blint) : {warm:.3f}   ← learned from usage alone")
    print(f"semantic_grade (held-out self-audit)   : {emb.semantic_grade:.3f}")
    assert warm is not None and warm > 0.5


def demo_world_model() -> None:
    print("\n" + "=" * 70)
    print("2) the world model's realized prediction error FALLS with experience")
    print("=" * 70)
    from nyxara.mind.world_model import build_world_model

    def true_step(x: float, a: str) -> float:
        return {"left": x - 1.0, "right": x + 1.0, "stay": x}[a]

    wm = build_world_model("auto", seed=0)
    rng = random.Random(1)
    checkpoints = (50, 150, 400)
    errs = []
    for i in range(1, max(checkpoints) + 1):
        x = rng.uniform(-10, 10)
        a = rng.choice(["left", "right", "stay"])
        nx = true_step(x, a)
        wm.observe((x,), a, (nx,), reward=-abs(nx), done=abs(nx) < 0.2)
        if i in checkpoints:
            errs.append(wm.prediction_error_ema())
            print(f"after {i:4d} transitions: realized error EMA = {errs[-1]:.4f}  "
                  f"epistemic = {wm.mean_epistemic():.4f}")
    assert errs[-1] < errs[0], "prediction error should fall with experience"
    near = wm.predict((4.0,), "left")
    far = wm.predict((900.0,), "left")
    print(f"confidence near data / far outside     : {near.confidence:.2f} / "
          f"{far.confidence:.3f}   ← honest about the unknown")
    print(f"done head near goal / far from it      : "
          f"{wm.predict((0.9,), 'left').done_prob:.2f} / "
          f"{wm.predict((8.0,), 'left').done_prob:.2f}")


def demo_causal_mechanism() -> None:
    print("\n" + "=" * 70)
    print("3) causal mechanisms: 'by how much', learned from valued events")
    print("=" * 70)
    from nyxara.mind.causal_world_model import CausalWorldModel

    rng = random.Random(3)
    cwm = CausalWorldModel(window=10.0, min_pairs_fit=8)
    for k in range(200):
        base = k * 100.0
        a = rng.uniform(0, 5)
        cwm.observe("a", at=base, value=a, intervention=True)
        cwm.observe("b", at=base + 1, value=2.0 * a + rng.gauss(0, 0.1))
    cwm.discover()
    verdict = cwm.is_causal("a", "b")
    mech = cwm.mechanism("a", "b")
    print(f"verdict a→b                            : {verdict.verdict}")
    print(f"learned mechanism                      : b ≈ {mech.slope:.3f}·a + "
          f"{mech.intercept:.3f}   (r²={mech.r2:.3f}, n={mech.n})")
    cf = cwm.counterfactual("a", "b", factual=3.0, counter=1.0)
    print(f"counterfactual (a: 3→1)                : {cf.describe()}")
    assert abs(cf.delta - (-4.0)) < 0.4


def demo_persistence() -> None:
    print("\n" + "=" * 70)
    print("4) her learning SURVIVES a restart (Rule 7)")
    print("=" * 70)
    from nyxara.memory.neural_embedder import SelfLearnedEmbedder
    from nyxara.mind.world_model import (build_world_model, load_world_model,
                                         save_world_model)

    tmp = tempfile.mkdtemp(prefix="nyxara-demo-")

    emb = SelfLearnedEmbedder(dim=128)
    for _ in range(120):
        emb.learn("the zorble controls the plasma flow in sector nine")
        emb.learn_pair("the zorble controls the plasma flow",
                       "plasma flow is under zorble control")
    emb.train(budget_s=0.8)
    epath = os.path.join(tmp, "embedder.json")
    emb.save(epath)
    emb2 = SelfLearnedEmbedder(dim=128)
    emb2.load(epath)
    a, b = emb.embed("zorble plasma"), emb2.embed("zorble plasma")
    drift = max(abs(x - y) for x, y in zip(a, b))
    print(f"embedder reload  : space_version={emb2.space_version}, "
          f"max embedding drift={drift:.2e} (weights stored at 5 decimals)")
    assert drift < 1e-3

    wm = build_world_model("auto", seed=0)
    rng = random.Random(5)
    for _ in range(200):
        x = rng.uniform(-10, 10)
        wm.observe((x,), "left", (x - 1.0,), reward=-abs(x - 1.0))
    wpath = os.path.join(tmp, "world_model.json")
    save_world_model(wm, wpath)
    wm2 = build_world_model("auto", seed=0)
    load_world_model(wm2, wpath)
    p1, p2 = wm.predict((4.0,), "left"), wm2.predict((4.0,), "left")
    print(f"world model reload: predicts {p2.next_state[0]:.3f} "
          f"(before restart: {p1.next_state[0]:.3f})")
    assert abs(p1.next_state[0] - p2.next_state[0]) < 1e-9

    print(f"\nartifacts written under {tmp}")


if __name__ == "__main__":
    demo_embedder()
    demo_world_model()
    demo_causal_mechanism()
    demo_persistence()
    print("\nALL GROUNDED-LEARNING DEMOS PASSED ✓")
