"""Tests for self-growing, persistent relational transfer (mind/transfer.py).

Hermetic: stdlib + numpy only, no LLM. These assert that a domain schema round-trips through
serialization (including nested higher-order relations), that a novel domain recovered from raw
text is distilled and learned, and that the growth survives a "restart" — a fresh engine on the
same memory rehydrates the learned domain. This is generalization NYXARA grows by her own action.
"""

from __future__ import annotations

from nyxara.memory.store import MemoryStore
from nyxara.mind.analogy import entity as e
from nyxara.mind.analogy import relation as r
from nyxara.mind.transfer import DomainSchema, RelationalTransferEngine, seed_library

_NOVEL = ("in this device the emitter zaps the mote and then the mote whirls "
          "the emitter continuously")


def test_schema_dict_round_trips_including_higher_order():
    schema = seed_library()[0]  # solar_system, carries a nested causes(attracts, revolves_around)
    back = DomainSchema.from_dict(schema.to_dict())
    assert back.name == schema.name
    assert {str(x) for x in back.relations} == {str(x) for x in schema.relations}
    assert back.keywords == schema.keywords
    # the higher-order relation survived the trip
    assert any(x.name == "causes" for x in back.relations)


def test_manual_learn_persists_and_hydrates():
    mem = MemoryStore()
    eng = RelationalTransferEngine(learn_from_experience=True, memory=mem)
    spring, mass = e("spring"), e("mass")
    eng.learn_domain(
        "harmonic_oscillator",
        [r("pulls", spring, mass), r("oscillates", mass, spring),
         r("causes", r("pulls", spring, mass), r("oscillates", mass, spring))],
        keywords=["pendulum", "vibration"])
    # a fresh engine on the SAME memory rehydrates the learned domain
    eng2 = RelationalTransferEngine(learn_from_experience=True, memory=mem)
    assert eng2.store.get("harmonic_oscillator") is not None
    got = eng2.store.get("harmonic_oscillator")
    assert any(x.name == "causes" for x in got.relations)


def test_novel_domain_is_distilled_from_lived_structure():
    mem = MemoryStore()
    eng = RelationalTransferEngine(learn_from_experience=True, memory=mem)
    before = len(eng.store)
    tr = eng.generalize(_NOVEL)
    assert tr is not None, "a novel-vocabulary domain must still transfer by structure"
    assert len(eng.store) > before, "a successfully-transferred novel domain must be learned"
    learned = [n for n in eng.store.names() if n.startswith("learned_")]
    assert learned, "the distilled domain should carry a 'learned_' label"


def test_growth_survives_restart():
    mem = MemoryStore()
    eng1 = RelationalTransferEngine(learn_from_experience=True, memory=mem)
    eng1.generalize(_NOVEL)
    learned1 = sorted(n for n in eng1.store.names() if n.startswith("learned_"))
    assert learned1
    # a brand-new engine (a "restart") on the same memory sees the learned domain
    eng2 = RelationalTransferEngine(learn_from_experience=True, memory=mem)
    learned2 = sorted(n for n in eng2.store.names() if n.startswith("learned_"))
    assert learned2 == learned1


def test_learn_from_experience_off_does_not_grow():
    mem = MemoryStore()
    eng = RelationalTransferEngine(learn_from_experience=False, memory=mem)
    before = len(eng.store)
    eng.generalize(_NOVEL)
    assert len(eng.store) == before, "growth must be gated by learn_from_experience"


def test_distilled_schema_is_deterministic_label():
    # the same query twice must merge into one schema, not fragment into two
    eng = RelationalTransferEngine(learn_from_experience=True)
    eng.generalize(_NOVEL)
    n1 = sorted(n for n in eng.store.names() if n.startswith("learned_"))
    eng.generalize(_NOVEL)
    n2 = sorted(n for n in eng.store.names() if n.startswith("learned_"))
    assert n1 == n2 and len(n1) == 1
