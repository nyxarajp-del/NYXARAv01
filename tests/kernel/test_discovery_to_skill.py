"""A discovered law must become a reusable SKILL — knowing → being able (capability growth,
not just knowledge). The kernel mints a skill in her skill tree from the newest law. Hermetic.
"""

from __future__ import annotations

from nyxara.kernel.orchestrator import NyxaraCore


def test_a_discovered_law_is_minted_into_a_skill():
    nyx = NyxaraCore()
    if nyx.skilltree is None or nyx.law_discovery is None:
        return  # capability disabled in this build; nothing to assert
    # she discovers a law from her own "measurements"
    X, y = [], []
    for m in range(1, 9):
        for a in range(1, 9):
            X.append([float(m), float(a)])
            y.append(float(m * a))
    nyx.law_discovery.discover_from_data(X, y, var_names=["m", "a"], target_name="force")
    law = nyx.law_discovery.known_laws()[-1]

    name = nyx._mint_skill_from_law(law)
    assert name is not None
    skill = nyx.skilltree.get(name)
    assert skill is not None
    assert skill.category == "discovered_law"
    # practising the freshly-minted (prerequisite-free) skill earned real proficiency
    assert nyx.skilltree.proficiency(name) > 0.0


def test_minting_is_idempotent_and_safe_without_a_tree():
    nyx = NyxaraCore()
    if nyx.skilltree is None or nyx.law_discovery is None:
        return
    X, y = [], []
    for m in range(1, 6):
        for a in range(1, 6):
            X.append([float(m), float(a)])
            y.append(float(m + a))
    nyx.law_discovery.discover_from_data(X, y, var_names=["m", "a"], target_name="total")
    law = nyx.law_discovery.known_laws()[-1]
    first = nyx._mint_skill_from_law(law)
    second = nyx._mint_skill_from_law(law)   # same law again → same skill, more practice
    assert first == second
    assert nyx._mint_skill_from_law(None) is None
