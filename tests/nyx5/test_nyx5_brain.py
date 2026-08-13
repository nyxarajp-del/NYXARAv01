"""NYX-5 facade wiring — every configured pillar is actually built on the brain.

Each pillar has its own unit test next to this one, so each was known to work in isolation.
Nothing tested the *facade*, and seven of them (holo-swarm, omni-forge, autopoiesis, mesh,
ontogenesis, retarget, phagocytosis) were never constructed by ``Nyx5Brain`` at all: the module
docstring claimed them, the config declared their flags, and no code anywhere read those flags.
Flipping one changed nothing, in either direction.

These tests pin both halves of that contract — a flag that is ON builds its faculty, and a flag
that is OFF leaves it absent — so a pillar can never again be config-gated in name only.
"""

from __future__ import annotations

import pytest

from nyxara.kernel.config import Nyx5Config
from nyxara.nyx5 import Nyx5Brain

# (attribute on the brain, config flag that gates it)
PILLARS = [
    ("chrono", "chrono_enabled"),
    ("sensorium", "sensorium_enabled"),
    ("immune", "immune_enabled"),
    ("intent", "intent_enabled"),
    ("concept", "concept_collapse_enabled"),
    ("axioms", "axiom_forge_enabled"),
    ("negentropy", "negentropy_enabled"),
    ("conduit", "conduit_enabled"),
    ("mirror", "mirroring_enabled"),
    ("continuum", "continuum_enabled"),
    ("threads", "threading_enabled"),
    ("voice", "voice_enabled"),
    ("dialectic", "dialectic_enabled"),
    ("anticipator", "anticipation_enabled"),
    # the seven that were declared but never built
    ("swarm", "holo_swarm_enabled"),
    ("omni_forge", "omni_forge_enabled"),
    ("autopoiesis", "autopoiesis_enabled"),
    ("mesh", "mesh_enabled"),
    ("ontogenesis", "ontogenesis_enabled"),
    ("retarget", "retarget_enabled"),
    ("phagocyte", "phagocytosis_enabled"),
]


@pytest.mark.parametrize("attr,flag", PILLARS)
def test_enabled_pillar_is_built(attr: str, flag: str) -> None:
    """Every pillar whose flag defaults ON is a real object on the brain, not None."""
    config = Nyx5Config()
    assert getattr(config, flag) is True, f"{flag} is expected to default ON"
    assert getattr(Nyx5Brain(config), attr) is not None, (
        f"{flag} is on, but brain.{attr} was never built — the flag is decorative")


@pytest.mark.parametrize("attr,flag", PILLARS)
def test_disabled_pillar_is_absent(attr: str, flag: str) -> None:
    """And the off-switch bites: a flag turned off leaves the faculty absent."""
    brain = Nyx5Brain(Nyx5Config(**{flag: False}))
    assert getattr(brain, attr) is None, (
        f"{flag} is off, but brain.{attr} was built anyway — the flag is not read")


def test_faculties_roster_matches_what_is_built() -> None:
    """faculties() reports what is live, and covers every pillar this facade owns."""
    brain = Nyx5Brain(Nyx5Config())
    live = brain.faculties()
    assert set(live) == {attr for attr, _ in PILLARS}
    # absence is reported as absence, never as a zeroed entry
    off = Nyx5Brain(Nyx5Config(mesh_enabled=False, retarget_enabled=False)).faculties()
    assert "mesh" not in off and "retarget" not in off
    assert "swarm" in off


def test_stats_reports_the_roster() -> None:
    """stats() carries the roster, so what she is made of is inspectable at runtime."""
    stats = Nyx5Brain(Nyx5Config()).stats()
    assert "faculties" in stats
    assert "phagocyte" in stats["faculties"]


def test_perceive_still_ticks_with_every_pillar_on() -> None:
    """Building the full stack does not disturb the perceive tick."""
    tick = Nyx5Brain(Nyx5Config()).perceive("the owner asks a question")
    assert tick.n_spikes > 0
    assert 0.0 <= tick.surprise <= 1.0
