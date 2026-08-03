"""NYXARA · sim — worlds she can run, and therefore worlds she can be checked against (🌍).

Thirteen simulators live here, and until now the package exported none of them: ``__init__.py``
was zero bytes, so every consumer had to reach into a submodule by hand and nothing advertised
what was available. That is part of why the causal stack and these worlds — which belong together
— had no import relationship at all.

They matter for one reason above the rest: each is **deterministic and has a known closed-form
answer**. An RC circuit's time constant really is ``R·C``; a wave on a string really does travel at
``√(T/μ)``. So they are not only environments to act in, they are *ground truth* — the only place
in the system where a learned causal graph can be checked against a law that is true by
construction rather than against something else she inferred. Grading her own homework is the
failure mode every self-improving system has to design around, and these worlds are the answer to
it (see :mod:`~nyxara.growth.causal_validation`).

The heavier environments (``physics_world``, ``embodied``, ``real_environment``, ``montecarlo``,
``envmodel``, ``sandbox``) load lazily through ``__getattr__``: they are large, and a consumer that
only wants the RC circuit should not pay for the physics engine.
"""

from __future__ import annotations

from typing import Any, List

# The small, closed-form law worlds — cheap to import, and the ones the causal validator uses.
from nyxara.sim.chem_world import GAS_CONSTANT, ReactionRun, ReactionWorld, arrhenius_k
from nyxara.sim.circuit_world import RCCircuit
from nyxara.sim.em_world import Charge, ElectrostaticWorld
from nyxara.sim.epi_world import EpidemicWorld, Outbreak
from nyxara.sim.optics_world import WaveString
from nyxara.sim.thermo_world import GasBox

__all__ = [
    # closed-form law worlds
    "RCCircuit",
    "GasBox",
    "WaveString",
    "Charge",
    "ElectrostaticWorld",
    "ReactionWorld",
    "ReactionRun",
    "arrhenius_k",
    "GAS_CONSTANT",
    "EpidemicWorld",
    "Outbreak",
    # lazily-loaded heavier environments
    "PhysicsWorld",
    "Body",
    "PhysicsAgent",
    "EmbodiedAgent",
    "embodied_stream",
    "RealEnvironment",
    "sensorimotor_tick",
    "MonteCarlo",
    "MonteCarloPlanner",
    "EnvironmentModel",
    "EnvironmentRegistry",
    "Sandbox",
]

# name -> module that defines it; resolved on first access so the heavy engines stay unimported
# until something actually asks for one.
_LAZY = {
    "PhysicsWorld": "physics_world", "Body": "physics_world", "PhysicsAgent": "physics_world",
    "EmbodiedAgent": "embodied", "embodied_stream": "embodied",
    "RealEnvironment": "real_environment", "sensorimotor_tick": "real_environment",
    "MonteCarlo": "montecarlo", "MonteCarloPlanner": "montecarlo",
    "EnvironmentModel": "envmodel", "EnvironmentRegistry": "envmodel",
    "Sandbox": "sandbox",
}


def __getattr__(name: str) -> Any:
    """Import a heavy environment on first use (PEP 562)."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(f"nyxara.sim.{module}"), name)


def __dir__() -> List[str]:
    return sorted(__all__)
