"""NYXARA · nyx001/layers/l00_substrate.py — Layer 0, computational birth (🌱, the seed).

The charter's Layer 0 is "the seed. No human knowledge. Only basic computation, random state, and
learning dynamics. We start small (Toy → Small → Medium → Scaled)."

:mod:`nyxara.nyx001.substrate` supplies the computation, the randomness and the dynamics. What was
still missing is the thing that *makes those a birth*: a single seeded moment where the scale is
chosen, the generator streams are laid out, and the system acquires an identity it can be measured
against later. :class:`Substrate` is that moment.

The scale ladder is not decoration. Each rung is a real (width, depth, context, vocabulary) tuple,
and the charter's own instruction is to climb it rather than to start at the top:

======  =====  =====  =======  ======  ================================================
rung    width  depth  context  vocab   what it is for
======  =====  =====  =======  ======  ================================================
toy      32      1      16       256   unit tests; a full cycle in milliseconds
small    64      2      32      1024   a developmental stage in seconds
medium  256      4     128      8192   the default: real capacity, minutes to train
scaled  512      6     256     32768   opt-in; hours on a CPU, and it will say so
======  =====  =====  =======  ======  ================================================

``medium`` is the default because it was asked for, and because it is the smallest rung at which
the concept and compression terms have enough width to separate anything. It is also the rung at
which a test stops being instant, which is a real cost and is stated here rather than discovered
in CI.

:meth:`Substrate.birth_certificate` is the identity: seed, rung, dimensions, parameter budget, and
the wall-clock moment of birth. Two runs with the same certificate are the same system, which is
what makes an ablation or a developmental comparison mean anything.

Honest: "computational birth" is a name for seeded initialisation, not an event with any further
significance. Nothing here is alive, and the module does not pretend otherwise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from nyxara.nyx001.substrate import HAS_NUMPY, count_params, rng_for

__all__ = ["Scale", "SCALES", "BirthCertificate", "Substrate"]


@dataclass(frozen=True)
class Scale:
    """One rung of the ladder. Frozen: a system does not change size mid-life."""

    name: str
    width: int
    depth: int
    context: int
    vocab: int

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "width": self.width, "depth": self.depth,
                "context": self.context, "vocab": self.vocab}


SCALES: Dict[str, Scale] = {
    "toy": Scale("toy", 32, 1, 16, 256),
    "small": Scale("small", 64, 2, 32, 1024),
    "medium": Scale("medium", 256, 4, 128, 8192),
    "scaled": Scale("scaled", 512, 6, 256, 32768),
}


@dataclass
class BirthCertificate:
    """Who this system is. Two runs sharing one of these are the same system."""

    seed: int = 42
    rung: str = "medium"
    width: int = 256
    depth: int = 4
    context: int = 128
    vocab: int = 8192
    born_at: float = field(default_factory=time.time)
    n_params: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"seed": self.seed, "rung": self.rung, "width": self.width, "depth": self.depth,
                "context": self.context, "vocab": self.vocab,
                "born_at": round(self.born_at, 3), "n_params": self.n_params}

    def same_system_as(self, other: "BirthCertificate") -> bool:
        """Identity ignores ``born_at`` — when it was born says nothing about what it is."""
        return (self.seed == other.seed and self.rung == other.rung
                and self.width == other.width and self.depth == other.depth
                and self.context == other.context and self.vocab == other.vocab)


class Substrate:
    """Layer 0: the seeded moment a NYX V.001 system comes into existence.

    Every layer above draws its generator from :meth:`stream`, so no two layers share a random
    sequence and adding a layer does not perturb the weights of the ones already built.
    """

    def __init__(self, *, seed: int = 42, rung: str = "medium") -> None:
        self.seed = int(seed)
        self.scale: Scale = SCALES.get(str(rung).lower(), SCALES["medium"])
        self.certificate = BirthCertificate(
            seed=self.seed, rung=self.scale.name, width=self.scale.width,
            depth=self.scale.depth, context=self.scale.context, vocab=self.scale.vocab)
        self._streams: Dict[str, int] = {}
        self._next_stream = 1
        self.available = bool(HAS_NUMPY)

    # ---- generator streams ---- #
    def stream(self, name: str) -> Any:
        """A generator private to ``name``, stable across runs with the same seed.

        The stream index is assigned on first request and remembered, so the *order* in which
        layers ask does determine their indices — but only once. A layer that asks for the same
        name always gets the same sequence for a given seed and a given set of layers.
        """
        if name not in self._streams:
            self._streams[name] = self._next_stream
            self._next_stream += 1
        return rng_for(self.seed, stream=self._streams[name])

    # ---- dimensions the layers read ---- #
    @property
    def width(self) -> int:
        return self.scale.width

    @property
    def depth(self) -> int:
        return self.scale.depth

    @property
    def context(self) -> int:
        return self.scale.context

    @property
    def vocab(self) -> int:
        return self.scale.vocab

    def note_params(self, *tensors: Any) -> int:
        """Record parameters as layers build them, so the certificate carries a real budget."""
        n = count_params(*tensors)
        self.certificate.n_params += n
        return n

    def birth_certificate(self) -> BirthCertificate:
        return self.certificate

    def stats(self) -> Dict[str, Any]:
        return {"available": self.available, "scale": self.scale.to_dict(),
                "streams": len(self._streams), "certificate": self.certificate.to_dict()}

    def to_dict(self) -> Dict[str, Any]:
        return {"seed": self.seed, "rung": self.scale.name, "streams": dict(self._streams),
                "certificate": self.certificate.to_dict()}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.seed = int(d.get("seed", self.seed))
            rung = str(d.get("rung", self.scale.name))
            self.scale = SCALES.get(rung, self.scale)
            self._streams = {str(k): int(v) for k, v in (d.get("streams") or {}).items()}
            self._next_stream = max(self._streams.values(), default=0) + 1
            cert = d.get("certificate") or {}
            if cert:
                self.certificate = BirthCertificate(
                    seed=int(cert.get("seed", self.seed)), rung=str(cert.get("rung", rung)),
                    width=int(cert.get("width", self.scale.width)),
                    depth=int(cert.get("depth", self.scale.depth)),
                    context=int(cert.get("context", self.scale.context)),
                    vocab=int(cert.get("vocab", self.scale.vocab)),
                    born_at=float(cert.get("born_at", time.time())),
                    n_params=int(cert.get("n_params", 0)))
        except Exception:  # noqa: BLE001 — a corrupt sidecar must not prevent a birth
            pass


def default_substrate(config: Optional[Any] = None) -> Substrate:
    """Build the substrate a config asks for, falling back to a seeded medium system."""
    seed = int(getattr(config, "seed", 42) or 42)
    rung = str(getattr(config, "scale", "medium") or "medium")
    return Substrate(seed=seed, rung=rung)
