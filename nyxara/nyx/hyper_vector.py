"""NYXARA · nyx/hyper_vector.py — structure a vector can hold (🧬, NYX V.02).

:mod:`nyxara.nyx.semantics` answers *how close* two concepts are. It cannot answer **which
role is bound to which filler** — and that is a different question. "The engine turns the
flywheel" and "the flywheel turns the engine" have identical concepts, identical similarity,
and opposite meanings. A bag of words cannot tell them apart. A bound hypervector can.

The algebra is already here and is already real —
:class:`~nyxara.cognition.hyper_dimensional_vectors.HyperSpace` has ``bind``, ``unbind``,
``bundle``, ``permute``, ``unpermute`` and cosine cleanup over a 10,000-dimensional item
memory. What was missing is a **seat** for it inside NYX:

* **Role-filler structure.** :meth:`VectorSymbolic.encode` binds each ``role → filler`` and
  bundles them into one vector; :meth:`VectorSymbolic.role_of` unbinds a role back out and
  snaps it to a clean symbol. Her concepts stop being a bag.
* **Order.** ``permute`` encodes sequence, so *A then B* and *B then A* are genuinely
  different vectors — measured here, not asserted.
* **Zero-shot analogy.** ``paris : france :: tokyo : ?`` with **no training and no examples** —
  the relation is transported and cleaned up.
* **Alongside semantics.** Similarity says how near; this says what is bound to what. Both, and
  neither pretending to be the other.

Honest framing (the rule this repo keeps, see ``docs/CAPABILITIES.md``):

* **Capacity is real.** A bundle holds only so many bindings before cleanup starts returning
  the wrong symbol. That limit is not hidden: every retrieval returns a **confidence and a
  margin**, and :meth:`VectorSymbolic.capacity_probe` measures where it actually breaks on this
  machine rather than quoting a number from a paper. This is *measured retrieval*, never
  "absolute retrieval".
* An analogy is only as good as the item memory it cleans up against. Ask for one over symbols
  she has never been given and she returns **nothing**, not a nearest-but-wrong guess.
* Nothing here is trained. That is the point of the zero-shot claim, and also its ceiling: the
  structure has to be *given* to her, in a frame or a sentence she was taught.

Pure standard library (numpy accelerates the algebra when present, identically either way).
Every public method is fail-soft.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["Bound", "Retrieved", "Capacity", "VectorSymbolic"]


@dataclass
class Bound:
    """One structure held in a single vector — and what went into it."""

    name: str = ""
    roles: Dict[str, str] = field(default_factory=dict)
    sequence: List[str] = field(default_factory=list)
    slots: int = 0
    dim: int = 0

    def render(self) -> str:
        parts = [f"{k}⊗{v}" for k, v in self.roles.items()]
        if self.sequence:
            parts.append(" ▸ ".join(self.sequence))
        return f"{self.name}: " + " ⊕ ".join(parts) if parts else self.name

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "roles": dict(self.roles),
                "sequence": list(self.sequence), "slots": self.slots, "dim": self.dim}


@dataclass
class Retrieved:
    """What came back out of a vector — with how much to trust it."""

    query: str = ""
    filler: Optional[str] = None
    score: float = 0.0
    margin: float = 0.0          # how far ahead of the runner-up — the real trust signal
    runner_up: Optional[str] = None
    trusted: bool = False
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"query": self.query, "filler": self.filler, "score": round(self.score, 6),
                "margin": round(self.margin, 6), "runner_up": self.runner_up,
                "trusted": self.trusted, "note": self.note}


@dataclass
class Capacity:
    """Where cleanup actually starts failing on this machine, measured rather than quoted."""

    dim: int = 0
    clean_up_to: int = 0         # the largest bundle that still retrieved everything
    broke_at: int = 0            # the first size at which something came back wrong
    tested_to: int = 0
    margins: List[Tuple[int, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"dim": self.dim, "clean_up_to": self.clean_up_to, "broke_at": self.broke_at,
                "tested_to": self.tested_to,
                "margins": [[n, round(m, 4)] for n, m in self.margins]}


class VectorSymbolic:
    """Role-filler structure, order, and zero-shot analogy — over her own concepts."""

    def __init__(self, brain: Any = None, *, dim: int = 10000, seed: int = 42,
                 min_margin: float = 0.05) -> None:
        self.brain = brain
        self.dim = max(64, int(dim))
        self.seed = int(seed)
        # A retrieval that barely beats the runner-up is not a retrieval. The margin is the
        # honest trust signal, not the raw cosine, which is high for everything in a bundle.
        self.min_margin = max(0.0, float(min_margin))

        self._space: Any = None
        self._items: Any = None
        self.bindings: Dict[str, Bound] = {}
        self._vectors: Dict[str, Any] = {}
        self.encoded = 0
        self.retrieved = 0
        self.untrusted = 0
        self.analogies = 0

    # ---- the algebra underneath ---------------------------------------------- #
    def _engine(self) -> Tuple[Any, Any]:
        if self._space is None:
            try:
                from nyxara.cognition.hyper_dimensional_vectors import HyperSpace, ItemMemory
                self._space = HyperSpace(dim=self.dim, seed=self.seed)
                self._items = ItemMemory(self._space)
            except Exception:  # noqa: BLE001 — without it there is no algebra and she says so
                self._space, self._items = None, None
        return self._space, self._items

    def available(self) -> bool:
        return self._engine()[0] is not None

    def symbol(self, name: str) -> Any:
        """The clean atomic vector for a name, minted on first use and stable after."""
        _space, items = self._engine()
        if items is None:
            return None
        return items.add(str(name))

    # ---- structure in one vector ---------------------------------------------- #
    def encode(self, name: str, roles: Optional[Dict[str, str]] = None, *,
               sequence: Optional[Sequence[str]] = None) -> Optional[Bound]:
        """Bind every ``role → filler``, bundle them, and keep the result under ``name``.

        With ``sequence``, order is encoded by permutation — so *A then B* and *B then A* are
        different vectors rather than the same bag.
        """
        try:
            space, items = self._engine()
            if space is None:
                return None
            roles = {str(k): str(v) for k, v in (roles or {}).items() if str(k) and str(v)}
            order = [str(s) for s in (sequence or []) if str(s)]
            if not roles and not order:
                return None

            parts: List[Any] = []
            for role, filler in roles.items():
                parts.append(space.bind(items.add(role), items.add(filler)))
            for position, item in enumerate(order):
                # Permutation by position is what makes order *structural* rather than a note
                # written beside the vector.
                parts.append(space.permute(items.add(item), shift=position + 1))

            vector = space.bundle(*parts) if len(parts) > 1 else parts[0]
            bound = Bound(name=str(name), roles=roles, sequence=order,
                          slots=len(parts), dim=self.dim)
            self.bindings[bound.name] = bound
            self._vectors[bound.name] = vector
            items.add(bound.name, vector)
            self.encoded += 1
            return bound
        except Exception:  # noqa: BLE001
            return None

    def role_of(self, name: str, role: str) -> Retrieved:
        """Unbind one role back out of a bundled structure, and say how much to trust it."""
        out = Retrieved(query=f"{name}.{role}")
        try:
            space, items = self._engine()
            if space is None:
                out.note = "no hyperdimensional algebra available"
                return out
            vector = self._vectors.get(str(name))
            if vector is None:
                out.note = f"nothing is bound under {name!r}"
                return out
            probe = space.unbind(vector, items.add(str(role)))
            # Exclude the structure's own name and the role: both are in the item memory and
            # neither is an answer to "what is bound here".
            result = items.cleanup(probe, exclude=[str(name), str(role)])
            self.retrieved += 1
            out.filler = getattr(result, "name", None)
            out.score = float(getattr(result, "score", 0.0))
            out.margin = float(getattr(result, "margin", 0.0))
            out.runner_up = getattr(result, "runner_up", None)
            out.trusted = out.margin >= self.min_margin and out.score > 0.0
            if not out.trusted:
                self.untrusted += 1
                out.note = (f"cleanup is not confident: {out.score:.3f} against "
                            f"{out.runner_up or 'nothing'} at margin {out.margin:.3f} — the "
                            f"bundle is fuller than this dimension retrieves cleanly")
            else:
                out.note = f"unbound cleanly (margin {out.margin:.3f})"
            return out
        except Exception as exc:  # noqa: BLE001
            out.note = f"{type(exc).__name__}: {exc}"
            return out

    def unpack(self, name: str) -> Dict[str, Retrieved]:
        """Every role in a structure, retrieved. Untrusted ones stay in, flagged."""
        out: Dict[str, Retrieved] = {}
        try:
            bound = self.bindings.get(str(name))
            if bound is None:
                return out
            for role in bound.roles:
                out[role] = self.role_of(name, role)
            return out
        except Exception:  # noqa: BLE001
            return out

    # ---- order ------------------------------------------------------------------ #
    def order_differs(self, a: Sequence[str], b: Sequence[str]) -> Optional[float]:
        """How different are two orderings of the same items? ``None`` without the algebra.

        The point of ``permute``: "A then B" and "B then A" are the same *bag* and must not be
        the same *vector*. This returns their similarity — near 1.0 would mean order was lost.
        """
        try:
            space, items = self._engine()
            if space is None:
                return None
            def encode(seq: Sequence[str]) -> Any:
                parts = [space.permute(items.add(str(s)), shift=i + 1)
                         for i, s in enumerate(seq)]
                return space.bundle(*parts) if len(parts) > 1 else parts[0]
            return float(space.similarity(encode(a), encode(b)))
        except Exception:  # noqa: BLE001
            return None

    # ---- zero-shot analogy -------------------------------------------------------- #
    def analogy(self, a: str, b: str, c: str) -> Retrieved:
        """``a : b :: c : ?`` — no training, no examples, no gradient.

        The real inference is over **structures**: given
        ``france = {capital⊗paris, currency⊗euro}`` and
        ``japan = {capital⊗tokyo, currency⊗yen}``, the mapping ``france ⊗ japan`` transports
        ``paris`` onto ``tokyo`` *without her ever having been told that those two correspond*.
        That is Kanerva's "dollar of Mexico", and it is genuinely zero-shot.

        Over bare atomic symbols there is no relation to transport — two random vectors share
        no structure — so that case is answered with a low margin and reported as untrusted
        rather than dressed up as an inference.
        """
        out = Retrieved(query=f"{a}:{b}::{c}:?")
        try:
            space, items = self._engine()
            if space is None:
                out.note = "no hyperdimensional algebra available"
                return out
            for name in (a, b, c):
                if str(name) not in items:
                    out.note = (f"{name!r} is not a symbol I hold — an analogy can only land "
                                f"on something already in my vocabulary, and guessing the "
                                f"nearest thing would not be an answer")
                    return out

            structural = str(a) in self.bindings and str(b) in self.bindings
            relation = space.bind(items.get(str(a)), items.get(str(b)))
            probe = space.bind(relation, items.get(str(c)))
            result = items.cleanup(probe, exclude=[str(a), str(b), str(c)])
            self.analogies += 1
            out.filler = getattr(result, "name", None)
            out.score = float(getattr(result, "score", 0.0))
            out.margin = float(getattr(result, "margin", 0.0))
            out.runner_up = getattr(result, "runner_up", None)
            out.trusted = out.margin >= self.min_margin and out.score > 0.0
            if out.trusted:
                out.note = ("transported the mapping between two structures and cleaned up — "
                            "no training, and she was never told these two correspond")
            elif out.filler is None:
                out.note = ("there is nothing left in my vocabulary to land on once the three "
                            "terms are excluded — an analogy needs somewhere to go")
            elif not structural:
                out.note = (f"{a!r} and {b!r} are bare symbols, not structures. Two random "
                            f"vectors share no relation to transport, so there is nothing here "
                            f"to carry across — encode them as records first")
            else:
                out.note = (f"the nearest symbol is {out.filler!r} but only by "
                            f"{out.margin:.3f}, which is not enough to call it the answer")
            return out
        except Exception as exc:  # noqa: BLE001
            out.note = f"{type(exc).__name__}: {exc}"
            return out

    def relate(self, name: str, pairs: Sequence[Tuple[str, str]]) -> int:
        """Bundle taught pairs into one relation vector, queryable by :meth:`across`.

        This is associative recall, not inference: what comes back is a pair member she was
        *given*. It is useful and it is not the same claim as :meth:`analogy`, so it has its
        own method and its own words.
        """
        made = 0
        try:
            space, items = self._engine()
            if space is None:
                return 0
            parts: List[Any] = []
            for left, right in pairs or ():
                left, right = str(left), str(right)
                if not left or not right:
                    continue
                parts.append(space.bind(items.add(left), items.add(right)))
                made += 1
            if not parts:
                return 0
            self._vectors[f"rel:{name}"] = space.bundle(*parts) if len(parts) > 1 else parts[0]
            return made
        except Exception:  # noqa: BLE001
            return made

    def across(self, name: str, key: str) -> Retrieved:
        """Query a bundled relation: ``across("capital-of", "tokyo")`` → ``japan``."""
        out = Retrieved(query=f"{name}[{key}]")
        try:
            space, items = self._engine()
            vector = self._vectors.get(f"rel:{name}")
            if space is None or vector is None:
                out.note = f"no relation is bundled under {name!r}"
                return out
            if str(key) not in items:
                out.note = f"{key!r} is not a symbol I hold"
                return out
            result = items.cleanup(space.unbind(vector, items.get(str(key))),
                                   exclude=[str(key)])
            self.retrieved += 1
            out.filler = getattr(result, "name", None)
            out.score = float(getattr(result, "score", 0.0))
            out.margin = float(getattr(result, "margin", 0.0))
            out.runner_up = getattr(result, "runner_up", None)
            out.trusted = out.margin >= self.min_margin and out.score > 0.0
            out.note = ("recalled from the bundled relation — this is associative recall of "
                        "something she was given, not an inference"
                        if out.trusted else
                        f"the bundle is too full to retrieve {key!r} cleanly "
                        f"(margin {out.margin:.3f})")
            if not out.trusted:
                self.untrusted += 1
            return out
        except Exception as exc:  # noqa: BLE001
            out.note = f"{type(exc).__name__}: {exc}"
            return out

    # ---- capacity, measured on this machine ---------------------------------------- #
    def capacity_probe(self, *, up_to: int = 24) -> Capacity:
        """Bundle N bindings and find where retrieval actually breaks. Not a quoted number.

        This is the honest counterpart to every "10,000-dimensional" claim: the dimension is
        real, and so is the point at which a fuller bundle stops coming back cleanly.
        """
        out = Capacity(dim=self.dim, tested_to=max(1, int(up_to)))
        try:
            space, items = self._engine()
            if space is None:
                return out
            for size in range(1, out.tested_to + 1):
                roles = {f"probe_role_{i}": f"probe_filler_{i}" for i in range(size)}
                name = f"__probe_{size}"
                if self.encode(name, roles) is None:
                    break
                worst = 1.0
                clean = True
                for role, filler in roles.items():
                    got = self.role_of(name, role)
                    worst = min(worst, got.margin)
                    if got.filler != filler:
                        clean = False
                out.margins.append((size, worst))
                if clean:
                    out.clean_up_to = size
                elif not out.broke_at:
                    out.broke_at = size
                    break
            for size in range(1, out.tested_to + 1):
                self.bindings.pop(f"__probe_{size}", None)
                self._vectors.pop(f"__probe_{size}", None)
            return out
        except Exception:  # noqa: BLE001
            return out

    # ---- alongside semantics ---------------------------------------------------------- #
    def compare(self, a: str, b: str) -> Dict[str, Any]:
        """Similarity *and* structure, side by side — neither pretending to be the other."""
        try:
            out: Dict[str, Any] = {"a": a, "b": b}
            semantics = getattr(self.brain, "semantics", None)
            if semantics is not None:
                sim = semantics.similarity(a, b)
                out["similarity"] = {"score": round(sim.score, 4), "rung": sim.rung,
                                     "known": sim.known}
            out["structure"] = {"a": self.bindings[a].to_dict() if a in self.bindings else None,
                                "b": self.bindings[b].to_dict() if b in self.bindings else None}
            out["note"] = ("similarity says how near two concepts are; structure says which "
                           "role is bound to which filler. 'The engine turns the flywheel' and "
                           "'the flywheel turns the engine' are identical on the first and "
                           "opposite on the second")
            return out
        except Exception:  # noqa: BLE001
            return {}

    # ---- introspection ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        try:
            _space, items = self._engine()
            return {
                "available": self.available(),
                "dim": self.dim,
                "vocabulary": len(items) if items is not None else 0,
                "structures": len(self.bindings),
                "encoded": self.encoded, "retrieved": self.retrieved,
                "untrusted": self.untrusted, "analogies": self.analogies,
                "min_margin": self.min_margin,
                "note": ("capacity is REAL: a bundle holds only so many bindings before "
                         "cleanup returns the wrong symbol, and capacity_probe() measures "
                         "where that happens on this machine rather than quoting a number. "
                         "Every retrieval carries a margin, and a retrieval that barely beats "
                         "its runner-up is reported as untrusted. This is measured retrieval, "
                         "never 'absolute retrieval'. Nothing here is trained — which is what "
                         "makes the analogy zero-shot, and also its ceiling: the structure has "
                         "to be given to her"),
            }
        except Exception:  # noqa: BLE001
            return {}

    def describe(self) -> str:
        """What ``/nyx analogy`` and ``/nyx bind`` print."""
        try:
            if not self.available():
                return "the hyperdimensional algebra is not available in this build."
            lines = [f"{self.dim}-dimensional, {len(self.bindings)} structure(s) held."]
            for bound in list(self.bindings.values())[:12]:
                lines.append("  " + bound.render())
            if self.retrieved:
                lines.append(f"retrievals: {self.retrieved}, of which "
                             f"{self.untrusted} were too close to call.")
            return "\n".join(lines)
        except Exception:  # noqa: BLE001
            return ""

    # ---- persistence ---------------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        # The vectors themselves are regenerable from the bindings and the seed, so only the
        # structure is stored — a 10,000-D bundle per concept would bloat the sidecar for
        # nothing.
        return {"encoded": self.encoded, "retrieved": self.retrieved,
                "untrusted": self.untrusted, "analogies": self.analogies,
                "bindings": [b.to_dict() for b in self.bindings.values()]}

    def load_dict(self, data: Any) -> bool:
        try:
            if not isinstance(data, dict):
                return False
            self.encoded = int(data.get("encoded", 0))
            self.retrieved = int(data.get("retrieved", 0))
            self.untrusted = int(data.get("untrusted", 0))
            self.analogies = int(data.get("analogies", 0))
            self.bindings, self._vectors = {}, {}
            for raw in data.get("bindings", []) or []:
                if not isinstance(raw, dict) or not raw.get("name"):
                    continue
                self.encode(str(raw["name"]),
                            {str(k): str(v) for k, v in (raw.get("roles") or {}).items()},
                            sequence=[str(s) for s in (raw.get("sequence") or [])])
            return True
        except Exception:  # noqa: BLE001 — a corrupt sidecar must never block boot
            return False
