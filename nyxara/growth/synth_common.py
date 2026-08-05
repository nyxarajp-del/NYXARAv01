"""NYXARA · growth/synth_common.py — the shared floor under the generators (🧱).

The measured failure this module exists to prevent: ``generate_domain_docs("code", 3000)``
returned 3000 documents containing **six distinct strings**. Six templates, each fully
deterministic, sampled three thousand times. The corpus dedup then collapsed them to six, the
tests passed, and the build reported success.

Nothing in the old design could have caught that, because **no generator declared how much it
could actually produce**. A template bank of six looks identical, from the outside, to a
generator with a state space of 10^8 — right up until you count the distinct outputs.

So every generator here declares :attr:`Generator.space`: an honest upper bound on how many
distinct documents it can emit. :func:`draw` never asks a generator for more than ``space //
10``, and names any generator that hit its cap in the report. A lazily-written generator now
fails *loudly*, at generation time, instead of silently contributing one string a million times.

The 10% rule is a birthday-bound argument, not superstition: drawing k items uniformly from a
space of size N yields roughly ``k²/2N`` collisions, so k = N/10 keeps expected duplicates near
5% of the draw — high enough to use the space efficiently, low enough that the corpus is not
mostly repeats.

``synth_data`` imports this; this must import **nothing** from ``synth_data``, or the façade and
its siblings form a cycle.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "Generator",
    "SynthReport",
    "doc",
    "draw",
    "TOOL_CALL_OPEN",
    "TOOL_CALL_CLOSE",
    "TOOL_RESULT_OPEN",
    "TOOL_RESULT_CLOSE",
]

#: The shape a tool trace takes at inference. Module constants so sft.py, the tool router and
#: the eval battery agree on one syntax rather than three that nearly match.
TOOL_CALL_OPEN = "<tool>"
TOOL_CALL_CLOSE = "</tool>"
TOOL_RESULT_OPEN = "<result>"
TOOL_RESULT_CLOSE = "</result>"

#: Never draw more than this fraction of a generator's declared state space.
_SPACE_FRACTION = 10


@dataclass(frozen=True)
class Generator:
    """One document producer, with an honest account of how much it can produce.

    ``fn`` returns ``(prompt, answer)`` or ``None`` when it declines this draw (an optional
    dependency is missing, or the random parameters landed somewhere degenerate). ``space`` is
    the number of *distinct* documents the generator could emit if you exhausted it — count the
    parameter combinations, do not guess, and err low. It is enforced, not decorative.
    """

    name: str
    domain: str
    fn: Callable[[random.Random], Optional[Tuple[str, str]]]
    space: int
    weight: float = 1.0
    #: The prover ``kind`` for this generator's claims, when they are checkable. Empty means the
    #: output is correct by construction and there is nothing for a solver to add.
    proof_kind: str = ""

    @property
    def cap(self) -> int:
        """The most documents :func:`draw` will request from this generator."""
        return max(1, self.space // _SPACE_FRACTION)


@dataclass
class SynthReport:
    """What was generated, what survived, and which generators ran out of room."""

    generated: int = 0
    verified: int = 0
    rejected: int = 0
    duplicates: int = 0
    certified: int = 0
    by_kind: Dict[str, int] = field(default_factory=dict)
    capped: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        if message and message not in self.notes:
            self.notes.append(message)

    def credit(self, name: str) -> None:
        self.by_kind[name] = self.by_kind.get(name, 0) + 1

    @property
    def acceptance(self) -> float:
        return self.verified / self.generated if self.generated else 0.0

    def summary(self) -> str:
        kinds = ", ".join(f"{k} {v}" for k, v in sorted(self.by_kind.items()))
        head = f"· {self.verified}/{self.generated} verified ({self.acceptance:.0%})"
        if self.certified:
            head += f", {self.certified} with proofs"
        if self.duplicates:
            head += f", {self.duplicates} duplicate"
        capped = (f"\n  · CAPPED (state space exhausted): {', '.join(sorted(self.capped))}"
                  if self.capped else "")
        notes = ("\n  · " + "\n  · ".join(self.notes)) if self.notes else ""
        return f"{head}{' [' + kinds + ']' if kinds else ''}{capped}{notes}"

    def to_dict(self) -> Dict[str, Any]:
        return {"generated": self.generated, "verified": self.verified,
                "rejected": self.rejected, "duplicates": self.duplicates,
                "certified": self.certified, "by_kind": dict(self.by_kind),
                "capped": list(self.capped), "acceptance": round(self.acceptance, 4),
                "notes": list(self.notes)}


def doc(prompt: str, answer: str, certificate: Any = None) -> str:
    """Render a supervised pair in her own template — train/inference parity.

    Routes through :mod:`~nyxara.growth.certify` so a generator that proved its claim can carry
    the witness into the document, and one that cannot simply passes ``None``.
    """
    try:
        from nyxara.growth.certify import render_certified_doc

        return render_certified_doc(prompt, answer, certificate)
    except Exception:  # noqa: BLE001 — a plain rendering beats losing the pair
        return f"{prompt}\n\n{answer}\n"


def draw(generators: Sequence[Generator], n: int, *, seed: int = 0,
         report: Optional[SynthReport] = None,
         certify: bool = False) -> Iterator[str]:
    """Stream ``n`` documents across ``generators``, respecting every declared state space.

    Allocation is proportional to ``weight``, then clipped to each generator's ``cap``. When the
    clipping leaves the request unmet, the shortfall is redistributed to generators that still
    have room — and if *every* generator is capped, the report says so by name rather than
    quietly returning fewer documents than asked for, or the same ones repeatedly.

    Exact duplicates are dropped here rather than left for the corpus dedup, so ``by_kind``
    counts distinct output and a narrow generator is visible immediately.
    """
    report = report if report is not None else SynthReport()
    n = max(0, int(n))
    usable = [g for g in generators if g.weight > 0 and g.space > 0]
    if n == 0 or not usable:
        return

    quota = _allocate(usable, n, report)
    seen: set = set()
    rng = random.Random(seed)

    for gen in usable:
        want = quota.get(gen.name, 0)
        if want <= 0:
            continue
        made = 0
        # Bounded attempts: a generator whose parameters often collide must not spin. 6x the
        # request is generous and always terminates.
        for _ in range(want * 6):
            if made >= want:
                break
            report.generated += 1
            try:
                built = gen.fn(rng)
            except Exception as exc:  # noqa: BLE001 — one bad draw is not a dead generator
                report.rejected += 1
                report.note(f"{gen.name} raised: {type(exc).__name__}: {exc}")
                continue
            if built is None:
                report.rejected += 1
                continue
            prompt, answer = built
            key = hash((prompt, answer))
            if key in seen:
                report.duplicates += 1
                continue
            seen.add(key)

            certificate = None
            if certify and gen.proof_kind:
                certificate = _certify(prompt, answer, gen)
                if certificate is not None:
                    report.certified += 1
            report.verified += 1
            report.credit(gen.name)
            made += 1
            yield doc(prompt, answer, certificate)
        if made < want:
            report.note(f"{gen.name} produced {made} of {want} requested")


def _allocate(generators: Sequence[Generator], n: int,
              report: SynthReport) -> Dict[str, int]:
    """Split ``n`` across generators by weight, never exceeding any generator's cap."""
    quota: Dict[str, int] = {}
    remaining = n
    pool = list(generators)
    # Repeatedly hand out the remaining budget by weight; generators that hit their cap drop
    # out and their shortfall flows to the rest. Terminates because `pool` strictly shrinks or
    # the budget is fully allocated.
    while remaining > 0 and pool:
        total_weight = sum(g.weight for g in pool) or 1.0
        capped_now: List[Generator] = []
        handed = 0
        for gen in pool:
            share = int(remaining * gen.weight / total_weight) or (1 if remaining > handed else 0)
            room = gen.cap - quota.get(gen.name, 0)
            take = min(share, room)
            if take > 0:
                quota[gen.name] = quota.get(gen.name, 0) + take
                handed += take
            if quota.get(gen.name, 0) >= gen.cap:
                capped_now.append(gen)
                if gen.name not in report.capped:
                    report.capped.append(gen.name)
        remaining -= handed
        for gen in capped_now:
            pool.remove(gen)
        if handed == 0:
            break
    if remaining > 0:
        report.note(f"{remaining} document(s) unallocated — every generator is at its state-space "
                    f"cap; widen a generator rather than drawing the same items again")
    return quota


def _certify(prompt: str, answer: str, gen: Generator) -> Any:
    """Best-effort machine-checkable witness for one item. ``None`` when undecided."""
    try:
        from nyxara.growth.certify import certify_claim

        return certify_claim(answer.strip().splitlines()[-1] if answer else prompt,
                             kind=gen.proof_kind)
    except Exception:  # noqa: BLE001 — an unavailable prover certifies nothing
        return None
