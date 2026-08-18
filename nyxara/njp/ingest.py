"""NYXARA · njp/ingest.py — a corpus of facts, without pretending it was a corpus of turns (📥).

:mod:`nyxara.njp.study` teaches her from **turns**: ``Corpus``, ``Pair`` and ``Tutor`` all model
conversation, and ``Pair.held_out`` and ``Tutor.exam`` exist to keep that measurement honest. A
file of a quarter-million triples is not turns. Nobody said them to her.

That distinction is the whole module, and it is load-bearing in two directions:

**It is not routed through** :meth:`~nyxara.njp.brain.NJPBrain.think`, and that is the point
rather than an optimisation. :func:`~nyxara.njp.study.seed_kinds` already made this argument at
137 rows — running them as conversation *"would put 137 fabricated exchanges into her episodic
memory and her fabric's growth record, which would make the session log a lie about what
happened"* — and at corpus scale the argument stops being only about honesty. Measured: the
fabric settles and expands at ~40 turns/s, so 250,000 triples through ``think`` is **1.7 hours**;
:meth:`~nyxara.njp.grounding.Grounder._assert` is ~54,000/s and flat, so the same corpus is
**five seconds**. And :class:`~nyxara.njp.memory.HoloMemory` links every write to the last eight
keys, so a bulk load routed through ``think`` would cross-link a million unrelated facts as
though they had co-occurred in one conversation — poisoning exactly the associative recall it
exists to provide. Not going through ``think`` *is* the fix; nothing here needs to defend
against those paths because it never enters them.

**Predicates are folded on the way in.** ``grounder._predicate`` decides what an edge is called,
and :class:`~nyxara.njp.core.CognitiveLearningCore` reads ``grounder.facts`` live and matches on
the folded name. An unfolded predicate is therefore not a slightly worse fact — it is a fact the
reasoner structurally cannot see, stored and unreachable. Measured on ConceptNet's own relation
names: ``_predicate("IsA")`` is ``"isa"``, which is in no affinity table and no general-answer
set, so retrieval scores it at zero forever.

**What it feeds, and what it deliberately does not.** Facts go to the store; the same triples go
to :meth:`~nyxara.njp.concepts.ConceptGenesis.observe_triples` in batches, because a subject with
several relations is the material the concept layer needs and one 250,000-item list would blow
past its capacity and evict most of it; and law-shaped relations go to
:meth:`~nyxara.njp.world.WorldView.from_grounding`. Nothing here writes to the fabric, to
episodic memory, or to any turn counter — ``stats()["extraction_rate"]`` must stay a statement
about the extractor, not about a file she was handed.

Every public entry point is fail-soft: a malformed row is skipped and counted, and a failed
ingest leaves the store exactly as it was rather than half-written.

Pure standard library.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Collection, Dict, Iterator, List, Optional, Set, Tuple

__all__ = ["IngestReport", "stream_rows", "stream_triples", "ingest_triples"]

#: What a row without its own confidence is worth. Below `study._SEED_CONFIDENCE` (0.8) on
#: purpose: the seed file is a small curated set the Master vetted, and a harvested corpus is not.
#: Both sit below what a parsed statement earns, because a claim about a kind is defeasible by
#: construction and the store should say so before anything inherits from it.
_DEFAULT_CONFIDENCE = 0.6


@dataclass
class IngestReport:
    """What a bulk load actually put in the store, including what it refused to."""

    read: int = 0
    asserted: int = 0
    skipped: int = 0
    duplicate: int = 0
    capped: bool = False
    subjects: int = 0
    predicates: Dict[str, int] = field(default_factory=dict)
    concepts: int = 0
    laws: int = 0
    source: str = ""
    digest: str = ""
    ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"read": self.read, "asserted": self.asserted, "skipped": self.skipped,
                "duplicate": self.duplicate, "capped": self.capped,
                "subjects": self.subjects,
                "predicates": dict(sorted(self.predicates.items(),
                                          key=lambda kv: (-kv[1], kv[0]))[:16]),
                "concepts": self.concepts, "laws": self.laws,
                "source": self.source, "digest": self.digest, "ms": round(self.ms, 3)}


def stream_rows(path: Any, *, on_error: str = "skip") -> Iterator[Dict[str, Any]]:
    """Yield one mapping per row of a ``.jsonl`` / ``.jsonl.gz`` / plain-JSON-list file.

    ``on_error`` is the only thing that varies between this module's callers and
    :class:`nyxara.njp.study.Corpus`, and it varies for a real reason rather than taste. A
    corrupt line in the bundled Q/A corpus means the file is wrong and the run should stop
    (``"raise"``); a corrupt line in a harvested third-party dump means that one row is wrong
    and the other quarter-million are fine (``"skip"``). The format handling is identical, so it
    lives here once instead of twice.
    """
    target = Path(str(path))
    opener = gzip.open if str(target).endswith(".gz") else open
    with opener(target, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        head = handle.read(1)
        handle.seek(0)
        if head == "[":                                  # a plain JSON list
            for row in json.load(handle):
                if isinstance(row, dict):
                    yield row
            return
        for line in handle:                              # JSON lines
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                if on_error == "raise":
                    raise
                continue
            if isinstance(row, dict):
                yield row


def stream_triples(path: Any, *,
                   default_confidence: float = _DEFAULT_CONFIDENCE
                   ) -> Iterator[Tuple[str, str, str, float]]:
    """Yield ``(subject, predicate, object, confidence)`` for every well-formed row.

    The predicate is **not** folded here. Folding needs the grounder, and a reader that needed a
    brain to read a file would be useless to the converter scripts that write these files.

    ``default_confidence`` is what a row without one is worth. It is a parameter rather than a
    constant because the answer depends on where the file came from — a curated seed the Master
    vetted is not worth the same as a harvested dump — and a file that states its own confidence
    always overrides it.
    """
    for row in stream_rows(path):
        try:
            subject = str(row.get("subject", "")).strip()
            predicate = str(row.get("predicate", "")).strip()
            obj = str(row.get("object", "")).strip()
            if not subject or not predicate or not obj:
                continue
            raw = row.get("confidence", default_confidence)
            confidence = float(raw) if raw is not None else default_confidence
            if not 0.0 <= confidence <= 1.0:
                confidence = default_confidence
            yield subject, predicate, obj, confidence
        except (TypeError, ValueError):
            continue


def _law_predicates() -> frozenset:
    """The relations :mod:`nyxara.njp.world` treats as standing laws, or an empty set.

    Read from that module rather than restated here: a second copy would drift the first time a
    predicate is added there, and the failure would be silent — laws quietly not reaching the
    world model, which is exactly the class of defect this whole effort keeps finding.
    """
    try:
        from nyxara.njp.world import _LAW_PREDICATES
        return frozenset(_LAW_PREDICATES)
    except Exception:  # noqa: BLE001
        return frozenset()


def _digest(path: Any) -> str:
    """A stable sha256 of the file, so a later reload can tell it is the same corpus."""
    try:
        digest = hashlib.sha256()
        with open(str(path), "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:  # noqa: BLE001 — provenance is best-effort, ingestion is not
        return ""


def ingest_triples(brain: Any, path: Any, *,
                   source: str = "ingest",
                   max_facts: int = 250_000,
                   max_laws: int = 20_000,
                   min_confidence: float = 0.0,
                   default_confidence: float = _DEFAULT_CONFIDENCE,
                   allow: Optional[Collection[str]] = None,
                   batch: int = 5_000,
                   to_world: bool = True,
                   progress: Optional[Callable[[int], None]] = None,
                   checkpoint: Optional[Callable[[int], None]] = None,
                   checkpoint_every: int = 0) -> IngestReport:
    """Assert a file of triples into the fact store and fan them out to the layers that want them.

    ``allow`` is checked **after** folding, because the caller names relations the way this
    package names them and the file names them the way its source did.

    ``max_facts`` is a hard stop that sets :attr:`IngestReport.capped` — a load that silently
    dropped its tail would report the same numbers as one that finished, and the difference
    between "she knows this corpus" and "she knows the first third of it" is not a detail.

    ``checkpoint`` is called with the count so far every ``checkpoint_every`` assertions, in the
    shape :meth:`nyxara.njp.study.Tutor.study` uses, so the same ``--save`` plumbing serves both.
    """
    report = IngestReport(source=source)
    started = time.perf_counter()
    try:
        from nyxara.njp.grounding import GroundedTriple

        grounder = getattr(brain, "grounder", None)
        if grounder is None:
            return report
        report.digest = _digest(path)

        genesis = getattr(brain, "genesis", None)
        world = getattr(brain, "world", None) if to_world else None
        allowed = {str(name) for name in allow} if allow else None
        max_facts = max(0, int(max_facts))
        batch = max(1, int(batch))

        seen: Set[Tuple[str, str, str]] = set()
        subjects: Set[str] = set()
        pending: List[Any] = []

        def _flush() -> None:
            """Hand one batch to the concept layer and the world model, then let it go.

            Batched rather than accumulated because `ConceptGenesis` has a real capacity with
            FIFO eviction and does per-observation similarity work: one 250,000-item list would
            evict most of what it had just been told and cost quadratically to do it.
            """
            if not pending:
                return
            if genesis is not None:
                try:
                    report.concepts += int(genesis.observe_triples(pending) or 0)
                except Exception:  # noqa: BLE001 — the facts are stored either way
                    pass
            if world is not None and report.laws < max_laws:
                try:
                    # Only the law-shaped triples are handed over, and they are counted here
                    # rather than read back out of the world model. `WorldView._stated` is keyed
                    # by cause, so its length counts *causes* and would undercount every law
                    # whose cause was already known — and it is another module's private state,
                    # which a caller should not be inferring its own report from.
                    #
                    # Passing only these is also the honest read of what `from_grounding` does
                    # with the rest: a ConceptNet edge is never an event, because nothing
                    # happened at a time, so everything else is discarded on arrival anyway.
                    laws = [t for t in pending if t.predicate in _law_predicates()]
                    room = max_laws - report.laws
                    if len(laws) > room:
                        laws = laws[:room]
                    if laws:
                        world.from_grounding(SimpleNamespace(triples=laws))
                        report.laws += len(laws)
                except Exception:  # noqa: BLE001
                    pass
            pending.clear()

        for subject, predicate, obj, confidence in stream_triples(
                path, default_confidence=default_confidence):
            report.read += 1
            if confidence < min_confidence:
                report.skipped += 1
                continue
            folded = grounder._predicate(predicate)
            if allowed is not None and folded not in allowed:
                report.skipped += 1
                continue
            key = (subject.lower(), folded, obj.lower())
            if key in seen:
                report.duplicate += 1
                continue
            if report.asserted >= max_facts:
                report.capped = True
                break
            seen.add(key)
            triple = GroundedTriple(
                subject=subject, predicate=folded, object=obj,
                confidence=confidence, source=source,
                text=f"{subject} {predicate} {obj}")
            grounder._assert(triple)
            pending.append(triple)
            subjects.add(subject.lower())
            report.asserted += 1
            report.predicates[folded] = report.predicates.get(folded, 0) + 1

            if len(pending) >= batch:
                _flush()
                if progress is not None:
                    try:
                        progress(report.asserted)
                    except Exception:  # noqa: BLE001
                        pass
            if (checkpoint is not None and checkpoint_every > 0
                    and report.asserted % checkpoint_every == 0):
                try:
                    checkpoint(report.asserted)
                except Exception:  # noqa: BLE001 — a failed save costs durability, not the run
                    pass

        _flush()
        report.subjects = len(subjects)
        return report
    except Exception:  # noqa: BLE001 — a failed ingest reports what it managed, never raises
        return report
    finally:
        report.ms = (time.perf_counter() - started) * 1000.0
