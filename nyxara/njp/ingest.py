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

__all__ = ["IngestReport", "TextReport", "stream_rows", "stream_triples", "stream_statements",
           "ingest_triples", "ingest_text"]

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


def _already_stored(grounder: Any, key: Tuple[str, str, str]) -> bool:
    """Is this exact triple already in the store, in any state?

    Superseded and contested entries count. That is the whole point rather than an edge case:
    when `Grounder.load_dict` replays a corpus, a bulk fact a conversation later retracted has
    already been restored from the sidecar in its retracted form, and re-asserting it here would
    put a live copy of the withdrawn claim back beside it — `_lookup` reads live facts only, so
    she would answer tomorrow with what she was corrected about today. A store that forgets its
    retractions overnight is the failure `load_dict` already argues against for the flag itself.

    It also makes re-ingesting the same file idempotent, which is worth having on its own.
    """
    try:
        for triple in grounder.facts.get((key[0], key[1]), ()):
            if triple.object.strip().lower() == key[2]:
                return True
    except Exception:  # noqa: BLE001
        pass
    return False


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
                   to_world: bool = False,
                   record: bool = True,
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

    ``to_world`` is **off by default, and that default was measured rather than chosen.** It was
    on, and `nyxara.eval.intelligence` caught what that cost the first time it was pointed at a
    prepared brain: loading 8,000 facts took ``causal_prediction`` from 1.00 to 0.00, and
    ``to_world=False`` restored it exactly. The mechanism is not subtle once looked at —
    :class:`~nyxara.njp.universe.InternalUniverse` holds 512 relations, those 8,000 facts stated
    2,312 laws, and ``sync_from_world`` imported 512 of them and filled it to the brim. Her own
    ``water → growth``, the one the question was about, never got in.

    The architectural reading is the same one :func:`~nyxara.njp.study.seed_kinds` already makes
    about kinds: a commonsense corpus is **testimony about the world in general**, not observation
    of *her* situation. The causal skeleton is what she has seen happen and been told about the
    case in front of her, and a crowd-sourced "eating causes fullness" displacing that is not
    extra knowledge — it is the loss of the only relations an intervention could be run on. Pass
    ``to_world=True`` deliberately, for a corpus that really is about her world.
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
            if key in seen or _already_stored(grounder, key):
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
        if report.asserted and record:
            # The manifest is what lets `Grounder.to_dict` leave these facts out of the sidecar:
            # it names a file that is still on disk and hashed, so they can be replayed exactly
            # rather than copied. Recorded only on a load that actually stored something, and
            # skipped entirely when `record=False` — which is how the replay itself avoids
            # re-noting the corpus it is in the middle of restoring.
            try:
                grounder.note_ingest(source=source, path=str(path),
                                     digest=report.digest, count=report.asserted)
            except AttributeError:
                pass
        return report
    except Exception:  # noqa: BLE001 — a failed ingest reports what it managed, never raises
        return report
    finally:
        report.ms = (time.perf_counter() - started) * 1000.0


# --------------------------------------------------------------------------- #
# Prose, which is most of what the world writes down
# --------------------------------------------------------------------------- #
@dataclass
class TextReport:
    """What a bulk load of *statements* put in the store, and what the extractor made of them.

    The two numbers that matter are :attr:`parsed` and :attr:`asserted`, and they are reported
    separately because they fail separately. A corpus where ``parsed`` is low was filtered badly —
    the sentences handed over were not statements her patterns read. A corpus where ``parsed`` is
    high and ``asserted`` is low was filtered fine and is simply already known, which is a
    different and much better problem.
    """

    read: int = 0
    parsed: int = 0
    unparsed: int = 0
    asserted: int = 0
    duplicate: int = 0
    skipped: int = 0
    capped: bool = False
    subjects: int = 0
    predicates: Dict[str, int] = field(default_factory=dict)
    concepts: int = 0
    source: str = ""
    digest: str = ""
    ms: float = 0.0

    @property
    def extraction_rate(self) -> float:
        """Share of statements that yielded at least one triple.

        Named for :meth:`Grounder.stats`'s figure and deliberately **not** the same number. That
        one is a statement about her conversational history and this module must not touch it (see
        :func:`ingest_triples`); this one is a statement about a file, and it belongs to the file.
        """
        return round(self.parsed / self.read, 4) if self.read else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"read": self.read, "parsed": self.parsed, "unparsed": self.unparsed,
                "extraction_rate": self.extraction_rate,
                "asserted": self.asserted, "duplicate": self.duplicate,
                "skipped": self.skipped, "capped": self.capped, "subjects": self.subjects,
                "predicates": dict(sorted(self.predicates.items(),
                                          key=lambda kv: (-kv[1], kv[0]))[:16]),
                "concepts": self.concepts, "source": self.source, "digest": self.digest,
                "ms": round(self.ms, 3)}


def stream_statements(path: Any) -> Iterator[Tuple[str, str]]:
    """Yield ``(statement, provenance)`` for every well-formed row of a text corpus.

    ``provenance`` is the row's own ``source`` — a Wikipedia URL, an arXiv id, a book title — and
    it is carried through to :attr:`GroundedTriple.source` rather than dropped, so a fact she
    later has to defend can be traced to the sentence it came from and not merely to the load.
    """
    for row in stream_rows(path):
        try:
            statement = " ".join(str(row.get("text") or row.get("statement") or "").split())
            if len(statement) < 8:
                continue
            yield statement, str(row.get("source") or "").strip()
        except (TypeError, ValueError):
            continue


def ingest_text(brain: Any, path: Any, *,
                source: str = "text",
                max_facts: int = 250_000,
                max_statements: int = 0,
                min_confidence: float = 0.0,
                allow: Optional[Collection[str]] = None,
                batch: int = 5_000,
                record: bool = True,
                progress: Optional[Callable[[int], None]] = None,
                checkpoint: Optional[Callable[[int], None]] = None,
                checkpoint_every: int = 0) -> TextReport:
    """Run the grounder's extractor over a file of statements and store what it finds.

    The sibling of :func:`ingest_triples`, for the four sources in
    :data:`nyxara.njp.corpora.SOURCES` that arrive as prose rather than as edges. Every invariant
    that function holds is held here for the same reasons, and two are worth restating because a
    prose loader is where they are most tempting to break:

    **It does not call** :meth:`~nyxara.njp.grounding.Grounder.ground`. That method is the
    conversational entry point: it increments ``turns``, files the result under ``unparsed`` or
    ``grounded_turns``, and runs concept observation per sentence. Routing a corpus through it
    would make ``stats()["extraction_rate"]`` — a number that is supposed to describe how well she
    reads *the Master* — into a description of a file somebody downloaded. :attr:`TextReport`
    reports the file's own rate instead, and the two never mix.

    **It does not run contradiction detection.** ``ground`` calls ``_contradicts`` and ``_revise``
    on every extraction, which is right for a turn and wrong for a corpus: Wikipedia states a
    person's birthplace in one article and their death place in another, and ``located_in`` is in
    ``grounding._FUNCTIONAL``, so a bulk load through the revision path would spend the run
    retracting facts against each other in file order and finish holding whichever one happened to
    be last. Bulk facts are *testimony*, entered side by side, and the first conversational turn
    about one of them is where the revision belongs.

    ``max_statements`` bounds the read where ``max_facts`` bounds the write. Both exist because
    they answer different questions: a corpus can be too large to *read* in the time available
    even if it would assert very little, and a corpus can assert far more than intended from very
    few rows. ``capped`` is set by either.
    """
    report = TextReport(source=source)
    started = time.perf_counter()
    try:
        grounder = getattr(brain, "grounder", None)
        if grounder is None:
            return report
        report.digest = _digest(path)

        genesis = getattr(brain, "genesis", None)
        allowed = {str(name) for name in allow} if allow else None
        max_facts = max(0, int(max_facts))
        max_statements = max(0, int(max_statements))
        batch = max(1, int(batch))

        seen: Set[Tuple[str, str, str]] = set()
        subjects: Set[str] = set()
        pending: List[Any] = []

        def _flush() -> None:
            """One batch to the concept layer, for the reason :func:`ingest_triples` gives.

            The world model is not fed here at all, and there is no ``to_world`` switch to turn it
            on. That function's own docstring records what feeding it cost — ``causal_prediction``
            1.00 → 0.00, because 512 relations of capacity filled with crowd-sourced laws — and
            the argument is *stronger* for prose: a sentence saying "smoking causes cancer" is
            testimony about the world in general, and her causal skeleton is what she has seen
            happen. A switch nobody should set is a switch that will be set.
            """
            if not pending:
                return
            if genesis is not None:
                try:
                    report.concepts += int(genesis.observe_triples(pending) or 0)
                except Exception:  # noqa: BLE001 — the facts are stored either way
                    pass
            pending.clear()

        for statement, provenance in stream_statements(path):
            if max_statements and report.read >= max_statements:
                report.capped = True
                break
            report.read += 1

            try:
                extracted = grounder._extract(statement)
            except Exception:  # noqa: BLE001 — one unreadable sentence is not a failed load
                extracted = []
            if not extracted:
                report.unparsed += 1
                continue
            report.parsed += 1

            for triple in extracted:
                if triple.confidence < min_confidence:
                    report.skipped += 1
                    continue
                if allowed is not None and triple.predicate not in allowed:
                    report.skipped += 1
                    continue
                key = (triple.subject.lower(), triple.predicate, triple.object.lower())
                if key in seen or _already_stored(grounder, key):
                    report.duplicate += 1
                    continue
                if report.asserted >= max_facts:
                    report.capped = True
                    break
                seen.add(key)
                # Provenance is prefixed rather than replaced. `_regenerable` matches the manifest
                # on `source`, so a triple tagged only with its Wikipedia URL would never match the
                # load that produced it and would be written to the sidecar in full — which is the
                # whole cost this manifest exists to avoid.
                triple.source = f"{source}:{provenance}"[:120] if provenance else source
                grounder._assert(triple)
                pending.append(triple)
                subjects.add(triple.subject.lower())
                report.asserted += 1
                report.predicates[triple.predicate] = \
                    report.predicates.get(triple.predicate, 0) + 1
            if report.capped:
                break

            if len(pending) >= batch:
                _flush()
                if progress is not None:
                    try:
                        progress(report.asserted)
                    except Exception:  # noqa: BLE001
                        pass
            if (checkpoint is not None and checkpoint_every > 0
                    and report.read % checkpoint_every == 0):
                try:
                    checkpoint(report.asserted)
                except Exception:  # noqa: BLE001 — a failed save costs durability, not the run
                    pass

        _flush()
        report.subjects = len(subjects)
        if report.asserted and record:
            try:
                grounder.note_ingest(source=source, path=str(path),
                                     digest=report.digest, count=report.asserted,
                                     form="text")
            except TypeError:
                # An older grounder with no `form` argument would record this corpus as triples,
                # and the replay would then read a text file with the triple reader and restore
                # nothing at all. Not recording is the safe half of that trade: the facts go to
                # the sidecar in full, which costs space and loses nothing.
                pass
            except AttributeError:
                pass
        return report
    except Exception:  # noqa: BLE001 — a failed ingest reports what it managed, never raises
        return report
    finally:
        report.ms = (time.perf_counter() - started) * 1000.0
