#!/usr/bin/env python3
"""Convert a ConceptNet 5.7 assertions dump into NYXARA's triple JSONL.

``nyxara.njp.ingest.ingest_triples`` reads ``{"subject", "predicate", "object", "confidence"}``
rows. This reshapes ConceptNet's five-column assertions TSV into exactly that, and it exists as a
separate offline script rather than as a loader inside the package for two reasons: the dump is
~475 MB and belongs on disk once rather than in every session, and no test may open a socket.

Standard library only. Examples
-------------------------------
    # the normal path: you already have the dump
    python scripts/prepare_conceptnet.py \\
        --in conceptnet-assertions-5.7.0.csv.gz --out conceptnet_en.jsonl.gz

    # fetch it first (~475 MB, one HTTP GET, no third-party packages)
    python scripts/prepare_conceptnet.py --download --out conceptnet_en.jsonl.gz

Then feed it to her:
    nyxara-study --ingest conceptnet_en.jsonl.gz

Three decisions in here are worth reading before changing the tables below.

**Relation names are translated, not folded.** ``Grounder._predicate`` normalises spelling and
folds synonyms, but it does **not** split CamelCase: measured, ``_predicate("IsA")`` is ``"isa"``
and ``_predicate("PartOf")`` is ``"partof"``. Those are not near-misses, they are names no
question ever forms, so the fact would be stored and unreachable. The map therefore emits this
package's own snake_case names and the ingest path folds from there.

**``/r/AtLocation`` is deliberately absent.** It would map to ``located_in``, which is in
``grounding._FUNCTIONAL`` — a relation that holds exactly one value, where a second is a
*contradiction* rather than an addition. ConceptNet lists dozens of locations per concept.
``_assert`` does not run contradiction detection, so the load itself would survive; the damage
arrives on the first conversational turn about where something is, when ``_revise`` fires against
an arbitrary one of forty. That is the ``is_a``-in-``_FUNCTIONAL`` failure documented in
``grounding.py``, reached by a side door.

**``/r/RelatedTo`` is deliberately absent.** It is the largest relation in ConceptNet by a wide
margin, carries no direction, and ``related_to`` is already what ``_predicate`` returns when it
cannot name a relation at all. Admitting millions of them would make "I could not read this" and
"these things are related" the same edge.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import tempfile
import urllib.request
from typing import Any, Dict, Iterator, List, Optional

#: The official 5.7.0 assertions dump. Verified reachable at 497,963,447 bytes.
_DUMP_URL = ("https://s3.amazonaws.com/conceptnet/downloads/2019/edges/"
             "conceptnet-assertions-5.7.0.csv.gz")

#: ConceptNet relation → the name this package uses for that edge.
#:
#: Every target is either reachable by retrieval (it appears in ``grounding._GENERAL_ANSWER`` or
#: ``_PREDICATE_AFFINITY``) or chainable by the Core (it carries a transitivity prior in
#: ``njp/core.py``), and `tests/njp/test_conceptnet.py` asserts that for every row so a future
#: addition cannot quietly emit a dead edge.
#:
#: The four with transitivity priors lead, because they are the ones ``core._compose`` can chain:
#: ``part_of`` 0.70, ``is_a`` 0.66, ``causes`` 0.62, ``requires`` 0.55.
_RELATIONS: Dict[str, str] = {
    "/r/IsA": "is_a",                  # _GENERAL_ANSWER, affinity 1.0, drives core._inherit
    "/r/PartOf": "part_of",            # _GENERAL_ANSWER; highest transitivity prior
    "/r/MadeOf": "consists_of",        # _GENERAL_ANSWER
    "/r/UsedFor": "purpose",           # _GENERAL_ANSWER
    "/r/HasSubevent": "occurs_when",   # _GENERAL_ANSWER
    "/r/DefinedAs": "means",           # _GENERAL_ANSWER, affinity 0.95 to is_a
    "/r/HasA": "has_part",             # appears in _PREDICATE_AFFINITY values
    "/r/Synonym": "also_known_as",     # feeds grounding._neighbours
    "/r/Causes": "causes",             # _LAW_PREDICATES -> world.state_law
    "/r/CausesDesire": "causes",
    "/r/HasPrerequisite": "requires",  # _LAW_PREDICATES
    "/r/MotivatedByGoal": "requires",
    # These two are what made schema induction work when it was tested: eight birds and six fish
    # as is_a / has_property / capable_of produced six schemas, 13/13 on a held-out fold, and
    # correct transfer (salmon -> swim, sparrow -> fly). `has_property` in particular is what
    # `ConceptGenesis` turns into invariants, and invariants are what a role is made of.
    #
    # They are honestly weaker than the rest on one axis and the tests say so: neither appears in
    # `_PREDICATE_AFFINITY`, and — measured — no question pattern reads into them either.
    # `_read_question("what is sparrow capable of?")` returns `('sparrow capable of', 'is_a')`.
    # So `_lookup(subject, 'capable_of')` finds them and the Core reasons over them, while a
    # Master asking in English does not reach them. That is a gap in the question grammar rather
    # than in this map, it is measured rather than assumed, and adding affinity rows would not
    # have closed it — which is why none are asserted here on a guess.
    "/r/CapableOf": "capable_of",
    "/r/HasProperty": "has_property",
}


def _surface(uri: str, *, lang: str) -> Optional[str]:
    """The plain phrase a ``/c/<lang>/<word>[/pos[/sense]]`` URI names, or None.

    The POS and sense suffixes are dropped rather than the row: ``/c/en/dog`` and ``/c/en/dog/n``
    are the same concept for a fact store, and the ingest path's dedup collapses the pair. Losing
    every sense-tagged edge instead would throw away most of the useful half of the dump.
    """
    parts = uri.split("/")
    if len(parts) < 4 or parts[1] != "c" or parts[2] != lang:
        return None
    word = parts[3].replace("_", " ").strip()
    if not word or len(word) > 64:
        return None
    return word


def _weight(meta: str) -> Optional[float]:
    try:
        return float(json.loads(meta).get("weight", 0.0))
    except (ValueError, TypeError, AttributeError):
        return None


def _rows(path: str) -> Iterator[List[str]]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 5:
                yield parts


def convert(path: str, *, lang: str = "en", min_weight: float = 2.0,
            relations: Optional[Dict[str, str]] = None,
            counters: Optional[Dict[str, int]] = None) -> Iterator[Dict[str, Any]]:
    """Yield one ingest-shaped row per assertion that survives every filter."""
    table = relations if relations is not None else _RELATIONS
    seen = counters if counters is not None else {}

    def bump(name: str) -> None:
        seen[name] = seen.get(name, 0) + 1

    for parts in _rows(path):
        bump("read")
        predicate = table.get(parts[1])
        if predicate is None:
            bump("dropped_relation")
            continue
        subject = _surface(parts[2], lang=lang)
        obj = _surface(parts[3], lang=lang)
        if subject is None or obj is None:
            bump("dropped_language")
            continue
        if subject == obj:
            bump("dropped_self")
            continue
        weight = _weight(parts[4])
        if weight is None:
            bump("dropped_unparsed")
            continue
        if weight < min_weight:
            bump("dropped_weight")
            continue
        bump("kept")
        # ConceptNet's weight counts independent contributors, so it is corroboration rather than
        # certainty, and it is capped well below what a stated fact earns. One contributor is
        # testimony; `--min-weight` is where the line between testimony and evidence is drawn.
        yield {"subject": subject, "predicate": predicate, "object": obj,
               "confidence": round(min(0.9, 0.5 + 0.1 * weight), 4)}


def _sample(rows: Iterator[Dict[str, Any]], *, cap: int, seed: int) -> List[Dict[str, Any]]:
    """Reservoir-sample (Vitter R) at most ``cap`` rows in one streaming pass.

    Sampled rather than truncated, for the reason ``study.Corpus.load`` gives: taking the first
    N rows of a file written in some order is not a smaller corpus, it is a different one. The
    ConceptNet dump is sorted by relation URI, so ``--cap`` on the head would mean everything she
    knows is whichever relation sorts first. Seeded, so the same ``(cap, seed)`` gives the same
    corpus on every machine — which is what makes a before/after comparison mean anything.
    """
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index < cap:
            out.append(row)
            continue
        j = rng.randint(0, index)
        if j < cap:
            out[j] = row
    return out


def _download(url: str, dest: str) -> str:
    """Fetch the dump with the standard library. The only code path here that opens a socket."""
    print(f"downloading {url}\n  → {dest} (~475 MB, this takes a while)", file=sys.stderr)
    handle, tmp = tempfile.mkstemp(dir=os.path.dirname(dest) or ".", suffix=".part")
    os.close(handle)
    try:
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 — a fixed, documented https URL
        os.replace(tmp, dest)
    except Exception:
        # A half-written dump that looks complete is worse than none: the next run would convert
        # it and report a corpus that is quietly a fraction of what was asked for.
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return dest


def _write(rows: List[Dict[str, Any]], out: str) -> int:
    opener = gzip.open if out.endswith(".gz") else open
    with opener(out, "wt", encoding="utf-8") as handle:  # type: ignore[operator]
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="infile", help="ConceptNet assertions .csv / .csv.gz")
    ap.add_argument("--out", dest="outfile", required=True, help="destination .jsonl / .jsonl.gz")
    ap.add_argument("--download", action="store_true",
                    help=f"fetch the dump from {_DUMP_URL} first (opens a socket)")
    ap.add_argument("--lang", default="en", help="ConceptNet language code (default: en)")
    ap.add_argument("--min-weight", type=float, default=2.0,
                    help="drop edges below this many independent contributors (default: 2.0)")
    ap.add_argument("--cap", type=int, default=250_000,
                    help="keep at most N triples, reservoir-sampled (0 = all)")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (default: 0)")
    ap.add_argument("--relation", action="append", default=None, metavar="/r/Name",
                    help="restrict to these ConceptNet relations (repeatable)")
    args = ap.parse_args(argv)

    source = args.infile
    if args.download:
        source = _download(_DUMP_URL, args.infile or "conceptnet-assertions-5.7.0.csv.gz")
    if not source:
        return int(bool(sys.stderr.write("need --in PATH or --download\n"))) or 2
    if not os.path.exists(source):
        sys.stderr.write(f"no such file: {source}\n")
        return 2

    table = _RELATIONS
    if args.relation:
        wanted = set(args.relation)
        unknown = wanted - set(_RELATIONS)
        if unknown:
            sys.stderr.write(f"unmapped relation(s): {', '.join(sorted(unknown))}\n")
            return 2
        table = {k: v for k, v in _RELATIONS.items() if k in wanted}

    counters: Dict[str, int] = {}
    rows = convert(source, lang=args.lang, min_weight=args.min_weight,
                   relations=table, counters=counters)
    kept = _sample(rows, cap=args.cap, seed=args.seed) if args.cap > 0 else list(rows)
    written = _write(kept, args.outfile)

    print(json.dumps({"read": counters.get("read", 0),
                      "kept": counters.get("kept", 0),
                      "written": written,
                      "sampled": counters.get("kept", 0) > written,
                      "dropped": {k[len("dropped_"):]: v for k, v in sorted(counters.items())
                                  if k.startswith("dropped_")}}, indent=1))
    print(f"\nuse it:  nyxara-study --ingest {args.outfile}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
