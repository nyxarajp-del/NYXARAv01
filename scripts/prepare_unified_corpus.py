#!/usr/bin/env python3
"""Compose the one corpus: every category of thing NJP can learn from, in a single record shape.

    scripts/knowledge/*.kb    (facts)      ┐
    scripts/experience/*.exp  (episodes)   ├──> nyxara/njp/data/world_unified.jsonl.gz
    scripts/unified/*.u       (the rest)   ┘

Why one file and one shape
--------------------------
The knowledge corpus taught her *what is*. The experience corpus taught her *what follows what,
and what it costs to be wrong about it*. Both are real and neither is intelligence, because the
organs that would turn one into the other are fed by neither: nothing was ever handed a rival
hypothesis to discriminate, a contradiction to revise, a concept to induce from members, a
principle to transfer to a domain it was never stated in, or a capability estimate to be wrong
about.

This file is the union. Every record carries a ``category`` and whatever *blocks* that category
needs, and :mod:`nyxara.njp.unified` routes each block to the organ that consumes it — the same
discipline as the other two generators, one level wider: **no block exists that no organ reads.**
``tests/njp/test_unified.py`` asserts that for all fifty categories, so a category cannot be added
here as a label with nothing behind it.

The record shape is the loop the Master named, with every stage optional::

    context → observation → knowledge → unknown → hypotheses → prediction → action → result
        → error → diagnosis → correction → belief → law → concept → generalization
        → counterfactual → transfer → confidence → verification

A knowledge record fills three of those and leaves seventeen empty; a discovery record fills
fifteen. That is the point of one shape rather than fifty: the *same* replay walks all of them,
so an organ that wants observations gets them from whichever category happens to carry them, and
adding a category later costs a block rather than a pipeline.

**What this is not.** It is not the scraped web, and no hand-written corpus can be. What a person
can write down is the *structure* — the laws, the rival hypotheses, the invariants, the transfers —
and that is exactly what a scraped corpus is worst at and what these organs need most.

The ``.u`` format
-----------------
One record per ``@record`` line, ``key = value`` under it, ``#`` comments, ``;;`` between items of
a list and ``|`` between the fields of an item::

    @record concept_formation birds_have_feathers
    domain     = biology
    context    = four animals are described and the shared invariant has to be found
    examples   = sparrow | feathers ;; eagle | feathers ;; owl | feathers ;; crow | feathers
    concept    = bird
    invariants = feathers
    generalize = penguin ?? feathers
    text       = a sparrow, an eagle, an owl and a crow all have feathers.

Every key's type comes from :data:`_FIELDS`, so a typo is a rejected file rather than a string
where a list was meant. Standard library only; nothing opens a socket.

Examples
--------
    python scripts/prepare_unified_corpus.py --out nyxara/njp/data/world_unified.jsonl.gz
    python scripts/prepare_unified_corpus.py --check
    python scripts/prepare_unified_corpus.py --category discovery --out -
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_U_DIR = os.path.join(_HERE, "unified")

sys.path.insert(0, _HERE)


class UnifiedError(ValueError):
    """A record that would carry a block no organ reads, with the file and line that wrote it."""


# --------------------------------------------------------------------------- #
# The fifty categories, and the organ each one exists to feed
# --------------------------------------------------------------------------- #
#: category -> (the attribute on the brain that consumes it, the blocks it must carry)
#:
#: The first element is a real attribute of :class:`~nyxara.njp.brain.NJPBrain` and
#: ``tests/njp/test_unified.py`` asserts that for every row. Guessing here is easy and silent: the
#: first version named ``selfmodel`` (it is ``self_model``) and ``hypotheses`` (the brain calls it
#: ``designer``), and both routed to ``None`` through a guard, so the report said zero capabilities
#: and zero hypotheses on a corpus full of both.
#:
#: The second element is what stops this being a list of aspirations. A category whose required
#: blocks are empty produces a record that routes nowhere, and the parser refuses it.
CATEGORIES: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    # --- what is ---------------------------------------------------------------------------- #
    "world_knowledge":     ("grounder",   ("knowledge",)),
    "language":            ("grounder",   ("text", "qa")),
    "qa":                  ("memory",     ("qa",)),
    "math":                ("grounder",   ("qa",)),
    "logic":               ("reasoner",   ("premises", "conclusion")),
    "science":             ("grounder",   ("knowledge",)),
    # --- what follows what ------------------------------------------------------------------ #
    "causal":              ("universe",   ("law",)),
    "counterfactual":      ("universe",   ("counterfactual",)),
    "world_state":         ("predictive", ("state", "action", "next_state")),
    "temporal":            ("world",      ("temporal",)),
    "spatial":             ("grounder",   ("spatial",)),
    # --- doing ------------------------------------------------------------------------------- #
    "planning":            ("goals",      ("goal", "steps")),
    "problem_solving":     ("goals",      ("problem", "attempts")),
    "experience":          ("predictor",  ("observation", "prediction", "result")),
    "prediction":          ("predictor",  ("prediction", "result")),
    "self_correction":     ("predictor",  ("prediction", "result", "correction")),
    # --- being wrong -------------------------------------------------------------------------- #
    "contradiction":       ("grounder",   ("contradiction",)),
    "uncertainty":         ("grounder",   ("claim", "epistemic")),
    "error_taxonomy":      ("predictor",  ("prediction", "result", "diagnosis", "evidence")),
    "calibration":         ("predictor",  ("prediction", "result", "confidence")),
    "knowledge_revision":  ("grounder",   ("knowledge", "contradiction")),
    # --- finding out ---------------------------------------------------------------------------#
    "hypothesis":          ("designer",   ("observation_text", "hypotheses")),
    "rival_hypotheses":    ("designer",   ("hypotheses", "experiment")),
    "experiment_design":   ("designer",   ("question", "experiment", "outcome")),
    "discovery":           ("universe",   ("variables", "series", "law")),
    "unknown_unknown":     ("assumptions", ("belief", "anomaly", "missing")),
    "curiosity":           ("curiosity",  ("unknown", "question")),
    "active_learning":     ("designer",   ("hypotheses", "experiment", "outcome")),
    "scientific_law":      ("universe",   ("variables", "series", "law")),
    "research":            ("goals",      ("question", "steps", "conclusion")),
    # --- abstraction ---------------------------------------------------------------------------#
    "concept_formation":   ("genesis",    ("examples", "concept")),
    "concept_composition": ("genesis",    ("parts", "concept")),
    "generalization":      ("discoverer", ("examples", "generalize", "concept")),
    "transfer":            ("grounder",   ("transfer",)),
    "analogy":             ("grounder",   ("analogy",)),
    "abstraction_rule":    ("discoverer", ("abstraction",)),
    # --- memory ---------------------------------------------------------------------------------#
    "memory_episode":      ("memory",     ("episode",)),
    "long_term":           ("levels",     ("episode", "consolidate")),
    "forgetting":          ("levels",     ("episode", "consolidate")),
    # --- others and self --------------------------------------------------------------------- #
    "social":              ("society",    ("agents", "claim")),
    "theory_of_mind":      ("society",    ("agents", "question")),
    "decision":            ("goals",      ("options", "choice")),
    "reinforcement":       ("predictor",  ("action", "reward")),
    "tool_use":            ("agent",     ("objective", "tool", "result")),
    "coding":              ("grounder",   ("problem", "solution")),
    "multimodal":          ("perceive",     ("modalities", "knowledge")),
    "grounding":           ("grounder",   ("observation_text", "knowledge")),
    "self_model":          ("self_model",  ("capability", "success")),
    "meta_learning":       ("self_model",  ("capability", "strategy", "success")),
    "strategy_selection":  ("metareason", ("problem", "strategy")),
    "creative_synthesis":  ("genesis",    ("parts", "concept", "novelty")),
}

#: Every field a record may carry, and how its text is read. A key not in here is a typo.
_LIST = "list"          # ``a ;; b ;; c``
_ITEMS = "items"        # ``a | b ;; c | d``  -> list of lists
_PAIRS = "pairs"        # ``q ?? a ;; q ?? a`` -> list of two-element lists
_TEXT = "text"
_NUMBER = "number"
_MAP = "map"            # ``k=v; k=v``        -> dict of floats

_FIELDS: Dict[str, str] = {
    "domain": _TEXT, "context": _TEXT, "text": _TEXT, "note": _TEXT,
    # what is
    "knowledge": _ITEMS, "qa": _PAIRS, "claim": _TEXT, "epistemic": _TEXT,
    "premises": _LIST, "conclusion": _TEXT, "valid": _TEXT,
    # what follows what
    "observation": _MAP, "order": _LIST, "law": _ITEMS, "counterfactual": _ITEMS,
    # A pattern needs more than one reading, and a record holds one `observation` map. `variables`
    # names the columns and `series` carries the rows, which is what a discovery actually looks
    # like: a table someone stared at until a law came out of it.
    "variables": _LIST, "series": _ITEMS,
    "state": _LIST, "next_state": _LIST, "temporal": _ITEMS, "spatial": _ITEMS,
    # doing
    "action": _TEXT, "actor": _TEXT, "object": _TEXT, "goal": _TEXT, "steps": _LIST,
    "problem": _TEXT, "attempts": _ITEMS, "solution": _TEXT, "reward": _NUMBER,
    "objective": _TEXT, "tool": _TEXT, "result": _TEXT, "options": _ITEMS, "choice": _TEXT,
    # being wrong
    "prediction": _ITEMS, "diagnosis": _TEXT, "correction": _TEXT, "confidence": _NUMBER,
    # `key | value` pairs handed to `predict.PredictionEngine.diagnose`. Its branches are reached
    # by *evidence* — `stimulus` with no `concepts` is a perception failure, `in_memory` without
    # `recalled` is a memory one — so a record that states a diagnosis and supplies no evidence
    # gets UNATTRIBUTED, correctly, and measures nothing. An empty value means absent.
    "evidence": _ITEMS,
    "contradiction": _PAIRS, "revision": _TEXT,
    # `subject | predicate | the old value | the new one`. The prose pair is for the language side
    # and cannot reach `_revise`: measured, `ground("the capital of myanmar is yangon")` extracts
    # no relation at all, so a revision test built on the sentences alone tests the pattern table.
    "clash": _ITEMS,
    # finding out
    "observation_text": _TEXT, "hypotheses": _ITEMS, "experiment": _ITEMS, "outcome": _ITEMS,
    "question": _TEXT, "unknown": _TEXT, "anomaly": _TEXT, "missing": _TEXT, "belief": _TEXT,
    # abstraction
    "examples": _ITEMS, "concept": _TEXT, "invariants": _LIST, "generalize": _PAIRS,
    # What the members *can do*, as opposed to what they *are*. Separate from `invariants` because
    # the relation differs and so does the only English that can ask for it: "is a seal warm
    # blooded" reaches `has_property`, and no phrasing of it reaches "breathes with lungs", which
    # is a `capable_of` claim wearing a property's field.
    "capabilities": _LIST,
    "parts": _LIST, "transfer": _ITEMS, "analogy": _ITEMS, "abstraction": _ITEMS,
    "novelty": _TEXT,
    # memory and self
    "episode": _ITEMS, "consolidate": _TEXT, "agents": _ITEMS, "capability": _TEXT,
    "success": _NUMBER, "strategy": _TEXT, "modalities": _ITEMS,
}

_READERS: Dict[str, Callable[[str], Any]] = {
    _TEXT: lambda raw: " ".join(raw.split()),
    _NUMBER: lambda raw: float(raw.strip()),
    _LIST: lambda raw: [" ".join(p.split()) for p in raw.split(";;") if p.strip()],
    _ITEMS: lambda raw: [[" ".join(f.split()) for f in item.split("|")]
                         for item in raw.split(";;") if item.strip()],
    _PAIRS: lambda raw: [[" ".join(f.split()) for f in item.split("??")]
                         for item in raw.split(";;") if item.strip()],
    _MAP: lambda raw: {" ".join(k.split()): float(v)
                       for k, _, v in (p.partition("=") for p in raw.split(";") if p.strip())},
}


# --------------------------------------------------------------------------- #
# Reading the .u files
# --------------------------------------------------------------------------- #
def parse_file(path: str) -> Iterator[Dict[str, Any]]:
    base = os.path.basename(path)
    record: Optional[Dict[str, Any]] = None
    key = ""
    with open(path, "rt", encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            line = raw.split(" #", 1)[0].rstrip() if " #" in raw else raw.rstrip()
            if line.lstrip().startswith("#") or not line.strip():
                continue
            where = f"{base}:{number}"
            if line.startswith("@record"):
                if record is not None:
                    yield _finish(record, where)
                parts = line.split()
                if len(parts) != 3:
                    raise UnifiedError(f"{where}: expected '@record <category> <id>'")
                category, rid = parts[1], parts[2]
                if category not in CATEGORIES:
                    raise UnifiedError(f"{where}: unknown category {category!r}. "
                                       f"Add it to CATEGORIES with the organ that reads it.")
                record, key = {"id": rid, "category": category, "where": where}, ""
                continue
            if record is None:
                raise UnifiedError(f"{where}: a field before any @record")
            if line.startswith((" ", "\t")) and key:
                record["_raw"][key] += " " + line.strip()      # a continued value
                continue
            name, sep, value = line.partition("=")
            if not sep:
                raise UnifiedError(f"{where}: not 'key = value': {line.strip()!r}")
            key = name.strip()
            if key not in _FIELDS:
                raise UnifiedError(f"{where}: unknown field {key!r}. "
                                   f"Add it to _FIELDS with how it should be read.")
            record.setdefault("_raw", {})
            if key in record["_raw"]:
                raise UnifiedError(f"{where}: {key!r} given twice")
            record["_raw"][key] = value.strip()
    if record is not None:
        yield _finish(record, f"{base}:end")


def _derive_inheritance(out: Dict[str, Any]) -> None:
    """Give a concept record the ``is_a`` path its own generalisation needs to be answerable.

    A record that says four birds have feathers and then asks about a kiwi is unanswerable unless
    something connects the kiwi to the kind and the kind to the property. Measured before this
    existed: the generalization probe asked four questions and she answered none, because nothing
    in the corpus had ever mentioned a kiwi. That is the corpus's gap, not hers.

    Derived rather than typed, so the claims cannot drift from the examples they come from:
    ``<concept> has_property <invariant>`` once per invariant, and ``<member> is_a <concept>`` for
    every example and every generalisation target.
    """
    concept = str(out.get("concept") or "")
    if not concept:
        return
    claims: List[List[str]] = list(out.get("knowledge") or [])
    seen = {tuple(c[:3]) for c in claims if len(c) >= 3}

    def add(subject: str, predicate: str, obj: str) -> None:
        row = (subject, predicate, obj)
        if subject and obj and subject != obj and row not in seen:
            seen.add(row)
            claims.append([subject, predicate, obj])

    for invariant in out.get("invariants") or []:
        add(concept, "has_property", str(invariant))
    for capability in out.get("capabilities") or []:
        add(concept, "capable_of", str(capability))
    for item in out.get("examples") or []:
        if item:
            add(str(item[0]), "is_a", concept)
    for pair in out.get("generalize") or []:
        if pair:
            add(str(pair[0]), "is_a", concept)
    if claims:
        out["knowledge"] = claims


def _finish(record: Dict[str, Any], where: str) -> Dict[str, Any]:
    """Type every field, check the category's required blocks are all non-empty, and clean up."""
    raw = record.pop("_raw", {})
    out: Dict[str, Any] = {"id": record["id"], "category": record["category"]}
    for key, text in raw.items():
        try:
            out[key] = _READERS[_FIELDS[key]](text)
        except (ValueError, TypeError) as exc:
            raise UnifiedError(f"{record['where']}: {key!r} is not a "
                               f"{_FIELDS[key]}: {text!r} ({exc})") from None
    _organ, required = CATEGORIES[record["category"]]
    # ``not out.get(key)`` was the first version and it was wrong in one place that matters: a
    # measured capability of 0.0 and a reward of 0 are *values*, and it read them as absent. The
    # self-model record for causal prediction after a bulk load is exactly that case — the number
    # is zero because the capability collapsed, which is the whole point of keeping it.
    empty = [key for key in required
             if key not in out or out[key] in ("", [], {}, None)]
    if empty:
        raise UnifiedError(f"{record['where']}: a {record['category']!r} record must carry "
                           f"{', '.join(required)} — missing or empty: {', '.join(empty)}")
    if record["category"] in ("concept_formation", "generalization"):
        # Either-or rather than a required field: a kind is defined by what its members *are*, by
        # what they *can do*, or by both, and demanding `invariants` forced a capability to be
        # written as a property — which is the mislabelling that made the whole generalisation
        # row unaskable, because no English form asks "is a seal breathes with lungs".
        if not (out.get("invariants") or out.get("capabilities")):
            raise UnifiedError(f"{record['where']}: a {record['category']!r} record must carry "
                               f"invariants or capabilities — it carries neither")
        _derive_inheritance(out)
    out.setdefault("domain", "general")
    return out


def records(directory: str = _U_DIR, *,
            wanted: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    if not os.path.isdir(directory):
        raise UnifiedError(f"no such directory: {directory}")
    out: List[Dict[str, Any]] = []
    seen: Dict[str, str] = {}
    for name in sorted(n for n in os.listdir(directory) if n.endswith(".u")):
        for record in parse_file(os.path.join(directory, name)):
            if record["id"] in seen:
                raise UnifiedError(f"duplicate record id {record['id']!r} "
                                   f"(already in {seen[record['id']]})")
            seen[record["id"]] = name
            out.append(record)
    if wanted:
        chosen = {str(c) for c in wanted}
        unknown = chosen - set(CATEGORIES)
        if unknown:
            raise UnifiedError(f"unknown category: {', '.join(sorted(unknown))}")
        out = [r for r in out if r["category"] in chosen]
    return out


# --------------------------------------------------------------------------- #
# Folding the other two corpora in
# --------------------------------------------------------------------------- #
def from_knowledge() -> Iterator[Dict[str, Any]]:
    """The ``.kb`` facts, one record per subject, with its QA beside it.

    Per subject rather than per triple: a record is a *thing she is being told about*, and forty
    thousand one-line records would make the unified file a triple store with extra keys.
    """
    import prepare_knowledge_corpus as kc

    rows = kc.parse()
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["subject"], []).append(row)
    qa_by_subject: Dict[str, List[List[str]]] = {}
    grounder_pairs = list(kc.qa_pairs(rows))
    subjects = sorted(grouped)
    lookup = {kc._norm(s): s for s in subjects}
    for pair in grounder_pairs:
        for subject in subjects:
            if subject in pair["answer"].lower():
                qa_by_subject.setdefault(subject, []).append([pair["question"], pair["answer"]])
                break
    for subject in subjects:
        triples = grouped[subject]
        yield {"id": f"kb.{subject.replace(' ', '_')}",
               "category": "world_knowledge",
               "domain": triples[0]["domain"],
               "knowledge": [[r["subject"], r["predicate"], r["object"]] for r in triples],
               "qa": qa_by_subject.get(subject, [])[:4],
               "text": " ".join(q[1] for q in qa_by_subject.get(subject, [])[:3])}
    del lookup


def from_experience() -> Iterator[Dict[str, Any]]:
    """The ``.exp`` episodes, unchanged in meaning, rewritten into the unified blocks."""
    import prepare_experience_corpus as pe

    for episode in pe.episodes(pe.scenarios()):
        cause, effect = episode["correction"]["cause"], episode["correction"]["effect"]
        yield {"id": f"exp.{episode['scenario']}.{episode['step']}",
               "category": "experience",
               "domain": episode["domain"],
               "context": episode["action"]["text"],
               "observation": episode["observation"],
               "order": list(episode["order"]),
               "state": list(episode["state_facts"]),
               "action": episode["action"]["action"],
               "actor": episode["action"]["actor"],
               "object": episode["action"]["object"],
               "prediction": [[episode["prediction"]["key"],
                               str(episode["prediction"]["expected"]),
                               str(episode["prediction"]["confidence"])]],
               "result": str(episode["observation"][effect]),
               "diagnosis": "world_model",
               "law": [[cause, "causes", effect, str(episode["correction"]["sign"])]],
               "counterfactual": [[episode["counterfactual"]["variable"],
                                   str(episode["counterfactual"]["value"]),
                                   episode["counterfactual"]["effect"],
                                   str(episode["counterfactual"]["direction"])]],
               "text": episode["text"]}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build(*, directory: str = _U_DIR, wanted: Optional[Sequence[str]] = None,
          include_knowledge: bool = True,
          include_experience: bool = True) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if include_knowledge and (not wanted or "world_knowledge" in wanted):
        out.extend(from_knowledge())
    if include_experience and (not wanted or "experience" in wanted):
        out.extend(from_experience())
    out.extend(records(directory, wanted=wanted))
    return out


def _write(rows: Iterable[Dict[str, Any]], path: str) -> int:
    written = 0
    if path in ("-", "/dev/stdout"):
        for row in rows:
            sys.stdout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
        return written
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    if path.endswith(".gz"):
        handle: Any = io.TextIOWrapper(
            gzip.GzipFile(path, "wb", compresslevel=9, mtime=0), encoding="utf-8")
    else:
        handle = open(path, "wt", encoding="utf-8")
    with handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=_U_DIR)
    ap.add_argument("--category", action="append", default=None, metavar="NAME")
    ap.add_argument("--out", metavar="PATH")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-knowledge", action="store_true", help="leave the .kb facts out")
    ap.add_argument("--no-experience", action="store_true", help="leave the .exp episodes out")
    args = ap.parse_args(argv)

    try:
        rows = build(directory=args.dir, wanted=args.category,
                     include_knowledge=not args.no_knowledge,
                     include_experience=not args.no_experience)
    except UnifiedError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    by_category: Dict[str, int] = {}
    by_domain: Dict[str, int] = {}
    blocks: Dict[str, int] = {}
    for row in rows:
        by_category[row["category"]] = by_category.get(row["category"], 0) + 1
        by_domain[row.get("domain", "general")] = by_domain.get(row.get("domain", "general"), 0) + 1
        for key in row:
            if key not in ("id", "category", "domain"):
                blocks[key] = blocks.get(key, 0) + 1
    covered = set(by_category)
    report: Dict[str, Any] = {
        "records": len(rows),
        "categories_covered": len(covered),
        "categories_total": len(CATEGORIES),
        "empty_categories": sorted(set(CATEGORIES) - covered),
        "by_category": dict(sorted(by_category.items())),
        "domains": len(by_domain),
        "blocks": dict(sorted(blocks.items(), key=lambda kv: (-kv[1], kv[0]))),
    }
    if not args.check:
        if not args.out:
            sys.stderr.write("nothing to do: pass --out PATH, or --check\n")
            return 2
        report["written"] = _write(rows, args.out)
    print(json.dumps(report, indent=1, ensure_ascii=False))
    if args.out and args.out not in ("-", "/dev/stdout"):
        print(f"\nuse it:  python -m nyxara.njp.unified --records {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
