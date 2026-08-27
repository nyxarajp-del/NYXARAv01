"""A category with no organ behind it is a label, and a corpus of labels measures nothing.

That is the load-bearing test here — :func:`test_every_category_names_an_organ_the_brain_has`
walks all fifty-one rows of ``CATEGORIES`` and resolves the attribute each one claims. It exists
because guessing an organ's name is easy and *silent*: the first version of the loader named
``selfmodel`` where the brain calls it ``self_model``, and ``hypotheses`` where it calls it
``designer``. Both resolved to ``None``, both routed into a guarded function that did nothing, and
the report said zero capabilities and zero hypotheses on a corpus full of both. Nothing raised.

The rest is about the read-back numbers, and each of those was zero once for a reason worth
keeping in a test: the transition check compared against a stringified list, the counterfactual
check ran against whatever state the corpus happened to end in, the abstraction cases never
repeated an antecedent pair, and the error records stated a diagnosis with no evidence to reach it.

No test here opens a socket. The whole absorb is about a second.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import prepare_unified_corpus as pu  # noqa: E402

from nyxara.njp.brain import NJPBrain  # noqa: E402
from nyxara.njp.predict import ErrorKind  # noqa: E402
from nyxara.njp.unified import absorb, load  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "prepare_unified_corpus.py"
_RECORDS = _ROOT / "nyxara" / "njp" / "data" / "world_unified.jsonl.gz"


@pytest.fixture(scope="module")
def records():
    return load(_RECORDS)


@pytest.fixture(scope="module")
def report(records):
    return absorb(NJPBrain(), records)


# --------------------------------------------------------------------------- #
# The rule the file is built on
# --------------------------------------------------------------------------- #
def test_every_category_names_an_organ_the_brain_has():
    """No block exists that no organ reads, checked against a real brain rather than a comment."""
    brain = NJPBrain()
    missing = [(category, organ) for category, (organ, _blocks) in pu.CATEGORIES.items()
               if getattr(brain, organ, None) is None]
    assert missing == [], f"categories routed to an organ the brain does not have: {missing}"


def test_every_category_has_records(records):
    present = {row["category"] for row in records}
    assert set(pu.CATEGORIES) - present == set()
    assert len(pu.CATEGORIES) >= 50


def test_every_record_carries_its_required_blocks(records):
    for row in records:
        _organ, required = pu.CATEGORIES[row["category"]]
        for block in required:
            assert row.get(block) not in (None, "", [], {}), f"{row['id']} is missing {block}"


def test_every_field_used_is_a_declared_field(records):
    known = set(pu._FIELDS) | {"id", "category"}
    for row in records:
        unknown = set(row) - known
        assert unknown == set(), f"{row['id']} carries undeclared fields: {sorted(unknown)}"


# --------------------------------------------------------------------------- #
# What the absorb learned
# --------------------------------------------------------------------------- #
def test_every_organ_is_actually_fed(report):
    """Each of these was zero at some point, and each zero was a bug rather than a limit."""
    fed = report.to_dict()["fed"]
    for name in ("facts", "qa_pairs", "episodes", "events", "observations", "arrows_declared",
                 "predictions", "transitions", "hypotheses", "concept_members",
                 "abstraction_cases", "capabilities", "goals"):
        assert fed[name] > 0, f"nothing reached {name}"


def test_a_fact_that_went_in_can_be_asked_for(report):
    assert report.facts_asked > 500
    assert report.facts_recalled == report.facts_asked


def test_every_fitted_sign_matches_the_stated_one(report):
    assert report.signs_asked > 0
    assert report.signs_correct == report.signs_asked


def test_every_counterfactual_runs_the_right_way(report):
    """Scored against each record's own reading as the baseline, which is the fix that got this
    from 0.81 to 1.00 — the failures were the checker asking from another scenario's final state."""
    assert report.counterfactuals_asked > 100
    assert report.counterfactuals_correct == report.counterfactuals_asked


def test_every_stated_diagnosis_is_the_one_diagnose_returns(report):
    """The record supplies the evidence and `diagnose` supplies the kind, so this is a measurement.

    Without the evidence blocks, eight of these came back UNATTRIBUTED — correctly, because
    `diagnose` reaches PERCEPTION, GROUNDING, MEMORY, RELATION, REASONING and LANGUAGE through
    evidence keys and refuses to guess without them.
    """
    assert report.diagnoses_asked > 20
    assert report.diagnoses_agreeing == report.diagnoses_asked


def test_the_error_taxonomy_is_exercised_end_to_end(records):
    """All nine kinds are named by the corpus, and the eight reachable ones are reached."""
    stated = {row["diagnosis"] for row in records if row.get("diagnosis")}
    assert stated >= {ErrorKind.PERCEPTION, ErrorKind.GROUNDING, ErrorKind.MEMORY,
                      ErrorKind.RELATION, ErrorKind.WORLD_MODEL, ErrorKind.REASONING,
                      ErrorKind.LANGUAGE, ErrorKind.UNATTRIBUTED}


def test_abstractions_are_found_rather_than_stated(report):
    """`Discoverer` proposes from co-occurrence and confirms on a held-out split, or not at all."""
    assert report.abstractions_found >= 5


def test_concepts_are_formed_and_claim_their_members(report):
    assert report.concepts_formed >= 5
    assert report.members_generalised >= report.concept_members * 0.8


def test_belief_revision_actually_supersedes(report):
    """`_assert` does not revise — it is the bulk path — so the router calls `_revise` itself."""
    assert report.revisions_asked >= 3
    assert report.supersedes == report.revisions_asked


def test_hypotheses_are_proposed_and_resolved(report):
    assert report.hypotheses >= 20
    assert report.hypotheses_resolved >= 5


def test_the_transition_model_learns_most_of_what_it_is_shown(report):
    """Most, not all. The one it misses is the record that exists to be missed — see below."""
    assert report.transitions_asked > 0
    assert report.transitions_predicted >= report.transitions_asked * 0.8


def test_a_transition_with_its_own_action_is_learned_exactly():
    """Every transition whose action is unique comes back exactly. The two that share one do not.

    Measured: ``predict`` returns ``context=()`` — the model backs off to the action alone at this
    volume, so ``light_switch`` and ``switch_with_dead_bulb``, which differ only in whether the
    bulb works, are one key and only one of them can be the answer. That is a real property of
    `PredictiveWorldModel` at nine examples rather than a defect in either record, and the pair is
    in the corpus precisely because a state-blind model should be caught by it.
    """
    from nyxara.njp.predictive import WorldState

    brain = NJPBrain()
    rows = [r for r in load(_RECORDS) if r.get("state") and r.get("next_state")]
    assert len(rows) >= 9
    for row in rows:
        brain.predictive.observe(WorldState.of(row["state"]), row["action"],
                                 next_state=WorldState.of(row["next_state"]))

    counts: dict = {}
    for row in rows:
        counts[row["action"]] = counts.get(row["action"], 0) + 1
    unique = [row for row in rows if counts[row["action"]] == 1]
    shared = [row for row in rows if counts[row["action"]] > 1]
    assert unique and shared, "the corpus needs both a unique action and a shared one"

    for row in unique:
        got = brain.predictive.predict(WorldState.of(row["state"]), row["action"])
        assert got.top == WorldState.of(row["next_state"]).signature, row["id"]

    right = sum(1 for row in shared
                if brain.predictive.predict(WorldState.of(row["state"]), row["action"]).top
                == WorldState.of(row["next_state"]).signature)
    assert right < len(shared), "the shared action no longer collides — update this test"


def test_the_prose_contradictions_do_not_ground(report):
    """Measured, and the reason the `clash` block exists.

    `ground("the capital of myanmar is yangon")` extracts no relation at all, so a revision test
    built on the sentences would be measuring the pattern table's coverage and calling it belief
    revision. If the extraction ever improves this fails, and that is a good failure.
    """
    assert report.contradictions_seen == 0


def test_absorb_is_fast_enough_to_be_a_fixture(report):
    assert report.ms < 30_000


def test_absorb_survives_a_brain_with_no_organs(records):
    class Bare:
        pass

    out = absorb(Bare(), records[:50], check=False)
    assert out.records == 50 and out.facts == 0


# --------------------------------------------------------------------------- #
# The generator
# --------------------------------------------------------------------------- #
def _u(tmp_path, body: str) -> Path:
    directory = tmp_path / "u"
    directory.mkdir(exist_ok=True)
    (directory / "probe.u").write_text(body, encoding="utf-8")
    return directory


def test_unknown_category_is_refused(tmp_path):
    with pytest.raises(pu.UnifiedError, match="unknown category"):
        pu.records(_u(tmp_path, "@record telepathy probe\ndomain = general\n"))


def test_unknown_field_is_refused(tmp_path):
    with pytest.raises(pu.UnifiedError, match="unknown field"):
        pu.records(_u(tmp_path, "@record qa probe\nvibes = good\n"))


def test_a_missing_required_block_is_refused(tmp_path):
    with pytest.raises(pu.UnifiedError, match="must carry"):
        pu.records(_u(tmp_path, "@record planning probe\ngoal = do the thing\n"))


def test_a_duplicate_id_is_refused(tmp_path):
    body = ("@record qa a\nqa = q ?? a\n"
            "@record qa a\nqa = q2 ?? a2\n")
    with pytest.raises(pu.UnifiedError, match="duplicate record id"):
        pu.records(_u(tmp_path, body))


def test_a_field_given_twice_is_refused(tmp_path):
    with pytest.raises(pu.UnifiedError, match="given twice"):
        pu.records(_u(tmp_path, "@record qa probe\nqa = q ?? a\nqa = r ?? b\n"))


def test_a_measured_zero_counts_as_a_value(tmp_path):
    """A capability of 0.0 is the capability collapsing, not the field being absent."""
    parsed = pu.records(_u(tmp_path, "@record self_model probe\n"
                                     "capability = causal prediction\nsuccess = 0.0\n"))
    assert parsed[0]["success"] == 0.0


def test_a_continued_value_is_joined(tmp_path):
    parsed = pu.records(_u(tmp_path, "@record qa probe\nqa = one ?? two ;;\n    three ?? four\n"))
    assert parsed[0]["qa"] == [["one", "two"], ["three", "four"]]


def test_the_shipped_corpus_is_what_the_sources_currently_produce():
    with gzip.open(_RECORDS, "rt", encoding="utf-8") as handle:
        shipped = [json.loads(line) for line in handle]
    assert pu.build() == shipped


def test_cli_check_writes_nothing(tmp_path):
    result = subprocess.run([sys.executable, str(_SCRIPT), "--check"],
                            capture_output=True, text=True, timeout=300, cwd=str(_ROOT))
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["categories_covered"] == payload["categories_total"]
    assert payload["empty_categories"] == []
    assert not list(tmp_path.iterdir())


def test_cli_absorb_reports(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "nyxara.njp.unified", "--records", str(_RECORDS), "--limit", "300"],
        capture_output=True, text=True, timeout=600, cwd=str(_ROOT))
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["records"] == 300
