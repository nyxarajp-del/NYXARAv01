"""A corpus that claims to teach a causal model has to be marked against a law it did not supply.

Every other test of a dataset in this suite can only check shape: that the rows parse, that the
predicates fold, that nothing is dropped. This one can do better, because each scenario states its
own ground truth and the generator *runs* it — so "she learned the arrow" is a claim with an answer
key, and `test_every_arrow_is_learned_with_the_right_sign` is that check.

The sign is the load-bearing one. `replay` calls `universe.declare` **without** a sign, so nothing
in the pipeline ever tells the simulator which way the effect moves; if the fitted slope comes back
right on twenty-seven scenarios including eleven curved ones, the numbers taught it. A test that
let the sign be handed over would be measuring the loader.

`test_nothing_is_fitted_without_a_skeleton` pins the architectural fact the module's docstring is
built on, so a future change to `_permitted` that quietly makes the gate permissive shows up here
rather than as an unexplained jump in the numbers.

No test here opens a socket. The whole replay is ~11 ms.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import prepare_experience_corpus as pe  # noqa: E402

from nyxara.njp.brain import NJPBrain  # noqa: E402
from nyxara.njp.experience import (  # noqa: E402
    Episode, ReplayReport, load_episodes, replay,
)
from nyxara.njp.predict import ErrorKind  # noqa: E402
from nyxara.njp.universe import InternalUniverse  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "prepare_experience_corpus.py"
_EPISODES = _ROOT / "nyxara" / "njp" / "data" / "world_experience.jsonl.gz"


@pytest.fixture(scope="module")
def episodes():
    return load_episodes(_EPISODES)


@pytest.fixture(scope="module")
def report(episodes) -> ReplayReport:
    return replay(NJPBrain(), episodes)


# --------------------------------------------------------------------------- #
# What the replay learned
# --------------------------------------------------------------------------- #
def test_every_arrow_is_learned_with_the_right_sign(report):
    """The sign is never supplied, so a correct one can only have come from the readings."""
    wrong = [r.scenario for r in report.results if not r.sign_correct]
    assert wrong == [], f"fitted the wrong direction for: {', '.join(wrong)}"
    assert report.sign_accuracy == 1.0
    assert report.signs_unlearned == 0


def test_curved_laws_are_learned_too(report):
    """Eleven of the scenarios are not straight lines. A linear fit still has to get the way."""
    curved = [r for r in report.results if r.true_slope is None]
    assert len(curved) >= 10
    assert all(r.sign_correct for r in curved)


def test_the_fitted_coefficient_is_close_to_the_real_one(report):
    """Direction is the cheap half. This is the claim that the magnitude was learned as well."""
    assert report.linear_scenarios >= 10
    assert report.linear_slope_error is not None
    assert report.linear_slope_error < 0.10, report.linear_slope_error


def test_every_counterfactual_runs_the_right_way(report):
    """``do(cause = the far end)``, scored on direction only — a magnitude would score the fit."""
    wrong = [r.scenario for r in report.results
             if r.counterfactual_asked and not r.counterfactual_correct]
    assert wrong == [], f"counterfactual went the wrong way for: {', '.join(wrong)}"
    assert report.counterfactuals_answered == report.counterfactuals_asked


def test_the_predictions_are_supposed_to_miss(report):
    """A corpus whose priors are already right attributes nothing and corrects nothing."""
    assert report.scored == report.predictions == report.episodes
    assert report.missed > report.episodes * 0.5
    assert report.mean_error > 0.15


def test_every_miss_is_attributed_to_the_world_model(report):
    """`diagnose` branch 4, reached by the organ — not the UNATTRIBUTED fallback."""
    assert set(report.diagnosed) == {ErrorKind.WORLD_MODEL}
    assert report.diagnosed[ErrorKind.WORLD_MODEL] == report.scored


def test_the_discrete_transition_model_gets_fed(report):
    assert report.transitions > 0


def test_replay_is_fast_enough_to_be_a_fixture(report):
    assert report.ms < 5_000


# --------------------------------------------------------------------------- #
# The architectural fact the module is built on
# --------------------------------------------------------------------------- #
def test_a_universe_with_no_world_fits_from_numbers_alone(episodes):
    """The permissive fallback, measured: slope 1.162 against a stated 1.2."""
    universe = InternalUniverse()
    rows = [e for e in episodes if e.scenario == "plant_growth"]
    assert rows, "the plant_growth scenario has gone missing"
    for episode in rows:
        universe.observe(dict(episode.observation), order=list(episode.order))
    relation = universe.relations.get(("water", "growth"))
    assert relation is not None and relation.usable and relation.oriented
    assert relation.slope == pytest.approx(1.2, abs=0.15)


def test_nothing_is_fitted_without_a_skeleton(episodes):
    """With a real WorldView and no stated law, `_permitted` refuses every arrow.

    This is why each scenario states that its arrow exists before supplying a number. If a future
    change makes the gate permissive, this test fails and the module docstring needs rewriting —
    which is the point of pinning it.
    """
    brain = NJPBrain()
    for episode in [e for e in episodes if e.scenario == "plant_growth"]:
        brain.universe.observe(dict(episode.observation), order=list(episode.order))
    assert ("water", "growth") not in brain.universe.relations
    assert brain.universe._permitted("water", "growth") is False


def test_replay_without_declaring_learns_nothing(episodes):
    """The same run with `declare=False`, so the docstring's claim is checked and not asserted."""
    quiet = replay(NJPBrain(), episodes, declare=False)
    assert quiet.relations == 0
    assert quiet.signs_correct == 0
    assert quiet.signs_unlearned == quiet.scenarios
    # The prediction loop is untouched by the skeleton, and must still run.
    assert quiet.scored == quiet.episodes


# --------------------------------------------------------------------------- #
# The generator
# --------------------------------------------------------------------------- #
def _scenario(tmp_path, body: str) -> Path:
    directory = tmp_path / "exp"
    directory.mkdir(exist_ok=True)
    (directory / "probe.exp").write_text(body, encoding="utf-8")
    return directory


_GOOD = """@scenario probe
domain      = physics
actor       = student
action      = push
object      = cart
cause       = force | newtons | 1 2 3 4 5
effect      = speed | metres per second
law         = linear a=0.0 b=2.0
belief      = linear a=10.0 b=-1.0
sentence    = the student pushed the cart with {cause} newtons and it reached {effect} metres per second
"""


def test_a_scenario_that_believes_the_truth_is_refused(tmp_path):
    """An episode with no error attributes nothing and corrects nothing."""
    body = _GOOD.replace("belief      = linear a=10.0 b=-1.0", "belief      = linear a=0.0 b=2.0")
    with pytest.raises(pe.ScenarioError, match="believes what is true"):
        pe.scenarios(_scenario(tmp_path, body))


def test_a_flat_law_is_refused(tmp_path):
    body = _GOOD.replace("law         = linear a=0.0 b=2.0", "law         = linear a=5.0 b=0.0")
    with pytest.raises(pe.ScenarioError, match="no direction"):
        pe.scenarios(_scenario(tmp_path, body))


def test_a_missing_field_is_refused(tmp_path):
    body = "\n".join(line for line in _GOOD.splitlines() if not line.startswith("law "))
    with pytest.raises(pe.ScenarioError, match="missing"):
        pe.scenarios(_scenario(tmp_path, body + "\n"))


def test_an_unknown_law_shape_is_refused(tmp_path):
    body = _GOOD.replace("law         = linear a=0.0 b=2.0", "law         = cubic a=0.0 b=2.0")
    with pytest.raises(pe.ScenarioError, match="unknown law shape"):
        pe.scenarios(_scenario(tmp_path, body))


def test_too_few_readings_are_refused(tmp_path):
    body = _GOOD.replace("| 1 2 3 4 5", "| 1 2")
    with pytest.raises(pe.ScenarioError, match="at least three"):
        pe.scenarios(_scenario(tmp_path, body))


def test_direction_is_ordered_and_sign_over_is_not():
    """The bug the counterfactual scoring had: a range has one direction, a move has two."""
    rising = pe.Law("linear", 0.0, 2.0)
    assert rising.sign_over([1, 5]) == 1
    assert rising.direction(1, 5) == 1
    assert rising.direction(5, 1) == -1
    falling = pe.Law("inverse", 0.0, 100.0)
    assert falling.sign_over([1, 5]) == -1
    assert falling.direction(5, 1) == 1


def test_the_generator_is_deterministic():
    """Same scenarios, same noise, same bytes — or every rebuild is a diff."""
    once = list(pe.episodes(pe.scenarios()))
    twice = list(pe.episodes(pe.scenarios()))
    assert once == twice


def test_shipped_episodes_are_what_the_scenarios_currently_produce():
    with gzip.open(_EPISODES, "rt", encoding="utf-8") as handle:
        shipped = [json.loads(line) for line in handle]
    assert list(pe.episodes(pe.scenarios())) == shipped


def test_every_episode_carries_every_stage_of_the_loop(episodes):
    """Each field is the argument to one call. A missing one is a stage that never runs."""
    for episode in episodes:
        assert episode.state_facts
        assert episode.action.get("actor") and episode.action.get("action")
        assert episode.prediction.get("key") and episode.prediction.get("expected") is not None
        assert len(episode.observation) == 2
        assert episode.order and len(episode.order) == 2
        assert episode.error.get("absolute") is not None
        assert episode.correction.get("sign") in (-1, 1)
        assert episode.counterfactual.get("direction") in (-1, 0, 1)
        assert episode.text and episode.truth.get("law")


def test_the_observation_is_the_two_variables_the_order_names(episodes):
    for episode in episodes:
        assert set(episode.observation) == set(episode.order)
        assert episode.order[0] == episode.cause and episode.order[1] == episode.effect


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def test_a_malformed_line_costs_that_line_and_not_the_file(tmp_path):
    path = tmp_path / "episodes.jsonl"
    path.write_text('{"scenario": "a", "step": 0}\nnot json\n{"scenario": "b", "step": 1}\n',
                    encoding="utf-8")
    loaded = load_episodes(path)
    assert [e.scenario for e in loaded] == ["a", "b"]


def test_limit_stops_early(episodes):
    assert len(load_episodes(_EPISODES, limit=10)) == 10


def test_replay_survives_a_brain_with_no_organs(episodes):
    """Duck-typed on purpose: a brain missing an organ replays the rest instead of failing."""
    class Bare:
        pass

    out = replay(Bare(), episodes[:20])
    assert out.episodes == 20 and out.scored == 0


# --------------------------------------------------------------------------- #
# The CLIs
# --------------------------------------------------------------------------- #
def test_generator_check_writes_nothing(tmp_path):
    result = subprocess.run([sys.executable, str(_SCRIPT), "--check"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["scenarios"] >= 25 and payload["episodes"] >= 150
    assert payload["falling"] > 0 and payload["rising"] > 0
    assert not list(tmp_path.iterdir())


def test_generator_writes_episodes(tmp_path):
    out = tmp_path / "e.jsonl"
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--scenario", "plant_growth", "--out", str(out)],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["written"] == 7
    assert json.loads(out.read_text(encoding="utf-8").splitlines()[0])["scenario"] == "plant_growth"


def test_replay_cli_reports(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "nyxara.njp.experience", "--episodes", str(_EPISODES)],
        capture_output=True, text=True, timeout=300, cwd=str(_ROOT))
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["sign_accuracy"] == 1.0
    assert payload["counterfactuals_correct"] == payload["counterfactuals_asked"]
