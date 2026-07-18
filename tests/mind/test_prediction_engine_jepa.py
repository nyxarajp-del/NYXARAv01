"""The prediction engine consumes the JEPA surface: grounded queries + energy signal."""

from __future__ import annotations

from typing import Any

from nyxara.mind.prediction_engine import PredictionEngine


class _Pred:
    confidence = 0.7
    expected_reward = 0.2


class _JEPAStub:
    """Minimal grounded-JEPA double: encode_state + plausibility + predict."""

    def __len__(self) -> int:
        return 10

    def encode_state(self, state: Any):  # marks the model as grounded
        return (0.0, 0.0)

    def predict(self, state: Any, action: Any) -> _Pred:
        self.predicted_with = state
        return _Pred()

    def plausibility(self, state: Any, action: Any, next_state: Any) -> float:
        self.scored = (state, action, next_state)
        return 0.8


def test_grounded_model_gets_real_query_not_zero_probe():
    stub = _JEPAStub()
    pe = PredictionEngine(world_model=stub)
    pe.predict("will the deploy succeed?")
    assert stub.predicted_with == "will the deploy succeed?"


def test_jepa_energy_signal_joins_reasoning_chain():
    stub = _JEPAStub()
    pe = PredictionEngine(world_model=stub)
    r = pe.predict("the service recovers", context="the service crashed")
    assert stub.scored == ("the service crashed", "the_service_recovers",
                           "the service recovers")
    assert any("jepa-energy plausibility=0.80" in step for step in r.reasoning_chain)


def test_no_energy_signal_without_context_or_surface():
    stub = _JEPAStub()
    pe = PredictionEngine(world_model=stub)
    r = pe.predict("the service recovers")          # no context → no energy signal
    assert not any("jepa-energy" in step for step in r.reasoning_chain)

    class _Plain:
        def __len__(self) -> int:
            return 5

        def predict(self, state: Any, action: Any) -> _Pred:
            return _Pred()

    pe2 = PredictionEngine(world_model=_Plain())    # no plausibility → no signal
    r2 = pe2.predict("q", context="c")
    assert not any("jepa-energy" in step for step in r2.reasoning_chain)
