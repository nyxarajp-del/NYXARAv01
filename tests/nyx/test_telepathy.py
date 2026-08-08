"""NYX V.02 · L-NEURAL-TELEPATHY — meaning as structure, and the honesty about what it is not.

The first assertion in this file is the one that matters most: nothing here reads a mind, and
the module says so. Everything after it is the part that *is* real — a frame that bypasses the
tokenizer, with the loss **measured** against the same content sent as prose rather than
asserted, and a shorthand minted only from repetitions she actually saw.
"""

from __future__ import annotations

from nyxara.kernel.config import NyxConfig
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.telepathy import Frame, SemanticStream

SPEC = Frame(concepts=["गुरुत्वाकर्षण", "flywheel", "torque"],
             roles={"agent": "engine", "patient": "flywheel"})
PROSE = "gravity ka flywheel par torque"
LONG = "run the full test suite and then commit only the docs"


def _stream(**kw) -> SemanticStream:
    kw.setdefault("mint_after", 3)
    return SemanticStream(NyxBrain(NyxConfig()), **kw)


# --------------------------------------------------------------------------- #
# The part that cannot be built, said out loud
# --------------------------------------------------------------------------- #
def test_the_module_states_there_is_no_mind_reading():
    note = _stream().stats()["note"]
    assert "NO brain-computer interface" in note
    assert "cannot be read by any software" in note


def test_describe_leads_with_what_it_cannot_do():
    said = _stream().describe()
    assert said.startswith("I cannot read your mind")


# --------------------------------------------------------------------------- #
# The vector channel — measured, not asserted
# --------------------------------------------------------------------------- #
def test_a_frame_bypasses_the_tokenizer_entirely():
    got = _stream().receive(SPEC)
    assert got.bypassed_tokenizer is True and got.loss == 0.0
    assert set(got.concepts) >= {"गुरुत्वाकर्षण", "flywheel", "torque", "engine"}


def test_prose_measurably_loses_what_a_frame_keeps():
    """The whole point, and it is a measurement: same content, two roads."""
    result = _stream().compare(PROSE, SPEC)
    assert result["frame_loss"] == 0.0
    assert result["prose_loss"] > 0.0


def test_the_loss_claim_is_scoped_to_past_the_frame():
    note = _stream().compare(PROSE, SPEC)["note"]
    assert "zero past the FRAME" in note or "zero past the frame" in note.lower()
    assert "not something software can measure" in note


def test_a_frame_actually_reaches_the_graph():
    stream = _stream()
    stream.receive(SPEC)
    assert "गुरुत्वाकर्षण" in stream.brain.graph
    assert stream.brain.graph.weight("गुरुत्वाकर्षण", "flywheel") > 0.0


def test_typed_role_bindings_become_relations():
    stream = _stream()
    stream.receive(SPEC)
    relations = stream.brain.semantics.relations("agent")
    assert any(r.b == "engine" for r in relations)


def test_an_empty_frame_carries_nothing_and_says_so():
    got = _stream().receive(Frame())
    assert got.concepts == [] and "carried nothing" in got.note


def test_a_frame_round_trips_through_a_dict():
    restored = Frame.from_dict(SPEC.to_dict())
    assert restored.content() == SPEC.content()


# --------------------------------------------------------------------------- #
# The streaming shape
# --------------------------------------------------------------------------- #
def test_frames_queue_and_drain_on_a_beat():
    stream = _stream()
    assert stream.push(SPEC) and stream.push({"concepts": ["engine"]})
    assert stream.pending == 2
    drained = stream.drain(limit=8)
    assert len(drained) == 2 and stream.pending == 0


def test_an_empty_frame_is_not_queued():
    assert _stream().push(Frame()) is False


def test_draining_is_budgeted():
    stream = _stream()
    for _ in range(5):
        stream.push(SPEC)
    assert len(stream.drain(limit=2)) == 2 and stream.pending == 3


class _Scrammed:
    def scrammed(self) -> bool:
        return True


def test_scram_stops_the_drain():
    stream = _stream()
    stream.push(SPEC)
    assert stream.drain(oversight=_Scrammed()) == []
    assert stream.pending == 1


def test_an_unreadable_oversight_is_treated_as_stop():
    class _Broken:
        @property
        def paused(self):
            raise RuntimeError("unreadable")
    assert SemanticStream._stopped(_Broken()) is True


# --------------------------------------------------------------------------- #
# Bidirectional — her state, as structure
# --------------------------------------------------------------------------- #
def test_she_emits_her_own_state_as_a_frame():
    stream = _stream()
    stream.brain.perceive("gravity pulls the apple down")
    frame = stream.emit()
    assert frame.kind == "state" and frame.concepts


def test_an_emitted_frame_can_be_received_by_another_node():
    """Machine-to-machine, this direction really is lossless."""
    sender = _stream()
    sender.brain.perceive("a heavier flywheel smooths an engine")
    frame = sender.emit()

    receiver = _stream()
    got = receiver.receive(frame)
    assert got.loss == 0.0
    assert set(got.concepts) == set(frame.content())


def test_an_emitted_frame_names_what_she_is_pursuing():
    stream = _stream()
    stream.brain.agenda.add("flywheel", kind="understand")
    frame = stream.emit()
    assert any(k.startswith("pursuing:") for k in frame.roles)


# --------------------------------------------------------------------------- #
# Intent compression — from repetition she actually saw
# --------------------------------------------------------------------------- #
def test_a_shorthand_is_minted_only_after_real_repetition():
    stream = _stream(mint_after=3)
    assert stream.observe(LONG) is None
    assert stream.observe(LONG) is None
    short = stream.observe(LONG)
    assert short is not None and short.trigger


def test_the_fourth_time_it_expands():
    stream = _stream(mint_after=3)
    for _ in range(3):
        stream.observe(LONG)
    trigger = next(iter({id(s): s for s in stream.shorthand.values()}.values())).trigger
    got = stream.expand(trigger)
    assert got is not None and got.concepts
    assert "learned from" in got.note and got.expanded_from == trigger


def test_the_full_text_still_expands_too():
    stream = _stream(mint_after=2)
    stream.observe(LONG)
    stream.observe(LONG)
    assert stream.expand(LONG) is not None


def test_an_unknown_phrase_expands_to_nothing_rather_than_guessing():
    assert _stream().expand("something she has never heard") is None


def test_something_already_short_is_not_compressed():
    assert _stream().observe("go") is None


def test_a_shorthand_carries_the_ordering_it_was_taught():
    stream = _stream(mint_after=2)
    stream.observe("fix the code but first run the tests")
    short = stream.observe("fix the code but first run the tests")
    assert short is not None
    assert "before" in short.frame.roles or "do" in short.frame.roles


def test_shorthands_are_capped():
    stream = _stream(mint_after=1, max_shorthand=2)
    for i in range(6):
        stream.observe(f"do the thing number {i} carefully and then stop")
    distinct = {id(s) for s in stream.shorthand.values()}
    assert len(distinct) <= 2


# --------------------------------------------------------------------------- #
# Through the brain, and across a restart
# --------------------------------------------------------------------------- #
def test_receive_and_emit_are_reachable_on_the_brain():
    brain = NyxBrain(NyxConfig())
    assert brain.receive_frame(SPEC).loss == 0.0
    assert brain.emit_frame() is not None


def test_learned_shorthand_survives_a_restart():
    brain = NyxBrain(NyxConfig(telepathy_mint_after=2))
    brain.telepathy.observe(LONG)
    short = brain.telepathy.observe(LONG)
    assert short is not None

    reborn = NyxBrain(NyxConfig(telepathy_mint_after=2))
    reborn.load_dict(brain.to_dict())
    got = reborn.telepathy.expand(short.trigger)
    assert got is not None and got.expanded_from == short.trigger


def test_disabling_the_layer_leaves_v01_behaviour():
    brain = NyxBrain(NyxConfig(telepathy_enabled=False))
    assert brain.telepathy is None
    assert brain.receive_frame(SPEC) is None and brain.emit_frame() is None


# --------------------------------------------------------------------------- #
# Fail-soft
# --------------------------------------------------------------------------- #
def test_everything_is_fail_soft_on_junk():
    stream = _stream()
    assert stream.receive(None).concepts == []
    assert stream.receive("not a frame").concepts == []
    assert stream.observe(None) is None
    assert stream.expand(None) is None
    assert stream.load_dict(None) is False
    assert stream.load_dict({"shorthand": ["junk", {"trigger": ""}]}) is True
    assert stream.shorthand == {}


def test_a_corrupt_frame_dict_becomes_an_empty_frame():
    assert Frame.from_dict("junk").empty
    assert Frame.from_dict({"concepts": None}).empty
