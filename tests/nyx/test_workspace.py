"""NYX V.01 · the conscious bottleneck — competition, the verifiable law, and learned attention."""

from __future__ import annotations

from nyxara.nyx.metacog import RecursiveMetaCognition
from nyxara.nyx.modules import Proposal, Situation, SpecialistModule
from nyxara.nyx.workspace import NyxWorkspace


class _Fixed(SpecialistModule):
    """A specialist that always bids the same thing — lets a test control the competition."""

    def __init__(self, name: str, *, confidence: float, priority: float = 0.5,
                 verifiable: bool = False, tags=frozenset(), silent: bool = False,
                 explode: bool = False, unavailable: bool = False) -> None:
        self.name = name
        self.priority = priority
        self.tags = tags or frozenset({name})
        self._confidence = confidence
        self._verifiable = verifiable
        self._silent = silent
        self._explode = explode
        self._unavailable = unavailable

    def available(self) -> bool:
        return not self._unavailable

    def consider(self, situation: Situation):
        if self._explode:
            raise RuntimeError("specialist on fire")
        if self._silent:
            return None
        return Proposal(source=self.name, content=f"{self.name} content",
                        confidence=self._confidence, verifiable=self._verifiable,
                        tags=self.tags)


def _sit(text: str = "a question") -> Situation:
    return Situation(stimulus=text, concepts=["question"])


# -------------------- the competition -------------------- #
def test_the_stronger_bid_reaches_consciousness():
    ws = NyxWorkspace(specialists=[_Fixed("weak", confidence=0.1),
                                   _Fixed("strong", confidence=0.95)])
    got = ws.deliberate(_sit())
    assert got.winner is not None and got.winner.source == "strong"
    assert got.starved is False


def test_every_bid_is_recorded_even_though_one_wins():
    """Losing is remembered — that is what makes `/nyx why` and calibration possible."""
    ws = NyxWorkspace(specialists=[_Fixed("a", confidence=0.9), _Fixed("b", confidence=0.2)])
    got = ws.deliberate(_sit())
    assert set(got.sources) == {"a", "b"}
    assert len(got.proposals) == 2


def test_a_verified_derivation_beats_a_more_confident_guess():
    """The repo's own law, enforced at the bottleneck: verifiable beats probabilistic."""
    ws = NyxWorkspace(specialists=[
        _Fixed("guesser", confidence=0.99, priority=0.5),
        _Fixed("derivation", confidence=0.6, priority=0.5, verifiable=True),
    ])
    got = ws.deliberate(_sit())
    assert got.winner.source == "derivation"


def test_silence_is_a_legitimate_bid():
    ws = NyxWorkspace(specialists=[_Fixed("quiet", confidence=0.9, silent=True),
                                   _Fixed("loud", confidence=0.3)])
    got = ws.deliberate(_sit())
    assert got.sources == ["loud"]


def test_nobody_bidding_is_starvation_not_a_crash():
    ws = NyxWorkspace(specialists=[_Fixed("quiet", confidence=0.9, silent=True)])
    got = ws.deliberate(_sit())
    assert got.starved is True and got.winner is None


def test_an_unavailable_specialist_is_skipped_and_reported():
    ws = NyxWorkspace(specialists=[_Fixed("gone", confidence=0.9, unavailable=True),
                                   _Fixed("here", confidence=0.4)])
    got = ws.deliberate(_sit())
    assert got.sources == ["here"]
    assert "gone" in ws.stats()["unavailable"]


def test_one_exploding_specialist_does_not_break_the_workspace():
    ws = NyxWorkspace(specialists=[_Fixed("bomb", confidence=0.9, explode=True),
                                   _Fixed("fine", confidence=0.4)])
    got = ws.deliberate(_sit())
    assert got.winner is not None and got.winner.source == "fine"


# -------------------- learned attention -------------------- #
def test_a_demoted_specialist_loses_access_to_consciousness():
    """The feedback loop that closes: being wrong repeatedly costs you the bottleneck."""
    metacog = RecursiveMetaCognition(min_samples=3)
    specialists = [_Fixed("loud", confidence=0.9, priority=0.6),
                   _Fixed("quiet", confidence=0.55, priority=0.6)]

    fresh = NyxWorkspace(specialists=specialists, metacog=metacog)
    assert fresh.deliberate(_sit()).winner.source == "loud"

    for i in range(8):                            # "loud" turns out to be confidently wrong
        metacog.observe_cycle(cycle_id=f"c{i}",
                              winner=Proposal(source="loud", content="x", confidence=0.9))
        metacog.observe_outcome(cycle_id=f"c{i}", correct=0.0)
    for i in range(8):                            # "quiet" turns out to be right
        metacog.observe_cycle(cycle_id=f"q{i}",
                              winner=Proposal(source="quiet", content="y", confidence=0.55))
        metacog.observe_outcome(cycle_id=f"q{i}", correct=1.0)

    learned = NyxWorkspace(specialists=specialists, metacog=metacog)
    assert learned.deliberate(_sit()).winner.source == "quiet"


def test_priority_stays_bounded_even_for_a_perfect_record():
    metacog = RecursiveMetaCognition(min_samples=1)
    for i in range(30):
        metacog.observe_cycle(cycle_id=f"c{i}",
                              winner=Proposal(source="derivation", content="x", confidence=1.0))
        metacog.observe_outcome(cycle_id=f"c{i}", correct=1.0)
    ws = NyxWorkspace(specialists=[_Fixed("derivation", confidence=1.0, verifiable=True)],
                      metacog=metacog)
    assert ws._priority(Proposal(source="derivation", content="x", confidence=1.0,
                                 verifiable=True)) <= 2.0


# -------------------- introspection -------------------- #
def test_stats_lists_the_cast_and_counts_cycles():
    ws = NyxWorkspace(specialists=[_Fixed("a", confidence=0.5)])
    ws.deliberate(_sit())
    ws.deliberate(_sit())
    stats = ws.stats()
    assert stats["cycles"] == 2 and stats["specialists"] == ["a"]


def test_default_cast_is_wired_and_bids_on_a_real_question():
    ws = NyxWorkspace()
    got = ws.deliberate(_sit("prove that (x+1)^2 = x^2 + 2x + 1"))
    # derivation needs sympy; without it the cast simply stays quieter — both are valid
    assert got.winner is None or got.winner.source in {
        "derivation", "graph", "memory", "creative", "ethics"}


def test_a_faculty_is_not_suppressed_for_having_been_right_recently():
    """IOR must damp repeating the same *content*, not whichever faculty last won.

    With a fixed cast bidding every cycle, keying inhibition on ``source:tags`` rotated the
    winner for reasons unrelated to the question — the same query answered worse the second
    time. The content fingerprint restores the mechanism's actual purpose.
    """
    ws = NyxWorkspace(specialists=[_Fixed("strong", confidence=0.9, priority=0.7),
                                   _Fixed("weak", confidence=0.3, priority=0.3)])
    winners = [ws.deliberate(_sit()).winner.source for _ in range(6)]
    assert winners.count("strong") >= 5


def test_the_content_fingerprint_is_stable_for_the_same_text():
    ws = NyxWorkspace(specialists=[])
    assert ws._fingerprint("hello there") == ws._fingerprint("  Hello There  ")
    assert ws._fingerprint("a") != ws._fingerprint("b")
