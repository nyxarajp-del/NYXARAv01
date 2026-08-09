"""NYX V.02 · open-domain reasoning — a seat for every domain, each answer carrying its tier.

The measurement this file exists to fix: outside physics, chemistry, maths and logic,
`FirstPrinciplesEngine.derive` returned `None` and the turn produced *silence*. The other half
of the file pins the rule that comes with reaching further — `verifiable=True` only where a
step was genuinely checked, and the label never drops off an unverified answer.
"""

from __future__ import annotations

from nyxara.kernel.config import NyxConfig
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.reason import TIERS, Chain, OpenDomainReasoner

SHOUT = "apple -> APPLE!\nmango -> MANGO!\ncar -> ?"
ALIEN = "zorblat frobs the quux and quux frobs the blim, does zorblat frob the blim"


def _reasoner(**kw) -> OpenDomainReasoner:
    return OpenDomainReasoner(**kw)


# --------------------------------------------------------------------------- #
# The verifiable tiers
# --------------------------------------------------------------------------- #
def test_an_exact_chain_is_checked():
    chain = _reasoner().solve("if x = 4 then what is x + 3")
    assert chain.answered and chain.tier == "exact"
    assert chain.verifiable and all(s.checked for s in chain.steps)
    assert "Derived and checked" in chain.label()


def test_a_certified_derivation_is_checked():
    chain = _reasoner().solve("balance Fe + O2 -> Fe2O3")
    assert chain.answered and chain.tier == "derived"
    assert chain.verifiable and chain.steps


def test_the_derived_tier_reads_the_derivations_own_certificate():
    """It does not assume the tier implies a proof — it reads `Derivation.verified`."""
    chain = _reasoner().solve("what is the escape velocity of earth")
    assert chain.tier == "derived"
    assert chain.verifiable == all(s.checked for s in chain.steps)


# --------------------------------------------------------------------------- #
# Past the four domains: an answer, and an honest label
# --------------------------------------------------------------------------- #
def test_a_field_with_no_prior_domain_is_modelled_not_ignored():
    """This is the gap: V.01 answered nothing at all here."""
    chain = _reasoner().solve(ALIEN)
    assert chain.answered
    assert chain.tier in ("modelled", "mapped")
    assert chain.verifiable is False


def test_an_unverified_answer_always_carries_its_label():
    chain = _reasoner().solve(ALIEN)
    assert "Plausible, not verified" in chain.label()
    assert chain.label() in chain.render()


def test_demonstrations_are_induced_rather_than_modelled():
    chain = _reasoner().solve(SHOUT)
    assert chain.tier == "induced" and "CAR!" in chain.answer


def test_two_demonstrations_are_fit_not_transfer():
    """Fit on the examples it was given is not evidence about the probe."""
    chain = _reasoner().solve(SHOUT)
    assert chain.verifiable is False
    assert "not transfer-tested" in chain.why


def test_three_demonstrations_can_earn_verified():
    chain = _reasoner().solve(
        "apple -> APPLE!\nmango -> MANGO!\nplum -> PLUM!\ncar -> ?")
    assert chain.tier == "induced" and chain.verifiable is True
    assert "held-out" in chain.why


# --------------------------------------------------------------------------- #
# The floor: her own graph, offered as association and nothing more
# --------------------------------------------------------------------------- #
def _brain(**kw) -> NyxBrain:
    return NyxBrain(NyxConfig(**kw))


def test_a_cold_graph_declines_rather_than_padding():
    """On day one there is nothing to associate, and saying so is the right answer."""
    brain = _brain()
    chain = brain.reason.solve("why does a heavier flywheel smooth an engine")
    if chain.answered:
        assert chain.tier != "associative"


def test_what_she_has_read_can_carry_an_associative_chain():
    brain = _brain()
    for line in ("a heavier flywheel stores more rotational inertia",
                 "rotational inertia resists a change in engine speed",
                 "a flywheel smooths the engine by resisting speed change"):
        brain.perceive(line)
    chain = brain.reason.solve("why does a heavier flywheel smooth an engine")
    assert chain.answered and chain.tier == "associative"
    assert chain.verifiable is False
    assert "not about the world" in chain.why


def test_the_associative_floor_can_be_switched_off():
    brain = _brain(reason_use_associative=False)
    assert brain.reason.available()["associative"] is False


# --------------------------------------------------------------------------- #
# Abstention is a result
# --------------------------------------------------------------------------- #
def test_when_no_tier_answers_she_says_so_and_names_what_she_tried():
    chain = _reasoner(use_generalization=False).solve("qqqq wwww")
    assert not chain.answered
    assert chain.tried and "no tier could answer this" in chain.why
    assert "not going to produce a paragraph" in chain.label()


def test_an_empty_question_is_not_an_error():
    assert _reasoner().solve("").answered is False
    assert _reasoner().solve(None).answered is False


# --------------------------------------------------------------------------- #
# Through the brain
# --------------------------------------------------------------------------- #
def test_reason_about_is_reachable_on_the_brain():
    chain = _brain().reason_about("if x = 4 then what is x + 3")
    assert chain is not None and chain.verifiable


def test_the_reason_specialist_is_seated():
    assert "reason" in [s.name for s in _brain().workspace.specialists]


def test_the_reason_specialist_never_claims_verified_without_a_checked_step():
    from nyxara.nyx.modules import ReasonSpecialist, Situation
    brain = _brain()
    bid = ReasonSpecialist().safe_consider(Situation(stimulus=ALIEN, brain=brain))
    assert bid is not None
    assert bid.verifiable is False
    assert bid.rationale.startswith(("modelled", "mapped"))


def test_a_verified_tier_reaches_the_specialist_as_verifiable():
    from nyxara.nyx.modules import ReasonSpecialist, Situation
    brain = _brain()
    bid = ReasonSpecialist().safe_consider(
        Situation(stimulus="if x = 4 then what is x + 3", brain=brain))
    assert bid is not None and bid.verifiable is True


def test_disabling_reason_leaves_v01_behaviour():
    brain = _brain(reason_enabled=False)
    assert brain.reason is None
    assert brain.reason_about("anything") is None


# --------------------------------------------------------------------------- #
# The two measured bugs in the machinery underneath
# --------------------------------------------------------------------------- #
def test_a_probe_row_is_the_query_not_a_demonstration():
    """`parse_demos` put "car -> ?" in the demo list, so induction was never reached."""
    from nyxara.mind.generalization import parse_demos
    demos, query = parse_demos(SHOUT)
    assert demos == [("apple", "APPLE!"), ("mango", "MANGO!")]
    assert query == "car"


def test_a_comma_inside_a_value_is_not_a_fragment_boundary():
    """Splitting on it turned "john smith -> Smith, John" into a demo mapping to "Smith"."""
    from nyxara.mind.generalization import parse_demos
    demos, query = parse_demos(
        "john smith -> Smith, John\nmary jones -> Jones, Mary\njon doe -> ?")
    assert demos == [("john smith", "Smith, John"), ("mary jones", "Jones, Mary")]
    assert query == "jon doe"


def test_the_name_reordering_task_now_induces_instead_of_inventing_a_domain():
    """Genesis used to read "jon" as a *relation* and answer with a theory of nothing."""
    from nyxara.mind.generalization import GeneralizationEngine
    res = GeneralizationEngine().generalize(
        "john smith -> Smith, John\nmary jones -> Jones, Mary\njon doe -> ?")
    assert res is not None and res.source == "skill_induction"
    assert "Doe, Jon" in res.answer


def test_domain_genesis_declines_on_a_demonstration_block():
    from nyxara.mind.generalization import GeneralizationEngine
    engine = GeneralizationEngine()
    assert engine._try_domain_genesis(SHOUT) is None
    # ...but is still available for a field that really has no prior domain
    assert engine._try_domain_genesis(ALIEN) is not None


def test_a_comma_separated_prompt_still_parses_as_before():
    from nyxara.mind.generalization import parse_demos
    demos, query = parse_demos("pattern: a -> A, b -> B. now: c")
    assert demos == [("a", "A"), ("b", "B")] and query == "c"


# --------------------------------------------------------------------------- #
# Fail-soft and honesty
# --------------------------------------------------------------------------- #
def test_stats_say_exactly_what_earns_verified():
    stats = _reasoner().stats()
    assert set(stats["tiers"]) == set(TIERS)
    assert "verifiable=True only for" in stats["note"]


def test_counters_track_verified_separately_from_answered():
    reasoner = _reasoner()
    reasoner.solve("if x = 4 then what is x + 3")
    reasoner.solve(ALIEN)
    stats = reasoner.stats()
    assert stats["answered"] == 2 and stats["verified"] == 1


def test_chain_serialises_with_its_label():
    got = Chain(question="q", answer="a", tier="modelled", engine="e").to_dict()
    assert got["label"].startswith("Plausible, not verified")
    assert got["verifiable"] is False
