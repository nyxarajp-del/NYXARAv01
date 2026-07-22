"""Tests for the Internal Opponent + the Hegelian Dialectic (P24 · F19, organs 7 & 13)."""

from nyxara.mind.inner_critic import Dialectic, InnerCritic

_BAD = ("A game changer that basically improves stuff somehow. "
        "A game changer that basically improves stuff somehow.")
_GOOD = ("A membrane valve routes heat through a copper lattice because the "
         "lattice stores charge; the stored charge drives the pump.")


def test_planted_flaws_are_caught():
    critic = InnerCritic()
    rep = critic.attack(_BAD, "idea")
    kinds = {o.kind for o in rep.objections}
    assert not rep.survived
    assert {"cliche", "vagueness", "redundancy", "unsupported"} <= kinds


def test_clean_mechanised_content_survives():
    critic = InnerCritic()
    rep = critic.attack(_GOOD, "idea")
    assert rep.survived
    assert rep.total_severity < critic.threshold


def test_contradiction_needs_resolution():
    critic = InnerCritic()
    raw = "It is always silent and never loud, always everything and nothing."
    resolved = ("It is always silent yet never loud — the tension between "
                "everything and nothing resolves in balance.")
    assert any(o.kind == "contradiction" for o in critic.attack(raw, "idea").objections)
    assert not any(o.kind == "contradiction"
                   for o in critic.attack(resolved, "idea").objections)


def test_degenerate_art_rejected_rich_art_released():
    critic = InnerCritic()
    flat = '<svg><circle cx="1" cy="1" r="0"/></svg>'
    rich = ('<svg><polyline points="100,100 180,220 300,90 210,40" '
            'stroke="hsl(10,70%,50%)"/><polyline points="120,300 200,260 310,330" '
            'stroke="hsl(190,70%,50%)"/></svg>')
    assert not critic.attack(flat, "art").survived
    assert critic.attack(rich, "art").survived


def test_svg_markup_repetition_is_not_redundancy():
    critic = InnerCritic()
    svg = "<svg>" + "".join(f'<rect x="{i}" y="{i}" width="2" height="2" '
                            f'fill="hsl({i * 10},60%,50%)"/>' for i in range(1, 30)) + "</svg>"
    assert not any(o.kind == "redundancy" for o in critic.attack(svg, "art").objections)


def test_attack_is_deterministic_and_counted():
    critic = InnerCritic()
    a = critic.attack(_BAD, "idea").to_dict()
    b = critic.attack(_BAD, "idea").to_dict()
    assert a == b
    assert critic.attacks == 2 and critic.rejections == 2


def test_dialectic_builds_real_antithesis_and_synthesis():
    d = Dialectic()
    triad = d.run("The open lattice binds light and lets memory rise.")
    assert "closed" in triad["antithesis"]
    assert "dark" in triad["antithesis"] and "fall" in triad["antithesis"]
    assert "resolves" in triad["synthesis"]
    # the synthesis carries both poles and passes the critic's contradiction probe
    assert not any(o.kind == "contradiction"
                   for o in InnerCritic().attack(triad["synthesis"], "idea").objections)


def test_persistence_round_trip():
    critic = InnerCritic(threshold=0.5)
    critic.attack(_BAD, "idea")
    clone = InnerCritic.from_dict(critic.to_dict())
    assert clone.threshold == critic.threshold
    assert clone.attacks == critic.attacks and clone.rejections == critic.rejections
