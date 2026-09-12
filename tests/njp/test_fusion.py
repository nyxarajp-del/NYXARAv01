"""The same shape in two subjects that never met (NJP V.40)."""
from __future__ import annotations


from nyxara.njp.explain import Explainer
from nyxara.njp.fusion import MIN_EDGES, Fusion


class T:
    __slots__ = ("object", "confidence", "superseded")

    def __init__(self, obj):
        self.object, self.confidence, self.superseded = obj, 1.0, False


class Store:
    def __init__(self, rows):
        self.facts = {}
        for a, r, b in rows:
            self.facts.setdefault((a.lower(), r), []).append(T(b))

    def _key(self, text):
        return " ".join(str(text or "").split()).lower()


def ring(names, relation="causes"):
    return [(names[i], relation, names[(i + 1) % len(names)]) for i in range(len(names))]


def fuser(rows, **kwargs):
    return Fusion(Explainer(Store(rows)), **kwargs)


LOOPS = (ring(["sensor", "controller", "actuator", "quantity"])
         + ring(["receptor", "hypothalamus", "effector", "temperature"]))
SEEDS = {"engineering": ["sensor"], "biology": ["receptor"]}


# --------------------------------------------------------------------------- #
# The finding
# --------------------------------------------------------------------------- #
def test_two_domains_with_no_shared_word_are_matched_on_shape_alone():
    got = fuser(LOOPS).analogies(SEEDS)
    assert len(got) == 1 and got[0].exact
    assert set(got[0].mapping) == {"sensor", "controller", "actuator", "quantity"}
    assert set(got[0].mapping.values()) == {"receptor", "hypothalamus", "effector",
                                            "temperature"}


def test_the_mapping_carries_every_edge():
    analogy = fuser(LOOPS).analogies(SEEDS)[0]
    carried = {(analogy.mapping[s], r, analogy.mapping[o]) for s, r, o in analogy.left.edges}
    assert carried == set(analogy.right.edges)


def test_an_abstraction_reaches_every_domain_that_has_the_shape():
    """Three domains sharing one shape are one idea, not three resemblances."""
    rows = LOOPS + ring(["wage", "spending", "output", "employment"])
    seeds = dict(SEEDS, economics=["wage"])
    got = fuser(rows).abstract(seeds)
    assert len(got) == 1
    assert got[0].reach == 3
    assert got[0].domains == ["biology", "economics", "engineering"]
    for role, fills in got[0].roles.items():
        assert len(fills) == 3, f"{role} is not filled in every domain"


def test_the_roles_are_positions_and_the_abstraction_is_unnamed():
    got = fuser(LOOPS).abstract(SEEDS)[0]
    assert all(role.startswith("role") for role in got.roles)
    gloss = got.gloss()
    assert "role0" in gloss and "sensor" in gloss and "receptor" in gloss


# --------------------------------------------------------------------------- #
# What it refuses
# --------------------------------------------------------------------------- #
def test_a_near_miss_is_a_miss():
    """The same undirected cycle with a source and a sink: same fingerprint, no bijection."""
    p, q, r, t = "p", "q", "r", "t"
    rows = LOOPS + [(p, "causes", q), (q, "causes", r), (t, "causes", r), (p, "causes", t)]
    got = fuser(rows).analogies(dict(SEEDS, decoy=["p"]))
    named = {a.left.seed for a in got} | {a.right.seed for a in got}
    assert "p" not in named


def test_a_chain_is_not_a_loop():
    rows = LOOPS + [("price", "causes", "supply"), ("supply", "causes", "demand"),
                    ("demand", "causes", "surplus")]
    got = fuser(rows).analogies(dict(SEEDS, economics=["price"]))
    assert all("price" not in (a.left.seed, a.right.seed) for a in got)


def test_two_subgraphs_from_one_domain_are_a_duplicate_not_an_analogy():
    rows = LOOPS + ring(["s2", "c2", "a2", "q2"])
    got = fuser(rows).analogies({"engineering": ["sensor", "s2"]})
    assert got == []


def test_a_shape_must_be_big_enough_to_be_a_shape():
    """Two nodes joined by one edge are isomorphic to every other such pair."""
    rows = [("x", "causes", "y"), ("p", "causes", "q")]
    assert fuser(rows).analogies({"one": ["x"], "two": ["p"]}) == []
    assert MIN_EDGES >= 3


def test_is_a_is_not_a_shape_relation():
    """Otherwise every taxonomy matches every other and the hierarchy itself is the insight."""
    rows = ring(["a", "b", "c", "d"], relation="is_a") + ring(["w", "x", "y", "z"],
                                                              relation="is_a")
    assert fuser(rows).analogies({"one": ["a"], "two": ["w"]}) == []


# --------------------------------------------------------------------------- #
# The radius, which was measured
# --------------------------------------------------------------------------- #
def test_a_loop_needs_a_radius_at_least_its_length():
    """At two hops the four-edge loop came back as a two-edge line and matched nothing."""
    assert fuser(LOOPS, radius=2).analogies(SEEDS) == []
    assert fuser(LOOPS, radius=3).analogies(SEEDS)


def test_the_gauntlet_paper_measures_the_bijection():
    """It scored 1.000 with two different sabotages until the decoy was fixed."""
    import nyxara.njp.fusion as fusion_module
    from nyxara.njp.explaingauntlet import run

    original = fusion_module.Fusion.match

    def blind(self, left, right):
        if left.size < self.min_edges or left.size != right.size:
            return None
        if len(left.nodes) != len(right.nodes):
            return None
        return fusion_module.Analogy(left=left, right=right,
                                     mapping=dict(zip(left.nodes, right.nodes)))

    on = run(limit=12, attacks=("fusion",)).paper("fusion").score
    fusion_module.Fusion.match = blind
    try:
        off = run(limit=12, attacks=("fusion",)).paper("fusion").score
    finally:
        fusion_module.Fusion.match = original
    assert on == 1.0 and off == 0.0
