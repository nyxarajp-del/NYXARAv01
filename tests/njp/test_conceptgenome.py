"""What a concept is made of, before it is called anything (NJP V.47)."""
from __future__ import annotations


from nyxara.njp.explain import Explainer
from nyxara.njp.conceptgenome import SLOT_RELATIONS, SLOTS, Genome, read_genome


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


PUMP = [("pump", "has_part", "impeller"), ("pump", "requires", "energy"),
        ("pump", "causes", "flow"), ("pump", "occurs_when", "the motor turns"),
        ("pump", "has_property", "conserves mass")]
HEART = [("heart", "has_part", "ventricle"), ("heart", "requires", "oxygen"),
         ("heart", "causes", "circulation"), ("heart", "occurs_when", "the muscle contracts"),
         ("heart", "has_property", "conserves mass")]


def genomes(rows):
    ex = Explainer(Store(rows))
    return ex


# --------------------------------------------------------------------------- #
# The eight slots
# --------------------------------------------------------------------------- #
def test_a_genome_has_exactly_the_eight_slots():
    got = Genome(subject="x")
    assert tuple(got.slots) == SLOTS and len(SLOTS) == 8


def test_each_relation_lands_in_its_own_slot():
    got = read_genome(genomes(PUMP), "pump")
    assert got.slots["roles"] == (("has_part", "impeller"),)
    assert got.slots["constraints"] == (("requires", "energy"),)
    assert got.slots["causal"] == (("causes", "flow"),)
    assert got.slots["temporal"] == (("occurs_when", "the motor turns"),)
    assert got.slots["invariants"] == (("has_property", "conserves mass"),)


def test_an_unclassified_relation_is_dropped_not_swept_into_relations():
    """That slot would absorb everything and the fingerprint would stop discriminating."""
    got = read_genome(genomes(PUMP + [("pump", "related_to", "anything")]), "pump")
    assert all("anything" not in obj for slot in SLOTS for _r, obj in got.slots[slot])
    assert "related_to" not in {r for row in SLOT_RELATIONS.values() for r in row}


def test_an_empty_slot_is_empty():
    got = read_genome(genomes(PUMP), "pump")
    assert got.slots["transformations"] == () and got.slots["exceptions"] == ()
    assert "transformations" not in got.filled


# --------------------------------------------------------------------------- #
# The fingerprint has no names in it
# --------------------------------------------------------------------------- #
def test_the_fingerprint_contains_no_vocabulary():
    got = read_genome(genomes(PUMP), "pump")
    flat = str(got.fingerprint)
    for word in ("pump", "impeller", "energy", "flow", "motor"):
        assert word not in flat


def test_two_concepts_sharing_no_word_have_the_same_fingerprint():
    ex = genomes(PUMP + HEART)
    assert read_genome(ex, "pump").fingerprint == read_genome(ex, "heart").fingerprint


# --------------------------------------------------------------------------- #
# Kinship, and where it differs
# --------------------------------------------------------------------------- #
def test_structural_kinship_is_found_across_vocabularies():
    ex = genomes(PUMP + HEART)
    got = read_genome(ex, "pump").compare(read_genome(ex, "heart"))
    assert got.aligned and got.score == 1.0
    assert got.mapping["impeller"] == "ventricle"
    assert got.mapping["energy"] == "oxygen"


def test_the_report_says_where_two_concepts_differ():
    """A single similarity score is exactly the thing that hides this."""
    ex = genomes(PUMP + [r for r in HEART if r[1] != "occurs_when"]
                 + [("heart", "becomes", "a scar")])
    got = read_genome(ex, "pump").compare(read_genome(ex, "heart"))
    assert "temporal" in got.differs and "transformations" in got.differs
    assert "roles" not in got.differs
    assert got.per_slot["roles"] == 1.0 and got.per_slot["temporal"] == 0.0


def test_a_partial_match_is_not_called_a_match():
    ex = genomes(PUMP + [("heart", "has_part", "ventricle")])
    got = read_genome(ex, "pump").compare(read_genome(ex, "heart"))
    assert got.aligned is False


def test_the_common_structure_is_never_named():
    ex = genomes(PUMP + HEART)
    text = read_genome(ex, "pump").compare(read_genome(ex, "heart")).gloss()
    assert "pump" in text and "heart" in text
    for invented in ("feedback", "regulation", "mechanism of"):
        assert invented not in text.lower()
