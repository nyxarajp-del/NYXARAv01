"""NYXARA · njp/passageschool.py — six lessons, then passages from subjects nobody taught (🏫).

:mod:`nyxara.njp.passage` induces its reading shapes from six demonstrations, all of them about
biology or weather. This examines it on passages from **economics, medicine, astronomy,
chemistry, computing, engineering, law, geology and meteorology** — subjects that appear in no
lesson, written by nobody who saw the lessons, about entities the reader has never met.

Three baselines run beside the reader on the identical passages, because a score with nothing to
compare it against is a number rather than a result:

* ``grounder`` — :mod:`nyxara.njp.grounding` alone. This is what the package could do before this
  module existed, and it is the honest floor for the whole change: everything above it is what
  reading a passage buys over parsing its sentences.
* ``cold`` — the reader with no lessons at all. It must score near zero, and if it does not, the
  shapes are not doing the work the lessons are credited with.
* four **falsification** runs, each with exactly one mechanism removed.

The papers are scored on what was actually asked for. ``gold`` holds only relations expressible
in the predicates the lessons taught; relations the passages plainly state in predicates nobody
demonstrated are counted separately as ``beyond`` and reported as a coverage gap rather than as
failures — marking her wrong for not knowing a relation nobody showed her would be scoring the
syllabus, not the reader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Set, Tuple

from nyxara.njp.passage import PassageReader, _bare, _toks, _words, taught_reader

__all__ = ["Item", "Paper", "Report", "HELD_OUT", "SEALED", "examine", "run",
           "grounder_baseline", "RUNS"]


def _key(text: str) -> str:
    """Compare on content words, determiner-free — the form the reader itself works in."""
    return " ".join(_words(_bare(_toks(str(text or "")))))


@dataclass
class Item:
    """One held-out passage and everything true of it that the taught predicates can say."""

    name: str = ""
    domain: str = ""
    text: str = ""
    concept: str = ""
    definition: str = ""
    kind: str = ""
    gold: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    #: Relations the passage states in predicates no lesson taught. Not failures — coverage.
    beyond: Tuple[str, ...] = ()
    #: What this item is in the exam for.
    papers: Tuple[str, ...] = ()
    condition: str = ""
    #: Who the last sentence is about, for the ``pronoun`` paper. Defaults to the concept, which
    #: is what makes the third item worth having: there the answer is **not** the concept, so a
    #: reader that always falls back to the topic gets it wrong and the resolver has to work.
    pronoun_expect: str = ""

    def pairs(self) -> Set[Tuple[str, str]]:
        return {(p, _key(o)) for p, objects in self.gold.items() for o in objects}


@dataclass
class Paper:
    """One subject's score, with the counts it was computed from kept beside it."""

    name: str = ""
    right: float = 0.0
    asked: int = 0
    got: int = 0
    wrong: int = 0

    @property
    def score(self) -> float:
        return round(self.right / self.asked, 4) if self.asked else 0.0

    @property
    def precision(self) -> float:
        return round(self.right / self.got, 4) if self.got else 0.0

    def to_dict(self) -> Dict[str, Any]:
        out = {"score": self.score, "asked": self.asked}
        if self.got:
            out["precision"] = self.precision
        return out


@dataclass
class Report:
    papers: Dict[str, Paper] = field(default_factory=dict)
    #: Relations produced that no gold row licenses, by item. The confabulation count.
    invented: Tuple[str, ...] = ()
    #: Relations the passages state that the taught predicates cannot express.
    gaps: Tuple[str, ...] = ()
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def mean(self) -> float:
        scores = [p.score for p in self.papers.values() if p.asked]
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    @property
    def recall(self) -> float:
        """Over the ``relations`` paper alone.

        Summing every paper was wrong and said so out loud: ``unseen_entities`` re-marks items
        the ``relations`` paper has already marked, and the fractional ``coordination`` marks
        have no ``got`` to divide by, so the first version of this printed a precision of
        **1.156** — a number that cannot exist and that nothing in the run was checking.
        """
        paper = self.papers.get("relations")
        return paper.score if paper else 0.0

    @property
    def precision(self) -> float:
        paper = self.papers.get("relations")
        return paper.precision if paper else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean, "recall": self.recall, "precision": self.precision,
                "papers": {n: p.to_dict() for n, p in sorted(self.papers.items())},
                "invented": list(self.invented), "gaps": list(self.gaps),
                "stats": self.stats}

    def render(self) -> str:
        lines = [f"mean {self.mean:.3f}   recall {self.recall:.3f}   "
                 f"precision {self.precision:.3f}"]
        for name, paper in sorted(self.papers.items()):
            if not paper.asked:
                continue
            lines.append(f"  {name:<16} {paper.score:.3f}  ({int(paper.right)}/{paper.asked})")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------------- #
#  the held-out passages
# --------------------------------------------------------------------------------------------- #
HELD_OUT: Tuple[Item, ...] = (
    Item(name="inflation", domain="economics", concept="inflation",
         text=("Inflation is a general rise in prices. Inflation requires an expanding money "
               "supply and produces a fall in real wages."),
         definition="general rise in prices", kind="rise",
         gold={"definition": ("a general rise in prices",),
               "requires": ("an expanding money supply",),
               "produces": ("a fall in real wages",)},
         papers=("definition", "kind", "relations", "unseen_entities")),
    Item(name="fermentation", domain="biology", concept="fermentation",
         text=("Fermentation is the reaction by which yeast releases energy from sugar. "
               "Fermentation occurs in the cytoplasm and requires anaerobic conditions."),
         definition="reaction by which yeast releases energy from sugar", kind="reaction",
         gold={"definition": ("the reaction by which yeast releases energy from sugar",),
               "occurs_in": ("yeast", "the cytoplasm"),
               "requires": ("anaerobic conditions",)},
         papers=("definition", "kind", "relations")),
    Item(name="nitrogen_fixation", domain="chemistry", concept="nitrogen fixation",
         text=("Nitrogen fixation is the process by which bacteria convert nitrogen into "
               "ammonia. Nitrogen fixation requires nitrogenase."),
         definition="process by which bacteria convert nitrogen into ammonia", kind="process",
         gold={"definition": ("the process by which bacteria convert nitrogen into ammonia",),
               "occurs_in": ("bacteria",), "uses": ("nitrogen",), "produces": ("ammonia",),
               "requires": ("nitrogenase",)},
         papers=("definition", "kind", "relations", "unseen_entities")),
    Item(name="electrolysis", domain="chemistry", concept="electrolysis",
         text=("Electrolysis is the process by which current splits water into hydrogen and "
               "oxygen. Electrolysis requires an electrolyte, two electrodes and a power "
               "supply."),
         definition="process by which current splits water into hydrogen and oxygen",
         kind="process",
         gold={"definition": ("the process by which current splits water into hydrogen and "
                              "oxygen",),
               "occurs_in": ("current",),
               "requires": ("an electrolyte", "two electrodes", "a power supply")},
         papers=("definition", "kind", "relations", "coordination")),
    Item(name="compilation", domain="computing", concept="compilation",
         text=("Compilation is the process by which a compiler turns source code into machine "
               "code. Compilation requires a parser, a type checker and a code generator."),
         definition="process by which a compiler turns source code into machine code",
         kind="process",
         gold={"definition": ("the process by which a compiler turns source code into machine "
                              "code",),
               "occurs_in": ("a compiler",),
               "requires": ("a parser", "a type checker", "a code generator")},
         papers=("definition", "kind", "relations", "coordination", "unseen_entities")),
    Item(name="dialysis", domain="medicine", concept="dialysis",
         text=("Dialysis is the treatment by which a machine removes waste from blood. "
               "Dialysis requires a semipermeable membrane and produces cleaned blood."),
         definition="treatment by which a machine removes waste from blood", kind="treatment",
         gold={"definition": ("the treatment by which a machine removes waste from blood",),
               "occurs_in": ("a machine",), "uses": ("waste",),
               "requires": ("a semipermeable membrane",),
               "produces": ("cleaned blood",)},
         papers=("definition", "kind", "relations", "unseen_entities")),
    Item(name="fusion", domain="astronomy", concept="stellar fusion",
         text=("Stellar fusion is the reaction by which a star converts hydrogen into helium. "
               "Stellar fusion requires enormous pressure and temperature."),
         definition="reaction by which a star converts hydrogen into helium", kind="reaction",
         gold={"definition": ("the reaction by which a star converts hydrogen into helium",),
               "occurs_in": ("a star",), "uses": ("hydrogen",), "produces": ("helium",),
               "requires": ("enormous pressure", "temperature")},
         papers=("definition", "kind", "relations", "coordination")),
    Item(name="welding", domain="engineering", concept="welding",
         text=("Welding is the process by which heat joins two metals. "
               "Welding requires a filler rod, shielding gas and a steady hand."),
         definition="process by which heat joins two metals", kind="process",
         gold={"definition": ("the process by which heat joins two metals",),
               "occurs_in": ("heat",), "uses": ("two metals",),
               "requires": ("a filler rod", "shielding gas", "a steady hand")},
         papers=("definition", "kind", "relations", "coordination")),
    Item(name="arbitration", domain="law", concept="arbitration",
         text=("Arbitration is the procedure by which an arbitrator settles a dispute outside "
               "court. Arbitration requires the consent of both parties."),
         definition="procedure by which an arbitrator settles a dispute outside court",
         kind="procedure",
         gold={"definition": ("the procedure by which an arbitrator settles a dispute outside "
                              "court",),
               "occurs_in": ("an arbitrator",),
               "requires": ("the consent of both parties",)},
         papers=("definition", "kind", "relations", "unseen_entities")),
    Item(name="subduction", domain="geology", concept="subduction",
         text=("Subduction is the movement by which one plate sinks beneath another. "
               "Subduction produces volcanoes and deep trenches."),
         definition="movement by which one plate sinks beneath another", kind="movement",
         gold={"definition": ("the movement by which one plate sinks beneath another",),
               "occurs_in": ("one plate",),
               "produces": ("volcanoes", "deep trenches")},
         papers=("definition", "kind", "relations", "coordination")),
    # -- the pronoun paper: the second sentence never names its subject --------------------- #
    Item(name="osmosis_pronoun", domain="biology", concept="osmosis",
         text=("Osmosis is the movement of water across a membrane. "
               "It requires a partially permeable membrane."),
         definition="movement of water across a membrane", kind="movement",
         gold={"definition": ("the movement of water across a membrane",),
               "requires": ("a partially permeable membrane",)},
         papers=("definition", "kind", "relations", "pronoun")),
    Item(name="corrosion_pronoun", domain="engineering", concept="corrosion",
         text=("Corrosion is the decay of metal in air. "
               "It requires moisture, oxygen and time."),
         definition="decay of metal in air", kind="decay",
         gold={"definition": ("the decay of metal in air",),
               "requires": ("moisture", "oxygen", "time")},
         papers=("definition", "kind", "relations", "pronoun", "coordination")),
    # A pronoun the topic does not name. A reader that files every clause under the passage's
    # subject gets this one wrong, which is the point of it: without it the ``no_pronouns``
    # ablation changed no score at all, and a mechanism no measurement can miss is a mechanism
    # nobody has shown to be doing anything.
    Item(name="silicon_pronoun", domain="engineering", concept="photovoltaics",
         text=("Photovoltaics is the generation of electricity from light. "
               "Silicon is the material that most cells use. "
               "It requires very high purity."),
         definition="generation of electricity from light", kind="generation",
         gold={"definition": ("the generation of electricity from light",
                              "the material that most cells use"),
               "requires": ("very high purity",)},
         pronoun_expect="silicon",
         papers=("definition", "kind", "relations", "pronoun")),
    # -- the condition paper --------------------------------------------------------------- #
    Item(name="precipitation", domain="meteorology", concept="precipitation",
         text=("Precipitation is the fall of water from cloud. "
               "Precipitation requires condensation nuclei if the air is very clean."),
         definition="fall of water from cloud", kind="fall",
         gold={"definition": ("the fall of water from cloud",),
               "requires": ("condensation nuclei",)},
         condition="the air is very clean",
         papers=("definition", "kind", "relations", "condition")),
    # -- the restraint papers: nothing taught is stated. Silence is the right answer -------- #
    Item(name="quiet_history", domain="history", concept="the treaty of westphalia",
         text=("The Treaty of Westphalia was signed in 1648. "
               "It ended a long European war."),
         gold={}, beyond=("signed_in=1648", "ended=a long european war"),
         papers=("restraint",)),
    Item(name="quiet_geography", domain="geography", concept="the danube",
         text=("The Danube flows through ten countries. "
               "It reaches the Black Sea."),
         gold={}, beyond=("flows_through=ten countries", "reaches=the black sea"),
         papers=("restraint",)),
    # -- the preservation paper ------------------------------------------------------------- #
    Item(name="viscosity", domain="physics", concept="viscosity",
         text=("Viscosity is the resistance of a fluid to flow. "
               "Viscosity requires a temperature to be meaningful."),
         definition="resistance of a fluid to flow", kind="resistance",
         gold={"definition": ("the resistance of a fluid to flow",),
               "requires": ("a temperature",)},
         papers=("definition", "kind", "relations", "preserved")),
)


#: A second exam, written after :data:`HELD_OUT` had already been used to find and fix defects,
#: from nine further subjects, and **run once**. That is the whole of its value: the first set
#: stopped being held out the moment a fix was made because of it, and a score on items the fixes
#: were shaped against is a report about the fixes. Whatever this returns is the number.
SEALED: Tuple[Item, ...] = (
    Item(name="pasteurisation", domain="food science", concept="pasteurisation",
         text=("Pasteurisation is the treatment by which heat kills bacteria in milk. "
               "Pasteurisation requires a holding time and a rapid chill."),
         definition="treatment by which heat kills bacteria in milk", kind="treatment",
         gold={"definition": ("the treatment by which heat kills bacteria in milk",),
               "occurs_in": ("heat",), "uses": ("bacteria",),
               "requires": ("a holding time", "a rapid chill")},
         papers=("definition", "kind", "relations", "coordination", "unseen_entities")),
    Item(name="sedimentation", domain="civil engineering", concept="sedimentation",
         text=("Sedimentation is the settling of solids in a tank. "
               "It requires still water, time and a shallow depth."),
         definition="settling of solids in a tank", kind="settling",
         gold={"definition": ("the settling of solids in a tank",),
               "requires": ("still water", "time", "a shallow depth")},
         papers=("definition", "kind", "relations", "coordination", "pronoun")),
    Item(name="tempering", domain="metallurgy", concept="tempering",
         text=("Tempering is the treatment by which reheating softens hardened steel. "
               "Tempering produces a tougher blade."),
         definition="treatment by which reheating softens hardened steel", kind="treatment",
         gold={"definition": ("the treatment by which reheating softens hardened steel",),
               "occurs_in": ("reheating",), "uses": ("hardened steel",),
               "produces": ("a tougher blade",)},
         papers=("definition", "kind", "relations", "unseen_entities")),
    Item(name="composting", domain="agriculture", concept="composting",
         text=("Composting is the process by which microbes break waste into humus. "
               "Composting requires moisture, air and warmth."),
         definition="process by which microbes break waste into humus", kind="process",
         gold={"definition": ("the process by which microbes break waste into humus",),
               "occurs_in": ("microbes",), "uses": ("waste",),
               "requires": ("moisture", "air", "warmth")},
         papers=("definition", "kind", "relations", "coordination")),
    Item(name="securitisation", domain="banking", concept="securitisation",
         text=("Securitisation is the practice by which a bank pools loans into a bond. "
               "Securitisation requires a rating agency."),
         definition="practice by which a bank pools loans into a bond", kind="practice",
         gold={"definition": ("the practice by which a bank pools loans into a bond",),
               "occurs_in": ("a bank",), "uses": ("loans",), "produces": ("a bond",),
               "requires": ("a rating agency",)},
         papers=("definition", "kind", "relations", "unseen_entities")),
    Item(name="dyeing", domain="textiles", concept="dyeing",
         text=("Dyeing is the process by which a mordant fixes colour into cloth. "
               "Dyeing requires hot water if the fibre is wool."),
         definition="process by which a mordant fixes colour into cloth", kind="process",
         gold={"definition": ("the process by which a mordant fixes colour into cloth",),
               "occurs_in": ("a mordant",), "uses": ("colour",), "produces": ("cloth",),
               "requires": ("hot water",)},
         condition="the fibre is wool",
         papers=("definition", "kind", "relations", "condition")),
    Item(name="quiet_music", domain="music", concept="the fugue",
         text=("The fugue reached its height under Bach. "
               "It spread across Europe in the eighteenth century."),
         gold={}, beyond=("height_under=bach", "spread_across=europe"),
         papers=("restraint",)),
    Item(name="quiet_sport", domain="sport", concept="the marathon",
         text=("The marathon covers forty-two kilometres. "
               "It was added to the first modern Olympics."),
         gold={}, beyond=("covers=forty-two kilometres", "added_to=the first modern olympics"),
         papers=("restraint",)),
    Item(name="navigation", domain="navigation", concept="dead reckoning",
         text=("Dead reckoning is the method by which a navigator estimates position from "
               "speed. Dead reckoning requires a known start, a heading and an elapsed time."),
         definition="method by which a navigator estimates position from speed", kind="method",
         gold={"definition": ("the method by which a navigator estimates position from speed",),
               "occurs_in": ("a navigator",), "uses": ("position",),
               "requires": ("a known start", "a heading", "an elapsed time")},
         papers=("definition", "kind", "relations", "coordination", "preserved")),
)


# --------------------------------------------------------------------------------------------- #
#  the examination
# --------------------------------------------------------------------------------------------- #
def _mark(report: Report, paper: str, right: float, asked: int = 1,
          got: int = 0) -> None:
    held = report.papers.setdefault(paper, Paper(name=paper))
    held.right += right
    held.asked += asked
    held.got += got


def examine(reader: PassageReader, items: Sequence[Item] = HELD_OUT) -> Report:
    """Read every held-out passage and mark what came back."""
    report = Report()
    invented: List[str] = []
    gaps: List[str] = []
    for item in items:
        obj = reader.read(item.text, concept=item.concept, source="school",
                          domain=item.domain)
        produced = {(r.predicate, _key(r.object)) for r in obj.relations}
        wanted = item.pairs()
        gaps.extend(f"{item.name}: {b}" for b in item.beyond)

        if "definition" in item.papers:
            _mark(report, "definition",
                  1.0 if _key(obj.definition) == _key(item.definition) else 0.0,
                  got=1 if obj.definition else 0)
        if "kind" in item.papers:
            _mark(report, "kind", 1.0 if _key(obj.kind) == _key(item.kind) else 0.0,
                  got=1 if obj.kind else 0)
        if "relations" in item.papers:
            hit = len(produced & wanted)
            _mark(report, "relations", hit, asked=len(wanted), got=len(produced))
            for predicate, obj_text in sorted(produced - wanted):
                invented.append(f"{item.name}: {predicate}={obj_text}")
        if "coordination" in item.papers:
            # Every gold predicate that was demonstrated with several objects must come back
            # with all of them, not with the coordinated span fused into one.
            lists = {p: objs for p, objs in item.gold.items() if len(objs) > 1}
            for predicate, objects in lists.items():
                want = {(predicate, _key(o)) for o in objects}
                _mark(report, "coordination", len(want & produced) / len(want))
        if "pronoun" in item.papers:
            # The relations of the last sentence must be filed under whoever it is about — not
            # under a pronoun, and not under nothing.
            last = obj.sentences[-1] if obj.sentences else ""
            subjects = {_key(r.subject) for r in obj.relations if r.sentence == last}
            want = _key(item.pronoun_expect or item.concept)
            _mark(report, "pronoun", 1.0 if subjects == {want} else 0.0)
        if "condition" in item.papers:
            carried = {r.condition for r in obj.relations if r.condition}
            _mark(report, "condition",
                  1.0 if _key(item.condition) in {_key(c) for c in carried} else 0.0)
        if "restraint" in item.papers:
            # Nothing the lessons taught is stated here. Producing a relation anyway is a
            # confabulation, and this is the only paper where silence scores full marks.
            _mark(report, "restraint", 1.0 if not obj.relations else 0.0,
                  got=len(obj.relations))
            invented.extend(f"{item.name}: {p}={o}" for p, o in sorted(produced))
        if "unseen_entities" in item.papers:
            hit = len(produced & wanted)
            _mark(report, "unseen_entities", hit, asked=len(wanted), got=len(produced))
        if "preserved" in item.papers:
            kept = (obj.text == item.text
                    and len(obj.sentences) == len(item.text.split(". "))
                    and any(obj.provenance.get(s) for s in obj.sentences))
            _mark(report, "preserved", 1.0 if kept else 0.0)
    report.invented = tuple(invented)
    report.gaps = tuple(gaps)
    report.stats = reader.stats()
    return report


# --------------------------------------------------------------------------------------------- #
#  the baselines
# --------------------------------------------------------------------------------------------- #
def grounder_baseline(items: Sequence[Item] = HELD_OUT) -> Report:
    """What the package could already do: :mod:`nyxara.njp.grounding`, sentence by sentence.

    Scored by exactly the same marker, so the comparison is like for like. Its predicates are
    its own — ``is_a`` rather than ``definition`` — so ``is_a`` is read as an attempt at the
    definition, which is the most generous reading available to it.
    """
    report = Report()
    invented: List[str] = []
    try:
        from nyxara.njp.grounding import Grounder
    except Exception:  # noqa: BLE001 - pragma: no cover
        return report
    grounder = Grounder()
    for item in items:
        produced: Set[Tuple[str, str]] = set()
        definition, subjects = "", set()
        for sentence in [s.strip() for s in item.text.split(". ") if s.strip()]:
            if not sentence.endswith("."):
                sentence += "."
            try:
                grounded = grounder.ground(sentence)
            except Exception:  # noqa: BLE001
                continue
            for triple in getattr(grounded, "triples", ()) or ():
                predicate = "definition" if triple.predicate == "is_a" else triple.predicate
                produced.add((predicate, _key(triple.object)))
                subjects.add(_key(triple.subject))
                if predicate == "definition" and not definition:
                    definition = triple.object
        wanted = item.pairs()
        if "definition" in item.papers:
            _mark(report, "definition", 1.0 if _key(definition) == _key(item.definition) else 0.0,
                  got=1 if definition else 0)
        if "kind" in item.papers:
            _mark(report, "kind", 0.0, got=0)      # the grounder has no separate kind at all
        if "relations" in item.papers:
            hit = len(produced & wanted)
            _mark(report, "relations", hit, asked=len(wanted), got=len(produced))
            invented.extend(f"{item.name}: {p}={o}" for p, o in sorted(produced - wanted))
        if "coordination" in item.papers:
            for predicate, objects in item.gold.items():
                if len(objects) < 2:
                    continue
                want = {(predicate, _key(o)) for o in objects}
                _mark(report, "coordination", len(want & produced) / len(want))
        if "pronoun" in item.papers:
            _mark(report, "pronoun", 1.0 if subjects == {_key(item.concept)} else 0.0)
        if "condition" in item.papers:
            _mark(report, "condition", 0.0)
        if "restraint" in item.papers:
            _mark(report, "restraint", 1.0 if not produced else 0.0, got=len(produced))
        if "unseen_entities" in item.papers:
            _mark(report, "unseen_entities", len(produced & wanted), asked=len(wanted),
                  got=len(produced))
        if "preserved" in item.papers:
            _mark(report, "preserved", 0.0)        # it keeps a sentence, never the passage
    report.invented = tuple(invented)
    report.stats = {"reader": "grounding.Grounder"}
    return report


#: Every run this school reports, and what removing the mechanism is supposed to prove.
RUNS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("taught", {}),
    ("frames_only", {"use_cued": False}),
    ("no_lists", {"expand_lists": False}),
    ("no_pronouns", {"resolve_pronouns": False}),
    ("no_topic", {"topic_subject": False}),
    ("witness_1", {"min_witnesses": 1}),
)


def run(items: Sequence[Item] = HELD_OUT) -> Dict[str, Report]:
    """Every run, in one dictionary: the reader, its ablations, and the two floors."""
    out: Dict[str, Report] = {}
    for name, options in RUNS:
        out[name] = examine(taught_reader(**options), items)
    out["cold"] = examine(PassageReader(), items)
    out["grounder"] = grounder_baseline(items)
    if items is HELD_OUT:
        out["sealed"] = examine(taught_reader(), SEALED)
        out["sealed_grounder"] = grounder_baseline(SEALED)
    return out


def main() -> None:  # pragma: no cover - a report, not a test
    reports = run()
    for name in ("grounder", "cold", "frames_only", "no_lists", "no_pronouns", "no_topic",
                 "witness_1", "taught", "sealed_grounder", "sealed"):
        report = reports.get(name)
        if report is None:
            continue
        print(f"\n=== {name} ===")
        print(report.render())
    taught = reports["taught"]
    if taught.invented:
        print("\nconfabulated:")
        for row in taught.invented[:20]:
            print("   ", row)
    if taught.gaps:
        print(f"\nbeyond the taught predicates: {len(taught.gaps)}")


if __name__ == "__main__":  # pragma: no cover
    main()
