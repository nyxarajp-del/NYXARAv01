"""NYXARA · njp/procedureschool.py — is a read procedure the procedure that was written (📏).

Thirty-six task definitions taken out of the corpus by a fixed shuffle and marked by hand, on
three fields: what the definition says she is **given**, the **action** its goal is headed by, and
the set of **outputs** it allows. Twenty-one of them are the held-out set the reader was fixed
against; fifteen are :data:`SEALED` and were marked at the same sitting, before any of the fixing,
and read exactly once.

The marking convention matters more than the marks, because a boundary I chose by taste would be
scored against a boundary the reader chose by rule:

* **given** is scored on **heads**, not phrases. The mark says *"a list of integers"* and the
  reader may say *"a list of integers and an integer k"*; both reduce through
  :func:`~nyxara.njp.passage._head_of` to ``list``, and neither is credited for a modifier. This
  removes my choice of where a noun phrase ends from the score entirely.
* **action** is one token and is scored exact. There is nothing to argue about.
* **outputs** is a set, scored on membership after case folding. A definition that names no answer
  space is marked with the empty set and is credited for returning one — a reader that invents an
  answer space out of a quoted example is doing damage, and two of the held-out items exist to
  catch exactly that.

Precision and recall are reported apart for ``given`` and ``outputs``. A reader that returns every
noun phrase in the definition would score a perfect recall on ``given``, and reporting only the
mean would hide it.

Four controls, and each removes one mechanism rather than turning a number down:

* **cold** — a reader constructed and never taught. The floor.
* **no_cued** — frames only. Every shape that kept the demonstration's own words survives; every
  shape that abstracted them is gone. What this costs is what abstraction bought, and if it costs
  nothing then the cued level is decoration.
* **no_frames** — the other half, and the sharper test: only the abstracted shapes remain, so a
  definition can only be read by structure it shares with a demonstration rather than by wording.
* **one_lesson** — taught the first demonstration alone. Whether fourteen lessons were
  fourteen lessons' worth.
"""

from __future__ import annotations

import gzip
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.passage import _head_of
from nyxara.njp.procedure import LESSONS, ProcedureReader, taught_procedures

__all__ = ["Mark", "Score", "Report", "AUDIT", "HELD_OUT", "SEALED", "CORPUS",
           "read_corpus", "grade", "examine", "actions", "run"]

CORPUS = Path(__file__).with_name("data") / "flan_instruction.jsonl.gz"

#: The seed that drew the audit out of the corpus. The 698 are sorted by task id and shuffled once
#: with this; :data:`AUDIT` is the first thirty-six. Recorded so the sample can be redrawn and
#: seen to be the sample, rather than a set of definitions chosen because they read well.
SEED = 53


@dataclass(frozen=True)
class Mark:
    """One definition, marked by hand. ``outputs`` empty means *it names no answer space*."""

    task: str = ""
    given: Tuple[str, ...] = ()
    action: str = ""
    outputs: Tuple[str, ...] = ()


@dataclass
class Score:
    right: int = 0
    said: int = 0
    wanted: int = 0

    @property
    def precision(self) -> float:
        return round(self.right / self.said, 4) if self.said else 1.0

    @property
    def recall(self) -> float:
        return round(self.right / self.wanted, 4) if self.wanted else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return round(2 * p * r / (p + r), 4) if (p + r) else 0.0


@dataclass
class Report:
    name: str = ""
    given: Score = field(default_factory=Score)
    outputs: Score = field(default_factory=Score)
    action_right: int = 0
    action_asked: int = 0
    #: Items where the definition names no answer space and the reader agreed. Reported apart
    #: because they are the ones a greedy reader loses and a mean would bury.
    silent_right: int = 0
    silent_asked: int = 0
    misread: List[Tuple[str, str, str, str]] = field(default_factory=list)

    @property
    def action(self) -> float:
        return round(self.action_right / self.action_asked, 4) if self.action_asked else 0.0

    @property
    def overall(self) -> float:
        """The mean of the three fields. Never quoted without the three beside it."""
        return round((self.given.f1 + self.action + self.outputs.f1) / 3, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "overall": self.overall,
                "given": {"f1": self.given.f1, "precision": self.given.precision,
                          "recall": self.given.recall},
                "action": self.action,
                "outputs": {"f1": self.outputs.f1, "precision": self.outputs.precision,
                            "recall": self.outputs.recall},
                "no_answer_space_kept": f"{self.silent_right}/{self.silent_asked}"}

    def render(self) -> str:
        return (f"{self.name:<14} overall {self.overall:.3f}   "
                f"given {self.given.f1:.3f} (p {self.given.precision:.3f} "
                f"r {self.given.recall:.3f})   action {self.action:.3f}   "
                f"outputs {self.outputs.f1:.3f} (p {self.outputs.precision:.3f} "
                f"r {self.outputs.recall:.3f})   "
                f"silent {self.silent_right}/{self.silent_asked}")


# --------------------------------------------------------------------------------------------- #
#  the audit
# --------------------------------------------------------------------------------------------- #
#: Twenty-one definitions marked by hand and used while the reader was being fixed.
HELD_OUT: Tuple[Mark, ...] = (
    Mark("task1194_kth_largest_element", ("a list of integers", "an integer k"), "find"),
    Mark("task1645_medical_question_pair_dataset_text_classification",
         ("a medical question pair hand-generated",), "classify", ("Similar", "Dissimilar")),
    Mark("task559_alt_translation_en_fi", ("a sentence in the English language",), "convert"),
    Mark("task1574_amazon_reviews_multi_language_identification",
         ("reviews of various products in multiple languages",), "predict",
         ("en", "ja", "de", "fr", "zh", "es")),
    Mark("task1509_evalution_antonyms", ("an adjective",), "generate"),
    Mark("task958_e2e_nlg_text_generation_parse", (), "parse"),
    Mark("task123_conala_sort_dictionary", ("a list of dictionaries",), "sort"),
    Mark("task744_eurlex_classification", ("an article of the legal acts",), "classify",
         ("Regulation", "Decision", "Directive")),
    # Names "True or False" — but as what the *generated question* can be answered by, not as
    # this task's own answer space. Marked empty on purpose.
    Mark("task1660_super_glue_question_generation",
         ("Wikipedia articles on a range of topics",), "write"),
    Mark("task017_mctaco_wrong_answer_generation_frequency", (), "write"),
    Mark("task1520_qa_srl_answer_generation", ("a sentence", "question"), "answer"),
    Mark("task434_alt_en_hi_answer_generation",
         ("a sentence in the English and Hindi language",), "check", ("Yes", "No")),
    Mark("task284_imdb_classification", ("a review of movie",), "classify",
         ("positive", "negative")),
    Mark("task184_break_generate_question",
         ("a set of steps that are required to answer a specific question",), "generate"),
    Mark("task525_parsinlu_movie_aspect_classification",
         ("a movie review",
          "a question about the reviewer's sentiment toward one aspect of the movie in Persian"),
         "infer",
         ("no sentiment expressed", "negative", "neutral", "positive", "mixed")),
    Mark("task512_twitter_emotion_classification", ("Twitter posts",), "label",
         ("sadness", "joy", "love", "anger", "fear", "surprise")),
    Mark("task1190_add_integer_to_list", ("a list of integers", "an integer k"), "add"),
    Mark("task203_mnli_sentence_generation",
         ("a statement", "the genre to which that statement belongs",
          "a label indicating if the statement should be agreed with"), "write"),
    Mark("task065_timetravel_consistent_sentence_classification",
         ("a short story consisting of exactly 5 sentences", "two options"), "select",
         ("Option 1", "Option 2")),
    Mark("task668_extreme_abstract_summarization", ("the abstract of a research paper",),
         "generate"),
    Mark("task1188_count_max_freq_char", ("a string with duplicate characters",), "return"),
)

#: Fifteen more, marked at the same sitting and read once, at the end. Nothing here was looked at
#: while the reader was being fixed, which is what makes the last row of the report mean anything.
SEALED: Tuple[Mark, ...] = (
    Mark("task157_count_vowels_and_consonants", (), "count"),
    Mark("task1148_maximum_ascii_value", ("a string with unique characters",), "return"),
    Mark("task292_storycommonsense_character_text_generation", ("a story",), "find"),
    Mark("task544_alt_translation_hi_en", ("a sentence in the Hindi language",), "convert"),
    Mark("task626_xlwic_sentence_based_on_given_word_sentence_generation", ("a word",), "respond"),
    Mark("task526_parsinlu_movie_overal_classification", ("a movie review in Persian",),
         "classify", ("negative", "neutral", "positive", "mixed")),
    Mark("task1332_check_leap_year", ("a year",), "check", ("1", "0")),
    Mark("task267_concatenate_and_reverse_all_elements_from_index_i_to_j",
         ("inputs 'i', 'j', and A",), "concatenate"),
    Mark("task091_all_elements_from_index_i_to_j", ("inputs i,j, and A",), "list"),
    Mark("task1420_mathqa_general", (), "answer", ("a", "b", "c", "d", "e")),
    Mark("task1192_food_flavor_profile", ("the name of an Indian food dish",), "classify",
         ("sweet", "spicy")),
    Mark("task080_piqa_answer_generation", ("the provided goal task in the input",), "describe"),
    Mark("task113_count_frequency_of_letter", (), "count"),
    Mark("task1142_xcsr_ar_commonsense_mc_classification",
         ("a question having multiple possible answers in Arabic language",), "choose",
         ("A", "B", "C", "D", "E")),
    Mark("task1139_xcsr_ru_commonsense_mc_classification",
         ("a question having multiple possible answers in Russian language",), "choose",
         ("A", "B", "C", "D", "E")),
)

AUDIT: Tuple[Mark, ...] = HELD_OUT + SEALED


# --------------------------------------------------------------------------------------------- #
#  the corpus
# --------------------------------------------------------------------------------------------- #
def read_corpus(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """The 698 distinct task definitions, or nothing if the corpus has not been built."""
    source = Path(path) if path is not None else CORPUS
    out: List[Dict[str, Any]] = []
    if not source.exists():
        return out
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
    return out


def sample(rows: Optional[Sequence[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """The thirty-six the audit marks, drawn the way they were drawn."""
    corpus = list(rows if rows is not None else read_corpus())
    corpus.sort(key=lambda r: str(r.get("task") or ""))
    random.Random(SEED).shuffle(corpus)
    return corpus[:len(AUDIT)]


def _by_task(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r.get("task") or ""): r for r in rows}


# --------------------------------------------------------------------------------------------- #
#  marking
# --------------------------------------------------------------------------------------------- #
def _heads(phrases: Sequence[str]) -> List[str]:
    return [h for h in (_head_of(p) for p in phrases) if h]


def _fold(values: Sequence[str]) -> List[str]:
    return [" ".join(str(v).split()).strip(" .,;:'\"").lower() for v in values if str(v).strip()]


def grade(reader: ProcedureReader, marks: Sequence[Mark], rows: Sequence[Dict[str, Any]],
          name: str = "") -> Report:
    """Read each marked definition and score it against the mark, field by field."""
    out = Report(name=name)
    table = _by_task(rows)
    for mark in marks:
        row = table.get(mark.task)
        if row is None:
            continue
        got = reader.read(str(row.get("instruction") or ""), task=mark.task,
                          source=str(row.get("source") or ""),
                          licence=str(row.get("licence") or ""))

        want_given, said_given = set(_heads(mark.given)), set(_heads(got.given))
        out.given.right += len(want_given & said_given)
        out.given.said += len(said_given)
        out.given.wanted += len(want_given)

        out.action_asked += 1
        if got.action == mark.action:
            out.action_right += 1
        else:
            out.misread.append((mark.task, "action", mark.action, got.action))

        want_out, said_out = set(_fold(mark.outputs)), set(_fold(got.outputs))
        out.outputs.right += len(want_out & said_out)
        out.outputs.said += len(said_out)
        out.outputs.wanted += len(want_out)
        if not want_out:
            out.silent_asked += 1
            if not said_out:
                out.silent_right += 1
            else:
                out.misread.append((mark.task, "outputs", "—", " | ".join(sorted(said_out))))
        elif want_out != said_out:
            out.misread.append((mark.task, "outputs", " | ".join(sorted(want_out)),
                                " | ".join(sorted(said_out))))
        if want_given != said_given:
            out.misread.append((mark.task, "given", " | ".join(sorted(want_given)),
                                " | ".join(sorted(said_given))))
    return out


def _filtered(keep: str) -> ProcedureReader:
    """A taught reader with one level of shape removed. The mechanism, not a threshold."""
    reader = taught_procedures()
    reader.shapes = {k: v for k, v in reader.shapes.items() if v.level == keep}
    return reader


def examine(rows: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Report]:
    corpus = list(rows if rows is not None else read_corpus())
    out: Dict[str, Report] = {}
    if not corpus:
        return out
    taught = taught_procedures()
    out["cold"] = grade(ProcedureReader(), HELD_OUT, corpus, "cold")
    out["_reader"] = taught  # type: ignore[assignment]
    out["no_cued"] = grade(_filtered("frame"), HELD_OUT, corpus, "frames only")
    out["no_frames"] = grade(_filtered("cued"), HELD_OUT, corpus, "cued only")
    out["one_lesson"] = grade(taught_procedures(LESSONS[:1]), HELD_OUT, corpus, "one lesson")
    out["taught"] = grade(taught, HELD_OUT, corpus, "taught")
    out["sealed"] = grade(taught, SEALED, corpus, "sealed")
    del out["_reader"]
    return out


# --------------------------------------------------------------------------------------------- #
#  what the 698 turn out to be
# --------------------------------------------------------------------------------------------- #
def actions(rows: Optional[Sequence[Dict[str, Any]]] = None,
            reader: Optional[ProcedureReader] = None) -> List[Tuple[str, int]]:
    """The taxonomy of the corpus, from the reading rather than imposed on it.

    Nothing decided in advance that a task definition names one of a dozen jobs. Each of the 698
    is read, the head verb of its goal is taken, and this is the tally. What comes out is the
    shape of what people actually ask for — and how much of the corpus the reader still cannot
    give a goal to at all, which is the row that matters.
    """
    corpus = list(rows if rows is not None else read_corpus())
    engine = reader if reader is not None else taught_procedures()
    tally: Dict[str, int] = {}
    for row in corpus:
        got = engine.read(str(row.get("instruction") or ""), task=str(row.get("task") or ""))
        tally[got.action or "—"] = tally.get(got.action or "—", 0) + 1
    return sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))


def coverage(rows: Optional[Sequence[Dict[str, Any]]] = None,
             reader: Optional[ProcedureReader] = None) -> Dict[str, Any]:
    """How much of the whole corpus each field is read on. Not accuracy — reach."""
    corpus = list(rows if rows is not None else read_corpus())
    engine = reader if reader is not None else taught_procedures()
    got = [engine.read(str(r.get("instruction") or ""), task=str(r.get("task") or ""))
           for r in corpus]
    n = max(1, len(got))
    return {
        "definitions": len(got),
        "with_goal": round(sum(1 for p in got if p.goal) / n, 4),
        "with_given": round(sum(1 for p in got if p.given) / n, 4),
        "with_outputs": round(sum(1 for p in got if p.outputs) / n, 4),
        "with_conditions": round(sum(1 for p in got if p.conditions) / n, 4),
        "with_caveats": round(sum(1 for p in got if p.caveats) / n, 4),
        "decidable": sum(1 for p in got if p.decides),
        "givens_per_definition": round(sum(len(p.given) for p in got) / n, 3),
    }


def run() -> Dict[str, Any]:  # pragma: no cover — a report, not a test
    corpus = read_corpus()
    if not corpus:
        print("no corpus; run scripts/stream_flan.py and scripts/merge_flan_shards.py")
        return {}
    print(f"{len(corpus)} distinct task definitions\n")
    got = examine(corpus)
    for key in ("cold", "no_cued", "no_frames", "one_lesson", "taught", "sealed"):
        print("  " + got[key].render())
    # One reader for the reach, the taxonomy and the report of what it learned -- a fresh one
    # would have read nothing, and ``generalising`` would print 0 whatever the truth was.
    reader = taught_procedures()
    print("\nreach over all", len(corpus), "definitions:")
    for key, value in coverage(corpus, reader).items():
        print(f"    {key:<24} {value}")
    print("\nwhat the corpus asks for:")
    for verb, count in actions(corpus, reader)[:20]:
        print(f"    {count:>4}  {verb}")
    print("\nwhat she worked out:")
    for key, value in reader.learned().items():
        print(f"    {key}: {value}")
    return {k: v.to_dict() for k, v in got.items()}


if __name__ == "__main__":  # pragma: no cover
    run()
