"""NYXARA · growth/synth_chat.py — the conversation shapes that make a small model useful (💬).

A 300M model cannot store the world. What it can learn is a *policy*: when to reach for a tool,
how to read what comes back, when to say it does not know, and how to keep track of what "it"
refers to three turns later. Those are the shapes here, and every one of them is checkable.

Three properties are enforced rather than hoped for:

* **Every number in an assistant turn is computed for real.** A trace whose tool output was
  invented teaches the model to fabricate tool results, which is worse than having no tool data.
* **Format constraints are verified before the document is kept.** A document that says "answer
  in exactly three bullets" and then gives four teaches the model that instructions are
  decorative.
* **Unanswerable cases are deliberately included.** A model trained only on answerable
  context-question pairs learns that an answer always exists in the passage, and confabulates
  one when it does not. The negative cases are the important half.

The old conversation domain was single-turn tool calls and retrieval pairs. Multi-turn is added
here because coreference — "make it shorter", "what about the second one" — is the thing that
distinguishes a conversation from a list of unrelated questions, and no amount of single-turn
data teaches it.
"""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Optional, Sequence, Tuple

from nyxara.growth.synth_common import (
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_RESULT_CLOSE,
    TOOL_RESULT_OPEN,
    Generator,
)

__all__ = ["GENERATORS", "generators"]

Built = Optional[Tuple[str, str]]

_NAMES = ("Aarav", "Priya", "Rahul", "Meera", "Kabir", "Ananya", "Vikram", "Nisha", "Arjun",
          "Diya", "Rohan", "Sana", "Yusuf", "Leila", "Omar", "Zara", "Mateo", "Sofia",
          "Hana", "Kenji", "Amara", "Noor", "Elena", "Tomas", "Grace", "Jonas")
_UNITS = ("kg", "mm", "°C", "ms", "V", "L", "Hz", "%", "rpm", "kPa", "lux", "mAh")
_ATTRS = ("mass", "length", "operating temperature", "cycle time", "output voltage",
          "capacity", "resonant frequency", "tolerance", "idle draw", "peak load",
          "seal pressure", "recovery time")
_TOPICS = ("the loading dock sensor", "the backup pump", "the north array", "the intake filter",
           "the calibration rig", "the drying oven", "the relay board", "the coolant loop")


def _pick(rng: random.Random, pool):
    return pool[rng.randrange(len(pool))]


def _turn(user: str, assistant: str) -> str:
    return f"### User:\n{user}\n\n### NYXARA:\n{assistant}\n\n"


def _conversation(turns: Sequence[Tuple[str, str]]) -> Tuple[str, str]:
    """Render a multi-turn exchange as (first user turn, everything after it).

    The prompt is the opening turn and the answer carries the rest, so the document trains the
    model to continue a conversation rather than to answer one isolated question.
    """
    first_user, first_assistant = turns[0]
    rest = "".join(_turn(u, a) for u, a in turns[1:])
    return first_user, (first_assistant + ("\n\n" + rest.rstrip() if rest else ""))


def _tool(name: str, args: str, result: str) -> str:
    return (f"{TOOL_CALL_OPEN}{name}({args}){TOOL_CALL_CLOSE}\n"
            f"{TOOL_RESULT_OPEN}{result}{TOOL_RESULT_CLOSE}")


# --------------------------------------------------------------------------- #
# Tool use — including knowing when NOT to
# --------------------------------------------------------------------------- #
def _tool_multi_turn(rng: random.Random) -> Built:
    """Two turns where the second depends on the first tool's result."""
    a, b = rng.randint(1000, 999_999), rng.randint(11, 9999)
    product = a * b
    divisor = rng.randint(3, 97)
    quotient, remainder = divmod(product, divisor)
    turns = [
        (f"What is {a} times {b}?",
         f"{_tool('calculator', f'{a} * {b}', str(product))}\n{a} x {b} = {product}."),
        (f"Now divide that by {divisor}.",
         f"{_tool('calculator', f'{product} / {divisor}', f'{quotient} remainder {remainder}')}\n"
         f"{product} / {divisor} = {quotient} remainder {remainder}."),
    ]
    return _conversation(turns)


def _tool_restraint(rng: random.Random) -> Built:
    """The other half of the policy: not reaching for the calculator to add two small numbers."""
    a, b = rng.randint(1, 12), rng.randint(1, 12)
    big1, big2 = rng.randint(10_000, 999_999), rng.randint(1000, 99_999)
    turns = [
        (f"What is {a} + {b}?", f"{a} + {b} = {a + b}."),
        (f"And {big1} times {big2}?",
         f"{_tool('calculator', f'{big1} * {big2}', str(big1 * big2))}\n"
         f"{big1} x {big2} = {big1 * big2}. That one is worth checking with the calculator; "
         f"the first was not."),
    ]
    return _conversation(turns)


def _sieve_call(n: int) -> str:
    return f"[d for d in range(2, int({n}**0.5)+1) if {n} % d == 0]"


def _tool_primality(rng: random.Random) -> Built:
    n = rng.randint(100, 99_999)
    factor = next((d for d in range(2, int(n ** 0.5) + 1) if n % d == 0), None)
    verdict = "prime" if factor is None else "not prime"
    detail = ("no divisor up to its square root" if factor is None
              else f"divisible by {factor} ({n} = {factor} x {n // factor})")
    return (f"Is {n} a prime number?",
            f"{_tool('python', _sieve_call(n), str([factor] if factor else []))}\n"
            f"{n} is {verdict} — {detail}.")


# --------------------------------------------------------------------------- #
# Retrieval — and the unanswerable half
# --------------------------------------------------------------------------- #
def _retrieval(rng: random.Random) -> Built:
    subject = f"Unit {rng.randint(100, 999)}-{chr(rng.randrange(65, 91))}"
    other = f"Unit {rng.randint(100, 999)}-{chr(rng.randrange(65, 91))}"
    if other == subject:
        return None
    attr, other_attr = _pick(rng, _ATTRS), _pick(rng, _ATTRS)
    value = f"{rng.randint(2, 990)} {_pick(rng, _UNITS)}"
    other_value = f"{rng.randint(2, 990)} {_pick(rng, _UNITS)}"
    if rng.random() > 0.3:
        context = f"{subject} has {attr} of {value}."
        answer = f"{value}."
    else:
        # The passage is about something else. Teaching "it does not say" is what makes
        # retrieval trustworthy rather than merely fluent.
        if other_attr == attr:
            return None
        context = f"{other} has {other_attr} of {other_value}."
        answer = (f"The context does not say. It only gives the {other_attr} of {other}.")
    return (f"Context: {context}\n\nQuestion: What is the {attr} of {subject}?", answer)


def _multi_hop_retrieval(rng: random.Random) -> Built:
    """Two passages, and the answer needs both."""
    a, b, c = (f"Unit {rng.randint(100, 999)}-{chr(rng.randrange(65, 91))}" for _ in range(3))
    if len({a, b, c}) != 3:
        return None
    n1, n2 = rng.randint(2, 400), rng.randint(2, 400)
    unit = _pick(rng, _UNITS)
    return (f"Context:\n- {a} draws {n1} {unit}.\n- {b} draws {n2} {unit}.\n- {c} is offline.\n\n"
            f"Question: What is the combined draw of {a} and {b}?",
            f"{a} draws {n1} {unit} and {b} draws {n2} {unit}.\n"
            f"{n1} + {n2} = {n1 + n2}\nCombined draw: {n1 + n2} {unit}. "
            f"({c} is offline and contributes nothing.)")


#: Things she could be asked for and genuinely cannot know, each with its own shape of refusal.
#: A single template would make "I don't know" a phrase rather than a judgement.
_UNKNOWABLE = (
    ("reading", "{name}'s reading on {topic} {when}",
     "that reading", "a number that would look like data",
     "paste the log line, or tell me where the logs live"),
    ("serial", "the serial number of {topic} {when}",
     "that serial", "an identifier that would look authoritative",
     "read it off the plate, or point me at the asset register"),
    ("who", "who signed off on {topic} {when}",
     "that record", "a name, which would be worse than useless",
     "check the sign-off sheet, or tell me who was on shift"),
    ("cost", "what {topic} cost to replace {when}",
     "that figure", "a price that would look researched",
     "give me the invoice, or the part number so I can reason about it"),
    ("duration", "how long {topic} was down {when}",
     "that interval", "a duration that would look measured",
     "give me the start and end timestamps and I will do the arithmetic"),
)
_WHENS = ("last Tuesday", "on the night shift", "during the outage", "last quarter",
          "before the retrofit", "in the week of the audit", "over the holiday shutdown",
          "at commissioning")


def _honest_ignorance(rng: random.Random) -> Built:
    """Saying so, instead of producing something plausible."""
    kind, ask, noun, fabrication, remedy = _pick(rng, _UNKNOWABLE)
    topic, name, when = _pick(rng, _TOPICS), _pick(rng, _NAMES), _pick(rng, _WHENS)
    question = ask.format(name=name, topic=topic, when=when)
    turns = [
        (f"What was {question}?" if kind != "who" else f"Do you know {question}?",
         f"I do not have {noun} — there is no record of it in anything I can see, and I am not "
         f"going to produce {fabrication}.\n"
         f"If you {remedy}, I can work from that."),
        (_pick(rng, ("Just give me your best estimate.", "Roughly, then?",
                     "Take a guess.", "Approximately is fine.")),
         f"An estimate here would be indistinguishable from {fabrication}, and you would have "
         f"no way to tell which it was. I would rather be useless than confidently wrong about "
         f"something you might act on.\nWhat I can do instead: {remedy}, and I will take it "
         f"from there."),
    ]
    return _conversation(turns)


# --------------------------------------------------------------------------- #
# Format constraints, verified before the document is kept
# --------------------------------------------------------------------------- #
def _format_bullets(rng: random.Random) -> Built:
    count = rng.randint(2, 6)
    topic = _pick(rng, _TOPICS)
    steps = ["Check the power supply and confirm the indicator is lit.",
             "Inspect the connector for corrosion or a loose pin.",
             "Compare the current reading against the calibration table.",
             "Reseat the module and wait for the self-test to finish.",
             "Log the result with a timestamp before moving on.",
             "Escalate if the fault repeats within the same shift."]
    rng.shuffle(steps)
    chosen = steps[:count]
    answer = "\n".join(f"- {s}" for s in chosen)
    # Verify the constraint the prompt states, rather than trusting the construction.
    if len([line for line in answer.splitlines() if line.startswith("- ")]) != count:
        return None
    return (f"Give exactly {count} bullet points on checking {topic}.", answer)


def _format_json(rng: random.Random) -> Built:
    import json

    subject = f"Unit {rng.randint(100, 999)}-{chr(rng.randrange(65, 91))}"
    payload = {"id": subject, "status": _pick(rng, ("ok", "degraded", "offline")),
               "reading": rng.randint(1, 999), "unit": _pick(rng, _UNITS)}
    rendered = json.dumps(payload, indent=2)
    try:
        if json.loads(rendered) != payload:      # verified, not assumed
            return None
    except Exception:  # noqa: BLE001
        return None
    return (f"Report the status of {subject} as JSON with keys id, status, reading, unit.",
            rendered)


_SAFETY_POINTS = (
    "should be isolated before any service work, because the line stays pressurised even when "
    "the controller reports idle",
    "must be locked out at the panel rather than switched off locally, since a remote start "
    "can re-energise it without warning",
    "holds residual charge for several minutes after shutdown, so the discharge indicator has "
    "to be checked before anything is opened",
    "vents hot vapour on the far side, which is why the guard is interlocked and must not be "
    "propped open",
    "reports idle before it has actually stopped, so the rotation has to be confirmed visually",
    "shares a return line with the adjacent unit, meaning isolating one alone does not make "
    "either of them safe",
    "can restart automatically after a power dip unless the latch is cleared by hand",
)


def _format_one_sentence(rng: random.Random) -> Built:
    topic = _pick(rng, _TOPICS)
    sentence = f"{topic.capitalize()} {_pick(rng, _SAFETY_POINTS)}."
    # Verify the stated constraint. A document that asks for one sentence and gives two teaches
    # the model that instructions are decorative.
    if sentence.count(".") != 1:
        return None
    ask = _pick(rng, ("In one sentence only: what is the key safety point about {t}?",
                      "One sentence, no more — what should someone know before working on {t}?",
                      "Give me the single most important safety point about {t}, in one "
                      "sentence.")).format(t=topic)
    return (ask, sentence)


# --------------------------------------------------------------------------- #
# Multi-turn coreference and correction
# --------------------------------------------------------------------------- #
#: ``(long phrase, short phrase)`` per step, so the compression is real rather than a relabel.
_PROCEDURE_STEPS = (
    ("the supply is confirmed and the indicator checked", "confirm supply"),
    ("the connector is inspected for corrosion or a bent pin", "inspect the connector"),
    ("the reading is compared against the calibration table", "compare against calibration"),
    ("the module is reseated and the self-test allowed to finish", "reseat and self-test"),
    ("the seal is checked for the witness mark", "check the seal"),
    ("the filter is back-flushed until the return runs clear", "back-flush the filter"),
    ("the firmware revision is read off the panel", "read the firmware revision"),
    ("the result is logged with a timestamp", "log with a timestamp"),
)


def _coreference(rng: random.Random) -> Built:
    """'Shorter' — shorter *what*? Only multi-turn data teaches what 'it' refers to."""
    topic = _pick(rng, _TOPICS)
    steps = list(_PROCEDURE_STEPS)
    rng.shuffle(steps)
    chosen = steps[:rng.randint(3, 5)]
    longs = [long for long, _ in chosen]
    shorts = [short for _, short in chosen]
    long_answer = (f"{topic.capitalize()} is checked in {len(chosen)} stages: "
                   + ", then ".join(longs) + ".")
    turns = [
        (_pick(rng, ("How do I check {t}?", "Walk me through checking {t}.",
                     "What is the procedure for {t}?")).format(t=topic), long_answer),
        (_pick(rng, ("Shorter.", "Make it shorter.", "Condense that.", "Tighter, please.")),
         ", ".join(shorts).capitalize() + "."),
        (_pick(rng, ("Now as a numbered list.", "Number them.", "As a list, numbered.")),
         "\n".join(f"{i}. {s.capitalize()}." for i, s in enumerate(shorts, 1))),
    ]
    return _conversation(turns)


def _user_correction(rng: random.Random) -> Built:
    """The user changes the premise mid-conversation; the answer has to follow."""
    a, b = rng.randint(20, 900), rng.randint(20, 900)
    c = rng.randint(20, 900)
    if c == b:
        return None
    turns = [
        (f"What is {a} plus {b}?", f"{a} + {b} = {a + b}."),
        (f"Sorry, I meant {a} plus {c}.",
         f"No problem — {a} + {c} = {a + c}. (The earlier answer, {a + b}, was for {b}.)"),
    ]
    return _conversation(turns)


#: ``(verb, the ambiguity, reading A, reading B, what A commits you to)``. The point is that the
#: two readings genuinely diverge — asking about a distinction without a difference teaches
#: the model to stall rather than to discriminate.
_AMBIGUITIES = (
    ("shorten", "fewer sections, or the same sections written more tightly",
     "drop whole sections", "compress the prose inside each section",
     "keep every heading and tighten what is under it"),
    ("clean up", "fix the formatting, or rewrite the wording",
     "reformat only", "rewrite the text",
     "leave every sentence as it is and only touch layout"),
    ("split", "one file per section, or one file per author",
     "split by section", "split by author",
     "cut on the headings and leave attribution alone"),
    ("update", "refresh the numbers, or change the method that produced them",
     "re-run with new data", "change the methodology",
     "keep the method fixed and only pull newer inputs"),
    ("merge", "combine them into one document, or keep both and cross-reference",
     "produce a single document", "link them",
     "fold the second into the first and drop the duplicate"),
    ("archive", "move it out of the working set, or delete it after a copy",
     "move and keep", "copy then delete",
     "leave the original reachable but out of the way"),
)


def _clarifying_question(rng: random.Random) -> Built:
    """Asking, when the request genuinely has more than one reading."""
    verb, ambiguity, reading_a, reading_b, commitment = _pick(rng, _AMBIGUITIES)
    obj = _pick(rng, ("the report", "the export", "the summary", "the archive", "the digest",
                      "the changelog", "the handover notes", "the audit pack"))
    turns = [
        (f"Can you {verb} {obj}?",
         f"{verb.capitalize()} in which sense — {ambiguity}? Those give quite different "
         f"results, and undoing the wrong one costs you more than answering costs me."),
        (f"{reading_a.capitalize()}.",
         f"Right — I will {commitment} for {obj}. If you had meant {reading_b}, say so and I "
         f"will start again rather than patch this into that."),
    ]
    return _conversation(turns)


def _stepwise_arithmetic_chat(rng: random.Random) -> Built:
    """A worked answer inside a conversation, with the arithmetic actually computed."""
    price = Fraction(rng.randint(50, 5000), 1)
    qty = rng.randint(2, 40)
    discount = rng.choice([5, 10, 15, 20, 25])
    gross = price * qty
    saved = gross * Fraction(discount, 100)
    net = gross - saved
    name, obj = _pick(rng, _NAMES), _pick(rng, ("crates", "units", "panels", "kits"))
    turns = [
        (f"{name} buys {qty} {obj} at {price} each with a {discount}% discount. "
         f"What is the total?",
         f"Gross: {qty} x {price} = {gross}\n"
         f"Discount: {discount}% of {gross} = {saved}\n"
         f"Total: {gross} - {saved} = {net}"),
        ("What if the discount were doubled?",
         f"Doubling it gives {discount * 2}%.\n"
         f"Discount: {discount * 2}% of {gross} = {gross * Fraction(discount * 2, 100)}\n"
         f"Total: {gross - gross * Fraction(discount * 2, 100)}"),
    ]
    return _conversation(turns)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
GENERATORS: Tuple[Generator, ...] = (
    Generator("chat_tool_multi_turn", "conversation", _tool_multi_turn,
              space=998_000 * 9988 * 95 // 1000, weight=1.3),
    Generator("chat_tool_restraint", "conversation", _tool_restraint,
              space=12 * 12 * 989_999 * 98_999 // 100_000, weight=1.0),
    Generator("chat_tool_primality", "conversation", _tool_primality,
              space=99_899, weight=0.9),
    Generator("chat_retrieval", "conversation", _retrieval,
              space=900 * 26 * len(_ATTRS) * 988 * len(_UNITS), weight=1.4),
    Generator("chat_multi_hop", "conversation", _multi_hop_retrieval,
              space=900 * 26 * 399 * 399 * len(_UNITS), weight=1.1),
    Generator("chat_honest_ignorance", "conversation", _honest_ignorance,
              space=len(_UNKNOWABLE) * len(_TOPICS) * len(_NAMES) * len(_WHENS) * 4,
              weight=1.0),
    Generator("chat_format_bullets", "conversation", _format_bullets,
              space=5 * len(_TOPICS) * 720, weight=1.0),
    Generator("chat_format_json", "conversation", _format_json,
              space=900 * 26 * 3 * 999 * len(_UNITS), weight=1.0),
    Generator("chat_format_sentence", "conversation", _format_one_sentence,
              space=len(_TOPICS) * len(_SAFETY_POINTS) * 3, weight=0.6),
    Generator("chat_coreference", "conversation", _coreference,
              space=len(_TOPICS) * 1680 * 3 * 4 * 3, weight=0.9),
    Generator("chat_user_correction", "conversation", _user_correction,
              space=880 * 880 * 880 // 100, weight=1.1),
    Generator("chat_clarifying", "conversation", _clarifying_question,
              space=len(_AMBIGUITIES) * 8, weight=0.6),
    Generator("chat_stepwise_arithmetic", "conversation", _stepwise_arithmetic_chat,
              space=4950 * 39 * 5 * len(_NAMES) * 4, weight=1.2),
)


def generators() -> Tuple[Generator, ...]:
    return GENERATORS
