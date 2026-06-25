"""NYXARA · cognition/language_grounding.py — learn dynamics from *language* (✶).

Until now NYXARA could only learn how the world behaves by being **fed** numeric
reinforcement-learning transitions ``(state, action, next_state, reward)`` — from the real
environment or a simulator. She could *read* text (RAG, sentiment, key-phrases) but reading
never **taught the world model anything**: a textbook paragraph could not become a learned
model. This module closes that gap.

:class:`LanguageGrounder` turns natural-language descriptions of how a world behaves —
"heating the water raises its temperature from 20 to 80 degrees" — into the very same
``Transition`` records the RL machinery already consumes, and feeds them to a
:class:`~nyxara.mind.world_model.WorldModel` via ``observe_many``. Because the default
``TransferWorldModel`` updates a :class:`~nyxara.mind.concept_hierarchy.ConceptHierarchy`
on every numeric transition, **reading text now forms concepts and trains dynamics** — no
hand-fed tuples required. Read two passages that both describe "heating raises temperature",
ask the model to predict, and it imagines the temperature going *up*.

Two extractors, one numeric space (NYXARA's floor-plus-ceiling philosophy):

* **deterministic** (always available, pure stdlib) — parses each sentence with the existing
  :mod:`nyxara.senses.nlp` toolkit: the causal/gerund **verb** becomes the action; explicit
  ranges ("from 20 to 80"), direction verbs ("raises"/"lowers") and comparative adjectives
  ("warmer"/"colder") become a signed Δ on a measured variable; sentence **sentiment** becomes
  the reward. This is the floor that never fails.
* **LLM-assisted** (optional ceiling) — when a *real* (non-mock) provider is configured, an
  :class:`~nyxara.mind.llm.LLM` is asked for structured JSON transitions, which flow through
  the *same* :class:`VariableRegistry`. Any failure silently degrades to the deterministic
  extractor; the LLM is never required.

The load-bearing invariant: ``TransferWorldModel.observe`` only updates the concept hierarchy
on the **numeric, fixed-width** path; symbolic or ragged-width transitions fall back to an
exact-memory kNN that forms *no* concepts. So every vector this module emits is a fixed-width,
all-float row produced by a stable variable→index :class:`VariableRegistry` (with reserved
capacity), so "temperature" is always the same dimension across sentences *and* across calls.

Pairs with :mod:`mind.world_model`, :mod:`mind.concept_hierarchy`, :mod:`senses.nlp`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.mind.world_model import Transition
from nyxara.senses import nlp

__all__ = [
    "VariableRegistry",
    "ExtractedChange",
    "LanguageGrounder",
    "extract_transitions_deterministic",
]


# --------------------------------------------------------------------------- #
# Lexicons — direction of change, and comparative adjectives → canonical vars
# --------------------------------------------------------------------------- #
# Verbs that *raise* a measured quantity (act on an explicit noun in the sentence).
_INCREASE_VERBS = frozenset("""
raise raises raised raising increase increases increased increasing rise rises risen rising
grow grows grew growing gain gains gained gaining add adds added adding boost boosts boosted
heat heats heated heating warm warms warmed warming lift lifts lifted lifting climb climbs
accelerate accelerates accelerated accelerating expand expands expanded expanding
""".split())

_DECREASE_VERBS = frozenset("""
lower lowers lowered lowering decrease decreases decreased decreasing fall falls fell falling
drop drops dropped dropping shrink shrinks shrank shrinking lose loses lost losing remove
removes removed removing reduce reduces reduced reducing cool cools cooled cooling chill chills
chilled chilling sink sinks sank sinking decelerate decelerates slow slows slowed slowing
contract contracts contracted contracting
""".split())

# Comparative adjectives map directly to a canonical variable and a sign — they carry both
# *what* changed and *which way*, even when the sentence contains no number at all.
_COMPARATIVE: Dict[str, Tuple[str, float]] = {
    "warmer": ("temperature", +1.0), "hotter": ("temperature", +1.0),
    "colder": ("temperature", -1.0), "cooler": ("temperature", -1.0),
    "faster": ("speed", +1.0), "quicker": ("speed", +1.0), "slower": ("speed", -1.0),
    "brighter": ("brightness", +1.0), "dimmer": ("brightness", -1.0), "darker": ("brightness", -1.0),
    "louder": ("volume", +1.0), "quieter": ("volume", -1.0), "softer": ("volume", -1.0),
    "bigger": ("size", +1.0), "larger": ("size", +1.0), "smaller": ("size", -1.0),
    "heavier": ("weight", +1.0), "lighter": ("weight", -1.0),
    "higher": ("level", +1.0), "taller": ("level", +1.0), "lower": ("level", -1.0),
    "wetter": ("moisture", +1.0), "drier": ("moisture", -1.0), "dryer": ("moisture", -1.0),
    "fuller": ("fullness", +1.0), "emptier": ("fullness", -1.0),
    "happier": ("happiness", +1.0), "sadder": ("happiness", -1.0),
    "richer": ("wealth", +1.0), "poorer": ("wealth", -1.0),
    "stronger": ("strength", +1.0), "weaker": ("strength", -1.0),
    "closer": ("distance", -1.0), "nearer": ("distance", -1.0), "farther": ("distance", +1.0),
    "further": ("distance", +1.0),
}

# Light irregular-verb normalisation so "rose"/"rises"/"rising" share the action "rise".
_IRREGULAR = {
    "rose": "rise", "risen": "rise", "fell": "fall", "fallen": "fall",
    "grew": "grow", "grown": "grow", "shrank": "shrink", "sank": "sink",
    "lost": "lose", "drove": "drive",
}

_NUMRANGE_RE = re.compile(r"from\s+(-?\d[\d,]*(?:\.\d+)?)\s+to\s+(-?\d[\d,]*(?:\.\d+)?)", re.I)
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _to_float(tok: str) -> float:
    return float(tok.replace(",", ""))


def _normalize_verb(word: str) -> str:
    """Crudely lemmatise an English verb to a stable action key (stdlib only)."""
    w = word.lower()
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            stem = w[: -len(suf)]
            # de-double a trailing consonant ("running" -> "run", not "runn")
            if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
                stem = stem[:-1]
            # restore a dropped 'e' ("raising" -> "rais" -> "raise")
            if suf == "ing" and stem and stem[-1] not in "aeiou" and not stem.endswith("e"):
                # only when it yields a known increase/decrease stem, to stay conservative
                if (stem + "e") in {"raise", "rise", "decrease", "increase"}:
                    stem += "e"
            return stem
    return w


# --------------------------------------------------------------------------- #
# VariableRegistry — a stable, fixed-width variable → dimension map
# --------------------------------------------------------------------------- #
class VariableRegistry:
    """Maps named state variables to **stable** vector dimensions.

    Every vector is emitted at a fixed ``capacity`` width (unused dims = 0.0) so the world
    model's numeric path — the only one that forms concepts — stays engaged across sentences
    *and* across separate :meth:`LanguageGrounder.learn` calls. The first variable seen claims
    dim 0, the next claims dim 1, and so on; the assignment never changes once made.
    """

    def __init__(self, *, capacity: int = 32) -> None:
        self.capacity = max(1, capacity)
        self._index: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._index)

    def names(self) -> List[str]:
        return sorted(self._index, key=lambda n: self._index[n])

    def index_of(self, name: str) -> int:
        key = name.strip().lower()
        if key not in self._index:
            slot = len(self._index)
            if slot >= self.capacity:                 # rare: grow rather than lose a variable
                self.capacity = slot + 1
            self._index[key] = slot
        return self._index[key]

    def width(self) -> int:
        return self.capacity

    def vectorize(self, values: Dict[str, float]) -> Tuple[float, ...]:
        """Build a fixed-width all-float row; named values placed at their stable dim."""
        for name in values:                            # register first so width is final
            self.index_of(name)
        row = [0.0] * self.capacity
        for name, val in values.items():
            row[self._index[name.strip().lower()]] = float(val)
        return tuple(row)


# --------------------------------------------------------------------------- #
# ExtractedChange — a parsed (action, before, after, reward) before vectorising
# --------------------------------------------------------------------------- #
@dataclass
class ExtractedChange:
    action: str
    variable: str
    before: float
    after: float
    reward: float = 0.0
    source: str = ""                                   # the sentence it came from

    @property
    def delta(self) -> float:
        return self.after - self.before


# --------------------------------------------------------------------------- #
# Deterministic extractor — the always-available floor
# --------------------------------------------------------------------------- #
def _find_action(tokens_lower: Sequence[str]) -> Optional[str]:
    """The causal verb of the sentence becomes the action key.

    Prefer a leading gerund ("Heating the water …" → ``heat``); otherwise the first
    increase/decrease verb; otherwise the first plausible verb-like content word.
    """
    # leading gerund / agentive process
    for t in tokens_lower:
        if t.endswith("ing") and t not in nlp.STOPWORDS and len(t) > 4:
            return _normalize_verb(t)
    # any directional verb
    for t in tokens_lower:
        if t in _INCREASE_VERBS or t in _DECREASE_VERBS:
            return _normalize_verb(t)
    return None


def _measured_variable(tokens_lower: Sequence[str], around: int) -> Optional[str]:
    """The noun a direction verb acts on — the first content word after the verb."""
    for t in tokens_lower[around + 1:]:
        if t not in nlp.STOPWORDS and len(t) > 2 and _NUMBER_RE.fullmatch(t) is None \
                and t not in _INCREASE_VERBS and t not in _DECREASE_VERBS:
            return t
    return None


def _extract_sentence(sentence: str) -> List[ExtractedChange]:
    """Parse ONE sentence into zero or more (action, variable, before→after, reward) changes."""
    toks = nlp.tokenize(sentence, lower=True)
    if not toks:
        return []
    reward = nlp.sentiment(sentence).polarity
    action = _find_action(toks)
    changes: List[ExtractedChange] = []

    # 1) explicit numeric range:  "<var> ... from 20 to 80"
    for m in _NUMRANGE_RE.finditer(sentence):
        before, after = _to_float(m.group(1)), _to_float(m.group(2))
        # the variable is the nearest content noun before "from"
        prefix = nlp.tokenize(sentence[: m.start()].lower())
        var = None
        for t in reversed(prefix):
            if t not in nlp.STOPWORDS and len(t) > 2 and _NUMBER_RE.fullmatch(t) is None:
                var = t
                break
        var = var or "value"
        act = action or ("raise" if after >= before else "lower")
        changes.append(ExtractedChange(act, var, before, after, reward, sentence))
    if changes:
        return changes

    # 2) comparative adjective:  "the room felt warmer" → +1 on temperature
    for t in toks:
        if t in _COMPARATIVE:
            var, sign = _COMPARATIVE[t]
            act = action or ("raise" if sign > 0 else "lower")
            changes.append(ExtractedChange(act, var, 0.0, sign, reward, sentence))
    if changes:
        return changes

    # 3) direction verb on a named variable:  "heating raises the temperature"
    for i, t in enumerate(toks):
        sign = 1.0 if t in _INCREASE_VERBS else (-1.0 if t in _DECREASE_VERBS else 0.0)
        if sign == 0.0:
            continue
        var = _measured_variable(toks, i)
        if var is None:
            continue
        act = action or _normalize_verb(t)
        changes.append(ExtractedChange(act, var, 0.0, sign, reward, sentence))
        break
    return changes


def extract_transitions_deterministic(
    text: str, registry: VariableRegistry
) -> List[Transition]:
    """Parse ``text`` into fixed-width numeric :class:`Transition` records (pure stdlib).

    All changes are collected first so the registry is fully populated, then every transition
    is vectorised at the registry's final width — guaranteeing batch-consistent dimensions.
    """
    changes: List[ExtractedChange] = []
    for sentence in nlp.sentences(text) or [text]:
        changes.extend(_extract_sentence(sentence))
    # populate the registry up-front so all rows share one width
    for ch in changes:
        registry.index_of(ch.variable)
    transitions: List[Transition] = []
    for ch in changes:
        before = registry.vectorize({ch.variable: ch.before})
        after = registry.vectorize({ch.variable: ch.after})
        transitions.append(Transition(before, ch.action, after, float(ch.reward)))
    return transitions


# --------------------------------------------------------------------------- #
# LLM-assisted extractor — the optional ceiling (degrades to the floor)
# --------------------------------------------------------------------------- #
_LLM_SYSTEM = (
    "You convert a passage describing how a world behaves into structured state transitions. "
    "Reply ONLY with JSON of the form "
    '{"transitions": [{"action": str, "before": {var: number}, "after": {var: number}, '
    '"reward": number}]}. Use consistent variable names across transitions. '
    "Numbers only in before/after; reward in [-1,1] (positive = good outcome). "
    "If nothing is described, return an empty list."
)


def extract_transitions_llm(
    text: str, registry: VariableRegistry, llm: Any
) -> List[Transition]:
    """Ask a real LLM for structured transitions, mapped through the shared registry.

    Returns ``[]`` on any failure so the caller can fall back to the deterministic floor.
    """
    try:
        data = llm.generate_json(
            f"Passage:\n{text}\n\nExtract the state transitions as JSON.",
            system=_LLM_SYSTEM,
        )
    except Exception:                                  # noqa: BLE001 — ceiling never required
        return []
    rows = (data or {}).get("transitions", []) if isinstance(data, dict) else []
    parsed: List[Tuple[str, Dict[str, float], Dict[str, float], float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        action = str(row.get("action") or "act")
        before = {str(k): float(v) for k, v in (row.get("before") or {}).items()
                  if _is_number(v)}
        after = {str(k): float(v) for k, v in (row.get("after") or {}).items()
                 if _is_number(v)}
        reward = float(row.get("reward") or 0.0) if _is_number(row.get("reward")) else 0.0
        if not before and not after:
            continue
        parsed.append((action, before, after, reward))
        for name in (*before, *after):
            registry.index_of(name)
    out: List[Transition] = []
    for action, before, after, reward in parsed:
        merged_after = {**before, **after}             # unchanged vars keep their value
        out.append(Transition(registry.vectorize(before),
                              action, registry.vectorize(merged_after), reward))
    return out


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# --------------------------------------------------------------------------- #
# LanguageGrounder — orchestrates extraction → learning
# --------------------------------------------------------------------------- #
class LanguageGrounder:
    """Read language, teach a world model. The bridge from prose to learned dynamics."""

    def __init__(self, *, registry: Optional[VariableRegistry] = None,
                 llm: Any = None, capacity: int = 32) -> None:
        self.registry = registry or VariableRegistry(capacity=capacity)
        self.llm = llm
        self.total_read = 0

    def _use_llm(self) -> bool:
        """Only use the LLM ceiling when a *real* (non-mock) provider is configured."""
        if self.llm is None:
            return False
        try:
            return self.llm.chosen_provider().name != "mock"
        except Exception:                              # noqa: BLE001
            return False

    def extract(self, text: str) -> List[Transition]:
        """Extract transitions from ``text`` — LLM ceiling if available, else the floor."""
        if self._use_llm():
            llm_rows = extract_transitions_llm(text, self.registry, self.llm)
            if llm_rows:
                return llm_rows
        return extract_transitions_deterministic(text, self.registry)

    def learn(self, text: str, world_model: Any) -> Dict[str, Any]:
        """Extract transitions from ``text`` and teach ``world_model`` from them.

        Because the default :class:`TransferWorldModel` updates its concept hierarchy on every
        numeric transition, this is how reading a passage *forms concepts and trains dynamics*.
        """
        transitions = self.extract(text)
        before_n = len(world_model) if world_model is not None else 0
        if world_model is not None and transitions:
            world_model.observe_many(transitions)
        report: Dict[str, Any] = {
            "transitions": len(transitions),
            "actions": sorted({t.action for t in transitions}),
            "variables": self.registry.names(),
            "world_transitions": (len(world_model) if world_model is not None else 0),
            "learned": (len(world_model) - before_n) if world_model is not None else 0,
            "via": "llm" if self._use_llm() else "deterministic",
        }
        if hasattr(world_model, "concept_report"):
            try:
                report["concepts"] = world_model.concept_report()
            except Exception:                          # noqa: BLE001
                pass
        self.total_read += 1
        return report


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.mind.world_model import build_world_model

    print("=" * 70)
    print("NYXARA language-grounding self-test")
    print("=" * 70)

    grounder = LanguageGrounder()
    wm = build_world_model("auto")

    textbook = (
        "Heating the water raises its temperature from 20 to 80 degrees, which is good. "
        "Heating the metal raises its temperature from 10 to 60 degrees. "
        "Cooling the water lowers its temperature from 80 to 30 degrees; it feels worse."
    )
    rep = grounder.learn(textbook, wm)
    print(f"\nread textbook → {rep['transitions']} transitions, "
          f"vars={rep['variables']}, actions={rep['actions']}")
    assert rep["transitions"] >= 3, "should extract a transition per numeric sentence"
    assert "temperature" in rep["variables"]

    # NYXARA now *predicts* the described dynamics — with no hand-fed numeric tuples.
    dim = grounder.registry.index_of("temperature")
    state = grounder.registry.vectorize({"temperature": 20.0})
    pred = wm.predict(state, "heat")
    print(f"predict(temp=20, 'heat') → temp' = {pred.next_state[dim]:.2f} "
          f"(conf={pred.confidence:.2f})")
    assert pred.next_state[dim] > state[dim], "learned from prose: heating must raise temp"
    print("\nlearned dynamics from LANGUAGE alone — heating raises temperature  ✓")
    print("\nALL SELF-TESTS PASSED ✓")
