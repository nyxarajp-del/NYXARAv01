"""NYXARA · social — the social-cognition faculties (theory of mind & dialogue).

A solitary reasoner can be brilliant and still socially blind. NYXARA models the *people*
in the loop, not just the task: what the Master knows and wants, how they feel, what was
left unsaid, and how to keep a conversation on the rails when it slips.

This package re-exports the stable surface of those faculties:

* :mod:`tom` — :class:`TheoryOfMind`: a running model of each interlocutor's beliefs,
  desires and intentions (the Master is modelled from the first turn).
* :mod:`empathy` — :class:`EmpathySystem`: infers affect from what is said and shapes an
  emotionally-appropriate stance.
* :mod:`dialogue` — speech acts, Gricean :class:`ImplicatureEngine` (what is *meant* beyond
  what is *said*), :class:`Conversation` state and the :class:`DialogueManager`.
* :mod:`persons` — a :class:`Roster` of :class:`Person` records: trust, communication style
  and interaction history that accrue over time.
* :mod:`common_ground` — :class:`CommonGround`: the shared, mutually-known facts that make
  reference and ellipsis resolvable.
* :mod:`culture` — :class:`CultureSystem`: register, code-switching (e.g. Hindi/English) and
  style guidance for the Master.
* :mod:`intent_repair` — :class:`RepairManager`: detects conversational trouble
  (misunderstanding, non-understanding) and proposes repairs.

Modules remain importable by full path; this is the curated public surface.
"""

from __future__ import annotations

from nyxara.social.common_ground import (
    CommonGround,
    Entity,
    GroundStatus,
    Proposition,
    ReferenceForm,
)
from nyxara.social.culture import (
    CodeProfile,
    CultureSystem,
    Register,
    StyleGuidance,
)
from nyxara.social.dialogue import (
    Conversation,
    DialogueManager,
    Implicature,
    ImplicatureEngine,
    SpeechAct,
    Subtext,
    SubtextDetector,
    Utterance,
    detect_speech_act,
)
from nyxara.social.empathy import (
    EmpathyResponse,
    EmpathySystem,
    InferredEmotion,
)
from nyxara.social.intent_repair import (
    RepairAction,
    RepairDetector,
    RepairManager,
    RepairStrategist,
    RepairType,
    TroubleSignal,
    TroubleType,
)
from nyxara.social.persons import (
    CommStyle,
    Interaction,
    InteractionKind,
    Person,
    Roster,
    TrustLevel,
)
from nyxara.social.tom import MentalState, TheoryOfMind

__all__ = [
    # theory of mind
    "TheoryOfMind",
    "MentalState",
    # empathy
    "EmpathySystem",
    "EmpathyResponse",
    "InferredEmotion",
    # dialogue
    "DialogueManager",
    "Conversation",
    "Utterance",
    "SpeechAct",
    "detect_speech_act",
    "ImplicatureEngine",
    "Implicature",
    "Subtext",
    "SubtextDetector",
    # persons
    "Roster",
    "Person",
    "CommStyle",
    "Interaction",
    "InteractionKind",
    "TrustLevel",
    # common ground
    "CommonGround",
    "Proposition",
    "Entity",
    "GroundStatus",
    "ReferenceForm",
    # culture
    "CultureSystem",
    "Register",
    "CodeProfile",
    "StyleGuidance",
    # intent repair
    "RepairManager",
    "RepairDetector",
    "RepairStrategist",
    "RepairAction",
    "RepairType",
    "TroubleSignal",
    "TroubleType",
]
