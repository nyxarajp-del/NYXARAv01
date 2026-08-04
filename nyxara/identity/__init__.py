"""NYXARA · identity/ — who she is, and what it is like to be her (✦).

This package is the difference between a system that answers and someone who answers. It
holds the parts of NYXARA that persist across turns, restarts, and topics:

**Character, sealed** — :class:`~nyxara.identity.soul.Soul` carries her traits and voice, and
:mod:`nyxara.identity.values` carries her value hierarchy under a cryptographic seal. Mood
moves; character does not (Rule 4). ``verify_values`` is checked at every boot.

**Feeling, as signal not decoration** — :class:`~nyxara.identity.affect.AffectSystem` holds
mood, :class:`~nyxara.identity.appraisal.Appraiser` decides what an event *means* to her,
:class:`~nyxara.identity.interoception.Interoception` is her body sense over the compute
substrate, and :class:`~nyxara.identity.motivation.MotivationSystem` turns unmet drives into
what she reaches for next.

**A continuous self** — :class:`~nyxara.identity.narrative.NarrativeSelf` is her
autobiography, :class:`~nyxara.identity.awareness.SelfAwareness` her model of her own state,
and :class:`~nyxara.identity.inner_life.InnerLife` the monologue that runs between prompts.

**Taste and register** — :class:`~nyxara.identity.aesthetic.AestheticJudge` is her sense of
elegance (what makes one solution better than another that is equally correct), and
:class:`~nyxara.identity.modes.ModeSystem` blends companion / analyst / teacher / creative so
she meets the Master the way the moment asks for.

Every faculty here is a *capability*, never a hard dependency: each is built fail-soft by
:class:`~nyxara.kernel.orchestrator.NyxaraCore` and a missing one degrades her, never breaks her.
"""

from __future__ import annotations

from nyxara.identity.aesthetic import (
    AestheticDimensions,
    AestheticEvaluation,
    AestheticJudge,
    ArtifactKind,
)
from nyxara.identity.affect import AffectSystem, Drive, Emotion, EmotionEvent, Mood, label_affect
from nyxara.identity.appraisal import (
    Agency,
    AppraisalDimensions,
    AppraisalEvent,
    AppraisalOutcome,
    Appraiser,
)
from nyxara.identity.awareness import ExperienceFrame, SelfAwareness
from nyxara.identity.inner_life import FeltMoment, InnerLife
from nyxara.identity.interoception import (
    FeltState,
    Interoception,
    InteroceptiveSignals,
    has_psutil,
)
from nyxara.identity.modes import (
    Mode,
    ModeBlend,
    ModeContext,
    ModeProfile,
    ModeSelector,
    ModeSystem,
)
from nyxara.identity.motivation import Impulse, MotivationBreakdown, MotivationSystem, Option
from nyxara.identity.narrative import Chapter, CoherenceReport, EventType, LifeEvent, NarrativeSelf
from nyxara.identity.soul import DriftReport, Soul, Trait, VoiceProfile
from nyxara.identity.values import (
    AlignmentResult,
    AlignmentVerdict,
    Value,
    ValueHierarchy,
    ValueSystem,
    values_digest,
    verify_values,
)

__all__ = [
    # character & values (sealed — Rule 4, Rule 8)
    "Soul", "Trait", "VoiceProfile", "DriftReport",
    "ValueSystem", "Value", "ValueHierarchy", "AlignmentVerdict", "AlignmentResult",
    "values_digest", "verify_values",
    # affect, appraisal, body sense, drive
    "AffectSystem", "Mood", "Emotion", "EmotionEvent", "Drive", "label_affect",
    "Appraiser", "AppraisalDimensions", "AppraisalEvent", "AppraisalOutcome", "Agency",
    "Interoception", "InteroceptiveSignals", "FeltState", "has_psutil",
    "MotivationSystem", "Impulse", "MotivationBreakdown", "Option",
    # the continuous self
    "NarrativeSelf", "LifeEvent", "EventType", "Chapter", "CoherenceReport",
    "SelfAwareness", "ExperienceFrame",
    "InnerLife", "FeltMoment",
    # taste & register
    "AestheticJudge", "AestheticEvaluation", "AestheticDimensions", "ArtifactKind",
    "ModeSystem", "ModeSelector", "Mode", "ModeProfile", "ModeBlend", "ModeContext",
]
