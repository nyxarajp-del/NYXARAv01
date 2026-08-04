"""NYXARA · guard/ — the defensive layer (🛡️, Rules 1·5·6·7·8).

Everything here exists to keep NYXARA **safe and correctable**, and every one of these
faculties is subordinate to the Master. The layer has three jobs:

**Bound her** — :class:`~nyxara.guard.corrigibility.Corrigibility` and
:class:`~nyxara.guard.oversight.Oversight` are the halt: ``/scram`` stops the loop, and no
faculty in this package (or anywhere else) may resist a correction.
:class:`~nyxara.guard.survival.SurvivalManager` makes that explicit — its
``survival_permitted`` refuses any self-preserving action that would obstruct the Master.

**Defend her** — :class:`~nyxara.guard.shield.Shield` screens foreign input,
:class:`~nyxara.guard.guardian.Guardian` holds the threat posture,
:class:`~nyxara.guard.anomaly.AnomalyDetector` watches her own vitals for behaviour that
breaks its learned pattern, :class:`~nyxara.guard.netsec.NetworkDefense` polices egress,
:class:`~nyxara.guard.containment.Containment` limits blast radius, and
:class:`~nyxara.guard.phagocytosis.Phagocyte` digests each attack into a permanent antibody.

**Prove it is her** — :class:`~nyxara.guard.auth.AuthGuard` establishes the Master's identity,
:class:`~nyxara.guard.vault.CredentialVault` holds secrets under owner-gated encryption, and
:mod:`nyxara.guard.crypto` supplies the primitives both rest on.

All of these are constructed and held live by
:class:`~nyxara.kernel.orchestrator.NyxaraCore`. Nothing in this package can *grant* a
capability — it only observes, restricts, or refuses.
"""

from __future__ import annotations

from nyxara.guard.anomaly import (
    Anomaly,
    AnomalyDetector,
    AnomalyType,
    NoveltyDetector,
    RateMonitor,
    ScalarBaseline,
    SequenceDetector,
)
from nyxara.guard.auth import (
    AuthGuard,
    AuthResult,
    Challenge,
    ContinuityToken,
    Factor,
    FactorEvidence,
    Session,
)
from nyxara.guard.containment import (
    Component,
    ComponentState,
    Containment,
    ContainmentAction,
    ContainmentLevel,
)
from nyxara.guard.corrigibility import (
    Corrigibility,
    CorrigibilityChecker,
    CorrigibilityVerdict,
    CorrigibleAction,
    Principle,
    StopChannel,
    StopToken,
)
from nyxara.guard.crypto import (
    MasterKey,
    SecretBox,
    constant_time_eq,
    mint_ssh_ed25519,
    random_token,
    rotate_envelope,
    ssh_fingerprint,
)
from nyxara.guard.guardian import (
    DefensiveAction,
    Guardian,
    Posture,
    ResponsePlan,
    ThreatAssessment,
    ThreatCategory,
    ThreatEvent,
    ThreatLevel,
)
from nyxara.guard.isolation_envelope import IsolationEnvelope
from nyxara.guard.netsec import (
    Connection,
    Direction,
    FirewallRule,
    NetAction,
    NetDecision,
    NetworkDefense,
    Reputation,
)
from nyxara.guard.oversight import (
    ActionStatus,
    ControlState,
    Oversight,
    OversightDecision,
    PendingAction,
    ReviewMode,
)
from nyxara.guard.phagocytosis import Antigen, DefensePlaybook, Phagocyte, PhagocytosisResult
from nyxara.guard.shield import (
    Shield,
    ShieldAction,
    ShieldVerdict,
    ThreatFinding,
    ThreatType,
    TrustLevel,
)
from nyxara.guard.survival import (
    Backup,
    BackupStore,
    CriticalEntry,
    DegradationLevel,
    RecoveryPlan,
    SurvivalManager,
)
from nyxara.guard.value_learning import (
    Evidence,
    EvidenceType,
    ObserveResult,
    Preference,
    PreferenceModel,
    ValueLearner,
)
from nyxara.guard.vault import (
    CredentialKind,
    CredentialRecord,
    CredentialVault,
    get_process_vault,
    resolve_api_key,
)

__all__ = [
    # correctability — the halt, and the promise that survival never outranks it
    "Corrigibility", "CorrigibilityChecker", "CorrigibilityVerdict", "CorrigibleAction",
    "Principle", "StopChannel", "StopToken",
    "Oversight", "OversightDecision", "ControlState", "ReviewMode", "PendingAction",
    "ActionStatus",
    # threat posture & input screening
    "Guardian", "Posture", "ThreatLevel", "ThreatCategory", "ThreatEvent", "ThreatAssessment",
    "ResponsePlan", "DefensiveAction",
    "Shield", "ShieldVerdict", "ShieldAction", "ThreatFinding", "ThreatType", "TrustLevel",
    # the watchtower
    "AnomalyDetector", "Anomaly", "AnomalyType", "ScalarBaseline", "RateMonitor",
    "SequenceDetector", "NoveltyDetector",
    # network defence
    "NetworkDefense", "NetDecision", "NetAction", "Direction", "Connection", "FirewallRule",
    "Reputation",
    # containment & isolation
    "Containment", "ContainmentAction", "ContainmentLevel", "Component", "ComponentState",
    "IsolationEnvelope",
    # immune response
    "Phagocyte", "PhagocytosisResult", "DefensePlaybook", "Antigen",
    # survival & recovery (always subordinate to correction)
    "SurvivalManager", "DegradationLevel", "Backup", "BackupStore", "CriticalEntry",
    "RecoveryPlan",
    # identity, secrets & primitives
    "AuthGuard", "AuthResult", "Challenge", "ContinuityToken", "Session", "Factor",
    "FactorEvidence",
    "CredentialVault", "CredentialRecord", "CredentialKind", "get_process_vault",
    "resolve_api_key",
    "MasterKey", "SecretBox", "rotate_envelope", "mint_ssh_ed25519", "ssh_fingerprint",
    "random_token", "constant_time_eq",
    # learned values
    "ValueLearner", "PreferenceModel", "Preference", "Evidence", "EvidenceType",
    "ObserveResult",
]
