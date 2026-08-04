"""NYXARA · kernel/ — the part that decides (⚙, the control law).

*The mind proposes; the kernel disposes; the Master is sovereign.* Everything in the rest of
the package produces **proposals**. This package is what turns a proposal into an act, or
refuses to.

:class:`~nyxara.kernel.orchestrator.NyxaraCore` is the whole mind wired together: it builds
every faculty, runs the perceive → attend → reason → gate → act cycle, and is the single
object the console, the server and the daemon all hold. Its per-turn output is a
:class:`~nyxara.kernel.orchestrator.CycleResult` carrying the
:class:`~nyxara.kernel.orchestrator.Disposition` and the gate verdicts that produced it.

Around it: :mod:`~nyxara.kernel.config` (every knob, one settings tree),
:mod:`~nyxara.kernel.rules` and :mod:`~nyxara.kernel.invariants` (the sealed constitution,
verified at boot and fail-closed on tampering), :mod:`~nyxara.kernel.errors` (typed failure,
retry and self-healing), :class:`~nyxara.kernel.bus.EventBus` and
:class:`~nyxara.kernel.jobqueue.JobQueue` (how work moves),
:class:`~nyxara.kernel.autonomic.AutonomicLoop` (the background mind between prompts),
:class:`~nyxara.kernel.workspace.GlobalWorkspace` and
:class:`~nyxara.kernel.stream.DefaultModeStream` (attention and idle thought),
:class:`~nyxara.kernel.presence.Presence` (wakefulness), and
:class:`~nyxara.kernel.replay.Recorder` (deterministic record/replay of a cognition run).

**Why this module is lazy.** ``nyxara.kernel.orchestrator`` imports ``nyxara.kernel.config``,
which means importing this package eagerly would import the orchestrator *while the kernel
package is still initialising* — a circular import. So the names below resolve on first
attribute access (PEP 562) instead. ``from nyxara.kernel import NyxaraCore`` works exactly as
it reads; nothing heavy is imported until you ask for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# public name -> submodule that defines it
_EXPORTS: dict[str, str] = {
    # the cycle
    "NyxaraCore": "orchestrator",
    "CycleResult": "orchestrator",
    "Disposition": "orchestrator",
    "Candidate": "orchestrator",
    # configuration
    "NyxaraSettings": "config",
    "get_settings": "config",
    "reload_settings": "config",
    "OwnerIdentity": "config",
    "Profile": "config",
    "FeatureFlags": "config",
    "ResourceLimits": "config",
    # the sealed constitution
    "Rule": "rules",
    "RuleSet": "rules",
    "RuleCategory": "rules",
    "Enforcement": "rules",
    "rules_digest": "rules",
    "verify_rules": "rules",
    "Invariant": "invariants",
    "InvariantSet": "invariants",
    "SystemState": "invariants",
    "BootReport": "invariants",
    "FormalReport": "invariants",
    "invariants_digest": "invariants",
    "boot_verify": "invariants",
    "enforce": "invariants",
    "prove_consistency": "invariants",
    # typed failure, retry, self-healing
    "NyxaraError": "errors",
    "ErrorCategory": "errors",
    "Severity": "errors",
    "InvariantViolation": "errors",
    "CorrigibilityError": "errors",
    "SecurityError": "errors",
    "AuthorizationError": "errors",
    "PromptInjectionError": "errors",
    "ValidationError": "errors",
    "ConfigurationError": "errors",
    "ResourceExhausted": "errors",
    "ExternalServiceError": "errors",
    "LLMError": "errors",
    "ToolError": "errors",
    "ErrorLedger": "errors",
    "ErrorRecord": "errors",
    "RetryPolicy": "errors",
    "retry": "errors",
    "aretry": "errors",
    "classify": "errors",
    "is_retryable": "errors",
    "wrap": "errors",
    # how work moves
    "EventBus": "bus",
    "Event": "bus",
    "Priority": "bus",
    "OverflowPolicy": "bus",
    "Subscription": "bus",
    "BusMetrics": "bus",
    "DeadLetter": "bus",
    "JobQueue": "jobqueue",
    "Job": "jobqueue",
    "JobStatus": "jobqueue",
    "Runtime": "runtime",
    "CircuitBreaker": "runtime",
    "CircuitBreakerConfig": "runtime",
    "CircuitOpenError": "runtime",
    "CircuitRegistry": "runtime",
    "CircuitState": "runtime",
    "Saga": "runtime",
    "SagaStep": "runtime",
    "SagaError": "runtime",
    "TaskHandle": "runtime",
    "ServiceRef": "runtime",
    "LifecycleState": "runtime",
    # the mind between prompts
    "AutonomicLoop": "autonomic",
    "Presence": "presence",
    "PresenceState": "presence",
    "PresenceMetrics": "presence",
    "StateProfile": "presence",
    "GlobalWorkspace": "workspace",
    "Broadcast": "workspace",
    "Coalition": "workspace",
    "Content": "workspace",
    "SalienceWeights": "workspace",
    "WorkspaceMetrics": "workspace",
    "DefaultModeStream": "stream",
    "Seed": "stream",
    "SeedPool": "stream",
    "ConceptGraph": "stream",
    "OpenProblem": "stream",
    "StreamMetrics": "stream",
    # substrate & determinism
    "compute_report": "compute",
    "recommend_device": "compute",
    "ComputeReport": "compute",
    "GPUInfo": "compute",
    "Recorder": "replay",
    "RecordEntry": "replay",
    "RandomProxy": "replay",
    "ReplayDivergence": "replay",
    "ReplayExhausted": "replay",
}

__all__ = sorted(_EXPORTS)

if TYPE_CHECKING:  # pragma: no cover — import-time cost is exactly what this module avoids
    from nyxara.kernel.autonomic import AutonomicLoop
    from nyxara.kernel.config import NyxaraSettings, get_settings
    from nyxara.kernel.orchestrator import Candidate, CycleResult, Disposition, NyxaraCore
    from nyxara.kernel.replay import Recorder


def __getattr__(name: str) -> Any:
    """Resolve a kernel export on first access (PEP 562) — see the module docstring."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value          # cache it: the lazy path runs once per name
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
