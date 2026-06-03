"""Kernel configuration — pydantic settings, dev/prod profiles, feature flags,
resource limits.

All tunable knobs live here so the rest of the system reads a single, validated
:class:`Settings` object instead of scattering ``os.environ`` lookups. Values can be
overridden by environment variables prefixed ``NYX_`` (e.g. ``NYX_PROFILE=prod``).
"""
from __future__ import annotations

import enum
import functools
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Profile(str, enum.Enum):
    DEV = "dev"
    PROD = "prod"
    SIM = "sim"  # everything routed to sandbox; zero real side effects


class FeatureFlags(BaseSettings):
    """Toggles for optional / experimental subsystems."""

    model_config = SettingsConfigDict(env_prefix="NYX_FF_", extra="ignore")

    continuous_stream: bool = True      # idle default-mode cognition
    prospective_memory: bool = True
    world_model: bool = True
    self_evolution: bool = False        # gated; off by default, sim-tested first
    web_access: bool = False            # off unless explicitly enabled by owner
    toolsmith: bool = False             # dynamic tool authoring; high-trust only
    speech_io: bool = False             # STT/TTS


class ResourceLimits(BaseSettings):
    """Hard ceilings the governor enforces on autonomous activity."""

    model_config = SettingsConfigDict(env_prefix="NYX_LIMIT_", extra="ignore")

    max_concurrent_faculties: int = 8
    max_proposal_depth: int = 6         # recursion guard on think→propose→think
    max_tokens_per_tick: int = 8_000
    max_tool_calls_per_min: int = 60
    max_spend_usd_per_day: float = 0.0  # 0 = no autonomous spend
    tick_budget_ms: int = 2_000         # soft per-tick deadline
    workspace_capacity: int = 7         # Miller's 7±2 conscious bottleneck


class Settings(BaseSettings):
    """Top-level, validated kernel configuration."""

    model_config = SettingsConfigDict(env_prefix="NYX_", extra="ignore")

    profile: Profile = Profile.DEV
    owner_id: str = "JP"                # canonical owner handle
    owner_name: str = "Jaypal Khoja"

    # Paths for durable state (created on demand; gitignored)
    state_dir: Path = Field(default=Path("nyxara_state"))
    audit_dir: Path = Field(default=Path("nyxara_state/audit_chain"))
    replay_dir: Path = Field(default=Path("nyxara_state/replay"))

    # Loop timing
    tick_hz: float = 5.0                # cognitive ticks per second when engaged
    idle_hz: float = 0.5                # ticks per second when idle

    # Invariant enforcement
    invariants_fail_closed: bool = True  # any gate error => deny
    verify_self_edits: bool = True

    features: FeatureFlags = Field(default_factory=FeatureFlags)
    limits: ResourceLimits = Field(default_factory=ResourceLimits)

    @field_validator("tick_hz", "idle_hz")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("tick rates must be positive")
        return v

    @property
    def is_prod(self) -> bool:
        return self.profile is Profile.PROD

    @property
    def sim_only(self) -> bool:
        return self.profile is Profile.SIM

    def ensure_dirs(self) -> None:
        for p in (self.state_dir, self.audit_dir, self.replay_dir):
            p.mkdir(parents=True, exist_ok=True)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide singleton settings (cached). Call once, read everywhere."""
    return Settings()


if __name__ == "__main__":  # pragma: no cover
    s = get_settings()
    print(f"profile={s.profile.value} owner={s.owner_id} "
          f"workspace_cap={s.limits.workspace_capacity} ff.stream={s.features.continuous_stream}")
