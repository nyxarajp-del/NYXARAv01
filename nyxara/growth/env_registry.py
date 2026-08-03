"""NYXARA · growth/env_registry.py — a persistent memory of every environment she has cracked (🗺).

Modelling an unknown system from first principles (:mod:`nyxara.growth.open_world`) is real work —
dozens of probes, a law search, a generalization check. Doing it *again* from scratch every time she
meets the same black box is not adaptation, it is amnesia. A mind that truly adapts to an environment
**remembers** it: the second time it is recognised at a glance and its law reused instantly.

This module is that memory. Each cracked system is distilled into a small, JSON-serialisable
:class:`EnvironmentProfile` — its input domain, the winning law *family and parameters* (enough to
rebuild the predictor with :func:`nyxara.growth.open_world.rebuild_predict`, no callable stored), a
calibrated confidence, and a cheap **fingerprint** (a handful of canonical ``action → observation``
pairs). :class:`EnvironmentRegistry` persists these under ``paths.data_dir`` and, given a fresh box,
:meth:`~EnvironmentRegistry.match` cheaply re-pokes it at a stored fingerprint: if the box answers as
the profile predicts, it *is* that environment and the saved law is handed straight back — recognition,
not re-derivation.

Everything is her own, deterministic, offline work: no LLM, no network. Persistence is best-effort and
never fatal — a missing or corrupt store degrades to "nothing remembered yet", and a failed save is
swallowed, so the registry can never crash an inquiry. Recognition still *verifies* against the live
box before trusting a memory, so a coincidentally-similar system is never mistaken for a known one.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from nyxara.growth.open_world import (DomainSpec, _Interactor, _from_vec, rebuild_predict)

__all__ = ["EnvironmentProfile", "EnvironmentRegistry"]


# --------------------------------------------------------------------------- #
# One remembered environment
# --------------------------------------------------------------------------- #
@dataclass
class EnvironmentProfile:
    """A compact, rebuildable record of one system NYXARA has already modelled."""
    label: str
    domain: Dict[str, Any]
    law_family: str
    law_description: str
    coeffs: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    # canonical (action, observation) pairs — the cheap identity check on re-encounter
    fingerprint: List[List[Any]] = field(default_factory=list)
    hits: int = 0                       # how often this environment has been recognised again
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "domain": dict(self.domain),
                "law_family": self.law_family, "law_description": self.law_description,
                "coeffs": dict(self.coeffs), "confidence": round(float(self.confidence), 4),
                "fingerprint": [list(p) for p in self.fingerprint], "hits": int(self.hits),
                "created_at": self.created_at, "updated_at": self.updated_at}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EnvironmentProfile":
        return cls(label=str(d.get("label", "unknown-system")),
                   domain=dict(d.get("domain", {})),
                   law_family=str(d.get("law_family", "")),
                   law_description=str(d.get("law_description", "")),
                   coeffs=dict(d.get("coeffs", {})),
                   confidence=float(d.get("confidence", 0.0)),
                   fingerprint=[list(p) for p in d.get("fingerprint", [])],
                   hits=int(d.get("hits", 0)),
                   created_at=float(d.get("created_at", time.time())),
                   updated_at=float(d.get("updated_at", time.time())))

    def rebuild(self) -> Optional[Any]:
        """Reconstruct this environment's predictor from its stored law params (or ``None``)."""
        spec = DomainSpec(**self.domain) if self.domain else DomainSpec()
        return rebuild_predict(self.law_family, self.coeffs, dims=spec.dims)


# --------------------------------------------------------------------------- #
# The persistent registry
# --------------------------------------------------------------------------- #
class EnvironmentRegistry:
    """A durable, self-consulting store of cracked environments.

    Parameters
    ----------
    path :
        Where to persist the JSON store. Defaults to ``settings.paths.data_dir/env_registry.json``.
    settings :
        Used only to locate the default ``path``. Optional.
    tol :
        Relative tolerance when checking a fingerprint reproduces on re-encounter.
    min_checks :
        How many fingerprint pairs must reproduce before a box is judged a known environment.
    max_profiles :
        Cap on stored profiles (oldest, least-hit dropped) so the store never grows without bound.
    """

    def __init__(self, *, path: Any = None, settings: Any = None, tol: float = 1e-3,
                 min_checks: int = 3, max_profiles: int = 512) -> None:
        self.tol = float(tol)
        self.min_checks = int(min_checks)
        self.max_profiles = int(max_profiles)
        self.path = self._resolve_path(path, settings)
        self._profiles: List[EnvironmentProfile] = []
        self._load()

    # ------------------------------------------------------------------ #
    # persistence (best-effort, never fatal)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_path(path: Any, settings: Any) -> Optional[Path]:
        if path is not None:
            return Path(path)
        try:
            if settings is None:
                from nyxara.kernel.config import get_settings
                settings = get_settings()
            return Path(settings.paths.data_dir) / "env_registry.json"
        except Exception:  # noqa: BLE001 — no settings ⇒ in-memory only
            return None

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._profiles = [EnvironmentProfile.from_dict(d) for d in raw.get("profiles", [])]
        except Exception:  # noqa: BLE001 — a corrupt store is simply "nothing remembered"
            self._profiles = []

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"version": 1, "profiles": [p.to_dict() for p in self._profiles]}
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self.path)          # atomic swap
        except Exception:  # noqa: BLE001 — a failed save must never crash an inquiry
            pass

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #
    def all(self) -> List[EnvironmentProfile]:
        return list(self._profiles)

    def __len__(self) -> int:
        return len(self._profiles)

    def match(self, system: Any, spec: DomainSpec) -> Optional[EnvironmentProfile]:
        """Return the stored profile this live box *is* — verified by re-poking it — else ``None``.

        Cheap by design: only the fingerprint's few actions are replayed, and the box must answer as
        the remembered profile did. A profile whose predictor cannot be rebuilt, or whose fingerprint
        no longer reproduces, is not a match — so recognition is never a blind cache hit."""
        box = _Interactor(system)
        for prof in self._profiles:
            if not self._domain_compatible(prof, spec):
                continue
            if self._fingerprint_reproduces(prof, box):
                prof.hits += 1
                prof.updated_at = time.time()
                self._save()
                return prof
        return None

    @staticmethod
    def _domain_compatible(prof: EnvironmentProfile, spec: DomainSpec) -> bool:
        d = prof.domain or {}
        return int(d.get("dims", 1)) == int(spec.dims) and str(d.get("kind", "real")) == spec.kind

    def _fingerprint_reproduces(self, prof: EnvironmentProfile, box: _Interactor) -> bool:
        checks = 0
        for pair in prof.fingerprint:
            if not pair or len(pair) < 2:
                continue
            action, expected = pair[0], pair[1]
            obs = box.run(action)
            if obs is None or not _close(obs, expected, self.tol):
                return False
            checks += 1
        return checks >= self.min_checks

    # ------------------------------------------------------------------ #
    # writes
    # ------------------------------------------------------------------ #
    def save(self, profile: EnvironmentProfile) -> None:
        """Insert or refresh a profile (keyed by label), then persist. Best-effort."""
        for i, existing in enumerate(self._profiles):
            if existing.label == profile.label:
                profile.hits = max(profile.hits, existing.hits)
                profile.created_at = existing.created_at
                profile.updated_at = time.time()
                self._profiles[i] = profile
                self._prune()
                self._save()
                return
        self._profiles.append(profile)
        self._prune()
        self._save()

    def build_profile(self, report: Any, *, fingerprint_size: int = 5) -> Optional[EnvironmentProfile]:
        """Distil an :class:`~nyxara.growth.open_world.UnderstandingReport` into a stored profile.

        Only a genuinely-modelled system with a *rebuildable* law is worth remembering; anything
        else returns ``None`` (we never persist a memory we could not reconstruct)."""
        winner = getattr(report, "winner", None)
        if winner is None or not getattr(winner, "coeffs", None):
            return None
        domain = dict(getattr(report, "domain", {}) or {})
        spec = DomainSpec(**domain) if domain else DomainSpec()
        if rebuild_predict(winner.family, winner.coeffs, dims=spec.dims) is None:
            return None
        # fingerprint: a spread of the probes actually observed, as (action, observation) pairs
        fp: List[List[Any]] = []
        for vec, obs in getattr(report, "samples", [])[:64]:
            if obs is None:
                continue
            fp.append([_from_vec(tuple(vec), spec), obs])
            if len(fp) >= fingerprint_size:
                break
        if len(fp) < self.min_checks:
            return None
        return EnvironmentProfile(
            label=str(getattr(report, "system", "unknown-system")),
            domain=domain,
            law_family=winner.family.value if hasattr(winner.family, "value") else str(winner.family),
            law_description=str(getattr(report, "law", "") or winner.description),
            coeffs=dict(winner.coeffs),
            confidence=float(getattr(report, "confidence", 0.0)),
            fingerprint=fp)

    def _prune(self) -> None:
        if len(self._profiles) <= self.max_profiles:
            return
        # keep the most-recognised, most-recent profiles
        self._profiles.sort(key=lambda p: (p.hits, p.updated_at), reverse=True)
        self._profiles = self._profiles[:self.max_profiles]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _close(a: Any, b: Any, tol: float) -> bool:
    """Tolerant equality: relative for numbers, exact for everything else (bools, strings)."""
    an = isinstance(a, (int, float)) and not isinstance(a, bool)
    bn = isinstance(b, (int, float)) and not isinstance(b, bool)
    if an and bn:
        try:
            return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
        except (TypeError, ValueError, OverflowError):
            return False
    return a == b


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import tempfile

    print("=" * 70)
    print("NYXARA Environment Registry self-test")
    print("=" * 70)

    from nyxara.growth.open_world import OpenWorldGeneralizer

    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp) / "env_registry.json"
        reg = EnvironmentRegistry(path=store, min_checks=3)
        assert len(reg) == 0

        system = lambda x: 3 * x - 2  # noqa: E731
        spec = DomainSpec(1, "real")
        rep = OpenWorldGeneralizer(seed=7).understand(system, domain=spec, label="line-A")
        prof = reg.build_profile(rep)
        assert prof is not None
        reg.save(prof)
        print(f"\n  saved            : {prof.label} → {prof.law_description} "
              f"(conf {prof.confidence:.2f})")

        # a fresh registry reads it back from disk
        reg2 = EnvironmentRegistry(path=store, min_checks=3)
        assert len(reg2) == 1
        print(f"  persisted        : reloaded {len(reg2)} profile(s) from disk ✓")

        # the SAME box is recognised without re-modelling
        hit = reg2.match(system, spec)
        assert hit is not None and hit.label == "line-A"
        pred = hit.rebuild()
        assert abs(pred((10.0,)) - (3 * 10 - 2)) < 1e-6
        print(f"  recognised       : rebuilt law predicts f(10)={pred((10.0,)):.4g} "
              f"(truth {3*10-2}) ✓")

        # a DIFFERENT box is not mistaken for the known one
        assert reg2.match(lambda x: 9 * x + 5, spec) is None
        print("  discriminates    : an unrelated box is NOT a false match ✓")

    print("\nALL SELF-TESTS PASSED ✓")
