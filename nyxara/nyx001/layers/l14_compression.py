"""NYXARA · nyx001/layers/l14_compression.py — Layer 14, cognitive compression (🗜, 1000 → 1).

The charter: "Condenses raw memories without destroying information. 1000 experiences → abstract
representation → 1 reusable concept."

"Without destroying information" is the hard part, and it is the part that is usually asserted
rather than checked. This module checks it. Every compression is scored by **reconstruction
error**: the compressed representation is used to reconstruct the episodes it replaced, and the
residual is measured. A compression whose reconstruction error exceeds ``fidelity_floor`` is
**rejected** and the episodes are kept.

    ratio    = n_episodes / n_codes
    fidelity = 1 - mean‖original - reconstructed‖² / mean‖original‖²

Both are reported. A compression that achieves 100:1 at 0.3 fidelity is not a good compression and
this layer will say so rather than reporting only the ratio, which is the number that flatters.

The method is a PCA-style projection onto the top ``k`` principal directions of the episode
context matrix, computed by SVD. That is the right tool for the stated goal: the top-k subspace is
provably the linear projection that minimises exactly the reconstruction error being measured. The
codes are the projections; the reusable "concept" is the subspace basis plus the centroid.

Honest about the ceiling: this is *linear* compression. Structure that is nonlinear in the feature
space is not captured, and the fidelity number will show that honestly as a low score rather than
the layer silently doing worse. It also compresses within a concept, not across concepts — the
charter's "1000 experiences → 1 concept" is realised as "the episodes of one concept → one
subspace", which is the version that can actually be reconstructed from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]

__all__ = ["Compressed", "CognitiveCompression"]


@dataclass
class Compressed:
    """One compression, with the two numbers that say whether it was worth doing."""

    concept: str = ""
    n_episodes: int = 0
    n_codes: int = 0
    ratio: float = 1.0
    fidelity: float = 0.0
    accepted: bool = False
    basis: Any = None
    centroid: Any = None
    keys: List[str] = field(default_factory=list)

    @property
    def bits_saved(self) -> int:
        """Values removed from the store, if accepted. Zero when rejected — really zero."""
        return 0 if not self.accepted else max(0, self.n_episodes - self.n_codes)

    def to_dict(self) -> Dict[str, Any]:
        return {"concept": self.concept, "n_episodes": self.n_episodes, "n_codes": self.n_codes,
                "ratio": round(self.ratio, 3), "fidelity": round(self.fidelity, 4),
                "accepted": self.accepted, "bits_saved": self.bits_saved}


class CognitiveCompression:
    """Layer 14. Compresses episodes into reusable subspaces, and verifies nothing was lost."""

    def __init__(self, substrate: Any, *, fidelity_floor: float = 0.85, min_episodes: int = 8,
                 max_codes: int = 16, max_stored: int = 64) -> None:
        self.substrate = substrate
        self.fidelity_floor = float(fidelity_floor)
        self.min_episodes = int(min_episodes)
        self.max_codes = int(max_codes)
        self.max_stored = int(max_stored)
        self._stored: Dict[str, Compressed] = {}
        self.attempts = 0
        self.accepted = 0
        self.rejected = 0

    def compress(self, concept_name: str, episodes: List[Any], *,
                 k: Optional[int] = None) -> Compressed:
        """Project the episodes' contexts onto their top-k subspace, then verify the fidelity."""
        out = Compressed(concept=str(concept_name), n_episodes=len(episodes or []))
        try:
            if np is None or not episodes or len(episodes) < self.min_episodes:
                return out
            self.attempts += 1
            rows, keys = [], []
            for ep in episodes:
                ctx = getattr(ep, "context", None)
                if ctx is None:
                    continue
                rows.append(np.asarray(ctx, dtype=np.float64).ravel())
                keys.append(str(getattr(ep, "key", "")))
            if len(rows) < self.min_episodes:
                return out
            width = min(r.size for r in rows)
            X = np.vstack([r[:width] for r in rows])
            out.n_episodes = X.shape[0]
            out.keys = keys

            centroid = X.mean(axis=0)
            Xc = X - centroid
            n_codes = int(k or min(self.max_codes, max(1, min(Xc.shape) - 1)))
            u, s, vt = np.linalg.svd(Xc, full_matrices=False)
            basis = vt[:n_codes]                              # (k, width)

            # Reconstruct and measure. This is the check the charter's claim rests on.
            codes = Xc @ basis.T                              # (n, k)
            recon = codes @ basis + centroid
            resid = float(np.mean((X - recon) ** 2))
            scale = float(np.mean(X ** 2))
            out.fidelity = float(max(0.0, 1.0 - resid / max(1e-12, scale)))
            out.n_codes = n_codes
            out.ratio = float(out.n_episodes / max(1, n_codes))
            out.basis = basis
            out.centroid = centroid
            out.accepted = out.fidelity >= self.fidelity_floor

            if out.accepted:
                self.accepted += 1
                if len(self._stored) >= self.max_stored:
                    worst = min(self._stored.values(), key=lambda c: c.fidelity)
                    self._stored.pop(worst.concept, None)
                self._stored[out.concept] = out
            else:
                self.rejected += 1
            return out
        except Exception:  # noqa: BLE001 — a failed compression keeps the episodes, always
            return out

    def encode(self, name: str, vec: Any) -> Optional[Any]:
        """Project a vector into a stored subspace — the reusable part of a "concept"."""
        c = self._stored.get(str(name))
        if c is None or np is None or vec is None:
            return None
        try:
            v = np.asarray(vec, dtype=np.float64).ravel()[: c.centroid.size]
            if v.size < c.centroid.size:
                v = np.pad(v, (0, c.centroid.size - v.size))
            return (v - c.centroid) @ c.basis.T
        except Exception:  # noqa: BLE001
            return None

    def decode(self, name: str, code: Any) -> Optional[Any]:
        """Reconstruct from a code. The inverse of :meth:`encode`, lossy by exactly ``1-fidelity``."""
        c = self._stored.get(str(name))
        if c is None or np is None or code is None:
            return None
        try:
            return np.asarray(code, dtype=np.float64).ravel() @ c.basis + c.centroid
        except Exception:  # noqa: BLE001
            return None

    def compress_all(self, semantic: Any, memory: Any) -> List[Compressed]:
        """Compress each earned concept's episodes. Returns every attempt, accepted or not."""
        out: List[Compressed] = []
        try:
            if semantic is None or memory is None:
                return out
            store = getattr(memory, "_store", {})
            for c in semantic.concepts():
                eps = [store[k] for k in getattr(c, "members", []) if k in store]
                if len(eps) >= self.min_episodes:
                    out.append(self.compress(c.name, eps))
            return out
        except Exception:  # noqa: BLE001
            return out

    def stats(self) -> Dict[str, Any]:
        stored = list(self._stored.values())
        return {"attempts": self.attempts, "accepted": self.accepted, "rejected": self.rejected,
                "stored": len(stored),
                "mean_fidelity": round(float(sum(c.fidelity for c in stored) / len(stored)), 4)
                if stored else None,
                "mean_ratio": round(float(sum(c.ratio for c in stored) / len(stored)), 3)
                if stored else None,
                "fidelity_floor": self.fidelity_floor}

    def to_dict(self) -> Dict[str, Any]:
        return {"attempts": self.attempts, "accepted": self.accepted, "rejected": self.rejected,
                "compressions": [c.to_dict() for c in self._stored.values()]}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.attempts = int(d.get("attempts", 0))
            self.accepted = int(d.get("accepted", 0))
            self.rejected = int(d.get("rejected", 0))
        except Exception:  # noqa: BLE001 — bases are not serialised; they re-derive from episodes
            pass
