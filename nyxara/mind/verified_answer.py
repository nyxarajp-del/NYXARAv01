"""NYXARA · mind/verified_answer.py — inference-time ceiling-break (search + verify).

A student distilled from a teacher is bounded by the teacher: imitation can only *approach*
the source. The way out, at inference time, is the same one AlphaZero used at train time —
stop imitating and select by **ground truth**. Given several candidate answers, this module
returns the best one by the strongest signal available, in order:

1. **Exact oracle.** If a verifiable faculty (exact math / logic / a machine-checkable
   :class:`~nyxara.growth.prover.Prover` certificate) can answer or certify the prompt, that
   answer is *provably* correct — it beats any single teacher sample outright.
2. **Self-consistency majority.** Otherwise, draw several independent samples from her own
   model and return the answer the samples most agree on. Best-of-N + majority vote is a
   well-established lift over a single greedy draft on reasoning tasks — search buys accuracy
   the single sample (the teacher's or hers) does not have.

Pure selection over candidates the caller supplies: it changes no weights, calls no network
itself, and reaches around no gate. The chosen text is still a *proposal* the kernel disposes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

__all__ = ["VerifiedAnswer", "best_verified_answer", "faculty_oracle"]

_WORD = re.compile(r"[A-Za-z0-9']+")


def _tokens(text: str) -> List[str]:
    return _WORD.findall((text or "").lower())


def _norm(text: str) -> str:
    """A normalised key for grouping equivalent answers (case/space/punctuation-insensitive)."""
    return " ".join(_tokens(text))


def _similarity(a: str, b: str) -> float:
    """Token-Jaccard overlap of two answers in [0, 1] — a cheap agreement measure."""
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass
class VerifiedAnswer:
    """The selected answer plus *how* it was chosen and *how sure* the selection was."""

    text: str
    confidence: float           # 1.0 for an exact oracle; agreement fraction for a majority
    method: str                 # "oracle" | "self_consistency" | "single"
    samples: int = 0            # how many candidates were considered
    agreement: float = 0.0      # mean pairwise agreement among the winning cluster

    def to_dict(self) -> dict:
        return {"text": self.text[:500], "confidence": round(self.confidence, 4),
                "method": self.method, "samples": self.samples,
                "agreement": round(self.agreement, 4)}


# Oracle signature: prompt -> (answer, confidence) or None when not decidable.
Oracle = Callable[[str], Optional[Tuple[str, float]]]


def faculty_oracle(prompt: str) -> Optional[Tuple[str, float]]:
    """The default exact oracle: NYXARA's verifiable reasoning faculties (exact math/logic).

    Returns ``(answer, confidence)`` when a faculty can answer ``prompt`` exactly, else ``None``.
    First the fixed verifiable faculties; then, for the whole broadened class they do not cover
    (arbitrary equations/systems, identities, calculus, factorisation…), the general compute-and-
    verify reducer — but *only* its exactly-certified (``verified``) results, so the oracle stays
    provably correct. Both are best-effort and import-guarded so this is usable on the barest
    machine; the compute path here uses no model (symbolic only), so it is cheap and deterministic."""
    try:
        from nyxara.mind.reasoning_faculties import solve_with_faculties
        hit = solve_with_faculties(prompt)
        if hit is not None:
            return hit
    except Exception:  # noqa: BLE001 — a missing/broken faculty simply means "no oracle here"
        pass
    try:
        from nyxara.mind.compute_reasoner import ComputeReasoner
        res = ComputeReasoner().solve(prompt)   # no proposer -> symbolic CAS only
        if res is not None and res.verified and (res.answer or "").strip():
            return res.answer, float(res.confidence)
    except Exception:  # noqa: BLE001 — the compute reducer is additive; abstain on any trouble
        pass
    return None


def _majority(candidates: List[str], *, agree_at: float = 0.6) -> Tuple[str, float, float]:
    """Pick the answer the candidates most agree on (self-consistency).

    Groups candidates into clusters of mutually-similar answers, returns the representative of
    the largest cluster, its share of the vote (``confidence``) and the cluster's mean pairwise
    agreement. Exact-normalised matches always cluster together; near-matches cluster when their
    token-Jaccard clears ``agree_at`` (so "x = 2" and "x=2." count as the same vote)."""
    if not candidates:
        return "", 0.0, 0.0
    clusters: List[List[str]] = []
    keys: List[str] = []
    for cand in candidates:
        k = _norm(cand)
        placed = False
        for i, key in enumerate(keys):
            if k == key or _similarity(cand, clusters[i][0]) >= agree_at:
                clusters[i].append(cand)
                placed = True
                break
        if not placed:
            clusters.append([cand])
            keys.append(k)
    best = max(clusters, key=len)
    # mean pairwise agreement inside the winning cluster (1.0 for a singleton)
    sims, pairs = 0.0, 0
    for i in range(len(best)):
        for j in range(i + 1, len(best)):
            sims += _similarity(best[i], best[j])
            pairs += 1
    agreement = sims / pairs if pairs else 1.0
    confidence = len(best) / len(candidates)
    return best[0], confidence, agreement


def best_verified_answer(prompt: str, generate: Callable[[], str], *,
                         samples: int = 5, oracle: Optional[Oracle] = faculty_oracle,
                         agree_at: float = 0.6) -> Optional[VerifiedAnswer]:
    """Select the best answer to ``prompt`` by ground truth, then self-consistency.

    ``generate`` produces one fresh candidate per call (the caller binds it to a non-zero
    temperature so repeated calls differ). Returns ``None`` only when no candidate could be
    produced and no oracle applies — so the caller can fall back to its normal path.

    Order of preference:
      1. An **exact oracle** answer (provably correct) — confidence 1.0, ``method="oracle"``.
      2. The **majority** answer across up to ``samples`` own-model drafts — ``method=
         "self_consistency"`` — or the single draft when only one was available.
    """
    # 1) exact oracle — provably correct beats every neural guess (and any teacher sample).
    if oracle is not None:
        try:
            hit = oracle(prompt)
        except Exception:  # noqa: BLE001 — a broken oracle just means "fall through to sampling"
            hit = None
        if hit is not None:
            ans, conf = hit
            if (ans or "").strip():
                return VerifiedAnswer(str(ans).strip(), float(conf), "oracle",
                                      samples=0, agreement=1.0)

    # 2) draw candidates and select by self-consistency majority vote.
    n = max(1, int(samples))
    candidates: List[str] = []
    for _ in range(n):
        try:
            txt = (generate() or "").strip()
        except Exception:  # noqa: BLE001 — a failed sample just lowers the effective count
            continue
        if txt:
            candidates.append(txt)
    if not candidates:
        return None
    if len(candidates) == 1:
        return VerifiedAnswer(candidates[0], 1.0, "single", samples=1, agreement=1.0)
    text, confidence, agreement = _majority(candidates, agree_at=agree_at)
    return VerifiedAnswer(text, confidence, "self_consistency",
                          samples=len(candidates), agreement=agreement)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA verified-answer self-test (offline)")
    print("=" * 70)

    # 1) oracle wins outright
    va = best_verified_answer("2 + 2", lambda: "five",
                              oracle=lambda p: ("4", 1.0))
    assert va is not None and va.text == "4" and va.method == "oracle"
    print(f"oracle              : {va.to_dict()}")

    # 2) majority vote rescues a noisy own model (3×'Paris' vs 1×'London', 1×'Lyon')
    bag = iter(["Paris", "London", "Paris", "Lyon", "Paris"])
    va = best_verified_answer("capital of France?", lambda: next(bag),
                              samples=5, oracle=lambda p: None)
    assert va is not None and va.text == "Paris" and va.method == "self_consistency"
    assert va.confidence == 0.6
    print(f"self-consistency    : {va.to_dict()}")

    # 3) no candidates, no oracle -> None (caller falls back)
    va = best_verified_answer("x", lambda: "", samples=3, oracle=lambda p: None)
    assert va is None
    print("empty -> None       : ✓ (caller falls back)")

    print("\nALL SELF-TESTS PASSED ✓")
