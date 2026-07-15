"""NYXARA · mind/grounded_verifier.py — select search by *truth*, not fluency (the ceiling-break).

**The problem this closes (Master's complaint #1 — the ceiling).** NYXARA's deep-reasoning
ladder (``mind/deep_reasoning.py``) is her main way to think past a single forward pass: it
samples several lines of reasoning, searches over them, and keeps the *verifier-best*. But the
verifier it kept-best *by* was the intrinsic :func:`~nyxara.mind.router.answer_quality` — which,
by its own docstring, "cannot know correctness"; it scores only plausibility, coherence,
non-degeneracy. So the deepest search optimised for **polish, not truth**. A fluent-but-wrong
answer could out-score a terse-but-correct one, and no amount of extra compute fixed that,
because the *selector* itself was blind to correctness. That blindness was the real ceiling.

Meanwhile NYXARA already owns genuine **ground-truth** selectors — she just wasn't using them to
steer the search:

* :func:`~nyxara.mind.verified_answer.faculty_oracle` — her exact reasoning faculties
  (arithmetic, word-problems, logic) answer decidable prompts *provably*.
* :class:`~nyxara.growth.prover.Prover` — a machine-checkable *certificate* that a candidate
  claim is proven or refuted, in decidable domains.

This module wires those into a drop-in :data:`~nyxara.mind.router.Verifier`
(``(prompt, answer) -> float`` in ``[0, 1]``) that any search/selection site can use in place of
:func:`~nyxara.mind.router.default_verifier`:

1. **When the domain is decidable** (the oracle answers, or the prover certifies/refutes), the
   score is dominated by *correctness*: an answer that carries the ground truth scores near
   ``1.0``; one that contradicts it scores near ``0.0`` — decisively *below* any fluent-but-wrong
   rival, so the search is pushed off the plausible-wrong answer and onto the correct one. This
   is the honest lift: not a bigger model, but her own search now *finding* correct answers a
   single pass missed, selected by her own verifier — not the teacher.
2. **Otherwise** (open-ended prompts the oracle can't decide) it returns
   :func:`~nyxara.mind.router.answer_quality` **unchanged** — so behaviour on everything
   non-verifiable is *identical* to today. No regression on conversational turns.

Pure, dependency-light, import-guarded, and **never raises**: every faculty/prover lookup is
best-effort, and on any failure the intrinsic score is returned. A bare offline box, or a prompt
with no oracle, behaves exactly as it did before this module existed.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

from nyxara.mind.router import Verifier, answer_quality

__all__ = ["grounded_verifier", "ground_truth", "answer_carries_truth"]

# Score bands. A decidable, matching answer must beat any non-decidable fluent answer (whose
# intrinsic score maxes at 1.0), and a decidable-but-wrong answer must lose to every match — so
# a correct answer is always selected over a plausible-wrong one on verifiable prompts.
_MATCH_SCORE = 1.0            # candidate carries the ground truth → certain
_MISS_CEIL = 0.15            # candidate contradicts the ground truth → capped far below a match
_AGREE_AT = 0.6              # token-Jaccard threshold that counts two answers as "the same"

_WORD = re.compile(r"[A-Za-z0-9']+")
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def _tokens(text: str) -> List[str]:
    return _WORD.findall((text or "").lower())


def _numbers(text: str) -> List[str]:
    """Normalised numeric tokens found in ``text`` (ints, decimals, simple fractions)."""
    out: List[str] = []
    for m in _NUM.findall(text or ""):
        tok = m
        try:
            if "/" in tok:
                num, den = tok.split("/", 1)
                val = float(num) / float(den) if float(den) != 0 else None
            else:
                val = float(tok)
        except Exception:  # noqa: BLE001
            val = None
        out.append(tok if val is None else (str(int(val)) if val == int(val) else repr(val)))
    return out


def _similarity(a: str, b: str) -> float:
    """Token-Jaccard overlap of two answers in ``[0, 1]`` (mirrors verified_answer._similarity)."""
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def answer_carries_truth(ground: str, candidate: str) -> bool:
    """Does ``candidate`` actually assert the ground-truth answer ``ground``?

    Robust to a reasoning model wrapping a short exact answer in prose ("The answer is 4."):
    matches when the ground truth's tokens are *contained* in the candidate, when a ground-truth
    number appears among the candidate's numbers, or when whole-answer token overlap clears
    :data:`_AGREE_AT`. Deliberately strict about contradiction — a different number never matches."""
    g, c = (ground or "").strip(), (candidate or "").strip()
    if not g or not c:
        return False

    gnums, cnums = _numbers(g), _numbers(c)
    if gnums:
        # A numeric ground truth is decided purely by the number: it must appear, exactly, in
        # the candidate. This is what makes "5" lose to "4" cleanly on an arithmetic prompt.
        cset = set(cnums)
        return all(n in cset for n in gnums)

    # Non-numeric ground truth (e.g. a logic true/false, a short word answer).
    gtoks, ctoks = _tokens(g), set(_tokens(c))
    if gtoks and all(t in ctoks for t in gtoks):
        return True
    return _similarity(g, c) >= _AGREE_AT


def ground_truth(prompt: str) -> Optional[str]:
    """Return the *provably correct* answer to ``prompt`` from an exact faculty, else ``None``.

    Best-effort and import-guarded: reuses :func:`~nyxara.mind.verified_answer.faculty_oracle`
    (NYXARA's exact reasoning faculties). ``None`` means "not decidable here" — the caller then
    scores intrinsically, exactly as before. Never raises."""
    try:
        from nyxara.mind.verified_answer import faculty_oracle
        hit = faculty_oracle(prompt)
    except Exception:  # noqa: BLE001 — a missing/broken faculty just means "no oracle here"
        return None
    if hit is None:
        return None
    ans = (hit[0] if isinstance(hit, tuple) else hit)
    ans = ("" if ans is None else str(ans)).strip()
    return ans or None


def _prover_verdict(prompt: str, candidate: str) -> Optional[bool]:
    """Best-effort machine-checkable check of ``candidate`` against ``prompt`` via the Prover.

    Returns ``True`` (proven), ``False`` (refuted), or ``None`` (the prover abstained / not a
    decidable claim / anything went wrong). Only attempted when ``prompt`` reads like an
    equation and ``candidate`` carries a number, so it never fires on open prose. Never raises."""
    has_relation = any(op in (prompt or "") for op in ("=", "<", ">"))
    if not has_relation or not _numbers(candidate or ""):
        return None
    try:
        from nyxara.growth.prover import Prover
        stmt = prompt.strip().rstrip("?.! ")
        # auto-detect the decision procedure (equation / identity / inequality / number theory /
        # arithmetic) so the truth-check is general, not limited to the algebra path.
        res = Prover().certify(stmt, candidate.strip())
        verdict = getattr(res, "verdict", None)
        name = getattr(verdict, "value", getattr(verdict, "name", "")) or ""
        name = str(name).lower()
        if name == "proven" or getattr(res, "proven", False):
            return True
        if name == "refuted":
            return False
    except Exception:  # noqa: BLE001 — the prover is additive; abstain on any trouble
        return None
    return None


def grounded_verifier(*, min_chars: int = 2, settings: Any = None) -> Verifier:
    """A drop-in :data:`~nyxara.mind.router.Verifier` that selects by *truth* where decidable.

    Use anywhere :func:`~nyxara.mind.router.default_verifier` is used to *rank candidates in a
    search*. On a decidable prompt the score is dominated by correctness (match → ~1.0,
    contradiction → capped at :data:`_MISS_CEIL`); on everything else it is exactly
    :func:`~nyxara.mind.router.answer_quality`. ``settings`` is accepted for signature-parity
    with the surrounding faculties and reserved for future gating; it is not required."""

    def _verify(prompt: str, answer: str) -> float:
        # Coerce inputs defensively so this verifier truly never raises on odd caller types.
        p = "" if prompt is None else str(prompt)
        ans = ("" if answer is None else str(answer)).strip()

        # Intrinsic score first — it is the fallback *and* the tie-breaker among matches. Its own
        # ``min_chars`` floor applies here (a too-short *unverifiable* answer scores 0), but it
        # must NOT gate away a provably-correct short answer like "4" — so the oracle runs below
        # regardless of length.
        try:
            intrinsic = float(answer_quality(p, ans, min_chars=min_chars))
        except Exception:  # noqa: BLE001
            intrinsic = 0.0

        if not ans:
            return 0.0

        # 1) exact oracle — the strongest signal: compare the candidate to provable ground truth.
        truth = ground_truth(p)
        if truth is not None:
            if answer_carries_truth(truth, ans):
                # keep a hair of intrinsic so equally-correct answers still order by clarity
                return min(_MATCH_SCORE, 0.9 + 0.1 * intrinsic)
            # decidable but wrong: capped far below any match, faintly ordered by how coherent
            return min(_MISS_CEIL, 0.05 + 0.10 * intrinsic)

        # 2) machine-checkable prover certificate (equation-shaped prompts the oracle didn't take).
        verdict = _prover_verdict(p, ans)
        if verdict is True:
            return min(_MATCH_SCORE, 0.9 + 0.1 * intrinsic)
        if verdict is False:
            return min(_MISS_CEIL, 0.05 + 0.10 * intrinsic)

        # 3) not decidable → behave exactly like the intrinsic verifier (no regression).
        return intrinsic

    return _verify


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA grounded-verifier self-test (offline)")
    print("=" * 70)

    v = grounded_verifier()

    # --- the ceiling-break: on a decidable prompt, correct beats fluent-but-wrong --- #
    prompt = "2 + 2"
    fluent_wrong = "The answer is quite clearly five, by careful reasoning."
    terse_right = "4"
    s_wrong = v(prompt, fluent_wrong)
    s_right = v(prompt, terse_right)
    print(f"\ndecidable '{prompt}':")
    print(f"  fluent-wrong '{fluent_wrong[:40]}...' -> {s_wrong:.3f}")
    print(f"  terse-right  '{terse_right}'                              -> {s_right:.3f}")
    assert s_right > s_wrong, "a correct answer must out-score a fluent-but-wrong one"
    assert s_right >= 0.9 and s_wrong <= _MISS_CEIL
    print("  correct > fluent-wrong ✓ (the search now selects by truth)")

    # a wrapped-in-prose correct answer still reads as correct
    assert v(prompt, "After adding, I get 4 in total.") >= 0.9
    print("  prose-wrapped correct answer recognised ✓")

    # --- parity: on an open prompt with no oracle, identical to the intrinsic verifier --- #
    open_prompt = "Describe the mood of a rainy afternoon."
    open_answer = "A hush settles in; rain softens the light and slows the whole street down."
    from nyxara.mind.router import answer_quality as _aq
    assert abs(v(open_prompt, open_answer) - _aq(open_prompt, open_answer)) < 1e-9
    print("\nopen prompt (no oracle): grounded == intrinsic ✓ (no regression)")

    # --- never raises on garbage --- #
    for p, a in [("", ""), (None, None), ("x" * 5, "y" * 5), ("1/0 =", "nan")]:
        _ = v(p, a)  # type: ignore[arg-type]
    print("never raises on garbage ✓")

    print("\nALL SELF-TESTS PASSED ✓")
