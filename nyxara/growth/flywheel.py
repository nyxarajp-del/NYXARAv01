"""NYXARA · growth/flywheel.py — the data flywheel (🛞 → 🧠, the moat).

A model becomes *hers* only when it learns from *her own* lived experience — not a teacher's
canned battery, but the real, verified turns she has actually handled well. This module is that
collector: every turn that clears **all the gates** and a **quality bar** is captured as a
supervised ``(prompt → answer)`` pair and appended to a local corpus. Over time that corpus is
a moat: training data no one else has, grown from her own use, that the foundry can fine-tune on
to lift her own model — the wrapper→own-AI transition, fuelled by experience.

Design rules:

* **Compose, never duplicate.** A captured pair is a :class:`~nyxara.growth.distill.DistillationExample`
  written to the *same* JSONL format the foundry already consumes — so the flywheel corpus is a
  drop-in alongside the distillation corpus, with **zero** changes to the foundry or
  :func:`~nyxara.growth.distill.load_distillation_docs`.
* **Gather-only.** It never trains, never acts, never reaches the world — it only ever *appends
  to a file*. Promotion of any model trained on it still clears the foundry's full gauntlet.
* **Quality over volume.** The bar is deliberately strict: a turn must have *cleared every gate*
  (so it is safe and honest), be a genuine answer (not a tool side-effect or a refusal), clear a
  confidence floor and length bounds, pass an optional external **verifier**, and be **new**
  (deduplicated by prompt). A small clean corpus beats a large noisy one.
* **Honest provenance.** Each pair records its source (``flywheel`` or ``flywheel-verified`` when
  a verifier confirmed it) so a later training run can weight or filter by trust.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

from nyxara.growth.distill import DistillationExample, load_distillation_docs

__all__ = ["FlywheelDecision", "DataFlywheel"]

# an external check that a (prompt, answer) pair is actually correct/good — returns True to keep
Verifier = Callable[[str, str], bool]


@dataclass
class FlywheelDecision:
    """Why a turn was (or was not) added to the corpus — auditable, never silent."""
    collected: bool
    reason: str
    example: Optional[DistillationExample] = None


@dataclass
class DataFlywheel:
    """Collects verified-good lived turns into a foundry-ready supervised corpus.

    Stateless about conversation; the only state is the on-disk JSONL and an in-memory set of
    seen prompt-hashes for deduplication (loaded from the store on first use).
    """

    store_path: Path
    min_confidence: float = 0.6
    min_chars: int = 8
    max_chars: int = 8000
    system: Optional[str] = None
    verifier: Optional[Verifier] = None
    _seen: set = field(default_factory=set)
    _loaded: bool = False

    # ---- construction from settings (so the core wires it in one line) ---- #
    @classmethod
    def from_settings(cls, settings: Any = None, *, verifier: Optional[Verifier] = None,
                      system: Optional[str] = None) -> "DataFlywheel":
        from nyxara.kernel.config import get_settings
        s = settings or get_settings()
        cfg = s.flywheel
        if cfg.store_path is not None:
            path = Path(cfg.store_path)
        else:
            # reuse the foundry root so the flywheel corpus sits beside the distillation corpus
            from nyxara.growth.distill import _foundry_root
            path = _foundry_root(s) / "flywheel.jsonl"
        return cls(store_path=Path(path), min_confidence=cfg.min_confidence,
                   min_chars=cfg.min_chars, max_chars=cfg.max_chars,
                   system=system, verifier=verifier)

    # ---- the corpus on disk ---- #
    def _ensure_loaded(self) -> None:
        """Load existing prompt-hashes once so restarts don't re-add the same pairs."""
        if self._loaded:
            return
        self._loaded = True
        for ex in self.examples():
            self._seen.add(self._key(ex.prompt))

    @staticmethod
    def _key(prompt: str) -> str:
        return hashlib.sha256((prompt or "").strip().lower().encode("utf-8")).hexdigest()

    def _append(self, ex: DistillationExample) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with self.store_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")

    def examples(self, *, limit: Optional[int] = None) -> List[DistillationExample]:
        if not self.store_path.exists():
            return []
        out: List[DistillationExample] = []
        for line in self.store_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(DistillationExample.from_dict(json.loads(line)))
            except Exception:  # noqa: BLE001 — a malformed line is skipped, never fatal
                continue
            if limit and len(out) >= limit:
                break
        return out

    def count(self) -> int:
        self._ensure_loaded()
        return len(self._seen)

    def training_docs(self, *, limit: Optional[int] = None) -> List[str]:
        """Render the corpus into foundry-ready training docs (train/inference parity)."""
        return load_distillation_docs(self.store_path, limit=limit)

    # ---- the quality bar ---- #
    def consider(self, prompt: str, answer: str, *, confidence: float = 1.0,
                 verified: Optional[bool] = None, certificate: str = "",
                 exact: bool = True) -> FlywheelDecision:
        """Apply the quality bar to one ``(prompt, answer)`` pair; collect it if it clears.

        ``verified`` overrides the configured verifier when given (so a caller that already
        knows a turn was verified — e.g. a math turn checked by the reasoning faculties — can say
        so directly). The bar, in order: non-empty, length bounds, confidence floor, optional
        verifier, novelty. Every rejection returns a reason; nothing is dropped silently.

        ``certificate`` is the machine-checkable witness, when the caller has one. It was
        previously impossible to pass: callers that had proved the pair — the synthesis curator
        holds the prover's own certificate — had nowhere to put it, so the corpus recorded only
        *that* something was verified and never *why*. ``exact`` distinguishes a decision
        procedure from sampled agreement; only exact witnesses are rendered into the document.
        """
        self._ensure_loaded()
        prompt = (prompt or "").strip()
        answer = (answer or "").strip()
        if not prompt or not answer:
            return FlywheelDecision(False, "empty prompt or answer")
        if len(answer) < self.min_chars:
            return FlywheelDecision(False, f"answer too short (<{self.min_chars} chars)")
        if len(answer) > self.max_chars:
            return FlywheelDecision(False, f"answer too long (>{self.max_chars} chars)")
        if confidence < self.min_confidence:
            return FlywheelDecision(False,
                                    f"confidence {confidence:.2f} < {self.min_confidence:.2f}")

        is_verified = verified
        if is_verified is None and self.verifier is not None:
            try:
                is_verified = bool(self.verifier(prompt, answer))
            except Exception:  # noqa: BLE001 — a flaky verifier never sinks the turn; treat as unknown
                is_verified = None
            if is_verified is False:
                return FlywheelDecision(False, "verifier rejected the answer")

        key = self._key(prompt)
        if key in self._seen:
            return FlywheelDecision(False, "duplicate prompt already in corpus")

        source = "flywheel-verified" if is_verified else "flywheel"
        ex = DistillationExample(prompt=prompt, answer=answer, system=self.system, source=source,
                                 certificate=(certificate or "").strip(), exact=bool(exact))
        self._append(ex)
        self._seen.add(key)
        return FlywheelDecision(True, f"collected ({source})", example=ex)

    # ---- corrections: the Master said the old answer was WRONG ---- #
    def retract(self, prompt: str) -> int:
        """Remove every stored pair for ``prompt`` from the corpus; return how many.

        Rewrites the JSONL without the matching lines and clears the prompt from the
        dedup set — a wrong answer must stop being training data, and the slot must
        reopen so the corrected answer can take it."""
        self._ensure_loaded()
        key = self._key(prompt)
        if key not in self._seen or not self.store_path.exists():
            return 0
        kept: List[str] = []
        dropped = 0
        for line in self.store_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                ex_prompt = json.loads(stripped).get("prompt", "")
            except Exception:  # noqa: BLE001 — keep malformed lines untouched
                kept.append(stripped)
                continue
            if self._key(ex_prompt) == key:
                dropped += 1
            else:
                kept.append(stripped)
        tmp = self.store_path.with_suffix(self.store_path.suffix + ".tmp")
        tmp.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
        import os
        os.replace(tmp, self.store_path)
        self._seen.discard(key)
        return dropped

    def consider_correction(self, prompt: str, wrong_answer: str, corrected_answer: str,
                            *, weight: int = 3) -> FlywheelDecision:
        """A correction from the Master: retract the stale wrong pair, then store the
        corrected one ``weight`` times.

        Repetition is the corpus's native weighting mechanism (the same doc appearing N
        times pulls the model N times harder), so a correction outweighs an ordinary
        collected turn — being corrected teaches MORE than being right did. The pair is
        stored as ``source="correction"``, first-class verified supervision."""
        self._ensure_loaded()
        prompt = (prompt or "").strip()
        corrected = (corrected_answer or "").strip()
        if not prompt or not corrected:
            return FlywheelDecision(False, "empty prompt or corrected answer")
        if len(corrected) > self.max_chars:
            return FlywheelDecision(False, f"answer too long (>{self.max_chars} chars)")
        self.retract(prompt)
        ex = DistillationExample(prompt=prompt, answer=corrected, system=self.system,
                                 source="correction")
        for _ in range(max(1, int(weight))):
            self._append(ex)
        self._seen.add(self._key(prompt))
        return FlywheelDecision(True, f"correction collected ×{max(1, int(weight))}", example=ex)
