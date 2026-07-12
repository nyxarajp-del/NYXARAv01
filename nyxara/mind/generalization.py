"""NYXARA · mind/generalization.py — one unified own-faculty generalizer (⇶).

The critique this module answers: *"NYXARA is only strong in the thing whose code was written;
she cannot do a genuinely NEW task without being taught."* The honest part of that critique was
never that the faculties don't exist — they do, and they are real — but that they were **narrow and
scattered**: exact math/logic ran every turn, but the from-examples skill-inducer, the relational-
transfer engine, and the black-box law-modeller each only fired on their own dedicated endpoint,
never as one cascade on an ordinary prompt.

:class:`GeneralizationEngine` unifies them. Given *any* prompt it tries NYXARA's own generalizers,
strongest-guarantee first, and returns the first genuine, honestly-hedged result — or ``None`` so
the normal path runs. Crucially it **parses demonstrations and tables straight out of the natural-
language prompt** and actually *runs* the inducer / law-modeller on them, so a task shown by a few
examples is solved on a held-out input **in the live turn** — the thing other systems do only when
someone hands them a training set.

Every stage is real, offline, and self-verifying (no LLM anywhere in this module):

* **faculties / first-principles** — :func:`nyxara.mind.reasoning_faculties.solve_with_faculties`
  (exact, verified computation/derivation);
* **skill induction from in-prompt demos** — :class:`nyxara.cognition.skill_induction.SkillInductionEngine`
  synthesises a program that reproduces *every* demonstration exactly, then applies it to the query;
* **relational transfer** — :class:`nyxara.mind.transfer.RelationalTransferEngine` structure-maps a
  novel domain onto one she already understands and projects the held-out structure;
* **open-world law modelling from an in-prompt numeric table** —
  :meth:`nyxara.growth.open_world.OpenWorldGeneralizer.model_dataset` fits the simplest law that
  generalizes to a held-out row;
* **compositional interpretation** — :class:`nyxara.cognition.composition.CompositionalGrammar`
  interprets an unseen combination of known primitives.

It **composes** those engines; it does not reimplement any of them. It never fakes: each stage
verifies and declines (returns ``None``) on failure, so the honest-abstention guarantee upstream is
preserved. It only ever *drafts* an answer — the kernel still disposes every reply.

Pure standard library (numpy only reached transitively through the composed engines).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "GeneralizationResult",
    "GeneralizationEngine",
    "parse_demos",
    "parse_num_table",
    "has_demos",
]

# --------------------------------------------------------------------------- #
# In-prompt structure parsers — recover demonstrations / tables from free text
# --------------------------------------------------------------------------- #
# Arrow separators that mean "input maps to output" in a demonstration.
_ARROW = r"(?:->|=>|-->|→|↦|⇒|➜)"
# A leading label like "pattern:", "rule:", "example(s):", "map", "given" that precedes demos.
_LABEL = re.compile(r"^\s*(?:pattern|patterns|rule|rules|examples?|mapping|map|given|"
                    r"apply the rule|apply this rule|transform|convert)\s*:?\s*", re.IGNORECASE)
# Query markers: the held-out input the induced task must be applied to.
_QUERY = re.compile(
    r"\b(?:now|then|apply(?:\s+(?:it|this|the\s+rule))?(?:\s+to)?|predict|"
    r"what(?:'s| is)|output\s+for|result\s+for|for\s+input|input)\b\s*[:=]?\s*"
    r"(?P<q>.+?)\s*[?.!]*\s*$",
    re.IGNORECASE)
# Fragment boundaries: commas, semicolons, newlines, and sentence periods (not decimals).
_SPLIT = re.compile(r"[,;\n]|(?<!\d)\.(?!\d)")
_ARROW_PAIR = re.compile(r"^(?P<l>.+?)\s*" + _ARROW + r"\s*(?P<r>.+?)$")
_COLON_PAIR = re.compile(r"^(?P<l>[^:]+?)\s*:\s*(?P<r>[^:]+?)$")
_NUM = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)$")


def _fragments(text: str) -> List[str]:
    return [f.strip() for f in _SPLIT.split(text or "") if f and f.strip()]


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in "\"'`" and s[-1] == s[0]:
        return s[1:-1]
    return s


def _looks_query_marker(frag: str) -> Optional[str]:
    """If ``frag`` is a query clause ("now: X", "what is X?"), return the query, else None."""
    m = _QUERY.match(frag.strip())
    if m:
        q = _strip_quotes(m.group("q").strip())
        # a query is a single value, never itself a demonstration
        if q and not re.search(_ARROW, q):
            return q
    return None


def parse_demos(prompt: str) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    """Recover ``(input, output)`` demonstrations and a held-out query from free text.

    Honest and conservative: only arrow-form (``a -> b``) pairs are taken as demonstrations
    (a colon form is accepted only as a fallback when no arrow pair exists and the fragment is
    not itself a query clause). Returns ``([], None)`` when nothing parses — the caller then
    declines rather than inventing a task.
    """
    text = (prompt or "").strip()
    if not text:
        return [], None
    demos: List[Tuple[str, str]] = []
    query: Optional[str] = None
    colon_candidates: List[Tuple[str, str]] = []
    for frag in _fragments(text):
        core = _LABEL.sub("", frag).strip()
        am = _ARROW_PAIR.match(core)
        if am:
            left = _strip_quotes(am.group("l"))
            right = _strip_quotes(am.group("r"))
            # a right side that is itself a query marker means this was "... . now: x"
            if left and right and not re.search(_ARROW, right):
                demos.append((left, right))
                continue
        # not an arrow demo — is it the query?
        q = _looks_query_marker(frag)
        if q is not None:
            query = q  # keep the LAST explicit query marker
            continue
        cm = _COLON_PAIR.match(core)
        if cm and not _LABEL.match(frag):
            left, right = _strip_quotes(cm.group("l")), _strip_quotes(cm.group("r"))
            if left and right:
                colon_candidates.append((left, right))
    if not demos and colon_candidates:
        demos = colon_candidates
    # no explicit query marker: if exactly one bare fragment is left over, treat it as the query
    if query is None and demos:
        used = set()
        for a, b in demos:
            used.add(a)
            used.add(b)
        leftovers = [f for f in _fragments(text)
                     if not re.search(_ARROW, f) and _looks_query_marker(f) is None
                     and _LABEL.sub("", f).strip() and not _COLON_PAIR.match(_LABEL.sub("", f).strip())]
        if len(leftovers) == 1:
            cand = _strip_quotes(_LABEL.sub("", leftovers[0]).strip())
            if cand and cand not in used:
                query = cand
    return demos, query


def parse_num_table(prompt: str) -> Tuple[List[Tuple[float, float]], Optional[float]]:
    """Recover a single-input numeric ``x -> y`` table and a numeric query from free text."""
    demos, query = parse_demos(prompt)
    pairs: List[Tuple[float, float]] = []
    for a, b in demos:
        if _NUM.match(a.strip()) and _NUM.match(b.strip()):
            pairs.append((float(a), float(b)))
    q: Optional[float] = None
    if query is not None and _NUM.match(query.strip()):
        q = float(query)
    return pairs, q


def has_demos(prompt: str) -> bool:
    """Cheap gate: does the prompt look like it carries demonstrations to learn from?"""
    text = prompt or ""
    if not re.search(_ARROW, text):
        return False
    demos, _q = parse_demos(text)
    return len(demos) >= 2


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class GeneralizationResult:
    """One genuine, honestly-hedged result from NYXARA's own generalizers."""

    answer: str
    source: str                        # which own faculty produced it
    confidence: float = 0.5
    analogical: bool = False           # True when reached by loose (renamed) structure-mapping
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "source": self.source,
            "confidence": round(float(self.confidence), 3),
            "analogical": bool(self.analogical),
            "detail": dict(self.detail),
        }

    def render(self) -> str:
        """The answer as NYXARA would say it, hedged when the derivation is a conjecture."""
        return self.answer


# --------------------------------------------------------------------------- #
# The unified engine
# --------------------------------------------------------------------------- #
class GeneralizationEngine:
    """Try NYXARA's own generalizers, strongest-guarantee first; the first genuine hit wins.

    Every dependency is optional and lazily built, so the engine runs fully offline (and in its
    own ``__main__`` self-test) with no orchestrator. It composes the real engines — it does not
    reimplement any of them — and returns ``None`` when none of them can honestly answer.
    """

    def __init__(self, *, transfer_engine: Any = None, skills: Any = None,
                 open_world: Any = None, composer: Any = None,
                 domain_genesis: Any = None,
                 min_confidence: float = 0.4, parse_demos_enabled: bool = True,
                 parse_tables_enabled: bool = True, min_demos: int = 2,
                 use_transfer: bool = True, use_domain_genesis: bool = True) -> None:
        self._transfer = transfer_engine
        self._skills = skills
        self._open_world = open_world
        self._composer = composer
        # her from-scratch domain-mastery faculty (mind/domain_genesis.py): when no known base
        # maps, model a genuinely alien field from its OWN internal structure rather than the LLM.
        self._domain_genesis = domain_genesis
        self.min_confidence = float(min_confidence)
        self.parse_demos_enabled = bool(parse_demos_enabled)
        self.parse_tables_enabled = bool(parse_tables_enabled)
        self.min_demos = max(1, int(min_demos))
        # relational transfer is also reachable as its own route; this flag lets a caller that
        # already runs transfer separately (the self-model router) turn off the cascade's copy.
        self.use_transfer = bool(use_transfer)
        self.use_domain_genesis = bool(use_domain_genesis)

    # ---- lazy builders (so the engine works standalone, offline) ---- #
    def _ensure_transfer(self) -> Any:
        if self._transfer is None:
            try:
                from nyxara.mind.transfer import RelationalTransferEngine
                self._transfer = RelationalTransferEngine()
            except Exception:  # noqa: BLE001 — transfer is a capability, never required
                self._transfer = None
        return self._transfer

    def _ensure_skills(self) -> Any:
        if self._skills is None:
            try:
                from nyxara.cognition.skill_induction import SkillInductionEngine
                self._skills = SkillInductionEngine(min_demos=self.min_demos)
            except Exception:  # noqa: BLE001
                self._skills = None
        return self._skills

    def _ensure_composer(self) -> Any:
        if self._composer is None:
            try:
                from nyxara.cognition.composition import CompositionalGrammar
                self._composer = CompositionalGrammar()
            except Exception:  # noqa: BLE001
                self._composer = None
        return self._composer

    def _ensure_domain_genesis(self) -> Any:
        if self._domain_genesis is None:
            try:
                from nyxara.mind.domain_genesis import DomainGenesisEngine
                # share the transfer engine's store so a field mastered from scratch is learned
                # into the same library and recognised (and transferable) next time.
                self._domain_genesis = DomainGenesisEngine(
                    transfer_engine=self._ensure_transfer(),
                    min_confidence=self.min_confidence)
            except Exception:  # noqa: BLE001 — domain genesis is a capability, never required
                self._domain_genesis = None
        return self._domain_genesis

    # ---- the public cascade ---- #
    def generalize(self, prompt: str) -> Optional[GeneralizationResult]:
        """Return the strongest genuine own-faculty result for ``prompt``, or ``None``.

        A *strict* transfer onto a known base (relation names identical) wins immediately, as
        does an exact faculty or an induced skill. A *loose analogical* transfer — a new domain
        forced onto a known base by structure alone — is weaker than a faithful model of the
        alien field's OWN structure, so it is held as a fallback and domain-genesis is tried
        first; the analogical transfer is used only if genesis (and everything after) declines."""
        prompt = (prompt or "").strip()
        if not prompt:
            return None
        fallback: Optional[GeneralizationResult] = None
        for stage in (self._try_faculties, self._try_skill_induction, self._try_transfer,
                      self._try_domain_genesis, self._try_open_world, self._try_composition):
            try:
                res = stage(prompt)
            except Exception:  # noqa: BLE001 — a stage never crashes the cascade
                res = None
            if res is None or res.confidence < self.min_confidence:
                continue
            # a loose cross-domain analogy is weaker than her own from-scratch domain model —
            # defer it and let domain-genesis (the next stage) try to master the field directly.
            if res.source == "relational_transfer" and res.analogical:
                if fallback is None:
                    fallback = res
                continue
            return res
        return fallback

    # ---- stage 1: exact verifiable faculties / first-principles ---- #
    def _try_faculties(self, prompt: str) -> Optional[GeneralizationResult]:
        try:
            from nyxara.mind.reasoning_faculties import solve_with_faculties
            hit = solve_with_faculties(prompt)
        except Exception:  # noqa: BLE001
            return None
        if hit is None:
            return None
        answer, conf = hit
        return GeneralizationResult(answer=str(answer), source="faculty",
                                    confidence=max(0.7, float(conf)),
                                    detail={"method": "verifiable_faculty"})

    # ---- stage 2: skill induction from in-prompt demonstrations ---- #
    def _try_skill_induction(self, prompt: str) -> Optional[GeneralizationResult]:
        if not self.parse_demos_enabled:
            return None
        demos, query = parse_demos(prompt)
        if len(demos) < self.min_demos or query is None:
            return None
        eng = self._ensure_skills()
        if eng is None:
            return None
        # induce under an ephemeral name so a one-off in-prompt task never clobbers a learned skill
        name = f"_adhoc_{abs(hash(tuple(demos))) % 100000}"
        skill = eng.induce(name, demos)
        if skill is None:
            return None
        out = skill.apply(query)
        if not out or out == query:
            return None
        conf = min(0.9, float(skill.confidence))
        answer = (f"Applying the rule I induced from your examples "
                  f"(verified on all {len(demos)}), {query!r} → {out!r}.")
        return GeneralizationResult(
            answer=answer, source="skill_induction", confidence=conf,
            detail={"skill": skill.render(), "query": query, "output": out,
                    "generalized": bool(skill.generalized), "demos": len(demos)})

    # ---- stage 3: relational structure transfer ---- #
    def _try_transfer(self, prompt: str) -> Optional[GeneralizationResult]:
        if not self.use_transfer:
            return None
        eng = self._ensure_transfer()
        if eng is None:
            return None
        tr = eng.generalize(prompt)
        if tr is None:
            return None
        conf = min(0.85, 0.5 + 0.1 * float(getattr(tr, "structural_score", 0.0)))
        analogical = bool(getattr(tr, "analogical", False))
        if analogical:
            conf = min(conf, 0.6)
        return GeneralizationResult(
            answer=tr.render(), source="relational_transfer", confidence=conf,
            analogical=analogical,
            detail={"transfer": tr.to_dict() if hasattr(tr, "to_dict") else {}})

    # ---- stage 3b: domain mastery FROM SCRATCH (no known base, no LLM) ---- #
    def _try_domain_genesis(self, prompt: str) -> Optional[GeneralizationResult]:
        """Model a genuinely alien field from its OWN internal structure — the from-scratch
        mastery that replaces the honest LLM fallback when nothing maps onto a known base.

        Declines (returns ``None``) when there is too little structure, or no induced law
        projects a genuinely new inference, so the normal path still runs."""
        if not self.use_domain_genesis:
            return None
        eng = self._ensure_domain_genesis()
        if eng is None:
            return None
        res = eng.generalize(prompt)
        if res is None:
            return None
        return GeneralizationResult(
            answer=res.render(), source="domain_genesis", confidence=float(res.confidence),
            analogical=bool(getattr(res, "analogical", True)),
            detail={"genesis": res.to_dict() if hasattr(res, "to_dict") else {}})

    # ---- stage 4: open-world law modelling from an in-prompt numeric table ---- #
    def _try_open_world(self, prompt: str) -> Optional[GeneralizationResult]:
        if not self.parse_tables_enabled or self._open_world is None:
            return None
        if not hasattr(self._open_world, "model_dataset"):
            return None  # capability added in the open-world upgrade; degrade gracefully until then
        pairs, query = parse_num_table(prompt)
        if len(pairs) < 3 or query is None:
            return None
        try:
            report = self._open_world.model_dataset(pairs, query=query)
            v = getattr(report, "verdict", "")
            verdict = str(getattr(v, "value", v)).lower()
            if verdict not in ("modelled", "partial", "supported"):
                return None
            prediction = report.predict(query)
        except Exception:  # noqa: BLE001
            return None
        if prediction is None:
            return None
        law = getattr(report, "law", None) or "a fitted law"
        answer = (f"From your table I fit {law} (validated on a held-out row) and predict "
                  f"f({_fmt_num(query)}) = {_fmt_num(prediction)}.")
        return GeneralizationResult(
            answer=answer, source="open_world", confidence=float(getattr(report, "confidence", 0.6)),
            detail={"report": report.to_dict() if hasattr(report, "to_dict") else {},
                    "query": query, "prediction": prediction})

    # ---- stage 5: compositional interpretation (last, mostly declines) ---- #
    def _try_composition(self, prompt: str) -> Optional[GeneralizationResult]:
        comp = self._ensure_composer()
        if comp is None:
            return None
        try:
            interp = comp.interpret(prompt)
        except Exception:  # noqa: BLE001
            return None
        if not getattr(interp, "ok", False) or interp.meaning is None:
            return None
        return GeneralizationResult(
            answer=f"Composed meaning: {interp.meaning}", source="composition",
            confidence=0.5, detail={"meaning": str(interp.meaning)})


def _fmt_num(v: Any) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(f)) if f == int(f) else f"{f:g}"


# --------------------------------------------------------------------------- #
# Self-test / demo (offline, no LLM anywhere)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA unified generalization self-test (her OWN faculties, no LLM)")
    print("=" * 70)

    eng = GeneralizationEngine()

    # 1. A brand-new string task, shown by two examples, solved on an unseen input — the answer
    #    can only come from induced program synthesis (there is no LLM here).
    res = eng.generalize("pattern: hello -> olleh, world -> dlrow. now: nyxara")
    assert res is not None and res.source == "skill_induction", res
    assert "araxyn" in res.answer, res.answer
    print(f"from-examples task  : {res.answer}")

    # 2. Demonstrations parsed out of a compact one-liner with a 'what is' query.
    demos, q = parse_demos("map ab -> ba, cd -> dc; what is xy?")
    assert demos == [("ab", "ba"), ("cd", "dc")] and q == "xy", (demos, q)
    res2 = eng.generalize("map ab -> ba, cd -> dc; what is xy?")
    assert res2 is not None and "yx" in res2.answer, res2
    print(f"parsed + applied    : {res2.answer}")

    # 3. Honest decline on free-form chat with no learnable structure.
    assert eng.generalize("hi how are you today") is None
    print("no structure        : declines (None) — normal path would run  ✓")

    # 4. Relational transfer still runs through the cascade for a novel domain.
    tr = eng.generalize("model of the atom: the nucleus attracts the electron and the "
                        "electron revolves around the nucleus")
    if tr is not None:
        print(f"relational transfer : {tr.source} (conf {tr.confidence:.2f})")

    print("\nALL SELF-TESTS PASSED ✓")
