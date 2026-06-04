"""NYXARA · mind/creative.py — lateral thinking, SCAMPER, analogy, writing, code-gen.

The divergent half of the mind. Where :mod:`mind.reasoner` converges toward the one
right answer, this faculty *deliberately spreads out* — generating many ideas, then
selecting the good ones (divergent → convergent), using structured creativity rather
than hoping the LLM stumbles onto something.

Techniques:

* **SCAMPER** — Substitute, Combine, Adapt, Modify, Put-to-other-use, Eliminate,
  Reverse: seven systematic transformations of a concept.
* **Lateral thinking** (de Bono) — random entry (force a connection to an unrelated
  stimulus), provocation (PO), assumption challenge, and reversal.
* **Analogy generation** — view the topic through evocative source domains
  (war, ecosystem, orchestra, immune system…) to import their structure.
* **Conceptual blending** — fuse two concepts into a hybrid.

Ideas are scored on **novelty × usefulness × feasibility** so the engine can prune a
divergent storm down to the few worth keeping. Open-ended **writing** and **code-gen**
are delegated to the (probabilistic) LLM when one is supplied, with template
scaffolds offline. Output is a :class:`~nyxara.mind.proposal.Proposal` — kernel disposes.

Pure standard library; optional LLM via duck-typed ``.generate``.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.memory.provenance import Provenance, SourceType
from nyxara.mind.proposal import Proposal, ProposalKind

__all__ = [
    "Technique",
    "Idea",
    "CreativeEngine",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2]


# --------------------------------------------------------------------------- #
# Techniques & ideas
# --------------------------------------------------------------------------- #
class Technique(str, Enum):
    SCAMPER = "scamper"
    LATERAL = "lateral"
    ANALOGY = "analogy"
    BLEND = "blend"


@dataclass
class Idea:
    content: str
    technique: str
    novelty: float = 0.5
    usefulness: float = 0.5
    feasibility: float = 0.5

    @property
    def score(self) -> float:
        return _clamp(0.4 * self.novelty + 0.35 * self.usefulness + 0.25 * self.feasibility)

    def to_dict(self) -> Dict[str, Any]:
        return {"content": self.content, "technique": self.technique,
                "novelty": round(self.novelty, 3), "usefulness": round(self.usefulness, 3),
                "feasibility": round(self.feasibility, 3), "score": round(self.score, 3)}


# per-technique creativity profile: (novelty, usefulness, feasibility) base
_PROFILE = {
    "substitute": (0.55, 0.7, 0.7), "combine": (0.7, 0.7, 0.6),
    "adapt": (0.6, 0.75, 0.65), "modify": (0.55, 0.6, 0.7),
    "put_to_other_use": (0.75, 0.6, 0.55), "eliminate": (0.6, 0.65, 0.7),
    "reverse": (0.7, 0.55, 0.5),
    "random_entry": (0.85, 0.5, 0.4), "provocation": (0.9, 0.45, 0.3),
    "challenge": (0.75, 0.6, 0.5), "lateral_reverse": (0.7, 0.55, 0.5),
    "analogy": (0.75, 0.6, 0.5), "blend": (0.8, 0.6, 0.5),
}

# evocative stimulus words for random entry & combination
_STIMULI = ["mirror", "river", "magnet", "shadow", "seed", "clock", "fire", "bridge",
            "swarm", "key", "echo", "lens", "thread", "storm", "compass", "mask"]

# metaphor source domains: name -> guiding principle to import
_DOMAINS = {
    "war": "concentrate force at the decisive point; defend in depth",
    "ecosystem": "balance many interdependent niches; let feedback self-regulate",
    "orchestra": "many specialists, one tempo, conducted toward a single intent",
    "immune_system": "detect anomalies early, remember past threats, respond in layers",
    "river": "follow the path of least resistance, but carve the landscape over time",
    "market": "let incentives and prices route resources without central dictate",
    "garden": "cultivate slowly, prune the weak, let the strong compound",
}


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class CreativeEngine:
    """Structured divergent generation + convergent selection. LLM optional."""

    def __init__(self, *, llm: Any = None, rng: Optional[random.Random] = None) -> None:
        self.llm = llm
        self.rng = rng or random.Random()

    def _idea(self, content: str, technique: str, topic: str) -> Idea:
        nov, use, feas = _PROFILE.get(technique, (0.5, 0.5, 0.5))
        # usefulness nudged by topical relevance (some overlap == grounded creativity)
        topic_w, idea_w = set(_words(topic)), set(_words(content))
        relevance = len(topic_w & idea_w) / len(topic_w) if topic_w else 0.0
        use = _clamp(0.6 * use + 0.4 * relevance + 0.2 * relevance)
        return Idea(content=content, technique=technique,
                    novelty=nov, usefulness=use, feasibility=feas)

    # ---- SCAMPER ---- #
    def scamper(self, concept: str, *, attribute: str = "core feature") -> List[Idea]:
        other = self.rng.choice(_STIMULI)
        ops = {
            "substitute": f"Substitute part of «{concept}»: replace its {attribute} "
                          f"with something from a different domain.",
            "combine": f"Combine «{concept}» with «{other}» to form a hybrid.",
            "adapt": f"Adapt «{concept}»: borrow a mechanism that works elsewhere.",
            "modify": f"Magnify «{concept}»'s {attribute} 10×; then shrink it to near-zero.",
            "put_to_other_use": f"Repurpose «{concept}» for a use it was never meant for.",
            "eliminate": f"Remove the {attribute} of «{concept}» — what simpler thing remains?",
            "reverse": f"Reverse «{concept}»: make its {attribute} work backwards.",
        }
        return [self._idea(text, op, concept) for op, text in ops.items()]

    # ---- lateral thinking ---- #
    def random_entry(self, concept: str, *, stimulus: Optional[str] = None) -> Idea:
        stim = stimulus or self.rng.choice(_STIMULI)
        return self._idea(
            f"Force-connect «{concept}» with «{stim}»: how could the logic of {stim} "
            f"reshape {concept}?", "random_entry", concept)

    def provocation(self, concept: str, *, attribute: str = "core feature") -> Idea:
        return self._idea(
            f"PO: «{concept}» has no {attribute}. Follow the movement — what new "
            f"possibility opens once that constraint is gone?", "provocation", concept)

    def challenge_assumption(self, assumption: str) -> Idea:
        return self._idea(
            f"Challenge the assumption that «{assumption}». What if the opposite were true?",
            "challenge", assumption)

    def reversal(self, concept: str) -> Idea:
        return self._idea(f"What if «{concept}» ran in reverse — effect before cause?",
                          "lateral_reverse", concept)

    def lateral(self, concept: str) -> List[Idea]:
        return [self.random_entry(concept), self.provocation(concept), self.reversal(concept)]

    # ---- analogy generation ---- #
    def analogies(self, topic: str, *, n: int = 3) -> List[Idea]:
        domains = list(_DOMAINS.items())
        self.rng.shuffle(domains)
        out = []
        for name, principle in domains[:n]:
            out.append(self._idea(
                f"«{topic}» as a {name}: {principle}.", "analogy", topic))
        return out

    # ---- conceptual blending ---- #
    def blend(self, concept_a: str, concept_b: str) -> Idea:
        return self._idea(
            f"Blend «{concept_a}» and «{concept_b}»: a {concept_a}-{concept_b} that "
            f"keeps the strengths of both.", "blend", f"{concept_a} {concept_b}")

    # ---- divergent brainstorm ---- #
    def brainstorm(self, topic: str, *,
                   techniques: Sequence[Technique] = (Technique.SCAMPER, Technique.LATERAL,
                                                      Technique.ANALOGY),
                   blend_with: Optional[str] = None) -> List[Idea]:
        ideas: List[Idea] = []
        if Technique.SCAMPER in techniques:
            ideas += self.scamper(topic)
        if Technique.LATERAL in techniques:
            ideas += self.lateral(topic)
        if Technique.ANALOGY in techniques:
            ideas += self.analogies(topic)
        if Technique.BLEND in techniques and blend_with:
            ideas.append(self.blend(topic, blend_with))
        # de-dup by content
        seen, unique = set(), []
        for i in ideas:
            if i.content not in seen:
                seen.add(i.content)
                unique.append(i)
        return unique

    # ---- convergent selection ---- #
    def select(self, ideas: Sequence[Idea], *, k: int = 3,
               novelty_weight: float = 0.5) -> List[Idea]:
        """Prune a divergent set, balancing raw score with a novelty preference."""
        def key(i: Idea) -> float:
            return (1 - novelty_weight) * i.score + novelty_weight * i.novelty
        return sorted(ideas, key=key, reverse=True)[:k]

    def best(self, topic: str, **kw: Any) -> Optional[Idea]:
        ideas = self.brainstorm(topic, **kw)
        ranked = self.select(ideas, k=1)
        return ranked[0] if ranked else None

    # ---- open-ended generation (LLM-backed) ---- #
    _CREATIVE_SYSTEM = ("You are NYXARA's creative faculty. Be original, vivid, and "
                        "concrete. Prefer unexpected but apt ideas over clichés.")

    def generate(self, prompt: str, *, temperature: float = 1.0) -> str:
        if self.llm is not None:
            return self.llm.generate(prompt, system=self._CREATIVE_SYSTEM,
                                     temperature=temperature)
        return f"[creative-scaffold] {prompt}"

    def write(self, brief: str, *, form: str = "prose") -> str:
        prompt = f"Write a {form} piece based on this brief:\n{brief}"
        if self.llm is not None:
            return self.generate(prompt)
        return f"# Draft ({form})\n# Brief: {brief}\n\n(Offline scaffold — supply an LLM for prose.)"

    def code(self, spec: str, *, language: str = "python") -> str:
        prompt = (f"Write clean, correct {language} code for this specification, with a "
                  f"docstring and edge-case handling:\n{spec}")
        if self.llm is not None:
            return self.generate(prompt, temperature=0.6)
        # offline: a minimal, real scaffold derived from the spec
        name = re.sub(r"[^a-z0-9]+", "_", spec.lower()).strip("_")[:40] or "task"
        return (f'def {name}(*args, **kwargs):\n'
                f'    """{spec}\n\n    TODO: implement (offline scaffold).\n    """\n'
                f'    raise NotImplementedError({spec!r})\n')

    # ---- control-law output ---- #
    def propose(self, topic: str, **kw: Any) -> Proposal:
        idea = self.best(topic, **kw)
        if idea is None:
            return Proposal(ProposalKind.ANSWER, "(no idea generated)",
                            source_faculty="creative", confidence=0.0)
        return Proposal(
            kind=ProposalKind.ANSWER, content=idea.content, source_faculty="creative",
            confidence=idea.score, rationale=f"creative technique: {idea.technique}",
            provenance=Provenance(SourceType.SELF_REFLECTION, confidence=idea.score,
                                  detail="creative", method="generated"),
            metadata={"technique": idea.technique, "novelty": idea.novelty})


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA creative-faculty self-test")
    print("=" * 70)

    eng = CreativeEngine(rng=random.Random(7))

    # SCAMPER produces all seven transformations
    scamper = eng.scamper("a security firewall")
    print(f"\nSCAMPER ({len(scamper)} ideas):")
    for i in scamper[:3]:
        print(f"  [{i.technique}] {i.content}")
    assert len(scamper) == 7

    # lateral thinking
    lat = eng.lateral("threat detection")
    print(f"\nlateral:")
    for i in lat:
        print(f"  [{i.technique}] {i.content[:70]}…")
    assert any(i.technique == "provocation" for i in lat)

    # analogy generation imports structure from other domains
    ana = eng.analogies("autonomous defense", n=3)
    print(f"\nanalogies:")
    for i in ana:
        print(f"  {i.content}")
    assert len(ana) == 3
    assert all(i.technique == "analogy" for i in ana)

    # conceptual blending
    b = eng.blend("chess", "cybersecurity")
    print(f"\nblend               : {b.content}")
    assert "chess" in b.content and "cybersecurity" in b.content

    # divergent -> convergent
    ideas = eng.brainstorm("protecting the owner", techniques=list(Technique),
                           blend_with="immune system")
    top = eng.select(ideas, k=3)
    print(f"\nbrainstormed {len(ideas)} ideas; top 3 by novelty-weighted score:")
    for i in top:
        print(f"  {i.score:.2f} [{i.technique}] {i.content[:60]}…")
    assert len(top) == 3
    assert all(top[i].score >= 0 for i in range(3))

    # reproducibility with a seed
    a1 = CreativeEngine(rng=random.Random(1)).brainstorm("x")
    a2 = CreativeEngine(rng=random.Random(1)).brainstorm("x")
    assert [i.content for i in a1] == [i.content for i in a2]
    print("\nreproducible w/ seed : OK")

    # offline code scaffold is real, runnable Python structure
    code = eng.code("compute fibonacci numbers")
    print(f"\noffline code-gen     :\n{code}")
    assert "def " in code and "fibonacci" in code

    # LLM-backed generation when a provider is supplied
    from nyxara.kernel.config import NyxaraSettings, Profile
    from nyxara.mind.llm import LLM
    eng2 = CreativeEngine(llm=LLM(settings=NyxaraSettings.for_profile(Profile.TEST)))
    poem = eng2.write("a haiku about loyalty")
    print(f"\nLLM write (mock)     : {poem[:80]}")
    assert poem

    # control-law output
    prop = eng.propose("defending the network")
    print(f"\nas proposal          : [{prop.metadata['technique']}] conf={prop.confidence:.2f}")
    assert isinstance(prop, Proposal) and prop.source_faculty == "creative"

    print("\nALL SELF-TESTS PASSED ✓")
