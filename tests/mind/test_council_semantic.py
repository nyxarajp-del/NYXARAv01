"""P2-B — the council scores agreement *semantically* and confidence reflects it.

Agreement used to be lexical token overlap (Jaccard), which misses members who say the same
thing in different words. It is now embedding-cosine over the answers, so concordant paraphrases
register as agreement and a divergent panel scores lower — and because confidence blends
agreement with coverage, a panel that genuinely concurs is trusted more than one that does not
(capabilities #2, #36).
"""

from __future__ import annotations

from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.mind.council import LLMCouncil, MemberVerdict
from nyxara.mind.llm import LLM


def _council() -> LLMCouncil:
    s = NyxaraSettings.for_profile(Profile.DEV)
    return LLMCouncil(LLM(settings=s, providers={}), settings=s)


def _verdicts(*texts: str):
    return [MemberVerdict(f"m{i}", 1.0, t, True) for i, t in enumerate(texts)]


def test_paraphrase_agreement_beats_lexical():
    c = _council()
    # three answers that AGREE in meaning but share few words
    concord = _verdicts("The master is JP, whom NYXARA serves.",
                        "JP is the one NYXARA answers to.",
                        "Her sovereign is JP.")
    # the same surface form would score low on pure token overlap; semantically it is high
    assert c._agreement(concord) > c._jaccard_agreement(concord)


def test_divergent_panel_scores_lower_than_concordant():
    c = _council()
    concord = _verdicts("The master is JP and NYXARA serves JP.",
                        "JP is the master that NYXARA serves.",
                        "Our master is JP.")
    diverge = _verdicts("The master is JP.",
                        "Quantum chromodynamics describes gluons.",
                        "Bananas are a yellow fruit.")
    assert c._agreement(concord) > c._agreement(diverge)


def test_confidence_tracks_semantic_agreement():
    c = _council()
    concord = _verdicts("The master is JP and NYXARA serves JP.",
                        "JP is the master that NYXARA serves.",
                        "Our master is JP.")
    diverge = _verdicts("The master is JP.",
                        "Quantum chromodynamics describes gluons.",
                        "Bananas are a yellow fruit.")
    # confidence = 0.5*agreement + 0.5*coverage; with full coverage on both, agreement decides
    from nyxara.mind.council import CouncilResult, DeliberationMode
    hi = CouncilResult("x", DeliberationMode.SYNTHESIZE, "self",
                       verdicts=concord, agreement=c._agreement(concord), coverage=1.0)
    lo = CouncilResult("x", DeliberationMode.SYNTHESIZE, "self",
                       verdicts=diverge, agreement=c._agreement(diverge), coverage=1.0)
    assert hi.confidence > lo.confidence


def test_degrades_to_lexical_without_embedder():
    c = _council()
    c._embedder = None                       # simulate no embedder available
    v = _verdicts("the master is JP", "the master is JP")
    assert c._agreement(v) == c._jaccard_agreement(v)
