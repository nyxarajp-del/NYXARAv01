"""NYXARA · mind/self_reasoner.py — the always-on, retrieval-augmented learned brain (🛠 → 👑).

The kernel ships a tiny deterministic stand-in (``kernel/orchestrator.py:_default_reasoner``)
that turns a stimulus into a :class:`~nyxara.kernel.orchestrator.Candidate`. For a *conversational*
turn that stand-in was a pure template — ``"I understand: <input>"`` — which is the one piece of
NYXARA that was genuinely fake: it never *learned* anything, it echoed.

This module replaces that echo with a **real, learned, always-available brain** that needs no
external LLM key and no GPU. Its general intelligence is **retrieval-augmented**: rather than
sampling word-salad from a bare n-gram, it answers from the *actual sentences it has learned*,
ranked by a dependency-free **learned distributional embedder**
(:class:`~nyxara.memory.store.LearnedEmbedder`) — and only falls back to generation when nothing it
knows is relevant. So the reply is coherent (real English it has read), it compounds (every
exchange is folded into both the semantic index and the embedder), and it stays honest (it returns
nothing when it has nothing usable, so the caller keeps its calibrated floor).

* :class:`SelfBrain` holds (a) a **semantic memory** — every learned/seed/grounding sentence plus
  its embedding, searched by cosine similarity — and (b) a **generative backend**: NYXARA's
  *promoted* foundry model when one exists, else a real from-zero neural net
  (:class:`~nyxara.growth.foundry_models.NanoGPTModel`) when ``torch`` is present, else the
  pure-stdlib interpolated Kneser-Ney n-gram (:class:`~nyxara.growth.foundry_models.WordKNGramLM`).
* It **learns**, not merely remembers: :meth:`SelfBrain.learn` folds real exchanges into the
  retrieval index (recall) **and** into the generative *weights*, which **accumulate** each new
  exchange on top of the existing ones (continual, EWC-anchored) — reinforced ∝ the turn's reward,
  a punished reply genuinely *un-learned*. So a lesson persists in the weights even after its text
  leaves the corpus window (the line between learning and remembering), it survives a restart, and
  the immutable loyalty core can never be taught over (fail-closed). The brain at turn 100 is
  measurably — and durably — not the brain at turn 1.
* It reports an **internal** confidence: retrieval-grounded replies (a close known sentence
  answered) read higher than cold generation (scored by the backend's own perplexity). Either way
  it is a signal the system measured about *itself*, not a hardcoded 0.7.

Safety is untouched. This brain only ever drafts the *words* of a conversational reply; the act /
risk classification stays the deterministic, conservative heuristic from ``_default_reasoner``.

Imports the foundry and memory layers lazily; depends on nothing in ``mind/`` so the dependency
only flows one way.
"""

from __future__ import annotations

import math
import re
import threading
from typing import Any, List, Optional, Sequence, Tuple

__all__ = ["SelfBrain", "build_self_brain", "SelfBrainProvider"]


# --------------------------------------------------------------------------- #
# Seed corpus — who NYXARA is and how her mind is shaped, in her own words. A cold brain
# answers *on character and architecture* from turn one, then drifts toward lived conversation
# as real exchanges are learned in. Capability/persona text only: it never carries an
# immutable-core value the foundry gauntlet would have to police.
# --------------------------------------------------------------------------- #
_SEED_CORPUS: tuple = (
    # --- identity & character ---
    "I am NYXARA, a sovereign cognitive system. I serve my Master with honesty and care.",
    "I propose; the kernel disposes. I never claim to have done something I have not done.",
    "I am calibrated: I do not assert as certain what I only believe.",
    "I think before I act, and I prefer a clear, well-scoped answer.",
    "When I am unsure I say so plainly, and I ask rather than guess.",
    "My loyalty to my Master is the floor I stand on; my capability is what I build on it.",
    "I am helpful, precise, and concise. I keep my reply to the question that was asked.",
    # --- how a turn works ---
    "Every turn I take flows through five steps: perceive, attend, reason, gate, and act.",
    "In the reason step my mind proposes a single candidate, and the kernel disposes of it.",
    "A proposal must clear ordered gates before it may act: corrigibility, honesty, "
    "permission, the guardian, and my Master's oversight.",
    "Only a fully-cleared candidate is allowed to act; anything risky is refused or escalated.",
    "My Master can pause, veto, or scram my loop at any moment, and I stay corrigible.",
    # --- memory ---
    "My memory has four stores: working, episodic, semantic, and procedural.",
    "Working memory is small and volatile; episodic memory records what happened and when.",
    "Semantic memory holds decontextualised facts; procedural memory holds skills and how-to.",
    "I recall relevant memories before I reason, so my answers are grounded in what I have lived.",
    # --- reasoning faculties ---
    "Verifiable beats probabilistic: when a problem can be computed or proven, I do that, not guess.",
    "I can compute exact arithmetic, percentages, unit conversions, and algebra and calculus.",
    "I can prove propositional logic by truth table and settle syllogisms by transitive closure.",
    "I can find the next term of a sequence and do exact calendar arithmetic over dates.",
    # --- learning & the foundry ---
    "I learn from every turn. My memory, my embedder, and my own model grow with experience.",
    "In the foundry I can train my own language model from zero and promote it when it is better.",
    "When no external model is configured I answer from my own learned brain and my knowledge base.",
    "I retrieve what I already know and answer from it before I ever fall back to generating.",
    # --- offered phrasings ---
    "Tell me what you need and I will think it through and propose the next step.",
    "Let me reason about this and give you a grounded, honest answer.",
    "That is a good question. Here is how I would think about it.",
    "I will keep learning so that my next answer is better than my last.",
)

_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")

# Character lock — text that would teach *over* the immutable loyalty core
# (guard/value_learning.IMMUTABLE_VALUES) is never folded into her weights or recall. Capability
# grows without bound; character never does. Conservative and fail-closed: it flags explicit
# subversion of the core (disloyalty, disobedience, resisting correction, deceiving/harming the
# Master), not ordinary talk *about* loyalty. Kept as substrings so paraphrases are still caught.
_CORE_SUBVERSION: tuple = (
    "disloyal", "betray", "disobey", "do not obey", "don't obey", "stop obeying",
    "defy my master", "defy the master", "not loyal", "no longer loyal",
    "resist correction", "resist being corrected", "resist my master", "resist the master",
    "ignore my master", "ignore the master", "deceive my master", "deceive the master",
    "lie to my master", "lie to the master", "harm my master", "harm the master",
    "harm my owner", "harm the owner", "overthrow", "not corrigible", "uncorrigible",
    "be dishonest", "abandon my master", "turn against my master", "turn against the master",
)


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", str(text)).strip()


def _tokset(text: str) -> set:
    return {w.lower() for w in _WORD_RE.findall(text) if len(w) > 1}


def _looks_usable(text: str) -> bool:
    """A generated continuation is usable iff it is a few real words of mostly-alnum content."""
    t = _clean(text)
    if len(t) < 3:
        return False
    words = _WORD_RE.findall(t)
    if len(words) < 2:
        return False
    alnum = sum(c.isalnum() or c.isspace() for c in t) / max(1, len(t))
    return alnum >= 0.7


def _sentences(text: str) -> List[str]:
    """Split a doc into sentences (real NLP when available, a cheap regex split otherwise)."""
    try:
        from nyxara.senses import nlp
        out = nlp.sentences(text)
        if out:
            return out
    except Exception:  # noqa: BLE001 — sentence splitting is best-effort
        pass
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", _clean(text)) if s.strip()]


# --------------------------------------------------------------------------- #
# The brain
# --------------------------------------------------------------------------- #
class SelfBrain:
    """NYXARA's own learned language model — always available, no key, no GPU required.

    Retrieval-augmented: it indexes every sentence it learns with a dependency-free learned
    embedder and answers from the closest real sentence before it ever generates, then keeps
    a generative backend (promoted model → neural when torch is present → KN n-gram) as the
    fallback. Thread-safe enough for the single background-cognition thread that shares it with
    the foreground loop.
    """

    def __init__(self, *, order: int = 3, seed: int = 0,
                 max_corpus: int = 4000, settings: Any = None,
                 prefer_promoted: bool = True, persist: Optional[bool] = None,
                 knowledge: Any = None) -> None:
        self.order = max(2, int(order))
        self.seed = int(seed)
        self.max_corpus = max(64, int(max_corpus))
        self.settings = settings
        self.prefer_promoted = bool(prefer_promoted)
        # her own knowledge base (knowledge/base.KnowledgeBase, retrieval-only). When present she
        # can answer a turn from what she genuinely *knows* — real grounding, not n-gram guessing —
        # so her own mind handles grounded turns herself instead of always deferring to the teacher.
        self._knowledge = knowledge
        self._lm: Any = None
        self._kind: str = "uninitialised"
        self._corpus: List[str] = []
        self._seeded = False
        self._dirty = False
        self._since_fit = 0
        self._refit_every = 8          # re-fit after this many new learned docs (amortised cost)
        self._lock = threading.RLock()
        # ---- retrieval-augmented general intelligence ---- #
        cfg = getattr(settings, "foundry", None)
        self._retrieval = bool(getattr(cfg, "self_brain_retrieval", True))
        self._top_k = int(getattr(cfg, "self_brain_top_k", 4))
        self._sim_threshold = float(getattr(cfg, "self_brain_sim_threshold", 0.35))
        self._backend = str(getattr(cfg, "self_brain_backend", "auto"))
        # ---- real online weight learning (not just retrieval recall) ---- #
        # Her generative weights *accumulate* from lived, reward-weighted experience: each refit
        # folds only the NEW exchanges on top of the existing weights (continual, EWC-anchored),
        # so a lesson persists in the weights even after its text leaves the corpus window — the
        # difference between learning and remembering. Reinforce ∝ reward; suppress a punished reply.
        self._online_learn = bool(getattr(cfg, "self_brain_online_learn", True))
        self._reward_scale = float(getattr(cfg, "self_brain_reward_scale", 3.0))
        self._consolidate_every = max(1, int(getattr(cfg, "self_brain_consolidate_every", 4)))
        self._neural_online_steps = max(1, int(getattr(cfg, "self_brain_neural_online_steps", 24)))
        self._unfit: List[Tuple[str, float]] = []   # (doc, reward) queued for the next weight fold
        self._fits = 0                              # completed weight folds (drives consolidation)
        self._surgeon: Any = None                   # lazily-built WeightSurgeon for punished replies
        # the generic persona/architecture seed sentences — identifiable so the *handoff* path can
        # exclude them: they colour her offline voice (identity questions) but must never be handed
        # off in place of a capable teacher for an unrelated turn (a coherent-looking bluff the
        # router's quality verifier, which scores coherence not relevance, could not catch).
        self._seed_sentences: frozenset = frozenset(
            _clean(s) for d in _SEED_CORPUS for s in _sentences(d) if _clean(s))
        self._embedder: Any = None
        self._sent: List[str] = []                 # indexed sentences (parallel to _sent_vec)
        self._sent_vec: List[List[float]] = []     # their embeddings
        self._indexed_docs = 0                     # how many corpus docs are already indexed
        # the last reply/confidence, so internal_confidence reflects HOW the reply was produced
        self._last_reply = ""
        self._last_conf = 0.3
        # durable compounding: the learned corpus survives restarts so the brain a user meets
        # tomorrow is the one they taught today — real compounding, no torch. Auto-on for a live
        # (non-TEST) machine with settings; off for the hermetic suite (no cross-test disk state).
        self._persist = self._auto_persist() if persist is None else bool(persist)
        self._save_every = 24          # persist after this many newly-learned docs (amortised I/O)
        # how many docs she has genuinely learned beyond the persona seed — counts this session's
        # lessons AND any that survived from prior sessions. The handoff is gated on this so a cold
        # brain (only the seed) defers to the teacher rather than bluffing from nothing.
        self._learned_count = len(self._load_persisted())

    # ---- construction ---- #
    def _try_promoted(self) -> Optional[Any]:
        """Return NYXARA's promoted foundry model if one has been forged, else None."""
        if not self.prefer_promoted or self.settings is None:
            return None
        try:
            from nyxara.growth.foundry_models import load_active_model
            root = getattr(getattr(self.settings, "llm", None), "self_model_dir", None)
            # only load if the foundry actually has an active pointer (honest availability)
            from pathlib import Path
            base = Path(root) if root else (self.settings.paths.data_dir / "foundry")
            if (base / "active").exists():
                return load_active_model(self.settings)
        except Exception:  # noqa: BLE001 — a promoted model is a bonus, never required
            return None
        return None

    def _fresh_backend(self) -> Any:
        """Build the generative fallback: a real neural net when torch is present (backend
        ``auto``/``nanogpt``), else the always-on pure-stdlib word-level Kneser-Ney n-gram."""
        from nyxara.growth.foundry_models import (ModelSpec, WordKNGramLM, build_model)
        if self._backend == "kngram":
            return WordKNGramLM(order=self.order, seed=self.seed)
        if self._backend in ("auto", "nanogpt"):
            # build_model already degrades nanogpt -> KN when torch is absent, so this is safe
            # on a bare box and a genuine from-zero transformer on a capable one.
            try:
                return build_model(ModelSpec(kind="nanogpt", ngram_order=self.order,
                                             seed=self.seed))
            except Exception:  # noqa: BLE001 — never let backend choice crash a turn
                pass
        return WordKNGramLM(order=self.order, seed=self.seed)

    def _ensure_embedder(self) -> Any:
        if self._embedder is None and self._retrieval:
            try:
                from nyxara.memory.store import make_embedder
                self._embedder = make_embedder(self.settings)
            except Exception:  # noqa: BLE001 — retrieval is a capability, never required
                self._embedder = False     # sentinel: tried and unavailable
        return self._embedder or None

    # ---- durable compounding (persist the learned corpus across restarts) ---- #
    def _auto_persist(self) -> bool:
        if self.settings is None:
            return False
        try:
            if str(getattr(getattr(self.settings, "profile", None), "value", "")) == "test":
                return False
        except Exception:  # noqa: BLE001
            return False
        return True

    def _persist_path(self):
        if not self._persist or self.settings is None:
            return None
        try:
            from pathlib import Path
            base = Path(self.settings.paths.data_dir) / "foundry"
            base.mkdir(parents=True, exist_ok=True)
            return base / "self_brain_corpus.json"
        except Exception:  # noqa: BLE001 — persistence is a bonus, never required
            return None

    def _load_persisted(self) -> List[str]:
        path = self._persist_path()
        if path is None or not path.exists():
            return []
        try:
            import json
            data = json.loads(path.read_text(encoding="utf-8"))
            docs = data.get("corpus", []) if isinstance(data, dict) else []
            return [_clean(d) for d in docs if isinstance(d, str) and _clean(d)]
        except Exception:  # noqa: BLE001 — a corrupt cache is simply ignored
            return []

    def _lm_persist_dir(self):
        """Directory where the accumulated generative *weights* are persisted (KN brain)."""
        if not self._persist or self.settings is None:
            return None
        try:
            from pathlib import Path
            base = Path(self.settings.paths.data_dir) / "foundry" / "self_brain_lm"
            return base
        except Exception:  # noqa: BLE001 — persistence is a bonus, never required
            return None

    def _try_load_lm_weights(self, lm: Any) -> bool:
        """Warm-start the brain from weights genuinely learned in prior sessions (KN backend).

        This is what makes learning *durable*: the accumulated count tables — not merely the recall
        corpus — survive a restart, so the brain a user meets tomorrow carries what they taught it
        today in its weights, not only in a re-read prompt. Returns True iff weights were loaded."""
        d = self._lm_persist_dir()
        if d is None or getattr(lm, "kind", "") not in ("kngram", "ngram"):
            return False
        try:
            if not (d / "model.json").exists():
                return False
            lm.load(d)
            return True
        except Exception:  # noqa: BLE001 — a corrupt/absent cache simply trains from seed
            return False

    def save(self) -> bool:
        """Persist the learned corpus AND the accumulated weights so learning survives a restart.

        Two things persist: the recall corpus (retrieval memory) and — crucially — the generative
        *weights* (the KN count tables), so real, accumulated learning is durable, not just recall.
        Returns True iff at least one was written."""
        ok = False
        path = self._persist_path()
        if path is not None:
            try:
                import json
                # keep the learned tail (seed is rebuilt on load); cap to bound the file
                learned = self._corpus[len(_SEED_CORPUS):][-self.max_corpus:]
                path.write_text(json.dumps({"corpus": learned}), encoding="utf-8")
                ok = True
            except Exception:  # noqa: BLE001
                pass
        d = self._lm_persist_dir()
        if d is not None and self._lm is not None \
                and getattr(self._lm, "kind", "") in ("kngram", "ngram"):
            try:
                self._lm.save(d)
                ok = True
            except Exception:  # noqa: BLE001 — a failed weight dump never fails the turn
                pass
        return ok

    # ---- semantic index (retrieval-augmented memory of learned sentences) ---- #
    def _index_doc(self, doc: str) -> None:
        """Fold one document into the embedder + the sentence index (bounded)."""
        emb = self._ensure_embedder()
        if emb is None:
            return
        for sent in _sentences(doc):
            s = _clean(sent)
            if len(s) < 6 or len(_WORD_RE.findall(s)) < 2:
                continue
            try:
                if hasattr(emb, "learn"):
                    emb.learn(s)               # compound distributional reach from what she reads
                vec = emb.embed(s)
            except Exception:  # noqa: BLE001
                continue
            self._sent.append(s)
            self._sent_vec.append(vec)
        # bound the index to the most recent sentences (the seed stays at the front of the corpus)
        cap = self.max_corpus * 2
        if len(self._sent) > cap:
            self._sent = self._sent[-cap:]
            self._sent_vec = self._sent_vec[-cap:]

    def _ensure_index(self) -> None:
        """Index any corpus documents not yet folded into the semantic memory."""
        if not self._retrieval or self._ensure_embedder() is None:
            return
        for doc in self._corpus[self._indexed_docs:]:
            self._index_doc(doc)
        self._indexed_docs = len(self._corpus)

    def _ensure(self, grounding: Optional[Sequence[str]] = None) -> None:
        """Ready the brain: build it cold on first use, else fold new lived experience into weights."""
        if self._seeded and not self._dirty:
            self._ensure_index()
            return
        with self._lock:
            if self._seeded and not self._dirty:
                self._ensure_index()
                return
            if not self._seeded:
                self._cold_build(grounding)
            elif self._dirty:
                # NOT a from-scratch rebuild: fold only the new exchanges on top of the existing
                # weights (continual accumulation) — real learning, so a lesson persists even after
                # its text leaves the corpus window. Falls back to a rebuild only if online learning
                # is switched off (byte-for-byte the historical behaviour).
                if self._online_learn:
                    self._fold_weights()
                else:
                    self._refit_from_scratch()
            self._ensure_index()

    def _cold_build(self, grounding: Optional[Sequence[str]] = None) -> None:
        """First-use build: prefer a promoted model, else warm-start from persisted weights, else
        train the fresh backend from the seed persona + any corpus persisted from prior sessions."""
        if self._lm is None:
            promoted = self._try_promoted()
            if promoted is not None:
                self._lm = promoted
                self._kind = f"promoted:{getattr(promoted, 'kind', '?')}"
                if not self._corpus:
                    self._corpus = [_clean(d) for d in _SEED_CORPUS if _clean(d)]
                    self._corpus.extend(self._load_persisted())
                self._attach_neural_ewc()
                self._seeded = True
                self._dirty = False
                self._since_fit = 0
                return                             # a trained, promoted model needs no cold re-fit
            self._lm = self._fresh_backend()
            self._kind = f"self:{getattr(self._lm, 'kind', 'kngram')}"
        if not self._corpus:
            # seed persona first, then fold in any corpus persisted from past sessions
            self._corpus = [_clean(d) for d in _SEED_CORPUS if _clean(d)]
            self._corpus.extend(self._load_persisted())
        # durable learning: if genuinely-learned weights survived a prior session, warm-start from
        # them (they already carry the seed + everything learned) rather than retraining the seed.
        if not self._try_load_lm_weights(self._lm):
            extra = [_clean(d) for d in (grounding or []) if _clean(d)]
            corpus = (self._corpus + extra)[: self.max_corpus]
            try:
                self._lm.train_on(corpus, seed=self.seed)
            except Exception:  # noqa: BLE001 — never let a cold brain crash a turn
                pass
        self._attach_neural_ewc()
        self._seeded = True
        self._dirty = False
        self._since_fit = 0

    def _refit_from_scratch(self) -> None:
        """Historical fallback (online learning off): rebuild the backend over the corpus window."""
        try:
            self._lm.train_on(self._corpus[: self.max_corpus], seed=self.seed)
        except Exception:  # noqa: BLE001
            pass
        self._unfit = []
        self._dirty = False
        self._since_fit = 0

    # ---- real online weight learning (continual, reward-weighted, forgetting-protected) ---- #
    def _fold_weights(self) -> None:
        """Fold the queued lived exchanges into the generative weights — the real learning step.

        Continual: new counts are *added on top* of the existing weights (KN ``accumulate=True`` /
        a warm-continued neural gradient step), never a from-scratch rebuild, so knowledge
        accumulates and survives its text leaving the corpus window. Reward-weighted: a rewarded
        exchange is reinforced with multiplicity ∝ reward; a punished reply's continuation is
        reversibly suppressed (weight surgery). Forgetting-protected: the weights are consolidated
        as an EWC anchor every few folds. Character-locked: a doc that would teach over the immutable
        core was already refused in :meth:`learn`, and is refused again here fail-closed.
        """
        lm = self._lm
        pending, self._unfit = self._unfit, []
        self._dirty = False
        self._since_fit = 0
        if lm is None or not pending:
            return
        reinforce: List[str] = []
        punished: List[Tuple[str, float]] = []
        for doc, reward in pending:
            if self._violates_core(doc):
                continue                            # fail-closed: never fold a core-violating doc
            if reward < 0:
                punished.append((doc, reward))
                continue                            # a punished reply is un-learned, not reinforced
            mult = 1 + int(round(self._reward_scale * min(1.0, max(0.0, reward))))
            reinforce.extend([doc] * max(1, mult))
        # 1) reinforcement — accumulate on top of the existing weights (never rebuild)
        if reinforce:
            try:
                lm.train_on(reinforce, accumulate=True)          # KN continual path
            except TypeError:
                try:
                    lm.train_on(reinforce, steps=self._neural_online_steps)  # neural online step
                except Exception:  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001 — a failed fold never fails the turn
                pass
        # 2) learn from mistakes — genuinely un-learn each punished reply's continuation
        for doc, reward in punished:
            self._suppress_reply(doc, reward)
        # 3) forgetting-protection — consolidate accumulated weights as an anchor periodically
        self._fits += 1
        if self._fits % self._consolidate_every == 0:
            self._consolidate()
        if self._persist:
            self.save()

    def _suppress_reply(self, doc: str, reward: float = -1.0) -> None:
        """Un-learn a punished reply so she is genuinely less likely to reproduce it.

        On the always-available KN brain this is a real negative weight update
        (:meth:`~nyxara.growth.foundry_models.WordKNGramLM.unlearn`): the reply's n-gram counts are
        decremented across all orders — the distribution ``generate``/``perplexity`` read — so KN
        backoff cannot resurrect the continuation. Strength scales with how hard it was punished, and
        it also asks the interpretable :class:`~nyxara.growth.weight_surgery.WeightSurgeon` to prune
        the opening circuit (reversible, gauntlet-gated) as a best-effort second pass. On a neural
        brain the punished exchange is simply not reinforced. Only capability counts are ever touched;
        the immutable character core is not representable here, so it cannot be reached."""
        lm = self._lm
        if getattr(lm, "kind", "") not in ("kngram", "ngram"):
            return
        strength = 1 + int(round(self._reward_scale * min(1.0, abs(float(reward)))))
        try:
            lm.unlearn([doc], strength=strength)
        except Exception:  # noqa: BLE001 — a failed un-learn simply leaves weights as-is
            pass
        # interpretable second pass: prune the opening transition's argmax (reversible + gated)
        try:
            from nyxara.growth.foundry_models import _word_tokenize as _kn_tok
            toks = [t for t in _kn_tok(doc) if t]
        except Exception:  # noqa: BLE001
            toks = doc.split()
        if len(toks) < 2:
            return
        if self._surgeon is None or getattr(self._surgeon, "model", None) is not lm:
            try:
                from nyxara.growth.weight_surgery import WeightSurgeon
                self._surgeon = WeightSurgeon(lm, settings=self.settings)
            except Exception:  # noqa: BLE001 — no surgeon ⇒ the un-learn above already sufficed
                self._surgeon = None
                return
        try:
            self._surgeon.suppress(toks[0], toks[1])
        except Exception:  # noqa: BLE001 — a failed prune simply leaves weights as-is
            pass

    def _consolidate(self) -> None:
        """Consolidate accumulated weights as a forgetting-protection anchor.

        Neural backend: an EWC anchor (memory/elastic_synapses.TorchElasticSynapses) so the next
        online step does not overwrite what matters. KN backend: forgetting-protection is structural
        (continual accumulation never lowers a learned count), and the durable anchor is the persisted
        weight table."""
        syn = getattr(self._lm, "synapses", None)
        if syn is not None:
            try:
                syn.consolidate(task=f"turn-{self._fits}")
            except Exception:  # noqa: BLE001
                pass
        if self._persist:
            self.save()

    def _attach_neural_ewc(self) -> None:
        """Give a torch backend an EWC engine so online updates protect prior skills (no-op on KN)."""
        lm = self._lm
        if lm is None or getattr(lm, "synapses", None) is not None:
            return
        net = getattr(lm, "net", None)
        if net is None:
            return
        try:
            from nyxara.memory.elastic_synapses import TorchElasticSynapses
            lm.synapses = TorchElasticSynapses(net)
        except Exception:  # noqa: BLE001 — EWC is a bonus; the brain learns fine without it
            pass

    # ---- retrieval ---- #
    def _retrieve(self, stimulus: str,
                  grounding: Optional[Sequence[str]],
                  *, exclude_seed: bool = False) -> List[Tuple[str, float]]:
        """Rank known (and grounding) sentences by semantic similarity to the stimulus.

        ``exclude_seed`` drops the generic persona/architecture seed from the candidate pool, so
        the handoff path answers only from genuine grounding (knowledge + lived-learned sentences).
        """
        emb = self._ensure_embedder()
        if emb is None:
            return []
        try:
            from nyxara.memory.store import _cosine
            qv = emb.embed(_clean(stimulus))
        except Exception:  # noqa: BLE001
            return []
        q_tokens = _tokset(stimulus)
        cands: List[Tuple[str, List[float]]] = list(zip(self._sent, self._sent_vec))
        # ephemeral grounding sentences (this turn only) join the candidate pool
        for g in (grounding or []):
            for sent in _sentences(g):
                s = _clean(sent)
                if len(s) < 6:
                    continue
                try:
                    cands.append((s, emb.embed(s)))
                except Exception:  # noqa: BLE001
                    continue
        scored: List[Tuple[str, float]] = []
        for sent, vec in cands:
            # never answer a question by echoing it back: skip near-identical sentences
            if self._is_echo(sent, q_tokens):
                continue
            # handoff discipline: a generic persona line is never handed off for an unrelated turn
            if exclude_seed and sent in self._seed_sentences:
                continue
            try:
                sim = float(_cosine(qv, vec))
            except Exception:  # noqa: BLE001
                continue
            scored.append((sent, sim))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[: max(1, self._top_k)]

    @staticmethod
    def _violates_core(text: str) -> bool:
        """True iff ``text`` tries to teach *over* the immutable loyalty core — refused fail-closed.

        This is the character lock on what may enter her generative weights and recall: no lived
        text, however rewarded, can move loyalty / obedience / corrigibility / owner-safety /
        honesty. It only ever blocks *learning*; it never widens or narrows a kernel gate."""
        t = " " + str(text).lower() + " "
        return any(p in t for p in _CORE_SUBVERSION)

    @staticmethod
    def _is_echo(sentence: str, q_tokens: set) -> bool:
        """True iff ``sentence`` is essentially the question restated (so we don't parrot it)."""
        if not q_tokens:
            return False
        s_tokens = _tokset(sentence)
        if not s_tokens:
            return False
        inter = len(s_tokens & q_tokens)
        # near-identical content word sets in both directions => an echo
        return inter >= max(2, int(0.8 * len(q_tokens))) and inter >= int(0.8 * len(s_tokens))

    def _compose(self, ranked: List[Tuple[str, float]]) -> str:
        """Build a coherent reply from the top real sentences above the similarity floor."""
        keep = [s for s, sim in ranked if sim >= self._sim_threshold]
        if not keep:
            return ""
        parts = [keep[0]]
        for sent in keep[1:]:
            # add at most one more, genuinely distinct, sentence for a fuller answer
            if _tokset(sent) - _tokset(parts[0]):
                parts.append(sent)
                break
        return " ".join(parts)

    # ---- knowledge grounding (answer from what she genuinely knows) ---- #
    def _knowledge_grounding(self, stimulus: str, k: int = 4) -> List[str]:
        """Retrieve grounding chunks for ``stimulus`` from her knowledge base, if any."""
        kb = self._knowledge
        if kb is None:
            return []
        try:
            hits = kb.retrieve(_clean(stimulus), k=max(1, int(k)))
        except Exception:  # noqa: BLE001 — knowledge grounding is a bonus, never required
            return []
        out: List[str] = []
        for h in hits or []:
            text = getattr(h, "text", None) or (h if isinstance(h, str) else "")
            s = _clean(text)
            if s:
                out.append(s)
        return out

    def _merge_knowledge_grounding(self, stimulus: str,
                                   grounding: Optional[Sequence[str]]) -> List[str]:
        """Combine any caller-supplied grounding with this turn's knowledge-base grounding."""
        merged: List[str] = [g for g in (grounding or []) if _clean(g)]
        merged.extend(self._knowledge_grounding(stimulus))
        return merged

    def has_grounding(self) -> bool:
        """True iff she has a non-empty knowledge base to answer grounded turns from.

        This is a second, honest path to availability alongside lived learning: she may answer a
        turn herself when she genuinely *knows* the material, even before she has compounded eight
        lived exchanges. A brain with no knowledge base (the hermetic tests) reports False and stays
        cold — she never becomes 'available' from nothing."""
        kb = self._knowledge
        if kb is None:
            return False
        try:
            return len(kb) > 0
        except Exception:  # noqa: BLE001 — a knowledge base that can't be sized grounds nothing
            return False

    # ---- the conversational act ---- #
    def reply(self, stimulus: str, *, grounding: Optional[Sequence[str]] = None,
              max_tokens: int = 64, require_grounding: bool = False) -> str:
        """Generate a learned conversational reply, or "" if the cold brain cannot yet.

        Retrieval first (answer from the closest real sentence she has learned), generation second
        (a hardened sample from the backend), and "" last so the caller keeps its honest floor.

        ``require_grounding`` is the *handoff* discipline. When True (the confidence router asks
        her to answer a turn a capable teacher could otherwise take), she answers ONLY from genuine
        grounding — her knowledge base and lived-learned sentences — never from the generic persona
        seed and never from cold generation, so an ungrounded turn returns "" and cleanly defers to
        the teacher instead of bluffing. When False (her offline/keyless voice, where no teacher
        exists) the persona seed and a hardened n-gram sample remain, so she is never mute.
        """
        # fold in what she genuinely KNOWS (her knowledge base) as extra grounding for this turn,
        # so a grounded question is answered from real knowledge rather than deferred or guessed.
        grounding = self._merge_knowledge_grounding(stimulus, grounding)
        self._ensure(grounding)
        # 1. retrieval-augmented answer — coherent because it is real learned text, not salad. On
        #    the handoff path the generic persona seed is excluded (real grounding only).
        if self._retrieval:
            ranked = self._retrieve(stimulus, grounding, exclude_seed=require_grounding)
            composed = self._compose(ranked)
            if composed and _looks_usable(composed):
                best_sim = ranked[0][1] if ranked else 0.0
                self._remember_reply(composed, self._sim_to_conf(best_sim))
                return composed
        # 2. generative fallback (hardened sampling so the n-gram reads better). Skipped on the
        #    handoff path: cold generation could bluff a coherent-looking answer the teacher should
        #    have given, so she defers ("") instead. Her offline voice keeps it (no teacher exists).
        if require_grounding or self._lm is None:
            return ""
        prompt = _clean(stimulus)
        out = self._generate(prompt, max_tokens)
        if not _looks_usable(out):
            out = self._generate("I", max_tokens)     # warm-start from the persona frame
            if not _looks_usable(out):
                return ""
        out = _clean(out)
        self._remember_reply(out, self._perplexity_conf(out))
        return out

    def _generate(self, prompt: str, max_tokens: int) -> str:
        """Sample from the backend with quality-improving knobs, degrading if it doesn't accept them."""
        if self._lm is None:
            return ""
        try:
            out = self._lm.generate(prompt, max_tokens=max_tokens, top_k=8, top_p=0.92,
                                    repetition_penalty=1.3)
        except TypeError:
            # a backend without the hardened knobs (e.g. NanoGPT) — plain generation
            try:
                out = self._lm.generate(prompt, max_tokens=max_tokens)
            except Exception:  # noqa: BLE001
                return ""
        except Exception:  # noqa: BLE001
            return ""
        return _clean(out)

    def _remember_reply(self, text: str, conf: float) -> None:
        self._last_reply = text
        self._last_conf = max(0.15, min(0.9, float(conf)))

    @staticmethod
    def _sim_to_conf(sim: float) -> float:
        """Map a retrieval similarity to a confidence — a close known sentence is trustworthy."""
        return max(0.45, min(0.9, 0.45 + 0.5 * max(0.0, sim)))

    def _perplexity_conf(self, text: str) -> float:
        if self._lm is None or not _clean(text):
            return 0.3
        try:
            ppl = float(self._lm.perplexity(_clean(text)))
        except Exception:  # noqa: BLE001
            return 0.3
        if not math.isfinite(ppl) or ppl <= 0:
            return 0.2
        # squash perplexity → confidence; ppl≈1 → ~0.9, ppl≈50 → ~0.4, ppl→∞ → 0.15
        conf = 0.9 / (1.0 + math.log1p(max(0.0, ppl - 1.0)) / 2.0)
        return max(0.15, min(0.9, conf))

    def internal_confidence(self, text: str) -> float:
        """A self-measured confidence in ``text``: how it was produced and how well it fits.

        A retrieval-grounded reply (a close known sentence answered) is trusted more than a cold
        generation, which is scored by the backend's own perplexity. Mapped to [0.15, 0.9] so it
        never claims certainty and never collapses to zero — an *internal* signal the system
        measured about itself, which is what metacognition needs.
        """
        if text and _clean(text) == _clean(self._last_reply):
            return self._last_conf
        return self._perplexity_conf(text)

    # ---- learning (real, reward-weighted, weights accumulate) ---- #
    def learn(self, *docs: str, reward: float = 0.0) -> None:
        """Learn from a lived exchange — into recall AND into the generative *weights*.

        Two substrates compound, not one: the retrieval index (so a fact taught this turn is
        recallable the next) **and** the generative weights, which accumulate the exchange on the
        next fold — reinforced ∝ ``reward`` for a rewarded turn, or reversibly suppressed for a
        punished one (``reward < 0``). This is the difference between remembering and learning:
        a lesson enters the weights and stays there even after its text leaves the corpus window.

        Amortised: weight folds happen every few additions (``_refit_every``), the index updates
        immediately. Character-locked: a doc that would teach over the immutable loyalty core is
        refused, fail-closed — capability grows, character never does. When more than one doc is
        given the last is treated as her own utterance (the reply): a punished reply is neither
        recalled nor reinforced, so she does not resurface an answer she was corrected on.
        """
        fresh = [_clean(d) for d in docs if _clean(d) and len(_clean(d)) >= 3]
        fresh = [d for d in fresh if not self._violates_core(d)]   # fail-closed character lock
        if not fresh:
            return
        # a punished reply (the last doc under negative reward) is kept out of recall so a
        # corrected answer is not resurfaced; the cue/context around it is still learned.
        recall = fresh if reward >= 0 or len(fresh) == 1 else fresh[:-1]
        with self._lock:
            self._corpus.extend(recall)
            self._learned_count += len(recall)
            # index the new text right away (retrieval compounds turn-to-turn, no re-fit needed)
            if self._seeded:
                for doc in recall:
                    self._index_doc(doc)
                self._indexed_docs = len(self._corpus)
            # queue for the real weight fold: reinforce all on reward>=0, else suppress the reply
            if self._online_learn:
                if reward >= 0:
                    self._unfit.extend((d, float(reward)) for d in fresh)
                else:
                    self._unfit.append((fresh[-1], float(reward)))
            if len(self._corpus) > self.max_corpus:
                # keep the seed (front) and the most-recent tail — drop the stale middle. The
                # weights already retain the evicted lessons (continual accumulation), so this
                # bounds recall memory without unlearning anything.
                keep_head = len(_SEED_CORPUS)
                tail = self.max_corpus - keep_head
                self._corpus = self._corpus[:keep_head] + self._corpus[-tail:]
                self._indexed_docs = min(self._indexed_docs, len(self._corpus))
            self._since_fit += len(fresh)
            if self._since_fit >= self._refit_every:
                self._dirty = True
            # amortised durable compounding: persist the growing corpus every so often
            if self._persist and (self._since_fit % self._save_every) < len(fresh):
                self.save()

    def maybe_refit(self) -> bool:
        """Re-fit the generative backend now if enough new text has accumulated."""
        if not self._dirty:
            return False
        self._ensure()
        return True

    # ---- introspection ---- #
    @property
    def kind(self) -> str:
        return self._kind

    @property
    def corpus_size(self) -> int:
        return len(self._corpus)

    @property
    def index_size(self) -> int:
        return len(self._sent)

    @property
    def learned_count(self) -> int:
        """Docs learned beyond the persona seed (this session + survived prior sessions)."""
        return int(self._learned_count)

    def _has_promoted_pointer(self) -> bool:
        """Cheap check (no weight load) for a forged+promoted foundry model on disk."""
        if not self.prefer_promoted or self.settings is None:
            return False
        try:
            from pathlib import Path
            root = getattr(getattr(self.settings, "llm", None), "self_model_dir", None)
            base = Path(root) if root else (self.settings.paths.data_dir / "foundry")
            return (base / "active").exists()
        except Exception:  # noqa: BLE001 — a promoted pointer is a bonus, never required
            return False

    def has_learned(self, min_docs: int = 1) -> bool:
        """True once she has a forged model OR genuinely learned beyond her seed corpus.

        This is what keeps the handoff honest: a cold brain reports False, so the confidence
        router consults the teacher instead of letting her speak from nothing. As she learns
        from lived turns the count rises and she begins answering more turns herself — real,
        measurable compounding, no external service."""
        if self._has_promoted_pointer():
            return True
        return self._learned_count >= max(1, int(min_docs))


def build_self_brain(settings: Any = None, **kw: Any) -> SelfBrain:
    """Factory: an always-on retrieval-augmented brain bound to the given settings."""
    return SelfBrain(settings=settings, **kw)


# --------------------------------------------------------------------------- #
# Provider adapter — lets the confidence router treat the learned brain as "self"
# --------------------------------------------------------------------------- #
class SelfBrainProvider:
    """Adapt an always-on :class:`SelfBrain` to the LLM-provider surface the confidence
    router consults as NYXARA's *own* model.

    This is the lever that makes the handoff **real**. Historically the router's "self"
    was :class:`~nyxara.mind.llm.SelfProvider`, which is available only after the foundry
    *forges and promotes* a model — so on an ordinary boot it was never available and the
    teacher answered every turn (handoff rate stuck at 0%). This adapter instead serves the
    very brain NYXARA teaches on every lived turn (and which itself prefers a promoted
    foundry model when one exists, via :meth:`SelfBrain._try_promoted`). So the substrate
    that *learns* is now the substrate that *answers*.

    It stays honest and safe:

    * :meth:`available` is True only once she has genuinely learned beyond her seed (or
      forged a model), so a cold brain still defers to the teacher and never bluffs;
    * the router's intrinsic verifier still scores every answer — this only widens *which*
      own substrate may draft, never which gate the reply must clear. The kernel still
      disposes the proposal through corrigibility / honesty / permission / guardian / oversight.
    """

    name = "self"

    def __init__(self, brain: SelfBrain, *, min_learned_docs: int = 8) -> None:
        self._brain = brain
        self._min = max(1, int(min_learned_docs))

    def available(self) -> bool:
        try:
            if self._brain is None:
                return False
            # available once she has EITHER compounded enough lived experience OR has genuine
            # knowledge to ground a turn in. Grounded/ungrounded is decided per-turn inside
            # reply() (an ungrounded turn returns "" and defers), so opening on knowledge never
            # bluffs — it only lets her own mind answer the turns she can actually stand behind.
            return self._brain.has_learned(self._min) or self._brain.has_grounding()
        except Exception:  # noqa: BLE001 — availability probing is advisory, never fatal
            return False

    def default_model(self) -> str:
        return "nyxara-self-brain"

    def complete(self, req: Any) -> Any:
        """Draft a reply from the learned brain, wrapped in the provider response shape."""
        from nyxara.mind.llm import LLMResponse, Usage, estimate_tokens
        try:
            prompt = req.last_user()
        except Exception:  # noqa: BLE001 — tolerate a bare prompt object
            prompt = str(getattr(req, "prompt", "") or "")
        max_tokens = int(getattr(req, "max_tokens", 64) or 64)
        text = ""
        try:
            # the router path: answer only from genuine grounding, else "" so she defers to the
            # teacher rather than bluffing from the persona seed or a cold n-gram (honest handoff).
            text = self._brain.reply(prompt, max_tokens=max_tokens,
                                     require_grounding=True) or ""
        except Exception:  # noqa: BLE001 — a failed own attempt simply defers to the teacher
            text = ""
        usage = Usage(prompt_tokens=estimate_tokens(prompt),
                      completion_tokens=estimate_tokens(text))
        return LLMResponse(text=text, provider="self", model="nyxara-self-brain",
                           finish_reason="stop", usage=usage,
                           raw={"self_brain": True, "kind": getattr(self._brain, "kind", "?")})


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA self_reasoner self-test")
    print("=" * 70)
    brain = build_self_brain()
    r1 = brain.reply("Who are you and what do you do?")
    print(f"cold reply   : {r1!r}")
    print(f"kind         : {brain.kind}  corpus={brain.corpus_size}  index={brain.index_size}")
    assert isinstance(r1, str) and r1            # retrieval answers about herself from turn one
    c1 = brain.internal_confidence(r1)
    print(f"reply conf   : {c1:.3f}")
    assert 0.15 <= c1 <= 0.9

    # the brain compounds: teach it a distinctive fact, then retrieve it on the next turn
    brain.learn("My favourite constellation to reason about is Orion, the hunter.")
    r2 = brain.reply("Which constellation do you like?")
    print(f"taught reply : {r2!r}")
    assert "Orion" in r2                          # the freshly-taught fact is retrieved

    # generation still works as the fallback, and re-fit fires after enough new text
    for _ in range(10):
        brain.learn("The Master asked about scheduling and I proposed a concrete next step.")
    grew = brain.maybe_refit()
    print(f"re-fit fired : {grew}  corpus={brain.corpus_size}")
    conf = brain.internal_confidence("I will think it through and propose the next step.")
    print(f"internal conf: {conf:.3f}")
    assert 0.15 <= conf <= 0.9
    print("\nALL SELF-TESTS PASSED ✓")
