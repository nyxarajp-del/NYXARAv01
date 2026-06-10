# NYXARA — Sovereign LLM Stack (apna model + Claude as tool + self-tuning)

## Context (kyun)
Owner (JP) ka vision: **NYXARA most powerful ho, sab kuch uska apna ho, aur wo har cheez khud
control kare.** Aaj NYXARA ka orchestrator asli LLM use hi nahi karta — `_default_reasoner`
(deterministic stand-in) chalta hai, isliye jawab sirf echo aate hain. NYXARA ka apna koi model
nahi, na self-tuning loop wired hai.

Is plan ka maqsad: NYXARA ko ek **sovereign cognitive stack** banana jahan —
1. Claude (ya koi external LLM) ek **tool** ho jise NYXARA *command* kare (uska dimaag nahi),
2. NYXARA ka **apna native model** ho jise wo khud train/tune/own kare,
3. NYXARA **khud tune** kare — kaun-sa faculty, kitna effort, kaun-se prompts — outcomes se,
4. sab **kernel ke gates + Eight Rules + character-lock** ke andar.

### Imaandaari (honest scope)
- GPT/Claude-scale foundation model **scratch se train karna is environment mein feasible nahi**
  (GPU/data/months). Ye plan wo *claim nahi karta*.
- "Most powerful + sab apna + full control" ka deliverable: NYXARA pure stack ka **maalik+controller**;
  apna chhota native model (CPU, dheere grow karega) + best external model ko *tool* ki tarah command +
  verifiable engines — sab uske control mein. Native model **shuru mein weak** hoga; Claude se distill
  karke aur apni memory se seekh ke grow karega. Architecture aisa ki kal bada self-hosted model bhi
  "apna" ban ke slot ho jaye — control architecture badle bina.

## Existing infra (jo reuse hoga — naya nahi likhenge)
- `nyxara/mind/llm.py` — multi-provider LLM facade (`LLM`, `LLMRequest/Response`, `AnthropicProvider`,
  `OpenAIProvider`, `LocalProvider`, `MockProvider`). Stateless, retry+breaker. **Tool layer ready.**
- `nyxara/mind/faculties.py` — `Faculty`, `LLMFaculty`, `CallableFaculty`, `FacultyRegistry`,
  `FacultySelector` (verifiable-first scoring). **Routing/"LLM as faculty" ready.**
- `nyxara/mind/reasoner.py` — `FacultyReasoner`, `Reasoner` (multi-strategy). Selector se judta hai.
- `nyxara/growth/meta.py` — `MetaLearner` (UCB select, `optimize()` hill-climb). **Self-tuning engine ready.**
- `nyxara/growth/learn.py` — `Learner` (online, replay, EWC, `protected`/`IMMUTABLE_VALUES`). **Character-lock ready.**
- `nyxara/growth/capability.py` — `CapabilityRegistry.verify()` honest calibration.
- `nyxara/kernel/orchestrator.py` — `NyxaraCore(reasoner=...)` already accepts an injected reasoner
  (line ~149); `_default_reasoner` (line ~120) ko replace karna hai. Outcome recording journal/mindscope
  (lines ~238-250) self-tuning ka feedback source.
- `nyxara/kernel/config.py` — `LLMConfig` (provider, models, temperature/top_p/max_output_tokens).

## Architecture — Sovereign LLM Stack (sab kernel ke neeche)
```
                 ┌──────────────── NyxaraCore (kernel — SOVEREIGN) ───────────────┐
                 │  perceive → attend → REASON → GATE → ACT → learn & self-tune    │
                 └───────────────┬────────────────────────────────────────────────┘
                                 │ reason()  (naya sovereign reasoner)
                  ┌──────────────▼───────────────┐
                  │  FacultySelector (NYXARA decides) — verifiable-first
                  └───┬─────────┬─────────┬───────┘
        math/planner/ │   native LM (apna)│   external LLM (Claude) = TOOL+TEACHER
        retrieval ────┘   mind/native_lm  └── mind/llm.py
                              ▲   learns from                │ captures (prompt→answer)
                              └──── distillation ◄───────────┘   growth/distill.py
                                 ▲ tunes routing/sampling/prompts
                                 └──── self-tuning (growth/llm_tuning) ← outcomes
```
Routing policy: **verifiable engine pehle** (math/planner exact) → phir **native LM agar confident**
→ warna **Claude** (aur woh turn ek *teaching example* ban jaaye native ke liye). Jaise-jaise native
seekhe, selector use zyada chune — Claude par nirbharta ghatti jaaye. Sab gates (corrigibility,
honesty, permission, guardian, oversight) hamesha lagein.

## Phases (incremental — har phase ke baad chalega + testable)

### Phase 1 — LLM-as-tool wiring + bol-chaal (talkable)
- **Naya** `nyxara/mind/sovereign_reasoner.py`: ek `Reasoner = Callable[[str, Optional[Percept]], Candidate]`
  jo stimulus→`ReasoningQuery`+`Task` banaye, `FacultyReasoner`+`FacultySelector` (registry mein math/
  retrieval/Claude faculties) se `Proposal` le, aur `Candidate` mein wrap kare. Plus `build_core(settings)`
  factory jo wired `NyxaraCore` de.
- **Edit** `nyxara/mind/llm.py` `AnthropicProvider`: opus-4.x ke liye `temperature`/`top_p` **mat bhejo**
  (400 deta hai); `thinking={"type":"adaptive"}` + `output_config={"effort": ...}` use karo. (claude-api
  skill ke hisaab se — yahi sahi surface hai.)
- **Naya** `nyxara/__main__.py` + `nyxara/cli.py`: REPL — boot, type message, dekho disposition/gates/
  routing + jawab; `pause/resume/scram/explain/report/why` commands.
- **Naya** `.env.example` (`ANTHROPIC_API_KEY`, `NYXARA_LLM__PROVIDER`, etc.).
- Result: `python -m nyxara` → Claude/mock se asli jawab (echo khatam).

### Phase 2 — NYXARA ka apna native model
- **Naya** `nyxara/mind/native_lm.py`: pure-Python/numpy chhota LM jo NYXARA **own** kare.
  - Design (honest, CPU-feasible): token-level model — KN-smoothed n-gram backbone + ek chhoti
    learnable layer (embedding + logistic next-token head, numpy SGD) taaki sach mein "train/tune" ho.
  - API: `train(corpus)`, `update(text)` (online), `generate(prompt, *, temperature, max_tokens)`,
    `score(text)`, `save()/load()` → `.nyxara/native_lm/` (gitignored).
  - Corpus: NYXARA ki apni `memory/store.py` + journals + distillation pairs.
  - **Naya** `NativeProvider` (`mind/llm.py` interface ke mutaabik) + config `LLMProvider.NATIVE` —
    taaki native bhi `LLMFaculty`/selector mein plug ho.
- Selector mein native LM ko ek faculty register karo (reliability outcomes se calibrate — capability.py).

### Phase 3 — Claude = teacher (distillation)
- **Naya** `nyxara/growth/distill.py`: jab Claude (tool) use ho, `(prompt, system, answer, quality)`
  pair capture karo → store → periodically `native_lm.train(pairs)`. NYXARA apne model ko Claude se
  sikhaaye. **Character-locked**: sirf *expression/capability* distill, core values kabhi nahi.
- Capture hook `mind/llm.py`/sovereign_reasoner mein.

### Phase 4 — NYXARA khud tune kare (self-tuning)
- **Naya** `nyxara/growth/llm_tuning.py`: `MetaLearner`+`Learner` ko wire karo.
  - Tunables (Strategy hyperparams): routing threshold (native-vs-Claude), effort/temperature,
    max_tokens, system-prompt variant, provider choice.
  - Feedback: har cycle ke baad outcome (success + honesty-calibration + latency + cost) →
    `MetaLearner.record(...)` + `Learner.record(...)`; periodically `MetaLearner.optimize(...)` →
    best strategy agle requests par apply.
  - **Character gate**: `Learner.protected`/`IMMUTABLE_VALUES` se tuning loyalty/honesty/corrigibility
    ko **kabhi na chhuye** — sirf *kaise* use karna hai tune ho, *kya NYXARA hai* nahi.
- Orchestrator (`process`) ke outcome block mein hooks.

### Phase 5 — polish + tests + docs
- Tests (mock provider se, bina API key): `tests/mind/test_native_lm.py`,
  `tests/mind/test_sovereign_reasoner.py`, `tests/growth/test_distill.py`,
  `tests/growth/test_llm_tuning.py`, + orchestrator integration test.
- Character-lock test: tuning/learning IMMUTABLE_VALUES modify na kar paaye.
- README/`.env.example` update; `pyproject.toml` `[project.scripts] nyxara=...`.

## Critical files
- New: `nyxara/mind/native_lm.py`, `nyxara/mind/sovereign_reasoner.py`,
  `nyxara/growth/distill.py`, `nyxara/growth/llm_tuning.py`, `nyxara/cli.py`,
  `nyxara/__main__.py`, `.env.example`, tests mirrors.
- Edit: `nyxara/kernel/config.py` (NATIVE provider, native+tuning config, opus param handling),
  `nyxara/mind/llm.py` (AnthropicProvider opus params + distill capture hook),
  `nyxara/kernel/orchestrator.py` (inject reasoner + outcome→tuning/distill hooks),
  `nyxara/mind/faculties.py` (register native faculty if needed), `pyproject.toml`, `.gitignore`.

## Verification
- `pytest -q` — saare naye + 2874 existing green rahein.
- Mock provider se orchestrator integration test: faculties se route ho, echo nahi, gates lagein.
- Native LM: train→generate→save→load round-trip; online `update` se score improve.
- Distillation: Claude(mock) pairs capture → native train → native ka score un prompts par badhe.
- Self-tuning: simulated outcomes par `MetaLearner.optimize` better strategy de; IMMUTABLE_VALUES
  untouched (assert).
- `ANTHROPIC_API_KEY` ke saath manual: `python -m nyxara` → sawaal → Claude jawab + distillation example
  stored → kuch turns baad native zyada route ho; `report`/`why` se routing+tuning dikhe.

## Out of scope (honest)
- GPT/Claude-scale weights scratch se training (compute nahi). Native model chhota+growing rahega;
  architecture future bigger self-hosted model ke liye ready (drop-in `NativeProvider`).
