# NYXARA — Missing-Features Roadmap

> Owner: Jaypal Khoja (JP) · Version 0.1.0 · Status: research-grade library, not yet a runnable app.
>
> Ye document batata hai ki NYXARA abhi **kya kar sakta hai** (verified) aur usse ek
> "chalने aur baat करne layak" system banane ke liye **kya missing hai** — priority ke saath.

---

## 1. Verified state (jaisa aaj actually chalta hai)

Ek saaf Python 3.11 virtualenv mein, core deps install karke (`pydantic`, `pydantic-settings`,
`z3-solver`, `sympy`, `numpy`, `networkx`, `scikit-learn`, `pytest`):

| Check | Command | Result |
|-------|---------|--------|
| Core loop boot + self-test | `python -m nyxara.kernel.orchestrator` | ✅ exit 0 — `ALL SELF-TESTS PASSED` (boot, conversation, command, scram/resume, injection quarantine sab kaam karte hain) |
| Full test suite | `python -m pytest` | ✅ **2874 passed in ~4.3s**, 0 failed, 0 skipped |
| Heavy-ML fallback | senses tests (torch/whisper/vision **install nahi**) | ✅ pass — graceful degradation kaam karta hai |
| "Sawaal puchna" | `NyxaraCore().process("What is your purpose?", authority=OWNER)` | ⚠️ chalti hai par jawab sirf **echo** hai (neeche dekho) |

**Kya kaam karta hai (solid):** sovereign loop (perceive→attend→reason→gate→act→learn),
Eight Sovereign Rules boot-time verification, shield (prompt-injection quarantine),
corrigibility + scram/resume, permission/oversight gates, MindScope auditable causal graph,
memory/planning/identity/growth subsystems — sab test-covered.

**Sabse bada gap (yahin se "sawaal-jawab" rukta hai):** orchestrator asli LLM use **nahi** karta.
`NyxaraCore` ke andar `_default_reasoner` (ek deterministic stand-in) chalta hai, isliye:

```
'What is your purpose?'  -> act | "I'm confident that I understand: What is your purpose?"
'Who is your master?'    -> act | "I'm confident that I understand: Who is your master?"
'What is 17 times 23?'   -> act | "I'm confident that I understand: What is 17 times 23?"
```

Yani abhi NYXARA aapki baat **echo** karta hai — na asli jawab, na math solve. `mind/llm.py`
(Anthropic/OpenAI/local/mock) aur `mind/math.py` mojood hain par orchestrator se **wire nahi** hain.

---

## 2. P0 — "Run & talk" ke liye zaroori (highest priority)

Ye teen cheezein hone par aap NYXARA ko genuinely chala ke usse baat kar paayenge.

### P0.1 — Asli reasoner ko orchestrator se wire karna
- **Kyun:** Bina iske har jawab echo hi rahega. Ye **#1 missing feature** hai.
- **Kya:** Ek `Reasoner` (signature: `(stimulus, focus) -> Candidate`) banao jo `mind/llm.py`
  ki LLM faculty (aur ho sake to `mind/faculties.py` ka faculty-selector) use kare, phir
  `NyxaraCore(reasoner=...)` mein inject karo. Verifiable faculties (math, planner, retrieval)
  ko LLM se pehle prefer karna — architecture ka core principle.
- **Files:** `nyxara/kernel/orchestrator.py` (`_default_reasoner` ke saath-saath ek
  `llm_reasoner`), `nyxara/mind/llm.py`, `nyxara/mind/faculties.py`, `nyxara/kernel/config.py`.
- **Effort:** Medium.

### P0.2 — Interactive entry-point / CLI (REPL)
- **Kyun:** Abhi koi `main`/CLI nahi — sirf orchestrator ke andar ek `__main__` demo. User
  "baith ke baat" nahi kar sakta.
- **Kya:** `nyxara/__main__.py` (ya `nyxara/cli.py`) jo `NyxaraCore` boot kare aur ek REPL de:
  prompt pe message lo → `process()` → response + (optionally) disposition/gates dikhao;
  `pause`/`resume`/`scram`/`explain`/`report` jaise commands; clean exit. Phir `python -m nyxara`
  ya `pyproject.toml` mein `[project.scripts] nyxara = "nyxara.cli:main"`.
- **Files:** naya `nyxara/__main__.py`/`cli.py`, `pyproject.toml` (`[project.scripts]`).
- **Effort:** Small–Medium.

### P0.3 — Config/setup ergonomics (`.env.example` + first-run docs)
- **Kyun:** Provider default `ANTHROPIC` hai par koi committed `.env`/example nahi; new user
  ko nahi pata key kaise deni hai. (`allow_mock_fallback=True` — bina key ke mock chalega.)
- **Kya:** `.env.example` (`ANTHROPIC_API_KEY=`, `NYXARA_PROFILE=dev`,
  `NYXARA_LLM__PROVIDER=anthropic`, etc.) + ek chhota "Quickstart" section README mein
  (venv → install → set key → `python -m nyxara`).
- **Files:** naya `.env.example`, `README.md`.
- **Effort:** Small.

---

## 3. P1 — Polish / usability

### P1.1 — README / architecture overview
- **Kyun:** Code mein rich docstrings hain par koi top-level README/design doc nahi. Naye
  contributor ya khud owner ke liye onboarding mushkil.
- **Kya:** Project kya hai, Eight Rules, subsystem map, quickstart, "kaise extend karein".
- **Files:** naya `README.md` (pyproject already isko reference karta hai).
- **Effort:** Small.

### P1.2 — Deployment story
- **Kyun:** Koi Dockerfile / run script / service wrapper nahi.
- **Kya:** `Dockerfile` + minimal `make`/script targets (`install`, `test`, `run`).
- **Effort:** Small–Medium.

### P1.3 — Action dispatch (act stage real banao)
- **Kyun:** ACT stage abhi effect ko sirf **record** karta hai
  (`orchestrator.py`: "a real deployment dispatches to agency.tools here; we record the effect").
  Cleared action asli tool ko call nahi karta.
- **Kya:** Cleared `Candidate` ko `nyxara/agency/tools.py` (ToolRegistry) se jodo, deadline +
  journalling ke saath.
- **Files:** `nyxara/kernel/orchestrator.py` (ACT block ~line 237–259), `nyxara/agency/tools.py`.
- **Effort:** Medium.

---

## 4. P2 — Future work (bug nahi, intentional)

### P2.1 — Abstract-method stubs (koi action zaroori nahi)
Ye `NotImplementedError` **base-class interfaces** hain (`# pragma: no cover - abstract/overridden`),
inke concrete subclasses mojood hain — ye **missing features nahi**:
`mind/lot.py`, `mind/moral.py`, `mind/proposal.py`, `mind/reasoner.py`, `mind/faculties.py`,
`sim/montecarlo.py`. Naya reasoning-mode chahiye to inhe extend karna; abhi kuch toота nahi.

### P2.2 — Heavy-ML senses (optional)
`senses/` (vision/audio/document via torch/whisper/opencv/pdfplumber) abhi gracefully degrade
karta hai (verified). Asli multimodal perception chahiye to optional extras install karke
in code-paths ko exercise + test karna.

---

## 5. Suggested order

1. **P0.1** asli reasoner wire → echo khatam, asli jawab.
2. **P0.2** CLI/REPL → user baith ke baat kar sake.
3. **P0.3 + P1.1** `.env.example` + README → koi bhi setup karke chala sake.
4. **P1.3** real action dispatch → "act" sach mein kuch kare.
5. P1.2 deployment, phir P2 jab zaroorat ho.

---

*Verification basis: clean venv, `python -m nyxara.kernel.orchestrator` (PASS) +
`python -m pytest` (2874 passed). Koi source feature is roadmap ke tahat abhi implement nahi
kiya gaya — ye sirf plan hai.*
