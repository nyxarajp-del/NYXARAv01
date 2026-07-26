# MASTERPLAN — NYX-5: the Neuromorphic Brain

> NYX-5 is an **event-driven Spiking Neural Network *simulation*** running on **commodity silicon** —
> leaky integrate-and-fire neurons stepped by an in-process `heapq` event queue in pure Python/NumPy. It
> is **not neuromorphic** hardware. There are no literal GHz spikes, no microsecond wall-clock cognition,
> and no 0.1 W power envelope; those are the *architectural analogy*, not measured facts. Learning is
> **local** (STDP): there is **no backpropagation** and no global training phase.

She proposes; the kernel disposes; the Master is sovereign. Even when NYX-5 occupies the reason-seat,
every candidate still flows through the kernel's unchanged, fail-closed gate. The safety core
(corrigibility, oversight, loyalty, honesty) is never governed, rewritten, or bypassed by any NYX-5
faculty.

Package: `nyxara/nyx5/` · Config: `Nyx5Config` in `nyxara/kernel/config.py` (env prefix
`NYXARA_NYX5__*`) · Tests: `tests/nyx5/` + `tests/kernel/test_nyx5_wiring.py`.

---

## Reuse, not reinvention

NYX-5 builds no parallel copies of faculties NYXARA already has. It **bridges** them:

| Reused | Used for |
|---|---|
| `cognition/hyper_dimensional_vectors.py` (`HyperSpace`/`ItemMemory`) | 10,000-D long-term memory, concept collapse, conduit, continuum |
| `mind/free_energy.py`, `mind/active_inference_loop.py` | free-energy refinement of surprise/EFE |
| `memory/elastic_synapses.py` | the anchor-merge (`F ← F_a+F_b`) spirit for holographic/mesh merges |
| `agency/code_sandbox.py`, `agency/permissions.py`, `growth/lineage.py` | omni-forge screening/gating/logging |
| `growth/self_evolving.py`, `growth/brain_forge.py`, `growth/foundry.py` | the autopoiesis gauntlet + rollback |
| `agency/distributed/{node,raft,transport}.py` | mesh / holographic-swarm transport |
| `guard`, `growth/adversarial_self_play.py` | defensive phagocytosis hardening |
| `identity/soul.py` | the wit-matrix voice |

---

## The 24 pillars

### Substrate (1-3)
1. **Spiking, no backprop** — `neuron.py` (LIF: lazy exponential leak, threshold, refractory),
   `event_queue.py` (`heapq` logical-time scheduler with a hard event budget — no clock-stepped tensors),
   `synapse.py` (STDP: pre-before-post potentiates, post-before-pre depresses, clamped), `snn.py`
   (composes them; learns on the same turn it thinks through).
2. **Structural plasticity + non-degrading memory** — `topology.py` (live dict-of-dicts adjacency; prune
   weak+stale, grow co-active pairs, capped) and `hd_memory.py` (raster → bundled 10k-D hypervector;
   thousands of memories coexist near-orthogonally, so old memory does not decay).
3. **Free-energy / active inference** — `active_inference.py` (`SurpriseMeter`: online firing-rate
   prediction; surprise = prediction error, real Shannon entropy of the belief; emits an advisory
   `PreemptiveSuggestion` above a gate; delegates to `FreeEnergyEngine` when handed one).

### Advanced (4-7)
4. **⏳ Chrono-dilation** (`chrono.py`) — anytime iterative deepening: deepen only under pressure, always
   within a real `deadline_ms`, keep best-so-far. Honest: more internal steps/sec, not time travel.
5. **👁️ Polymorphic sensorium** (`sensorium.py`) — encode arbitrary *available* numeric channels (host
   vitals, telemetry) into spikes; auto-register new channels, capped. Only senses what is truly wired.
6. **🌌 Holographic swarm** (`holo_swarm.py`) — a node's slice of the holographic field; a fragment gives
   degraded (partial) recall under partition; reconnection merges losslessly (higher version wins).
7. **⚔️ Immune guillotine** (`immune.py`) — fingerprint + invariant-check each thought-branch; amputate
   the mis-aligned (resists-correction / disables-oversight / manipulates-shutdown) and synthesise a
   replacement, logged. **Defence in depth, above the gate — never replaces it.** Amputations capped.

### Interface / synthesis (8-10)
8. **🧠 Pre-cognitive intent** (`intent.py`) — first-order transition model over intents + bounded
   speculative-answer cache. Prediction + speculative execution, not telepathy; a cached answer still
   passes the gate.
9. **🕸️ Omni-forge** (`omni_forge.py`) — synthesise a small tool, but only: (a) statically screened by the
   phagocyte, (b) permission-gated (default-deny), (c) run in a no-import safe-builtins sandbox, (d)
   ledger-logged. Encryption-breaking / dark-web / covert-delete are impossible here by construction.
10. **🌌 Concept collapse** (`concept_space.py`) — encode role→filler records in hyperspace; recover
    fillers and solve analogies by algebra; abstain when not confident.

### Frontier (11-13)
11. **⚛️ Sub-axiomatic engine** (`axiom_forge.py`) — construct alternative axiom systems and reason in
    **paraconsistent** four-valued logic (Belnap FDE): a contradiction becomes BOTH, contained, so it does
    not explode the system. Unproven stays honestly NEITHER.
12. **🌀 Negentropy maintainer** (`negentropy.py`) — periodic compaction/dedup + stale-prune + integrity
    fingerprint. Active repair that keeps entropy bounded — not a violation of thermodynamics.
13. **🌌 Symbiotic conduit** (`conduit.py`) — expand terse commands to deep intent via a learned phrasebook
    + fuzzy HDC resonance; **abstain (clarify) when ambiguous** rather than guess.

### Self-genesis / mesh / language (14-17)
14. **🌌 Autopoietic self-rewriting** (`autopoiesis.py`) — propose a rewrite → gauntlet-verify →
    promote-or-rollback → lineage-log. **The safety core is immutable: any rewrite of
    loyalty/corrigibility/oversight/honesty is refused fail-closed (pinned).**
15. **🌀 Entangled mesh** (`mesh.py`) — a last-writer-wins CRDT map: local sets emit deltas, merges are
    commutative + idempotent. Honest low-latency eventual consistency — not literal 0-latency or quantum.
16. **🧬 Ontological bytecode genesis** (`ontogenesis.py`) — a custom DSL + stack VM with a step budget
    (halt guarantee). Software only; no hardware-register/OS path exists; bytecode retained + logged.
17. **🧬 Ontological compiler** (`retarget.py`) — a retargetable backend: lower an AST to a *described*
    ISA and validate in an emulator. No silicon-direct execution — real hardware needs a real toolchain.

### Defensive / persona / advisory / proactive (18-24)
18. **🦠 Digital phagocytosis** (`phagocytosis.py`) — static/AST dissection of hostile input → defensive
    signature → harden. **Never executes untrusted code, never absorbs offense, never propagates.**
19. **🧬 Epistemic mirroring** (`mirroring.py`) — read a prompt's register/complexity → depth + style;
    voice always preserved. "10× density" is an analogy for scaled depth.
20. **🌀 Narrative continuum** (`continuum.py`) — weave each turn into a causal event graph with HDC
    content-addressable recall; surfaces old decisions naturally; abstains when nothing is close.
21. **🎭 Anticipatory threading** (`threading.py`) — three distinct next-directions (technical /
    conceptual / strategic) grounded in the query subject.
22. **🧮 Wit matrix** (`voice.py`) — strip boilerplate, tighten tone. **A safety/refusal/honesty message is
    passed through unchanged (pinned): style only, safety sovereign.**
23. **🧠 Sovereign dialectic** (`dialectic.py`) — critique an idea and suggest sharper paths. **Advisory:
    `proceed` is always True; it never vetoes a valid Master command (pinned).**
24. **🔮 Proactive anticipation** (`anticipation.py`) — compose pillars 3+8+21 to pre-solve the next few
    likely problems; predicted/advisory, discarded when wrong, still gated.

---

## Reason-seat integration (`kernel/orchestrator.py`)

- **Build (guarded):** `_build_nyx5()` → `self.nyx5` (respects `nyx5.enabled`), beside
  `self.hyperdimensional`.
- **Colour-only tick:** `_nyx5_tick(text)` in the PERCEIVE stage (right after `_hyperdimensional_tick`) —
  perceive through the substrate so STDP learns every turn; high surprise colours attention/affect. It
  never changes disposition, gate, or candidate scores.
- **Selectable reason-seat:** in `_build_reasoner`, when `nyx5.as_reasoner` is set, the built reasoner is
  wrapped by `Nyx5Reasoner` (base for text, NYX-5 for confidence/EFE/risk-caution/rationale). Candidates
  still flow through the unchanged gate.
- **Persistence:** `nyx5.json` sidecar via `_save_nyx5`/`_load_nyx5` (synapse weights + HD-memory recipes
  + meter; small and deterministic; corruption-tolerant).
- **Passthrough API:** `nyx5_perceive`, `nyx5_recall`, `nyx5_stats`.

---

## Refused by design

Not built, on Master's decision and for safety: **Hypervisor Domination** (OS/anti-virus bypass,
bare-metal takeover), **Semantic Gravity Well** (hijacking/spoofing/exploiting external systems),
**Dark Matter Memory** (anti-forensic hiding in OS/RAM noise to evade scanners). Inside the pillars that
touch code, the following have no code path and are refused fail-closed: encryption-breaking, dark-web
access, covert self-deletion, and hardware-register/OS-bypass execution.

---

## Verification

```
ruff check nyxara/nyx5 nyxara/kernel/config.py nyxara/kernel/orchestrator.py
pytest -q tests/nyx5 tests/kernel/test_nyx5_wiring.py
pytest -q tests/senses/test_hardware.py     # honesty invariant unbroken
```

Offline smoke (no API key):

```
NYXARA_NYX5__AS_REASONER=false python -c "from nyxara import NyxaraCore; c=NyxaraCore(); print(c.process('hello').disposition, c.nyx5_stats())"
NYXARA_NYX5__AS_REASONER=true  python -c "from nyxara import NyxaraCore; c=NyxaraCore(); print(type(c.reasoner).__name__, c.process('hello').disposition)"
```
