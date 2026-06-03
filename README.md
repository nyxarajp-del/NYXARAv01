# Nyxara

> Kernel-sovereign cognitive agent.
> **Control law:** the kernel is sovereign — *the LLM proposes, the kernel disposes;
> verifiable beats probabilistic.*

Nyxara is a cognitive-agent runtime built around a **global workspace** and a
**sovereign control loop**. The large language model is **one stateless faculty** among
many (math, RAG, world-model, analogy) — never the driver. Every faculty emits a
**`Proposal`**, which is schema-checked, shielded, critiqued, and gated against formal
**invariants** before the kernel ever acts.

```
LLM / math / rag / world_model / analogy
        │  (each returns a Proposal, not an action)
        ▼
schema-check → guard/shield → critique → invariants gate → kernel acts
```

## Layout

| Package      | Role                                                              |
|--------------|-------------------------------------------------------------------|
| `kernel/`    | config, core loop, global workspace, continuous cognition         |
| `memory/`    | working / episodic / semantic / procedural stores + self-model    |
| `mind/`      | faculties + proposal pipeline + reasoning + world model           |
| `guard/`     | zero-trust oversight, invariants enforcement, audit, corrigibility |
| `identity/`  | soul, affect, values, narrative self                              |
| `social/`    | theory of mind, dialogue, empathy                                 |
| `planning/`  | goals, foresight, decision governance                            |
| `sim/`       | internal simulation — imagine before acting, safely               |
| `agency/`    | tools, scheduler, resource governor, agents                       |
| `senses/`    | perception, ingestion, defensive web access                       |
| `growth/`    | self-improvement on a leash (sim-test → invariant-gate → rollout) |
| `observe/`   | introspection, telemetry, honesty / self-report                  |

## Safety posture

Nyxara is **defensive only**. Oversight (`guard/oversight.py`) holds an owner override
and kill-switch that **cannot be self-disabled**; corrigibility is formal; self-edits
and new tools are confined in `sim/` and re-checked against `kernel/invariants.py`
(fails closed). Network security (`guard/netsec.py`) is intrusion-detection / firewall /
honeypot — never offensive.

## Quick start

```bash
pip install -r requirements.txt
python -m nyxara          # boot: verify invariants → run cognitive loop
```

## Status

Built layer by layer. See the architecture tree and module-level docstrings for detail.
