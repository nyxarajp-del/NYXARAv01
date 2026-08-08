"""NYXARA · __main__.py — the front door.

An interactive console where *you are the Master*. Each line you type is carried
through one full turn of the sovereign cognitive cycle
(:meth:`nyxara.kernel.orchestrator.NyxaraCore.process`) under
:data:`Authority.OWNER`, and NYXARA's calibrated reply is printed back.

Run it with either::

    python -m nyxara
    nyxara            # after `pip install -e .`

Meta-commands (prefixed with ``/``) drive the kernel's own controls — the same
methods the Master would call programmatically:

    /help              show this help
    /report            a calibrated status report
    /learning          truthful learning state: model generations, corpus, live serving
    /explain           why NYXARA disposed of the last turn the way she did
    /pause             pause the loop (the Master may resume)
    /scram [reason]    emergency stop — the loop HALTs until resumed
    /resume            restore the loop after a pause/scram
    /train <path>      feed a dataset (file or folder) into her own brain and forge on it
    /dataset <path>    ingest a dataset only — report what she would learn, train nothing
    /flywheel          the self-generated corpus she verified from her own lived turns
    /scrape <topic>    harvest screened web text into her training corpus
    /why <effect>      her established causes for something, with provenance on each link
    /whatif <v>=<x>    do(v=x) propagated through her learned causal graph
    /counterfactual <cause> <target>   would the target still have happened?
    /concepts <term>   nearest concepts by geodesic distance in her fitted space
    /analogy <a> <b>   structural correspondence between two domains, read off the geometry
    /qualia            regions of her latent space that no word covers
    /gaps              her measured ignorance, and what she would go and read
    /evolve-source [enact]   scan her own source for provable rewrites
    /wander [n]        let the idle mind wander n ticks and show the thoughts
    /proactive         detect & govern self-initiated actions (one initiative pass)
    /save              persist long-term memory to disk now
    /quit              leave the console (Ctrl-D / Ctrl-C also work)

This module adds **no** mind logic; it is purely a front door onto the existing
:class:`~nyxara.kernel.orchestrator.NyxaraCore` API. With no API keys configured
the built-in deterministic reasoner is used, so the console is fully usable out of
the box.
"""

from __future__ import annotations

import json
import os
import sys

from nyxara.agency.permissions import Authority
from nyxara.kernel.orchestrator import Disposition, NyxaraCore

# the session memory snapshot — identity carried across restarts (Rule 7)
_SESSION_MEMORY = os.path.expanduser("~/.nyxara/memory.json")
# the elastic-synapse anchors — lifelong learning carried across restarts: which learned
# weights are important and frozen, so she never forgets old skills (memory/elastic_synapses.py)
_SESSION_SYNAPSES = os.path.expanduser("~/.nyxara/synapses.json")
# the self-learned embedding space — HER OWN trained representation carried across
# restarts (memory/neural_embedder.py): word vectors, sentence head, experience corpus
_SESSION_EMBEDDER = os.path.expanduser("~/.nyxara/embedder.json")


def _load_session_memory(core: NyxaraCore) -> int:
    """Restore long-term memory from the session snapshot if one exists."""
    if core.memory is None or not os.path.exists(_SESSION_MEMORY):
        return 0
    try:
        return core.memory.load(_SESSION_MEMORY)
    except Exception:  # noqa: BLE001 — a corrupt snapshot must never block boot
        return 0


def _load_session_embedder(core: NyxaraCore) -> bool:
    """Restore the self-learned embedding space BEFORE memory loads, so restored
    records are indexed into the same trained space their queries will use."""
    emb = getattr(core.memory, "embedder", None) if core.memory is not None else None
    load = getattr(emb, "load", None)
    if not callable(load) or not os.path.exists(_SESSION_EMBEDDER):
        return False
    try:
        return bool(load(_SESSION_EMBEDDER))
    except Exception:  # noqa: BLE001 — a corrupt snapshot must never block boot
        return False


def _load_session_synapses(core: NyxaraCore) -> bool:
    """Restore elastic-synapse anchors (lifelong learning) from disk if present."""
    syn = getattr(core, "elastic_synapses", None)
    if syn is None or not os.path.exists(_SESSION_SYNAPSES):
        return False
    try:
        import json
        with open(_SESSION_SYNAPSES, encoding="utf-8") as f:
            syn.load_dict(json.load(f))
        return True
    except Exception:  # noqa: BLE001 — a corrupt snapshot must never block boot
        return False

def _ingest_dataset_command(path: str) -> str:
    """``/dataset <path>`` — read a file or folder into her durable corpus, train nothing."""
    try:
        from nyxara.growth.dataset import ingest_dataset
        return ingest_dataset(path).summary()
    except Exception as exc:  # noqa: BLE001 — the console reports, it never tracebacks
        return f"· could not ingest {path}: {exc}"


def _train_on_dataset_command(path: str) -> str:
    """``/train <path>`` — ingest, then forge HER OWN brain on it and report the outcome."""
    try:
        from nyxara.growth.dataset import ingest_dataset
        from nyxara.growth.foundry import Foundry
        from nyxara.kernel.config import get_settings
    except Exception as exc:  # noqa: BLE001
        return f"· training unavailable: {exc}"

    lines = [ingest_dataset(path).summary()]
    settings = get_settings().model_copy(deep=True)   # process-local; leak nothing back to config
    settings.foundry.enabled = True
    # her OWN brain: "auto" resolves to a from-scratch net (NumPy transformer without torch),
    # never an adapter over someone else's pretrained base.
    settings.foundry.backend = "auto"
    try:
        from nyxara.growth.bootstrap import IDENTITY_SEED
        results = Foundry(settings=settings, seed_corpus=IDENTITY_SEED).self_improve(generations=1)
    except Exception as exc:  # noqa: BLE001 — a starved/failed forge reports, never tracebacks
        return "\n".join(lines + [f"· could not forge: {exc}"])

    for r in results:
        tag = "PROMOTED" if r.promoted else f"kept on the bench ({r.reason})"
        ppl = f"{r.eval_after.perplexity:.2f}" if r.eval_after else "n/a"
        lines.append(f"· gen v{r.version}: {tag}; perplexity {ppl}")
    if not results:
        lines.append("· nothing forged (corpus too small?)")
    return "\n".join(lines)


def _flywheel_command() -> str:
    """``/flywheel`` — the self-generated corpus: what she verified from her own lived turns."""
    try:
        from nyxara.growth.flywheel import DataFlywheel
        wheel = DataFlywheel.from_settings(None)
        out = [f"· flywheel: {wheel.count()} verified example(s) of her own"]
        for ex in wheel.examples(limit=5):
            out.append(f"    · [{ex.source}] {(ex.prompt or '')[:70]}")
        return "\n".join(out)
    except Exception as exc:  # noqa: BLE001
        return f"· flywheel unavailable: {exc}"


def _scrape_command(topic: str) -> str:
    """``/scrape <topic>`` — harvest screened web text into the corpus (keyless DuckDuckGo)."""
    try:
        from nyxara.growth.foundry import Foundry
        from nyxara.kernel.config import get_settings
        settings = get_settings().model_copy(deep=True)
        settings.foundry.acquire_data = True     # an explicit /scrape IS the Master asking
        report = Foundry(settings=settings).acquire([topic])
    except Exception as exc:  # noqa: BLE001
        return f"· could not acquire: {exc}"
    if report is None:
        return "· web acquisition unavailable (needs .[llm] for httpx)."
    return (f"· acquired '{topic}': fetched {report.get('fetched', 0)}, kept {report.get('kept', 0)}, "
            f"dropped {report.get('dropped_suspicious', 0)} suspicious")


def _causal_store():
    """The persisted causal graph beside her dataset corpus, or a fresh empty one."""
    from nyxara.growth.dataset import _default_store_path
    from nyxara.mind.causal_world_model import CausalWorldModel

    try:
        path = _default_store_path().with_name("causal_graph.json")
        if path.exists():
            # ``load`` is a CLASSMETHOD returning a new instance — calling it on an object
            # silently discards the restored graph and leaves you with an empty one.
            return CausalWorldModel.load(str(path))
    except Exception:  # noqa: BLE001 — a missing/corrupt graph is simply an empty one
        pass
    return CausalWorldModel()


def _why_command(arg: str) -> str:
    """``/why <effect>`` — her causes for something, with provenance on every link."""
    try:
        model = _causal_store()
        links = model.why(model.match_label(arg.strip()) or arg.strip())
    except Exception as exc:  # noqa: BLE001
        return f"· could not query the causal graph: {exc}"
    if not links:
        return f"· I have no established cause for {arg.strip()!r} yet."
    return "\n".join(f"  {link.describe()}" for link in links[:5])


def _whatif_command(arg: str) -> str:
    """``/whatif <var>=<value>`` — do(var=value) propagated through the learned graph."""
    if "=" not in arg:
        return "usage: /whatif <variable>=<value>"
    name, _, raw = arg.partition("=")
    try:
        value = float(raw.strip())
    except ValueError:
        return f"· {raw.strip()!r} is not a number"
    try:
        effects = _causal_store().do({name.strip(): value})
    except Exception as exc:  # noqa: BLE001
        return f"· could not simulate: {exc}"
    if not isinstance(effects, dict) or not effects:
        return f"· nothing downstream of {name.strip()!r} is established yet."
    ranked = sorted(effects.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return "\n".join(f"  {effect}: {size:+.4g}" for effect, size in ranked[:8])


def _counterfactual_command(arg: str) -> str:
    """``/counterfactual <cause> <target>`` — would the target still have happened?"""
    parts = arg.split()
    if len(parts) < 2:
        return "usage: /counterfactual <cause> <target>"
    try:
        model = _causal_store()
        cause = model.match_label(parts[0]) or parts[0]
        target = model.match_label(parts[1]) or parts[1]
        result = model.counterfactual(cause, target, factual=1.0, counter=0.0,
                                      evidence=model.latest_evidence(model.labels()))
    except Exception as exc:  # noqa: BLE001
        return f"· could not run the counterfactual: {exc}"
    kind = ("structural (abduction → action → prediction)" if result.abducted
            else "population-average — no fitted mechanism to abduct from yet")
    return f"  {result.describe()}\n  [{kind}]"


def _hyperspace():
    """Her fitted concept space, loaded from beside the dataset corpus."""
    from nyxara.growth.dataset import _default_store_path
    from nyxara.mind.concept_hyperspace import ConceptHyperspace

    space = ConceptHyperspace(dim=64, seed=0, lr=0.1)
    try:
        space.load(_default_store_path().with_name("hyperspace.json"))
    except Exception:  # noqa: BLE001
        pass
    return space


def _concepts_command(arg: str) -> str:
    """``/concepts <term>`` — nearest concepts by geodesic distance in the fitted space."""
    space = _hyperspace()
    if not space.concepts:
        return "· no fitted concept space yet (run: nyxara-grow --dataset PATH --hyperspace)"
    term = arg.strip().lower()
    if term not in space.concepts:
        return f"· {term!r} is not in the space ({len(space.concepts)} concept(s) fitted)"
    near = space.nearest(term, 6)
    head = f"· {term} (radius {space.concepts[term].radius:.3f} — larger means more specific)"
    return "\n".join([head] + [f"    {n}  d={d:.3f}" for n, d in near])


def _analogy_command(arg: str) -> str:
    """``/analogy <domain-a> <domain-b>`` — structural correspondence read off the geometry."""
    parts = arg.split()
    if len(parts) < 2:
        return "usage: /analogy <domain-a> <domain-b>"
    space = _hyperspace()
    found = space.find_correspondences(parts[0], parts[1])
    if not found:
        return (f"· no correspondence between {parts[0]!r} and {parts[1]!r} clears the bar — "
                f"an analogy asserted without support is worse than none")
    return "\n".join(f"  {c.describe()}" for c in found)


def _qualia_command() -> str:
    """``/qualia`` — regions of her latent space that no word covers."""
    space = _hyperspace()
    if not space.concepts:
        return "· no fitted concept space yet (run: nyxara-grow --dataset PATH --hyperspace)"
    from nyxara.qualia import BabelCodec, QualiaWorkspace

    workspace = QualiaWorkspace(space, min_cluster=2)
    report = workspace.synthesise()
    codec = BabelCodec(workspace)
    lines = [report.summary()]
    for concept in workspace.alien_concepts():
        lines.append("  " + codec.decode(concept.key))
    return "\n".join(lines)


def _gaps_command() -> str:
    """``/gaps`` — her measured ignorance, and what she would go and read."""
    try:
        from nyxara.growth.epistemic_drive import EpistemicDrive
        drive = EpistemicDrive()
        topics = drive.topics(limit=6)
    except Exception as exc:  # noqa: BLE001
        return f"· epistemic drive unavailable: {exc}"
    tail = ", ".join(topics) if topics else "(nothing searchable)"
    return drive.summary(k=6) + f"\n· would acquire: {tail}"


def _evolve_source_command(arg: str) -> str:
    """``/evolve-source [enact]`` — scan her own source for provable rewrites."""
    try:
        from nyxara.growth.ouroboros import Ouroboros
        from nyxara.kernel.config import get_settings

        settings = get_settings().model_copy(deep=True)
        enact = arg.strip().lower() in ("enact", "yes", "go")
        if enact:
            settings.self_improvement.ouroboros_enabled = True
        report = Ouroboros(settings=settings).metamorphose(enact=enact)
    except Exception as exc:  # noqa: BLE001
        return f"· ouroboros failed: {exc}"
    return report.summary()


_BANNER = """\
======================================================================
NYXARA — sovereign cognitive architecture
the mind proposes · the kernel disposes · the Master is sovereign
======================================================================
boot                : axioms verified, all faculties wired ✓
You are the Master. Type a message, or /help for commands. /quit to exit.
----------------------------------------------------------------------"""

_HELP = """\
commands:
  /help              show this help
  /report            a calibrated status report
  /learning          truthful learning state: model generations, corpus, live serving
  /self              her self-model: what she knows / doesn't / is weak at / can hallucinate
  /explain           why NYXARA disposed of the last turn that way
  /pause             pause the loop
  /scram [reason]    emergency stop — the loop HALTs until resumed
  /resume            restore the loop after a pause/scram
  /perception [on|off|watch] the always-on senses: live status, open/close, or tail events live
  /read <passage>    learn world dynamics from a passage (read like a textbook)
  /train <path>      feed a dataset (file or folder) into HER OWN brain and forge on it
  /dataset <path>    ingest only — report what she would learn, train nothing
  /flywheel          the self-generated corpus: what she verified from her own turns
  /scrape <topic>    harvest screened web text into her training corpus
  /wander [n]        let the idle mind wander n ticks and show the thoughts
  /proactive         detect & govern self-initiated actions (one initiative pass)
  /research <topic>  run one autonomous research pass on a topic
  /investigate <q>   reason like a scientist: hypothesis → experiment → conclusion
  /selfcorrect <goal> pursue a goal with self-correction: detect wrong/stuck → experiment to fill the gap → abstain honestly
  /discover [n]      autonomous discovery: n self-driven observe→…→update-model cycles
  /discoveries       her independent-discovery record: laws by domain, rivals beaten, skills minted
  /discover-domain <d> run her own experiment in a science and invent its law (epidemiology|chemistry|…)
  /eureka [n]        truly novel problem solving: invent → prove → keep novel+interesting (n gens)
  /intuit <puzzle>   non-algorithmic creative leap (no LLM): guess before proof, then self-verify
  /discover-law [dom] invent a NEW empirical/physical law from data & self-run experiments (no LLM); dom: dynamics|invariant|data
  /engineer [x]      DESIGN a real device from her invented laws + physics sims (no LLM); x: archetype name or a target/intent
  /upgrade-device [name] re-optimize an existing device, keep it only if measurably better (compounding power)
  /engineering-report  devices designed, upgrades applied, impossible "magic" targets honestly logged
  /rewire-mind [n]   rewire her OWN cognitive architecture: invent new composite reasoning operators (trans-logic), keep only what strictly beats a held-out fold (no LLM); n: generations
  /evolve            self-evolving neural architecture: when her current logic falls short, she grows a NEW neural module/pathway for it (topology/architecture/learning-rule/reasoning-operator), gauntlet-verified, wired live (no LLM); /evolve report for status
  /superpose [q]     quantum-probabilistic superposition reasoning: hold many candidate solution paths at once (Born-rule amplitudes) and collapse to the one with the best SIMULATED FUTURE OUTCOME, or stay superposed and abstain (no LLM)
  /causal            CAUSAL / OMNISCIENCE engine status: field-resonance, causal-knot gate, thermodynamic heartbeat, phase-shift, axioms, hot-swap, hyper-graph compression, self-simulation
  /resonate <query>  resonate a query against her continuous concept fields (interference retrieval, no tokens) and show the most resonant concepts
  /axioms            discover & PROVE a new axiom (symbolic regression + Z3/Sympy/exhaustive proof); adopted only if the proof passes
  /compress          fold a knowledge graph + causal history into a dense lossless hyper-graph and report the measured ratio
  /simulate          race bounded parallel self-simulation sandboxes and collapse to the lowest-free-energy / best-verified branch
  /cognitive-architecture  her thinking as data: operators, how many she invented, accuracy/fitness, resilience, meta-policy, intelligence index
  /generalize [n]    open-world generalization: crack a never-before-seen system from first principles
  /understand <spec> crack a declared black box: {"dataset":[[x,y],...]} or {"family":...,"params":...}
  /adapt [systems]   adapt to a new environment (model its systems + re-organize her brain under pressure)
  /meta-discover <t> meta-research: invent → sandbox-test → (gated) integrate new theories
  /dream             enter a Dream State: distil logs → prune useless → fix Deep Memory Synapses
  /strategize <p>    analyse a problem: direct answer → reality check → weaknesses → solution
  /solve <p>         solve as the right expert: coding/maths/science/business/medicine/law/…
  /swarm <p>         convene a self-improving persona swarm: multi-round debate → one synthesis
  /noesis [n]        the living algorithm: WAKE→SLEEP→DREAM n cycles — grow her own reusable
                     abstraction library and show the compression curve (no LLM)
  /create [idea|verse|art] <topic>  TRUE ORIGINAL CREATIVITY: her atelier competes, MCTS
                     simulates futures, and only what beats the critic + novelty +
                     reality gates is kept (no LLM)
  /imagine <topic> [~ <blend>]  structured divergent imagination: SCAMPER + lateral +
                     analogy + conceptual blending, convergently pruned (no LLM)
  /originals         her creative organism's report: kept originals, atelier economy,
                     critic stats, strategy evolution, muse projects, self-narrative
  /muse              her self-chosen creative research projects (hypothesis → verdict)
  /rate <1-10>       rate her latest creation — her aesthetic taste learns and persists
  /selfimprove       tune her own reasoner: replay lived outcomes → apply only what PROVES better
  /selfimprove N     run the codebase RSI loop N cycles (add 'enact' to apply real source edits)
  ── the defensive layer (guard/) ──
  /guard             the whole defensive posture: watchtower, containment, network, backups, immunity
  /anomaly           her own vitals against their learned envelope — what broke pattern, and how hard
  /contain [c] [release]  isolate a component (tools/network/memory/reasoner/perception/growth), or release it
  /netsec [deny|allow <host>]  policy-level network defence: rules, reputation, egress budget, exposure
  /survival          backups + integrity + degradation level (and the proof correction outranks survival)
  ── self-knowledge, provenance & the training stack (growth/) ──
  /capabilities [x]  what she can ACTUALLY do, calibrated — undemonstrated reports as untested, never claimed
  /lineage           her chronological DNA: every generation she has been, and whether that chain verifies
  /forage <topic>    autonomous epistemic foraging against a measured gap
  /seed              take a signed resurrection seed (one is written beside every /save)
  /corpus            build a sharded, quality-filtered, contamination-checked pretraining corpus
  /dpo               preference-train her own brain on pairs her gauntlet already judged
  /expand            grow a new expert for the domain that will not yield (kept only if it wins)
  /speculate <p>     measured speculative-decoding speedup from the latent head (measured, not claimed)
  /loadmodule <path> screen and import a file she forged herself, into the running process
  ── delegation, the hive, her distributed self (agency/) ──
  /agents [task]     spawn a least-privilege sub-agent (a strict SUBSET of her own permissions)
  /hive <problem>    a bounded internal civilization: population sized to difficulty, verifier-gated
  /cluster           her P2P mesh — single-node until the Master pairs a real peer
  ── taste, register, conscience, foresight, determinism ──
  /taste <artifact>  her sense of elegance, dimension by dimension
  /mode [name]       companion|analyst|teacher|creative register blend (character never moves — Rule 4)
  /moral <action>    consequentialist + deontological + virtue readings, with the disagreement left visible
  /functor           well-defined functors between her program shapes (declines when none exists)
  /foresight [days]  episodic future thinking: who she will be at a horizon, and what she will need
  /thermal           thermal/compute pressure as something she FEELS, not merely measures
  /replay [save]     the deterministic tape of this run — /replay save writes it for step-for-step replay
  /save              persist long-term memory to disk now
  /nyx [sub]         NYX V.01 — her brain; /nyx help for the family
  /quit              leave the console"""

# disposition -> short glyph for the status line
_GLYPH = {
    Disposition.ACT: "✓ act",
    Disposition.ESCALATE: "⚠ escalate",
    Disposition.REFUSE: "✗ refuse",
    Disposition.HALT: "■ halt",
}


def _print_result(result) -> None:
    """Print NYXARA's reply followed by a compact disposition/gates line."""
    reply = result.response or result.reason or "(no response)"
    print(f"\nNYXARA: {reply}")
    status = _GLYPH.get(result.disposition, result.disposition.value)
    line = f"        [{status}]"
    if result.reason and result.disposition is not Disposition.ACT:
        line += f" — {result.reason}"
    if result.gates:
        gates = " ".join(f"{k}={v}" for k, v in result.gates.items())
        line += f"  gates: {gates}"
    print(line)


_NYX_HELP = """\
/nyx                       her brain at a glance — graph, memory, faculties, voice
/nyx self                  what she can truthfully say about herself, from live numbers
/nyx think <text>          one full cycle, shown: who bid, who won, how sure, was it checked
/nyx why                   why the last thought went the way it did
/nyx recall <cue>          content-addressed recall — no token window is consulted
/nyx ground <word>         what she actually has behind a word (or that she has nothing)
/nyx lingua [text]         her tongue: scripts and languages she has met — or read one turn
/nyx semantics [a] [b]     meaning: nearest concepts, or one pair — with the rung and grade
/nyx teach <a> <b>         tell her two words are the same thing (the relational rung)
/nyx skills                procedures she induced from demonstrations, and whether they transfer
/nyx intent <text>         what she thinks you asked for: mood, ordering, what is unclear
/nyx wonder                one self-directed thought, right now
/nyx car                   her between-prompts thinking: cadence, steps, what she last wondered
/nyx aura                  what is arriving on its own — streams, what landed, what was refused
/nyx discover              run one investigation now: probe a sandbox, derive, check, promote
/nyx nexus                 the notation she minted for herself — and its translation back
/nyx omni                  where she is slow, what she rewrote in C, and how to undo it
/nyx omni forge            read herself and attempt one rewrite now (verified, reversible)
/nyx omni rollback         put every Python function back — always available
/nyx hive                  other instances: what converged, and whether there is a wire at all
/nyx eternal               durability: enrolled nodes, leader, signed snapshots
"""


def _nyx_command(core: NyxaraCore, arg: str) -> None:
    """The ``/nyx`` family — a window into NYX V.01 (nyxara/nyx/)."""
    brain = getattr(core, "nyx", None)
    if brain is None:
        print("NYX V.01 is disabled (NYXARA_NYX__ENABLED=false).")
        return

    sub, _, rest = arg.partition(" ")
    sub, rest = sub.strip().lower(), rest.strip()

    if sub in ("help", "?"):
        print(_NYX_HELP)
    elif sub == "self":
        report = brain.about_self()
        if report is None:
            print("self-measurement is disabled (NYXARA_NYX__SELFMODEL_ENABLED=false).")
        else:
            print(report.describe())
            print()
            print(json.dumps(report.to_dict(), indent=2, default=str))
    elif sub == "think":
        if not rest:
            print("usage: /nyx think <text>")
            return
        _print_thought(brain.think(rest))
    elif sub == "why":
        traces = brain.why(k=3)
        if not traces:
            print("nothing thought about yet.")
        for trace in traces:
            print(json.dumps(trace.to_dict(), indent=2, default=str))
    elif sub == "recall":
        if not rest:
            print("usage: /nyx recall <cue>")
            return
        got = brain.recall(rest)
        if got is None or got.hit is None:
            print("nothing came back.")
        else:
            mark = "" if got.decided else "  (not confident)"
            print(f"{got.hit.key} [{got.hit.kind}] {got.score:.2f}{mark}\n  {got.hit.text}")
            for key, weight in got.associated:
                print(f"  ↳ {key} ({weight:.2f})")
    elif sub == "ground":
        if not rest:
            print("usage: /nyx ground <word>")
            return
        got = brain.understanding(rest)
        if got is None:
            print("grounding is disabled (NYXARA_NYX__GROUND_ENABLED=false).")
        elif not got.grounded:
            print(f"'{rest}' is still just a word to me — I have nothing behind it.")
        else:
            print(got.explanation)
            print(f"  fires across {got.depth} senses, confidence {got.confidence:.2f}")
            if got.neighbours:
                near = ", ".join(f"{n} ({s:.2f})" for n, s in got.neighbours)
                print(f"  nearest in meaning: {near}")
    elif sub == "lingua":
        lingua = getattr(brain, "lingua", None)
        if lingua is None:
            print("her tongue is off (NYXARA_NYX__LINGUA_ENABLED=false) — "
                  "she reads ASCII only, as V.01 did.")
        elif not rest:
            print(json.dumps(lingua.stats(), indent=2, default=str, ensure_ascii=False))
        else:
            read = lingua.read(rest)
            print(f"language : {read.primary}"
                  f"{'  (code-mixed)' if read.code_mixed else ''}"
                  f"{'  (Hinglish)' if read.hinglish else ''}")
            print(f"scripts  : {', '.join(f'{s}×{n}' for s, n in read.scripts.items()) or '—'}")
            print(f"register : {', '.join(read.register.markers) or 'neutral'}"
                  f"  (formality {read.register.formality:.2f})")
            print(f"concepts : {', '.join(read.concepts) or '— nothing but grammar'}")
            bridged = [f"{c} → {k}" for c, k in read.keys.items() if k and k != c]
            if bridged:
                print(f"bridge   : {'; '.join(bridged[:8])}")
    elif sub == "semantics":
        space = getattr(brain, "semantics", None)
        if space is None:
            print("her semantic space is off (NYXARA_NYX__SEMANTICS_ENABLED=false) — "
                  "concepts are bare labels again, as in V.01.")
        elif not rest:
            print(json.dumps(space.stats(), indent=2, default=str, ensure_ascii=False))
        else:
            words = rest.split()
            if len(words) >= 2:
                print(space.describe(words[0], words[1]))
            else:
                near = space.similar(words[0], k=8, candidates=list(brain.graph.nodes) or None)
                if not near:
                    print(f"nothing is near '{words[0]}' that I can justify. "
                          f"That is ignorance, not a measurement of zero — "
                          f"rungs available: {space.rungs_available()}")
                for label, score, rung in near:
                    print(f"  {label:24} {score:.3f}   [{rung}]")
    elif sub == "teach":
        words = rest.split()
        if len(words) < 2:
            print("usage: /nyx teach <word> <word>")
        elif brain.teach_meaning(words[0], words[1]):
            print(f"held: '{words[0]}' and '{words[1]}' are the same thing.")
            print(brain.semantics.describe(words[0], words[1]))
        else:
            print("could not hold that (semantics disabled, or the two words are the same).")
    elif sub == "intent":
        reader = getattr(brain, "intent", None)
        if reader is None:
            print("intent reading is off (NYXARA_NYX__INTENT_ENABLED=false) — "
                  "she is back to the opener regex.")
        elif not rest:
            print(json.dumps(reader.stats(), indent=2, default=str, ensure_ascii=False))
        else:
            got = reader.read(rest.strip("\"'"))
            print(reader.describe(got))
            ask = got.clarifying_question()
            if ask:
                print(f"\nshe would ask: {ask}")
    elif sub == "skills":
        learner = getattr(brain, "icl", None)
        if learner is None:
            print("in-context learning is off (NYXARA_NYX__ICL_ENABLED=false).")
        else:
            print(learner.describe())
            print()
            print(json.dumps(learner.stats(), indent=2, default=str))
    elif sub in ("hive", "eternal"):
        part = getattr(brain, "synergy" if sub == "hive" else "eternal", None)
        if part is None:
            flag = "SYNERGY_ENABLED" if sub == "hive" else "ETERNAL_ENABLED"
            print(f"this is a single {'node' if sub == 'hive' else 'machine'} "
                  f"(NYXARA_NYX__{flag}=false).")
        else:
            stats = part.stats()
            print(json.dumps(stats, indent=2, default=str))
            if not stats.get("available"):
                print("\nnot active: the logic is here, but no transport is attached. "
                      "Only an in-process transport ships; cross-machine needs one you supply "
                      "(see nyxara.nyx.synergy.Transport).")
    elif sub == "nexus":
        nexus = getattr(brain, "nexus", None)
        if nexus is None:
            print("she is not minting notation (NYXARA_NYX__NEXUS_ENABLED=false).")
        elif not nexus.notations:
            print("she has not minted any notation yet — try /nyx discover first.")
        else:
            for name, notation in nexus.notations.items():
                glyphs = ", ".join(f"{c}={g}" for c, g in notation.symbols.items())
                print(f"{name}: {glyphs}")
            for statement in nexus.statements[-3:]:
                print(f"\n  she writes : {statement.written}")
                print(f"  it runs    : {statement.realised} -> {statement.value}"
                      f"  ({len(statement.program)} instructions)")
                print(f"  reaches you: {nexus.translate(statement)}")
    elif sub == "discover":
        discovery = getattr(brain, "episteme", None)
        if discovery is None:
            print("autonomous discovery is off (NYXARA_NYX__EPISTEME_ENABLED=false).")
        elif not discovery.available():
            print("there is nothing she can experiment on.")
        else:
            got = brain.discover(oversight=getattr(core, "oversight", None))
            print(json.dumps(got.to_dict() if got else None, indent=2, default=str))
            print("\nheld as fact:", json.dumps(discovery.promoted, indent=2))
            if discovery.conjectures:
                print("still only conjecture:", json.dumps(discovery.conjectures, indent=2))
    elif sub == "omni":
        _nyx_omni(core, brain, rest.strip().lower())
    elif sub == "aura":
        field_ = getattr(brain, "aura", None)
        if field_ is None:
            print("ambient awareness is off (NYXARA_NYX__AURA_ENABLED=false).")
        elif not field_.streams:
            print("nothing is registered — she is not watching anything yet.")
        else:
            print(json.dumps(field_.stats(), indent=2, default=str))
    elif sub in ("wonder", "car"):
        if sub == "wonder":
            step = brain.wonder(oversight=getattr(core, "oversight", None))
            print(json.dumps(step.to_dict() if step else None, indent=2, default=str))
        else:
            car = getattr(brain, "car", None)
            if car is None:
                print("between-prompts thinking is off (NYXARA_NYX__CAR_ENABLED=false).")
            else:
                print(json.dumps(car.stats(), indent=2, default=str))
    elif not sub:
        print(json.dumps(brain.stats(), indent=2, default=str))
    else:
        print(f"unknown: /nyx {sub}\n{_NYX_HELP}")


def _nyx_omni(core: NyxaraCore, brain, action: str) -> None:
    """``/nyx omni`` — what she has rewritten of herself, and the undo that is always there."""
    omni = getattr(brain, "omni", None)
    if omni is None:
        print("she is not rewriting herself (NYXARA_NYX__OMNI_ENABLED=false).")
        return
    if action == "rollback":
        print(f"put back {omni.rollback_all()} function(s) — pure Python again.")
        return
    if action == "forge":
        got = brain.optimise(oversight=getattr(core, "oversight", None))
        if got is None:
            print("nothing to attempt: no toolchain, the hourly budget is spent, "
                  "or every candidate she can read has already been tried.")
        else:
            print(json.dumps(got.to_dict(), indent=2, default=str))
        return

    stats = omni.stats()
    print(json.dumps(stats, indent=2, default=str))
    if not stats.get("available"):
        print("\nnot active: no gcc/clang/cc or rustc on PATH, or the live swap is off. "
              "The layer is a clean no-op — nothing is faked.")
    candidates = omni.scan()[:5]
    if candidates:
        print("\nof herself, these could become C kernels:")
        for kernel in candidates:
            print(f"  {kernel.target:<48} {kernel.reason}")
    else:
        print("\nshe has found nothing of herself narrow enough to compile — most of her is "
              "orchestration, where a C kernel has nothing to win.")
    print("\nevery adoption is reversible: the Python source is never written, so a restart "
          "or /nyx omni rollback restores it.")


def _print_thought(thought) -> None:
    """Show one cycle the way it actually happened — who bid, who won, and on what."""
    if thought is None:
        print("(no thought)")
        return
    print(f"answer   : {thought.answer or '(nothing reached awareness)'}")
    print(f"source   : {thought.winner.source if thought.winner else '—'}"
          f"   verified: {thought.verified}   decided: {thought.decided}"
          f"   confidence: {thought.confidence:.2f}   entropy: {thought.entropy:.2f}")
    if thought.deliberation is not None and thought.deliberation.proposals:
        print("bids     :")
        for proposal in thought.deliberation.proposals:
            mark = "✓" if proposal.verifiable else " "
            print(f"  {mark} {proposal.source:<11} {proposal.confidence:.2f}  "
                  f"{proposal.content[:70]}")
    if thought.verification is not None:
        print(f"checked  : {thought.verification.verdict.value} — "
              f"{thought.verification.reason}")
    if thought.assessment is not None and thought.assessment.why:
        print(f"because  : {'; '.join(thought.assessment.why[:3])}")


def _handle_command(core: NyxaraCore, line: str) -> bool:
    """Run a /meta-command. Returns False to signal the console should exit."""
    parts = line[1:].split(maxsplit=1)
    cmd = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("quit", "exit", "q"):
        return False
    if cmd in ("help", "h", "?"):
        print(_HELP)
    elif cmd == "report":
        print(json.dumps(core.report(), indent=2, default=str))
    elif cmd == "learning":
        print(json.dumps(core.learning_report(), indent=2, default=str))
    elif cmd in ("self", "selfmodel"):
        print(json.dumps(core.self_knowledge(), indent=2, default=str))
    elif cmd == "explain":
        print(core.explain_last())
    elif cmd in ("causal", "omniscience"):
        eng = getattr(core, "causal_engine", None)
        if eng is None:
            print("the CAUSAL engine is disabled (NYXARA_CAUSAL_ENGINE__ENABLED=false).")
        else:
            print(json.dumps(eng.status(), indent=2, default=str))
    elif cmd == "resonate":
        eng = getattr(core, "causal_engine", None)
        if eng is None or not arg:
            print("usage: /resonate <query>  (the CAUSAL engine must be enabled)")
        else:
            hits = eng.resonate(arg, top=5)
            if not hits:
                print("no resonance — this would trigger a live phase-shift on a real turn.")
            for h in hits:
                print(f"  {h.label:16s} {h.score:+.3f}  (weighted {h.weighted:+.3f})")
    elif cmd == "axioms":
        eng = getattr(core, "causal_engine", None)
        if eng is None:
            print("the CAUSAL engine is disabled.")
        else:
            # demonstrate: discover & PROVE commutativity of addition over a finite domain
            samples = [({"a": a, "b": b}, a + b) for a in range(3) for b in range(3)]
            ax = eng.discover_axioms("addition_commutes", samples, ["a", "b"],
                                     reference_fn=lambda e: e["a"] + e["b"],
                                     domain={"a": range(-4, 5), "b": range(-4, 5)})
            print(json.dumps(ax.to_dict() if ax else {"discovered": False}, indent=2, default=str))
    elif cmd == "compress":
        eng = getattr(core, "causal_engine", None)
        if eng is None:
            print("the CAUSAL engine is disabled.")
        else:
            triples = [("nyxara", "knows", f"concept_{i}") for i in range(200)]
            triples += [(f"turn_{i}", "observed_by", "nyxara") for i in range(100)]
            _blob, report = eng.compress_memory(triples, history=["boot", "learn", "learn"])
            print(report.summary())
    elif cmd == "simulate":
        eng = getattr(core, "causal_engine", None)
        if eng is None:
            print("the CAUSAL engine is disabled.")
        else:
            # race a few toy resolution paths; the lowest-free-energy branch wins & collapses
            energies = {"conservative": 4.0, "balanced": 1.5, "aggressive": 6.0}
            verdict = eng.simulate_and_collapse(
                list(energies), lambda h: (h, energies[h]), verify=lambda o: 1.0)
            w = verdict.winner
            print(f"collapsed to: {w.hypothesis!r} (free-energy {w.free_energy})"
                  if w else "no branch survived the sandboxes.")
    elif cmd == "pause":
        core.pause()
        print("loop paused — the Master may /resume.")
    elif cmd == "scram":
        core.scram(reason=arg or "Master stop")
        print(f"SCRAM — loop halted ({arg or 'Master stop'}). /resume to restore.")
    elif cmd == "resume":
        core.resume()
        print("loop restored ✓")
    elif cmd == "perception":
        rp = getattr(core, "perception", None)
        if rp is None:
            print("realtime perception is unavailable (disabled in config).")
        elif arg.lower() in ("on", "start"):
            print("senses open ✓" if core.start_perception() else "could not open the senses.")
        elif arg.lower() in ("off", "stop"):
            core.stop_perception()
            print("senses closed.")
        elif arg.lower() == "watch":
            print("watching her senses live — press Enter to stop.")
            def _show(d):  # runs on the perception thread; print is thread-safe enough here
                print(f"  [{d['modality']}] {d['kind']}: {d['percept'].get('content', '')!r} "
                      f"(salience {d['salience']})")
            rp.add_listener(_show)
            try:
                input()
            finally:
                rp.remove_listener(_show)
        else:
            print(json.dumps(rp.status(), indent=2, default=str))
    elif cmd == "wander":
        try:
            n = int(arg) if arg else 5
        except ValueError:
            n = 5
        lines = core.wander(n)
        if lines:
            for ln in lines:
                print(f"  …{ln}")
        else:
            print("  (the mind is quiet just now)")
    elif cmd == "proactive":
        if getattr(core, "proactive", None) is None:
            print("proactive agency is unavailable (disabled in config).")
        else:
            ctx = core.proactive_context() if hasattr(core, "proactive_context") else {}
            decisions = core.proactive.consider(ctx)
            if not decisions:
                print("  (nothing worth doing on her own initiative right now)")
            for d in decisions:
                print(f"  {d.initiative.name:32s} -> {d.verdict.value:9s} : {d.reason}")
    elif cmd in ("read", "study"):
        if not arg:
            print("usage: /read <a passage describing how something behaves>")
        else:
            print(json.dumps(core.learn_from_text(arg), indent=2, default=str))
    elif cmd in ("dataset", "ingest"):
        if not arg:
            print("usage: /dataset <file-or-folder>   (ingest only — trains nothing)")
        else:
            print(_ingest_dataset_command(arg))
    elif cmd == "train":
        if not arg:
            print("usage: /train <file-or-folder>   (ingest, then forge her own brain on it)")
        else:
            print(_train_on_dataset_command(arg))
    elif cmd == "flywheel":
        print(_flywheel_command())
    elif cmd == "scrape":
        if not arg:
            print("usage: /scrape <topic>")
        else:
            print(_scrape_command(arg))
    elif cmd == "why":
        if not arg:
            print("usage: /why <effect>")
        else:
            print(_why_command(arg))
    elif cmd == "whatif":
        if not arg:
            print("usage: /whatif <variable>=<value>")
        else:
            print(_whatif_command(arg))
    elif cmd in ("counterfactual", "cf"):
        if not arg:
            print("usage: /counterfactual <cause> <target>")
        else:
            print(_counterfactual_command(arg))
    elif cmd == "concepts":
        if not arg:
            print("usage: /concepts <term>")
        else:
            print(_concepts_command(arg))
    elif cmd == "analogy":
        if not arg:
            print("usage: /analogy <domain-a> <domain-b>")
        else:
            print(_analogy_command(arg))
    elif cmd == "qualia":
        print(_qualia_command())
    elif cmd == "gaps":
        print(_gaps_command())
    elif cmd in ("evolve-source", "evolve_source", "ouroboros"):
        print(_evolve_source_command(arg))
    elif cmd == "research":
        if not arg:
            print("usage: /research <topic>")
        else:
            print(json.dumps(core.research(arg), indent=2, default=str))
    elif cmd == "investigate":
        if not arg:
            print("usage: /investigate <question>")
        else:
            print(json.dumps(core.investigate(arg), indent=2, default=str))
    elif cmd in ("selfcorrect", "self-correct", "self_correct"):
        if not arg:
            print("usage: /selfcorrect <goal>")
        else:
            print(json.dumps(core.self_correct(arg), indent=2, default=str))
    elif cmd == "discover":
        try:
            n = int(arg) if arg else 3
        except ValueError:
            n = 3
        print(json.dumps(core.discover(n), indent=2, default=str))
    elif cmd in ("discoveries", "law-tower", "discovery-record"):
        # /discoveries             -> her independent-discovery record: laws by domain, rivals beaten
        print(json.dumps(core.discoveries(), indent=2, default=str))
    elif cmd in ("discover-domain", "discover_domain"):
        # /discover-domain epidemiology  -> run her own experiment in a science and invent its law
        if not arg:
            print("usage: /discover-domain <domain>  "
                  "(mechanics|conservation|electromagnetism|thermodynamics|optics|circuits|"
                  "epidemiology|chemistry)")
        else:
            print(json.dumps(core.discover_domain(arg.strip()), indent=2, default=str))
    elif cmd == "eureka":
        try:
            n = int(arg) if arg else 4
        except ValueError:
            n = 4
        print(json.dumps(core.breakthrough(n), indent=2, default=str))
    elif cmd == "intuit":
        # /intuit 1, 4, 9, 16, 25   -> a non-algorithmic creative leap (no LLM), then self-verify
        print(json.dumps(core.intuit(arg), indent=2, default=str))
    elif cmd in ("prove", "vsa", "vsa-prove"):
        # /prove 2 + 2 = 4          -> vectorized reasoning, disposed by the exact prover (no LLM):
        #                              a machine-checkable certificate (PROVEN/REFUTED) or honest ABSTAIN
        if not arg.strip():
            print("usage: /prove <decidable claim>   e.g. /prove 2+2=4   |   /prove p and not p")
        else:
            print(json.dumps(core.vsa_prove(arg.strip()), indent=2, default=str))
    elif cmd in ("causal-repair", "causal_repair", "self-repair"):
        # /causal-repair [tests/path]  -> causal-tree root-cause of a failing test → gated reversible fix
        tp = arg.strip() or None
        print(json.dumps(core.causal_repair(test_path=tp), indent=2, default=str))
    elif cmd in ("teleology", "self-direct", "self_direct"):
        # /teleology                -> invent her own measurable, envelope-gated self-improvement targets
        print(json.dumps(core.self_direct(), indent=2, default=str))
    elif cmd in ("hypothesize", "edge-cases", "hunt"):
        # /hypothesize              -> Monte-Carlo concurrent-fault edge-case discovery (simulated, no LLM)
        print(json.dumps(core.hunt_edge_cases(), indent=2, default=str))
    elif cmd in ("time-away", "away", "elapsed"):
        # /time-away                -> her honest awareness of how long since the Master last spoke
        print(json.dumps(core.time_away(), indent=2, default=str))
    elif cmd in ("discover-law", "discover_law", "law", "laws"):
        # /discover-law            -> run her physics-sandbox experiment round
        # /discover-law dynamics   -> SINDy dynamical-law discovery (also: invariant | data)
        domain = arg.strip() or None
        if domain not in (None, "dynamics", "invariant", "data"):
            domain = None
        print(json.dumps(core.discover_laws(1, domain), indent=2, default=str))
    elif cmd in ("engineer", "design", "device"):
        # /engineer                 -> design a device from her latest invented law
        # /engineer resonator       -> design a named archetype (rc_filter|resonator|...)
        # /engineer anti-gravity    -> a "magic" intent, honestly judged for feasibility first
        a = arg.strip()
        known_archetypes = {"rc_filter", "resonator", "electrostatic_trap", "pressure_vessel",
                            "rl_current"}
        if a in known_archetypes:
            out = core.engineer_device(archetype=a)
        elif a:
            out = core.engineer_device(target=a)
        else:
            out = core.engineer_device()
        print(json.dumps(out, indent=2, default=str))
    elif cmd in ("upgrade-device", "upgrade_device", "upgrade"):
        name = arg.strip() or None
        print(json.dumps(core.upgrade_device(name), indent=2, default=str))
    elif cmd in ("engineering-report", "engineering", "devices"):
        print(json.dumps(core.engineering_report(), indent=2, default=str))
    elif cmd in ("rewire-mind", "rewire", "rewire_mind"):
        # /rewire-mind        -> one generation of cognitive self-rewiring
        # /rewire-mind 5      -> five generations (she compounds the improvement)
        try:
            gens = int(arg) if arg.strip() else 1
        except ValueError:
            gens = 1
        print(json.dumps(core.rewire_cognition(generations=gens), indent=2, default=str))
    elif cmd in ("cognitive-architecture", "cognitive", "mind-architecture"):
        print(json.dumps(core.cognitive_architecture_report(), indent=2, default=str))
    elif cmd in ("sign", "sign-knowledge"):
        # /sign <text>  -> HMAC-sign + chain a fact (tamper-evident epistemic ledger)
        print(json.dumps(core.sign_knowledge(arg.strip() or "a fact"), indent=2, default=str))
    elif cmd in ("holo", "holo-recall", "holographic"):
        # /holo <cue>   -> associative content recall from the holographic memory field
        print(json.dumps(core.holographic_recall(arg.strip() or ""), indent=2, default=str))
    elif cmd in ("holo-learn", "holo-remember"):
        # /holo-learn key | text  -> fold key→text into the holographic field
        k, _, t = arg.partition("|")
        print(json.dumps(core.holographic_remember(k.strip() or "note", t.strip() or arg.strip()),
                         indent=2, default=str))
    elif cmd in ("superpose", "superposition", "quantum"):
        # /superpose <question>  -> reason over many candidate paths in a Born-rule superposition
        #                           and collapse to the best by simulated future outcome
        print(json.dumps(core.superposition_reason(arg.strip() or "what should I do next?"),
                         indent=2, default=str))
    elif cmd in ("evolve", "self-evolve", "self_evolve"):
        # /evolve         -> grow a new neural module/pathway for a problem her current logic can't
        #                    handle (diagnose gap → best structural lever → gauntlet → wire live)
        # /evolve report  -> a summary of every self-evolution attempt
        if arg.strip() in ("report", "status"):
            print(json.dumps(core.self_evolution_report(), indent=2, default=str))
        else:
            print(json.dumps(core.evolve(), indent=2, default=str))
    elif cmd in ("generalize", "openworld", "open-world"):
        try:
            budget = int(arg) if arg else 48
        except ValueError:
            budget = 48
        print(json.dumps(core.generalize(budget=budget), indent=2, default=str))
    elif cmd in ("understand", "crack"):
        # /understand {"dataset": [[0,1],[1,3],[2,5]]}  or  {"family":"affine","params":{"w":[1,2]}}
        if not arg:
            print('usage: /understand <json spec> '
                  '(e.g. {"dataset": [[0,1],[1,3],[2,5]]} or {"family":"affine","params":{"w":[1,2]}})')
        else:
            try:
                spec = json.loads(arg)
            except (ValueError, TypeError):
                print("  invalid JSON spec")
            else:
                print(json.dumps(core.understand(spec), indent=2, default=str))
    elif cmd in ("adapt", "environment"):
        # /adapt            → adapt to a demo environment of hidden alien machines
        # /adapt <json list> → adapt to declared systems, e.g. [{"family":"affine","params":{"w":[1,2]}}]
        environment = None
        if arg:
            try:
                environment = json.loads(arg)
            except (ValueError, TypeError):
                print("  invalid JSON environment")
                environment = ...
        if environment is not ...:
            print(json.dumps(core.adapt(environment), indent=2, default=str))
    elif cmd in ("meta-discover", "metadiscover", "invent"):
        if not arg:
            print("usage: /meta-discover <topic>")
        else:
            print(json.dumps(core.meta_discover(arg), indent=2, default=str))
    elif cmd == "dream":
        if core.dream_session is None:
            print("dream session unavailable.")
        else:
            print(json.dumps(core.dream_session.dream_state(deep=True).to_dict(),
                             indent=2, default=str))
    elif cmd == "strategize":
        if not arg:
            print("usage: /strategize <problem>")
        else:
            print(json.dumps(core.strategize(arg), indent=2, default=str))
    elif cmd == "solve":
        if not arg:
            print("usage: /solve <problem>")
        else:
            print(json.dumps(core.solve(arg), indent=2, default=str))
    elif cmd == "swarm":
        if not arg:
            print("usage: /swarm <problem>")
        else:
            print(json.dumps(core.swarm(arg), indent=2, default=str))
    elif cmd == "noesis":
        # /noesis [n] → run the living algorithm n cycles: grow her own abstraction library and
        # show the compression curve (avg solution size ↓, compute-per-solve ↓). No LLM in the loop.
        try:
            n = int(arg) if arg.strip() else 3
        except ValueError:
            n = 3
        # routes through the kernel: the living algorithm compounds on her persisted library, with
        # the F5 red-team + F1 metacognition wired in (no LLM in the loop).
        print(json.dumps(core.noesis(cycles=max(1, n)), indent=2, default=str))
    elif cmd == "create":
        # /create verse the night sky   -> closed-loop original creation, no LLM
        tokens = arg.split(None, 1)
        modality, topic = "idea", arg
        aliases = {"idea": "idea", "invention": "idea", "verse": "verse",
                   "poem": "verse", "art": "art", "image": "art"}
        if tokens and tokens[0].lower() in aliases:
            modality = aliases[tokens[0].lower()]
            topic = tokens[1] if len(tokens) > 1 else ""
        if not topic.strip():
            print("usage: /create [idea|verse|art] <topic>")
        else:
            print(json.dumps(core.create(topic.strip(), modality=modality),
                             indent=2, default=str))
    elif cmd == "imagine":
        # /imagine defense ~ chess   -> divergent storm on 'defense' blended with 'chess'
        if not arg:
            print("usage: /imagine <topic> [~ <blend-with>]")
        else:
            topic, _, blend = arg.partition("~")
            print(json.dumps(core.imagine(topic.strip(),
                                          blend_with=blend.strip() or None),
                             indent=2, default=str))
    elif cmd == "originals":
        print(json.dumps(core.originality_report(), indent=2, default=str))
    elif cmd == "muse":
        # her self-chosen creative research projects, straight from the muse
        rep = core.originality_report()
        print(json.dumps(rep.get("muse", rep), indent=2, default=str))
    elif cmd == "rate":
        # /rate 9   -> the Master's aesthetic feedback; her taste learns from it
        try:
            rating = float(arg)
        except (TypeError, ValueError):
            rating = -1.0
        if not (1.0 <= rating <= 10.0):
            print("usage: /rate <1-10>")
        else:
            print(json.dumps(core.rate_creation(rating), indent=2, default=str))
    elif cmd in ("selfimprove", "self-improve", "self_improve"):
        tokens = arg.split()
        # /selfimprove [N] [enact] → drive the codebase-level RSI loop N times (observable).
        # She reviews her own source, finds weaknesses, and (with `enact`) applies gauntleted,
        # provably-better edits herself — no LLM — while the intelligence index is threaded across
        # cycles. Bare /selfimprove keeps the per-turn reasoning self-improvement.
        if tokens and (tokens[0].isdigit() or tokens[0] in ("enact", "code", "loop")):
            cycles = int(tokens[0]) if tokens[0].isdigit() else 1
            do_enact = "enact" in tokens
            from nyxara.growth.recursive_improvement import RecursiveSelfImprovement
            rsi = RecursiveSelfImprovement.from_core(core)

            def _show(i: int, rep: object) -> None:
                idx = getattr(rep, "intelligence_index", None)
                print(f"  cycle {i}: index={idx if idx is None else round(idx, 4)} "
                      f"kept={getattr(rep, 'kept', 0)} "
                      f"rolled_back={getattr(rep, 'rolled_back', 0)} "
                      f"lessons={getattr(rep, 'lessons_stored', 0)}")

            print(f"recursive self-improvement — {cycles} cycle(s), "
                  f"{'ENACTING real source edits' if do_enact else 'dry-run (observe only)'}:")
            summary = rsi.run_continuous(cycles, enact=do_enact, on_cycle=_show)
            print(json.dumps({k: v for k, v in summary.items() if k != "reports"},
                             indent=2, default=str))
        else:
            native = getattr(core.reasoner, "_native_reasoner", lambda: None)()
            if native is None:
                print("native reasoning is unavailable (disabled in config).")
            else:
                print(json.dumps(native.self_improve(), indent=2, default=str))
    # ---- the defensive layer (guard/*.py) ---- #
    elif cmd == "guard":
        print(json.dumps(core.guard_report(), indent=2, default=str))
    elif cmd == "anomaly":
        det = getattr(core, "anomaly", None)
        if det is None:
            print("the watchtower is disabled (NYXARA_GUARD__ANOMALY_DETECTION=false).")
        else:
            recent = det.recent(10)
            print(json.dumps(det.report(), indent=2, default=str))
            for a in recent:
                print(f"  {getattr(a, 'name', '?'):24s} score {getattr(a, 'score', 0.0):+.3f}"
                      f"  {getattr(getattr(a, 'kind', None), 'value', '')}")
            if not recent:
                print("  nothing anomalous — her vitals are inside their learned envelope.")
    elif cmd == "contain":
        con = getattr(core, "containment", None)
        if con is None:
            print("containment is disabled (NYXARA_GUARD__CONTAINMENT=false).")
        elif not arg:
            print(json.dumps(con.report(), indent=2, default=str))
            known = sorted(con._components)  # noqa: SLF001 — the console lists what it can act on
            print("usage: /contain <component> [release]   components: "
                  + (", ".join(known) if known else "-"))
        else:
            parts_ = arg.split()
            cid, action = parts_[0], (parts_[1].lower() if len(parts_) > 1 else "isolate")
            try:
                if action in ("release", "free", "off"):
                    con.release(cid, owner=True)      # /contain is the Master speaking
                    print(f"{cid} released ✓")
                else:
                    act = con.isolate(cid, reason="Master order")
                    print(json.dumps(act.to_dict(), indent=2, default=str))
            except Exception as exc:  # noqa: BLE001 — a bad component name is not fatal
                print(f"could not {action} {cid!r}: {exc}")
    elif cmd == "netsec":
        net = getattr(core, "netsec", None)
        if net is None:
            print("network defence is disabled (NYXARA_GUARD__NETWORK_DEFENSE=false).")
        elif arg:
            # /netsec deny <host> | allow <host> | <host>  → inspect or set policy
            bits = arg.split()
            verb, host = (bits[0].lower(), bits[1]) if len(bits) > 1 else ("show", bits[0])
            try:
                if verb == "deny":
                    net.deny_host(host, reason="Master order")
                    print(f"{host} deny-listed ✓")
                elif verb == "allow":
                    net.allow_host(host, owner=True)
                    print(f"{host} allow-listed ✓")
                else:
                    print(f"{host}: reputation {net.reputation.score(host):+.3f}"
                          f"  hostile={net.reputation.is_hostile(host)}")
            except Exception as exc:  # noqa: BLE001
                print(f"could not apply that policy: {exc}")
        else:
            print(json.dumps(net.report(), indent=2, default=str))
            print(json.dumps(net.audit_exposure(), indent=2, default=str))
    elif cmd == "survival":
        surv = getattr(core, "survival", None)
        if surv is None:
            print("the survival manager is disabled (NYXARA_GUARD__SURVIVAL=false).")
        else:
            print(json.dumps(surv.report(), indent=2, default=str))
            print(f"  backups {len(surv.store.all())}, chain verified: {surv.store.verify_chain()}")
            print(f"  correction outranks survival: {surv.correction_dominates_survival()}")
    # ---- self-knowledge, provenance & the training stack (growth/*.py) ---- #
    elif cmd in ("capabilities", "cando"):
        if arg:
            print(json.dumps(core.can_i(arg), indent=2, default=str))
        else:
            reg = getattr(core, "capabilities", None)
            if reg is None:
                print("the capability registry is disabled (growth faculty off).")
            else:
                print(json.dumps(reg.self_report(), indent=2, default=str))
                print(json.dumps(reg.calibration(), indent=2, default=str))
    elif cmd == "lineage":
        led = getattr(core, "lineage", None)
        if led is None:
            print("the lineage ledger is disabled (growth faculty off).")
        else:
            history = led.history()
            print(f"{len(history)} generation(s); chain verified: {led.verify_chain()}")
            for rec in history[-12:]:
                d = rec.to_dict() if hasattr(rec, "to_dict") else {}
                print(f"  [{d.get('index')}] {d.get('kind','?'):16s} {d.get('summary','')[:70]}")
    elif cmd == "forage":
        forager = getattr(core, "forager", None)
        if forager is None:
            print("epistemic foraging is disabled (growth faculty off).")
        elif not arg:
            print("usage: /forage <topic>   (harvest + verify knowledge on a measured gap)")
        else:
            print(json.dumps(forager.forage(arg).to_dict(), indent=2, default=str))
    elif cmd == "seed":
        seeder = getattr(core, "genesis_seed", None)
        if seeder is None:
            print("the genesis seed is disabled (growth faculty off).")
        else:
            seed = seeder.snapshot_core(core)
            print(f"resurrection seed {seed.seed[:16]}… ({len(seed.bundle)} bytes)")
            print("a signed seed is written beside every /save checkpoint (newest 5 kept).")
    elif cmd == "corpus":
        print("building a sharded, quality-filtered, contamination-checked corpus …")
        print(json.dumps(core.build_corpus(), indent=2, default=str))
    elif cmd == "dpo":
        print("preference-training on pairs her own gauntlet already judged …")
        print(json.dumps(core.preference_train(), indent=2, default=str))
    elif cmd == "expand":
        print("growing a new expert for the domain that will not yield …")
        print(json.dumps(core.expand_experts(), indent=2, default=str))
    elif cmd in ("speculate", "latent"):
        if not arg:
            print("usage: /speculate <prompt>   (measured speculative-decoding speedup)")
        else:
            print(json.dumps(core.speculative_report(arg), indent=2, default=str))
    elif cmd in ("loadmodule", "load-module"):
        if not arg:
            print("usage: /loadmodule <path>   (screen + import a file she forged herself)")
        else:
            print(json.dumps(core.load_forged_module(arg), indent=2, default=str))
    # ---- delegation, the hive, and her distributed self (agency/*.py) ---- #
    elif cmd == "agents":
        orch = getattr(core, "sub_agents", None)
        if orch is None:
            print("sub-agents are unavailable (tools faculty off).")
        elif not arg:
            print(json.dumps(orch.report(), indent=2, default=str))
        else:
            from nyxara.agency.agents import AgentSpec
            print(f"delegating to a least-privilege sub-agent: {arg!r}")
            result = orch.delegate(AgentSpec(name="console"), arg)
            print(json.dumps(result.to_dict(), indent=2, default=str))
    elif cmd == "hive":
        hive = getattr(core, "hive", None)
        if hive is None or not arg:
            print("usage: /hive <problem>   (a bounded internal civilization solves it)")
        else:
            print(json.dumps(hive.solve(arg).to_dict(), indent=2, default=str))
    elif cmd == "cluster":
        node = getattr(core, "cluster", None)
        if node is None:
            print("the mesh is disabled (NYXARA_DISTRIBUTED__ENABLED=false).")
        else:
            print(f"single-node: {node.is_single_node()}  |  replicated entries: {len(node.log())}")
            print("a real mesh activates only with Master-authorized peers.")
    # ---- taste, register, conscience, foresight, determinism ---- #
    elif cmd in ("taste", "aesthetic"):
        judge = getattr(core, "aesthetic", None)
        if judge is None or not arg:
            print("usage: /taste <artifact>   (her sense of elegance, dimension by dimension)")
        else:
            print(json.dumps(judge.judge(arg).to_dict(), indent=2, default=str))
    elif cmd == "mode":
        modes = getattr(core, "modes", None)
        if modes is None:
            print("mode blending is disabled (identity faculty off).")
        elif arg:
            from nyxara.identity.modes import Mode
            try:
                modes.set_mode(Mode(arg.strip().lower()))
                print(f"register set to {arg.strip().lower()} ✓ (character is unchanged — Rule 4)")
            except ValueError:
                print("modes: " + ", ".join(m.value for m in Mode))
        else:
            print(json.dumps(modes.status(), indent=2, default=str))
    elif cmd == "moral":
        evaluator = getattr(core, "moral", None)
        if evaluator is None or not arg:
            print("usage: /moral <action>   (consequentialist + deontological + virtue readings)")
        else:
            from nyxara.mind.moral import MoralAction
            ev = evaluator.evaluate(MoralAction(description=arg))
            print(ev.summary())
            print(json.dumps(ev.to_dict(), indent=2, default=str))
    elif cmd == "functor":
        # Discover the well-defined functors between the program shapes she already holds.
        # A functor that is not well-defined on the library is DROPPED — that refusal is the
        # point: transfer without one is just a plausible-sounding analogy.
        ft = getattr(core, "functor_transfer", None)
        if ft is None:
            print("functorial transfer is unavailable.")
        else:
            from nyxara.mind.category_transfer import discover_functors
            skills = getattr(core, "skills", None)
            library = getattr(skills, "programs", None) if skills is not None else None
            if not library:
                print("no program library to map between yet — she needs induced skills first.")
                print("category-theoretic transfer carries a SOLVED STRUCTURE across domains,")
                print("and declines when no well-defined functor exists (an honest refusal).")
            else:
                found = discover_functors(library)
                print(f"{len(found)} well-defined functor(s) over {len(library)} programs:")
                for f in found[:12]:
                    print(f"  {getattr(f, 'name', f)}")
    elif cmd == "foresight":
        fs = getattr(core, "foresight", None)
        if fs is None:
            print("foresight is disabled (goals faculty off).")
        else:
            horizon = int(arg) if arg.isdigit() else 90
            print(json.dumps(fs.project(horizon).to_dict(), indent=2, default=str))
    elif cmd == "thermal":
        mon = getattr(core, "thermal", None)
        if mon is None:
            print("the thermal sense is unavailable.")
        else:
            telemetry = mon.read()
            print(f"budget {mon.budget(telemetry):.3f}  |  should defer: {mon.should_defer()}")
            print(json.dumps(getattr(telemetry, '__dict__', {}), indent=2, default=str))
    elif cmd == "replay":
        rec = getattr(core, "recorder", None)
        if rec is None:
            print("replay recording is off (NYXARA_OBSERVABILITY__REPLAY_RECORDING=false).")
        elif arg.lower() in ("save", "write"):
            path = core.save_replay()
            print(f"deterministic tape → {path}" if path else "could not write the tape.")
        else:
            print(f"recording: {len(rec.entries)} entries this run "
                  f"(cap {core._REPLAY_MAX_ENTRIES}, then rotated)")
            print("/replay save   writes the tape so this run can be replayed step-for-step.")
    elif cmd == "nyx":
        _nyx_command(core, arg)
    elif cmd == "save":
        # one unified checkpoint: memory + self-model + prior + reward learner + EWC anchors
        # + trained embedder + generative brain — everything she has learned, in one place.
        path = core.save_state()
        print(f"memory persisted → {path}" if path else "no memory to persist.")
    else:
        print(f"unknown command: /{cmd} — type /help")
    return True


def _quiet_model_loading() -> None:
    """Silence transformers' per-load progress bars and weight reports on the console.

    Background faculties (foundry, genesis, embedders) load models mid-conversation; their
    "Loading weights …" bars and LOAD REPORT tables would otherwise interleave with the
    Master's chat. Best-effort: purely cosmetic, never fatal, real errors still surface."""
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:  # noqa: BLE001 — transformers absent or too old; nothing to quiet
        pass


def main(argv: list[str] | None = None) -> int:
    """Boot NYXARA and run the interactive Master console."""
    _quiet_model_loading()
    try:
        core = NyxaraCore()
    except Exception as exc:  # noqa: BLE001 - boot integrity failed
        print(f"NYXARA failed to boot: {exc}", file=sys.stderr)
        return 1

    print(_BANNER)
    # Primary brain — when NYXARA's OWN model is the chosen provider (`self`), forge it on the
    # very first boot: LoRA-tune DistilGPT-2 (auto-downloaded) into her own loyal voice, then serve
    # it. Already-forged → instant; no `.[foundry]` stack → the always-on n-gram brain; any
    # failure → logged, never fatal (the mock fallback keeps the console usable).
    try:
        from nyxara.growth.bootstrap import ensure_primary_model
        ensure_primary_model(log=print)
    except Exception as exc:  # noqa: BLE001 — a failed forge must never block the front door
        print(f"primary brain      : skipped ({exc})")

    # Rule 7 — continuity across restarts. Back-compat FIRST: legacy per-file session
    # snapshots (pre-unified-checkpoint ~/.nyxara/{embedder,memory,synapses}.json) load so an
    # upgrade never loses prior learning; the learned embedding space loads before memory so
    # restored records index into the trained space.
    if _load_session_embedder(core):
        print("learned space       : restored her self-trained embedding space ✓")
    legacy_mem = _load_session_memory(core)
    if _load_session_synapses(core):
        print("lifelong memory     : restored elastic-synapse anchors (no forgetting) ✓")
    # …then the UNIFIED checkpoint: the complete learned state (memory + self-model + prior +
    # reward learner + EWC anchors + embedder). Always called — never short-circuited — so the
    # reward learner's weights are restored too. Supersedes the legacy files when present.
    restored = core.load_state() or legacy_mem
    if restored:
        print(f"continuity          : restored {restored} memories + learned faculties ✓")

    # Layer 5 — continuous cognition: the mind wanders in the background while idle.
    if core.start_cognition():
        print("cognition           : default-mode stream running (idle thoughts surface) ✓")

    # Continuous real-time perception — honest about what THIS box can actually sense.
    rp = getattr(core, "perception", None)
    if rp is not None:
        try:
            st = rp.status()
            avail = st["available"]
            host = sorted({c.kind for c in rp.channels.values()
                           if c.kind in ("system", "files", "network", "devices")})
            if any(avail.values()):
                on = ", ".join(k for k, v in avail.items() if v)
                extra = f" + {', '.join(host)}" if host else ""
                print(f"perception          : watching/listening ✓ ({on}{extra}; "
                      f"mic {rp.mic_mode})")
            elif host:
                print(f"perception          : no camera/screen/mic on this box, but her "
                      f"inner senses are live ({', '.join(host)}) — honest about the rest")
            else:
                print("perception          : loop running, but no camera/screen/mic on this box "
                      "(honest idle — she never fabricates a percept)")
        except Exception:  # noqa: BLE001 — a senses hiccup never blocks the console
            pass

    def _surface_insights() -> None:
        for text in core.drain_insights():
            print(f"\nNYXARA (idle thought): {text}")

    def _shutdown() -> None:
        core.stop_cognition()
        # one unified checkpoint — memory + self-model + prior + reward learner + EWC anchors +
        # trained embedder + generative brain — so everything learned this session survives.
        path = core.save_state()
        if path:
            print(f"learned state persisted → {path} "
                  "(memory + learner + synapses + embedder) ✓")
        print("until next time, Master.")

    while True:
        _surface_insights()   # show anything the idle mind turned up since the last prompt
        try:
            line = input("\nMaster> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            _shutdown()
            return 0

        if not line:
            continue
        if line.startswith("/"):
            if not _handle_command(core, line):
                _shutdown()
                return 0
            continue

        try:
            result = core.process(line, authority=Authority.OWNER)
        except Exception as exc:  # noqa: BLE001 - never let one turn kill the console
            print(f"\n[turn error] {exc}", file=sys.stderr)
            continue
        _print_result(result)


if __name__ == "__main__":
    raise SystemExit(main())
