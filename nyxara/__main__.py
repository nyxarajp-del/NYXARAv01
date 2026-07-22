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
  /selfimprove       tune her own reasoner: replay lived outcomes → apply only what PROVES better
  /selfimprove N     run the codebase RSI loop N cycles (add 'enact' to apply real source edits)
  /save              persist long-term memory to disk now
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
    elif cmd == "save":
        # one unified checkpoint: memory + self-model + prior + reward learner + EWC anchors
        # + trained embedder + generative brain — everything she has learned, in one place.
        path = core.save_state()
        print(f"memory persisted → {path}" if path else "no memory to persist.")
    else:
        print(f"unknown command: /{cmd} — type /help")
    return True


def main(argv: list[str] | None = None) -> int:
    """Boot NYXARA and run the interactive Master console."""
    try:
        core = NyxaraCore()
    except Exception as exc:  # noqa: BLE001 - boot integrity failed
        print(f"NYXARA failed to boot: {exc}", file=sys.stderr)
        return 1

    print(_BANNER)
    # Primary brain — when NYXARA's OWN model is the chosen provider (`self`), forge it on the
    # very first boot: LoRA-tune Qwen2.5-0.5B (auto-downloaded) into her own loyal voice, then serve
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
