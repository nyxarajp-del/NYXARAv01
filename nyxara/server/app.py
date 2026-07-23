"""NYXARA · server/app.py — the FastAPI application over the sovereign loop.

Routes (all of ``/v1`` require the bearer token when one is configured):

* ``GET  /health``               — liveness, unauthenticated.
* ``GET  /v1/report``            — a calibrated status report.
* ``GET  /v1/learning``          — truthful learning state (generations, corpus, serving).
* ``POST /v1/chat``              — one turn: ``{message}`` → the disposed response.
* ``POST /v1/agent``             — a multi-step gated goal: ``{goal, max_steps?}``.
* ``POST /v1/self_correct``      — a goal pursued with self-correction & epistemic uncertainty:
                                   detect wrong/stuck, experiment to fill the gap, abstain honestly.
* ``POST /v1/self_improve``      — the codebase RSI loop, observable: ``{cycles?, enact?}`` → the
                                   intelligence-index trajectory + kept/rolled-back edit tallies.
* ``POST /v1/research``          — one autonomous research pass: ``{topic}``.
* ``POST /v1/investigate``       — the scientist loop: ``{question}`` → hypothesis/conclusion.
* ``POST /v1/discover``          — the autonomous discovery loop: ``{cycles?}`` → belief updates.
* ``POST /v1/breakthrough``      — truly novel problem solving: ``{generations?, population?}``
  → invent → prove → keep novel + interesting.
* ``POST /v1/discover-law``      — Frontier Law Discovery: ``{rounds?, domain?}`` → invent a NEW
  empirical/physical law from data & self-run experiments (no LLM); corroborated or honest abstain.
* ``POST /v1/generalize``        — open-world generalization: ``{budget?}`` → crack a hidden,
  never-before-seen alien machine from first principles (observe→hypothesize→test→model).
* ``POST /v1/meta_discover``     — meta-research: ``{topic}`` → invent → test → (gated) integrate.
* ``GET  /v1/discoveries``       — her independent-discovery record: the law tower she built from
  her own curiosity — laws by domain, rivals beaten, skills minted, director tallies.
* ``POST /v1/discover_domain``   — run her own experiment in a named science: ``{domain}`` → she
  poses the question, runs the experiment, and invents the governing law (zero-to-discovery, no LLM).
* ``POST /v1/dream``             — a deep Dream State: ``{deep?}`` → distil / prune / fix synapses.
* ``POST /v1/strategize``        — strategic analysis: ``{problem}`` → six-part framework.
* ``POST /v1/solve``             — domain-aware general intelligence: ``{problem}`` → solved
                                   as the right kind of expert (coding/maths/science/…).
* ``POST /v1/control/{action}``  — sovereign control: pause / resume / scram (opt-in).
* ``GET  /v1/perception``        — the always-on senses: live status (devices, channels, events).
* ``POST /v1/perception/{action}`` — open/close her eyes and ears: start / stop.
* ``WS   /v1/perception/ws``     — live event feed: every percept streams as it lands.
* ``POST /v1/memory/save|load``  — persist / restore long-term memory (Rule 7 continuity).
* ``WS   /v1/ws``                — a streaming chat socket (token via ``?token=``).

The control law is untouched: each request runs ``NyxaraCore.process`` end to end, so an
over-eager request is refused or escalated exactly as it would be at the console. Auth is a
single bearer token — the Master's credential, this being a single-Master system.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from nyxara.agency.permissions import Authority
from nyxara.kernel.config import NyxaraSettings, Profile, get_settings

__all__ = ["create_app", "run"]


# --------------------------------------------------------------------------- #
# Request bodies
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    # The caller authenticates as the Master; non-owner authorities only ever de-escalate.
    authority: str = "owner"


class AgentRequest(BaseModel):
    goal: str = Field(..., min_length=1)
    # No upper bound here — the server clamps to ``server.max_agent_steps``.
    max_steps: Optional[int] = Field(default=None, ge=1)
    authority: str = "owner"


class SelfCorrectRequest(BaseModel):
    goal: str = Field(..., min_length=1)
    # No upper bound here — the server clamps to ``server.max_agent_steps``.
    max_steps: Optional[int] = Field(default=None, ge=1)
    authority: str = "owner"


class SelfImproveRequest(BaseModel):
    # How many codebase-level RSI cycles to run (bounded — never a literal infinite loop).
    cycles: int = Field(default=1, ge=1, le=50)
    # Apply real, gauntleted source edits (True) or observe the loop read-only (default).
    enact: bool = False


class DelegateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    subgoal: str = Field(..., min_length=1)
    max_steps: Optional[int] = Field(default=None, ge=1)
    # delegates run AUTONOMOUS by default so their risky moves escalate, never auto-act
    authority: str = "autonomous"


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=1)


class InvestigateRequest(BaseModel):
    question: str = Field(..., min_length=1)


class DiscoverRequest(BaseModel):
    # How many self-driven discovery cycles to run; the server keeps this bounded.
    cycles: int = Field(default=3, ge=1, le=50)


class GeneralizeRequest(BaseModel):
    # Probe budget for cracking a hidden alien machine from first principles; kept bounded.
    budget: int = Field(default=48, ge=8, le=256)


class UnderstandRequest(BaseModel):
    # Hand NYXARA a black box to crack — declaratively, since a callable can't cross the wire.
    # Provide EITHER a dataset of observed rows OR a named law family + params to rebuild a box.
    dataset: Optional[List[List[float]]] = None       # [[x, y], ...] observed input→output rows
    family: Optional[str] = None                      # a law family name, e.g. "affine"
    params: Dict[str, Any] = Field(default_factory=dict)
    dims: int = Field(default=1, ge=1, le=6)
    kind: str = "real"                                # "real" | "int" | "bool"
    low: float = -6.0
    high: float = 6.0
    label: Optional[str] = None
    budget: int = Field(default=48, ge=8, le=256)


class AdaptRequest(BaseModel):
    # A brand-new environment as a list of declarative systems ({family, params, ...}); she models
    # each with her own faculties and, under real pressure, re-organizes her own brain. Bounded.
    systems: List[Dict[str, Any]] = Field(default_factory=list)
    budget: int = Field(default=48, ge=8, le=256)
    label: str = "environment"


class BreakthroughRequest(BaseModel):
    # Open-ended novel-discovery generations and per-generation population; both kept bounded.
    generations: int = Field(default=4, ge=1, le=50)
    population: int = Field(default=24, ge=5, le=200)


class DiscoverLawRequest(BaseModel):
    # How many law-discovery rounds to run (bounded), and an optional domain to steer the search:
    # "dynamics" (SINDy), "invariant" (Noether), "data" (self-generated), or omitted (physics sandbox).
    rounds: int = Field(default=1, ge=1, le=20)
    domain: Optional[str] = Field(default=None)


class MetaDiscoverRequest(BaseModel):
    topic: str = Field(..., min_length=1)


class DiscoverDomainRequest(BaseModel):
    # A science domain for her to run her own experiment in and discover its governing law:
    # "mechanics", "conservation", "electromagnetism", "thermodynamics", "optics", "circuits",
    # "epidemiology", or "chemistry". She poses the question, runs the experiment, invents the law.
    domain: str = Field(..., min_length=1)


class DreamRequest(BaseModel):
    # Run a deep "Dream State" (distil logs, delete useless ones, fix Deep Memory Synapses).
    deep: bool = True


class StrategizeRequest(BaseModel):
    problem: str = Field(..., min_length=1)


class SolveRequest(BaseModel):
    problem: str = Field(..., min_length=1)


class ControlRequest(BaseModel):
    reason: str = ""


class MemoryPathRequest(BaseModel):
    path: Optional[str] = None


class TemporalTickRequest(BaseModel):
    # how many millisecond-scale beats to drive the fractal hierarchy through (nested
    # meso roll-ups and macro observations fire at their boundaries).
    beats: int = Field(default=1, ge=1)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _authority(value: str) -> Authority:
    """Map a request's authority string to the enum, defaulting to OWNER (the Master)."""
    try:
        return Authority(str(value).strip().lower())
    except ValueError:
        return Authority.OWNER


def _missing_dep_error() -> "ImportError":
    return ImportError(
        "The NYXARA API server needs FastAPI + uvicorn. Install the optional extra:\n"
        "    pip install -e \".[server]\"")


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app(core: Any = None, *, settings: Optional[NyxaraSettings] = None) -> Any:
    """Construct the FastAPI app bound to a (lazily-built) :class:`NyxaraCore`."""
    try:
        from fastapi import (Depends, FastAPI, Header, HTTPException, WebSocket,
                             WebSocketDisconnect, status)
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise _missing_dep_error() from exc

    settings = settings or get_settings()
    cfg = settings.server

    # Resolve the configured token once (never logged).
    token: Optional[str] = (cfg.api_token.get_secret_value() if cfg.api_token else None) or None

    # Fail-closed: a production server must be authenticated.
    if settings.profile is Profile.PROD and not token:
        raise RuntimeError(
            "refusing to start an unauthenticated server in PROD — set "
            "NYXARA_SERVER__API_TOKEN to a strong secret")

    if core is None:
        from nyxara.kernel.orchestrator import NyxaraCore
        core = NyxaraCore()

    # ---- always-on background mind ---- #
    # When ``server.autonomic`` is set (the ``nyxara-daemon`` / service path), start the
    # AutonomicLoop over this same core once uvicorn's event loop is running, and stop it
    # cleanly on shutdown. Wrapped so a background-mind hiccup never takes the API down.
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _lifespan(app_: Any):
        loop = None
        runtime = None
        # Lifelong continuity: the always-on daemon MUST resume prior learning on startup —
        # the whole point of "gets better every minute" is that minute N+1 builds on minute N,
        # across restarts, not from a blank slate. Load the complete learned state (memory +
        # self-model + reward learner + EWC anchors + embedder) before the mind starts.
        try:
            restored = app_.state.core.load_state()
            print(f"NYXARA continuity: restored learned state on startup "
                  f"({restored} memory records + learner/synapses/embedder)")
        except Exception as lexc:  # noqa: BLE001 — a cold start must never be blocked by restore
            print(f"NYXARA continuity: startup restore skipped ({lexc})")
        # Continuous real-time perception: the core auto-starts it at construction; here we
        # just report honestly what this box can actually perceive (never fabricated).
        try:
            rp = getattr(app_.state.core, "perception", None)
            if rp is not None:
                st = rp.status()
                print(f"NYXARA live perception: running={st['running']} "
                      f"available={st['available']} mic={st['mic_mode']}")
            else:
                print("NYXARA live perception: disabled by config")
        except Exception:  # noqa: BLE001 — the senses report must never block startup
            pass
        if cfg.autonomic:
            try:
                from nyxara.kernel.autonomic import AutonomicLoop
                from nyxara.kernel.presence import Presence
                from nyxara.kernel.runtime import Runtime
                from nyxara.agency.health import HealthMonitor
                core = app_.state.core
                # When the Master has granted autonomous internet (or full control), run the
                # background mind at "max level": inner_life draws its agenda from her own
                # proactive engine (incl. the internet-research initiative) + default-mode
                # stream, so the always-on daemon continuously researches and acts on the live
                # web on her own cadence — still through every sovereign gate. Off by config
                # (autonomous_internet=false, full_control=false) it keeps the fixed reflective
                # repertoire, unchanged.
                inner_life = bool(settings.agency.full_control
                                  or settings.agency.autonomous_internet)

                # Presence — arousal state machine gives the background mind a *cadence* (calm
                # when idle/asleep, active when alert) and gates self-initiated proposals, so
                # she paces herself without ever needing the Master online.
                presence = None
                try:
                    presence = Presence(settings=settings)
                except Exception:  # noqa: BLE001 — presence is a capability, never required
                    presence = None

                # HealthMonitor — heartbeats + resource pressure with bounded self-healing, so
                # the background mind can recover from degradation on its own (fail-closed on
                # security/invariant classes, which always escalate to the Master).
                health = None
                try:
                    health = HealthMonitor()
                    health.register("autonomic", heartbeat_timeout=cfg.autonomic_interval_s * 4)
                    if getattr(core, "governor", None) is not None:
                        health.bind_governor(core.governor)
                except Exception:  # noqa: BLE001 — health is a capability, never required
                    health = None

                loop = AutonomicLoop(
                    core,
                    interval_s=cfg.autonomic_interval_s,
                    growth_every=cfg.autonomic_growth_every,
                    self_optimize_every=cfg.autonomic_self_optimize_every,
                    self_optimize_debug=cfg.autonomic_self_optimize_debug,
                    inner_life=inner_life,
                    decision_mode=cfg.autonomic_decision_mode,
                    presence=presence,
                    health=health,
                )

                # Supervise the loop in-process: if the background mind ever crashes it is
                # auto-restarted with backoff (complementing systemd's Restart=always, which
                # only restarts the whole process). One long-running, self-healing mind.
                runtime = Runtime(name="nyxara-autonomic")

                async def _mind_factory(_loop=loop):
                    _loop._running = True
                    await _loop._run(None)

                runtime.supervise(_mind_factory, name="autonomic-mind",
                                  max_restarts=1_000_000, backoff=1.0, max_backoff=30.0)
                app_.state.autonomic = loop
                app_.state.autonomic_runtime = runtime
                print(f"NYXARA background mind (AutonomicLoop) started "
                      f"[interval {cfg.autonomic_interval_s}s, growth_every "
                      f"{cfg.autonomic_growth_every}, self_optimize_every "
                      f"{cfg.autonomic_self_optimize_every}, inner_life {inner_life}, "
                      f"decision_mode {cfg.autonomic_decision_mode}, supervised]")

                # Deep self-directed cognition: the AutonomicLoop above decides + acts, but the
                # richer "think on her own and create her own work" engines (the default-mode
                # stream, and idle_maintenance — dream replay, the autonomous scientist, the
                # eureka engine, active curiosity, continuous RSI growth — plus the micro-agent
                # civilization) live behind NyxaraCore.start_cognition(). The console starts it;
                # the always-on daemon must too, or a deployed NYXARA never runs them. Started
                # here, LLM-free and oversight-gated; stopped cleanly in the finally below.
                deep = bool(getattr(cfg, "autonomic_deep_cognition", True)
                            and getattr(settings.features, "continuous_cognition", True))
                app_.state.deep_cognition = False
                if deep:
                    try:
                        if core.start_cognition():
                            app_.state.deep_cognition = True
                            print("NYXARA deep cognition started "
                                  "[default-mode stream + idle_maintenance "
                                  "(dream/scientist/eureka/curiosity/growth) + civilization]")
                    except Exception as cexc:  # noqa: BLE001 — never let cognition block the API
                        print(f"NYXARA deep cognition failed to start: {cexc}")
            except Exception as exc:  # noqa: BLE001 — never let the mind block the server
                app_.state.autonomic = None
                app_.state.autonomic_runtime = None
                print(f"NYXARA background mind failed to start: {exc}")
        else:
            app_.state.autonomic = None
            app_.state.autonomic_runtime = None
        try:
            yield
        finally:
            # Checkpoint everything the daemon learned this run before the process exits, so no
            # minute of learning is lost on shutdown (the periodic autosave covers crashes; this
            # covers a clean stop). Best-effort — a failed save must never hang shutdown.
            try:
                saved = app_.state.core.save_state()
                print(f"NYXARA continuity: checkpointed learned state on shutdown ({saved})")
            except Exception:  # noqa: BLE001 — durability is best-effort, never blocks shutdown
                pass
            # close the senses (idempotent — stop_cognition also closes them)
            try:
                app_.state.core.stop_perception()
            except Exception:  # noqa: BLE001 — best-effort clean shutdown
                pass
            # stop the deep-cognition thread first (it is what start_cognition launched)
            if getattr(app_.state, "deep_cognition", False):
                try:
                    app_.state.core.stop_cognition()
                except Exception:  # noqa: BLE001 — best-effort clean shutdown
                    pass
                app_.state.deep_cognition = False
            if loop is not None:
                loop._running = False  # signal the background mind to exit its next check
            if runtime is not None:
                try:
                    await runtime.stop()
                except Exception:  # noqa: BLE001 — best-effort clean shutdown
                    pass
            elif loop is not None:
                try:
                    await loop.stop()
                except Exception:  # noqa: BLE001
                    pass
            app_.state.autonomic = None
            app_.state.autonomic_runtime = None

    app = FastAPI(title="NYXARA", version=settings.schema_version,
                  description="Sovereign cognitive architecture — authenticated API.",
                  lifespan=_lifespan)

    if cfg.cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=list(cfg.cors_origins),
            allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    # ---- auth dependency ---- #
    def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
        if token is None:
            return  # open (dev convenience); PROD forbids this above
        expected = f"Bearer {token}"
        # constant-time-ish compare; header may be absent
        import hmac
        if not authorization or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="invalid or missing bearer token",
                                headers={"WWW-Authenticate": "Bearer"})

    auth = [Depends(require_auth)]

    # ---- routes ---- #
    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "instance": settings.instance_name,
                "profile": settings.profile.value, "version": settings.schema_version}

    @app.get("/v1/report", dependencies=auth)
    def report() -> dict:
        return core.report()

    @app.get("/status", dependencies=auth)
    def power_status() -> dict:
        """The power surface: which faculties are LIVE right now, plus every feature flag."""
        return core.power_report()

    @app.get("/v1/learning", dependencies=auth)
    def learning() -> dict:
        """Truthful learning state: trained generations, corpus growth, live serving."""
        return core.learning_report()

    @app.get("/v1/scale", dependencies=auth)
    def scale() -> dict:
        """Honest effective scale (Problem #1): small model × the amplifiers she can spend now."""
        return core.scale_report()

    @app.post("/v1/chat", dependencies=auth)
    def chat(req: ChatRequest) -> dict:
        result = core.process(req.message, authority=_authority(req.authority))
        return result.to_dict()

    @app.post("/v1/agent", dependencies=auth)
    def agent(req: AgentRequest) -> dict:
        steps = min(req.max_steps or cfg.max_agent_steps, cfg.max_agent_steps)
        run = core.agent(req.goal, authority=_authority(req.authority), max_steps=steps)
        return run.to_dict()

    @app.post("/v1/self_correct", dependencies=auth)
    def self_correct(req: SelfCorrectRequest) -> dict:
        # A multi-step goal pursued with active self-correction & epistemic uncertainty engaged:
        # she predicts-then-verifies each move, detects loops, and runs a real experiment to fill
        # a knowledge gap instead of spinning — abstaining or escalating honestly when unsure.
        steps = min(req.max_steps or cfg.max_agent_steps, cfg.max_agent_steps)
        return core.self_correct(req.goal, authority=_authority(req.authority), max_steps=steps)

    @app.post("/v1/self_improve", dependencies=auth)
    def self_improve(req: SelfImproveRequest) -> dict:
        # The codebase-level recursive self-improvement loop, made observable: NYXARA reviews her
        # own source, detects weaknesses, and (when ``enact``) applies gauntleted, provably-better
        # edits herself — no LLM — while the intelligence index is threaded across cycles. Bounded
        # by ``cycles`` and halted cleanly by a closed oversight gate; the reports block is dropped
        # to keep the response small (the trajectory + tallies are the signal the Master watches).
        from nyxara.growth.recursive_improvement import RecursiveSelfImprovement
        summary = RecursiveSelfImprovement.from_core(core).run_continuous(
            int(req.cycles), enact=bool(req.enact))
        return {k: v for k, v in summary.items() if k != "reports"}

    @app.post("/v1/delegate", dependencies=auth)
    def delegate(req: DelegateRequest) -> dict:
        steps = min(req.max_steps or cfg.max_agent_steps, cfg.max_agent_steps)
        result = core.delegate(req.name, req.subgoal, max_steps=steps,
                               authority=_authority(req.authority))
        return result.to_dict()

    @app.post("/v1/research", dependencies=auth)
    def research(req: ResearchRequest) -> dict:
        return core.research(req.topic)

    @app.post("/v1/investigate", dependencies=auth)
    def investigate(req: InvestigateRequest) -> dict:
        return core.investigate(req.question)

    @app.post("/v1/discover", dependencies=auth)
    def discover(req: DiscoverRequest) -> dict:
        return core.discover(req.cycles)

    @app.post("/v1/generalize", dependencies=auth)
    def generalize(req: GeneralizeRequest = GeneralizeRequest()) -> dict:
        # No system can cross the wire, so this demonstrates the capability on a hidden,
        # randomly-parameterized alien machine she has never seen — observe→hypothesize→test→model.
        return core.generalize(budget=req.budget)

    @app.post("/v1/understand", dependencies=auth)
    def understand(req: UnderstandRequest) -> dict:
        # A callable can't cross HTTP, so the Master hands her a *declarative* black box: either a
        # dataset of observed input→output rows, or a named law family + params she rebuilds into a
        # box and cracks by probing. Her own faculties model it — no LLM.
        spec: Dict[str, Any] = {}
        if req.dataset:
            spec["dataset"] = req.dataset
        if req.family:
            spec.update({"family": req.family, "params": req.params, "dims": req.dims,
                         "kind": req.kind, "low": req.low, "high": req.high})
        if req.label:
            spec["label"] = req.label
        if not spec:
            return {"error": "provide either 'dataset' or 'family'"}
        return core.understand(spec, budget=req.budget)

    @app.post("/v1/adapt", dependencies=auth)
    def adapt(req: AdaptRequest) -> dict:
        # Drop her into a brand-new environment (declared as a list of systems): she models each with
        # her own faculties and, under real pressure, structurally re-organizes her own brain.
        environment = req.systems if req.systems else None
        return core.adapt(environment, budget=req.budget, label=req.label)

    @app.post("/v1/breakthrough", dependencies=auth)
    def breakthrough(req: BreakthroughRequest) -> dict:
        return core.breakthrough(req.generations, req.population)

    @app.post("/v1/discover-law", dependencies=auth)
    def discover_law(req: DiscoverLawRequest = DiscoverLawRequest()) -> dict:
        # She invents a NEW empirical/physical law — by symbolic regression, SINDy, invariant
        # discovery, or experiments she runs herself in the physics sandbox — with no LLM in the
        # loop. Corroborated on held-out + extrapolation data, or an honest abstention.
        return core.discover_laws(req.rounds, req.domain)

    @app.post("/v1/meta_discover", dependencies=auth)
    def meta_discover(req: MetaDiscoverRequest) -> dict:
        return core.meta_discover(req.topic)

    @app.get("/v1/discoveries", dependencies=auth)
    def discoveries() -> dict:
        # Her independent-discovery record: the law tower she built from her own curiosity — laws by
        # domain, rivals beaten on decisive tests, skills minted, and the director's tallies.
        return core.discoveries()

    @app.post("/v1/discover_domain", dependencies=auth)
    def discover_domain(req: DiscoverDomainRequest) -> dict:
        # She runs her own experiment in the named science and discovers its governing law — genuine
        # zero-to-discovery from a domain she was pointed at, no equation handed to her, no LLM.
        return core.discover_domain(req.domain)

    @app.post("/v1/dream", dependencies=auth)
    def dream(req: DreamRequest = DreamRequest()) -> dict:
        if core.dream_session is None:
            return {"error": "dream session unavailable"}
        return core.dream_session.dream_state(deep=req.deep).to_dict()

    @app.post("/v1/strategize", dependencies=auth)
    def strategize(req: StrategizeRequest) -> dict:
        return core.strategize(req.problem)

    @app.post("/v1/solve", dependencies=auth)
    def solve(req: SolveRequest) -> dict:
        return core.solve(req.problem)

    if cfg.enable_control:
        @app.post("/v1/control/{action}", dependencies=auth)
        def control(action: str, req: ControlRequest = ControlRequest()) -> dict:
            act = action.strip().lower()
            if act == "pause":
                core.pause()
            elif act == "resume":
                core.resume()
            elif act == "scram":
                core.scram(reason=req.reason or "Master stop (api)")
            else:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                    detail=f"unknown control action '{action}'")
            return {"action": act, "control": core.oversight.state.value}

    # ---- continuous real-time perception (the always-on senses) ---- #
    @app.get("/v1/perception", dependencies=auth)
    def perception_status() -> dict:
        rp = getattr(core, "perception", None)
        if rp is None:
            return {"error": "realtime perception unavailable (disabled by config)"}
        return rp.status()

    @app.post("/v1/perception/{action}", dependencies=auth)
    def perception_control(action: str) -> dict:
        rp = getattr(core, "perception", None)
        if rp is None:
            return {"error": "realtime perception unavailable (disabled by config)"}
        act = action.strip().lower()
        if act == "start":
            core.start_perception()
        elif act == "stop":
            core.stop_perception()
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"unknown perception action '{action}'")
        return {"action": act, "running": rp.running}

    @app.websocket("/v1/perception/ws")
    async def perception_ws(socket: WebSocket) -> None:
        """A live window into her senses: every PerceptEvent streams here as it lands."""
        if token is not None and socket.query_params.get("token") != token:
            await socket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        rp = getattr(core, "perception", None)
        await socket.accept()
        if rp is None:
            await socket.send_json({"error": "realtime perception unavailable"})
            await socket.close()
            return
        import asyncio
        loop_ = asyncio.get_running_loop()
        queue: "asyncio.Queue[dict]" = asyncio.Queue(maxsize=256)

        def _on_event(d: dict) -> None:  # runs on the perception thread
            try:
                loop_.call_soon_threadsafe(queue.put_nowait, d)
            except Exception:  # noqa: BLE001 — a full queue drops, never blocks sensing
                pass

        rp.add_listener(_on_event)
        try:
            await socket.send_json({"hello": "perception feed", "status": rp.status()})
            while True:
                ev = await queue.get()
                await socket.send_json(ev)
        except WebSocketDisconnect:
            return
        finally:
            rp.remove_listener(_on_event)

    # ---- fractal temporal hierarchies (the multi-dimensional mind) ---- #
    @app.get("/v1/temporal/report", dependencies=auth)
    def temporal_report() -> dict:
        ft = getattr(core, "fractal_temporal", None)
        if ft is None:
            return {"error": "fractal temporal hierarchy unavailable"}
        return ft.report()

    @app.get("/v1/temporal/awareness", dependencies=auth)
    def temporal_awareness() -> dict:
        ft = getattr(core, "fractal_temporal", None)
        if ft is None:
            return {"error": "fractal temporal hierarchy unavailable"}
        latest = ft.awareness()
        return latest.to_dict() if latest is not None else {"awareness": None}

    @app.post("/v1/temporal/tick", dependencies=auth)
    def temporal_tick(req: TemporalTickRequest = TemporalTickRequest()) -> dict:
        ft = getattr(core, "fractal_temporal", None)
        if ft is None:
            return {"error": "fractal temporal hierarchy unavailable"}
        n = max(1, min(req.beats, 100_000))
        ft.run_for(n)
        return ft.report()

    @app.post("/v1/memory/save", dependencies=auth)
    def memory_save(req: MemoryPathRequest = MemoryPathRequest()) -> dict:
        return {"saved": core.save_state(req.path)}

    @app.post("/v1/memory/load", dependencies=auth)
    def memory_load(req: MemoryPathRequest = MemoryPathRequest()) -> dict:
        return {"loaded": core.load_state(req.path)}

    @app.websocket("/v1/ws")
    async def ws(socket: WebSocket) -> None:
        # Authenticate before accepting: token via ?token= query param.
        if token is not None and socket.query_params.get("token") != token:
            await socket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        await socket.accept()
        try:
            while True:
                data = await socket.receive_json()
                message = str(data.get("message", "")).strip()
                if not message:
                    await socket.send_json({"error": "empty message"})
                    continue
                result = core.process(message, authority=_authority(data.get("authority", "owner")))
                await socket.send_json(result.to_dict())
        except WebSocketDisconnect:
            return

    app.state.core = core
    app.state.settings = settings
    return app


def run(core: Any = None, *, settings: Optional[NyxaraSettings] = None,
        **uvicorn_kwargs: Any) -> None:
    """Serve the app with uvicorn (blocking)."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise _missing_dep_error() from exc
    settings = settings or get_settings()
    app = create_app(core=core, settings=settings)
    uvicorn_kwargs.setdefault("host", settings.server.host)
    uvicorn_kwargs.setdefault("port", settings.server.port)
    uvicorn.run(app, **uvicorn_kwargs)
