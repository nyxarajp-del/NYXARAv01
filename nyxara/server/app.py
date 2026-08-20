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
* ``POST /v1/prove``            — vectorized reasoning, disposed by the exact prover:
  ``{statement, kind?, candidate_answer?}`` → PROVEN/REFUTED/ABSTAIN + certificate (no LLM).
* ``POST /v1/causal-repair``    — Causal Code Engine: ``{test_path?, max_fixes?}`` → causal-tree
  root-cause of a failing test, then gated reversible repair of the causal root module.
* ``POST /v1/teleology``        — Recursive Self-Directed Teleology: ``{launch?}`` → invent her own
  measurable self-improvement targets, envelope-gated (out-of-envelope rejected). No LLM.
* ``POST /v1/synthesize-scenarios`` — Epistemic Auto-Evolution: ``{n?, max_tests?}`` → Monte-Carlo
  concurrent-fault edge cases over simulated conditions, self-hypothesized + tested. No LLM.
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


class TrainDatasetRequest(BaseModel):
    # A server-side path to a dataset file or folder (.txt/.md, .jsonl/.json, .csv/.tsv,
    # .pdf/.docx). Not an upload — the same posture as every other route here.
    path: str = Field(..., min_length=1)
    # Forge generations to run after ingesting. Bounded — never an unattended marathon.
    generations: int = Field(default=1, ge=1, le=10)
    # Ingest and report what she WOULD learn without training anything.
    ingest_only: bool = False
    # Injection posture: a Master-supplied file is not untrusted web text, so "warn" keeps a
    # flagged record and counts it; "drop" enforces the stricter acquire.py stance.
    screen: Optional[str] = Field(default=None, pattern="^(warn|drop|off)$")


class CausalQueryRequest(BaseModel):
    # "why": causes of `target`. "do": effects of setting `cause` to `value`.
    # "counterfactual": would `target` still have happened without `cause`?
    cause: Optional[str] = None
    target: Optional[str] = None
    value: float = 1.0


class HyperspaceRequest(BaseModel):
    term: Optional[str] = None                  # for /embed and /nearest
    source_domain: Optional[str] = None         # for /analogy
    target_domain: Optional[str] = None
    k: int = Field(default=5, ge=1, le=50)


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


class VsaProveRequest(BaseModel):
    statement: str = Field(..., min_length=1)
    kind: Optional[str] = None
    candidate_answer: Optional[str] = None


class CausalRepairRequest(BaseModel):
    test_path: Optional[str] = None
    max_fixes: int = Field(default=3, ge=0, le=20)


class TeleologyRequest(BaseModel):
    launch: bool = False


class SynthesizeScenariosRequest(BaseModel):
    n: int = Field(default=8, ge=1, le=128)
    max_tests: Optional[int] = Field(default=None, ge=0, le=128)


class AbyssTimelinesRequest(BaseModel):
    actions: List[str]
    state: Optional[List[float]] = None          # omitted ⇒ the embodied agent's live state
    branches: int = Field(default=512, ge=2, le=100_000)
    horizon: int = Field(default=6, ge=1, le=64)
    risk_aversion: float = Field(default=0.5, ge=0.0, le=5.0)
    budget_ms: Optional[float] = Field(default=None, gt=0.0)


class AbyssButterflyRequest(BaseModel):
    state: Optional[List[float]] = None          # omitted ⇒ the embodied agent's live state
    horizon: int = Field(default=6, ge=1, le=64)
    delta: float = Field(default=0.02, gt=0.0)
    labels: Optional[List[str]] = None


class ControlRequest(BaseModel):
    reason: str = ""


class MemoryPathRequest(BaseModel):
    path: Optional[str] = None


class TemporalTickRequest(BaseModel):
    # how many millisecond-scale beats to drive the fractal hierarchy through (nested
    # meso roll-ups and macro observations fire at their boundaries).
    beats: int = Field(default=1, ge=1)


# ---- the defensive layer (guard/*.py) ---- #
class ContainRequest(BaseModel):
    component: str
    # isolate (default) cuts the component off; release restores it and is owner-only.
    action: str = "isolate"
    reason: str = ""


class NetPolicyRequest(BaseModel):
    host: str
    # deny | allow — both are Master-authority policy changes on the live firewall.
    action: str = "deny"
    reason: str = ""


class PhagocytoseRequest(BaseModel):
    """A hostile sample to digest into a permanent antibody."""
    sample: str
    kind: str = "prompt_injection"


# ---- self-knowledge & the training stack (growth/*.py) ---- #
class CapabilityRequest(BaseModel):
    capability: str


class ForageRequest(BaseModel):
    topic: str


class CorpusRequest(BaseModel):
    tokens_budget: Optional[int] = Field(default=None, ge=1)
    out_dir: Optional[str] = None


class PreferenceTrainRequest(BaseModel):
    limit: int = Field(default=256, ge=1, le=8192)
    flywheel_path: Optional[str] = None


class ExpandRequest(BaseModel):
    # per-domain measured losses; the weakest one gets the new expert.
    losses: Optional[Dict[str, float]] = None


class SpeculateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=32, ge=1, le=512)


class LoadModuleRequest(BaseModel):
    path: str


# ---- delegation & the hive (agency/*.py) ---- #
class AgentTaskRequest(BaseModel):
    task: str
    name: str = "api"


class HiveRequest(BaseModel):
    problem: str
    difficulty: float = Field(default=0.5, ge=0.0, le=1.0)


# ---- taste, register, conscience, foresight ---- #
class AestheticRequest(BaseModel):
    artifact: str
    kind: Optional[str] = None


class ModeRequest(BaseModel):
    mode: Optional[str] = None


class MoralRequest(BaseModel):
    action: str


class ForesightRequest(BaseModel):
    horizon_days: int = Field(default=90, ge=1, le=3650)


# --- NYX V.01 (nyxara/nyx/) — the unified brain ------------------------------ #
class NyxThinkRequest(BaseModel):
    stimulus: str
    remember: bool = True


class NyxRecallRequest(BaseModel):
    cue: str
    k: int = Field(default=5, ge=1, le=64)


class NyxGroundRequest(BaseModel):
    word: str


class NyxChronosRequest(BaseModel):
    decision: str


# --- NYX V.02 (nyxara/nyx/) — the nineteen ----------------------------------- #
class NyxTextRequest(BaseModel):
    """One string in. Used by the routes whose whole input is a sentence."""

    text: str


class NyxAuthorRequest(BaseModel):
    spec: str
    name: str = ""
    load: Optional[bool] = None


class NyxActRequest(BaseModel):
    request: str
    tool: str = ""
    args: Dict[str, Any] = Field(default_factory=dict)
    # Default to the honest thing over HTTP: say what she *would* do. Acting is opt-in per call
    # and still goes through the registry's unchanged capability/risk/governor pipeline.
    dry_run: bool = True


class NyxAxiomRequest(BaseModel):
    problem: str
    name: str = ""


class NyxEvolveRequest(BaseModel):
    # ``force`` waives the *cadence* — run now rather than wait for the next slot. It does not
    # waive the sample floor, which is an epistemic guard rather than a scheduling one; that
    # needs its own name so it can never be waived as a side effect of asking her to run now.
    force: bool = False
    ignore_sample_floor: bool = False


class NyxDebateRequest(BaseModel):
    claim: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    verified: bool = False


class NyxCausalRequest(BaseModel):
    cause: str
    effect: str
    question: str = ""


class NyxGoalRequest(BaseModel):
    request: str
    # Same rule as /v1/njp routes: decompose and price by default, execute only when asked.
    run: bool = False


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
        # Her own brains, provisioned before she serves a single request.
        #
        # This ran only in ``__main__.main`` — the interactive console — so the daemon and the
        # HTTP server, which are how she actually runs unattended, never installed the cortex
        # runtime, never fetched the weights, and never warmed them. Nothing failed: the ladder
        # walked past the missing rung and answered on the n-gram floor, correctly and silently,
        # for the entire life of the process. "Boot her and it just works" has to mean every
        # front door, not the one a person happens to be sitting at.
        #
        # It is the same call the console makes, with the same guarantees: already provisioned is
        # a file check, every download and install is flag-gated, and any failure is logged and
        # never raised — so a cold or offline box still serves, one rung lower, and says so.
        try:
            from nyxara.growth.bootstrap import ensure_primary_model
            ensure_primary_model(log=print)
        except Exception as bexc:  # noqa: BLE001 — a failed provision must never block the port
            print(f"NYXARA primary brain: provisioning skipped ({bexc})")
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
                      f"{cfg.autonomic_growth_every}, inner_life {inner_life}, "
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

    @app.post("/v1/train", dependencies=auth)
    def train_dataset(req: TrainDatasetRequest) -> dict:
        # Hand NYXARA a dataset and let her OWN brain train on it (growth/dataset.py →
        # growth/foundry.py). Ingestion is durable: the records land in her corpus store, so every
        # later forge trains on them too, not just this call. The forge backend is pinned to
        # "auto" — a from-scratch net (the NumPy transformer on a torch-less box), never an
        # adapter over someone else's pretrained base. Every promotion still clears the full
        # gauntlet, so a bad dataset can only produce a candidate that is refused, never a
        # disloyal live brain.
        from nyxara.growth.dataset import ingest_dataset
        from nyxara.growth.foundry import Foundry

        # a process-local copy so the route never mutates the app's live configuration
        train_settings = settings.model_copy(deep=True)
        train_settings.foundry.enabled = True
        train_settings.foundry.backend = "auto"
        report = ingest_dataset(req.path, settings=train_settings, screen=req.screen)
        out: dict = {"ingest": report.to_dict()}
        if req.ingest_only:
            return out
        try:
            from nyxara.growth.bootstrap import IDENTITY_SEED
            results = Foundry(settings=train_settings,
                              seed_corpus=IDENTITY_SEED).self_improve(generations=req.generations)
        except Exception as exc:  # noqa: BLE001 — a starved forge is a reported fact, not a 500
            out["error"] = str(exc)
            return out
        out["generations"] = [r.to_dict() for r in results]
        out["promoted"] = sum(1 for r in results if r.promoted)
        return out

    def _causal_graph() -> Any:
        """The persisted causal graph beside her dataset corpus, or a fresh empty one."""
        from nyxara.growth.dataset import _default_store_path
        from nyxara.mind.causal_world_model import CausalWorldModel

        try:
            path = _default_store_path(settings).with_name("causal_graph.json")
            if path.exists():
                # ``load`` is a CLASSMETHOD returning a new instance — calling it on an object
                # silently discards the restored graph and leaves you with an empty one.
                return CausalWorldModel.load(str(path))
        except Exception:  # noqa: BLE001 — a missing/corrupt graph is simply an empty one
            pass
        return CausalWorldModel()

    def _concept_space() -> Any:
        from nyxara.growth.dataset import _default_store_path
        from nyxara.mind.concept_hyperspace import ConceptHyperspace

        space = ConceptHyperspace(dim=64, seed=0, lr=0.1)
        try:
            space.load(_default_store_path(settings).with_name("hyperspace.json"))
        except Exception:  # noqa: BLE001
            pass
        return space

    @app.post("/v1/causal/{action}", dependencies=auth)
    def causal(action: str, req: CausalQueryRequest) -> dict:
        # Pearl's ladder over the graph she learned from the Master's data (and, at lower and
        # clearly-labelled confidence, from prose). Every link reports whether it was MEASURED or
        # merely asserted in text — a sentence someone wrote must never read like a measurement.
        model = _causal_graph()
        if action == "why":
            if not req.target:
                raise HTTPException(status_code=400, detail="target required")
            label = model.match_label(req.target) or req.target
            links = model.why(label)
            return {"target": label,
                    "causes": [{"cause": link.cause, "confidence": link.confidence,
                                "strength": link.strength, "measured": link.measured,
                                "describe": link.describe()} for link in links[:10]]}
        if action == "do":
            if not req.cause:
                raise HTTPException(status_code=400, detail="cause required")
            label = model.match_label(req.cause) or req.cause
            effects = model.do({label: float(req.value)})
            return {"intervention": {label: req.value},
                    "effects": effects if isinstance(effects, dict) else {}}
        if action == "counterfactual":
            if not (req.cause and req.target):
                raise HTTPException(status_code=400, detail="cause and target required")
            cause = model.match_label(req.cause) or req.cause
            target = model.match_label(req.target) or req.target
            result = model.counterfactual(cause, target, factual=1.0, counter=0.0,
                                          evidence=model.latest_evidence(model.labels()))
            return {"describe": result.describe(), "delta": result.delta,
                    "abducted": result.abducted}
        if action == "validate":
            # the honest one: check the learner against sim/ worlds whose laws are true by
            # construction, so the answer is not another thing she inferred
            from nyxara.growth.causal_validation import validate_all
            return {"reports": [r.to_dict() for r in validate_all()]}
        raise HTTPException(status_code=404, detail=f"unknown causal action {action!r}")

    @app.post("/v1/hyperspace/{action}", dependencies=auth)
    def hyperspace(action: str, req: HyperspaceRequest) -> dict:
        space = _concept_space()
        if not space.concepts:
            return {"error": "no fitted concept space yet — run nyxara-grow --hyperspace"}
        if action == "embed":
            if not req.term:
                raise HTTPException(status_code=400, detail="term required")
            concept = space.concepts.get(req.term.lower())
            if concept is None:
                return {"term": req.term, "known": False}
            return {"term": req.term, "known": True, "point": concept.point,
                    "radius": concept.radius}
        if action == "nearest":
            if not req.term:
                raise HTTPException(status_code=400, detail="term required")
            return {"term": req.term,
                    "nearest": [{"name": n, "distance": d}
                                for n, d in space.nearest(req.term.lower(), req.k)]}
        if action == "analogy":
            if not (req.source_domain and req.target_domain):
                raise HTTPException(status_code=400,
                                    detail="source_domain and target_domain required")
            found = space.find_correspondences(req.source_domain, req.target_domain)
            # an empty list is a real answer: a correspondence below the bar is not returned,
            # because an analogy asserted without support is worse than none
            return {"correspondences": [c.to_dict() for c in found]}
        raise HTTPException(status_code=404, detail=f"unknown hyperspace action {action!r}")

    @app.post("/v1/qualia/{action}", dependencies=auth)
    def qualia(action: str, req: HyperspaceRequest) -> dict:
        space = _concept_space()
        if not space.concepts:
            return {"error": "no fitted concept space yet — run nyxara-grow --hyperspace"}
        from nyxara.qualia import BabelCodec, QualiaWorkspace

        workspace = QualiaWorkspace(space, min_cluster=2)
        report = workspace.synthesise()
        codec = BabelCodec(workspace)
        if action == "concepts":
            # `usable` is gated on certification: an unnamed cluster is not automatically a truth,
            # so it may sit in the workspace and still be withheld from an answer.
            return {"report": report.to_dict(),
                    "alien": [c.to_dict() for c in workspace.alien_concepts()],
                    "usable": [c.to_dict() for c in workspace.usable()]}
        if action == "decode":
            return {"described": codec.describe_all(certified_only=False)}
        if action == "encode":
            if not req.term:
                raise HTTPException(status_code=400, detail="term required")
            return {"term": req.term, "point": codec.encode(req.term.lower())}
        raise HTTPException(status_code=404, detail=f"unknown qualia action {action!r}")

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

    @app.post("/v1/prove", dependencies=auth)
    def prove(req: VsaProveRequest) -> dict:
        # Vectorized reasoning disposed by the exact prover: a machine-checkable certificate
        # (PROVEN/REFUTED) or an honest ABSTAIN — never token-guessing, no LLM.
        return core.vsa_prove(req.statement, kind=req.kind,
                              candidate_answer=req.candidate_answer)

    @app.post("/v1/causal-repair", dependencies=auth)
    def causal_repair(req: CausalRepairRequest = CausalRepairRequest()) -> dict:
        # Causal-tree root-cause of a failing test → gated, reversible repair of the causal root.
        return core.causal_repair(test_path=req.test_path, max_fixes=req.max_fixes)

    @app.post("/v1/teleology", dependencies=auth)
    def teleology(req: TeleologyRequest = TeleologyRequest()) -> dict:
        # Invent her own measurable self-improvement targets — hard-filtered through the owner
        # alignment envelope (out-of-envelope targets rejected before adoption). No LLM.
        return core.self_direct(launch=req.launch)

    @app.post("/v1/abyss/timelines", dependencies=auth)
    def abyss_timelines(req: AbyssTimelinesRequest) -> dict:
        # Abyss · 1 — branch the present into many futures over the shared world model and rank
        # the candidate actions by risk-aware outcome (CVaR tail, not just the mean). Reports
        # itself blind rather than ranking noise when the model has not learned enough.
        return core.simulate_futures(req.actions, state=req.state, branches=req.branches,
                                     horizon=req.horizon, risk_aversion=req.risk_aversion,
                                     budget_ms=req.budget_ms)

    @app.post("/v1/abyss/butterfly", dependencies=auth)
    def abyss_butterfly(req: AbyssButterflyRequest = AbyssButterflyRequest()) -> dict:
        # Abyss · 2 — perturb each numeric dimension of the present in turn and rank them by how
        # far the tiny change cascades. Answers which small detail of now most controls next.
        return core.butterfly_scan(state=req.state, horizon=req.horizon, delta=req.delta,
                                   labels=req.labels)

    @app.post("/v1/synthesize-scenarios", dependencies=auth)
    def synthesize_scenarios(req: SynthesizeScenariosRequest = SynthesizeScenariosRequest()) -> dict:
        # Monte-Carlo concurrent-fault edge-case discovery over simulated conditions (no real host
        # stressed, no LLM): generate never-seen scenarios, self-hypothesize, test in the sandbox.
        return core.hunt_edge_cases(n=req.n, max_tests=req.max_tests)

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

    # ------------------------------------------------------------------ #
    # THE DEFENSIVE LAYER (guard/*.py)
    # Read paths are plain reports; the two write paths (containment, network
    # policy) only ever RESTRICT — nothing here can grant a capability, and
    # /scram, oversight and corrigibility are untouched by all of it.
    # ------------------------------------------------------------------ #
    @app.get("/v1/guard", dependencies=auth)
    def guard_status() -> dict:
        return core.guard_report()

    @app.get("/v1/guard/anomalies", dependencies=auth)
    def guard_anomalies(limit: int = 20) -> dict:
        det = getattr(core, "anomaly", None)
        if det is None:
            return {"enabled": False, "reason": "anomaly detection is disabled"}
        recent = det.recent(max(1, min(int(limit), 200)))
        return {"enabled": True, "report": det.report(),
                "recent": [a.to_dict() for a in recent]}

    @app.post("/v1/guard/contain", dependencies=auth)
    def guard_contain(req: ContainRequest) -> dict:
        con = getattr(core, "containment", None)
        if con is None:
            raise HTTPException(status_code=503, detail="containment is disabled")
        try:
            if req.action.strip().lower() in ("release", "free"):
                # release is owner-exclusive; this route is already behind the Master's token
                return {"released": con.release(req.component, owner=True).to_dict()}
            return {"contained": con.isolate(
                req.component, reason=req.reason or "API order").to_dict()}
        except Exception as exc:  # noqa: BLE001 — an unknown component is a 400, not a 500
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/guard/network", dependencies=auth)
    def guard_network(req: NetPolicyRequest) -> dict:
        net = getattr(core, "netsec", None)
        if net is None:
            raise HTTPException(status_code=503, detail="network defence is disabled")
        try:
            if req.action.strip().lower() == "allow":
                net.allow_host(req.host, owner=True)
            else:
                net.deny_host(req.host, reason=req.reason or "API order")
            return {"host": req.host, "action": req.action, "report": net.report()}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/guard/phagocytose", dependencies=auth)
    def guard_phagocytose(req: PhagocytoseRequest) -> dict:
        """Digest a hostile sample into a permanent antibody (guard/phagocytosis.py)."""
        phagocyte = getattr(core, "phagocyte", None)
        if phagocyte is None:
            raise HTTPException(status_code=503, detail="the immune layer is disabled")
        result = phagocyte.engulf(req.sample, kind=req.kind)
        return {"result": result.to_dict() if hasattr(result, "to_dict") else str(result),
                "antibodies": len(phagocyte.playbook.antibodies())}

    @app.get("/v1/guard/survival", dependencies=auth)
    def guard_survival() -> dict:
        surv = getattr(core, "survival", None)
        if surv is None:
            return {"enabled": False, "reason": "the survival manager is disabled"}
        return {"enabled": True, "report": surv.report(),
                "backups": len(surv.store.all()),
                "chain_verified": surv.store.verify_chain(),
                "correction_dominates_survival": surv.correction_dominates_survival()}

    # ------------------------------------------------------------------ #
    # SELF-KNOWLEDGE, PROVENANCE & THE TRAINING STACK (growth/*.py)
    # ------------------------------------------------------------------ #
    @app.get("/v1/capabilities", dependencies=auth)
    def capabilities() -> dict:
        reg = getattr(core, "capabilities", None)
        if reg is None:
            return {"enabled": False, "reason": "the capability registry is disabled"}
        return {"enabled": True, "can_do": reg.self_report(),
                "calibration": reg.calibration()}

    @app.post("/v1/capabilities/can_i", dependencies=auth)
    def can_i(req: CapabilityRequest) -> dict:
        """An honest, evidence-backed answer — undemonstrated reports as untested."""
        return core.can_i(req.capability)

    @app.get("/v1/lineage", dependencies=auth)
    def lineage() -> dict:
        led = getattr(core, "lineage", None)
        if led is None:
            return {"enabled": False, "reason": "the lineage ledger is disabled"}
        history = led.history()
        return {"enabled": True, "generations": len(history), "head": led.head(),
                "chain_verified": led.verify_chain(),
                "history": [r.to_dict() for r in history[-50:]]}

    @app.post("/v1/forage", dependencies=auth)
    def forage(req: ForageRequest) -> dict:
        forager = getattr(core, "forager", None)
        if forager is None:
            raise HTTPException(status_code=503, detail="epistemic foraging is disabled")
        return forager.forage(req.topic).to_dict()

    @app.post("/v1/seed", dependencies=auth)
    def genesis_seed() -> dict:
        seeder = getattr(core, "genesis_seed", None)
        if seeder is None:
            raise HTTPException(status_code=503, detail="the genesis seed is disabled")
        seed = seeder.snapshot_core(core)
        # the signed bundle itself is never returned over the wire — only its identity
        return {"seed": seed.seed, "bytes": len(seed.bundle)}

    @app.post("/v1/train/corpus", dependencies=auth)
    def train_corpus(req: CorpusRequest = CorpusRequest()) -> dict:
        return core.build_corpus(tokens_budget=req.tokens_budget, out_dir=req.out_dir)

    @app.post("/v1/train/dpo", dependencies=auth)
    def train_dpo(req: PreferenceTrainRequest = PreferenceTrainRequest()) -> dict:
        return core.preference_train(limit=req.limit, flywheel_path=req.flywheel_path)

    @app.post("/v1/train/expand", dependencies=auth)
    def train_expand(req: ExpandRequest = ExpandRequest()) -> dict:
        return core.expand_experts(losses=req.losses)

    @app.post("/v1/speculate", dependencies=auth)
    def speculate(req: SpeculateRequest) -> dict:
        return core.speculative_report(req.prompt, max_tokens=req.max_tokens)

    @app.post("/v1/module/load", dependencies=auth)
    def load_module(req: LoadModuleRequest) -> dict:
        """Screen and import a file she forged herself. Confined to the forged root."""
        return core.load_forged_module(req.path)

    # ------------------------------------------------------------------ #
    # DELEGATION, THE HIVE, HER DISTRIBUTED SELF (agency/*.py)
    # ------------------------------------------------------------------ #
    @app.get("/v1/agents", dependencies=auth)
    def agents_status() -> dict:
        orch = getattr(core, "sub_agents", None)
        if orch is None:
            return {"enabled": False, "reason": "sub-agents are unavailable"}
        return {"enabled": True, **orch.report()}

    @app.post("/v1/agents", dependencies=auth)
    def agents_delegate(req: AgentTaskRequest) -> dict:
        """Delegate to a sub-agent holding a strict SUBSET of her own permissions."""
        orch = getattr(core, "sub_agents", None)
        if orch is None:
            raise HTTPException(status_code=503, detail="sub-agents are unavailable")
        from nyxara.agency.agents import AgentSpec
        return orch.delegate(AgentSpec(name=req.name), req.task).to_dict()

    @app.post("/v1/hive", dependencies=auth)
    def hive(req: HiveRequest) -> dict:
        council = getattr(core, "hive", None)
        if council is None:
            raise HTTPException(status_code=503, detail="the hive council is unavailable")
        return council.solve(req.problem, difficulty=req.difficulty).to_dict()

    @app.get("/v1/cluster", dependencies=auth)
    def cluster() -> dict:
        node = getattr(core, "cluster", None)
        if node is None:
            return {"enabled": False, "reason": "the mesh is disabled"}
        return {"enabled": True, "single_node": node.is_single_node(),
                "log_entries": len(node.log())}

    # ------------------------------------------------------------------ #
    # TASTE, REGISTER, CONSCIENCE, FORESIGHT, DETERMINISM
    # Advisory faculties: each returns its reading as data. None of them can
    # change a disposition — that stays with the kernel's gates alone.
    # ------------------------------------------------------------------ #
    @app.post("/v1/aesthetic", dependencies=auth)
    def aesthetic(req: AestheticRequest) -> dict:
        judge = getattr(core, "aesthetic", None)
        if judge is None:
            raise HTTPException(status_code=503, detail="the aesthetic judge is disabled")
        kwargs = {}
        if req.kind:
            from nyxara.identity.aesthetic import ArtifactKind
            try:
                kwargs["kind"] = ArtifactKind(req.kind.strip().lower())
            except ValueError as exc:
                raise HTTPException(status_code=400,
                                    detail=f"unknown artifact kind {req.kind!r}") from exc
        return judge.judge(req.artifact, **kwargs).to_dict()

    @app.post("/v1/mode", dependencies=auth)
    def mode(req: ModeRequest = ModeRequest()) -> dict:
        """Set or read the register blend. Character is sealed and never moves (Rule 4)."""
        modes = getattr(core, "modes", None)
        if modes is None:
            raise HTTPException(status_code=503, detail="mode blending is disabled")
        if req.mode:
            from nyxara.identity.modes import Mode
            try:
                modes.set_mode(Mode(req.mode.strip().lower()))
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"unknown mode {req.mode!r}; "
                           f"expected one of {[m.value for m in Mode]}") from exc
        return modes.status()

    @app.post("/v1/moral", dependencies=auth)
    def moral(req: MoralRequest) -> dict:
        """Consequentialist + deontological + virtue readings, disagreement left visible."""
        evaluator = getattr(core, "moral", None)
        if evaluator is None:
            raise HTTPException(status_code=503, detail="moral evaluation is disabled")
        from nyxara.mind.moral import MoralAction
        evaluation = evaluator.evaluate(MoralAction(description=req.action))
        return {"summary": evaluation.summary(), "contested": evaluation.contested,
                **evaluation.to_dict()}

    @app.post("/v1/foresight", dependencies=auth)
    def foresight(req: ForesightRequest = ForesightRequest()) -> dict:
        fs = getattr(core, "foresight", None)
        if fs is None:
            raise HTTPException(status_code=503, detail="foresight is disabled")
        return fs.project(req.horizon_days).to_dict()

    @app.get("/v1/thermal", dependencies=auth)
    def thermal() -> dict:
        mon = getattr(core, "thermal", None)
        if mon is None:
            return {"enabled": False, "reason": "the thermal sense is unavailable"}
        telemetry = mon.read()
        return {"enabled": True, "budget": mon.budget(telemetry),
                "should_defer": mon.should_defer(),
                "telemetry": {k: v for k, v in vars(telemetry).items()
                              if not k.startswith("_")} if hasattr(telemetry, "__dict__") else {}}

    @app.get("/v1/replay", dependencies=auth)
    def replay_status() -> dict:
        rec = getattr(core, "recorder", None)
        if rec is None:
            return {"enabled": False, "reason": "replay recording is off"}
        return {"enabled": True, "entries": len(rec.entries),
                "cap": core._REPLAY_MAX_ENTRIES}

    @app.post("/v1/replay/save", dependencies=auth)
    def replay_save(req: MemoryPathRequest = MemoryPathRequest()) -> dict:
        path = core.save_replay(req.path)
        if path is None:
            raise HTTPException(status_code=503, detail="no tape to write")
        return {"saved": path}

    @app.post("/v1/eval/generalization", dependencies=auth)
    def eval_generalization() -> dict:
        """Does she generalize HERSELF, or only via the LLM? Answered with a number."""
        from nyxara.eval.generalization import run_generalization_benchmark
        return run_generalization_benchmark().to_dict()

    # ----------------------------------------------------------------------- #
    # NJP V.01 — the brain (nyxara/njp/). Every route degrades to a 404-shaped payload
    # rather than an error when the brain is disabled, and none of them can move a turn past
    # the sovereign gate: /v1/chat is still the only way to *act*.
    # ----------------------------------------------------------------------- #
    def _brain() -> Any:
        return getattr(core, "njp", None)

    # Organs readable over the wire, by URL segment. Read-only: this route never acts.
    _NJP_ORGANS = ("fabric", "truth", "soulsync", "evolver", "pulse", "ledger",
                   "learner", "calculate")
    # Where the URL segment does not match the attribute on the brain.
    _ORGAN_ATTR = {"calculate": "calculator"}

    def _off(flag: str = "ENABLED") -> dict:
        return {"available": False,
                "reason": f"NJP V.01 is disabled (NYXARA_NJP__{flag}=false)."}

    @app.get("/v1/njp/status", dependencies=auth)
    def njp_status() -> dict:
        """Her brain measured, not described: fabric shape, every organ, and whether she grows."""
        brain = _brain()
        return brain.stats() if brain is not None else _off()

    @app.get("/v1/njp/fabric", dependencies=auth)
    def njp_fabric() -> dict:
        """The automaton itself: cells, synapses, mean degree, and the growth counters."""
        brain = _brain()
        return brain.fabric.stats() if brain is not None else _off()

    @app.get("/v1/njp/ledger", dependencies=auth)
    def njp_ledger() -> dict:
        """Then vs now. Whether she is actually more than she was, as numbers."""
        brain = _brain()
        return brain.report() if brain is not None else _off("LEDGER_ENABLED")

    @app.post("/v1/njp/think", dependencies=auth)
    def njp_think(req: NyxThinkRequest) -> dict:
        """One whole turn — and the fabric is measurably larger when it returns."""
        brain = _brain()
        if brain is None:
            return _off()
        thought = brain.think(req.stimulus)
        return thought.to_dict() if thought is not None else {}

    @app.post("/v1/njp/recall", dependencies=auth)
    def njp_recall(req: NyxThinkRequest) -> dict:
        """Content-addressed recall. No token window is consulted."""
        brain = _brain()
        if brain is None or brain.memory is None:
            return _off("MEMORY_ENABLED")
        got = brain.memory.recall(req.stimulus)
        return got.to_dict() if got is not None else {"recall": None}

    @app.post("/v1/njp/anticipate", dependencies=auth)
    def njp_anticipate(req: NyxThinkRequest) -> dict:
        """What she expects next, WITHOUT settling the fabric.

        Comes back ``trusted=false`` with a reason when the manifold has not earned the
        prediction — an unearned foresight is reported as absent, never spent."""
        brain = _brain()
        if brain is None:
            return _off()
        got = brain.anticipate(req.stimulus)
        return got.to_dict() if hasattr(got, "to_dict") else {"anticipated": None}

    @app.post("/v1/njp/expand", dependencies=auth)
    def njp_expand() -> dict:
        """Run one plasticity pass now: potentiate, grow, prune, mint cells if warranted."""
        brain = _brain()
        if brain is None:
            return _off()
        got = brain.fabric.expand()
        return got.to_dict() if got is not None else {}

    @app.post("/v1/njp/evolve", dependencies=auth)
    def njp_evolve() -> dict:
        """One self-rewrite attempt: measure, propose, verify the claim, keep or roll back.

        Refuses when oversight is paused or scrammed, and when the standing authorisation to
        enact is not set. Every outcome is journalled."""
        got = core.njp_evolve()
        return got.to_dict() if hasattr(got, "to_dict") else _off("EVOLVE_ENABLED")

    @app.post("/v1/njp/pulse", dependencies=auth)
    def njp_pulse() -> dict:
        """One beat of the continuous loop: expand, consolidate on cadence, evolve on cadence."""
        return core.njp_tick() or _off("PULSE_ENABLED")

    def _organ(name: str, flag: str) -> dict:
        """One organ's report, or the same 404-shaped payload every njp route degrades to.

        An organ that is switched off is *absent* rather than reporting zeros — the distinction
        this whole package keeps, carried out to the wire so a caller can tell "off" from "idle".
        """
        brain = _brain()
        if brain is None:
            return _off("NJP_ENABLED")
        organ = getattr(brain, name, None)
        if organ is None:
            return _off(flag)
        try:
            return organ.stats()
        except Exception:  # noqa: BLE001
            return {"error": f"{name} stats failed"}

    @app.get("/v1/njp/grounding", dependencies=auth)
    def njp_grounding() -> dict:
        """Entities, relations and learned extraction patterns — what words became structure."""
        return _organ("grounder", "GROUNDING_ENABLED")

    @app.get("/v1/njp/world", dependencies=auth)
    def njp_world() -> dict:
        """Events, causal links and what she can explain."""
        return _organ("world", "WORLD_ENABLED")

    @app.get("/v1/njp/memory-levels", dependencies=auth)
    def njp_memory_levels() -> dict:
        """Working / episodic / semantic / procedural / autobiographical, with retention."""
        return _organ("levels", "LEVELS_ENABLED")

    @app.get("/v1/njp/discover", dependencies=auth)
    def njp_discover() -> dict:
        """Abstractions proposed, confirmed on held-out episodes, or refuted and deleted."""
        return _organ("discoverer", "DISCOVER_ENABLED")

    @app.get("/v1/njp/reason", dependencies=auth)
    def njp_reason() -> dict:
        """How deep she has been going, and how often she declined to decide."""
        return _organ("reasoner", "REASON_ENABLED")

    @app.get("/v1/njp/self", dependencies=auth)
    def njp_self() -> dict:
        """The measured capability map: what she is reliably good at, and what she is not."""
        return _organ("self_model", "SELF_MODEL_ENABLED")

    @app.get("/v1/njp/meta", dependencies=auth)
    def njp_meta() -> dict:
        """Her own strategies as bandit arms, and where compute is currently allocated."""
        return _organ("meta", "META_ENABLED")

    @app.get("/v1/njp/goals", dependencies=auth)
    def njp_goals() -> dict:
        """Mission → Goal → Subgoal → Task, with rolled-up progress and what is blocked."""
        return _organ("goals", "GOALS_ENABLED")

    @app.get("/v1/njp/curiosity", dependencies=auth)
    def njp_curiosity() -> dict:
        """Her own open questions, priced by expected value of information."""
        return _organ("curiosity", "CURIOSITY_ENABLED")

    @app.get("/v1/njp/shadow", dependencies=auth)
    def njp_shadow() -> dict:
        """What she does not know — gaps priced by information gain, and her weak faculties."""
        brain = _brain()
        if brain is None:
            return _off("NJP_ENABLED")
        try:
            return brain.shadow()
        except Exception:  # noqa: BLE001
            return {"error": "shadow failed"}

    @app.get("/v1/njp/attention", dependencies=auth)
    def njp_attention() -> dict:
        """What has been winning her attention, and what has been losing."""
        return _organ("attention", "ATTENTION_ENABLED")

    @app.get("/v1/njp/predict", dependencies=auth)
    def njp_predict() -> dict:
        """Prediction accuracy, and errors broken down by which organ they were blamed on."""
        return _organ("predictor", "PREDICT_ENABLED")

    @app.post("/v1/njp/prove", dependencies=auth)
    def njp_prove(req: NyxThinkRequest) -> dict:
        """Machine-check a claim, where the claim admits one.

        Returns INEXPRESSIBLE — and no verdict at all — for anything that is not a formula.
        That refusal is the point of the route: a proof runs on formally expressible claims, and
        "gravity pulls the apple down" is not one."""
        brain = _brain()
        if brain is None:
            return _off()
        try:
            from nyxara.njp.prove import ProofCore
            cert = ProofCore().prove(req.stimulus)
            return cert.to_dict() if hasattr(cert, "to_dict") else {"status": str(cert)}
        except Exception:  # noqa: BLE001 — no prover installed ⇒ say so, never fake a verdict
            return {"available": False, "reason": "the proof core is not available here"}

    @app.post("/v1/njp/intent", dependencies=auth)
    def njp_intent(req: NyxThinkRequest) -> dict:
        """What was actually asked — mood, ordering constraints, and what she would ask back."""
        brain = _brain()
        if brain is None or brain.intent is None:
            return _off("INTENT_ENABLED")
        got = brain.intent.read(req.stimulus)
        return got.to_dict() if hasattr(got, "to_dict") else {}

    @app.post("/v1/njp/truth", dependencies=auth)
    def njp_truth(req: NyxThinkRequest) -> dict:
        """Put a claim through the Truth-Seeking Gauntlet and see every source's reading."""
        brain = _brain()
        if brain is None or brain.truth is None:
            return _off("TRUTH_ENABLED")
        return brain.truth.judge(req.stimulus, context=brain).to_dict()

    @app.get("/v1/njp/{organ}", dependencies=auth)
    def njp_organ(organ: str) -> dict:
        """Read-only state for one organ. Never acts, never changes anything."""
        brain = _brain()
        if brain is None:
            return _off()
        # `learner` and `calculate` are the attribute names on the brain, not the display names:
        # the Core is `brain.learner` because `brain.core` is already the kernel back-reference,
        # and the calculator is `brain.calculator`. The allowlist is keyed by the URL segment and
        # `_ORGAN_ATTR` maps it to whatever the attribute is actually called.
        if organ not in _NJP_ORGANS:
            return {"available": False, "reason": f"no NJP organ named {organ!r}."}
        part = getattr(brain, _ORGAN_ATTR.get(organ, organ), None)
        if part is None:
            return _off(f"{organ.upper()}_ENABLED")
        return part.stats()

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
