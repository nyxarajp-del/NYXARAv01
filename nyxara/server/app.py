"""NYXARA · server/app.py — the FastAPI application over the sovereign loop.

Routes (all of ``/v1`` require the bearer token when one is configured):

* ``GET  /health``               — liveness, unauthenticated.
* ``GET  /v1/report``            — a calibrated status report.
* ``POST /v1/chat``              — one turn: ``{message}`` → the disposed response.
* ``POST /v1/agent``             — a multi-step gated goal: ``{goal, max_steps?}``.
* ``POST /v1/research``          — one autonomous research pass: ``{topic}``.
* ``POST /v1/investigate``       — the scientist loop: ``{question}`` → hypothesis/conclusion.
* ``POST /v1/discover``          — the autonomous discovery loop: ``{cycles?}`` → belief updates.
* ``POST /v1/control/{action}``  — sovereign control: pause / resume / scram (opt-in).
* ``POST /v1/memory/save|load``  — persist / restore long-term memory (Rule 7 continuity).
* ``WS   /v1/ws``                — a streaming chat socket (token via ``?token=``).

The control law is untouched: each request runs ``NyxaraCore.process`` end to end, so an
over-eager request is refused or escalated exactly as it would be at the console. Auth is a
single bearer token — the Master's credential, this being a single-Master system.
"""

from typing import Any, Optional

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


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=1)


class InvestigateRequest(BaseModel):
    question: str = Field(..., min_length=1)


class DiscoverRequest(BaseModel):
    # How many self-driven discovery cycles to run; the server keeps this bounded.
    cycles: int = Field(default=3, ge=1, le=50)


class ControlRequest(BaseModel):
    reason: str = ""


class MemoryPathRequest(BaseModel):
    path: Optional[str] = None


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

    app = FastAPI(title="NYXARA", version=settings.schema_version,
                  description="Sovereign cognitive architecture — authenticated API.")

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

    @app.post("/v1/chat", dependencies=auth)
    def chat(req: ChatRequest) -> dict:
        result = core.process(req.message, authority=_authority(req.authority))
        return result.to_dict()

    @app.post("/v1/agent", dependencies=auth)
    def agent(req: AgentRequest) -> dict:
        steps = min(req.max_steps or cfg.max_agent_steps, cfg.max_agent_steps)
        run = core.agent(req.goal, authority=_authority(req.authority), max_steps=steps)
        return run.to_dict()

    @app.post("/v1/research", dependencies=auth)
    def research(req: ResearchRequest) -> dict:
        return core.research(req.topic)

    @app.post("/v1/investigate", dependencies=auth)
    def investigate(req: InvestigateRequest) -> dict:
        return core.investigate(req.question)

    @app.post("/v1/discover", dependencies=auth)
    def discover(req: DiscoverRequest) -> dict:
        return core.discover(req.cycles)

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
