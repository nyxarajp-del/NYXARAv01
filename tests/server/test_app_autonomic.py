"""The always-on daemon path: the server can host the background mind (AutonomicLoop).

Hermetic — the core runs on the deterministic offline reasoner, and the app is driven
through FastAPI's in-process TestClient. We assert the lifespan wiring only (that the
loop is started and cleanly stopped), not the loop's cognitive behaviour."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from nyxara.kernel.config import NyxaraSettings, Profile  # noqa: E402
from nyxara.kernel.orchestrator import NyxaraCore  # noqa: E402
from nyxara.server.app import create_app  # noqa: E402


def _core() -> NyxaraCore:
    return NyxaraCore(enable_memory=False, enable_tools=False, enable_skills=False)


def _app(*, autonomic: bool):
    s = NyxaraSettings.for_profile(Profile.DEV)
    s.server.autonomic = autonomic
    s.server.autonomic_interval_s = 0.02   # tick fast so any wiring bug surfaces
    return create_app(core=_core(), settings=s)


def test_autonomic_off_by_default_leaves_no_background_loop() -> None:
    app = _app(autonomic=False)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert getattr(app.state, "autonomic", "missing") is None


def test_autonomic_on_starts_loop_on_boot_and_stops_on_shutdown() -> None:
    app = _app(autonomic=True)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        # started with the running server event loop, supervised for in-process auto-restart
        assert app.state.autonomic is not None
        assert app.state.autonomic.running is True
        assert app.state.autonomic_runtime is not None
    # lifespan shutdown must tear the background mind down cleanly
    assert app.state.autonomic is None
    assert app.state.autonomic_runtime is None


def test_autonomic_on_starts_deep_cognition_and_stops_on_shutdown() -> None:
    # the always-on daemon must also run the deep self-directed engines (default-mode stream +
    # idle_maintenance — dream/scientist/eureka/curiosity/growth — + civilization), not only the
    # narrow decide→act loop. These live behind core.start_cognition(), which the console starts
    # and the server previously did not.
    app = _app(autonomic=True)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert app.state.deep_cognition is True
        assert app.state.core._cognition_thread is not None
        assert app.state.core._cognition_thread.is_alive()
    # clean teardown stops the cognition thread
    assert app.state.deep_cognition is False
    assert app.state.core._cognition_thread is None


def test_autonomic_off_leaves_no_deep_cognition() -> None:
    app = _app(autonomic=False)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert getattr(app.state, "deep_cognition", False) is False
