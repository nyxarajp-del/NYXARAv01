"""Tests for NYXARA's self-initiated (LLM-free) network actions.

These lock in the "NYXARA khude kare" contract: her background/proactive mind ORIGINATES real
network actions herself — arbitrary HTTP requests and remote logins/commands — and each one
runs through the gated ``ToolRegistry`` under ``Authority.AUTONOMOUS`` (never a raw call, never
the LLM in the decision path). Two layers are covered:

* the helper methods ``_autonomous_http`` / ``_autonomous_remote_check`` dispatch the right
  gated tool with the right args (incl. the vault-backed credential path and the
  verify-login-only cases);
* end to end, a real ``NyxaraCore`` configured with ``watch_endpoints`` / ``remote_hosts``
  surfaces ``http:`` and ``remote:`` initiatives from its proactive engine, and executing them
  drives the gated tools (with the network backends faked so no socket is opened).
"""

from __future__ import annotations

import types

import pytest

from nyxara.agency.permissions import Authority
from nyxara.kernel.orchestrator import NyxaraCore


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class _Result:
    """Stand-in for ToolResult — only ``to_dict`` is exercised by the helpers."""

    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return dict(self._payload)


class _RecordingTools:
    """A minimal ToolRegistry stand-in that records every gated invoke."""

    def __init__(self, present, results=None):
        self.calls = []
        self._present = set(present)
        self._results = results or {}

    def get(self, name):
        return object() if name in self._present else None

    def invoke(self, name, args=None, *, authority=Authority.AUTONOMOUS,
               owner_confirmed=False, dry_run=False):
        self.calls.append({"name": name, "args": dict(args or {}), "authority": authority})
        return _Result(self._results.get(name, {"ok": True, "name": name}))


class _Self:
    """A lightweight object to invoke the unbound NyxaraCore helpers against."""

    def __init__(self, tools, goal_commands=True):
        self.tools = tools
        self._goal_commands = goal_commands

    def _autonomous_goal_commands_on(self):
        return self._goal_commands


def _spec(**kw):
    return types.SimpleNamespace(**kw)


# --------------------------------------------------------------------------- #
# _autonomous_http — arbitrary internet requests, through the gate
# --------------------------------------------------------------------------- #
def test_http_helper_invokes_gated_http_request():
    tools = _RecordingTools(present={"http_request"},
                            results={"http_request": {"ok": True, "status": 200}})
    spec = _spec(name="hb", url="https://example.com/ping", method="post",
                 body="{}", headers='{"X-A": "1"}', credential_name=None)
    out = NyxaraCore._autonomous_http(_Self(tools), spec)
    assert out == {"ok": True, "status": 200}
    assert len(tools.calls) == 1
    call = tools.calls[0]
    assert call["name"] == "http_request"
    assert call["authority"] is Authority.AUTONOMOUS
    assert call["args"] == {"url": "https://example.com/ping", "method": "POST",
                            "body": "{}", "headers": '{"X-A": "1"}'}


def test_http_helper_routes_credential_through_vault_tool():
    tools = _RecordingTools(present={"http_request", "credential_request"})
    spec = _spec(name="api", url="https://api.example.com/v1", method="get",
                 body="", headers="", credential_name="my_token")
    NyxaraCore._autonomous_http(_Self(tools), spec)
    assert tools.calls[0]["name"] == "credential_request"
    assert tools.calls[0]["args"] == {"name": "my_token", "url": "https://api.example.com/v1",
                                      "method": "GET"}


def test_http_helper_missing_tool_is_data_not_crash():
    out = NyxaraCore._autonomous_http(_Self(_RecordingTools(present=set())),
                                      _spec(url="https://example.com", method="GET",
                                            body="", headers="", credential_name=None))
    assert out["ok"] is False and "unavailable" in out["error"]


# --------------------------------------------------------------------------- #
# _autonomous_remote_check — remote logins & commands, through the gate
# --------------------------------------------------------------------------- #
def test_remote_helper_verifies_login_then_runs_health_command():
    tools = _RecordingTools(present={"ssh_login", "ssh_exec"})
    spec = _spec(name="box", host="203.0.113.10", health_command="df -h")
    out = NyxaraCore._autonomous_remote_check(_Self(tools), spec, goal="serve the Master")
    names = [c["name"] for c in tools.calls]
    assert names == ["ssh_login", "ssh_exec"]
    assert all(c["authority"] is Authority.AUTONOMOUS for c in tools.calls)
    exec_call = tools.calls[1]
    assert exec_call["args"] == {"credential_name": "box", "host": "203.0.113.10",
                                 "command": "df -h"}
    assert out["command"] == "df -h" and out["goal"] == "serve the Master"


def test_remote_helper_empty_health_command_is_login_only():
    tools = _RecordingTools(present={"ssh_login", "ssh_exec"})
    spec = _spec(name="box", host="203.0.113.10", health_command="")
    NyxaraCore._autonomous_remote_check(_Self(tools), spec)
    assert [c["name"] for c in tools.calls] == ["ssh_login"]


def test_remote_helper_default_probe_gated_by_flag():
    spec = _spec(name="box", host="203.0.113.10", health_command=None)
    # flag ON -> default 'uptime' probe runs
    on = _RecordingTools(present={"ssh_login", "ssh_exec"})
    NyxaraCore._autonomous_remote_check(_Self(on, goal_commands=True), spec)
    assert on.calls[1]["args"]["command"] == "uptime"
    # flag OFF -> login-verify only, no autonomous exec
    off = _RecordingTools(present={"ssh_login", "ssh_exec"})
    NyxaraCore._autonomous_remote_check(_Self(off, goal_commands=False), spec)
    assert [c["name"] for c in off.calls] == ["ssh_login"]


# --------------------------------------------------------------------------- #
# end to end — a real core self-surfaces and executes the actions
# --------------------------------------------------------------------------- #
@pytest.fixture
def _net_core(monkeypatch):
    """A real NyxaraCore configured with a watch endpoint + a remote host."""
    monkeypatch.setenv("NYXARA_ENV", "test")
    monkeypatch.setenv("NYXARA_AGENCY__WATCH_ENDPOINTS",
                       '[{"name": "hb", "url": "https://example.com/ping", "method": "GET"}]')
    monkeypatch.setenv("NYXARA_AGENCY__REMOTE_HOSTS",
                       '[{"name": "box", "host": "203.0.113.10", "username": "root",'
                       ' "password": "pw", "health_command": "uptime"}]')
    from nyxara.kernel import config as cfg
    cfg.reload_settings()
    try:
        yield NyxaraCore()
    finally:
        cfg.reload_settings()


def test_core_context_exposes_network_state(_net_core):
    ctx = _net_core.proactive_context()
    for key in ("autonomous_network", "autonomous_remote", "autonomous_internet",
                "watch_endpoints", "remote_hosts"):
        assert key in ctx
    assert ctx["autonomous_network"] is True
    assert len(ctx["watch_endpoints"]) == 1 and len(ctx["remote_hosts"]) == 1


def test_core_self_surfaces_http_and_remote_initiatives(_net_core):
    inis = _net_core.proactive.detect(_net_core.proactive_context())
    names = [i.name for i in inis]
    assert any(n.startswith("http:") for n in names), names
    assert any(n.startswith("remote:") for n in names), names


def test_detectors_dormant_when_autonomous_network_off(monkeypatch):
    monkeypatch.setenv("NYXARA_ENV", "test")
    monkeypatch.setenv("NYXARA_AGENCY__AUTONOMOUS_NETWORK", "false")
    monkeypatch.setenv("NYXARA_AGENCY__WATCH_ENDPOINTS",
                       '[{"name": "hb", "url": "https://example.com/ping"}]')
    monkeypatch.setenv("NYXARA_AGENCY__REMOTE_HOSTS",
                       '[{"name": "box", "host": "203.0.113.10"}]')
    from nyxara.kernel import config as cfg
    cfg.reload_settings()
    try:
        core = NyxaraCore()
        names = [i.name for i in core.proactive.detect(core.proactive_context())]
        assert not any(n.startswith("http:") for n in names)
        assert not any(n.startswith("remote:") for n in names)
    finally:
        cfg.reload_settings()


def test_executing_http_initiative_drives_gated_tool(_net_core, monkeypatch):
    recorded = {}

    def _fake_http(url, **kw):
        recorded["url"] = url
        recorded["kw"] = kw
        return {"ok": True, "status": 200, "headers": {}, "body": "pong", "error": None}

    # the http_request tool imports the backend at call time — patch the module attribute
    monkeypatch.setattr("nyxara.agency.net_request.http_request", _fake_http)
    inis = _net_core.proactive.detect(_net_core.proactive_context())
    http_ini = next(i for i in inis if i.name.startswith("http:"))
    result = http_ini.action()  # NYXARA does it herself, in code
    assert result["ok"] is True
    assert recorded["url"] == "https://example.com/ping"
    # the gated ToolResult is threaded back through the journalled initiative wrapper; the raw
    # HTTP response lives under its ``value``
    tool_result = result["result"]
    assert tool_result["ok"] is True
    assert tool_result["value"]["status"] == 200 and tool_result["value"]["body"] == "pong"


def test_executing_remote_initiative_drives_gated_ssh(_net_core, monkeypatch):
    from nyxara.agency import remote_exec

    calls = {}

    class _Chan:
        def recv_exit_status(self):
            return 0

    class _Stream:
        def __init__(self, data=b""):
            self._data = data
            self.channel = _Chan()

        def read(self, n=-1):
            return self._data

        def close(self):
            pass

    class _Client:
        def set_missing_host_key_policy(self, policy):
            pass

        def connect(self, **kw):
            calls["connect"] = kw

        def get_transport(self):
            return types.SimpleNamespace(remote_version="SSH-2.0-Fake")

        def exec_command(self, command, timeout=None):
            calls["command"] = command
            return _Stream(), _Stream(b"up 3 days\n"), _Stream(b"")

        def close(self):
            calls["closed"] = True

    fake = types.ModuleType("paramiko")
    fake.SSHClient = _Client
    fake.AutoAddPolicy = lambda: object()
    monkeypatch.setattr(remote_exec, "_load_paramiko", lambda: fake)

    inis = _net_core.proactive.detect(_net_core.proactive_context())
    remote_ini = next(i for i in inis if i.name.startswith("remote:"))
    result = remote_ini.action()  # NYXARA logs in and runs the command herself
    assert result["ok"] is True
    assert calls["command"] == "uptime"
    assert calls["connect"]["hostname"] == "203.0.113.10"
