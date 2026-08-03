"""Tests for agency/remote_exec.py — guarded SSH login & remote command execution.

Lock in the errors-as-data contract (never raises), fail-closed host vetting, and the
default-tools wiring (ssh_login / ssh_exec registered under REMOTE_EXEC, credential
resolution from remote_hosts, and http_request forwarding custom headers). Uses a fake
paramiko so no real network is touched.
"""

from __future__ import annotations

import types


from nyxara.agency import remote_exec
from nyxara.agency.permissions import Capability, RiskTier
from nyxara.agency.tools import ToolRegistry
from nyxara.agency.default_tools import build_default_tools


# -------------------- host vetting (fail-closed) -------------------- #
def test_vet_host_refuses_empty_and_malformed():
    assert remote_exec._vet_host("") == "missing host"
    assert "malformed" in remote_exec._vet_host("bad host")
    assert "malformed" in remote_exec._vet_host("user@host")


def test_vet_host_private_gate():
    # refused by default when allow_private is False
    assert "refused" in remote_exec._vet_host("127.0.0.1", allow_private=False)
    assert "refused" in remote_exec._vet_host("10.0.0.5", allow_private=False)
    # reachable when the Master opts in
    assert remote_exec._vet_host("127.0.0.1", allow_private=True) is None
    # a normal public host always passes
    assert remote_exec._vet_host("example.com") is None


# -------------------- errors-as-data, never raises -------------------- #
def test_ssh_exec_refuses_before_dialling():
    r = remote_exec.ssh_exec("", "echo hi")
    assert r["ok"] is False and r["error"] == "missing host"
    r = remote_exec.ssh_exec("example.com", "   ", allow_private=True)
    assert r["ok"] is False and r["error"] == "empty command"


def test_paramiko_absent_returns_data(monkeypatch):
    monkeypatch.setattr(remote_exec, "_load_paramiko", lambda: None)
    r = remote_exec.ssh_exec("example.com", "echo hi", allow_private=True)
    assert r["ok"] is False and "paramiko not installed" in r["error"]
    r2 = remote_exec.ssh_login("example.com", allow_private=True)
    assert r2["ok"] is False and "paramiko not installed" in r2["error"]


# -------------------- success path via a fake paramiko -------------------- #
def _fake_paramiko(*, exit_status=0, stdout=b"hello\n", stderr=b""):
    """Build a stand-in paramiko module whose client records the call and returns fixtures."""
    mod = types.ModuleType("paramiko")
    calls = {}

    class _Chan:
        def recv_exit_status(self):
            return exit_status

    class _Stream:
        def __init__(self, data):
            self._data = data
            self.channel = _Chan()

        def read(self, n=-1):
            return self._data

        def close(self):
            pass

    class _Transport:
        remote_version = "SSH-2.0-FakeServer"

    class _Client:
        def set_missing_host_key_policy(self, policy):
            pass

        def connect(self, **kw):
            calls["connect"] = kw

        def get_transport(self):
            return _Transport()

        def exec_command(self, command, timeout=None):
            calls["command"] = command
            return _Stream(b""), _Stream(stdout), _Stream(stderr)

        def close(self):
            calls["closed"] = True

    mod.SSHClient = _Client
    mod.AutoAddPolicy = lambda: object()
    mod._calls = calls
    return mod


def test_ssh_exec_success(monkeypatch):
    fake = _fake_paramiko(exit_status=0, stdout=b"ok\n")
    monkeypatch.setattr(remote_exec, "_load_paramiko", lambda: fake)
    r = remote_exec.ssh_exec("203.0.113.10", "echo ok", username="root",
                             password="pw", allow_private=True)
    assert r["ok"] is True and r["exit_status"] == 0 and r["stdout"] == "ok\n"
    assert fake._calls["command"] == "echo ok"
    assert fake._calls["connect"]["username"] == "root"
    assert fake._calls["connect"]["password"] == "pw"
    assert fake._calls.get("closed") is True


def test_ssh_exec_nonzero_exit_is_not_ok(monkeypatch):
    fake = _fake_paramiko(exit_status=2, stdout=b"", stderr=b"boom\n")
    monkeypatch.setattr(remote_exec, "_load_paramiko", lambda: fake)
    r = remote_exec.ssh_exec("203.0.113.10", "false", username="root",
                             password="pw", allow_private=True)
    assert r["ok"] is False and r["exit_status"] == 2 and r["stderr"] == "boom\n"


def test_ssh_login_success(monkeypatch):
    fake = _fake_paramiko()
    monkeypatch.setattr(remote_exec, "_load_paramiko", lambda: fake)
    r = remote_exec.ssh_login("203.0.113.10", username="root", password="pw",
                              allow_private=True)
    assert r["ok"] is True and r["banner"] == "SSH-2.0-FakeServer"
    assert fake._calls.get("closed") is True  # verify-only: connection is closed


def test_connect_failure_becomes_data(monkeypatch):
    fake = _fake_paramiko()

    def _boom(**kw):
        raise OSError("connection refused")

    fake.SSHClient.connect = lambda self, **kw: _boom(**kw)
    monkeypatch.setattr(remote_exec, "_load_paramiko", lambda: fake)
    r = remote_exec.ssh_exec("203.0.113.10", "echo hi", allow_private=True)
    assert r["ok"] is False and "connection refused" in r["error"]


# -------------------- default-tools wiring -------------------- #
def test_remote_tools_registered_with_capability():
    reg = ToolRegistry()
    build_default_tools(reg)
    login = reg.get("ssh_login")
    exec_ = reg.get("ssh_exec")
    assert login.capability is Capability.REMOTE_EXEC and login.reversible is True
    assert login.risk is RiskTier.MODERATE
    assert exec_.capability is Capability.REMOTE_EXEC and exec_.reversible is False
    assert exec_.risk is RiskTier.HIGH


def test_http_request_forwards_headers(monkeypatch):
    import nyxara.agency.net_request as nr
    captured = {}

    def fake(url, **kw):
        captured["url"] = url
        captured.update(kw)
        return {"ok": True, "status": 200, "headers": {}, "body": "", "error": None}

    monkeypatch.setattr(nr, "http_request", fake)
    reg = ToolRegistry()
    build_default_tools(reg)
    handler = reg.get("http_request").handler
    handler("https://api.example.com", "POST", '{"x":1}', '{"Authorization":"Bearer t"}')
    assert captured["headers"] == {"Authorization": "Bearer t"}
    # malformed headers fail as data, never raise
    r = handler("https://api.example.com", "GET", "", "not-json")
    assert r["ok"] is False and "invalid headers JSON" in r["error"]


def test_ssh_exec_tool_resolves_stored_credential(monkeypatch):
    # Configure a named remote credential and verify the tool resolves it.
    monkeypatch.setenv(
        "NYXARA_AGENCY__REMOTE_HOSTS",
        '[{"name":"gw","host":"203.0.113.10","port":2222,"username":"root","password":"sekret"}]',
    )
    from nyxara.kernel import config as cfg
    cfg.reload_settings()
    try:
        import nyxara.agency.remote_exec as re_mod
        seen = {}

        def fake_exec(host, command, **kw):
            seen["host"] = host
            seen["command"] = command
            seen.update(kw)
            return {"ok": True, "host": host, "exit_status": 0, "stdout": "", "stderr": "",
                    "error": None, "truncated": False}

        monkeypatch.setattr(re_mod, "ssh_exec", fake_exec)
        reg = ToolRegistry()
        build_default_tools(reg)
        handler = reg.get("ssh_exec").handler
        handler(command="uptime", credential_name="gw")
        assert seen["host"] == "203.0.113.10"
        assert seen["username"] == "root"
        assert seen["password"] == "sekret"
        assert seen["port"] == 2222
        assert seen["command"] == "uptime"
    finally:
        monkeypatch.delenv("NYXARA_AGENCY__REMOTE_HOSTS", raising=False)
        cfg.reload_settings()
