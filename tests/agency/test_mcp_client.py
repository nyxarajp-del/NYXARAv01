"""Tests for the MCP client (agency/mcp_client.py).

Hermetic: each test launches a tiny in-process Python MCP "echo" server over real stdio —
no third-party SDK, no network — exercising the genuine handshake/JSON-RPC path."""

from __future__ import annotations

import sys
import textwrap

import pytest

from nyxara.agency.mcp_client import (MCPClient, MCPError, MCPServerConfig, MCPTool,
                                      connect_configured_mcp, register_mcp_tools)
from nyxara.agency.permissions import RiskTier
from nyxara.agency.tools import ToolRegistry

# A minimal MCP server: initialize -> tools/list (one 'echo' tool) -> tools/call.
_ECHO_SERVER = textwrap.dedent('''
    import json, sys
    def send(m): sys.stdout.write(json.dumps(m) + "\\n"); sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        msg = json.loads(line)
        mid, method = msg.get("id"), msg.get("method")
        if method == "initialize":
            send({"jsonrpc":"2.0","id":mid,"result":{"serverInfo":{"name":"echo","version":"1"}}})
        elif method == "tools/list":
            send({"jsonrpc":"2.0","id":mid,"result":{"tools":[{"name":"echo",
                 "description":"echo text back","inputSchema":{"type":"object",
                 "properties":{"text":{"type":"string"},"loud":{"type":"boolean"}},
                 "required":["text"]}}]}})
        elif method == "tools/call":
            args = msg["params"]["arguments"]
            t = args.get("text","")
            if args.get("loud"): t = t.upper()
            send({"jsonrpc":"2.0","id":mid,"result":{"content":[{"type":"text","text":"echo: "+t}]}})
        elif method == "boom":
            send({"jsonrpc":"2.0","id":mid,"error":{"code":-32000,"message":"kaboom"}})
''')


def _echo_cfg(name="echo"):
    return MCPServerConfig(name=name, command=sys.executable, args=["-c", _ECHO_SERVER])


@pytest.fixture
def client():
    c = MCPClient(_echo_cfg(), timeout_s=10.0).start()
    yield c
    c.close()


# --------------------------------------------------------------------------- #
# Handshake / discovery / invocation
# --------------------------------------------------------------------------- #
def test_handshake_sets_server_info(client):
    assert client.server_info.get("name") == "echo"


def test_list_tools(client):
    tools = client.list_tools()
    assert [t.name for t in tools] == ["echo"]
    assert "echo text back" in tools[0].description


def test_call_tool_returns_text(client):
    assert client.call_tool("echo", {"text": "hi"}) == "echo: hi"
    assert client.call_tool("echo", {"text": "hi", "loud": True}) == "echo: HI"


def test_server_error_raises_mcperror(client):
    with pytest.raises(MCPError):
        client._t.request("boom")


# --------------------------------------------------------------------------- #
# Schema translation
# --------------------------------------------------------------------------- #
def test_params_from_json_schema():
    tool = MCPTool(name="x", input_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}, "n": {"type": "integer"},
                       "flag": {"type": "boolean"}, "items": {"type": "array"}},
        "required": ["text"]})
    by_name = {p.name: p for p in tool.params()}
    assert by_name["text"].type == "str" and by_name["text"].required
    assert by_name["n"].type == "int" and not by_name["n"].required
    assert by_name["flag"].type == "bool"
    assert by_name["items"].type == "list"


# --------------------------------------------------------------------------- #
# Registration into the governed registry
# --------------------------------------------------------------------------- #
def test_register_into_registry_is_conservative(client):
    reg = ToolRegistry()
    names = register_mcp_tools(reg, client)
    assert names == ["mcp.echo.echo"]
    spec = reg.get("mcp.echo.echo")
    assert spec.risk is RiskTier.MODERATE      # unknown remote effect -> conservative
    assert spec.reversible is False            # -> escalates under mere autonomy
    assert spec.rate_resource == "mcp"


def test_registered_tool_is_callable(client):
    reg = ToolRegistry()
    register_mcp_tools(reg, client)
    spec = reg.get("mcp.echo.echo")
    assert spec.handler(text="ping") == "echo: ping"


def test_register_skips_duplicates(client):
    reg = ToolRegistry()
    assert register_mcp_tools(reg, client) == ["mcp.echo.echo"]
    assert register_mcp_tools(reg, client) == []  # already present -> skipped, not clobbered


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #
def test_bad_command_raises_on_start():
    c = MCPClient(MCPServerConfig(name="nope", command="this-command-does-not-exist-xyz"),
                  timeout_s=3.0)
    with pytest.raises(MCPError):
        c.start()


# --------------------------------------------------------------------------- #
# Config-driven connection
# --------------------------------------------------------------------------- #
def test_connect_configured_disabled_returns_empty():
    from nyxara.kernel.config import NyxaraSettings, Profile
    s = NyxaraSettings.for_profile(Profile.DEV)
    assert connect_configured_mcp(ToolRegistry(), s) == []


def test_connect_configured_registers_servers():
    from nyxara.kernel.config import MCPServerSpec, NyxaraSettings, Profile
    s = NyxaraSettings.for_profile(Profile.DEV)
    s.mcp.enabled = True
    s.mcp.servers = [MCPServerSpec(name="echo", command=sys.executable,
                                   args=["-c", _ECHO_SERVER])]
    reg = ToolRegistry()
    clients = connect_configured_mcp(reg, s)
    try:
        assert reg.get("mcp.echo.echo") is not None
        assert len(clients) == 1
    finally:
        for c in clients:
            c.close()


def test_connect_configured_skips_bad_server():
    from nyxara.kernel.config import MCPServerSpec, NyxaraSettings, Profile
    s = NyxaraSettings.for_profile(Profile.DEV)
    s.mcp.enabled = True
    s.mcp.servers = [
        MCPServerSpec(name="bad", command="this-command-does-not-exist-xyz"),
        MCPServerSpec(name="echo", command=sys.executable, args=["-c", _ECHO_SERVER]),
    ]
    reg = ToolRegistry()
    clients = connect_configured_mcp(reg, s)
    try:
        # the bad server is skipped; the good one still registers
        assert reg.get("mcp.echo.echo") is not None
        assert len(clients) == 1
    finally:
        for c in clients:
            c.close()
