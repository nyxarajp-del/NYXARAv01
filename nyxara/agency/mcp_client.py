"""NYXARA · agency/mcp_client.py — a Model Context Protocol (MCP) client (🔌 → 👑).

MCP is the open standard for connecting an AI to external tools and data: an MCP *server*
exposes a set of tools over JSON-RPC; a *client* discovers and calls them. This module makes
NYXARA an MCP client, so the entire MCP ecosystem (filesystem, git, databases, browsers,
SaaS connectors, …) becomes available to her — **through the existing governed pipeline**.
A remote MCP tool is registered as an ordinary :class:`~nyxara.agency.tools.ToolSpec`, so it
clears the same capability / risk / authority / sandbox / governor gates as any native tool.
The mind proposes the call; the kernel disposes of it. The network is a longer arm, not a
loophole.

Design rules:

* **Stdlib only.** A minimal but real JSON-RPC-2.0-over-stdio transport (newline-delimited
  messages, the MCP stdio framing) built on ``subprocess`` — no third-party MCP SDK needed,
  matching NYXARA's "works on a bare machine" philosophy.
* **Conservative by default.** Remote tools are of unknown effect, so they register at
  ``MODERATE`` risk and ``reversible=False`` — under mere autonomy they *escalate to the
  Master* rather than auto-run. The owner can confirm, or tune the risk per server.
* **Fail-soft.** A server that won't start, a tool that errors, or a slow call never crashes
  the loop: it raises an :class:`MCPError` the caller turns into a graceful refusal/observation.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nyxara.agency.permissions import Capability, RiskTier
from nyxara.agency.tools import ToolParam, ToolRegistry, ToolSpec

__all__ = [
    "MCPError", "MCPServerConfig", "MCPTool", "MCPClient", "register_mcp_tools",
    "connect_configured_mcp",
]

PROTOCOL_VERSION = "2024-11-05"

_RISK = {
    "trivial": RiskTier.TRIVIAL, "low": RiskTier.LOW, "moderate": RiskTier.MODERATE,
    "high": RiskTier.HIGH, "critical": RiskTier.CRITICAL,
}

# JSON-schema type -> NYXARA ToolParam type token (see tools._TYPE_TOKENS).
_JSON_TYPE = {
    "string": "str", "integer": "int", "number": "float", "boolean": "bool",
    "array": "list", "object": "dict",
}


class MCPError(RuntimeError):
    """An MCP transport, handshake, or tool-call failure."""


# --------------------------------------------------------------------------- #
# Config & tool description
# --------------------------------------------------------------------------- #
@dataclass
class MCPServerConfig:
    """How to launch/identify one MCP server (stdio transport)."""

    name: str                                   # a short id used to namespace its tools
    command: str                                # executable, e.g. "npx" or "python"
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)   # extra env (merged over os.environ)
    cwd: Optional[str] = None


@dataclass
class MCPTool:
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)

    def params(self) -> List[ToolParam]:
        """Translate the tool's JSON-schema into NYXARA ToolParams."""
        schema = self.input_schema or {}
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        required = set(schema.get("required", []) or [])
        out: List[ToolParam] = []
        for pname, pschema in (props.items() if isinstance(props, dict) else []):
            jtype = (pschema or {}).get("type") if isinstance(pschema, dict) else None
            token = _JSON_TYPE.get(jtype, "any")
            out.append(ToolParam(
                name=pname, type=token, required=pname in required,
                description=(pschema or {}).get("description", "") if isinstance(pschema, dict) else ""))
        return out


# --------------------------------------------------------------------------- #
# Stdio JSON-RPC transport
# --------------------------------------------------------------------------- #
class _StdioTransport:
    """Newline-delimited JSON-RPC 2.0 over a subprocess's stdin/stdout (MCP stdio)."""

    def __init__(self, cfg: MCPServerConfig, *, timeout_s: float) -> None:
        self.cfg = cfg
        self.timeout_s = timeout_s
        self._proc: Optional[subprocess.Popen] = None
        self._inbox: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self._next_id = 0

    def start(self) -> None:
        env = dict(os.environ)
        env.update(self.cfg.env or {})
        try:
            self._proc = subprocess.Popen(
                [self.cfg.command, *self.cfg.args],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                cwd=self.cfg.cwd, env=env, text=True, bufsize=1)
        except (OSError, ValueError) as exc:
            raise MCPError(f"could not launch MCP server {self.cfg.name!r}: {exc}") from exc
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._inbox.put(json.loads(line))
            except json.JSONDecodeError:
                continue  # server log noise on stdout — ignore non-JSON lines

    def _send(self, msg: Dict[str, Any]) -> None:
        if not self._proc or self._proc.stdin is None or self._proc.poll() is not None:
            raise MCPError(f"MCP server {self.cfg.name!r} is not running")
        try:
            self._proc.stdin.write(json.dumps(msg) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise MCPError(f"MCP server {self.cfg.name!r} pipe closed: {exc}") from exc

    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        self._next_id += 1
        rid = self._next_id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        deadline = time.monotonic() + self.timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MCPError(f"timeout waiting for {method!r} from {self.cfg.name!r}")
            try:
                msg = self._inbox.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                if self._proc and self._proc.poll() is not None:
                    raise MCPError(f"MCP server {self.cfg.name!r} exited during {method!r}")
                continue
            if msg.get("id") != rid:
                continue  # a notification or another response — skip
            if "error" in msg:
                err = msg["error"]
                raise MCPError(f"{method} failed: {err.get('message', err)}")
            return msg.get("result")

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        except OSError:
            pass
        finally:
            self._proc = None


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class MCPClient:
    """Connect to one MCP server, discover its tools, and call them."""

    def __init__(self, cfg: MCPServerConfig, *, timeout_s: float = 30.0) -> None:
        self.cfg = cfg
        self.timeout_s = timeout_s
        self._t = _StdioTransport(cfg, timeout_s=timeout_s)
        self._started = False
        self.server_info: Dict[str, Any] = {}

    # ---- lifecycle ---- #
    def start(self) -> "MCPClient":
        """Launch the server and perform the MCP initialize handshake."""
        self._t.start()
        result = self._t.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "nyxara", "version": "0.1.0"},
        })
        self.server_info = (result or {}).get("serverInfo", {})
        # Per spec, the client confirms it is ready before issuing further requests.
        self._t.notify("notifications/initialized")
        self._started = True
        return self

    def close(self) -> None:
        self._t.close()
        self._started = False

    def __enter__(self) -> "MCPClient":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- discovery & invocation ---- #
    def list_tools(self) -> List[MCPTool]:
        result = self._t.request("tools/list") or {}
        tools = result.get("tools", []) if isinstance(result, dict) else []
        out: List[MCPTool] = []
        for t in tools:
            if not isinstance(t, dict) or "name" not in t:
                continue
            out.append(MCPTool(name=t["name"], description=t.get("description", ""),
                               input_schema=t.get("inputSchema", {}) or {}))
        return out

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """Call a remote tool; return its text content (or the raw result)."""
        result = self._t.request("tools/call", {"name": name, "arguments": arguments or {}})
        if not isinstance(result, dict):
            return result
        if result.get("isError"):
            raise MCPError(f"tool {name!r} reported an error: {_content_text(result)}")
        text = _content_text(result)
        return text if text else result


def _content_text(result: Dict[str, Any]) -> str:
    """Flatten MCP content blocks (``[{type:'text', text:...}, …]``) into a string."""
    parts: List[str] = []
    for block in result.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(parts).strip()


# --------------------------------------------------------------------------- #
# Registration into the governed ToolRegistry
# --------------------------------------------------------------------------- #
def register_mcp_tools(registry: ToolRegistry, client: MCPClient, *,
                       capability: Capability = Capability.TOOL_CALL,
                       risk: RiskTier = RiskTier.MODERATE,
                       reversible: bool = False,
                       cost: float = 0.0) -> List[str]:
    """Discover ``client``'s tools and register each as a governed NYXARA tool.

    Returns the namespaced names registered. Remote effects are unknown, so the defaults are
    conservative (MODERATE / irreversible): an autonomous call escalates to the Master, who
    can approve it. A tool whose name already exists is skipped (never clobbered).
    """
    registered: List[str] = []
    for tool in client.list_tools():
        local_name = f"mcp.{client.cfg.name}.{tool.name}"
        if registry.get(local_name) is not None:
            continue

        def _handler(_remote=tool.name, **kwargs: Any) -> Any:
            return client.call_tool(_remote, kwargs)

        desc = tool.description or f"MCP tool {tool.name} from server {client.cfg.name!r}"
        registry.register(ToolSpec(
            name=local_name, handler=_handler, description=desc, params=tool.params(),
            capability=capability, risk=risk, reversible=reversible, cost=cost,
            rate_resource="mcp"))
        registered.append(local_name)
    return registered


def connect_configured_mcp(registry: ToolRegistry, settings: Any) -> List[MCPClient]:
    """Connect every server in ``settings.mcp.servers`` and register its tools.

    Returns the live clients (kept alive by the caller so their tools stay callable). Any
    single server that fails to start or hand-shake is skipped — one bad server never blocks
    the others, and never crashes the boot.
    """
    cfg = getattr(settings, "mcp", None)
    if cfg is None or not getattr(cfg, "enabled", False) or not getattr(cfg, "servers", None):
        return []
    clients: List[MCPClient] = []
    for spec in cfg.servers:
        client = MCPClient(
            MCPServerConfig(name=spec.name, command=spec.command, args=list(spec.args),
                            env=dict(spec.env), cwd=spec.cwd),
            timeout_s=cfg.timeout_s)
        try:
            client.start()
            register_mcp_tools(registry, client, risk=_RISK.get(spec.risk, RiskTier.MODERATE),
                               reversible=spec.reversible)
            clients.append(client)
        except Exception:  # noqa: BLE001 — a bad server is skipped, never fatal
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
    return clients


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import sys
    import textwrap

    # A tiny in-process MCP echo server so the demo is hermetic (no external deps).
    server_src = textwrap.dedent('''
        import json, sys
        def send(m): sys.stdout.write(json.dumps(m) + "\\n"); sys.stdout.flush()
        for line in sys.stdin:
            line = line.strip()
            if not line: continue
            msg = json.loads(line)
            mid, method = msg.get("id"), msg.get("method")
            if method == "initialize":
                send({"jsonrpc":"2.0","id":mid,"result":{"serverInfo":{"name":"echo"}}})
            elif method == "tools/list":
                send({"jsonrpc":"2.0","id":mid,"result":{"tools":[{"name":"echo",
                     "description":"echo text back","inputSchema":{"type":"object",
                     "properties":{"text":{"type":"string"}},"required":["text"]}}]}})
            elif method == "tools/call":
                t = msg["params"]["arguments"].get("text","")
                send({"jsonrpc":"2.0","id":mid,"result":{"content":[{"type":"text","text":"echo: "+t}]}})
    ''')

    print("=" * 70)
    print("NYXARA mcp-client self-test")
    print("=" * 70)
    cfg = MCPServerConfig(name="echo", command=sys.executable, args=["-c", server_src])
    with MCPClient(cfg, timeout_s=10.0) as client:
        print(f"server_info : {client.server_info}")
        tools = client.list_tools()
        print(f"tools       : {[t.name for t in tools]}")
        assert tools and tools[0].name == "echo"
        out = client.call_tool("echo", {"text": "hello MCP"})
        print(f"call result : {out!r}")
        assert out == "echo: hello MCP"

        reg = ToolRegistry()
        names = register_mcp_tools(reg, client)
        print(f"registered  : {names}")
        assert names == ["mcp.echo.echo"]
        spec = reg.get("mcp.echo.echo")
        assert spec is not None and spec.risk is RiskTier.MODERATE and spec.reversible is False
    print("\nALL SELF-TESTS PASSED ✓")
