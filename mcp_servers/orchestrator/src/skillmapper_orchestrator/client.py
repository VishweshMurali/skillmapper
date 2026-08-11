from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from skillmapper_orchestrator.server_specs import SERVER_SPECS


class OrchestratorToolError(RuntimeError):
    """A remote MCP tool call returned is_error=True -- surfaced clearly
    rather than letting a failed call feed None/garbage into the retry loop.
    """

    def __init__(self, server_name: str, tool_name: str, content: Any):
        super().__init__(f"{server_name}.{tool_name} returned an error: {content}")
        self.server_name = server_name
        self.tool_name = tool_name
        self.content = content


class OrchestratorClients:
    def __init__(self, sessions: dict[str, ClientSession]):
        self.sessions = sessions

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any], timeout: float | None = None
    ) -> dict[str, Any]:
        session = self.sessions[server_name]
        result = await session.call_tool(tool_name, arguments, read_timeout_seconds=timeout)
        if result.is_error:
            raise OrchestratorToolError(server_name, tool_name, result.content)
        return result.structured_content


@asynccontextmanager
async def connect_all() -> AsyncIterator[OrchestratorClients]:
    """Opens all 5 servers' ClientSessions once for the duration of one
    orchestrator run -- subprocess spawn + Chroma/BM25/Ollama-client init
    costs get paid once, not once per tool call (same reasoning as why
    AgentRuntime is built once per process today).
    """
    async with AsyncExitStack() as stack:
        sessions: dict[str, ClientSession] = {}
        for name, params in SERVER_SPECS.items():
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            sessions[name] = session
        yield OrchestratorClients(sessions)
