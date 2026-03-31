"""
Standard MCP Client for RAE-Core.

Provides a unified interface to call tools on MCP servers.
"""

import os
from typing import Any, cast

import structlog
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = structlog.get_logger(__name__)


class RAEMCPClient:
    """
    Standard MCP Client for RAE.
    Handles connection to MCP servers via STDIO.
    """

    def __init__(
        self,
        command: str = "python",
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ):
        self.command = command
        self.args = args or []
        self.env = env or os.environ.copy()
        self._session: ClientSession | None = None
        self._stdio_mgr: Any | None = None

    async def connect(self):
        """Establish connection to the MCP server."""
        if self._session:
            return

        logger.info("connecting_to_mcp_server", command=self.command, args=self.args)

        server_params = StdioServerParameters(
            command=self.command, args=self.args, env=self.env
        )

        # We use a simple session management here.
        # In a real app, you might want to use AsyncExitStack.
        self._stdio_mgr = stdio_client(server_params)
        read, write = await self._stdio_mgr.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.initialize()

        logger.info("mcp_server_connected")

    async def disconnect(self):
        """Close the connection."""
        if self._stdio_mgr:
            await self._stdio_mgr.__aexit__(None, None, None)
            self._session = None
            self._stdio_mgr = None
            logger.info("mcp_server_disconnected")

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool on the MCP server."""
        if not self._session:
            await self.connect()

        if not self._session:
            raise RuntimeError("Failed to initialize MCP session")

        try:
            result = await self._session.call_tool(name, arguments)
            # MCP result content is a list of content blocks
            # We assume it's text/json for now
            if hasattr(result, "content") and result.content:
                # Basic heuristic: if it looks like JSON, parse it
                import json

                content_block = result.content[0]
                if hasattr(content_block, "text"):
                    text = getattr(content_block, "text")
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, dict):
                            return cast(dict[str, Any], parsed)
                        return {"result": parsed}
                    except Exception:
                        return {"text": text}
            return {}
        except Exception as e:
            logger.error("mcp_tool_call_failed", tool=name, error=str(e))
            raise

    async def list_tools(self) -> list[Any]:
        """List available tools on the server."""
        if not self._session:
            await self.connect()

        if not self._session:
            return []

        result = await self._session.list_tools()
        return result.tools
