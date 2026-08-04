"""MCP server exposing the Genie realtime voice API as Model Context Protocol tools.

Wraps the deployed realtime voice API (STT / TTS / voice-agent WebSocket routes
plus the metadata HTTP endpoints) so an MCP client — Cursor, Claude Desktop, or
any MCP host — can transcribe audio, synthesize speech, talk to the voice agent,
and read capabilities/benchmarks. See ``mcp_server.server`` for the entry point.
"""
from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
