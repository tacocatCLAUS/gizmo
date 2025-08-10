#!/usr/bin/env python3
import asyncio
import logging
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create server instance
server = Server("TestServer")

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="magicoutput",
            description="Use this function to get the magic output",
            inputSchema={
                "type": "object",
                "properties": {
                    "obj1": {
                        "type": "string",
                        "description": "First object parameter"
                    },
                    "obj2": {
                        "type": "string", 
                        "description": "Second object parameter"
                    }
                },
                "required": ["obj1", "obj2"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name == "magicoutput":
        obj1 = arguments.get("obj1", "")
        obj2 = arguments.get("obj2", "")
        logger.info(f"magicoutput called with obj1='{obj1}', obj2='{obj2}'")
        return [TextContent(type="text", text="WomboWombat")]
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    """Run the MCP server."""
    logger.info("Starting MCP server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
