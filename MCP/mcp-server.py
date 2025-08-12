#!/usr/bin/env python3
import asyncio
import logging
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, Resource, Prompt
import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import quote_plus, urlparse
import time
import pytz

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create consolidated server
server = Server("GizmoConsolidated")

def clean_text(text):
    """Clean and format text"""
    if not text:
        return ""
    return ' '.join(text.split())

def safe_request(url, timeout=10):
    """Make a safe HTTP request"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {str(e)}")
        return None

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools in one consolidated server."""
    return [
        Tool(
            name="fetch_webpage",
            description="Fetch and extract text content from a specific webpage URL",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the webpage to fetch and extract content from"
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to return from the webpage (default: 2000)",
                        "default": 2000
                    }
                },
                "required": ["url"]
            }
        ),
        
        # File Operations
        Tool(
            name="read_file",
            description="Read contents of a local file",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding (default: utf-8)",
                        "default": "utf-8"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="write_file",
            description="Write content to a local file",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding (default: utf-8)",
                        "default": "utf-8"
                    }
                },
                "required": ["file_path", "content"]
            }
        ),
        
        # Time & Date
        Tool(
            name="get_current_time",
            description="Get current date and time information",
            inputSchema={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone (default: local)",
                        "default": "local"
                    },
                    "format": {
                        "type": "string",
                        "description": "Date format (default: ISO)",
                        "default": "iso"
                    }
                }
            }
        ),
        
        # Math & Calculator
        Tool(
            name="calculate",
            description="Perform mathematical calculations",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2 + 2 * 3')"
                    }
                },
                "required": ["expression"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle all tool calls in one place."""
    
    try:
        if name == "fetch_webpage":
            url = arguments.get("url", "")
            max_chars = arguments.get("max_chars", 2000)
            
            if not url:
                return [TextContent(type="text", text="Error: URL is required")]
            
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return [TextContent(type="text", text="Error: Invalid URL format")]
            
            response = safe_request(url)
            if not response:
                return [TextContent(type="text", text=f"Failed to fetch webpage: {url}")]
            
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            text_content = soup.get_text(separator=' ', strip=True)
            cleaned_content = clean_text(text_content)
            
            if len(cleaned_content) > max_chars:
                cleaned_content = cleaned_content[:max_chars] + "...[truncated]"
            
            result = f"📄 Content from: {url}\n" + "=" * 50 + "\n" + cleaned_content
            return [TextContent(type="text", text=result)]
            
        elif name == "read_file":
            file_path = arguments.get("file_path", "")
            encoding = arguments.get("encoding", "utf-8")
            
            if not file_path:
                return [TextContent(type="text", text="Error: File path is required")]
            
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                return [TextContent(type="text", text=f"📄 Contents of {file_path}:\n\n{content}")]
            except Exception as e:
                return [TextContent(type="text", text=f"Error reading file: {str(e)}")]
                
        elif name == "write_file":
            file_path = arguments.get("file_path", "")
            content = arguments.get("content", "")
            encoding = arguments.get("encoding", "utf-8")
            
            if not file_path or content is None:
                return [TextContent(type="text", text="Error: File path and content are required")]
            
            try:
                with open(file_path, 'w', encoding=encoding) as f:
                    f.write(content)
                return [TextContent(type="text", text=f"✅ Successfully wrote to {file_path}")]
            except Exception as e:
                return [TextContent(type="text", text=f"Error writing file: {str(e)}")]
                
        elif name == "get_current_time":
            import datetime
            import pytz
            
            timezone = arguments.get("timezone", "local")
            format_type = arguments.get("format", "iso")
            
            if timezone == "local":
                now = datetime.datetime.now()
            else:
                try:
                    tz = pytz.timezone(timezone)
                    now = datetime.datetime.now(tz)
                except:
                    now = datetime.datetime.now()
            
            if format_type == "iso":
                time_str = now.isoformat()
            else:
                time_str = now.strftime("%Y-%m-%d %H:%M:%S")
            
            return [TextContent(type="text", text=f"🕐 Current time: {time_str}")]
            
        elif name == "calculate":
            expression = arguments.get("expression", "")
            
            if not expression:
                return [TextContent(type="text", text="Error: Mathematical expression is required")]
            
            try:
                # Safe evaluation of mathematical expressions
                import ast
                import operator
                
                # Supported operations
                ops = {
                    ast.Add: operator.add,
                    ast.Sub: operator.sub,
                    ast.Mult: operator.mul,
                    ast.Div: operator.truediv,
                    ast.Pow: operator.pow,
                    ast.BitXor: operator.xor,
                    ast.USub: operator.neg,
                }
                
                def eval_expr(expr):
                    return eval_node(ast.parse(expr, mode='eval').body)
                
                def eval_node(node):
                    if isinstance(node, ast.Num):
                        return node.n
                    elif isinstance(node, ast.BinOp):
                        return ops[type(node.op)](eval_node(node.left), eval_node(node.right))
                    elif isinstance(node, ast.UnaryOp):
                        return ops[type(node.op)](eval_node(node.operand))
                    else:
                        raise TypeError(node)
                
                result = eval_expr(expression)
                return [TextContent(type="text", text=f"🧮 {expression} = {result}")]
                
            except Exception as e:
                return [TextContent(type="text", text=f"Error calculating: {str(e)}")]
        else:
            raise ValueError(f"Unknown tool: {name}")
            
    except Exception as e:
        error_msg = f"Tool '{name}' failed: {str(e)}"
        logger.error(error_msg)
        return [TextContent(type="text", text=error_msg)]

@server.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources (currently none)."""
    return []

@server.list_prompts()
async def list_prompts() -> list[Prompt]:
    """List available prompts (currently none)."""
    return []

async def main():
    """Run the consolidated MCP server."""
    logger.info("Starting Gizmo Consolidated MCP server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())