#!/usr/bin/env python3
"""
Simple MCP Tools Discovery - Get tools, update skills.txt, and generate examples for new tools
Using full MCP Client SDK for robust server communication
"""

import json
import asyncio
import os
import sys
import warnings
import requests
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
devmode = False

def manager(message=None, pos_var=None):
    """
    If devmode is False, set the log level to None (no logs).
    If devmode is True, print the given message if it is not None.
    """
    if devmode == False:
        from yacana import LoggerManager
        LoggerManager.set_log_level(None)
    else:
        if message is not None:
            print(message + pos_var)

# Add the parent directory to Python path to resolve model module import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.modelbuilder import build

# Import MCP SDK components
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp import types
from pydantic import AnyUrl

# Suppress asyncio ResourceWarnings on Windows
warnings.filterwarnings("ignore", category=ResourceWarning)

class MCPServerManager:
    """Manages MCP server connections and tool discovery using the full SDK"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.devmode = False
    
    def _load_config(self) -> Dict[str, Any]:
        """Load MCP configuration from mcp.json"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"mcp.json not found at {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def _log(self, message: str, error: bool = False):
        """Log messages if in dev mode"""
        if self.devmode:
            file = sys.stderr if error else sys.stdout
            manager(message, file=file)
    
    async def discover_all_tools(self) -> Dict[str, Any]:
        """Discover tools from all configured MCP servers"""
        servers = self.config.get("mcpServers", {})
        all_tools = {}
        
        for server_name, server_config in servers.items():
            try:
                self._log(f"Discovering tools for server: {server_name}")
                tools = await self._discover_server_tools(server_name, server_config)
                all_tools[server_name] = tools
                
                if 'tools' in tools:
                    self._log(f"Found {len(tools['tools'])} tools in {server_name}")
                elif 'error' in tools:
                    self._log(f"Error in {server_name}: {tools['error']}", error=True)
                    
            except Exception as e:
                self._log(f"ERROR discovering {server_name}: {e}", error=True)
                all_tools[server_name] = {"error": str(e)}
        
        return all_tools
    
    async def _discover_server_tools(self, server_name: str, server_config: Dict[str, Any]) -> Dict[str, Any]:
        """Discover tools from a single MCP server using the SDK"""
        try:
            # Determine server type and create appropriate client
            if server_config.get("command"):
                return await self._discover_stdio_server(server_name, server_config)
            elif server_config.get("url"):
                return await self._discover_http_server(server_name, server_config)
            else:
                return {"error": "Server config must have either 'command' or 'url'"}
                
        except Exception as e:
            return {"error": f"Failed to discover tools: {str(e)}"}
    
    async def _discover_stdio_server(self, server_name: str, server_config: Dict[str, Any]) -> Dict[str, Any]:
        """Discover tools from a stdio MCP server"""
        try:
            # Prepare environment
            env = os.environ.copy()
            if 'env' in server_config:
                env.update(server_config['env'])
            
            # Handle special cases for different servers
            env = self._prepare_server_environment(server_name, server_config, env)
            
            # Auto-add -y flag for npx commands
            args = server_config.get('args', [])
            if server_config['command'] == 'npx' and '-y' not in args:
                args = ['-y'] + args
            
            # Create server parameters
            server_params = StdioServerParameters(
                command=server_config['command'],
                args=args,
                env=env
            )
            
            # Connect and discover tools
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize the connection
                    await session.initialize()
                    
                    # Get all available information
                    tools_result = {"tools": [], "resources": [], "prompts": []}
                    
                    # List tools
                    try:
                        tools_response = await session.list_tools()
                        tools_result["tools"] = [
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "inputSchema": tool.inputSchema
                            }
                            for tool in tools_response.tools
                        ]
                    except Exception as e:
                        self._log(f"Error listing tools for {server_name}: {e}", error=True)
                    
                    # List resources
                    try:
                        resources_response = await session.list_resources()
                        tools_result["resources"] = [
                            {
                                "uri": str(resource.uri),
                                "name": resource.name,
                                "description": resource.description,
                                "mimeType": resource.mimeType
                            }
                            for resource in resources_response.resources
                        ]
                    except Exception as e:
                        self._log(f"Error listing resources for {server_name}: {e}", error=True)
                    
                    # List prompts
                    try:
                        prompts_response = await session.list_prompts()
                        tools_result["prompts"] = [
                            {
                                "name": prompt.name,
                                "description": prompt.description,
                                "arguments": [
                                    {
                                        "name": arg.name,
                                        "description": arg.description,
                                        "required": arg.required
                                    }
                                    for arg in (prompt.arguments or [])
                                ]
                            }
                            for prompt in prompts_response.prompts
                        ]
                    except Exception as e:
                        self._log(f"Error listing prompts for {server_name}: {e}", error=True)
                    
                    return tools_result
                    
        except Exception as e:
            return {"error": f"STDIO connection failed: {str(e)}"}
    
    async def _discover_http_server(self, server_name: str, server_config: Dict[str, Any]) -> Dict[str, Any]:
        """Discover tools from an HTTP MCP server"""
        try:
            server_url = server_config["url"]
            
            # Handle authentication if configured
            auth = None
            if "auth" in server_config:
                # This would need to be implemented based on the auth config
                # For now, we'll skip auth and just try basic connection
                pass
            
            async with streamablehttp_client(server_url, auth=auth) as (read, write, _):
                async with ClientSession(read, write) as session:
                    # Initialize the connection
                    await session.initialize()
                    
                    # Get all available information
                    tools_result = {"tools": [], "resources": [], "prompts": []}
                    
                    # List tools
                    try:
                        tools_response = await session.list_tools()
                        tools_result["tools"] = [
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "inputSchema": tool.inputSchema
                            }
                            for tool in tools_response.tools
                        ]
                    except Exception as e:
                        self._log(f"Error listing tools for {server_name}: {e}", error=True)
                    
                    # List resources
                    try:
                        resources_response = await session.list_resources()
                        tools_result["resources"] = [
                            {
                                "uri": str(resource.uri),
                                "name": resource.name,
                                "description": resource.description,
                                "mimeType": resource.mimeType
                            }
                            for resource in resources_response.resources
                        ]
                    except Exception as e:
                        self._log(f"Error listing resources for {server_name}: {e}", error=True)
                    
                    # List prompts
                    try:
                        prompts_response = await session.list_prompts()
                        tools_result["prompts"] = [
                            {
                                "name": prompt.name,
                                "description": prompt.description,
                                "arguments": [
                                    {
                                        "name": arg.name,
                                        "description": arg.description,
                                        "required": arg.required
                                    }
                                    for arg in (prompt.arguments or [])
                                ]
                            }
                            for prompt in prompts_response.prompts
                        ]
                    except Exception as e:
                        self._log(f"Error listing prompts for {server_name}: {e}", error=True)
                    
                    return tools_result
                    
        except Exception as e:
            return {"error": f"HTTP connection failed: {str(e)}"}
    
    def _prepare_server_environment(self, server_name: str, server_config: Dict[str, Any], env: Dict[str, str]) -> Dict[str, str]:
        """Prepare environment variables for specific servers"""
        
        # Special handling for cli-mcp-server
        if server_name == 'cli-mcp-server':
            if 'ALLOWED_DIR' not in env:
                env['ALLOWED_DIR'] = os.getcwd()
            else:
                # Convert Unix-style paths to Windows paths on Windows
                allowed_dir = env['ALLOWED_DIR']
                if sys.platform == 'win32':
                    if allowed_dir.startswith('/c/'):
                        env['ALLOWED_DIR'] = allowed_dir.replace('/c/', 'C:\\', 1).replace('/', '\\')
                    elif allowed_dir.startswith('/Users/'):
                        env['ALLOWED_DIR'] = allowed_dir.replace('/Users/', 'C:\\Users\\', 1).replace('/', '\\')
                    elif allowed_dir.startswith('/home/'):
                        env['ALLOWED_DIR'] = allowed_dir.replace('/home/', 'C:\\Users\\', 1).replace('/', '\\')
            self._log(f"CLI MCP Server env: ALLOWED_DIR={env.get('ALLOWED_DIR')}")
        
        # Add more server-specific environment handling here as needed
        
        return env


def get_existing_tools(skills_file: str) -> set:
    """Parse existing tools from skills.txt"""
    if not os.path.exists(skills_file):
        return set()
    
    with open(skills_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    existing_tools = set()
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('- `') and '`' in line[3:]:
            tool_name = line.split('`')[1]
            existing_tools.add(tool_name)
    
    return existing_tools


def update_skills_file(skills_file: str, all_tools: Dict[str, Any], new_tool_examples: str):
    """Update skills.txt with new tools and examples"""
    if not os.path.exists(skills_file):
        manager(f"ERROR: {skills_file} not found", file=sys.stderr)
        return
    
    with open(skills_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Find existing tools in Available MCP Tools section
    existing_tools = get_existing_tools(skills_file)
    
    # Collect all new tools
    new_tools_list = []
    for server_name, server_info in all_tools.items():
        if 'tools' in server_info:
            for tool in server_info['tools']:
                tool_name = tool.get('name', 'unnamed')
                if tool_name not in existing_tools:
                    new_tools_list.append(f"- `{tool_name}`")
    
    if not new_tools_list:
        manager("No new tools to add to skills.txt")
        return
    
    # Find where to insert new tools (after Available MCP Tools:)
    tools_section_idx = -1
    for i, line in enumerate(lines):
        if "Available MCP Tools:" in line:
            tools_section_idx = i
            break
    
    if tools_section_idx == -1:
        manager("ERROR: Could not find 'Available MCP Tools:' section", file=sys.stderr)
        return
    
    # Find where to insert (after last existing tool)
    insert_idx = tools_section_idx + 1
    while insert_idx < len(lines) and (lines[insert_idx].strip().startswith('-') or lines[insert_idx].strip() == ''):
        if lines[insert_idx].strip().startswith('-'):
            insert_idx += 1
        else:
            break
    
    # Insert new tools
    for tool_line in reversed(new_tools_list):
        lines.insert(insert_idx, tool_line)
    
    # Find Example Usage Patterns section and insert examples at the top
    examples_idx = -1
    for i, line in enumerate(lines):
        if "Example Usage Patterns:" in line:
            examples_idx = i
            break
    
    if examples_idx == -1:
        manager("ERROR: Could not find 'Example Usage Patterns:' section", file=sys.stderr)
        return
    
    # Insert new examples right after "Example Usage Patterns:" line
    # Add blank line first, then examples, then another blank line
    lines.insert(examples_idx + 1, '')
    
    example_lines = new_tool_examples.strip().split('\n')
    for i, example_line in enumerate(example_lines):
        lines.insert(examples_idx + 2 + i, example_line)
    
    # Add blank line after examples
    lines.insert(examples_idx + 2 + len(example_lines), '')
    
    # Write updated content
    with open(skills_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    manager(f"✅ Updated {skills_file} with {len(new_tools_list)} new tools and examples")
    print(f"Added {len(new_tools_list)} new tools.")


def query_ai(prompt_text: str) -> str:
    """Query Hack Club AI API"""
    try:
        response = requests.post(
            "https://ai.hackclub.com/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "messages": [{"role": "user", "content": prompt_text}],
                "model": "meta-llama/llama-4-maverick-17b-128e-instruct"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"API Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Request Error: {str(e)}"


async def main(devmode):
    """Main function"""
    # Get paths
    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(script_dir)
    mcp_config_path = os.path.join(parent_dir, "mcp.json")
    skills_file = os.path.join(parent_dir, "model", "skills.txt")
    
    # Create server manager
    try:
        mcp_manager = MCPServerManager(mcp_config_path)
        # mcp_manager.devmode = True  # Enable for debugging
        
        # Discover all tools
        print("Looking for MCP tools...")
        tools_data = await mcp_manager.discover_all_tools()
        
        # Get existing tools from skills.txt
        existing_tools = get_existing_tools(skills_file)
        
        # Find new tools only
        new_tools = {}
        total_new_count = 0
        
        for server_name, server_info in tools_data.items():
            if 'tools' in server_info:
                new_server_tools = []
                for tool in server_info['tools']:
                    tool_name = tool.get('name', 'unnamed')
                    if tool_name not in existing_tools:
                        new_server_tools.append(tool)
                        total_new_count += 1
                
                if new_server_tools:
                    new_tools[server_name] = {"tools": new_server_tools}
        
        if not new_tools:
            print("No new tools found. Skills file is up to date.")
            return
        
        manager(f"Found {total_new_count} new tools across {len(new_tools)} servers")
        
        # Create prompt for AI with only new tools
        prompt = """You will be given a JSON object containing a list of tools with their names, descriptions, and inputSchemas.
Your task:
1. For each tool, create an example interaction consisting of:
   - A "User:" line with a realistic example request based on the tool's purpose.
   - A "Gizmo:" line with a short natural reply that sounds like a helpful assistant.
   - A tool call in the format: ⚡️<tool_name>({...arguments...})
     - Arguments must match the tool's inputSchema.
     - Fill required fields with realistic sample values.
     - Include default fields if they appear in the schema.
2. Output format:
   - No titles, no explanations.
   - Each example in the form:
     ```
     User: "<example request>"
     Gizmo: <assistant reply>
     ⚡️<tool_name>({<json args>})
     ```
   - Separate each example with a blank line.
   - No extra commentary or text outside of the examples.

Here are the NEW tools:
""" + json.dumps(new_tools, indent=2)
        
        # Query AI for examples
        manager("Generating examples for new tools...")
        ai_response = query_ai(prompt)
        
        # Update skills.txt
        update_skills_file(skills_file, tools_data, ai_response)
        
        manager("\n" + "="*60)
        manager("GENERATED EXAMPLES FOR NEW TOOLS:")
        manager("="*60)
        manager(ai_response)
        
        # Rebuild model
        manager("\nRebuilding model...")
        build(
            os.path.join(parent_dir, "model", "system.txt"),
            os.path.join(parent_dir, "model", "skills.txt"),
            os.path.join(parent_dir, "model", "Modelfile"),
            "gizmo",
            "wizardlm2:7b"
        )
        print("Model rebuild complete")
                
    except Exception as e:
        manager(f"Error: {e}", file=sys.stderr)
        return 1


def serverupdate(devmode=False):
    """Entry point for the server update function"""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(main(devmode))
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)


if __name__ == "__main__":
    serverupdate()