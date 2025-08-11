#!/usr/bin/env python3
"""
MCP Tools Discovery Script - Fixed Version
Reads mcp.json file and discovers all available MCP tools from configured servers
"""

import json
import subprocess
import asyncio
import sys
import os
from typing import Dict, List, Any
import logging
import requests
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MCPToolsDiscovery:
    def __init__(self, config_file: str = "mcp.json"):
        """Initialize with mcp.json config file"""
        self.config_file = config_file
        self.config = self.load_config()
        self.discovered_tools = {}

    def load_config(self) -> Dict:
        """Load configuration from mcp.json file"""
        try:
            if not os.path.exists(self.config_file):
                logger.error(f"Configuration file {self.config_file} not found!")
                return {"mcpServers": {}}
            
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                logger.info(f"✅ Loaded configuration from {self.config_file}")
                return config
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in {self.config_file}: {e}")
            return {"mcpServers": {}}
        except Exception as e:
            logger.error(f"❌ Error loading {self.config_file}: {e}")
            return {"mcpServers": {}}

    async def send_mcp_request(self, process, request: Dict, timeout: int = 10, expect_response: bool = True) -> Dict:
        """Send an MCP request and optionally get response with timeout"""
        try:
            request_str = json.dumps(request) + "\n"
            print(f"      📤 Sending: {request.get('method', 'unknown')} (id: {request.get('id', 'N/A')})")
            
            process.stdin.write(request_str.encode())
            await process.stdin.drain()
            
            if not expect_response:
                # For notifications, we don't expect a response
                print(f"      📝 Notification sent (no response expected)")
                return {"success": True}
            
            # Read response with timeout
            start_time = asyncio.get_event_loop().time()
            
            while True:
                try:
                    remaining_time = timeout - (asyncio.get_event_loop().time() - start_time)
                    if remaining_time <= 0:
                        return {"error": f"Timeout after {timeout} seconds"}
                    
                    response_line = await asyncio.wait_for(
                        process.stdout.readline(), 
                        timeout=min(remaining_time, 2.0)
                    )
                    
                    if not response_line:
                        return {"error": "No response received (stream ended)"}
                    
                    response_text = response_line.decode().strip()
                    if not response_text:
                        continue
                        
                    print(f"      📥 Received: {response_text[:100]}...")
                    
                    # Try to parse as JSON
                    try:
                        response = json.loads(response_text)
                        # If it's a valid JSON-RPC response, return it
                        if "jsonrpc" in response:
                            return response
                        else:
                            print(f"      ⚠️ Non JSON-RPC response: {response_text}")
                            continue
                    except json.JSONDecodeError as e:
                        print(f"      ⚠️ JSON decode error: {e}")
                        continue
                        
                except asyncio.TimeoutError:
                    return {"error": f"Timeout after {timeout} seconds"}
            
        except BrokenPipeError:
            return {"error": "Communication error: Connection lost (broken pipe)"}
        except ConnectionResetError:
            return {"error": "Communication error: Connection reset"}
        except Exception as e:
            return {"error": f"Communication error: {type(e).__name__}: {str(e)}"}

    async def discover_server_tools(self, server_name: str, server_config: Dict) -> Dict[str, Any]:
        """Discover tools from a single MCP server"""
        print(f"\n🔍 Discovering tools from server: {server_name}")
        
        # Auto-add -y flag for npx commands if not present
        if server_config['command'] == 'npx' and '-y' not in server_config.get('args', []):
            server_config['args'] = ['-y'] + server_config.get('args', [])
            print(f"   ⚙️ Added -y flag to npx command for auto-install")
        
        print(f"   Command: {server_config['command']} {' '.join(server_config.get('args', []))}")
        
        server_info = {
            "name": server_name,
            "command": server_config['command'],
            "args": server_config.get('args', []),
            "env": server_config.get('env', {}),
            "tools": [],
            "resources": [],
            "prompts": [],
            "status": "unknown",
            "error": None
        }
        
        process = None
        
        try:
            # Prepare environment
            env = os.environ.copy()
            
            # Ensure PATH includes common Node.js installation directories
            # This helps when subprocess doesn't inherit the full PATH
            if sys.platform == "win32":
                # Windows paths
                node_paths = [
                    r"C:\Program Files\nodejs",
                    r"C:\Program Files (x86)\nodejs",
                    os.path.expanduser(r"~\AppData\Roaming\npm"),
                    os.path.expanduser(r"~\.npm-global\bin")
                ]
                current_path = env.get('PATH', '')
                for node_path in node_paths:
                    if os.path.exists(node_path) and node_path not in current_path:
                        env['PATH'] = node_path + os.pathsep + current_path
                        current_path = env['PATH']
            
            if 'env' in server_config:
                env.update(server_config['env'])
            
            # Start the MCP server process with timeout
            print(f"   🚀 Starting server process...")
            
            # Debug: print PATH for npx commands
            if server_config['command'] == 'npx':
                print(f"   🔍 DEBUG: Current PATH: {env.get('PATH', 'NOT SET')}")
                # Also try to find npx in common locations
                import shutil
                npx_path = shutil.which('npx', path=env.get('PATH'))
                print(f"   🔍 DEBUG: npx found at: {npx_path if npx_path else 'NOT FOUND'}")
            
            try:
                # On Windows, npx needs to be run through shell or with .cmd extension
                if sys.platform == "win32" and server_config['command'] == 'npx':
                    # Use shell mode for npx on Windows
                    cmd_string = f"{server_config['command']} {' '.join(server_config.get('args', []))}"
                    process = await asyncio.wait_for(
                        asyncio.create_subprocess_shell(
                            cmd_string,
                            stdin=asyncio.subprocess.PIPE,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            env=env
                        ),
                        timeout=15
                    )
                else:
                    process = await asyncio.wait_for(
                        asyncio.create_subprocess_exec(
                            server_config['command'],
                            *server_config.get('args', []),
                            stdin=asyncio.subprocess.PIPE,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            env=env
                        ),
                        timeout=15
                    )
            except asyncio.TimeoutError:
                server_info["status"] = "failed"
                server_info["error"] = "Timeout starting server process"
                print(f"   ❌ Timeout starting server process")
                return server_info
            
            # Give the server a moment to start
            await asyncio.sleep(1)
            
            # Check if process is still running
            if process.returncode is not None:
                stderr_output = await process.stderr.read()
                stderr_text = stderr_output.decode() if stderr_output else "No error output"
                server_info["status"] = "failed"
                server_info["error"] = f"Server process exited immediately with code {process.returncode}. Stderr: {stderr_text}"
                print(f"   ❌ Server process exited immediately")
                print(f"   Error output: {stderr_text}")
                return server_info
            
            # Send initialization request
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                        "prompts": {}
                    },
                    "clientInfo": {
                        "name": "mcp-discovery",
                        "version": "1.0.0"
                    }
                }
            }
            
            print(f"   📤 Sending initialization request...")
            init_response = await self.send_mcp_request(process, init_request, timeout=10)
            
            if "error" in init_response:
                server_info["status"] = "failed"
                server_info["error"] = f"Initialization failed: {init_response['error']}"
                print(f"   ❌ Initialization failed: {init_response['error']}")
                return server_info
            
            if "result" not in init_response:
                server_info["status"] = "failed"
                server_info["error"] = f"Invalid initialization response: {init_response}"
                print(f"   ❌ Invalid initialization response")
                return server_info
            
            print(f"   ✅ Server initialized successfully")
            
            # CRITICAL: Send initialized notification (this was missing!)
            initialized_notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            print(f"   📤 Sending initialized notification...")
            await self.send_mcp_request(process, initialized_notification, expect_response=False)
            
            # Give server time to process the notification
            await asyncio.sleep(0.5)
            
            server_info["status"] = "connected"
            
            # Now we can safely request tools
            tools_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            }
            
            print(f"   📤 Requesting tools list...")
            tools_response = await self.send_mcp_request(process, tools_request, timeout=10)
            
            if "result" in tools_response and "tools" in tools_response["result"]:
                server_info["tools"] = tools_response["result"]["tools"]
                print(f"   🔧 Found {len(server_info['tools'])} tools")
                for tool in server_info["tools"]:
                    name = tool.get('name', 'unnamed')
                    desc = tool.get('description', 'no description')
                    print(f"      - {name}: {desc}")
                    
                    # Display input schema/parameters
                    input_schema = tool.get('inputSchema', {})
                    if input_schema:
                        self._print_tool_parameters(input_schema, indent="        ")
                    else:
                        print(f"        📝 No parameters defined")
            elif "error" in tools_response:
                print(f"   ⚠️ Tools discovery failed: {tools_response['error']}")
                server_info["tools_error"] = tools_response["error"]
            else:
                print(f"   ⚠️ Unexpected tools response: {tools_response}")
            
            # Discover resources
            resources_request = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "resources/list",
                "params": {}
            }
            
            print(f"   📤 Requesting resources list...")
            resources_response = await self.send_mcp_request(process, resources_request, timeout=5)
            
            if "result" in resources_response and "resources" in resources_response["result"]:
                server_info["resources"] = resources_response["result"]["resources"]
                print(f"   📁 Found {len(server_info['resources'])} resources")
            elif "error" in resources_response:
                print(f"   ⚠️ Resources discovery failed: {resources_response['error']}")
                server_info["resources_error"] = resources_response["error"]
            
            # Discover prompts
            prompts_request = {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "prompts/list",
                "params": {}
            }
            
            print(f"   📤 Requesting prompts list...")
            prompts_response = await self.send_mcp_request(process, prompts_request, timeout=5)
            
            if "result" in prompts_response and "prompts" in prompts_response["result"]:
                server_info["prompts"] = prompts_response["result"]["prompts"]
                print(f"   💬 Found {len(server_info['prompts'])} prompts")
            elif "error" in prompts_response:
                print(f"   ⚠️ Prompts discovery failed: {prompts_response['error']}")
                server_info["prompts_error"] = prompts_response["error"]
            
        except FileNotFoundError:
            server_info["status"] = "failed"
            server_info["error"] = f"Command '{server_config['command']}' not found"
            print(f"   ❌ Command not found: {server_config['command']}")
        except Exception as e:
            server_info["status"] = "failed"
            server_info["error"] = str(e)
            print(f"   ❌ Error: {str(e)}")
        
        finally:
            # Clean up process
            if process:
                print(f"   🔄 Closing connection...")
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5)
                    print(f"   ✅ Process terminated gracefully")
                except asyncio.TimeoutError:
                    print(f"   ⚠️ Force killing process...")
                    process.kill()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=2)
                    except:
                        pass
                except Exception as e:
                    print(f"   ⚠️ Error during cleanup: {e}")
        
        return server_info

    def _print_tool_parameters(self, schema: Dict, indent: str = ""):
        """Print tool parameter schema in a readable format"""
        if not schema or schema.get('type') != 'object':
            return
        
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        
        if not properties:
            print(f"{indent}📝 No parameters")
            return
        
        print(f"{indent}📝 Parameters:")
        for param_name, param_info in properties.items():
            is_required = param_name in required
            req_indicator = "🔴" if is_required else "🔵"
            param_type = param_info.get('type', 'unknown')
            param_desc = param_info.get('description', 'No description')
            default_val = param_info.get('default')
            
            print(f"{indent}  {req_indicator} {param_name} ({param_type}): {param_desc}")
            
            if default_val is not None:
                print(f"{indent}     Default: {default_val}")
            
            # Handle enum values
            if 'enum' in param_info:
                enum_vals = ', '.join([str(v) for v in param_info['enum']])
                print(f"{indent}     Options: {enum_vals}")
            
            # Handle nested objects
            if param_type == 'object' and 'properties' in param_info:
                print(f"{indent}     Object properties:")
                self._print_tool_parameters(param_info, indent + "       ")

    async def discover_all_tools(self) -> Dict[str, Any]:
        """Discover tools from all configured MCP servers"""
        print("🚀 Starting MCP Tools Discovery")
        print(f"📄 Reading configuration from: {self.config_file}")
        
        if "mcpServers" not in self.config:
            print("❌ No 'mcpServers' section found in configuration")
            return {}
        
        servers = self.config["mcpServers"]
        print(f"🔍 Found {len(servers)} configured servers")
        
        discovery_results = {}
        
        for server_name, server_config in servers.items():
            server_info = await self.discover_server_tools(server_name, server_config)
            discovery_results[server_name] = server_info
        
        self.discovered_tools = discovery_results
        return discovery_results

    def print_summary(self):
        """Print a summary of all discovered tools"""
        print("\n" + "="*60)
        print("📊 MCP TOOLS DISCOVERY SUMMARY")
        print("="*60)
        
        total_tools = 0
        total_resources = 0
        total_prompts = 0
        active_servers = 0
        
        for server_name, server_info in self.discovered_tools.items():
            status_emoji = "✅" if server_info["status"] == "connected" else "❌"
            print(f"\n{status_emoji} Server: {server_name}")
            print(f"   Status: {server_info['status']}")
            
            if server_info["status"] == "connected":
                active_servers += 1
                tools_count = len(server_info["tools"])
                resources_count = len(server_info["resources"])
                prompts_count = len(server_info["prompts"])
                
                total_tools += tools_count
                total_resources += resources_count
                total_prompts += prompts_count
                
                print(f"   Tools: {tools_count}")
                print(f"   Resources: {resources_count}")
                print(f"   Prompts: {prompts_count}")
                
                if server_info["tools"]:
                    print("   Available Tools:")
                    for tool in server_info["tools"]:
                        name = tool.get('name', 'unnamed')
                        print(f"     • {name}")
                        
                        # Show parameter summary
                        input_schema = tool.get('inputSchema', {})
                        if input_schema and input_schema.get('properties'):
                            properties = input_schema['properties']
                            required = input_schema.get('required', [])
                            req_count = len([p for p in properties.keys() if p in required])
                            opt_count = len(properties) - req_count
                            print(f"       Parameters: {req_count} required, {opt_count} optional")
                        else:
                            print(f"       Parameters: none")
            else:
                print(f"   Error: {server_info.get('error', 'Unknown error')}")
        
        print(f"\n📈 TOTALS:")
        print(f"   Active Servers: {active_servers}/{len(self.discovered_tools)}")
        print(f"   Total Tools: {total_tools}")
        print(f"   Total Resources: {total_resources}")
        print(f"   Total Prompts: {total_prompts}")

    def save_results(self, output_file: str = "mcp_discovery_results.json"):
        """Save discovery results to a JSON file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.discovered_tools, f, indent=2)
            print(f"\n💾 Results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

    def print_detailed_tools(self, server_name: str = None):
        """Print detailed information about tools including full parameter schemas"""
        print("\n" + "="*80)
        print("🔧 DETAILED TOOLS INFORMATION")
        print("="*80)
        
        servers_to_show = [server_name] if server_name else list(self.discovered_tools.keys())
        
        for srv_name in servers_to_show:
            if srv_name not in self.discovered_tools:
                print(f"❌ Server '{srv_name}' not found")
                continue
                
            server_info = self.discovered_tools[srv_name]
            if server_info["status"] != "connected" or not server_info["tools"]:
                print(f"\n📭 No tools available for server: {srv_name}")
                continue
            
            print(f"\n🏢 Server: {srv_name}")
            print("-" * 60)
            
            for tool in server_info["tools"]:
                name = tool.get('name', 'unnamed')
                desc = tool.get('description', 'No description provided')
                
                print(f"\n🔧 Tool: {name}")
                print(f"   Description: {desc}")
                
                input_schema = tool.get('inputSchema', {})
                if input_schema:
                    self._print_tool_parameters(input_schema, "   ")
                else:
                    print("   📝 No parameters defined")
                print()

    def generate_tool_usage_examples(self, output_file: str = "mcp_tool_examples.md"):
        """Generate markdown file with tool usage examples"""
        try:
            with open(output_file, 'w') as f:
                f.write("# MCP Tools Usage Examples\n\n")
                f.write("Generated tool documentation with parameter examples.\n\n")
                
                for server_name, server_info in self.discovered_tools.items():
                    if server_info["status"] != "connected" or not server_info["tools"]:
                        continue
                    
                    f.write(f"## Server: {server_name}\n\n")
                    
                    for tool in server_info["tools"]:
                        name = tool.get('name', 'unnamed')
                        desc = tool.get('description', 'No description provided')
                        
                        f.write(f"### {name}\n\n")
                        f.write(f"**Description:** {desc}\n\n")
                        
                        input_schema = tool.get('inputSchema', {})
                        if input_schema and input_schema.get('properties'):
                            f.write("**Parameters:**\n\n")
                            properties = input_schema['properties']
                            required = input_schema.get('required', [])
                            
                            for param_name, param_info in properties.items():
                                is_required = param_name in required
                                param_type = param_info.get('type', 'unknown')
                                param_desc = param_info.get('description', 'No description')
                                default_val = param_info.get('default')
                                
                                req_text = "**Required**" if is_required else "Optional"
                                f.write(f"- `{param_name}` ({param_type}) - {req_text}\n")
                                f.write(f"  {param_desc}\n")
                                
                                if default_val is not None:
                                    f.write(f"  Default: `{default_val}`\n")
                                
                                if 'enum' in param_info:
                                    enum_vals = ', '.join([f"`{v}`" for v in param_info['enum']])
                                    f.write(f"  Options: {enum_vals}\n")
                                f.write("\n")
                            
                            # Generate example usage
                            f.write("**Example usage:**\n\n")
                            example_params = {}
                            for param_name, param_info in properties.items():
                                param_type = param_info.get('type', 'string')
                                default_val = param_info.get('default')
                                
                                if default_val is not None:
                                    example_params[param_name] = default_val
                                elif param_type == 'string':
                                    if 'enum' in param_info:
                                        example_params[param_name] = param_info['enum'][0]
                                    else:
                                        example_params[param_name] = f"example_{param_name}"
                                elif param_type == 'integer':
                                    example_params[param_name] = 42
                                elif param_type == 'boolean':
                                    example_params[param_name] = True
                                elif param_type == 'array':
                                    example_params[param_name] = ["example"]
                                else:
                                    example_params[param_name] = f"<{param_type}>"
                            
                            f.write("```json\n")
                            f.write(json.dumps(example_params, indent=2))
                            f.write("\n```\n\n")
                        else:
                            f.write("**Parameters:** None\n\n")
                        
                        f.write("---\n\n")
            
            print(f"\n📖 Tool examples generated: {output_file}")
            
        except Exception as e:
            print(f"❌ Error generating examples: {e}")

def update_skills_file(self, skills_file: str = "skills.txt"):
    """Update skills.txt file with discovered MCP tools"""
    try:
        # Read existing skills file if it exists
        existing_content = ""
        if os.path.exists(skills_file):
            with open(skills_file, 'r', encoding='utf-8') as f:
                existing_content = f.read()
        
        # Find the MCP Tool Usage section
        mcp_section_start = existing_content.find("[Skill: MCP Tool Usage]")
        if mcp_section_start == -1:
            print(f"❌ Could not find '[Skill: MCP Tool Usage]' section in {skills_file}")
            return
        
        # Find where to insert new tools (after "Available MCP Tools:")
        tools_list_start = existing_content.find("Available MCP Tools:", mcp_section_start)
        if tools_list_start == -1:
            print(f"❌ Could not find 'Available MCP Tools:' section in {skills_file}")
            return
        
        # Find the end of the tools list (before the 🛑 Important MCP Rules:)
        tools_list_end = existing_content.find("🛑 Important MCP Rules:", tools_list_start)
        if tools_list_end == -1:
            print(f"❌ Could not find end of tools list in {skills_file}")
            return
        
        # Extract existing tools list
        existing_tools_section = existing_content[tools_list_start:tools_list_end]
        existing_tools_lines = existing_tools_section.split('\n')
        
        # Parse existing tools to avoid duplicates
        existing_tool_names = set()
        for line in existing_tools_lines:
            if line.strip().startswith('- `'):
                tool_name = line.split('`')[1] if '`' in line else ""
                if tool_name:
                    existing_tool_names.add(tool_name)
        
        # Generate new tools list and example patterns
        new_tools = []
        new_examples = []
        
        for server_name, server_info in self.discovered_tools.items():
            if server_info["status"] == "connected" and server_info["tools"]:
                for tool in server_info["tools"]:
                    tool_name = tool.get('name', 'unnamed')
                    tool_desc = tool.get('description', 'No description')
                    
                    if tool_name not in existing_tool_names:
                        new_tools.append(f"- `{tool_name}` - {tool_desc}")
                        
                        # Generate example pattern for this tool
                        if tool_name == "recommend-mcp-servers":
                            example = (
                                'User: "What is a good MCP server for AWS Lambda deployment?"\n'
                                'Gizmo: I\'ll find some MCP servers for AWS Lambda deployment.\n'
                                'しrecommend-mcp-servers({"query": "MCP Server for AWS Lambda Python3.9 deployment"})'
                            )
                            new_examples.append(example)
        
        if not new_tools:
            print(f"✅ No new tools to add to {skills_file}")
            return
        
        # Find where to insert new examples (after existing examples)
        examples_section_start = existing_content.find("Example Usage Patterns:")
        if examples_section_start == -1:
            print(f"❌ Could not find 'Example Usage Patterns:' section in {skills_file}")
            return
        
        # Find the end of existing examples
        examples_end = existing_content.find("User: \"What's 2+2?\"", examples_section_start)
        if examples_end == -1:
            examples_end = existing_content.find("User: \"Save this code.\"", examples_section_start)
        if examples_end == -1:
            print(f"❌ Could not find end of examples section in {skills_file}")
            return
        
        # Insert new tools and examples
        lines = existing_content.split('\n')
        
        # Insert new tools
        insert_line = -1
        for i, line in enumerate(lines):
            if "Available MCP Tools:" in line:
                # Find the last tool in the existing list
                j = i + 1
                while j < len(lines) and (lines[j].strip().startswith('-') or lines[j].strip() == ''):
                    if lines[j].strip().startswith('-'):
                        insert_line = j
                    j += 1
                break
        
        if insert_line == -1:
            print(f"❌ Could not find where to insert new tools in {skills_file}")
            return
        
        for tool_line in new_tools:
            insert_line += 1
            lines.insert(insert_line, tool_line)
        
        # Insert new examples
        examples_insert_line = -1
        for i, line in enumerate(lines):
            if "Example Usage Patterns:" in line:
                # Find the last example
                j = i + 1
                while j < len(lines) and (lines[j].strip().startswith('User:') or 
                                         lines[j].strip().startswith('Gizmo:') or 
                                         lines[j].strip().startswith('し') or
                                         lines[j].strip() == ''):
                    if lines[j].strip().startswith('User:'):
                        examples_insert_line = j
                    j += 1
                break
        
        if examples_insert_line == -1:
            print(f"❌ Could not find where to insert new examples in {skills_file}")
            return
        
        for example in reversed(new_examples):
            for example_line in reversed(example.split('\n')):
                examples_insert_line += 1
                lines.insert(examples_insert_line, example_line)
            # Add blank line after each example
            examples_insert_line += 1
            lines.insert(examples_insert_line, '')
        
        # Write updated content back to file
        updated_content = '\n'.join(lines)
        with open(skills_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"\n✅ Updated {skills_file} with:")
        print(f"   - {len(new_tools)} new tools")
        print(f"   - {len(new_examples)} new example patterns")
        for tool_line in new_tools:
            print(f"      {tool_line}")
        
    except Exception as e:
        print(f"❌ Error updating skills file: {e}")

async def main():
    """Main function to run the MCP tools discovery"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Discover MCP tools from mcp.json configuration")
    parser.add_argument("--config", "-c", default="mcp.json", help="Path to MCP configuration file (default: mcp.json)")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--server", "-s", help="Test only a specific server by name")
    parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed parameter information")
    parser.add_argument("--examples", "-e", help="Generate tool usage examples to markdown file")
    parser.add_argument("--update-skills", "-u", help="Update skills.txt with new tool examples (specify skills file path, default: skills.txt)")
    parser.add_argument("--api-url", help="Custom AI API URL (default: Hack Club AI)")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create discovery instance
    discovery = MCPToolsDiscovery(config_file=args.config)
    
    # If testing a specific server, filter the config
    if args.server:
        if args.server in discovery.config.get("mcpServers", {}):
            discovery.config["mcpServers"] = {args.server: discovery.config["mcpServers"][args.server]}
            print(f"🎯 Testing only server: {args.server}")
        else:
            print(f"❌ Server '{args.server}' not found in configuration")
            return
    
    # Discover all tools
    await discovery.discover_all_tools()
    
    # Print summary
    discovery.print_summary()
    
    # Print detailed info if requested
    if args.detailed:
        discovery.print_detailed_tools(args.server)
    
    # Save results if requested
    if args.output:
        discovery.save_results(args.output)
    
    # Generate examples if requested
    if args.examples:
        discovery.generate_tool_usage_examples(args.examples)
    
    # Update skills file if requested
    if args.update_skills is not None:
        skills_file = args.update_skills if args.update_skills else "skills.txt"
        # Handle absolute paths that start with / on Windows
        if skills_file.startswith('/'):
            # Convert Unix-style absolute path to relative path
            skills_file = skills_file.lstrip('/')
        discovery.update_skills_file(skills_file)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Discovery interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)