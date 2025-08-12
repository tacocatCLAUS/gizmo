#!/usr/bin/env python3
"""
Simple MCP Tools Discovery - Get tools, update skills.txt, and generate examples for new tools
"""

import json
import asyncio
import os
import sys
import warnings
import requests
from model.modelbuilder import build

# Suppress asyncio ResourceWarnings on Windows
warnings.filterwarnings("ignore", category=ResourceWarning)

async def get_mcp_tools():
    """Get all MCP tools and return as dict"""
    
    # Load mcp.json
    if not os.path.exists("mcp.json"):
        manager("ERROR: mcp.json not found", file=sys.stderr)
        return {}
    
    with open("mcp.json", 'r') as f:
        config = json.load(f)
    
    servers = config.get("mcpServers", {})
    all_tools = {}
    
    for server_name, server_config in servers.items():
        try:
            tools = await discover_server_tools(server_name, server_config)
            all_tools[server_name] = tools
        except Exception as e:
            manager(f"ERROR discovering {server_name}: {e}", file=sys.stderr)
    
    return all_tools

async def discover_server_tools(server_name, server_config):
    """Get tools from one server"""
    
    # Auto-add -y flag for npx
    if server_config['command'] == 'npx' and '-y' not in server_config.get('args', []):
        server_config['args'] = ['-y'] + server_config.get('args', [])
    
    process = None
    try:
        # Setup environment
        env = os.environ.copy()
        if 'env' in server_config:
            env.update(server_config['env'])
        
        # Start server process
        if sys.platform == "win32" and server_config['command'] == 'npx':
            cmd_string = f"{server_config['command']} {' '.join(server_config.get('args', []))}"
            process = await asyncio.create_subprocess_shell(
                cmd_string,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
        else:
            process = await asyncio.create_subprocess_exec(
                server_config['command'],
                *server_config.get('args', []),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
        
        await asyncio.sleep(1)  # Let server start
        
        # Check if process died
        if process.returncode is not None:
            return {"error": "Server process exited immediately"}
        
        # Initialize server
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                "clientInfo": {"name": "simple-discovery", "version": "1.0.0"}
            }
        }
        
        await send_request(process, init_request)
        
        # Send initialized notification
        init_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        await send_request(process, init_notification, expect_response=False)
        
        await asyncio.sleep(0.5)
        
        # Get tools
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        tools_response = await send_request(process, tools_request)
        
        if "result" in tools_response and "tools" in tools_response["result"]:
            return {"tools": tools_response["result"]["tools"]}
        else:
            return {"error": "Failed to get tools", "response": tools_response}
            
    except Exception as e:
        return {"error": str(e)}
    
    finally:
        if process:
            try:
                # Close stdin first to signal shutdown
                if process.stdin and not process.stdin.is_closing():
                    process.stdin.close()
                    await asyncio.sleep(0.1)
                
                # Wait for graceful exit
                try:
                    await asyncio.wait_for(process.wait(), timeout=1)
                except asyncio.TimeoutError:
                    # Force terminate if needed
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=1)
                    except asyncio.TimeoutError:
                        process.kill()
                        try:
                            await asyncio.wait_for(process.wait(), timeout=0.5)
                        except:
                            pass
            except Exception:
                pass

async def send_request(process, request, expect_response=True):
    """Send JSON-RPC request to MCP server"""
    try:
        request_str = json.dumps(request) + "\n"
        process.stdin.write(request_str.encode())
        await process.stdin.drain()
        
        if not expect_response:
            return {"success": True}
        
        # Read response with timeout
        response_line = await asyncio.wait_for(process.stdout.readline(), timeout=10)
        
        if not response_line:
            return {"error": "No response"}
        
        response_text = response_line.decode().strip()
        return json.loads(response_text)
        
    except Exception as e:
        return {"error": str(e)}
    
devmode = False
def manager(message=None, pos_var=None, flush=False):
    if devmode:
        if message:
            if pos_var:
                print(message + pos_var)
            else:
                print(message)

def get_existing_tools(skills_file):
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

def update_skills_file(skills_file, all_tools, new_tool_examples):
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
                tool_desc = tool.get('description', 'No description')
                if tool_name not in existing_tools:
                    new_tools_list.append(f"- `{tool_name}` - {tool_desc}")
    
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

def query_ai(prompt_text):
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

def serverupdate():
    """Main function"""
    skills_file = "model/skills.txt"
    
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            tools_data = loop.run_until_complete(get_mcp_tools())
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            finally:
                loop.close()
        
        # Get existing tools from skills.txt
        existing_tools = get_existing_tools(skills_file)
        
        # Find new tools only
        new_tools = {}
        for server_name, server_info in tools_data.items():
            if 'tools' in server_info:
                new_server_tools = []
                for tool in server_info['tools']:
                    tool_name = tool.get('name', 'unnamed')
                    if tool_name not in existing_tools:
                        new_server_tools.append(tool)
                
                if new_server_tools:
                    new_tools[server_name] = {"tools": new_server_tools}
        
        if not new_tools:
            manager("No new tools found. Skills file is up to date.")
            return
        
        manager(f"Found {sum(len(s['tools']) for s in new_tools.values())} new tools")
        
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
   - No extra commentary or text outside of the examples. SO NO ADDITIONAL TEXT. NO "(I'll wait for your confirmation before proceeding, or is there anything else I can help you with?)" OR ANYTHING ELSE.
Example of expected style:
User: "What's the latest news about AI developments?"
Gizmo: I'll search for the latest AI news for you.
⚡️web_search({"query": "latest AI news", "max_results": 5})

User: "Summarize this article: example.com/article"
Gizmo: Sure — I'll fetch the article text for you.
⚡️fetch_webpage({"url": "https://example.com/article", "max_chars": 1500})

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
        build("model/system.txt", "model/skills.txt", "model/Modelfile", "gizmo", "wizardlm2:7b")
                
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    serverupdate()