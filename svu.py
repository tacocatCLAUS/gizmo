#!/usr/bin/env python3
"""
Simple MCP Tools Discovery - Get tools, generate prompt, and query AI
"""

import json
import asyncio
import os
import sys
import warnings
import requests

# Suppress asyncio ResourceWarnings on Windows
warnings.filterwarnings("ignore", category=ResourceWarning)

async def get_mcp_tools():
    """Get all MCP tools and return as dict"""
    
    # Load mcp.json
    if not os.path.exists("mcp.json"):
        print("ERROR: mcp.json not found", file=sys.stderr)
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
            print(f"ERROR discovering {server_name}: {e}", file=sys.stderr)
    
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

def main():
    """Main function with proper asyncio handling for Windows"""
    if sys.platform == "win32":
        # Use ProactorEventLoop on Windows to avoid some cleanup issues
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        # Create new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            tools_data = loop.run_until_complete(get_mcp_tools())
        finally:
            # Properly close the loop
            try:
                # Cancel any remaining tasks
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            finally:
                loop.close()
        
        # Create the prompt with discovered tools
        prompt = """You will be given a JSON object containing a list of tools with their names, descriptions, and inputSchemas.
Your task:
1. For each tool, create an example interaction consisting of:
   - A "User:" line with a realistic example request based on the tool's purpose.
   - A "Gizmo:" line with a short natural reply that sounds like a helpful assistant.
   - A tool call in the format: し<tool_name>({...arguments...})
     - Arguments must match the tool's inputSchema.
     - Fill required fields with realistic sample values.
     - Include default fields if they appear in the schema.
2. Output format:
   - No titles, no explanations.
   - Each example in the form:
     ```
     User: "<example request>"
     Gizmo: <assistant reply>
     し<tool_name>({<json args>})
     ```
   - Separate each example with a blank line.
   - No extra commentary or text outside of the examples.
Example of expected style:
User: "What's the latest news about AI developments?"
Gizmo: I'll search for the latest AI news for you.
しweb_search({"query": "latest AI news", "max_results": 5})

User: "Summarize this article: example.com/article"
Gizmo: Sure — I'll fetch the article text for you.
しfetch_webpage({"url": "https://example.com/article", "max_chars": 1500})

Here are the tools:
""" + json.dumps(tools_data, indent=2)
        
        # Query the AI
        print("Querying Hack Club AI...")
        ai_response = query_ai(prompt)
        print("\n" + "="*60)
        print("AI RESPONSE:")
        print("="*60)
        print(ai_response)
                
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()