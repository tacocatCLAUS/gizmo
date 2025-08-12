import os
import sys
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

from yacana import Task, OllamaAgent, OpenAiAgent, LoggerManager
from pathlib import Path
from termcolor import colored, cprint
from survey import routines
from langchain_chroma import Chroma
from RAG.populate_database import parse, clear_database
from Libraries.filepicker import select_file
from Libraries.svu import serverupdate
from RAG.get_embedding_function import get_embedding_function
from langchain.prompts import ChatPromptTemplate
import shutil
import asyncio
import threading
import queue
from typing import Any, Dict, List
from pydantic import BaseModel, Field, create_model
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json
import re

# Configuration
openai = False
openai_model = "gpt-3.5-turbo"
devmode = False
db_clear = True
use_mcp = True

system_prompt_path = Path("model/system.txt")
system_prompt = system_prompt_path.read_text(encoding="utf-8")
skills_prompt_path = Path("model/skills.txt")
openai_api_key = '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'
ollama_agent = OllamaAgent("ʕ•ᴥ•ʔ Gizmo", "gizmo")
openai_agent = OpenAiAgent("ʕ•ᴥ•ʔ Gizmo", openai_model, system_prompt=system_prompt, api_token=openai_api_key)
agent = ollama_agent
stream_state = {"stream": "true"}
db_query = False
addfile = 'N'
CHROMA_PATH = "RAG/chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
[File]
{context}

---

Answer the question based on the above context: {question}
"""

MCP_TEMPLATE = """
The user asked this question: {question} and you answered like this: {answer} with this information: {tool_result} continue where you stopped and answer with this new information. 
"""

class MCPServerConfig:
    """Configuration for a single MCP server"""
    def __init__(self, name: str, command: str, args: List[str], cwd: str = None, env: Dict[str, str] = None):
        self.name = name
        self.command = command
        self.args = args
        self.cwd = cwd or str(Path.cwd())
        self.env = env or {}

def load_mcp_config(config_path: str = "mcp.json") -> Dict[str, MCPServerConfig]:
    """Load MCP server configurations from JSON file"""
    serverupdate()
    config_file = Path(config_path)
    if not config_file.exists():
        # Create default config file
        default_config = {
            "mcpServers": {
                "local-server": {
                    "command": sys.executable,
                    "args": ["mcp-server.py"]
                }
            }
        }
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        cprint(f"ʕ•ᴥ•ʔ Created default MCP config at {config_path}", 'yellow', attrs=["bold"])
    
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        servers = {}
        mcp_servers = config_data.get("mcpServers", {})
        
        for server_name, server_config in mcp_servers.items():
            servers[server_name] = MCPServerConfig(
                name=server_name,
                command=server_config["command"],
                args=server_config.get("args", []),
                cwd=server_config.get("cwd"),
                env=server_config.get("env")
            )
        
        return servers
    except Exception as e:
        cprint(f"ʕ•ᴥ•ʔ Error loading MCP config: {str(e)}", 'red', attrs=["bold"])
        return {}

class OllamaMCP:
    def __init__(self, server_config: MCPServerConfig):
        self.server_config = server_config
        self.server_params = StdioServerParameters(
            command=server_config.command,
            args=server_config.args,
            cwd=server_config.cwd,
            env=server_config.env
        )
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.initialized = threading.Event()
        self.tools: list[Any] = []
        self.thread = threading.Thread(target=self._run_background, daemon=True)
        self.thread.start()

    def _run_background(self):
        asyncio.run(self._async_run())

    async def _async_run(self):
        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    manager(f"Initializing MCP session for {self.server_config.name}...")
                    await session.initialize()
                    self.session = session
                    manager(f"Listing available tools for {self.server_config.name}...")
                    tools_result = await session.list_tools()
                    self.tools = tools_result.tools
                    manager(f"Found {len(self.tools)} tools in {self.server_config.name}: {[tool.name for tool in self.tools]}")
                    self.initialized.set()

                    while True:
                        try:
                            tool_name, arguments = self.request_queue.get(block=False)
                        except queue.Empty:
                            await asyncio.sleep(0.01)
                            continue

                        if tool_name is None:
                            break
                        try:
                            result = await session.call_tool(tool_name, arguments)
                            self.response_queue.put(result)
                        except Exception as e:
                            self.response_queue.put(f"Error: {str(e)}")
        except Exception as e:
            manager(f"MCP Session Initialization Error for {self.server_config.name}: {str(e)}")
            self.initialized.set()
            self.response_queue.put(f"MCP initialization error: {str(e)}")

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if not self.initialized.wait(timeout=30):
            raise TimeoutError(f"MCP session for {self.server_config.name} did not initialize in time.")
        self.request_queue.put((tool_name, arguments))
        result = self.response_queue.get()
        return result

    def shutdown(self):
        self.request_queue.put((None, None))
        self.thread.join()
        manager(f"MCP session {self.server_config.name} shut down.")

class MCPManager:
    """Manages multiple MCP server connections"""
    def __init__(self, config_path: str = "mcp.json"):
        self.config_path = config_path
        self.clients: Dict[str, OllamaMCP] = {}
        self.all_tools: Dict[str, str] = {}  # tool_name -> server_name mapping
        
    def initialize(self):
        """Initialize all MCP servers from config"""
        server_configs = load_mcp_config(self.config_path)
        
        if not server_configs:
            cprint("ʕ•ᴥ•ʔ No MCP servers configured", 'yellow', attrs=["bold"])
            return
        
        successful_connections = 0
        for server_name, server_config in server_configs.items():
            try:
                manager(f"ʕ•ᴥ•ʔ Connecting to MCP server: {server_name}")
                client = OllamaMCP(server_config)
                
                if client.initialized.wait(timeout=30):
                    self.clients[server_name] = client
                    # Map tools to their server
                    for tool in client.tools:
                        self.all_tools[tool.name] = server_name
                    successful_connections += 1
                    manager(f"ʕ•ᴥ•ʔ Connected to {server_name} with {len(client.tools)} tools")
                else:
                    manager(f"ʕ•ᴥ•ʔ Connection to {server_name} timed out")
                    
            except Exception as e:
                manager(f"ʕ•ᴥ•ʔ Failed to connect to {server_name}: {str(e)}")
        
        if successful_connections > 0:
            cprint(f"ʕ•ᴥ•ʔ Successfully connected to {successful_connections} MCP server(s)", 'green', attrs=["bold"])
            manager(f"ʕ•ᴥ•ʔ Available tools: {list(self.all_tools.keys())}")
        else:
            cprint("ʕ•ᴥ•ʔ No MCP servers available", 'red', attrs=["bold"])
    
    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on the appropriate server"""
        server_name = self.all_tools.get(tool_name)
        if not server_name:
            raise ValueError(f"Tool '{tool_name}' not found in any connected server")
        
        client = self.clients.get(server_name)
        if not client:
            raise ValueError(f"Server '{server_name}' not connected")
        
        return client.call_tool(tool_name, arguments)
    
    def shutdown_all(self):
        """Shutdown all MCP connections"""
        for client in self.clients.values():
            client.shutdown()
        self.clients.clear()
        self.all_tools.clear()
        cprint("ʕ•ᴥ•ʔ All MCP connections shut down", 'yellow', attrs=["bold"])

def dbclear():
    if db_clear:
        clear_database()
    else:
        cprint('ʕ•ᴥ•ʔ Persistent memory is on.', 'yellow', attrs=["bold"])

def manager(message=None, pos_var=None, flush=False):
    if not devmode:
        LoggerManager.set_log_level(None)
    else:
        if message:
            if pos_var:
                print(message + pos_var)
            else:
                print(message)

def set_agent():
    global agent
    agent = openai_agent if openai else ollama_agent

def streaming_callback(chunk: str):
    """Streaming callback that detects tool calls and pauses streaming"""
    
    manager(f"🔧 [DEBUG] Streaming callback received chunk: {repr(chunk[:50])}...")
    
    # Check for tool call pattern
    if "⚡️" in chunk:
        stream_state["stream"] = "false"
        manager(f"\n🔧 [Tool Call Detected - Processing...]")
        manager(f"🔧 [DEBUG] Tool call detected in chunk: {repr(chunk)}")
        return
    
    # Only print if streaming is active
    if stream_state["stream"] == "true":
        print(f"{chunk}", end="", flush=True)
    else:
        manager(f"🔧 [DEBUG] Streaming paused, not printing: {repr(chunk[:30])}...")
    
    return chunk

def incorporate_tool_results(original_request="", partial_response="", tool_result="Tool Failed. Just tell me that and suggest alternatives if you can."):
    """Continue the conversation with tool results incorporated"""
    
    stream_state = {"stream": "true"}
    manager(f"\n🔧 [Tool Complete - Resuming Stream...]")
    
    continuation_prompt = f"""The user asked: {original_request}

You started to answer: {partial_response}

Tool result: {tool_result}

Continue your response naturally, incorporating this tool result. Don't repeat what you already said, just continue from where you left off with the new information."""
    
    Task(continuation_prompt, agent, streaming_callback=streaming_callback).solve()

def query_rag(request):
    embedding_function = get_embedding_function(openai)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(request, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=request)
    response_text = Task(prompt, agent, streaming_callback=streaming_callback).solve()
    sources = [doc.metadata.get("id", None) for doc, _ in results]
    formatted_response = f"\nSources: {sources}"
    if stream_state["stream"] == "true":
        print(formatted_response)
    return response_text

def handle_tool_execution(response_content, mcp_manager, original_request):
    """Handle tool execution if a tool call was detected"""
    
    manager(f"🔧 [DEBUG] Handle tool execution - stream state: {stream_state}")
    manager(f"🔧 [DEBUG] Response content type: {type(response_content)}")
    
    # Handle both string and object responses
    content_str = ""
    if hasattr(response_content, 'content'):
        content_str = response_content.content
    elif isinstance(response_content, str):
        content_str = response_content
    else:
        content_str = str(response_content)
    
    # Check if there's a tool call in the final response even if streaming didn't detect it
    if "⚡️" in content_str and not stream_state["stream"] == "false":
        manager(f"🔧 [DEBUG] Found ⚡️ in final response, forcing tool execution")
        stream_state["stream"] == "false"
    
    if not mcp_manager or not stream_state["stream"] == "false":
        manager(f"🔧 [DEBUG] Skipping tool execution - mcp_manager: {mcp_manager is not None}, paused: {stream_state['stream'] == 'false'}")
        return
    
    try:
        manager(f"🔧 [DEBUG] Content string: {repr(content_str)}")
        tool_name, arguments = parse_tool_call(content_str)
        if tool_name:
            cprint(f"ʕ•ᴥ•ʔ Using {tool_name}...", 'yellow', attrs=["bold"])
            result = mcp_manager.call_tool(tool_name, arguments)
            manager(f"🔧 Result: {result}")
            incorporate_tool_results(original_request, content_str, str(result))
        else:
            manager(f"🔧 [DEBUG] No tool found in content, resuming stream")
            stream_state["stream"] == "true"
    except Exception as e:
        manager(f"🔧 Tool execution failed: {str(e)}")
        manager(f"🔧 [DEBUG] Exception: {e}")
        incorporate_tool_results(original_request, content_str, "Tool Failed. Just tell me that and suggest alternatives if you can.")

def parse_tool_call(content: str):
    """Parse tool call syntax like: ⚡️tool_name({...json...})"""
    manager(f"🔧 [DEBUG] Parsing content: {repr(content)}")
    pattern = r"⚡️(\w+)\s*\((\{.*?\})\)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        tool_name = match.group(1)
        json_str = match.group(2)
        manager(f"🔧 [DEBUG] Found tool: {tool_name} with args: {json_str}")
        try:
            arguments = json.loads(json_str)
        except json.JSONDecodeError as e:
            manager(f"🔧 JSON parsing error in tool call: {e}")
            arguments = {}
        return tool_name, arguments
    else:
        manager(f"🔧 [DEBUG] No tool call pattern found")
    return None, {}

# Initialize MCP Manager if enabled
mcp_manager = None
if use_mcp:
    try:
        mcp_manager = MCPManager("mcp.json")
        mcp_manager.initialize()
    except Exception as e:
        cprint(f"ʕ•ᴥ•ʔ MCP manager initialization failed: {str(e)}, continuing without MCP...", 'red', attrs=["bold"])
        mcp_manager = None

# Main execution
dbclear()
manager()
set_agent()
cprint('ʕ•ᴥ•ʔฅ Gizmo', 'yellow', attrs=["bold"])
Task("I have no questions. introduce yourself. dont mention your skills at all. be breif.", agent, streaming_callback=streaming_callback).solve()

while True:
    print('\n')
    cprint('(•ᴗ•) You', 'blue', attrs=["bold"])
    request = routines.input()
    if request.strip().lower() == "bye":
        break
    addfile = routines.input('📄 (Y/N): ')
    if addfile == 'Y':
        file_path = select_file()
        db_query = True
        if file_path:
            dest_dir = os.path.join(os.getcwd(), "RAG", "data")
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(file_path, dest_dir)
            parse()
            filename = Path(file_path).name
            print(f"ʕ•ᴥ•ʔ I processed {filename}")
            addfile = 'N'
        else:
            cprint("Error.", 'red')
            manager("[SYSTEM] Error. No path added by user/library.")
    print('\n')
    cprint('ʕ•ᴥ•ʔ Gizmo', 'yellow', attrs=["bold"])
    
    # Reset stream state before each request
    stream_state["stream"] = "true"
    
    if db_query:
        message = query_rag(request)
        content_str = message.content if hasattr(message, 'content') else str(message)
        handle_tool_execution(content_str, mcp_manager, request)
    else:
        message = Task(request, agent, streaming_callback=streaming_callback).solve()
        content_str = message.content if hasattr(message, 'content') else str(message)
        handle_tool_execution(content_str, mcp_manager, request)

if mcp_manager:
    mcp_manager.shutdown_all()