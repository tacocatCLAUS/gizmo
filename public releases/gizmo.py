import time
import os
import subprocess
import sys
from pathlib import Path

# Configuration
devmode = False
db_clear = True
use_mcp = True

def check_and_install_requirements():
    """Check and install requirements from model/requirements.txt"""
    requirements_path = Path("model/requirements.txt")
    
    if not requirements_path.exists():
        return True, "No requirements.txt found - skipping"
    
    try:
        # Read requirements
        with open(requirements_path, 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        if not requirements:
            return True, f"Requirements file empty - {len(requirements)} packages"
        
        # Try to install requirements
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_path)
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            return True, f"Libraries installed - {len(requirements)} packages"
        else:
            return False, f"Failed to install requirements: {result.stderr[:100]}"
            
    except FileNotFoundError:
        return True, "No requirements.txt found - skipping"
    except Exception as e:
        return False, f"Error checking requirements: {str(e)[:100]}"

# Auto-install rich if not available
try:
    from rich.console import Console
    from rich.text import Text
    from survey import routines
    from termcolor import colored, cprint
    from yacana import Task, OllamaAgent, OpenAiAgent, LoggerManager
    from pathlib import Path
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
except ImportError:
    check_and_install_requirements()
    # Restart the script after installing rich
    os.execv(sys.executable, ['python'] + "model/modelbuilder.py")
    os.execv(sys.executable, ['python'] + sys.argv)

console = Console()

def check_python():
    """Check if Python is properly installed"""
    try:
        version = sys.version.split()[0]
        return True, f"Python {version}"
    except:
        return False, "Python not found"

def check_pip():
    """Check if pip is available"""
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return True, "pip available"
        else:
            return False, "pip not found"
    except:
        return False, "pip not available"

def check_ollama():
    """Check if Ollama is installed and accessible"""
    try:
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip().split('\n')[0]
            return True, f"Ollama {version}"
        else:
            return False, "Ollama not responding"
    except FileNotFoundError:
        return False, "Ollama not installed"
    except Exception as e:
        return False, f"Ollama error: {str(e)[:50]}"

def post_sequence():
    """Simulate IBM Netfinity 7000 POST sequence with real checks"""
    # Static messages first
    static_messages = [
        ("Gizmo 7000 MCP Version 12", 1.2),
        ("(c) MIT License CLAUS 2025", 0.8),
        ("Requirements Test in progress...", 1.5),
    ]
    
    for message, delay in static_messages:
        post_text = Text(message)
        post_text.stylize("bold green")
        console.print(post_text)
        time.sleep(delay)
    
    # Dynamic checks
    checks = [
        ("001: Python", check_python),
        ("002: pip", check_pip),
        ("003: Libraries", check_and_install_requirements),
        ("00F: Ollama", check_ollama),
    ]
    
    for check_id, check_func in checks:
        # Perform actual check in background
        success, message = check_func()
        
        if success:
            result_text = Text(f"  {check_id}: {message} - OK")
            result_text.stylize("bold green")
            console.print(result_text)
        else:
            result_text = Text(f"  {check_id}: {message} - FAILED")
            result_text.stylize("bold red")
            console.print(result_text)
            
            # Exit on critical failures
            if check_func in [check_python, check_ollama]:
                error_text = Text(f"\nCRITICAL ERROR: {message}")
                error_text.stylize("bold red")
                console.print(error_text)
                console.print(Text("Please install the missing component and run again.", "bold red"))
                sys.exit(1)
        
        time.sleep(0.8)
    
    # Continue with remaining static messages
    remaining_messages = [
        ("  010: Hard Disk 0: IBM DCAS-34330 (4.3 GB) - OK", 0.5),
        ("  011: CD-ROM Drive Detected", 0.4),
        ("  012: Network Controller: Intel EtherExpress Pro/100 - OK", 0.5),
        ("Boot Device Priority: SCSI ID 0", 0.8),
        ("Loading Operating System...", 1.2),
        ("Yacana Boot Agent v4.02", 0.6),
        ("Press F to quit.", 0.4),
        ("Starting CUDA 21.3.3 kernel...", 1.0),
        ("CPU0: Initializing MMU... done", 0.3),
        ("CPU1: Initializing MMU... done", 0.3),
        ("SCSI subsystem initialized", 0.4),
        ("eth0: link up, 100Mbps full duplex", 0.4),
        ("Mounting root filesystem (/dev/sda1)... done", 0.6),
        ("Starting inetd, syslogd, and sshd... done", 0.5),
        ("System ready.", 1.0),
        ("", 0.8)  # Empty line for spacing
    ]
    
    for message, delay in remaining_messages:
        if message:  # Only print non-empty messages
            post_text = Text(message)
            post_text.stylize("bold green")
            console.print(post_text)
        else:
            console.print()  # Print empty line
        time.sleep(delay)

def show_ascii_art():
    """Display the ASCII art logo"""
    ascii_art = [
        " ███            █████████   ███                                     ",
        "░░░███         ███░░░░░███ ░░░                                      ",
        "  ░░░███      ███     ░░░  ████   █████████ █████████████    ██████ ",
        "    ░░░███   ░███         ░░███  ░█░░░░███ ░░███░░███░░███  ███░░███",
        "     ███░    ░███    █████ ░███  ░   ███░   ░███ ░███ ░███ ░███ ░███",
        "   ███░      ░░███  ░░███  ░███    ███░   █ ░███ ░███ ░███ ░███ ░███",
        " ███░         ░░█████████  █████  █████████ █████░███ █████░░██████ ",
        "░░░            ░░░░░░░░░  ░░░░░  ░░░░░░░░░ ░░░░░ ░░░ ░░░░░  ░░░░░░  "
    ]
    
    # Define gradient colors (from top to bottom)
    colors = ["#cca7df", "#b68cd4", "#b78cd7", "#9174b6", "#684d8f", "#503973", "#302d54", "#1c1444"]
    
    for i, line in enumerate(ascii_art):
        text = Text(line)
        text.stylize(f"bold {colors[i]}")
        console.print(text)

def show_welcome_selection():
    """Show the welcome selection menu after ASCII art"""
    welcome_text = """
                   ╔════════════════════════════════════╗
                   ║   ʕ•ᴥ•ʔ Your local ai assistant.   ║
                   ║  --------------------------------  ║
                   ║    1. Show startup each time.      ║
                   ║    2. Skip startup in future.      ║
                   ╚════════════════════════════════════╝
    """
    print(welcome_text)
    
    while True:
        choice = routines.input('').strip()
        if choice in ['1', '2']:
            return choice
        else:
            print("Please enter 1 or 2.")

def get_startup_preference():
    """Get user preference for startup behavior"""
    settings_path = Path("model/setting.txt")
    
    # If devmode is on, always show the selection menu (ignore saved settings)
    if devmode:
        show_ascii_art()
        choice = show_welcome_selection()
        return choice
    
    # Check if setting already exists
    if settings_path.exists():
        try:
            with open(settings_path, 'r') as f:
                setting = f.read().strip()
                if setting in ['1', '2']:
                    return setting
        except:
            pass  # If file is corrupted, ask again
    
    # First time - show ASCII art then selection
    show_ascii_art()
    choice = show_welcome_selection()
    
    # Save the preference
    settings_path.parent.mkdir(exist_ok=True)
    with open(settings_path, 'w') as f:
        f.write(choice)
    
    return choice

def startup():
    """Main startup function that handles user preferences"""
    preference = get_startup_preference()
    
    if preference == '1':
        # Show full startup sequence
        post_sequence()
        time.sleep(4)
        os.system('cls' if os.name == 'nt' else 'clear')
        time.sleep(2)
        show_ascii_art()
        print("\nStarting Gizmo...")
        time.sleep(2)
    elif preference == '2':
        # Skip startup sequence entirely
        print("Starting Gizmo...")
        time.sleep(1)
    
    # Clear screen one final time before starting main program
    os.system('cls' if os.name == 'nt' else 'clear')

import os
import sys
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# Configuration
openai = False
openai_model = "gpt-3.5-turbo"

system_prompt_path = Path("model/system.txt")
system_prompt = system_prompt_path.read_text(encoding="utf-8")
skills_prompt_path = Path("model/skills.txt")
openai_api_key = ''
ollama_agent = OllamaAgent("ʕ•ᴥ•ʔ Gizmo", "gizmo")
openai_agent = OpenAiAgent("ʕ•ᴥ•ʔ Gizmo", openai_model, system_prompt=system_prompt, api_token=openai_api_key)
agent = ollama_agent

# Global state - need to accumulate chunks to detect split emoji
stream_state = {"stream": "true", "buffer": ""}

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
            manager(f"ʕ•ᴥ•ʔ Successfully connected to {successful_connections} MCP server(s)")
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
    """Enhanced streaming callback that buffers and cleans output"""
    
    manager(f"🔧 [DEBUG] Streaming callback received chunk: {repr(chunk[:50])}...")
    
    # Add chunk to buffer for pattern detection
    stream_state["buffer"] += chunk
   
    # Check for tool call pattern in accumulated buffer
    if "⚡️" in stream_state["buffer"] and stream_state["stream"] == "true":
        stream_state["stream"] = "false"
        
        # Clear the current line to remove any printed lightning bolt
        print("\r\033[K", end="")  # Clear current line
        
        # Extract and print only the clean part before the tool call
        clean_part = stream_state["buffer"].split("⚡️")[0].strip()
        if clean_part:
            print(clean_part, end="", flush=True)
        
        manager(f"\n🔧 [Tool Call Detected - Processing...]")
        return chunk
    
    # Only print if streaming is still active
    if stream_state["stream"] == "true":
        print(chunk, end="", flush=True)
    
    return chunk

def incorporate_tool_results(original_request="", partial_response="", tool_result="Tool Failed. Just tell me that and suggest alternatives if you can."):
    """Continue the conversation with tool results incorporated"""
    
    manager(f"\n🔧 [Tool Complete - Resuming Stream...]")
    
    continuation_prompt = f"""The user asked: {original_request}

You started to answer: {partial_response}

Tool result: {tool_result}

Continue your response naturally, incorporating this tool result. Don't repeat what you already said, just continue from where you left off with the new information."""
    
    # Simple callback that just prints - no streaming control needed for continuation
    def continuation_callback(chunk: str):
        print(chunk, end="", flush=True)
        return chunk
    
    Task(continuation_prompt, agent, streaming_callback=continuation_callback).solve()

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
    if "⚡️" in content_str and stream_state["stream"] == "true":
        manager(f"🔧 [DEBUG] Found ⚡️ in final response, forcing tool execution")
        stream_state["stream"] = "false"
    
    if not mcp_manager or stream_state["stream"] == "true":
        manager(f"🔧 [DEBUG] Skipping tool execution - mcp_manager: {mcp_manager is not None}, stream stopped: {stream_state['stream'] == 'false'}")
        return
    
    try:
        manager(f"🔧 [DEBUG] Content string: {repr(content_str)}")
        tool_name, arguments = parse_tool_call(content_str)
        if tool_name:
            cprint(f"ʕ•ᴥ•ʔ Using {tool_name}...", attrs=["bold"])
            result = mcp_manager.call_tool(tool_name, arguments)
            manager(f"🔧 Result: {result}")
            incorporate_tool_results(original_request, content_str, str(result))
        else:
            manager(f"🔧 [DEBUG] No tool found in content")
    except Exception as e:
        manager(f"🔧 Tool execution failed: {str(e)}")
        manager(f"🔧 [DEBUG] Exception: {e}")
        incorporate_tool_results(original_request, content_str, "Tool Failed. Just tell me that and suggest alternatives if you can.")

def parse_tool_call(content: str):
    """Parse tool call syntax like: ⚡️tool_name({...json...})"""
    manager(f"🔧 [DEBUG] Parsing content for tool calls...")
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

# Run the startup sequence based on user preference

# Main execution
dbclear()
manager()
set_agent()
startup()
cprint('ʕ•ᴥ•ʔอ… Gizmo', 'yellow', attrs=["bold"])
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
    stream_state["buffer"] = ""
    
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