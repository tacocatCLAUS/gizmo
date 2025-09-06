import time
import os
import subprocess
import sys
import warnings
from pathlib import Path
from rich.console import Console
from rich.text import Text
from survey import routines
from termcolor import colored, cprint
from yacana import Task, OllamaAgent, OpenAiAgent, LoggerManager

# Comprehensive warning suppression for asyncio cleanup issues
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*Event loop is closed.*")
warnings.filterwarnings("ignore", message=".*I/O operation on closed pipe.*")
warnings.filterwarnings("ignore", message=".*unclosed transport.*")
warnings.filterwarnings("ignore", message=".*unclosed.*")

# Set environment variable to suppress asyncio debug
os.environ["PYTHONWARNINGS"] = "ignore::ResourceWarning,ignore::RuntimeWarning"
os.environ["PYTHONASYNCIODEBUG"] = "0"

# Redirect stderr to suppress ignored exceptions during cleanup
import io
from contextlib import redirect_stderr

class ErrorFilter:
    """Filter to suppress specific error messages based on devmode"""
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.suppress_patterns = [
            "Exception ignored in",
            "RuntimeError: Event loop is closed",
            "ValueError: I/O operation on closed pipe",
            "ResourceWarning",
            "unclosed transport",
            "<function BaseSubprocessTransport.__del__",
            "<function _ProactorBasePipeTransport.__del__",
            "Traceback (most recent call last)",
            "File \"C:\\Program Files\\WindowsApps\\",
            "at 0x",
            "self.close()",
            "proto.pipe.close()",
            "self._loop.call_soon",
            "self._check_closed()",
            "info.append",
            "self._sock.fileno()",
            "asyncio",
            "raise RuntimeError",
            "raise ValueError",
            "Event loop is closed",
            "I/O operation on closed pipe",
            "^^^",  # Error indication symbols
            "^^^^^",  # More error indication symbols
        ]
        self.in_traceback = False
        self.devmode = False  # Will be set after config is loaded
    
    def set_devmode(self, devmode):
        """Set devmode after config is loaded"""
        self.devmode = devmode
    
    def write(self, text):
        # If devmode is off, suppress all asyncio-related errors
        if not self.devmode:
            lines = text.split('\n')
            filtered_lines = []
            
            for line in lines:
                # Special handling for bear emoji messages with error prefixes
                if "ʕ•ᴥ•ʔ" in line and ("RuntimeError:" in line or "ValueError:" in line):
                    # Extract just the bear message part
                    bear_message = line[line.find("ʕ•ᴥ•ʔ"):]
                    filtered_lines.append(bear_message)
                    continue
                
                # Check if this line should be suppressed
                should_suppress = any(pattern in line for pattern in self.suppress_patterns)
                
                # Track if we're in a traceback
                if "Traceback (most recent call last)" in line:
                    self.in_traceback = True
                    should_suppress = True
                elif self.in_traceback and (line.strip() == "" or not line.startswith(" ") and not line.startswith("  ")):
                    # End of traceback
                    self.in_traceback = False
                elif self.in_traceback:
                    should_suppress = True
                    
                # Also suppress standalone error messages and concatenated errors
                if ("RuntimeError" in line and "Event loop is closed" in line) or \
                   ("ValueError" in line and "I/O operation on closed pipe" in line) or \
                   line.strip().startswith("^") or \
                   line.strip() == "RuntimeError:" or \
                   line.strip() == "ValueError:" or \
                   line.strip().startswith("RuntimeError:") or \
                   line.strip().startswith("ValueError:") or \
                   (line.count("RuntimeError:") > 1) or \
                   (line.count("ValueError:") > 1) or \
                   ("RuntimeError:" in line and "ValueError:" in line):
                    should_suppress = True
                    
                if not should_suppress:
                    filtered_lines.append(line)
            
            if filtered_lines:
                filtered_text = '\n'.join(filtered_lines)
                if filtered_text.strip():  # Only write if there's actual content
                    self.original_stderr.write(filtered_text)
        else:
            # In devmode, show everything
            self.original_stderr.write(text)
    
    def flush(self):
        self.original_stderr.flush()
    
    def fileno(self):
        return self.original_stderr.fileno()

# Install the error filter
sys.stderr = ErrorFilter(sys.__stderr__)
sys.stdout = ErrorFilter(sys.__stdout__)
from langchain_chroma import Chroma
from RAG.populate_database import parse, clear_database
from Libraries.filepicker import select_file
from Libraries.voicehandling import clean_function_text
from Libraries.svu import serverupdate
from Libraries.config_manager import ConfigManager
from Libraries.lagcleaner import clear_vram_and_reset as vramclear
from voice.f5 import f5
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
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
config_manager = ConfigManager()
config = config_manager.load_config()

# Config...uration
devmode = config.get('devmode', False)
db_clear = config.get('db_clear', True) 
use_mcp = config.get('use_mcp', True)
voice = config.get('voice', False)
openai = config.get('openai', False)
openai_model = config.get('openai_model', "gpt-3.5-turbo")
hc = config.get('hc', False)
hc_model = config.get('hc_model', "meta-llama/llama-4-maverick-17b-128e-instruct")
rag_model = config.get('rag_model', "ollama")
openai_api_key = config.get('openai_api_key', "")

# Set devmode in the error filter after config is loaded
sys.stderr.set_devmode(devmode)
sys.stdout.set_devmode(devmode)


# Use API key from config or environment variable - don't hardcode sensitive data
# openai_api_key is already loaded from config above

system_prompt_path = Path("model/system.txt")
system_prompt = system_prompt_path.read_text(encoding="utf-8")
skills_prompt_path = Path("model/skills.txt")
ollama_agent = OllamaAgent("ʕ•ᴥ•ʔ Gizmo", "gizmo")
hc_agent = OpenAiAgent("ʕ•ᴥ•ʔ Gizmoʰᶜ", hc_model, system_prompt, endpoint="https://ai.hackclub.com", api_token="leave blank")
openai_agent = OpenAiAgent("ʕ•ᴥ•ʔ Gizmoᴳᴾᵀ", openai_model, system_prompt, api_token=openai_api_key)
agent = ollama_agent

# Global state - need to accumulate chunks to detect split emoji
stream_state = {"stream": "true", "buffer": ""}

# Variable Declarations

db_query = False
addfile = 'N'
file_path = None  # Initialize to prevent NameError
filename = None   # Initialize to prevent NameError
message = {'content': ''}
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
    global devmode
    serverupdate(devmode)
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
        self.shutdown_event = threading.Event()
        self.tools: list[Any] = []
        self.loop = None
        self.session = None
        self.read_transport = None
        self.write_transport = None
        self.thread = threading.Thread(target=self._run_background, daemon=True)
        self.thread.start()

    def _run_background(self):
        """Run asyncio in a separate thread with proper cleanup"""
        try:
            # Create a new event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # Run the async coroutine
            self.loop.run_until_complete(self._async_run())
            
        except Exception as e:
            manager(f"Background thread error for {self.server_config.name}: {e}")
            self.initialized.set()
        finally:
            # Cleanup event loop properly
            if self.loop and not self.loop.is_closed():
                try:
                    # Cancel all running tasks
                    pending = asyncio.all_tasks(self.loop)
                    for task in pending:
                        task.cancel()
                    
                    if pending:
                        # Wait for tasks to be cancelled
                        self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    
                    # Close the loop
                    self.loop.close()
                except Exception as cleanup_error:
                    manager(f"Loop cleanup error for {self.server_config.name}: {cleanup_error}")

    async def _async_run(self):
        """Main async function with proper resource management"""
        session = None
        read_stream = None
        write_stream = None
        
        try:
            async with stdio_client(self.server_params) as (read, write):
                read_stream, write_stream = read, write
                
                async with ClientSession(read, write) as session:
                    self.session = session
                    manager(f"Initializing MCP session for {self.server_config.name}...")
                    
                    await session.initialize()
                    
                    manager(f"Listing available tools for {self.server_config.name}...")
                    tools_result = await session.list_tools()
                    self.tools = tools_result.tools
                    manager(f"Found {len(self.tools)} tools in {self.server_config.name}: {[tool.name for tool in self.tools]}")
                    self.initialized.set()

                    # Main message loop
                    while not self.shutdown_event.is_set():
                        try:
                            tool_name, arguments = self.request_queue.get(timeout=0.1)
                            
                            if tool_name is None:  # Shutdown signal
                                break
                                
                            result = await session.call_tool(tool_name, arguments)
                            self.response_queue.put(result)
                            
                        except queue.Empty:
                            continue
                        except Exception as e:
                            self.response_queue.put(f"Error: {str(e)}")
                            
        except asyncio.CancelledError:
            manager(f"MCP session for {self.server_config.name} was cancelled")
        except Exception as e:
            manager(f"MCP Session Error for {self.server_config.name}: {str(e)}")
            self.response_queue.put(f"MCP initialization error: {str(e)}")
        finally:
            # Ensure initialization event is set even on error
            self.initialized.set()
            
            # Clean up session reference
            self.session = None
            
            manager(f"Async session cleanup completed for {self.server_config.name}")

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if not self.initialized.wait(timeout=30):
            raise TimeoutError(f"MCP session for {self.server_config.name} did not initialize in time.")
        self.request_queue.put((tool_name, arguments))
        result = self.response_queue.get()
        return result

    def shutdown(self):
        """Shutdown the MCP connection with proper cleanup"""
        try:
            # Signal shutdown to the async loop
            self.shutdown_event.set()
            
            # Send shutdown signal to the request queue
            self.request_queue.put((None, None))
            
            # Wait for the thread to finish gracefully
            self.thread.join(timeout=10.0)
            
            if self.thread.is_alive():
                manager(f"Warning: MCP session {self.server_config.name} thread did not shut down gracefully")
                # Force cleanup if thread is still alive
                if self.loop and not self.loop.is_closed():
                    try:
                        # Try to close the loop from here if possible
                        pending = asyncio.all_tasks(self.loop)
                        for task in pending:
                            task.cancel()
                    except Exception as force_cleanup_error:
                        manager(f"Force cleanup error for {self.server_config.name}: {force_cleanup_error}")
                        
        except Exception as e:
            manager(f"Error during MCP session {self.server_config.name} shutdown: {e}")
        finally:
            # Clear any remaining references
            self.session = None
            self.loop = None
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
    if openai:
        agent = openai_agent
    elif hc:
        agent = hc_agent
    else:
        agent = ollama_agent

def voicecheck():
    if voice == True and stream_state["stream"] == "true":
        # Handle both dict and OllamaTextMessage object types
        if hasattr(message, 'content'):
            vctxt = clean_function_text(message.content)
        elif isinstance(message, dict) and 'content' in message:
            vctxt = clean_function_text(message['content'])
        else:
            # Fallback - convert to string
            vctxt = clean_function_text(str(message))
        print("\n")
        f5(vctxt)


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
    
    # Simple callback that just prints - no streaming control needed for continuation bad implementation...
#    def continuation_callback(chunk: str):
#       print(chunk, end="", flush=True)
#      return chunk
    stream_state["stream"] = "true"
    stream_state["buffer"] = ""  # Clear the buffer to prevent old ⚡️ patterns from interfering
    message = Task(continuation_prompt, agent, streaming_callback=streaming_callback).solve()
    voicecheck()

def query_rag(request):
    embedding_function = get_embedding_function(rag_model)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(request, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=request)
    message = Task(prompt, agent, streaming_callback=streaming_callback).solve()
    response_text = message
    sources = [doc.metadata.get("id", None) for doc, _ in results]
    formatted_response = f"\nSources: {sources}"
    if stream_state["stream"] == "true":
        print(formatted_response)
    voicecheck()
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
            try:
                result = mcp_manager.call_tool(tool_name, arguments)
                manager(f"🔧 Result: {result}")
                incorporate_tool_results(original_request, content_str, str(result))
            except  Exception as tool_error:
                manager(f"🔧 Tool execution failed: {str(tool_error)}")
                # Try the same tool call again but with failure context
                failure_context = f"The tool call failed with error: {str(tool_error)}. Suggest alternatives."
                stream_state["stream"] = "true"  # Reset stream state for retry
                incorporate_tool_results(original_request, content_str, failure_context)
        else:
            manager(f"🔧 [DEBUG] No tool found in content")
            failure_context = f"The tool call failed with error: Tool not found. Suggest alternatives."
            stream_state["stream"] = "true"  # Reset stream state for retry
            incorporate_tool_results(original_request, content_str, failure_context)
    except Exception as e:
        manager(f"🔧 Tool execution failed: {str(e)}")
        manager(f"🔧 [DEBUG] Exception: {e}")
        failure_context = f"Tool call failed with error: {str(e)}. Please suggest alternatives."
        incorporate_tool_results(original_request, content_str, failure_context)

def parse_tool_call(content: str):
    """Parse tool call syntax like: ⚡️tool_name({...json...})"""
    manager(f"🔧 [DEBUG] Parsing content for tool calls...")
    # Updated pattern to include hyphens and other characters in tool names
    pattern = r"⚡️([a-zA-Z0-9_-]+)\s*\((\{.*?\})\)"
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
        manager(f"🔧 [DEBUG] No tool call pattern found in: {repr(content)}")
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
cprint('ʕ•ᴥ•ʔอ… Gizmo', 'yellow', attrs=["bold"])
message = Task("I have no questions. introduce yourself. dont mention your skills at all. be breif.", agent, streaming_callback=streaming_callback).solve()
voicecheck()

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
            if rag_model != "none":
                parse()
            filename = Path(file_path).name
            print(f"ʕ•ᴥ•ʔ I processed {filename}")
            addfile = 'N'
        else:
            cprint("Error.", 'red')
            manager("[SYSTEM] Error. No path added by user/library.")
    else:
        db_query = False
    print('\n')
    cprint('ʕ•ᴥ•ʔ Gizmo', 'yellow', attrs=["bold"])
    
    # Reset stream state before each request
    stream_state["stream"] = "true"
    stream_state["buffer"] = ""
    text_extensions = (
    ".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".xml", ".ini", ".log",
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".cs", ".html", ".htm",
    ".css", ".scss", ".go", ".rs", ".rb", ".php", ".sh", ".bat", ".toml"
)

    
    if db_query == True and rag_model != "none" and file_path and file_path.endswith((".txt", ".pdf")):
        message = query_rag(request)
        content_str = message.content if hasattr(message, 'content') else str(message)
        handle_tool_execution(content_str, mcp_manager, request)
    elif (db_query == True and rag_model == "none") or (file_path and file_path.endswith((text_extensions))):
        if filename:  # Only proceed if filename is available
            context_text = os.path.join(os.getcwd(), "RAG", "data", filename)
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=request)
            message = Task(prompt, agent, streaming_callback=streaming_callback).solve()
            voicecheck()
            content_str = message.content if hasattr(message, 'content') else str(message)
            handle_tool_execution(content_str, mcp_manager, request)
        else:
            # Fallback to regular processing if filename is not available
            message = Task(request, agent, streaming_callback=streaming_callback).solve()
            voicecheck()
            content_str = message.content if hasattr(message, 'content') else str(message)
            handle_tool_execution(content_str, mcp_manager, request)
    else:
        message = Task(request, agent, streaming_callback=streaming_callback).solve()
        voicecheck()
        content_str = message.content if hasattr(message, 'content') else str(message)
        handle_tool_execution(content_str, mcp_manager, request)
if mcp_manager:
    mcp_manager.shutdown_all()

vramclear(devmode)