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
from RAG.get_embedding_function import get_embedding_function
from langchain.prompts import ChatPromptTemplate
import shutil
import asyncio
import threading
import queue
from typing import Any
from pydantic import BaseModel, Field, create_model
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json
import re

# Configuration
openai = False
openai_model = "gpt-3.5-turbo"
devmode = True
db_clear = True
use_mcp = True

openai_api_key = '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'

system_prompt_path = Path("setup/system.txt")
system_prompt = system_prompt_path.read_text(encoding="utf-8")
skills_prompt_path = Path("setup/skills.txt")
skills = skills_prompt_path.read_text(encoding="utf-8")
system_prompt = system_prompt + "\n\n" + skills
ollama_agent = OllamaAgent("ʕ•ᴥ•ʔ Gizmo", "gizmo")
openai_agent = OpenAiAgent("ʕ•ᴥ•ʔ Gizmo", openai_model, system_prompt=system_prompt, api_token=openai_api_key)
agent = ollama_agent
stream_state = {"stream": "active"}
final_request = ""
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

class OllamaMCP:
    def __init__(self, server_params: StdioServerParameters):
        self.server_params = server_params
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
                    manager("Initializing MCP session...")
                    await session.initialize()
                    self.session = session
                    manager("Listing available tools...")
                    tools_result = await session.list_tools()
                    self.tools = tools_result.tools
                    manager(f"Found {len(self.tools)} tools: {[tool.name for tool in self.tools]}")
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
            print("MCP Session Initialization Error:", str(e))
            self.initialized.set()
            self.response_queue.put(f"MCP initialization error: {str(e)}")

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if not self.initialized.wait(timeout=30):
            raise TimeoutError("MCP session did not initialize in time.")
        self.request_queue.put((tool_name, arguments))
        result = self.response_queue.get()
        return result

    def shutdown(self):
        self.request_queue.put((None, None))
        self.thread.join()
        print("MCP session shut down.")

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

def streaming(chunk: str):
    if "し" in chunk:
        stream_state["stream"] = "paused"
        manager(f"\n🔧 [Tool Call Detected - Processing...]", flush=True)
        return chunk
    if stream_state.get("stream", "active") == "active":
        print(f"{chunk}", end="", flush=True)
    return chunk

def detect_mcp_call(chunk: str) -> bool:
    # Just check for the presence of し anywhere
    return "し" in chunk

def resume_streaming(contextual_response="", contextual_request="", result="Tool Failed. Just tell me that and suggest alternatives if you can."):
    # reprompt with both result and old prompt
    stream_state["stream"] = "active"
    manager(f"\n🔧 [Tool Complete - Resuming...]", flush=True)
    Task(f"The user asked this question: {contextual_request} and you answered like this: {contextual_response} with this information: {result} continue where you stopped and answer with this new information and dont perform another web search. YOU MUST ANSWER IN THE ENGLISH LANGUAGE AND ONLY THE ENGLISH LANGUAGE!", agent, streaming_callback=streaming).solve()


def query_rag(request):
    embedding_function = get_embedding_function(openai)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(request, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=request)
    response_text = Task(prompt, agent, streaming_callback=streaming).solve()
    sources = [doc.metadata.get("id", None) for doc, _ in results]
    formatted_response = f"\nSources: {sources}"
    if stream_state.get("stream", "active") == "active":
        print(formatted_response)
    return response_text

def handle_tool_execution(response_content, mcp_client):
    contextual_response = response_content
    contextual_request = request
    if not mcp_client or stream_state.get("stream") != "paused":
        return
    try:
        tool_name, arguments = parse_tool_call(response_content)
        if tool_name:
            cprint(f"ʕ•ᴥ•ʔ Using {tool_name}...", 'yellow', attrs=["bold"])
            result = mcp_client.call_tool(tool_name, arguments)
            manager(f"🔧 Result: {result}")
            resume_streaming(contextual_response, contextual_request, result)
    except Exception as e:
        print(f"🔧 Tool execution failed: {str(e)}")
        resume_streaming(contextual_response, contextual_request, result)

def parse_tool_call(content: str):
    """
    Parse tool call syntax like:
    しtool_name({...json...})
    """
    pattern = r"し(\w+)\s*\((\{.*?\})\)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        tool_name = match.group(1)
        json_str = match.group(2)
        try:
            arguments = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"🔧 JSON parsing error in tool call: {e}")
            arguments = {}
        return tool_name, arguments
    return None, {}

# Initialize MCP if enabled
mcp_client = None
if use_mcp:
    try:
        server_parameters = StdioServerParameters(
            command=sys.executable,
            args=["mcp-server.py"],
            cwd=str(Path.cwd())
        )
        mcp_client = OllamaMCP(server_parameters)
        if mcp_client.initialized.wait(timeout=30):
            print(f"ʕ•ᴥ•ʔ MCP tools available: {[tool.name for tool in mcp_client.tools]}")
        else:
            print("ʕ•ᴥ•ʔ MCP initialization timed out, continuing without MCP...")
            mcp_client = None
    except Exception as e:
        print(f"ʕ•ᴥ•ʔ MCP initialization failed: {str(e)}, continuing without MCP...")
        mcp_client = None

# Main execution
dbclear()
manager()
set_agent()
cprint('ʕ•ᴥ•ʔฅ Gizmo', 'yellow', attrs=["bold"])
Task("I have no questions. introduce yourself. dont mention your skills at all. be breif.", agent, streaming_callback=streaming).solve()

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
    if db_query:
        message = query_rag(request)
        handle_tool_execution(message.content, mcp_client)
    else:
        message = Task(request, agent, streaming_callback=streaming).solve()
        handle_tool_execution(message.content, mcp_client)

if mcp_client:
    mcp_client.shutdown()
