import os
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

from yacana import Task, OllamaAgent, OpenAiAgent, LoggerManager
from pathlib import Path
from ScrapeSearchEngine.ScrapeSearchEngine import Duckduckgo
from tavily import TavilyClient
from ollama import chat
from ollama import ChatResponse
from langchain.prompts import ChatPromptTemplate
from termcolor import colored, cprint
from survey import routines
from langchain_chroma import Chroma
from RAG.populate_database import parse, clear_database
from filepicker import select_file
from RAG.get_embedding_function import get_embedding_function
import shutil

# Configuration
openai = True # Use OpenAI instead of Ollama model.
devmode = False  # Set to True for development mode, False for production
db_clear = True  # Set to True to clear the database on startup, False to keep it persistent
tavily_api = True  # Set to True to use Tavily API, False to use DuckDuckGo

# API's etc.
userAgent = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Safari/605.1.15')
openai_api_key = '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'
tavily_api_key = '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'

# Set Variables.
system_prompt_path = Path("setup/system.txt")
system_prompt = system_prompt_path.read_text()
skills_prompt_path = Path("setup/skills.txt")
skills = skills_prompt_path.read_text()
system_prompt = system_prompt + "\n\n" + skills  # Optional spacing between the two
ollama_agent = OllamaAgent("ʕ•ᴥ•ʔ Gizmo", "gizmo")
openai_agent = OpenAiAgent("ʕ•ᴥ•ʔ Gizmo", "gpt-3.5-turbo", system_prompt=system_prompt, api_token=openai_api_key)   
agent = ollama_agent
client = TavilyClient(tavily_api_key)
stream_state = {"stream": "true"}
final_request = ""
db_query = False
api_state = {"api": "true"}
addfile = 'N'
CHROMA_PATH = "RAG/chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
[File]
{context}

---

Answer the question based on the above context: {question}
"""

def dbclear():
    if db_clear == True:
        clear_database()
    else: 
        cprint('ʕ•ᴥ•ʔ Persistent memory is on.', 'yellow', attrs=["bold"])

def manager(message=None):
    if devmode == False:
        LoggerManager.set_log_level(None)
    else:
        if message is not None:
            print(message)

def tavily():
    if tavily_api == True:
        api_state = {"api": "true"}
        manager('[SYSTEM] Using Tavily API.')
    else:
        api_state = {"api": "false"}
        manager('[SYSTEM] Using DuckDuckGo instead.')

def openai(): # This function is used to set the agent to OpenAI.
    if openai == True:
        agent = openai_agent
        manager('[SYSTEM] OpenAI agent selected...')
    else :
        agent = ollama_agent

def streaming(chunk: str): # This function is used to stream the response from the agent. And stop if the agent performs a web request.
    if "し" in chunk:
        stream_state["stream"] = "false"
        manager('[SYSTEM] web request received...')
        return
    else:
        if stream_state["stream"] == "true":
            print(f"{chunk}", end="", flush=True)
        else:
            return
        
def web(content):
    if stream_state["stream"] == "false":
        print("searching web...")
        manager(f"[SYSTEM] Web search: {content}")
        # Split the content on the pipe symbol and strip any extra whitespace
        parts = [part.strip() for part in content.split("|")]
        if len(parts) < 4:
            manager("[SYSTEM] Not enough parts in content for web search. Skipping.")
            cprint('Error. Please try again.', 'red', attrs=['bold'])
            return
        # Assign variables based on their position
        search_1 = parts[1]
        primary_search = parts[2]
        search_2 = parts[3]
        if api_state["api"] == "true":
            links_1 = ''
            summarize = f'Sumarrize this data and make it breif while still containing the most information you can. Dont mention anything about you summarizing only give the summary. Only include a summary:{client.search(query=primary_search, max_results=2,include_answer="basic")}'
            print('ʕ•ᴥ•ʔ I am fetching the api...')
            response: ChatResponse = chat(model='gemma3:1b', messages=[
            {
                'role': 'user',
                'content': summarize
            },
            ])
            links_3 = response['message']['content']
            print('ʕ•ᴥ•ʔ I am summarizing...')
        else:
            summarize = f'Sumarrize this data and make it breif while still containing the most information you can. Dont mention anything about you summarizing only give the summary. Only include a summary: {Duckduckgo(search_1, userAgent)}'
            print('ʕ•ᴥ•ʔ I am scraping the web...')
            response: ChatResponse = chat(model='gemma3:1b', messages=[
            {
                'role': 'user',
             'content': summarize
         },
          ])
        links_1 = response['message']['content']
        links_3 = ''
        print(f'{links_1}{links_3}')
        stream_state["stream"] = "true"      
        # final_request = f"し original question: {request} use this data: {links_1} {links_3}"
        # message = Task(final_request, ollama_agent, streaming_callback=streaming).solve() --- llm considers the summary of the web search as the original question very grueling and annoying
    else:
        return
    
def query_rag(request):
    # Prepare the DB.
    embedding_function = get_embedding_function(openai)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(request, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=request)
    # print(prompt)
    response_text = Task(prompt, agent, streaming_callback=streaming).solve()
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"\nSources: {sources}"
    print(formatted_response)
    return response_text

# original question
dbclear()
manager()
cprint('ʕ•ᴥ•ʔฅ Gizmo', 'yellow', attrs=["bold"])
message = Task("I have no questions. introduce yourself. dont mention your skills at all. be breif.", agent, streaming_callback=streaming).solve()
# second question
while True:
    print('\n')
    cprint('(•ᴗ•) You', 'blue', attrs=["bold"])
    request = routines.input()
    if request.strip().lower() == "bye":
        break
    addfile = routines.input('📄 (Y/N): ')
    if addfile == 'Y':
        file_path = select_file()
        db_query = True # Set db_query to True if a file is added
        if file_path:
            # Use the rag folder in the current directory
            dest_dir = os.path.join(os.getcwd(), "RAG", "data")
            # Create the destination directory if it doesn't exist
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(file_path, dest_dir)
            parse()
            filename = Path(file_path).name
            print(f"ʕ•ᴥ•ʔ I processed {filename}")
            addfile = 'N'  # Reset addfile to 'N' after processing  
        else:
            cprint("Error.", 'red')
            manager("[SYSTEM] Error. No path added by user/library.")
    print('\n')
    cprint('ʕ•ᴥ•ʔ Gizmo', 'yellow', attrs=["bold"])
    if db_query:
        # Use the database query logic for every question
        message = query_rag(request)
        web(message.content)
    else:
        # Use your normal logic
        message = Task(request, agent, streaming_callback=streaming).solve()
        web(message.content)

