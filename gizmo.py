from yacana import Task, OllamaAgent
from pathlib import Path
from ScrapeSearchEngine.ScrapeSearchEngine import Duckduckgo
from tavily import TavilyClient
from ollama import chat
from ollama import ChatResponse
from langchain.prompts import ChatPromptTemplate
from termcolor import colored, cprint
from log import manager
from survey import routines
from langchain_chroma import Chroma
from RAG.populate_database import parse, clear_database
from filepicker import select_file
from RAG.get_embedding_function import get_embedding_function
import shutil
import os


manager()

# system_prompt_path = Path("system.txt")
# system_prompt = system_prompt_path.read_text()

# Read from skills.txt
# skills_prompt_path = Path("skills.txt")
# skills = skills_prompt_path.read_text()

# Combine both texts
# combined_prompt = system_prompt + "\n\n" + skills  # Optional spacing between the two
userAgent = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Safari/605.1.15')
ollama_agent = OllamaAgent("ʕ•ᴥ•ʔ Gizmo", "gizmo")
client = TavilyClient("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")
db_clear = True
stream_state = {"stream": "true"}
final_request = ""
db_query = False
api_state = {"api": "true"}
addfile = 'N'
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
[File]
{context}

---

Answer the question based on the above context: {question}
"""

def db_clear():
    if db_clear == True:
        clear_database()
    else: 
        cprint('ʕ•ᴥ•ʔ Persistent memory is on.', 'yellow', attrs=["bold"])

def streaming(chunk: str):
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
    
def query_rag_gizmo(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    ollama_agent = OllamaAgent("ʕ•ᴥ•ʔ Gizmo", "gizmo")
    response_text = Task(prompt, ollama_agent, streaming_callback=streaming).solve()
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"\nSources: {sources}"
    print(formatted_response)
    return response_text
    


# original question
db_clear()
cprint('ʕ•ᴥ•ʔฅ Gizmo', 'yellow', attrs=["bold"])
message = Task("I have no questions. introduce yourself. dont mention your skills at all. be breif.", ollama_agent, streaming_callback=streaming).solve()
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
        if file_path:
            # Expand the tilde to the full home directory path
            dest_dir = os.path.expanduser("~/RAG/data/")
            shutil.copy(file_path, dest_dir)
            parse()
            filename = Path(file_path).name
            print(f"ʕ•ᴥ•ʔ I processed {filename}")
            addfile = 'N'  # Reset addfile to 'N' after processing  
            db_query = True 
        else:
            cprint("Error.", 'red')
            manager("[SYSTEM] Error. No path added by user/library.")
    print('\n')
    cprint('ʕ•ᴥ•ʔ Gizmo', 'yellow', attrs=["bold"])
    if db_query:
        # Use the database query logic for every question
        message = query_rag_gizmo(request)
        web(message.content)
    else:
        # Use your normal logic
        message = Task(request, ollama_agent, streaming_callback=streaming).solve()
        web(message.content)

