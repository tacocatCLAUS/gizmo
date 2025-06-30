from yacana import Task, OllamaAgent
from pathlib import Path
from ScrapeSearchEngine.ScrapeSearchEngine import Duckduckgo
from tavily import TavilyClient
from ollama import chat
from ollama import ChatResponse
from termcolor import colored, cprint
from log import manager
from survey import routines
import datetime

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

stream_state = {"stream": "true"}
final_request = ""
api_state = {"api": "true"}

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
        print('ʕ•ᴥ•ʔ I am summarizing...')
        links_3 = ''
        print(f'{links_1}{links_3}')
        stream_state["stream"] = "true"      
        # final_request = f"し original question: {request} use this data: {links_1} {links_3}"
        # message = Task(final_request, ollama_agent, streaming_callback=streaming).solve() --- llm considers the summary of the web search as the original question very grueling and annoying
    else:
        return

# original question
cprint('ʕ•ᴥ•ʔฅ Gizmo', (227, 118, 41), attrs=["bold"])
message = Task("I have no questions. introduce yourself. dont mention your skills at all. be breif.", ollama_agent, streaming_callback=streaming).solve()
# second question
while True:
    print('\n')
    cprint('(•ᴗ•) You', (12, 110, 176), attrs=["bold"])
    date = datetime.datetime.now()
    request = routines.input() + date.strftime('%x')
    if request.strip().lower() == "bye":
        break
    print('\n')
    cprint('ʕ•ᴥ•ʔ Gizmo', (227, 118, 41), attrs=["bold"])
    message = Task(request, ollama_agent, streaming_callback=streaming).solve()
    web(message.content)