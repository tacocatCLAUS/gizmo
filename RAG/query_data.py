import os
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from yacana import Task, OllamaAgent
from get_embedding_function import get_embedding_function
import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now you can import your module
from log import manager

CHROMA_PATH = "chroma"
stream_state = {"stream": "true"}
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
[File]
{context}

---

Answer the question based on the above context: {question}
"""

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

def main():
    manager()
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    ollama_agent = OllamaAgent("ʕ•ᴥ•ʔ Gizmo", "gizmo")
    response_text = Task(prompt, ollama_agent, streaming_callback=streaming).solve()
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
