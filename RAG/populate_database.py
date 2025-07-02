import os
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

import argparse
import shutil
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from .get_embedding_function import get_embedding_function
from langchain_chroma import Chroma

if os.path.exists("RAG/chroma"):
    CHROMA_PATH = "RAG/chroma"  # Called from project root
else:
    CHROMA_PATH = "chroma"      # Called from RAG directory
# Dynamic path detection - works when called from project root or RAG directory
if os.path.exists("RAG/data"):
    DATA_PATH = "RAG/data"  # Called from project root
else:
    DATA_PATH = "data"      # Called from RAG directory

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.lower().endswith(".pdf"):
            print("Found PDF:", os.path.join(root, file))

def parse():
    # Check if the database should be cleared (using the --reset flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Load and split PDFs
    pdf_documents = load_pdf()
    pdf_chunks = split_pdfs(pdf_documents)
    add_to_chroma(pdf_chunks)

    # Load and split TXTs
    txt_documents = load_txt()
    txt_chunks = split_txt(txt_documents)
    add_to_chroma(txt_chunks)

    # Load and split MDs
    md_documents = load_md()
    md_chunks = split_markdown(md_documents)
    add_to_chroma(md_chunks)

def load_pdf():
    # Use PyPDFDirectoryLoader directly on the data directory
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    docs = document_loader.load()
    print(f"Loaded {len(docs)} PDF documents")
    return docs

def load_txt():
    document_loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader)
    return document_loader.load()

def load_md():
    document_loader = DirectoryLoader(DATA_PATH, glob="**/*.md", loader_cls=TextLoader)
    return document_loader.load()

def split_pdfs(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} PDF chunks")
    return chunks

def split_txt(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def split_markdown(documents: list[Document]):
    # Use the same splitter as for txt and pdf for simplicity
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function(True)
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)

def calculate_chunk_ids(chunks):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        os.makedirs(CHROMA_PATH, exist_ok=True)
        if os.path.exists(DATA_PATH):
            shutil.rmtree(DATA_PATH)
            os.makedirs(DATA_PATH, exist_ok=True)

if __name__ == "__main__":
    parse()
    # If this file is run directly, populate the database.