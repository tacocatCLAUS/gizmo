from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

# from langchain_aws import BedrockEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_function(rag_model):
    if rag_model == "openai":
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    else:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings


# embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
