from langchain_ollama import OllamaEmbeddings
from langchain_aws import BedrockEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_function(openai):
    if openai == True:
        # tbd
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    else:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings


# embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
