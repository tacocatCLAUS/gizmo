from langchain_ollama import OllamaEmbeddings
from langchain_aws import BedrockEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_function(openai):
    if openai == True:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    else:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings


# embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
