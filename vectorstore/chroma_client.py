import os
import requests
import chromadb
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

class SafeOllamaEmbeddings(Embeddings):
    """
    Wrapper around OllamaEmbeddings that intercepts connection errors and raises
    clear instructions on how to start Ollama and pull the models.
    """
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.embeddings = OllamaEmbeddings(model=model, base_url=base_url)

    def _handle_error(self, e: Exception):
        raise RuntimeError(
            "Ollama connection failed. Please ensure Ollama is running and the models are pulled.\n\n"
            "Instructions:\n"
            "1. Run `ollama serve` in a terminal window.\n"
            "2. Pull the required models:\n"
            "   `ollama pull nomic-embed-text`\n"
            "   `ollama pull llama3.1`"
        ) from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            self._handle_error(e)

    def embed_query(self, text: str) -> List[float]:
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            self._handle_error(e)

def get_chroma_vectorstore(persist_directory: str = "./chroma_db", collection_name: str = "codebase_rag") -> Chroma:
    """
    Returns a persistent Chroma DB client wrapped in LangChain's Chroma vector store.
    """
    # Get ollama host from env or default
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    embeddings = SafeOllamaEmbeddings(model="nomic-embed-text", base_url=ollama_host)
    
    # Instantiate persistent chromadb client
    client = chromadb.PersistentClient(path=persist_directory)
    
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )

def delete_file_from_vectorstore(vectorstore: Chroma, file_path: str):
    """
    Deletes all chunks associated with a specific file path from Chroma DB.
    """
    try:
        # Get all chunk ids for the given file path
        # Chroma's get allows filtering by metadata
        results = vectorstore.get(where={"file_path": file_path})
        if results and "ids" in results and results["ids"]:
            vectorstore.delete(ids=results["ids"])
            print(f"Deleted {len(results['ids'])} existing chunks for {file_path}")
    except Exception as e:
        print(f"Warning: Failed to delete old chunks for {file_path} from vectorstore: {e}")
