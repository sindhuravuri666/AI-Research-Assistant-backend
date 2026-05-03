from typing import List
from app.utils.ollama_client import OllamaClient


class EmbeddingService:
    def __init__(self, client: OllamaClient):
        self.client = client

    def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        return self.client.embed_texts(chunks)

    def embed_query(self, query: str) -> List[float]:
        embeddings = self.client.embed_texts([query])
        return embeddings[0]
