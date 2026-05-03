from typing import List, Dict
from app.storage.vector_store import LocalFAISSStore
from app.services.embedding_service import EmbeddingService
from app.models.schemas import Citation
from app.core.config import TOP_K


class RetrievalService:
    def __init__(self, vector_store: LocalFAISSStore, embedding_service: EmbeddingService):
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        embedding = self.embedding_service.embed_query(query)
        return self.vector_store.search(embedding, top_k=top_k)

    def retrieve_citations(self, query: str, top_k: int = TOP_K) -> List[Citation]:
        hits = self.retrieve(query, top_k=top_k)
        return [Citation(
            paper_id=hit["paper_id"],
            file_name=hit["file_name"],
            page_number=hit["page_number"],
            chunk_index=hit["chunk_index"],
            snippet=hit["text"][:320]
        ) for hit in hits]
