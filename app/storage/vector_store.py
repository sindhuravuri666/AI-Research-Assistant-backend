import pickle
from pathlib import Path
from typing import List, Dict, Optional

import faiss
import numpy as np
from app.core.config import INDEX_FILE, METADATA_FILE, EMBED_DIM


class LocalFAISSStore:
    def __init__(self):
        self.index_path = Path(INDEX_FILE)
        self.meta_path = Path(METADATA_FILE)
        self.index = None
        self.metadata = []
        self._load_or_create_index()

    def _load_or_create_index(self):
        if self.index_path.exists() and self.meta_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.meta_path, "rb") as handle:
                    self.metadata = pickle.load(handle)
                return
            except Exception:
                pass
        self.index = faiss.IndexFlatL2(EMBED_DIM)
        self.metadata = []

    def save(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "wb") as handle:
            pickle.dump(self.metadata, handle)

    def add_embeddings(self, embeddings: List[List[float]], metadatas: List[Dict]):
        arr = np.array(embeddings, dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.index.d:
            raise ValueError(f"Embedding size {arr.shape[1]} does not match index dimension {self.index.d}")
        self.index.add(arr)
        self.metadata.extend(metadatas)
        self.save()

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
        arr = np.array([query_embedding], dtype="float32")
        distances, indices = self.index.search(arr, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx].copy()
            item["distance"] = float(dist)
            results.append(item)
        return results

    def get_all_papers(self) -> List[Dict]:
        papers = {}
        for chunk in self.metadata:
            paper_id = chunk["paper_id"]
            if paper_id not in papers:
                papers[paper_id] = {
                    "paper_id": paper_id,
                    "file_name": chunk["file_name"],
                    "pages": chunk.get("pages", 0),
                    "chunks": 0,
                }
            papers[paper_id]["chunks"] += 1
        return list(papers.values())

    def get_paper_chunks(self, paper_id: str) -> List[Dict]:
        return [chunk for chunk in self.metadata if chunk["paper_id"] == paper_id]

    def needs_initialization(self) -> bool:
        return self.index is None or self.index.ntotal == 0
