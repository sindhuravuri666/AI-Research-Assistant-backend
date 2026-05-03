import requests
from typing import List, Optional, Dict
from app.core.config import OLLAMA_HOST, OLLAMA_EMBED_MODEL, OLLAMA_LLM_MODEL


class OllamaClientError(Exception):
    pass


class OllamaClient:
    def __init__(self, host: str = OLLAMA_HOST):
        self.host = host.rstrip("/")

    def _request(self, path: str, json: Dict, timeout: int = 60) -> Dict:
        try:
            response = requests.post(f"{self.host}{path}", json=json, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            raise OllamaClientError(f"Ollama request failed: {exc}") from exc

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        payload = {
            "model": OLLAMA_EMBED_MODEL,
            "text": texts,
        }
        data = self._request("/embed", payload)
        if "embeddings" not in data:
            raise OllamaClientError("Embedding response did not include embeddings")
        return data["embeddings"]

    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 512) -> str:
        payload = {
            "model": OLLAMA_LLM_MODEL,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = self._request("/predict", payload)
        if "output" not in data:
            raise OllamaClientError("LLM response did not include output")
        if isinstance(data["output"], list):
            return "\n".join(str(item) for item in data["output"])
        return str(data["output"])

    def check_health(self) -> bool:
        try:
            response = requests.get(f"{self.host}/v1/models", timeout=10)
            response.raise_for_status()
            return True
        except Exception:
            return False
