from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

try:
    from langchain_core.embeddings import Embeddings
except Exception:  # pragma: no cover - compatibility fallback
    class Embeddings:  # type: ignore[no-redef]
        def embed_documents(self, texts):
            raise NotImplementedError

        def embed_query(self, text):
            raise NotImplementedError

from config.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL, OPENAI_API_KEY


class LocalTransformerEmbeddings(Embeddings):
    """Minimal local embedding wrapper for sentence-transformer style models."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        device: str | None = None,
        max_length: int | None = None,
        normalize: bool = True,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_name = self._resolve_model_name(model_name)
        self.batch_size = batch_size
        self.device = device or os.getenv("EMBEDDING_DEVICE") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.max_length = max_length or int(os.getenv("EMBEDDING_MAX_LENGTH", "512"))
        self.normalize = normalize

        model_path = Path(self.model_name)
        local_files_only = model_path.exists()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            local_files_only=local_files_only,
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_model_name(model_name: str) -> str:
        model_path = Path(model_name)
        if model_path.exists():
            return str(model_path.resolve())
        return model_name

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        safe_texts = [text if text and text.strip() else " " for text in texts]
        encoded = self.tokenizer(
            safe_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with self.torch.no_grad():
            output = self.model(**encoded)
            token_embeddings = output.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
            masked_embeddings = token_embeddings * attention_mask
            pooled = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
            if self.normalize:
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)

        return pooled.detach().cpu().numpy().astype("float32").tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            embeddings.extend(self._embed_batch(texts[start : start + self.batch_size]))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class EmbeddingService:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or EMBEDDING_MODEL
        self.embeddings = self._get_embeddings()

    def _get_embeddings(self):
        if self.model_name.startswith("text-embedding"):
            if not OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY is empty. Set EMBEDDING_MODEL=./m3e-small "
                    "or provide an OpenAI key for OpenAI embeddings."
                )
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                api_key=OPENAI_API_KEY,
                model=self.model_name,
            )

        return LocalTransformerEmbeddings(self.model_name)

    def embed_documents(self, documents: Iterable[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(list(documents))

    def embed_query(self, query: str) -> List[float]:
        return self.embeddings.embed_query(query)
