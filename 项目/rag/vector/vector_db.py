from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
from langchain_core.documents import Document

from config.config import (
    EMBEDDING_BATCH_SIZE,
    SCORE_THRESHOLD,
    TOP_K,
    VECTOR_DB_BACKEND,
    VECTOR_DB_PATH,
)
from rag.vector.embedding import EmbeddingService

try:
    from langchain_chroma import Chroma
except Exception:  # pragma: no cover - optional dependency
    Chroma = None


class SimpleVectorStore:
    """Small persistent cosine-similarity index used when Chroma is unavailable."""

    def __init__(self, persist_directory: str | Path, embedding_function):
        self.persist_directory = Path(persist_directory)
        self.index_file = self.persist_directory / "simple_index.jsonl"
        self.embedding_function = embedding_function
        self._rows: list[dict] = []
        self._embeddings: np.ndarray | None = None

    def reset(self) -> None:
        if self.persist_directory.exists():
            shutil.rmtree(self.persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._rows = []
        self._embeddings = None

    def load(self) -> "SimpleVectorStore":
        if not self.index_file.exists():
            raise FileNotFoundError(f"Vector index not found: {self.index_file}")

        rows: list[dict] = []
        vectors: list[list[float]] = []
        with self.index_file.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                item = json.loads(line)
                rows.append(
                    {
                        "text": item["text"],
                        "metadata": item.get("metadata", {}),
                    }
                )
                vectors.append(item["embedding"])

        self._rows = rows
        self._embeddings = self._normalize(np.asarray(vectors, dtype="float32"))
        return self

    def add_documents(
        self,
        documents: Sequence[Document],
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        with self.index_file.open("a", encoding="utf-8") as file:
            for start in range(0, len(documents), batch_size):
                batch = list(documents[start : start + batch_size])
                texts = [doc.page_content for doc in batch]
                embeddings = self.embedding_function.embed_documents(texts)

                for doc, embedding in zip(batch, embeddings):
                    item = {
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                        "embedding": embedding,
                    }
                    file.write(json.dumps(item, ensure_ascii=False) + "\n")

        self.load()

    def similarity_search_with_score(
        self,
        query: str,
        k: int = TOP_K,
    ) -> list[tuple[Document, float]]:
        if self._embeddings is None:
            self.load()

        assert self._embeddings is not None
        query_vector = np.asarray(self.embedding_function.embed_query(query), dtype="float32")
        query_vector = self._normalize(query_vector.reshape(1, -1))[0]
        # Avoid BLAS-backed matrix multiplication here. On some Windows
        # environments numpy matmul can conflict with torch-loaded runtimes.
        scores = np.sum(self._embeddings * query_vector.reshape(1, -1), axis=1)
        top_indices = np.argsort(scores)[::-1][:k]

        results: list[tuple[Document, float]] = []
        for index in top_indices:
            row = self._rows[int(index)]
            document = Document(page_content=row["text"], metadata=row["metadata"])
            results.append((document, float(scores[int(index)])))
        return results

    def get_documents_by_page(self, filename: str, page_number: int | str) -> list[Document]:
        if self._embeddings is None:
            self.load()

        page_text = str(page_number)
        documents: list[Document] = []
        for row in self._rows:
            metadata = row["metadata"] or {}
            row_filename = metadata.get("filename") or metadata.get("source")
            row_page = metadata.get("page_number") or metadata.get("page")
            if row_filename == filename and str(row_page) == page_text:
                documents.append(Document(page_content=row["text"], metadata=metadata))

        return sorted(
            documents,
            key=lambda doc: int(doc.metadata.get("chunk_id") or 0),
        )

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)


class VectorDB:
    def __init__(
        self,
        persist_directory: str | Path | None = None,
        backend: str | None = None,
        embedding_service: EmbeddingService | None = None,
    ):
        self.persist_directory = Path(persist_directory or VECTOR_DB_PATH)
        self.embedding_service = embedding_service or EmbeddingService()
        self.backend = self._resolve_backend(backend or VECTOR_DB_BACKEND)
        self.db = None

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        backend = (backend or "auto").lower()
        if backend == "auto":
            return "chroma" if Chroma is not None else "simple"
        if backend == "chroma" and Chroma is None:
            raise ImportError(
                "langchain_chroma/chromadb is not installed. Install requirements "
                "or set VECTOR_DB_BACKEND=simple."
            )
        if backend not in {"chroma", "simple"}:
            raise ValueError("VECTOR_DB_BACKEND must be one of: auto, chroma, simple")
        return backend

    def create(
        self,
        documents: Sequence[Document],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        recreate: bool = False,
    ):
        if recreate and self.persist_directory.exists():
            shutil.rmtree(self.persist_directory)

        if self.backend == "chroma":
            assert Chroma is not None
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self.db = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embedding_service.embeddings,
            )
            self.add_documents(documents, batch_size=batch_size)
        else:
            self.db = SimpleVectorStore(
                self.persist_directory,
                self.embedding_service.embeddings,
            )
            if recreate:
                self.db.reset()
            self.db.add_documents(documents, batch_size=batch_size)
        return self.db

    def load(self):
        if self.backend == "chroma":
            assert Chroma is not None
            self.db = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embedding_service.embeddings,
            )
        else:
            self.db = SimpleVectorStore(
                self.persist_directory,
                self.embedding_service.embeddings,
            ).load()
        return self.db

    def add_documents(
        self,
        documents: Sequence[Document],
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        if self.db is None:
            self.create(documents, batch_size=batch_size)
            return

        for start in range(0, len(documents), batch_size):
            batch = list(documents[start : start + batch_size])
            self.db.add_documents(batch)

        if hasattr(self.db, "persist"):
            self.db.persist()

    def search(
        self,
        query: str,
        k: int = TOP_K,
        score_threshold: float | None = SCORE_THRESHOLD,
    ) -> list[tuple[Document, float]]:
        if self.db is None:
            self.load()

        results = self.db.similarity_search_with_score(query=query, k=k)
        threshold = SCORE_THRESHOLD if score_threshold is None else score_threshold
        if threshold <= 0:
            return results

        if self.backend == "chroma":
            max_distance = 1 - threshold
            return [(doc, score) for doc, score in results if score <= max_distance]
        return [(doc, score) for doc, score in results if score >= threshold]

    def relevance_score(self, raw_score: float) -> float:
        if self.backend == "chroma":
            return 1 - raw_score
        return raw_score

    def get_documents_by_page(self, filename: str, page_number: int | str) -> list[Document]:
        if self.db is None:
            self.load()

        if self.backend == "simple":
            return self.db.get_documents_by_page(filename, page_number)

        items = self.db.get(where={"filename": filename})
        documents: list[Document] = []
        page_text = str(page_number)
        for text, metadata in zip(items.get("documents", []), items.get("metadatas", [])):
            row_page = metadata.get("page_number") or metadata.get("page")
            if str(row_page) == page_text:
                documents.append(Document(page_content=text, metadata=metadata))

        return sorted(
            documents,
            key=lambda doc: int(doc.metadata.get("chunk_id") or 0),
        )

    def stats(self) -> dict:
        if self.backend == "simple":
            db = self.db or self.load()
            return {
                "backend": self.backend,
                "persist_directory": str(self.persist_directory),
                "document_count": len(db._rows),
            }

        collection = (self.db or self.load())._collection
        return {
            "backend": self.backend,
            "persist_directory": str(self.persist_directory),
            "document_count": collection.count(),
        }
