from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class RerankResult:
    index: int
    score: float


class LocalPageReranker:
    """Thin wrapper around a local sentence-transformers CrossEncoder reranker."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 8,
        max_chars: int = 1200,
    ) -> None:
        self.model_name = (model_name or "").strip()
        self.batch_size = max(1, int(batch_size))
        self.max_chars = max(200, int(max_chars))
        self._model = None
        self._backend = ""
        self.error = ""

    @property
    def enabled(self) -> bool:
        return bool(self.model_name)

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.enabled:
            raise RuntimeError("reranker model is empty")
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            self._backend = "sentence_transformers"
            return self._model
        except ImportError:
            pass
        except Exception as exc:
            self.error = str(exc)
            first_error = exc

        try:
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(self.model_name, use_fp16=False)
            self._backend = "flag_embedding"
            return self._model
        except Exception as exc:
            self.error = f"{first_error}; {exc}" if "first_error" in locals() else str(exc)
            raise

    def rerank(self, query: str, passages: Iterable[str]) -> list[RerankResult]:
        passages = [self._trim_passage(passage) for passage in passages]
        if not passages:
            return []
        model = self._load_model()
        pairs = [(query, passage) for passage in passages]
        if self._backend == "flag_embedding":
            scores = model.compute_score(pairs, batch_size=self.batch_size)
        else:
            scores = model.predict(pairs, batch_size=self.batch_size)
        return [
            RerankResult(index=index, score=float(score))
            for index, score in enumerate(scores)
        ]

    def _trim_passage(self, passage: str) -> str:
        text = " ".join(str(passage or "").split())
        if len(text) <= self.max_chars:
            return text
        head = self.max_chars // 2
        tail = self.max_chars - head
        return f"{text[:head]}\n...\n{text[-tail:]}"
