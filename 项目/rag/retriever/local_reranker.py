from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class RerankResult:
    index: int
    score: float


class LocalPageReranker:
    """Thin wrapper around local CrossEncoder/FlagEmbedding/HF rerankers."""

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
        if self._prefer_hf_sequence():
            return self._load_hf_sequence_reranker()
        if self._prefer_flag_embedding():
            return self._load_flag_reranker()
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

        return self._load_flag_reranker(first_error if "first_error" in locals() else None)

    def _load_flag_reranker(self, first_error: Exception | None = None):
        try:
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(self.model_name, use_fp16=False)
            self._backend = "flag_embedding"
            return self._model
        except Exception as exc:
            self.error = f"{first_error}; {exc}" if first_error is not None else str(exc)
            raise

    def _load_hf_sequence_reranker(self, first_error: Exception | None = None):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            self._model = (tokenizer, model, device)
            self._backend = "hf_sequence"
            return self._model
        except Exception as exc:
            self.error = f"{first_error}; {exc}" if first_error is not None else str(exc)
            raise

    def _prefer_hf_sequence(self) -> bool:
        model = self.model_name.lower()
        return "bge-reranker-v2" in model

    def _prefer_flag_embedding(self) -> bool:
        model = self.model_name.lower()
        return "bge-reranker-v2" in model or "flag" in model

    def rerank(self, query: str, passages: Iterable[str]) -> list[RerankResult]:
        passages = [self._trim_passage(passage) for passage in passages]
        if not passages:
            return []
        model = self._load_model()
        pairs = [(query, passage) for passage in passages]
        if self._backend == "hf_sequence":
            scores = self._predict_hf_sequence(query, passages)
        elif self._backend == "flag_embedding":
            scores = model.compute_score(pairs, batch_size=self.batch_size)
        else:
            try:
                scores = model.predict(pairs, batch_size=self.batch_size)
            except Exception as exc:
                self._model = None
                self.error = str(exc)
                if self._prefer_hf_sequence():
                    self._load_hf_sequence_reranker(exc)
                    scores = self._predict_hf_sequence(query, passages)
                else:
                    model = self._load_flag_reranker(exc)
                    scores = model.compute_score(pairs, batch_size=self.batch_size)
        return [
            RerankResult(index=index, score=float(score))
            for index, score in enumerate(scores)
        ]

    def _predict_hf_sequence(self, query: str, passages: list[str]) -> list[float]:
        import torch

        tokenizer, model, device = self._model
        scores: list[float] = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start : start + self.batch_size]
            encoded = tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits.reshape(-1)
            scores.extend(float(value) for value in logits.detach().cpu().tolist())
        return scores

    def _trim_passage(self, passage: str) -> str:
        text = " ".join(str(passage or "").split())
        if len(text) <= self.max_chars:
            return text
        head = self.max_chars // 2
        tail = self.max_chars - head
        return f"{text[:head]}\n...\n{text[-tail:]}"
