from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document


@dataclass
class PageCandidate:
    filename: str
    page_number: int
    keyword_score: float
    bonus_score: float


class PageTextIndex:
    def __init__(self, chunks_path: str | Path):
        self.chunks_path = Path(chunks_path)
        self.pages: dict[tuple[str, int], list[Document]] = {}
        self.page_texts: dict[tuple[str, int], str] = {}
        self.page_tokens: dict[tuple[str, int], Counter] = {}
        self.page_lengths: dict[tuple[str, int], int] = {}
        self.idf: dict[str, float] = {}
        self.avg_doc_length = 1.0
        self._load()

    def _load(self) -> None:
        if not self.chunks_path.exists():
            return

        with self.chunks_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                filename = row.get("filename")
                page_number = row.get("page_number")
                text = (row.get("text") or "").strip()
                if not filename or page_number is None or not text:
                    continue
                metadata = {key: value for key, value in row.items() if key != "text"}
                key = (filename, int(page_number))
                self.pages.setdefault(key, []).append(
                    Document(page_content=text, metadata=metadata)
                )

        for key, documents in self.pages.items():
            documents.sort(key=lambda doc: int(doc.metadata.get("chunk_id") or 0))
            self.page_texts[key] = "\n".join(doc.page_content for doc in documents)

        self._build_bm25()

    def _build_bm25(self) -> None:
        document_frequency: Counter[str] = Counter()
        total_length = 0
        for key, text in self.page_texts.items():
            tokens = Counter(self.tokenize(text))
            self.page_tokens[key] = tokens
            doc_length = sum(tokens.values())
            self.page_lengths[key] = doc_length
            total_length += doc_length
            document_frequency.update(tokens.keys())

        total_docs = max(len(self.page_tokens), 1)
        self.avg_doc_length = total_length / total_docs if total_docs else 1.0
        self.idf = {
            token: math.log((total_docs - freq + 0.5) / (freq + 0.5) + 1)
            for token, freq in document_frequency.items()
        }

    @staticmethod
    def normalize_text(text: str) -> str:
        return re.sub(r"\s+", "", str(text or "").lower())

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        normalized = cls.normalize_text(text)
        tokens = re.findall(r"[a-zA-Z]+\d*[a-zA-Z0-9.-]*|\d+(?:\.\d+)?%?", normalized)
        for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
            for ngram_size in (2, 3, 4):
                if len(sequence) >= ngram_size:
                    tokens.extend(
                        sequence[index : index + ngram_size]
                        for index in range(len(sequence) - ngram_size + 1)
                    )
        return tokens

    def keyword_score(self, query: str, key: tuple[str, int]) -> float:
        query_tokens = Counter(self.tokenize(query))
        page_tokens = self.page_tokens.get(key, Counter())
        doc_length = self.page_lengths.get(key, 1) or 1
        score = 0.0
        k1 = 1.5
        b = 0.75
        for token in query_tokens:
            frequency = page_tokens.get(token, 0)
            if not frequency:
                continue
            denominator = frequency + k1 * (1 - b + b * doc_length / self.avg_doc_length)
            score += self.idf.get(token, 0.0) * frequency * (k1 + 1) / denominator
        return score

    def bonus_score(self, query: str, key: tuple[str, int]) -> float:
        filename, _ = key
        page_text = self.normalize_text(self.page_texts.get(key, ""))
        score = 0.0

        for company_name in ("联邦制药", "凌云股份", "广联达", "伊利股份", "千味央厨"):
            if company_name in query and company_name in filename:
                score += 0.12

        for figure_number in re.findall(r"(?:图表|图|表)\s*(\d+)", query):
            if re.search(r"(?:图表|图|表)\s*" + re.escape(figure_number), self.page_texts.get(key, "")):
                score += 0.35
            elif figure_number in page_text:
                score += 0.08

        query_numbers = set(re.findall(r"\d+(?:\.\d+)?%?", self.normalize_text(query)))
        if query_numbers:
            number_hits = sum(1 for number in query_numbers if number in page_text)
            score += min(0.12, number_hits * 0.025)

        return score

    def exact_phrase_score(self, query: str, key: tuple[str, int]) -> float:
        page_text = self.normalize_text(self.page_texts.get(key, ""))
        score = 0.0
        for term in self.important_terms(query):
            if term not in page_text:
                continue
            if re.fullmatch(r"\d{4}年|\d+(?:\.\d+)?%?", term):
                score += 0.02
            elif len(term) >= 8:
                score += 0.12
            elif len(term) >= 5:
                score += 0.08
            else:
                score += 0.04
        return min(score, 0.45)

    def early_page_penalty(self, query: str, key: tuple[str, int]) -> float:
        page_number = key[1]
        if page_number > 5:
            return 0.0
        if any(
            cue in query
            for cue in ("发展历程", "营收结构", "平台战略", "本期内容提要")
        ):
            return 0.0
        return {
            1: 0.20,
            2: 0.14,
            3: 0.11,
            4: 0.09,
            5: 0.07,
        }.get(page_number, 0.0)

    def important_terms(self, query: str) -> set[str]:
        normalized_query = self.normalize_text(query)
        terms = set(
            re.findall(
                r"[a-zA-Z]+\d*[a-zA-Z0-9+.-]*|\d{4}年|\d+(?:\.\d+)?%?",
                normalized_query,
            )
        )
        stop_phrases = (
            "根据",
            "请问",
            "如何",
            "分析",
            "评估",
            "关于",
            "公司",
            "报告",
            "深度",
            "研究",
            "能否",
            "详细",
            "具体",
            "哪些",
            "什么",
            "情况",
            "方面",
            "文件内容",
            "文件",
            "内容",
            "发展",
            "未来",
            "进行",
            "实现",
            "影响",
            "主要",
        )
        for sequence in re.findall(r"[\u4e00-\u9fff]{4,}", normalized_query):
            cleaned = sequence
            for phrase in stop_phrases:
                cleaned = cleaned.replace(phrase, " ")
            for part in re.findall(r"[\u4e00-\u9fff]{4,}", cleaned):
                if len(part) <= 12:
                    terms.add(part)
                for ngram_size in (4, 5, 6, 7, 8):
                    if len(part) >= ngram_size:
                        terms.update(
                            part[index : index + ngram_size]
                            for index in range(len(part) - ngram_size + 1)
                        )

        domain_terms = (
            "发展历程",
            "重大产品临床",
            "营业收入",
            "净利润",
            "减重适应症",
            "热成型",
            "客户拓展",
            "新能源汽车",
            "力传感器",
            "施工总承包",
            "数字孪生",
            "数字造价",
            "数字施工",
            "数字设计",
            "智慧工地",
            "材料核算",
            "供应链管理",
            "奶酪业务",
            "营销策略",
            "产品创新",
            "体重管理",
            "风险调整",
            "销售峰值",
            "营收结构",
            "股价走势",
            "竞争格局",
            "大客户资源",
            "产品质量",
            "成本控制",
            "工程造价",
            "装配式建筑",
            "AECORE",
            "BIM5D",
            "EPC",
            "5G",
            "云计算",
        )
        for term in domain_terms:
            if term.lower() in query.lower():
                terms.add(self.normalize_text(term))
        return {term for term in terms if len(term) >= 2}

    def rank(self, query: str, top_k: int = 30) -> list[PageCandidate]:
        candidates: list[PageCandidate] = []
        for key in self.page_texts:
            keyword = self.keyword_score(query, key)
            bonus = self.bonus_score(query, key)
            if keyword <= 0 and bonus <= 0:
                continue
            candidates.append(
                PageCandidate(
                    filename=key[0],
                    page_number=key[1],
                    keyword_score=keyword,
                    bonus_score=bonus,
                )
            )
        candidates.sort(key=lambda item: (item.keyword_score, item.bonus_score), reverse=True)
        return candidates[:top_k]

    def get_documents_by_page(self, filename: str, page_number: int | str) -> list[Document]:
        return list(self.pages.get((filename, int(page_number)), []))

    def build_page_content(
        self,
        filename: str,
        page_number: int | str,
        query: str,
        max_chars: int,
        preferred_chunk_ids: Iterable[int] | None = None,
    ) -> tuple[str, list[int]]:
        documents = self.get_documents_by_page(filename, page_number)
        preferred = set(preferred_chunk_ids or [])
        if not documents:
            return "", []

        scored_documents = []
        for document in documents:
            chunk_id = int(document.metadata.get("chunk_id") or 0)
            text = document.page_content or ""
            score = self._chunk_relevance(query, text)
            if chunk_id in preferred:
                score += 2.0
            if "【OCR补充】" in text or "【表格抽取】" in text:
                score += 0.6
            scored_documents.append((score, chunk_id, document))

        total_text = "\n".join(doc.page_content for _, _, doc in scored_documents)
        if len(total_text) <= max_chars:
            return total_text, [chunk_id for _, chunk_id, _ in sorted(scored_documents, key=lambda item: item[1])]

        selected: list[tuple[int, Document]] = []
        used_chars = 0
        for _, chunk_id, document in sorted(scored_documents, key=lambda item: (item[0], -item[1]), reverse=True):
            text = (document.page_content or "").strip()
            if not text:
                continue
            next_size = used_chars + len(text) + 1
            if next_size > max_chars and selected:
                continue
            selected.append((chunk_id, document))
            used_chars = next_size
            if used_chars >= max_chars:
                break

        selected.sort(key=lambda item: item[0])
        merged = "\n".join(document.page_content for _, document in selected)
        if len(merged) > max_chars:
            merged = merged[:max_chars].rstrip() + "..."
        return merged, [chunk_id for chunk_id, _ in selected]

    def _chunk_relevance(self, query: str, text: str) -> float:
        query_tokens = set(self.tokenize(query))
        text_tokens = set(self.tokenize(text))
        if not query_tokens or not text_tokens:
            return 0.0
        overlap = len(query_tokens & text_tokens) / len(query_tokens)
        score = overlap
        query_numbers = set(re.findall(r"\d+(?:\.\d+)?%?", self.normalize_text(query)))
        if query_numbers:
            number_hits = sum(1 for number in query_numbers if number in self.normalize_text(text))
            score += min(0.4, number_hits * 0.1)
        return score
