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


@dataclass
class SectionProfile:
    filename: str
    title: str
    start_page: int
    end_page: int
    profile_text: str


class PageTextIndex:
    WEAK_QUERY_TERMS = (
        "如何",
        "分析",
        "评估",
        "评价",
        "看待",
    )

    def __init__(self, chunks_path: str | Path):
        self.chunks_path = Path(chunks_path)
        self.pages: dict[tuple[str, int], list[Document]] = {}
        self.page_texts: dict[tuple[str, int], str] = {}
        self.page_tokens: dict[tuple[str, int], Counter] = {}
        self.page_lengths: dict[tuple[str, int], int] = {}
        self.section_profiles: dict[str, list[SectionProfile]] = {}
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
        self._build_section_profiles()

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

    @classmethod
    def query_text_for_keyword(cls, query: str) -> str:
        normalized = cls.normalize_text(query)
        filtered = normalized
        for term in cls.WEAK_QUERY_TERMS:
            filtered = filtered.replace(cls.normalize_text(term), " ")
        return filtered if len(cls.tokenize(filtered)) >= 2 else normalized

    def keyword_score(self, query: str, key: tuple[str, int]) -> float:
        query_tokens = Counter(self.tokenize(self.query_text_for_keyword(query)))
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

    def _build_section_profiles(self) -> None:
        self.section_profiles = {}
        files = sorted({filename for filename, _ in self.page_texts})
        for filename in files:
            pages = sorted(page for file_name, page in self.page_texts if file_name == filename)
            if not pages:
                continue
            starts = self._extract_toc_sections(filename, pages)
            if len(starts) < 2:
                starts = self._extract_heading_sections(filename, pages)
            if not starts:
                starts = [(pages[0], "全文")]

            merged_by_page: dict[int, list[str]] = {}
            page_set = set(pages)
            for page_number, title in starts:
                if page_number not in page_set or not title:
                    continue
                title = self._clean_section_title(title)
                if not title:
                    continue
                merged_by_page.setdefault(page_number, [])
                if title not in merged_by_page[page_number]:
                    merged_by_page[page_number].append(title)

            distinct_starts = sorted(merged_by_page)
            profiles: list[SectionProfile] = []
            for index, start_page in enumerate(distinct_starts):
                next_start = distinct_starts[index + 1] if index + 1 < len(distinct_starts) else pages[-1] + 1
                end_page = max(start_page, min(next_start - 1, pages[-1]))
                title = "；".join(merged_by_page[start_page][:3])
                profile_text = self._build_section_profile_text(filename, title, start_page, end_page)
                if profile_text:
                    profiles.append(
                        SectionProfile(
                            filename=filename,
                            title=title,
                            start_page=start_page,
                            end_page=end_page,
                            profile_text=profile_text,
                        )
                    )
            self.section_profiles[filename] = profiles

    def _extract_toc_sections(self, filename: str, pages: list[int]) -> list[tuple[int, str]]:
        starts: list[tuple[int, str]] = []
        toc_pages = pages[: min(12, len(pages))]
        for page_number in toc_pages:
            text = self.page_texts.get((filename, page_number), "")
            for line in self._clean_lines(text):
                if self._is_noise_heading(line) or line.startswith(("图表", "图 ", "图")):
                    continue
                parsed = self._parse_toc_line(line)
                if parsed is not None:
                    starts.append(parsed)
        return self._deduplicate_section_starts(starts)

    def _extract_heading_sections(self, filename: str, pages: list[int]) -> list[tuple[int, str]]:
        starts: list[tuple[int, str]] = []
        for page_number in pages:
            text = self.page_texts.get((filename, page_number), "")
            for line in self._clean_lines(text)[:12]:
                if self._is_noise_heading(line):
                    continue
                if self._looks_like_section_heading(line):
                    starts.append((page_number, line))
                    break
        return self._deduplicate_section_starts(starts)

    @classmethod
    def _parse_toc_line(cls, line: str) -> tuple[int, str] | None:
        text = line.strip()
        patterns = (
            r"^(.{2,90}?)(?:\.{2,}|…{2,}|·{2,}|\s{2,})\s*-?\s*(\d{1,3})\s*-?\s*$",
            r"^(.{2,90}?)\s+-\s*(\d{1,3})\s*-\s*$",
        )
        for pattern in patterns:
            match = re.match(pattern, text)
            if not match:
                continue
            title, page = match.group(1).strip(), int(match.group(2))
            if cls._looks_like_section_heading(title) or cls._looks_like_plain_toc_title(title):
                return page, title
        return None

    @staticmethod
    def _looks_like_plain_toc_title(line: str) -> bool:
        if len(line) < 4 or len(line) > 80:
            return False
        if line.startswith(("目录", "内容目录", "正文目录", "图表目录", "资料来源", "免责声明")):
            return False
        return bool(re.search(r"[\u4e00-\u9fff]{3,}", line))

    @staticmethod
    def _looks_like_section_heading(line: str) -> bool:
        text = line.strip()
        if len(text) < 4 or len(text) > 100:
            return False
        if text.startswith(("图表", "图 ", "表 ", "资料来源", "免责声明", "请务必阅读")):
            return False
        return bool(
            re.match(r"^[一二三四五六七八九十]{1,3}[、.．]\s*\S+", text)
            or re.match(r"^\d+(?:\.\d+){0,3}\.?\s*\S+", text)
        )

    @classmethod
    def _deduplicate_section_starts(cls, starts: list[tuple[int, str]]) -> list[tuple[int, str]]:
        seen: set[tuple[int, str]] = set()
        cleaned: list[tuple[int, str]] = []
        for page, title in starts:
            title = cls._clean_section_title(title)
            if page <= 0 or not title:
                continue
            key = (page, title)
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(key)
        cleaned.sort(key=lambda item: (item[0], item[1]))
        return cleaned

    @staticmethod
    def _clean_section_title(title: str) -> str:
        text = re.sub(r"\s+", " ", str(title or "")).strip(" .。·…-")
        text = re.sub(r"^目录\s*", "", text).strip()
        return text[:120]

    @staticmethod
    def _is_noise_heading(line: str) -> bool:
        return bool(
            not line
            or line.startswith(("请务必阅读", "免责声明", "证券研究报告", "公司深度报告", "资料来源", "注："))
            or line in {"目 录", "目录", "正文目录", "内容目录", "图表目录", "CONTENTS"}
        )

    @classmethod
    def _clean_lines(cls, text: str) -> list[str]:
        return [
            re.sub(r"\s+", " ", line).strip()
            for line in str(text or "").splitlines()
            if re.sub(r"\s+", " ", line).strip()
        ]

    def _build_section_profile_text(
        self,
        filename: str,
        title: str,
        start_page: int,
        end_page: int,
    ) -> str:
        pieces: list[str] = [title]
        for page_number in range(start_page, end_page + 1):
            text = self.page_texts.get((filename, page_number), "")
            if not text:
                continue
            lines = self._clean_lines(text)
            selected_lines: list[str] = []
            for line in lines[:30]:
                if self._is_noise_heading(line):
                    continue
                if (
                    self._looks_like_section_heading(line)
                    or re.match(r"^(?:图表|图|表)\s*\d{1,3}\s*[:：]", line)
                    or ("：" in line and len(line) <= 80)
                ):
                    selected_lines.append(line)
                if len(selected_lines) >= 5:
                    break
            body = " ".join(
                line
                for line in lines
                if not self._is_noise_heading(line) and not line.startswith(("图表目录", "图表 "))
            )
            if body:
                selected_lines.append(body[:220])
            pieces.extend(selected_lines[:6])
            if sum(len(piece) for piece in pieces) > 3000:
                break
        profile = "\n".join(piece for piece in pieces if piece)
        return profile[:3500]

    def section_keyword_score(self, query: str, section: SectionProfile) -> float:
        query_tokens = Counter(self.tokenize(self.query_text_for_keyword(query)))
        section_tokens = Counter(self.tokenize(section.profile_text))
        if not query_tokens or not section_tokens:
            return 0.0
        doc_length = sum(section_tokens.values()) or 1
        score = 0.0
        k1 = 1.5
        b = 0.75
        for token in query_tokens:
            frequency = section_tokens.get(token, 0)
            if not frequency:
                continue
            denominator = frequency + k1 * (1 - b + b * doc_length / self.avg_doc_length)
            score += self.idf.get(token, 0.0) * frequency * (k1 + 1) / denominator
        return score

    def section_entity_overlap(self, query: str, section: SectionProfile) -> float:
        terms = self.important_terms(query)
        if not terms:
            return 0.0
        profile_text = self.normalize_text(section.profile_text)
        hits = sum(1 for term in terms if term in profile_text)
        return hits / max(len(terms), 1)

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
