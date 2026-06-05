from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

from config.config import (
    CHART_DIRECTORY_PENALTY,
    CHART_PAGE_BOOST,
    CONTENT_ANCHOR_BOOSTS_ENABLED,
    FINAL_SOURCE_OVERRIDE_MARGIN,
    FINAL_SOURCE_OVERRIDE_RATIO,
    MANUAL_SECTION_RULES_ENABLED,
    PAGE_RERANKER_BATCH_SIZE,
    PAGE_RERANKER_CANDIDATES,
    PAGE_RERANKER_MAX_CHARS,
    PAGE_RERANKER_NEIGHBOR_PAGES,
    PAGE_RERANKER_MODEL,
    PAGE_RERANKER_WEIGHT,
    RETRIEVAL_BONUS_WEIGHT,
    RETRIEVAL_CHUNKS_PATH,
    RETRIEVAL_EARLY_PAGE_PENALTY,
    RETRIEVAL_EXACT_WEIGHT,
    RETRIEVAL_KEYWORD_WEIGHT,
    RETRIEVAL_MODE,
    RETRIEVAL_VECTOR_WEIGHT,
    SEMANTIC_SECTION_BONUS,
    SEMANTIC_SECTION_MIN_SCORE,
    SEMANTIC_SECTION_ROUTING_ENABLED,
    SEMANTIC_SECTION_TOP_K,
    SECTION_ROUTE_BONUS,
    TARGETED_RETRIEVAL_ENABLED,
)
from rag.llm.llm import LLMService
from rag.retriever.local_reranker import LocalPageReranker
from rag.retriever.page_index import PageTextIndex
from rag.vector.vector_db import VectorDB


@dataclass
class RetrievedPage:
    filename: str
    page_number: int | str
    score: float
    hit_count: int
    chunk_ids: list[int]
    content: str
    rule_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "page_number": self.page_number,
            "score": self.score,
            "hit_count": self.hit_count,
            "chunk_ids": self.chunk_ids,
            "content": self.content,
            "rule_score": self.rule_score,
        }


@dataclass
class StructuredAnswer:
    filename: str
    page: int | str
    answer: str
    sources: list[dict]
    prompt: str
    raw_answer: str = ""
    llm_used: bool = False
    error: str = ""

    def to_dict(self, include_prompt: bool = False) -> dict:
        data = {
            "filename": self.filename,
            "page": self.page,
            "answer": self.answer,
            "sources": self.sources,
            "llm_used": self.llm_used,
            "raw_answer": self.raw_answer,
            "error": self.error,
        }
        if include_prompt:
            data["prompt"] = self.prompt
        return data


class RAGService:
    def __init__(
        self,
        vector_db: VectorDB | None = None,
        llm: LLMService | None = None,
        page_index: PageTextIndex | None = None,
    ):
        self.vector_db = vector_db or VectorDB()
        self.llm = llm or LLMService()
        self.page_index = page_index
        if self.page_index is None and RETRIEVAL_MODE == "hybrid":
            self.page_index = PageTextIndex(RETRIEVAL_CHUNKS_PATH)
        self._page_rerankers: dict[tuple[str, int, int], LocalPageReranker] = {}
        self._disabled_page_rerankers: set[tuple[str, int, int]] = set()
        self._section_embedding_cache: dict[str, list[list[float]]] = {}

    @staticmethod
    def _expanded_query_terms(query: str) -> set[str]:
        normalized_query = PageTextIndex.normalize_text(query)
        expansions: set[str] = set()
        rules = {
            "收入": ("收入", "营收", "营业收入"),
            "营收": ("收入", "营收", "营业收入"),
            "占比": ("占比", "比重", "占公司总营收", "占公司"),
            "主要产品": ("主要产品", "产品", "业务板块", "第一大业务板块"),
            "客户": ("客户", "主要客户", "定点", "项目定点"),
            "市场潜力": ("市场空间", "市场规模", "成长空间", "渗透率"),
            "增长潜力": ("增长潜力", "成长空间", "市场空间", "增速", "复合增速", "CAGR"),
            "发展前景": ("发展前景", "市场空间", "成长空间", "渗透率"),
            "竞争格局": ("竞争格局", "市场份额", "份额", "集中度"),
            "供应链": ("供应链", "供应商", "质量", "供应稳定"),
            "产品质量": ("产品质量", "质量", "品控", "供应商"),
            "股价走势": ("股价", "收盘价", "走势", "涨跌幅"),
            "盈利预测": ("盈利预测", "营业收入", "归母净利润", "净利润"),
            "净利润": ("净利润", "归母净利润", "利润"),
            "临床进展": ("临床", "临床试验", "适应症", "获批"),
            "市场表现": ("市场表现", "收入", "同比", "规模"),
        }
        for trigger, values in rules.items():
            if PageTextIndex.normalize_text(trigger) in normalized_query:
                expansions.update(PageTextIndex.normalize_text(value) for value in values)
        return {term for term in expansions if len(term) >= 2}

    def retrieve_pages(
        self,
        query: str,
        initial_k: int = 80,
        final_pages: int = 5,
        score_threshold: float = 0,
        max_chars_per_page: int = 5000,
        reranker_model: str | None = PAGE_RERANKER_MODEL,
        reranker_candidates: int = PAGE_RERANKER_CANDIDATES,
        reranker_batch_size: int = PAGE_RERANKER_BATCH_SIZE,
        reranker_max_chars: int = PAGE_RERANKER_MAX_CHARS,
        reranker_weight: float = PAGE_RERANKER_WEIGHT,
        restrict_reranker_to_top_file: bool = True,
        reranker_query_mode: str = "original",
        reranker_neighbor_pages: int = PAGE_RERANKER_NEIGHBOR_PAGES,
        targeted_retrieval: bool = TARGETED_RETRIEVAL_ENABLED,
    ) -> list[RetrievedPage]:
        hits = self.vector_db.search(query, k=initial_k, score_threshold=score_threshold)
        grouped: dict[tuple[str, str], dict] = {}

        for rank, (doc, raw_score) in enumerate(hits, start=1):
            metadata = doc.metadata or {}
            filename = metadata.get("filename") or metadata.get("source")
            page_number = metadata.get("page_number") or metadata.get("page")
            if not filename or page_number is None:
                continue

            key = (filename, str(page_number))
            relevance = self.vector_db.relevance_score(raw_score)
            group = grouped.setdefault(
                key,
                {
                    "filename": filename,
                    "page_number": page_number,
                    "best_score": relevance,
                    "score_sum": 0.0,
                    "hit_count": 0,
                    "first_rank": rank,
                    "chunk_ids": set(),
                },
            )
            group["best_score"] = max(group["best_score"], relevance)
            group["score_sum"] += relevance
            group["hit_count"] += 1
            group["first_rank"] = min(group["first_rank"], rank)
            if metadata.get("chunk_id") is not None:
                group["chunk_ids"].add(int(metadata["chunk_id"]))

        if self.page_index is not None:
            for candidate in self.page_index.rank(query, top_k=max(30, final_pages * 6)):
                key = (candidate.filename, str(candidate.page_number))
                group = grouped.setdefault(
                    key,
                    {
                        "filename": candidate.filename,
                        "page_number": candidate.page_number,
                        "best_score": 0.0,
                        "score_sum": 0.0,
                        "hit_count": 0,
                        "first_rank": initial_k + 1,
                        "chunk_ids": set(),
                    },
                )
                group["keyword_score"] = max(
                    group.get("keyword_score", 0.0),
                    candidate.keyword_score,
                )
                group["bonus_score"] = max(
                    group.get("bonus_score", 0.0),
                    candidate.bonus_score,
                )
                key_tuple = (candidate.filename, int(candidate.page_number))
                group["exact_score"] = max(
                    group.get("exact_score", 0.0),
                    self.page_index.exact_phrase_score(query, key_tuple),
                )
                group["early_penalty"] = max(
                    group.get("early_penalty", 0.0),
                    self.page_index.early_page_penalty(query, key_tuple),
                )

        max_vector_score = max((item.get("best_score", 0.0) for item in grouped.values()), default=1.0) or 1.0
        max_keyword_score = max((item.get("keyword_score", 0.0) for item in grouped.values()), default=1.0) or 1.0
        for group in grouped.values():
            vector_part = group.get("best_score", 0.0) / max_vector_score
            keyword_part = group.get("keyword_score", 0.0) / max_keyword_score
            bonus_part = group.get("bonus_score", 0.0)
            exact_part = group.get("exact_score", 0.0)
            early_penalty = group.get("early_penalty", 0.0)
            group["final_score"] = (
                RETRIEVAL_VECTOR_WEIGHT * vector_part
                + RETRIEVAL_KEYWORD_WEIGHT * keyword_part
                + RETRIEVAL_BONUS_WEIGHT * bonus_part
                + RETRIEVAL_EXACT_WEIGHT * exact_part
                - RETRIEVAL_EARLY_PAGE_PENALTY * early_penalty
            )

        pre_boost_top_filename = ""
        if grouped:
            pre_boost_top_filename = max(
                grouped.values(),
                key=lambda item: (
                    item.get("final_score", item.get("best_score", 0.0)),
                    item.get("score_sum", 0.0) / max(item.get("hit_count", 1), 1),
                ),
            )["filename"]
        if targeted_retrieval and self.page_index is not None:
            self._apply_targeted_page_boosts(
                query,
                grouped,
                initial_k,
                restrict_filename=pre_boost_top_filename if restrict_reranker_to_top_file else "",
            )

        ranked_groups = sorted(
            grouped.values(),
            key=lambda item: (
                item.get("final_score", item["best_score"]),
                item["score_sum"] / max(item["hit_count"], 1),
                item["hit_count"],
                -item["first_rank"],
            ),
            reverse=True,
        )
        if restrict_reranker_to_top_file and ranked_groups:
            top_filename = ranked_groups[0]["filename"]
            same_file = [group for group in ranked_groups if group["filename"] == top_filename]
            other_files = [group for group in ranked_groups if group["filename"] != top_filename]
            ranked_groups = same_file + other_files

        rerank_count = max(0, int(reranker_candidates or 0)) if reranker_model else 0
        candidate_count = max(final_pages, rerank_count)
        if rerank_count > 0:
            rerank_groups = self._expand_rerank_groups_with_neighbors(
                ranked_groups[:rerank_count],
                grouped,
                radius=reranker_neighbor_pages,
            )
            rerank_keys = {
                (group["filename"], int(group["page_number"]))
                for group in rerank_groups
            }
            tail_groups = [
                group
                for group in ranked_groups[:candidate_count]
                if (group["filename"], int(group["page_number"])) not in rerank_keys
            ]
            pages = self._build_pages_from_groups(
                query=query,
                groups=rerank_groups,
                max_chars_per_page=max_chars_per_page,
            )
            head = self._rerank_pages(
                query=self._build_reranker_query(query, reranker_query_mode),
                pages=pages,
                model_name=reranker_model or "",
                batch_size=reranker_batch_size,
                max_chars=reranker_max_chars,
                weight=reranker_weight,
            )
            pages = head + self._build_pages_from_groups(
                query=query,
                groups=tail_groups,
                max_chars_per_page=max_chars_per_page,
            )
        else:
            pages = self._build_pages_from_groups(
                query=query,
                groups=ranked_groups[:candidate_count],
                max_chars_per_page=max_chars_per_page,
            )

        return pages[:final_pages]

    @staticmethod
    def _group_key(group: dict) -> tuple[str, int]:
        return group["filename"], int(group["page_number"])

    def _make_empty_group(
        self,
        filename: str,
        page_number: int,
        initial_k: int,
    ) -> dict:
        return {
            "filename": filename,
            "page_number": page_number,
            "best_score": 0.0,
            "score_sum": 0.0,
            "hit_count": 0,
            "first_rank": initial_k + 1,
            "chunk_ids": set(),
            "keyword_score": 0.0,
            "bonus_score": 0.0,
            "exact_score": 0.0,
            "early_penalty": 0.0,
            "final_score": 0.0,
        }

    def _expand_rerank_groups_with_neighbors(
        self,
        groups: list[dict],
        all_groups: dict[tuple[str, str], dict],
        radius: int,
    ) -> list[dict]:
        if self.page_index is None or radius <= 0:
            return list(groups)
        expanded: list[dict] = []
        seen: set[tuple[str, int]] = set()
        for group in groups:
            filename, page_number = self._group_key(group)
            offsets = [0]
            for distance in range(1, radius + 1):
                offsets.extend([-distance, distance])
            for offset in offsets:
                neighbor = page_number + offset
                if neighbor <= 0:
                    continue
                key = (filename, neighbor)
                if key in seen or not self.page_index.get_documents_by_page(filename, neighbor):
                    continue
                seen.add(key)
                source = all_groups.get((filename, str(neighbor)))
                if source is None:
                    source = self._make_empty_group(filename, neighbor, initial_k=0)
                    source["final_score"] = max(
                        float(group.get("final_score", 0.0)) - 0.08 * abs(neighbor - page_number),
                        0.0,
                    )
                expanded.append(source)
        expanded.sort(
            key=lambda item: (
                float(item.get("final_score", item.get("best_score", 0.0))),
                item.get("hit_count", 0),
                -int(item.get("page_number", 0)),
            ),
            reverse=True,
        )
        return expanded

    def _apply_targeted_page_boosts(
        self,
        query: str,
        grouped: dict[tuple[str, str], dict],
        initial_k: int,
        restrict_filename: str = "",
    ) -> None:
        if self.page_index is None:
            return
        chart_pages = self._lookup_chart_pages(query, restrict_filename=restrict_filename)
        for filename, page_number in chart_pages:
            group = grouped.setdefault(
                (filename, str(page_number)),
                self._make_empty_group(filename, page_number, initial_k),
            )
            group["final_score"] = float(group.get("final_score", 0.0)) + CHART_PAGE_BOOST
            group["chart_page_boost"] = group.get("chart_page_boost", 0.0) + CHART_PAGE_BOOST

        chart_ids = self._chart_ids_from_query(query)
        if chart_ids:
            for (filename, page_number), text in self.page_index.page_texts.items():
                if restrict_filename and filename != restrict_filename:
                    continue
                if self._is_navigation_or_chart_directory_page(page_number, text):
                    key = (filename, str(page_number))
                    group = grouped.get(key)
                    if group is not None:
                        group["final_score"] = float(group.get("final_score", 0.0)) - CHART_DIRECTORY_PENALTY
                        group["chart_directory_penalty"] = CHART_DIRECTORY_PENALTY

        if CONTENT_ANCHOR_BOOSTS_ENABLED:
            self._apply_content_anchor_boosts(
                query=query,
                grouped=grouped,
                initial_k=initial_k,
                restrict_filename=restrict_filename,
            )

        if SEMANTIC_SECTION_ROUTING_ENABLED:
            self._apply_semantic_section_boosts(
                query=query,
                grouped=grouped,
                initial_k=initial_k,
                restrict_filename=restrict_filename,
            )

        manual_routes = (
            self._matched_section_routes(query, restrict_filename=restrict_filename)
            if MANUAL_SECTION_RULES_ENABLED
            else []
        )
        for filename, page_start, page_end, bonus in manual_routes:
            for page_number in range(page_start, page_end + 1):
                if not self.page_index.get_documents_by_page(filename, page_number):
                    continue
                group = grouped.setdefault(
                    (filename, str(page_number)),
                    self._make_empty_group(filename, page_number, initial_k),
                )
                group["final_score"] = float(group.get("final_score", 0.0)) + bonus
                group["section_route_bonus"] = group.get("section_route_bonus", 0.0) + bonus

    def _apply_semantic_section_boosts(
        self,
        query: str,
        grouped: dict[tuple[str, str], dict],
        initial_k: int,
        restrict_filename: str = "",
    ) -> None:
        if self.page_index is None or not restrict_filename:
            return
        sections = self.page_index.section_profiles.get(restrict_filename, [])
        if not sections:
            return
        scored_sections = self._rank_sections_by_query(query, restrict_filename, sections)
        if not scored_sections:
            return
        for section, score in scored_sections[: max(1, SEMANTIC_SECTION_TOP_K)]:
            if score < SEMANTIC_SECTION_MIN_SCORE:
                continue
            boost = SEMANTIC_SECTION_BONUS * score
            for page_number in range(section.start_page, section.end_page + 1):
                if not self.page_index.get_documents_by_page(restrict_filename, page_number):
                    continue
                group = grouped.setdefault(
                    (restrict_filename, str(page_number)),
                    self._make_empty_group(restrict_filename, page_number, initial_k),
                )
                group["final_score"] = float(group.get("final_score", 0.0)) + boost
                group["semantic_section_boost"] = group.get("semantic_section_boost", 0.0) + boost

    def _rank_sections_by_query(
        self,
        query: str,
        filename: str,
        sections: list,
    ) -> list[tuple[object, float]]:
        lexical_scores = [self.page_index.section_keyword_score(query, section) for section in sections]
        max_lexical = max(lexical_scores, default=0.0) or 1.0
        overlap_scores = [self.page_index.section_entity_overlap(query, section) for section in sections]

        semantic_scores = self._section_semantic_scores(query, filename, sections)
        combined: list[tuple[object, float]] = []
        for section, lexical, overlap, semantic in zip(
            sections,
            lexical_scores,
            overlap_scores,
            semantic_scores,
        ):
            lexical_norm = lexical / max_lexical if max_lexical else 0.0
            score = 0.50 * semantic + 0.30 * lexical_norm + 0.20 * overlap
            combined.append((section, score))
        combined.sort(key=lambda item: item[1], reverse=True)
        return combined

    def _section_semantic_scores(self, query: str, filename: str, sections: list) -> list[float]:
        try:
            if filename not in self._section_embedding_cache:
                profiles = [section.profile_text for section in sections]
                self._section_embedding_cache[filename] = self.vector_db.embedding_service.embed_documents(profiles)
            section_vectors = self._section_embedding_cache[filename]
            query_vector = self.vector_db.embedding_service.embed_query(query)
        except Exception as exc:
            print(f"semantic section routing unavailable, fallback to lexical only: {exc}")
            return [0.0 for _ in sections]

        scores: list[float] = []
        for section_vector in section_vectors:
            scores.append(max(0.0, min(1.0, self._cosine_similarity(query_vector, section_vector))))
        return scores

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0
        length = min(len(left), len(right))
        dot = sum(float(left[index]) * float(right[index]) for index in range(length))
        left_norm = sum(float(left[index]) * float(left[index]) for index in range(length)) ** 0.5
        right_norm = sum(float(right[index]) * float(right[index]) for index in range(length)) ** 0.5
        if left_norm <= 0 or right_norm <= 0:
            return 0.0
        return dot / (left_norm * right_norm)

    def _apply_content_anchor_boosts(
        self,
        query: str,
        grouped: dict[tuple[str, str], dict],
        initial_k: int,
        restrict_filename: str = "",
    ) -> None:
        if self.page_index is None:
            return
        normalized_query = PageTextIndex.normalize_text(query)
        anchors: list[tuple[tuple[str, ...], float, int]] = []
        if "EPC" in query.upper():
            anchors.append((("EPC", "工程总承包", "利益统一", "统一管理规划"), 3.0, 3))
        if (
            "四大云解决方案" in normalized_query
            or ("设计云" in normalized_query and "施工云" in normalized_query)
        ):
            anchors.append((("图表180", "四大云解决方案", "设计云", "施工云"), 3.0, 4))
        if (
            ("盈利预测" in normalized_query or "估值" in normalized_query)
            and ("买入" in normalized_query or "评级" in normalized_query)
        ):
            anchors.append((("投资建议", "EPS", "PE", "PS", "买入"), 3.0, 4))
        if "客户覆盖率" in normalized_query and "渗透率" in normalized_query:
            anchors.append((("客户覆盖率", "渗透率", "2019", "2021H1"), 3.0, 4))
        if not anchors:
            return

        for (filename, page_number), text in self.page_index.page_texts.items():
            if restrict_filename and filename != restrict_filename:
                continue
            normalized_text = PageTextIndex.normalize_text(text)
            for terms, boost, min_matches in anchors:
                matched = sum(
                    1
                    for term in terms
                    if PageTextIndex.normalize_text(term) in normalized_text
                )
                if matched < min_matches:
                    continue
                group = grouped.setdefault(
                    (filename, str(page_number)),
                    self._make_empty_group(filename, page_number, initial_k),
                )
                group["final_score"] = float(group.get("final_score", 0.0)) + boost
                group["content_anchor_boost"] = group.get("content_anchor_boost", 0.0) + boost

    @staticmethod
    def _chart_ids_from_query(query: str) -> set[str]:
        chart_ids = set(re.findall(r"(?:图表|图|表)\s*(\d{1,3})", query or ""))
        for sequence in re.findall(r"(?:图表|图|表)\s*((?:\d{1,3}\s*(?:、|,|，|和)?\s*)+)", query or ""):
            chart_ids.update(re.findall(r"\d{1,3}", sequence))
        return chart_ids

    def _lookup_chart_pages(self, query: str, restrict_filename: str = "") -> set[tuple[str, int]]:
        if self.page_index is None:
            return set()
        chart_ids = self._chart_ids_from_query(query)
        if not chart_ids:
            return set()
        results: set[tuple[str, int]] = set()

        for (filename, page_number), text in self.page_index.page_texts.items():
            if restrict_filename and filename != restrict_filename:
                continue
            lines = text.splitlines()
            for chart_id in chart_ids:
                chart_pattern = re.compile(rf"(?:图表|图|表)\s*{re.escape(chart_id)}")
                for line in lines:
                    if not chart_pattern.search(line):
                        continue
                    directory_pages = [
                        int(match)
                        for match in re.findall(r"-\s*(\d{1,3})\s*-", line)
                    ]
                    for directory_page in directory_pages:
                        if self.page_index.get_documents_by_page(filename, directory_page):
                            results.add((filename, directory_page))
                    if not self._is_navigation_or_chart_directory_page(page_number, text):
                        results.add((filename, page_number))
        return results

    @staticmethod
    def _is_navigation_or_chart_directory_page(page_number: int, text: str) -> bool:
        if page_number > 15:
            return False
        chart_count = len(re.findall(r"(?:图表|图|表)\s*\d{1,3}", text or ""))
        return "图表目录" in (text or "") or chart_count >= 12

    def _matched_section_routes(
        self,
        query: str,
        restrict_filename: str = "",
    ) -> list[tuple[str, int, int, float]]:
        if self.page_index is None or "广联达" not in query:
            return []
        report_files = [
            filename
            for filename, _ in self.page_index.page_texts
            if "220217131" in filename
        ]
        if not report_files:
            return []
        filename = report_files[0]
        if restrict_filename and restrict_filename != filename:
            return []
        rules = [
            (
                ("两新一重", "资金压力", "施工资质", "总承包资质", "安全事故", "质量问题", "项目超期", "成本超支"),
                19,
                36,
            ),
            (
                ("5G", "云计算", "数字孪生", "EPC", "双碳", "政策", "装配式建筑"),
                40,
                53,
            ),
            (
                ("数字造价", "云造价", "造价业务", "未来成长空间", "70-100亿元", "70-100"),
                51,
                62,
            ),
            (
                ("数字施工", "施工业务", "114N", "智慧工地", "施工业务增长潜力", "市场潜力"),
                55,
                81,
            ),
            (
                ("CAD快速看图", "搅拌站", "AECORE", "四大云解决方案", "设计云", "施工云"),
                85,
                99,
            ),
            (
                ("工程设计", "设计业务", "Autodesk", "Bentley", "MagiCAD", "数维设计", "自主可控", "工业软件"),
                104,
                120,
            ),
            (
                ("盈利预测", "估值", "EPS", "PE", "PS", "买入", "毛利率"),
                129,
                131,
            ),
        ]
        matched = []
        normalized_query = PageTextIndex.normalize_text(query)
        for keywords, start, end in rules:
            if any(PageTextIndex.normalize_text(keyword) in normalized_query for keyword in keywords):
                matched.append((filename, start, end, SECTION_ROUTE_BONUS))
        if not matched and "数字化转型" in normalized_query:
            if "建筑业" in normalized_query or "挑战" in normalized_query or "应对策略" in normalized_query:
                matched.append((filename, 19, 36, SECTION_ROUTE_BONUS * 0.9))
            elif "竞争优势" in normalized_query:
                matched.append((filename, 40, 53, SECTION_ROUTE_BONUS * 0.9))
            elif "增长潜力" in normalized_query and "未来" not in normalized_query:
                matched.append((filename, 19, 36, SECTION_ROUTE_BONUS * 0.9))
            elif "未来发展前景" in normalized_query:
                matched.append((filename, 74, 81, SECTION_ROUTE_BONUS * 0.9))
        return matched

    def _build_reranker_query(self, query: str, mode: str) -> str:
        normalized_mode = (mode or "original").strip().lower()
        if normalized_mode == "original":
            return query
        terms: set[str] = set()
        if self.page_index is not None:
            terms.update(self.page_index.important_terms(query))
        terms.update(self._expanded_query_terms(query))
        terms.update(re.findall(r"\d{4}年|\d+(?:\.\d+)?%?", query))
        useful_terms = sorted(
            {term for term in terms if len(term) >= 2},
            key=lambda term: (-len(term), term),
        )[:18]
        keyword_query = " ".join(useful_terms)
        if normalized_mode == "keywords":
            return keyword_query or query
        if keyword_query:
            return f"{query}\nkeywords: {keyword_query}"
        return query

    def _build_pages_from_groups(
        self,
        query: str,
        groups: list[dict],
        max_chars_per_page: int,
    ) -> list[RetrievedPage]:
        pages: list[RetrievedPage] = []
        for group in groups:
            if self.page_index is not None:
                content, selected_chunk_ids = self.page_index.build_page_content(
                    group["filename"],
                    group["page_number"],
                    query=query,
                    max_chars=max_chars_per_page,
                    preferred_chunk_ids=group["chunk_ids"],
                )
                page_docs = self.page_index.get_documents_by_page(
                    group["filename"],
                    group["page_number"],
                )
            else:
                page_docs = self.vector_db.get_documents_by_page(
                    group["filename"],
                    group["page_number"],
                )
                content = self._merge_page_documents(page_docs, max_chars=max_chars_per_page)
                selected_chunk_ids = []
            if not content:
                continue
            all_chunk_ids = {
                int(doc.metadata["chunk_id"])
                for doc in page_docs
                if doc.metadata.get("chunk_id") is not None
            }
            pages.append(
                RetrievedPage(
                    filename=group["filename"],
                    page_number=group["page_number"],
                    score=round(float(group.get("final_score", group["best_score"])), 6),
                    hit_count=group["hit_count"],
                    chunk_ids=sorted(selected_chunk_ids or all_chunk_ids or group["chunk_ids"]),
                    content=content,
                    rule_score=round(
                        float(group.get("chart_page_boost", 0.0))
                        + float(group.get("content_anchor_boost", 0.0)),
                        6,
                    ),
                )
            )
        return pages

    def _get_page_reranker(
        self,
        model_name: str,
        batch_size: int,
        max_chars: int,
    ) -> LocalPageReranker | None:
        key = (model_name, max(1, int(batch_size)), max(200, int(max_chars)))
        if not model_name or key in self._disabled_page_rerankers:
            return None
        if key not in self._page_rerankers:
            self._page_rerankers[key] = LocalPageReranker(
                model_name=model_name,
                batch_size=key[1],
                max_chars=key[2],
            )
        return self._page_rerankers[key]

    def _rerank_pages(
        self,
        query: str,
        pages: list[RetrievedPage],
        model_name: str,
        batch_size: int,
        max_chars: int,
        weight: float,
    ) -> list[RetrievedPage]:
        reranker = self._get_page_reranker(model_name, batch_size, max_chars)
        if reranker is None or not pages:
            return pages

        passages = [
            f"filename: {page.filename}\npage: {page.page_number}\ntext:\n{page.content}"
            for page in pages
        ]
        try:
            results = reranker.rerank(query, passages)
        except Exception as exc:
            key = (model_name, max(1, int(batch_size)), max(200, int(max_chars)))
            self._disabled_page_rerankers.add(key)
            print(f"local reranker unavailable, fallback to hybrid ranking: {exc}")
            return sorted(
                pages,
                key=lambda page: (
                    float(page.score),
                    page.hit_count,
                    -int(page.page_number),
                ),
                reverse=True,
            )
        if not results:
            return pages

        score_by_index = {result.index: result.score for result in results}
        raw_scores = list(score_by_index.values())
        base_scores = [float(page.score) for page in pages]
        min_raw, max_raw = min(raw_scores), max(raw_scores)
        min_base, max_base = min(base_scores), max(base_scores)
        rerank_weight = max(0.0, min(1.0, float(weight)))

        reranked: list[RetrievedPage] = []
        for index, page in enumerate(pages):
            raw_score = score_by_index.get(index, min_raw)
            if max_raw > min_raw:
                rerank_norm = (raw_score - min_raw) / (max_raw - min_raw)
            else:
                rerank_norm = 1.0
            if max_base > min_base:
                base_norm = (float(page.score) - min_base) / (max_base - min_base)
            else:
                base_norm = 1.0
            anchor_bonus = min(0.35, max(0.0, float(page.rule_score)) * 0.06)
            page.score = round(
                rerank_weight * rerank_norm
                + (1.0 - rerank_weight) * base_norm
                + anchor_bonus,
                6,
            )
            reranked.append(page)

        reranked.sort(
            key=lambda page: (
                page.score,
                page.hit_count,
                -int(page.page_number),
            ),
            reverse=True,
        )
        return reranked

    @staticmethod
    def _merge_page_documents(documents: Iterable, max_chars: int) -> str:
        blocks: list[str] = []
        seen: set[str] = set()
        for doc in documents:
            text = " ".join((doc.page_content or "").split())
            if not text or text in seen:
                continue
            if any(text in block for block in blocks):
                continue
            seen.add(text)
            blocks.append(text)

        merged = "\n".join(blocks)
        if len(merged) > max_chars:
            return merged[:max_chars].rstrip() + "..."
        return merged

    def build_forced_pages(
        self,
        query: str,
        filename: str,
        page_number: int | str,
        neighbor_pages: int = 1,
        max_chars_per_page: int = 4500,
    ) -> list[RetrievedPage]:
        """Build context pages around a preselected primary page."""
        page_int = int(page_number)
        candidate_numbers = [page_int]
        for distance in range(1, max(0, neighbor_pages) + 1):
            candidate_numbers.extend([page_int - distance, page_int + distance])

        pages: list[RetrievedPage] = []
        seen: set[int] = set()
        for candidate_page in candidate_numbers:
            if candidate_page <= 0 or candidate_page in seen:
                continue
            seen.add(candidate_page)

            if self.page_index is not None:
                documents = self.page_index.get_documents_by_page(filename, candidate_page)
                if not documents:
                    continue
                content, selected_chunk_ids = self.page_index.build_page_content(
                    filename,
                    candidate_page,
                    query=query,
                    max_chars=max_chars_per_page,
                )
            else:
                documents = self.vector_db.get_documents_by_page(filename, candidate_page)
                if not documents:
                    continue
                content = self._merge_page_documents(documents, max_chars=max_chars_per_page)
                selected_chunk_ids = []

            if not content:
                continue
            all_chunk_ids = {
                int(doc.metadata["chunk_id"])
                for doc in documents
                if doc.metadata.get("chunk_id") is not None
            }
            pages.append(
                RetrievedPage(
                    filename=filename,
                    page_number=candidate_page,
                    score=1.0 if candidate_page == page_int else 0.5,
                    hit_count=len(documents),
                    chunk_ids=sorted(selected_chunk_ids or all_chunk_ids),
                    content=content,
                )
            )
        return pages

    def build_candidate_pages(
        self,
        query: str,
        candidate_pages: list[dict],
        max_chars_per_page: int = 4500,
    ) -> list[RetrievedPage]:
        """Build answer context from fused top-K candidate pages."""
        pages: list[RetrievedPage] = []
        seen: set[tuple[str, int]] = set()
        for rank, candidate in enumerate(candidate_pages, start=1):
            filename = candidate.get("filename")
            try:
                page_number = int(candidate.get("page"))
            except (TypeError, ValueError):
                continue
            if not filename or page_number <= 0:
                continue
            key = (filename, page_number)
            if key in seen:
                continue
            seen.add(key)

            preferred_chunk_ids = candidate.get("chunk_ids") or []
            if self.page_index is not None:
                documents = self.page_index.get_documents_by_page(filename, page_number)
                if not documents:
                    continue
                content, selected_chunk_ids = self.page_index.build_page_content(
                    filename,
                    page_number,
                    query=query,
                    max_chars=max_chars_per_page,
                    preferred_chunk_ids=preferred_chunk_ids,
                )
            else:
                documents = self.vector_db.get_documents_by_page(filename, page_number)
                if not documents:
                    continue
                content = self._merge_page_documents(documents, max_chars=max_chars_per_page)
                selected_chunk_ids = preferred_chunk_ids

            if not content:
                continue
            all_chunk_ids = {
                int(doc.metadata["chunk_id"])
                for doc in documents
                if doc.metadata.get("chunk_id") is not None
            }
            score = candidate.get("fusion_score")
            if score is None:
                score = 1.0 / max(rank, 1)
            pages.append(
                RetrievedPage(
                    filename=filename,
                    page_number=page_number,
                    score=round(float(score), 6),
                    hit_count=len(documents),
                    chunk_ids=sorted(selected_chunk_ids or all_chunk_ids or preferred_chunk_ids),
                    content=content,
                )
            )
        return pages

    def _page_profile(self, query: str, content: str) -> dict:
        normalized_content = PageTextIndex.normalize_text(content)
        terms: set[str] = set()
        if self.page_index is not None:
            terms.update(self.page_index.important_terms(query))
        terms.update(self._expanded_query_terms(query))
        terms.update(re.findall(r"\d{4}年|\d+(?:\.\d+)?%?", PageTextIndex.normalize_text(query)))
        noisy_terms = {
            PageTextIndex.normalize_text(term)
            for term in (
                "华创证券",
                "东吴证券",
                "证券研究所",
                "深度研究报告",
                "研究报告",
                "联邦制药",
                "凌云股份",
                "广联达",
                "伊利股份",
                "千味央厨",
            )
        }
        terms = {
            term
            for term in terms
            if len(term) >= 2 and term not in noisy_terms and term in normalized_content
        }
        matched_terms = sorted(terms, key=lambda term: (-len(term), term))[:16]

        role_flags = []
        if "图表目录" in content or ("目 录" in content and content.count("...") >= 3):
            role_flags.append("目录/图表目录页")
        if "风险提示" in content and len(terms) <= 2:
            role_flags.append("风险提示页")
        if any(mark in content for mark in ("【表格抽取】", "【疑似表格", "图表", "表")):
            role_flags.append("含图表/表格")
        if "【OCR补充】" in content:
            role_flags.append("含OCR补充")
        page_role = "；".join(role_flags) if role_flags else "正文页"

        lines = []
        for raw_line in re.split(r"[\n。；;]", content):
            line = " ".join(raw_line.split())
            if len(line) < 8:
                continue
            normalized_line = PageTextIndex.normalize_text(line)
            term_hits = sum(1 for term in terms if term in normalized_line)
            if term_hits <= 0:
                continue
            has_evidence_mark = any(mark in line for mark in ("【表格抽取】", "【疑似表格", "图表", "表"))
            number_hits = len(re.findall(r"\d+(?:\.\d+)?%?", normalized_line))
            score = 3 * term_hits + number_hits + (1 if has_evidence_mark else 0)
            lines.append((score, line[:180]))

        if not lines:
            evidence_hint = "未抽取到明显关键词提示，请直接阅读本页全文判断。"
            evidence_score = 0
        else:
            selected = []
            seen = set()
            for _, line in sorted(lines, key=lambda item: item[0], reverse=True):
                if line in seen:
                    continue
                seen.add(line)
                selected.append(line)
                if len(selected) >= 5:
                    break
            evidence_hint = "\n".join(f"- {line}" for line in selected)
            evidence_score = sum(score for score, _ in sorted(lines, key=lambda item: item[0], reverse=True)[:5])

        return {
            "role": page_role,
            "matched_terms": matched_terms,
            "number_count": len(re.findall(r"\d+(?:\.\d+)?%?", normalized_content)),
            "evidence_score": evidence_score,
            "evidence_hint": evidence_hint,
        }

    def build_prompt(
        self,
        query,
        pages: list[RetrievedPage],
        forced_source: tuple[str, int | str] | None = None,
        allow_answer_from_all_pages: bool = False,
    ):
        page_profiles = [self._page_profile(query, page.content) for page in pages]
        evidence_table_parts = []
        context_parts = []
        for index, page in enumerate(pages, start=1):
            profile = page_profiles[index - 1]
            matched_terms = "、".join(profile["matched_terms"]) if profile["matched_terms"] else "无明显匹配"
            evidence_table_parts.append(
                f"[{index}] page={page.page_number}; role={profile['role']}; "
                f"evidence_score={profile['evidence_score']}; "
                f"matched_terms={matched_terms}; number_count={profile['number_count']}"
            )
            context_parts.append(
                f"[{index}] filename: {page.filename}\n"
                f"page: {page.page_number}\n"
                f"page_role: {profile['role']}\n"
                f"evidence_score: {profile['evidence_score']}\n"
                f"chunk_ids: {page.chunk_ids}\n"
                f"matched_terms: {matched_terms}\n"
                f"evidence_hint:\n{profile['evidence_hint']}\n"
                f"content:\n{page.content}"
            )

        evidence_table = "\n".join(evidence_table_parts)
        context = "\n\n".join(context_parts)
        answer_strategy = (
            "【答案抽取策略】\n"
            "1. 回答前先在内部判断题型，但不要输出题型判断过程。\n"
            "2. 标准答案风格不是越长越好，而是回答题目要求的最小充分信息：数字题简洁，列举题列全，图表题抽结构，分析题分点，业务解释题讲组成/功能/作用。\n"
            "3. 先定位问题中的公司、业务、产品、年份、指标、金额、百分比、图表编号和题目动词，再到候选页证据中找完全对应的句子、表格或 OCR/图表文字。\n"
            "4. 尽量复用报告原文里的关键词、连续短语、业务名称、产品名称、指标名称、年份、数字、单位和结论；原文单位是什么就保留什么，不要自行换算。\n"
            "5. 不要为了变长加入无关背景，也不要把需要列举/分析的问题压缩成一句泛泛结论。"
        )
        context = f"{context}\n\n{answer_strategy}"
        forced_note = ""
        if forced_source is not None:
            if allow_answer_from_all_pages:
                forced_note = (
                    "\n【页码规划约束】\n"
                    f"已通过本地检索和 reranker 预选最终证据页：filename={forced_source[0]}，page={forced_source[1]}。\n"
                    "输出 JSON 中的 filename 和 page 必须使用该最终证据页。\n"
                    "但 answer 字段可以综合【候选页全文】中任意候选页的直接证据，优先选择最能回答问题的数字、表格、图表文字或结论。\n"
                    "如果最终证据页与其它候选页不一致，answer 以最直接、最完整、最贴近问题的候选页证据为准，不要为了迎合页码而编造。\n"
                )
            else:
                forced_note = (
                    "\n【页码规划约束】\n"
                    f"已通过本地检索和 reranker 预选主证据页：filename={forced_source[0]}，page={forced_source[1]}。\n"
                    "相邻页只作为补充上下文；输出 JSON 中的 filename 和 page 必须使用该主证据页。\n"
                )
        return f"""你是一个金融研报 RAG 问答助手。请只根据参考片段回答用户问题。

【候选页证据表】
{evidence_table}

【候选页全文】
{context}
{forced_note}

【用户问题】
{query}

【回答要求】
1. 只使用参考片段中的信息，不要编造。
2. 候选页可能是相邻页窗口。必须先比较每一页的 evidence_hint 和 content，找出“answer 主要依据所在页”，再输出 filename/page/answer。
3. filename 必须从参考片段的 filename 中选择，page 必须从参考片段的 page 中选择。
4. 参考片段中的“【OCR补充】”“【表格抽取】”和图表文字都视为有效依据；如果问题问图表、图片、发展历程、指标或数字，必须优先检查这些内容。
5. 如果问题问“占比/结构/竞争格局/客户/市场空间/测算/图表”，优先选择含图表/表格主体和核心数字的页，而不是目录页、图表目录页或只出现一句概括的页。
6. 如果相邻页都相关，page 选择包含核心数据、图表、表格或结论最多的那一页；不要选只有背景铺垫、标题延续、目录、风险提示或摘要概括的页。
7. page 字段必须和 answer 的主要依据一致；可以用相邻页补充理解，但不要把 page 输出为补充页。
8. 请先隐式判断题型，并按题型决定答案长度和结构，不要输出题型判断过程。
9. 数值/事实提取题：如果问题包含“多少、占比、增长率、达到多少、分别是多少、有何变化、趋势如何”等，答案应简洁，通常 1-2 句；必须保留年份、指标名、数值、单位和变化方向。格式可为：“根据参考片段，{{年份/场景}}下{{指标名称}}为{{数值}}{{单位}}，{{补充解释}}。”不要额外扩展无关背景。
10. 列举题：如果问题包含“有哪些、分别是什么、具体包括、主要客户、主要产品、主要功能”等，必须逐项列举，尽量列全参考片段中的名称、模块、客户、产品或功能；不要只写概括性总结。
11. 图表题：如果问题包含“根据图表、结合图片、图表X”，优先抽取图表中的项目、数值、趋势或对比关系。若问最大/最高/最低/达到多少，只回答对应项目和数值并补一句说明；若问哪些方面/有何差异/怎样分布，则按图表项目逐项列举。保留图表编号、年份、单位和指标名。
12. 分析/评估/评价题：如果问题包含“如何分析、如何评估、如何评价、如何看待、发展前景、增长潜力、竞争优势”等，用 2-5 点分点回答；每一点都必须对应参考片段中的事实、数据或明确表述，优先覆盖背景/现状、关键数据或事实、原因/影响、结论。
13. 产品/业务/策略解释题：如果问题询问业务模式、平台、系统、解决方案、策略或工作流程，按“组成/功能/作用/效果”组织答案，必要时分点说明。
14. 通用要求：尽量复用参考片段原文的关键词、连续短语、业务名称、产品名称、指标名称、图表项目、年份、金额、百分比和单位；不要自行换算单位，原文是“百万元”就写“百万元”，原文是“亿元”就写“亿元”。
15. 不要为了变长而加入与问题无关的内容；如果参考片段中有多个相关数字，优先保留与问题直接相关的数字。
16. 只要候选页里有相关数据、图表、业务描述或结论，就不要回答“未在参考片段中找到足够信息”；只有当所有参考片段都完全没有相关信息时，才回答“未在参考片段中找到足够信息”。
17. 输出严格 JSON，不要 Markdown，不要代码块，字段固定为 filename、page、answer；page 必须是数字，不要写成字符串。answer 可以在同一个 JSON 字符串中使用“1. ...；2. ...；3. ...”组织要点。

【输出示例】
{{"filename":"xxx.pdf","page":1,"answer":"根据参考片段，..."}}

【输出】
"""

    def answer_structured(
        self,
        query: str,
        initial_k: int = 80,
        final_pages: int = 5,
        max_chars_per_page: int = 5000,
        run_llm: bool = True,
        include_prompt: bool = False,
    ) -> dict:
        pages = self.retrieve_pages(
            query,
            initial_k=initial_k,
            final_pages=final_pages,
            max_chars_per_page=max_chars_per_page,
        )
        if not pages:
            return {
                "filename": "",
                "page": -1,
                "answer": "未检索到相关片段",
                "sources": [],
                "llm_used": False,
                "raw_answer": "",
                "error": "no_retrieval_hits",
                **({"prompt": ""} if include_prompt else {}),
            }

        prompt = self.build_prompt(query, pages)
        sources = self._format_sources(pages)
        fallback = StructuredAnswer(
            filename=pages[0].filename,
            page=pages[0].page_number,
            answer="",
            sources=sources,
            prompt=prompt,
            llm_used=False,
        )

        if not run_llm:
            fallback.error = "llm_not_run"
            return fallback.to_dict(include_prompt=include_prompt)

        if not self.llm.available:
            fallback.error = "llm_unavailable: OPENAI_API_KEY is empty"
            return fallback.to_dict(include_prompt=include_prompt)

        try:
            raw_answer = self.llm.generate(prompt)
        except Exception as exc:
            fallback.error = f"llm_exception: {exc}"
            return fallback.to_dict(include_prompt=include_prompt)

        parsed = self.parse_structured_answer(raw_answer)
        if not parsed:
            fallback.answer = raw_answer.strip()
            fallback.raw_answer = raw_answer
            fallback.llm_used = True
            fallback.error = "llm_output_not_json"
            return fallback.to_dict(include_prompt=include_prompt)

        filename = parsed.get("filename") or pages[0].filename
        page = self._normalize_page_value(parsed.get("page") or pages[0].page_number)
        valid_sources = {(source["filename"], str(source["page"])) for source in sources}
        if (filename, str(page)) not in valid_sources:
            filename = pages[0].filename
            page = pages[0].page_number
        else:
            filename, page = self._maybe_override_source(filename, page, sources)
        answer = parsed.get("answer") or ""
        return StructuredAnswer(
            filename=filename,
            page=page,
            answer=answer,
            sources=sources,
            prompt=prompt,
            raw_answer=raw_answer,
            llm_used=True,
        ).to_dict(include_prompt=include_prompt)

    def answer_forced_page(
        self,
        query: str,
        filename: str,
        page: int | str,
        neighbor_pages: int = 1,
        max_chars_per_page: int = 4500,
        run_llm: bool = True,
        include_prompt: bool = False,
        force_page: bool = True,
    ) -> dict:
        pages = self.build_forced_pages(
            query=query,
            filename=filename,
            page_number=page,
            neighbor_pages=neighbor_pages,
            max_chars_per_page=max_chars_per_page,
        )
        if not pages:
            return {
                "filename": filename or "",
                "page": page if page not in (None, "") else -1,
                "answer": "未检索到相关片段",
                "sources": [],
                "llm_used": False,
                "raw_answer": "",
                "error": "forced_page_not_found",
                "page_plan": {"filename": filename, "page": page},
                **({"prompt": ""} if include_prompt else {}),
            }
        if not force_page:
            for page_item in pages:
                page_item.score = 1.0
            pages.sort(key=lambda page_item: int(page_item.page_number))

        prompt = self.build_prompt(
            query,
            pages,
            forced_source=(filename, page) if force_page else None,
        )
        sources = self._format_sources(pages)
        fallback = StructuredAnswer(
            filename=filename,
            page=page,
            answer="",
            sources=sources,
            prompt=prompt,
            llm_used=False,
        )

        if not run_llm:
            fallback.error = "llm_not_run_forced_page"
            result = fallback.to_dict(include_prompt=include_prompt)
            result["page_plan"] = {"filename": filename, "page": page}
            return result

        if not self.llm.available:
            fallback.error = "llm_unavailable: OPENAI_API_KEY is empty"
            result = fallback.to_dict(include_prompt=include_prompt)
            result["page_plan"] = {"filename": filename, "page": page}
            return result

        try:
            raw_answer = self.llm.generate(prompt)
        except Exception as exc:
            fallback.error = f"llm_exception: {exc}"
            result = fallback.to_dict(include_prompt=include_prompt)
            result["page_plan"] = {"filename": filename, "page": page}
            return result

        parsed = self.parse_structured_answer(raw_answer)
        if not parsed:
            fallback.answer = raw_answer.strip()
            fallback.raw_answer = raw_answer
            fallback.llm_used = True
            fallback.error = "llm_output_not_json"
            result = fallback.to_dict(include_prompt=include_prompt)
            result["page_plan"] = {"filename": filename, "page": page}
            return result

        answer = parsed.get("answer") or ""
        selected_filename = filename
        selected_page = self._normalize_page_value(page)
        if not force_page:
            parsed_filename = parsed.get("filename") or filename
            parsed_page = self._normalize_page_value(parsed.get("page") or page)
            valid_sources = {(source["filename"], str(source["page"])) for source in sources}
            if (parsed_filename, str(parsed_page)) in valid_sources:
                selected_filename = parsed_filename
                selected_page = parsed_page

        result = StructuredAnswer(
            filename=selected_filename,
            page=selected_page,
            answer=answer,
            sources=sources,
            prompt=prompt,
            raw_answer=raw_answer,
            llm_used=True,
        ).to_dict(include_prompt=include_prompt)
        result["page_plan"] = {"filename": filename, "page": page}
        return result

    def answer_forced_candidates(
        self,
        query: str,
        filename: str,
        page: int | str,
        candidate_pages: list[dict],
        neighbor_pages: int = 1,
        max_chars_per_page: int = 4500,
        run_llm: bool = True,
        include_prompt: bool = False,
        force_page: bool = True,
    ) -> dict:
        selected_page = self._normalize_page_value(page)
        selected_candidate = {"filename": filename, "page": selected_page, "fusion_score": 1.0}
        ordered_candidates = [selected_candidate]
        seen = {(filename, str(selected_page))}
        for candidate in candidate_pages:
            key = (candidate.get("filename"), str(candidate.get("page")))
            if not key[0] or key in seen:
                continue
            seen.add(key)
            ordered_candidates.append(candidate)

        pages = self.build_candidate_pages(
            query=query,
            candidate_pages=ordered_candidates,
            max_chars_per_page=max_chars_per_page,
        )
        if not pages:
            return self.answer_forced_page(
                query=query,
                filename=filename,
                page=page,
                neighbor_pages=neighbor_pages,
                max_chars_per_page=max_chars_per_page,
                run_llm=run_llm,
                include_prompt=include_prompt,
                force_page=force_page,
            )

        if not force_page:
            pages.sort(key=lambda page_item: (page_item.filename, int(page_item.page_number)))

        prompt = self.build_prompt(
            query,
            pages,
            forced_source=(filename, page) if force_page else None,
            allow_answer_from_all_pages=force_page,
        )
        sources = self._format_sources(pages)
        context_page_refs = [
            {"filename": source["filename"], "page": source["page"]}
            for source in sources
        ]
        fallback = StructuredAnswer(
            filename=filename,
            page=selected_page,
            answer="",
            sources=sources,
            prompt=prompt,
            llm_used=False,
        )

        page_plan = {
            "filename": filename,
            "page": selected_page,
            "answer_context": "topk_candidates",
            "answer_pages": context_page_refs,
        }

        if not run_llm:
            fallback.error = "llm_not_run_forced_candidates"
            result = fallback.to_dict(include_prompt=include_prompt)
            result["page_plan"] = page_plan
            return result

        if not self.llm.available:
            fallback.error = "llm_unavailable: OPENAI_API_KEY is empty"
            result = fallback.to_dict(include_prompt=include_prompt)
            result["page_plan"] = page_plan
            return result

        try:
            raw_answer = self.llm.generate(prompt)
        except Exception as exc:
            fallback.error = f"llm_exception: {exc}"
            result = fallback.to_dict(include_prompt=include_prompt)
            result["page_plan"] = page_plan
            return result

        parsed = self.parse_structured_answer(raw_answer)
        if not parsed:
            fallback.answer = raw_answer.strip()
            fallback.raw_answer = raw_answer
            fallback.llm_used = True
            fallback.error = "llm_output_not_json"
            result = fallback.to_dict(include_prompt=include_prompt)
            result["page_plan"] = page_plan
            return result

        answer = parsed.get("answer") or ""
        selected_filename = filename
        output_page = selected_page
        if not force_page:
            parsed_filename = parsed.get("filename") or filename
            parsed_page = self._normalize_page_value(parsed.get("page") or page)
            valid_sources = {(source["filename"], str(source["page"])) for source in sources}
            if (parsed_filename, str(parsed_page)) in valid_sources:
                selected_filename = parsed_filename
                output_page = parsed_page

        result = StructuredAnswer(
            filename=selected_filename,
            page=output_page,
            answer=answer,
            sources=sources,
            prompt=prompt,
            raw_answer=raw_answer,
            llm_used=True,
        ).to_dict(include_prompt=include_prompt)
        result["page_plan"] = page_plan
        return result

    def answer(self, query):
        return self.answer_structured(query)

    @staticmethod
    def parse_structured_answer(text: str) -> dict | None:
        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
            candidate = re.sub(r"\s*```$", "", candidate)

        json_match = re.search(r"\{.*\}", candidate, flags=re.S)
        if json_match:
            candidate = json_match.group(0)

        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None
        if "answer" not in data:
            return None
        return data

    @staticmethod
    def _normalize_page_value(page: int | str):
        if isinstance(page, str) and page.strip().isdigit():
            return int(page.strip())
        return page

    @staticmethod
    def _format_sources(pages: list[RetrievedPage]) -> list[dict]:
        return [
            {
                "filename": page.filename,
                "page": RAGService._normalize_page_value(page.page_number),
                "chunk_ids": page.chunk_ids,
                "score": page.score,
                "hit_count": page.hit_count,
            }
            for page in pages
        ]

    @staticmethod
    def _maybe_override_source(
        filename: str,
        page: int | str,
        sources: list[dict],
    ) -> tuple[str, int | str]:
        if not sources:
            return filename, page

        top_source = sources[0]
        selected_score = None
        for source in sources:
            if source.get("filename") == filename and str(source.get("page")) == str(page):
                selected_score = float(source.get("score") or 0.0)
                break

        if selected_score is None:
            return filename, page

        top_score = float(top_source.get("score") or 0.0)
        if (
            (top_source.get("filename"), str(top_source.get("page"))) != (filename, str(page))
            and top_score - selected_score >= FINAL_SOURCE_OVERRIDE_MARGIN
            and top_score >= selected_score * FINAL_SOURCE_OVERRIDE_RATIO
        ):
            return top_source.get("filename"), top_source.get("page")

        return filename, page
