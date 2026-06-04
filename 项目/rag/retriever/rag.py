from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

from config.config import (
    FINAL_SOURCE_OVERRIDE_MARGIN,
    FINAL_SOURCE_OVERRIDE_RATIO,
    RETRIEVAL_BONUS_WEIGHT,
    RETRIEVAL_CHUNKS_PATH,
    RETRIEVAL_EARLY_PAGE_PENALTY,
    RETRIEVAL_EXACT_WEIGHT,
    RETRIEVAL_KEYWORD_WEIGHT,
    RETRIEVAL_MODE,
    RETRIEVAL_VECTOR_WEIGHT,
)
from rag.llm.llm import LLMService
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

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "page_number": self.page_number,
            "score": self.score,
            "hit_count": self.hit_count,
            "chunk_ids": self.chunk_ids,
            "content": self.content,
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

        pages: list[RetrievedPage] = []
        for group in ranked_groups[:final_pages]:
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
                )
            )

        return pages

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
        forced_note = ""
        if forced_source is not None:
            forced_note = (
                "\n【页码规划约束】\n"
                f"已通过检索重排预选主证据页：filename={forced_source[0]}，page={forced_source[1]}。\n"
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
8. answer 要直接回答问题，尽量沿用参考片段原文中的关键词和表述，必须保留关键数字、时间、业务名称、图表项目或结论；不要泛泛扩展。
9. 如果问题要求“分析/评估/如何看待”，先列出参考片段里的具体依据，再给结论；优先用 2-5 个短句或分号分隔要点。
10. 只要候选页里有相关数据、图表、业务描述或结论，就不要回答“未在参考片段中找到足够信息”。
11. 只有当所有参考片段都完全没有相关信息时，才回答“未在参考片段中找到足够信息”。
12. 输出严格 JSON，不要 Markdown，不要代码块，字段固定为 filename、page、answer；page 必须是数字，不要写成字符串。

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
