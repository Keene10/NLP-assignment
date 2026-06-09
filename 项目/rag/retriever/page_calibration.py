from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from config.config import (
    PAGE_RERANKER_BATCH_SIZE,
    PAGE_RERANKER_MAX_CHARS,
    PAGE_RERANKER_MODEL,
    STRICT_LOCAL_RERANKER,
)
from rag.retriever.local_reranker import LocalPageReranker
from rag.retriever.page_index import PageTextIndex
from rag.retriever.rag import RAGService, RetrievedPage


DETAIL_TYPES = {"chart", "number", "financial", "period_data", "list"}
DETAIL_TERMS = ("具体", "解决方案", "产品定位", "主要功能", "应用场景", "分别")
ENTITY_SUFFIXES = (
    "平台",
    "系统",
    "方案",
    "业务",
    "产品",
    "工具",
    "模型",
    "药物",
    "品牌",
    "政策",
    "规划",
    "技术",
    "客户",
    "服务",
    "市场",
    "行业",
    "项目",
    "功能",
    "策略",
    "模式",
)
ADVANCED_ENTITY_SUFFIXES = ENTITY_SUFFIXES + (
    "云",
    "端",
    "线",
    "盒",
)
FINANCIAL_TERMS = (
    "营业收入",
    "营收",
    "净利润",
    "归母净利润",
    "毛利率",
    "收入",
    "利润",
    "EPS",
    "PE",
    "PS",
    "ROE",
    "CAGR",
    "盈利预测",
    "估值",
    "买入评级",
    "市场份额",
)


@dataclass
class PageDebug:
    filename: str
    page: int
    hybrid_score: float
    adjusted_score: float
    anchor_score: float
    anchor_hits: dict[str, list[str]]
    summary_penalty: bool = False


@dataclass
class StageResult:
    filename: str
    page: int
    pages: list[RetrievedPage]
    debug: dict


def normalize_text(text: str) -> str:
    return PageTextIndex.normalize_text(text)


def classify_question(question: str) -> str:
    if re.search(r"(?:图表|图|表)\s*\d{1,3}", question or ""):
        return "chart"
    if any(term in question for term in ("多少", "分别是多少", "达到多少", "占比", "增长率", "市场份额", "变化趋势")):
        return "number"
    if any(term in question for term in FINANCIAL_TERMS):
        return "financial"
    if re.search(r"(?:Q[1-4]|H[12]|一季度|二季度|三季度|四季度|第三季度|上半年|下半年)", question or "", re.I):
        return "period_data"
    if any(term in question for term in ("有哪些", "分别是什么", "具体有哪些", "主要客户", "主要功能", "应用场景")):
        return "list"
    if any(term in question for term in ("如何分析", "如何评估", "如何评价", "如何看待", "发展前景", "增长潜力", "竞争优势")):
        return "analysis"
    return "default"


def should_apply_summary_penalty(question: str, question_type: str) -> bool:
    return question_type in DETAIL_TYPES or any(term in question for term in DETAIL_TERMS)


def chart_ids(question: str) -> list[str]:
    ids = set(re.findall(r"(?:图表|图|表)\s*(\d{1,3})", question or ""))
    for sequence in re.findall(r"(?:图表|图|表)\s*((?:\d{1,3}\s*(?:、|,|，|和)?\s*)+)", question or ""):
        ids.update(re.findall(r"\d{1,3}", sequence))
    return sorted(ids, key=lambda value: int(value))


def year_terms(question: str) -> list[str]:
    return sorted(set(re.findall(r"(?:19|20)\d{2}年?", question or "")))


def period_terms(question: str) -> list[str]:
    pattern = r"(?:20\d{2}\s*[Hh][12]|[Qq][1-4]|[Hh][12]|一季度|二季度|三季度|四季度|第三季度|上半年|下半年)"
    return sorted(set(re.findall(pattern, question or "")))


def number_unit_terms(question: str) -> list[str]:
    units = "亿元|百万元|万元|元|亿|万|%|pct|百分点|倍|家|个|项|吨|万吨|支|万支|亿支|平方米|万平方米|人|万人|户"
    pattern = rf"\d+(?:,\d{{3}})*(?:\.\d+)?\s*(?:{units})"
    return sorted(set(re.findall(pattern, question or "", flags=re.I)))


def mixed_entity_terms(question: str) -> list[str]:
    pattern = r"\b[A-Za-z]{2,}[A-Za-z0-9+./-]*\d*[A-Za-z0-9+./-]*\b|\b\d+[A-Za-z][A-Za-z0-9+./-]*\b"
    terms = {term.strip() for term in re.findall(pattern, question or "") if len(term.strip()) >= 2}
    return sorted(terms, key=lambda item: (-len(item), item))[:12]


def advanced_mixed_entity_terms(question: str) -> list[str]:
    pattern = (
        r"\b[A-Za-z]{2,}[A-Za-z0-9+./-]*\d*[A-Za-z0-9+./-]*\b"
        r"|\b\d+(?:[A-Za-z]+|\+[A-Za-z0-9]+)[A-Za-z0-9+./-]*\b"
        r"|\b[A-Za-z0-9]+(?:\+[A-Za-z0-9]+)+\b"
    )
    terms = {term.strip() for term in re.findall(pattern, question or "") if len(term.strip()) >= 2}
    return sorted(terms, key=lambda item: (-len(item), item))[:16]


def quoted_terms(question: str) -> list[str]:
    terms = set()
    for term in re.findall(r"[“\"'《]([^”\"'》]{2,30})[”\"'》]", question or ""):
        cleaned = term.strip()
        if cleaned and not cleaned.endswith((".pdf", "PDF")):
            terms.add(cleaned)
    for term in re.findall(r"\b[A-Za-z0-9]+(?:\+[A-Za-z0-9]+)+\b", question or ""):
        terms.add(term.strip())
    return sorted(terms, key=lambda item: (-len(item), item))[:12]


def entity_phrases(question: str) -> list[str]:
    suffix_pattern = "|".join(re.escape(term) for term in ENTITY_SUFFIXES)
    pattern = rf"[\u4e00-\u9fffA-Za-z0-9+./-]{{2,24}}(?:{suffix_pattern})"
    phrases = set()
    for raw in re.findall(pattern, question or ""):
        cleaned = re.sub(r"^(根据|关于|请问|请|能否|详细解释一下|如何分析|如何评估|如何评价|如何看待)", "", raw)
        if len(cleaned) >= 3:
            phrases.add(cleaned)
    return sorted(phrases, key=lambda item: (-len(item), item))[:16]


def advanced_entity_phrases(question: str) -> list[str]:
    suffix_pattern = "|".join(re.escape(term) for term in ADVANCED_ENTITY_SUFFIXES)
    pattern = rf"[\u4e00-\u9fffA-Za-z0-9+./-]{{1,24}}(?:{suffix_pattern})"
    phrases = set(entity_phrases(question))
    for raw in re.findall(pattern, question or ""):
        cleaned = re.sub(
            r"^(根据|关于|请问|请|能否|详细解释一下|如何分析|如何评估|如何评价|如何看待|有哪些|具体有哪些)",
            "",
            raw,
        )
        cleaned = re.sub(r"^(在|其|该|的|和|与)", "", cleaned)
        if len(cleaned) >= 2 and not re.fullmatch(r"(业务|产品|客户|市场|行业|技术|政策|方案|系统|平台)", cleaned):
            phrases.add(cleaned)
    for term in quoted_terms(question):
        if len(term) <= 24:
            phrases.add(term)
    return sorted(phrases, key=lambda item: (-len(item), item))[:20]


def financial_terms(question: str) -> list[str]:
    normalized_question = normalize_text(question)
    terms = [
        term
        for term in FINANCIAL_TERMS
        if normalize_text(term) in normalized_question
    ]
    return sorted(set(terms), key=lambda item: (-len(item), item))


def is_strong_entity_question(question: str, page_index: PageTextIndex | None = None) -> bool:
    if chart_ids(question) or mixed_entity_terms(question):
        return True
    if len(entity_phrases(question)) >= 1:
        return True
    if page_index is not None and len(page_index.important_terms(question)) >= 3:
        return True
    return False


def has_strong_anchor_hits(hits: dict[str, list[str]]) -> bool:
    strong_categories = {
        "charts",
        "numbers",
        "financial_terms",
        "mixed_entities",
        "entity_phrases",
        "quoted_terms",
        "year_range",
    }
    return any(hits.get(category) for category in strong_categories)


def _add_hits(score: float, hits: dict[str, list[str]], category: str, values: Iterable[str]) -> float:
    values = [value for value in values if value]
    if values:
        hits.setdefault(category, [])
        for value in values:
            if value not in hits[category]:
                hits[category].append(value)
    return score


def is_chart_directory_page(page_number: int, text: str) -> bool:
    if page_number > 15:
        return False
    chart_count = len(re.findall(r"(?:图表|图|表)\s*\d{1,3}", text or ""))
    return "图表目录" in (text or "") or chart_count >= 12


def list_feature_score(text: str) -> float:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    bullet_lines = sum(1 for line in lines if re.match(r"^(?:[-*•]|[0-9一二三四五六七八九十]+[.、])", line))
    separator_lines = sum(1 for line in lines if line.count("、") + line.count("；") + line.count(";") >= 2)
    colon_lines = sum(1 for line in lines if "：" in line or ":" in line)
    return min(4.0, bullet_lines * 0.6 + separator_lines * 0.5 + colon_lines * 0.25)


def evidence_anchor_score(
    question: str,
    page_text: str,
    page_number: int = 0,
    page_index: PageTextIndex | None = None,
    question_type: str | None = None,
    advanced: bool = False,
) -> tuple[float, dict[str, list[str]]]:
    question_type = question_type or classify_question(question)
    normalized_text = normalize_text(page_text)
    hits: dict[str, list[str]] = {}
    score = 0.0

    matched_years = [term for term in year_terms(question) if normalize_text(term) in normalized_text]
    score += min(4.0, 2.0 * len(matched_years))
    _add_hits(score, hits, "years", matched_years)

    if advanced:
        years = year_terms(question)
        if len(years) >= 2:
            coverage = [term for term in years if normalize_text(term) in normalized_text]
            if len(coverage) >= 2:
                score += 3.0
                _add_hits(score, hits, "year_range", coverage)

    matched_periods = [term for term in period_terms(question) if normalize_text(term) in normalized_text]
    score += min(4.0, 2.0 * len(matched_periods))
    _add_hits(score, hits, "periods", matched_periods)

    matched_charts = []
    for chart_id in chart_ids(question):
        pattern = rf"(?:图表|图|表)\s*{re.escape(chart_id)}"
        if re.search(pattern, page_text or ""):
            matched_charts.append(chart_id)
            score += 1.0 if is_chart_directory_page(page_number, page_text) else 5.0
    _add_hits(score, hits, "charts", matched_charts)

    matched_numbers = [term for term in number_unit_terms(question) if normalize_text(term) in normalized_text]
    score += min(9.0, 3.0 * len(matched_numbers))
    _add_hits(score, hits, "numbers", matched_numbers)

    matched_financial = [term for term in financial_terms(question) if normalize_text(term) in normalized_text]
    score += min(6.0, 1.5 * len(matched_financial))
    _add_hits(score, hits, "financial_terms", matched_financial)

    mixed_terms = advanced_mixed_entity_terms(question) if advanced else mixed_entity_terms(question)
    matched_mixed = [term for term in mixed_terms if normalize_text(term) in normalized_text]
    score += min(7.0, 2.0 * len(matched_mixed))
    _add_hits(score, hits, "mixed_entities", matched_mixed)

    phrases = advanced_entity_phrases(question) if advanced else entity_phrases(question)
    matched_phrases = [term for term in phrases if normalize_text(term) in normalized_text]
    score += min(8.0, 2.0 * len(matched_phrases))
    _add_hits(score, hits, "entity_phrases", matched_phrases)

    if advanced:
        matched_quotes = [term for term in quoted_terms(question) if normalize_text(term) in normalized_text]
        score += min(6.0, 3.0 * len(matched_quotes))
        _add_hits(score, hits, "quoted_terms", matched_quotes)

    if page_index is not None:
        matched_terms = [
            term
            for term in page_index.important_terms(question)
            if len(term) >= 3 and normalize_text(term) in normalized_text
        ]
        score += min(5.0, 0.75 * len(matched_terms))
        _add_hits(score, hits, "important_terms", sorted(matched_terms, key=lambda item: (-len(item), item))[:12])

    if question_type == "list":
        list_score = list_feature_score(page_text)
        if list_score:
            score += list_score
            hits["list_features"] = [f"{list_score:.2f}"]

    return round(score, 4), hits


class HybridPageExperiment:
    def __init__(
        self,
        rag: RAGService,
        candidate_pages: int = 50,
        topk_anchor: int = 10,
        summary_penalty: float = 0.45,
        anchor_replace_margin: float = 3.0,
        neighbor_replace_margin: float = 2.0,
        reranker_weight: float = 0.3,
        reranker_margin_threshold: float = 0.05,
    ) -> None:
        self.rag = rag
        self.candidate_pages = candidate_pages
        self.topk_anchor = topk_anchor
        self.summary_penalty = summary_penalty
        self.anchor_replace_margin = anchor_replace_margin
        self.neighbor_replace_margin = neighbor_replace_margin
        self.reranker_weight = reranker_weight
        self.reranker_margin_threshold = reranker_margin_threshold
        self.reranker = LocalPageReranker(
            PAGE_RERANKER_MODEL,
            batch_size=PAGE_RERANKER_BATCH_SIZE,
            max_chars=PAGE_RERANKER_MAX_CHARS,
        )

    @property
    def page_index(self) -> PageTextIndex | None:
        return self.rag.page_index

    def retrieve_hybrid_candidates(
        self,
        question: str,
        initial_k: int = 200,
        max_chars_per_page: int = 1800,
    ) -> list[RetrievedPage]:
        return self.rag.retrieve_pages(
            question,
            initial_k=initial_k,
            final_pages=self.candidate_pages,
            max_chars_per_page=max_chars_per_page,
            reranker_model="",
            reranker_candidates=0,
        )

    def run_all_stages(self, question: str, candidates: list[RetrievedPage]) -> dict[str, StageResult]:
        question_type = classify_question(question)
        strong_entity = is_strong_entity_question(question, self.page_index)
        page_debug = self._page_debug(question, candidates, question_type, apply_summary_penalty=False)
        stage_a_pages = self._sort_by_adjusted(candidates, page_debug, use_adjusted=False)
        hybrid_top1_page = int(stage_a_pages[0].page_number)
        stage_a = self._stage(
            "A_pure_hybrid",
            question,
            question_type,
            strong_entity,
            stage_a_pages,
            page_debug,
            hybrid_top1_page=hybrid_top1_page,
        )

        page_debug_b = self._page_debug(question, candidates, question_type, apply_summary_penalty=True)
        stage_b_pages = self._sort_by_adjusted(candidates, page_debug_b, use_adjusted=True)
        stage_b = self._stage(
            "B_summary_penalty",
            question,
            question_type,
            strong_entity,
            stage_b_pages,
            page_debug_b,
            hybrid_top1_page=hybrid_top1_page,
        )

        stage_c_pages, anchor_debug = self._anchor_calibrate(question, stage_b_pages, page_debug_b, question_type, strong_entity)
        stage_c = self._stage(
            "C_anchor_calibration",
            question,
            question_type,
            strong_entity,
            stage_c_pages,
            page_debug_b,
            extra_debug=anchor_debug,
            hybrid_top1_page=hybrid_top1_page,
        )

        stage_d_pages, neighbor_debug = self._neighbor_calibrate(question, stage_c_pages, question_type)
        stage_d = self._stage(
            "D_neighbor_calibration",
            question,
            question_type,
            strong_entity,
            stage_d_pages,
            self._page_debug(question, stage_d_pages, question_type, apply_summary_penalty=True),
            extra_debug={**anchor_debug, **neighbor_debug},
            hybrid_top1_page=hybrid_top1_page,
        )

        stage_e_pages, reranker_debug = self._selective_rerank(question, stage_d_pages, question_type, strong_entity)
        stage_e = self._stage(
            "E_selective_reranker",
            question,
            question_type,
            strong_entity,
            stage_e_pages,
            self._page_debug(question, stage_e_pages, question_type, apply_summary_penalty=True),
            extra_debug={**anchor_debug, **neighbor_debug, **reranker_debug},
            hybrid_top1_page=hybrid_top1_page,
        )

        stage_f_base_debug = self._page_debug(
            question,
            candidates,
            question_type,
            apply_summary_penalty=True,
            advanced=True,
        )
        stage_f_base_pages = self._sort_by_adjusted(candidates, stage_f_base_debug, use_adjusted=True)
        stage_f_anchor_pages, stage_f_anchor_debug = self._anchor_calibrate(
            question,
            stage_f_base_pages,
            stage_f_base_debug,
            question_type,
            strong_entity,
        )
        stage_f_neighbor_pages, stage_f_neighbor_debug = self._neighbor_calibrate(
            question,
            stage_f_anchor_pages,
            question_type,
            advanced=True,
        )
        stage_f_pages, stage_f_reranker_debug = self._selective_rerank(
            question,
            stage_f_neighbor_pages,
            question_type,
            strong_entity,
            advanced=True,
        )
        stage_f = self._stage(
            "F_advanced_guarded",
            question,
            question_type,
            strong_entity,
            stage_f_pages,
            self._page_debug(
                question,
                stage_f_pages,
                question_type,
                apply_summary_penalty=True,
                advanced=True,
            ),
            extra_debug={**stage_f_anchor_debug, **stage_f_neighbor_debug, **stage_f_reranker_debug},
            hybrid_top1_page=hybrid_top1_page,
        )

        return {
            "A_pure_hybrid": stage_a,
            "B_summary_penalty": stage_b,
            "C_anchor_calibration": stage_c,
            "D_neighbor_calibration": stage_d,
            "E_selective_reranker": stage_e,
            "F_advanced_guarded": stage_f,
        }

    def _page_debug(
        self,
        question: str,
        pages: list[RetrievedPage],
        question_type: str,
        apply_summary_penalty: bool,
        advanced: bool = False,
    ) -> dict[tuple[str, int], PageDebug]:
        page_debug = {}
        penalize_summary = should_apply_summary_penalty(question, question_type)
        for page in pages:
            page_number = int(page.page_number)
            anchor_score, anchor_hits = evidence_anchor_score(
                question,
                page.content,
                page_number=page_number,
                page_index=self.page_index,
                question_type=question_type,
                advanced=advanced,
            )
            summary_penalty = apply_summary_penalty and penalize_summary and page_number <= 2
            penalty = self.summary_penalty if summary_penalty else 0.0
            if advanced and apply_summary_penalty and penalize_summary and page_number <= 3:
                detail_text = any(term in question for term in ("具体", "解决方案", "产品定位", "应用场景", "主要功能"))
                penalty = max(penalty, 0.95 if detail_text or question_type == "list" else 0.65)
                summary_penalty = True
            adjusted_score = float(page.score) - penalty
            page_debug[(page.filename, page_number)] = PageDebug(
                filename=page.filename,
                page=page_number,
                hybrid_score=float(page.score),
                adjusted_score=round(adjusted_score, 6),
                anchor_score=anchor_score,
                anchor_hits=anchor_hits,
                summary_penalty=summary_penalty,
            )
        return page_debug

    @staticmethod
    def _sort_by_adjusted(
        pages: list[RetrievedPage],
        debug: dict[tuple[str, int], PageDebug],
        use_adjusted: bool,
    ) -> list[RetrievedPage]:
        def sort_key(page: RetrievedPage) -> tuple[float, int, int]:
            item = debug[(page.filename, int(page.page_number))]
            score = item.adjusted_score if use_adjusted else item.hybrid_score
            return score, page.hit_count, -int(page.page_number)

        return sorted(pages, key=sort_key, reverse=True)

    def _anchor_calibrate(
        self,
        question: str,
        pages: list[RetrievedPage],
        debug: dict[tuple[str, int], PageDebug],
        question_type: str,
        strong_entity: bool,
    ) -> tuple[list[RetrievedPage], dict]:
        if not pages:
            return pages, {"anchor_calibrated_page": None}
        eligible = question_type in DETAIL_TYPES or strong_entity
        current = pages[0]
        current_debug = debug[(current.filename, int(current.page_number))]
        top_candidates = pages[: max(1, self.topk_anchor)]
        best = max(
            top_candidates,
            key=lambda page: (
                debug[(page.filename, int(page.page_number))].anchor_score,
                debug[(page.filename, int(page.page_number))].adjusted_score,
            ),
        )
        best_debug = debug[(best.filename, int(best.page_number))]
        replaced = (
            eligible
            and (best.filename, int(best.page_number)) != (current.filename, int(current.page_number))
            and best_debug.anchor_score >= current_debug.anchor_score + self.anchor_replace_margin
        )
        if not replaced:
            return pages, {
                "anchor_calibrated_page": int(current.page_number),
                "anchor_replaced": False,
                "anchor_best_page": int(best.page_number),
                "anchor_best_score": best_debug.anchor_score,
                "anchor_current_score": current_debug.anchor_score,
            }
        reordered = [best] + [
            page
            for page in pages
            if (page.filename, int(page.page_number)) != (best.filename, int(best.page_number))
        ]
        return reordered, {
            "anchor_calibrated_page": int(best.page_number),
            "anchor_replaced": True,
            "anchor_best_page": int(best.page_number),
            "anchor_best_score": best_debug.anchor_score,
            "anchor_current_score": current_debug.anchor_score,
        }

    def _neighbor_calibrate(
        self,
        question: str,
        pages: list[RetrievedPage],
        question_type: str,
        advanced: bool = False,
    ) -> tuple[list[RetrievedPage], dict]:
        if not pages or self.page_index is None:
            return pages, {"neighbor_calibrated_page": int(pages[0].page_number) if pages else None}
        current = pages[0]
        current_score, _ = evidence_anchor_score(
            question,
            current.content,
            page_number=int(current.page_number),
            page_index=self.page_index,
            question_type=question_type,
            advanced=advanced,
        )
        neighbors = []
        for offset in (-1, 1):
            neighbor_page = int(current.page_number) + offset
            if neighbor_page <= 0:
                continue
            content, chunk_ids = self.page_index.build_page_content(
                current.filename,
                neighbor_page,
                query=question,
                max_chars=1800,
            )
            if not content:
                continue
            score, hits = evidence_anchor_score(
                question,
                content,
                page_number=neighbor_page,
                page_index=self.page_index,
                question_type=question_type,
                advanced=advanced,
            )
            neighbors.append((score, hits, neighbor_page, content, chunk_ids))
        if not neighbors:
            return pages, {
                "neighbor_calibrated_page": int(current.page_number),
                "neighbor_replaced": False,
                "neighbor_current_score": current_score,
            }
        best_score, best_hits, best_page, best_content, best_chunk_ids = max(
            neighbors,
            key=lambda item: item[0],
        )
        if advanced and not has_strong_anchor_hits(best_hits):
            return pages, {
                "neighbor_calibrated_page": int(current.page_number),
                "neighbor_replaced": False,
                "neighbor_guarded": True,
                "neighbor_best_page": best_page,
                "neighbor_best_score": best_score,
                "neighbor_best_hits": best_hits,
                "neighbor_current_score": current_score,
            }
        replaced = best_score >= current_score + self.neighbor_replace_margin
        if not replaced:
            return pages, {
                "neighbor_calibrated_page": int(current.page_number),
                "neighbor_replaced": False,
                "neighbor_best_page": best_page,
                "neighbor_best_score": best_score,
                "neighbor_current_score": current_score,
            }
        neighbor = RetrievedPage(
            filename=current.filename,
            page_number=best_page,
            score=max(float(current.score) - 0.01, 0.0),
            hit_count=current.hit_count,
            chunk_ids=best_chunk_ids,
            content=best_content,
            rule_score=0.0,
        )
        reordered = [neighbor] + [
            page
            for page in pages
            if (page.filename, int(page.page_number)) != (current.filename, best_page)
        ]
        return reordered, {
            "neighbor_calibrated_page": best_page,
            "neighbor_replaced": True,
            "neighbor_best_page": best_page,
            "neighbor_best_score": best_score,
            "neighbor_best_hits": best_hits,
            "neighbor_current_score": current_score,
        }

    def _selective_rerank(
        self,
        question: str,
        pages: list[RetrievedPage],
        question_type: str,
        strong_entity: bool,
        advanced: bool = False,
    ) -> tuple[list[RetrievedPage], dict]:
        if len(pages) < 2:
            return pages, {"use_reranker": False}
        margin = float(pages[0].score) - float(pages[1].score)
        use_reranker = question_type in {"analysis", "default"} or margin <= self.reranker_margin_threshold
        if question_type in DETAIL_TYPES and strong_entity and margin > self.reranker_margin_threshold:
            use_reranker = False
        if not use_reranker:
            return pages, {
                "use_reranker": False,
                "reranker_margin": round(margin, 6),
            }
        candidate_pages = pages[: max(2, min(10, len(pages)))]
        passages = [
            f"filename: {page.filename}\npage: {page.page_number}\ntext:\n{page.content}"
            for page in candidate_pages
        ]
        try:
            rerank_results = self.reranker.rerank(question, passages)
        except Exception as exc:
            if STRICT_LOCAL_RERANKER:
                raise RuntimeError(
                    f"local reranker unavailable: {PAGE_RERANKER_MODEL}. "
                    "Set PAGE_RERANKER_MODEL to a local model directory or make sure the "
                    "HuggingFace cache is available in offline mode."
                ) from exc
            return pages, {
                "use_reranker": False,
                "reranker_error": str(exc),
                "reranker_margin": round(margin, 6),
            }
        raw_by_index = {item.index: item.score for item in rerank_results}
        raw_values = list(raw_by_index.values()) or [0.0]
        base_values = [float(page.score) for page in candidate_pages]
        min_raw, max_raw = min(raw_values), max(raw_values)
        min_base, max_base = min(base_values), max(base_values)
        rescored = []
        for index, page in enumerate(candidate_pages):
            raw_score = raw_by_index.get(index, min_raw)
            rerank_norm = (raw_score - min_raw) / (max_raw - min_raw) if max_raw > min_raw else 1.0
            base_norm = (float(page.score) - min_base) / (max_base - min_base) if max_base > min_base else 1.0
            anchor_score, _ = evidence_anchor_score(
                question,
                page.content,
                page_number=int(page.page_number),
                page_index=self.page_index,
                question_type=question_type,
                advanced=advanced,
            )
            anchor_bonus = min(0.12, anchor_score * 0.01)
            combined = self.reranker_weight * rerank_norm + (1.0 - self.reranker_weight) * base_norm + anchor_bonus
            rescored.append((combined, page, raw_score))
        rescored.sort(key=lambda item: (item[0], float(item[1].score), -int(item[1].page_number)), reverse=True)
        selected_pages = [page for _, page, _ in rescored]
        selected_keys = {(page.filename, int(page.page_number)) for page in selected_pages}
        selected_pages.extend(
            page
            for page in pages
            if (page.filename, int(page.page_number)) not in selected_keys
        )
        reranker_guarded = False
        if advanced and selected_pages:
            original = pages[0]
            selected = selected_pages[0]
            if (selected.filename, int(selected.page_number)) != (original.filename, int(original.page_number)):
                original_anchor, original_hits = evidence_anchor_score(
                    question,
                    original.content,
                    page_number=int(original.page_number),
                    page_index=self.page_index,
                    question_type=question_type,
                    advanced=True,
                )
                selected_anchor, selected_hits = evidence_anchor_score(
                    question,
                    selected.content,
                    page_number=int(selected.page_number),
                    page_index=self.page_index,
                    question_type=question_type,
                    advanced=True,
                )
                if (
                    (question_type in DETAIL_TYPES or strong_entity)
                    and has_strong_anchor_hits(original_hits)
                    and original_anchor >= selected_anchor + 2.0
                ):
                    reranker_guarded = True
                    selected_pages = [original] + [
                        page
                        for page in selected_pages
                        if (page.filename, int(page.page_number)) != (original.filename, int(original.page_number))
                    ]
        return selected_pages, {
            "use_reranker": True,
            "reranker_margin": round(margin, 6),
            "reranker_weight": self.reranker_weight,
            "reranker_top_page": int(selected_pages[0].page_number),
            "reranker_guarded": reranker_guarded,
            "reranker_raw_scores": [
                {
                    "page": int(page.page_number),
                    "combined": round(combined, 6),
                    "raw": round(float(raw), 6),
                }
                for combined, page, raw in rescored[:5]
            ],
        }

    def _stage(
        self,
        name: str,
        question: str,
        question_type: str,
        strong_entity: bool,
        pages: list[RetrievedPage],
        debug: dict[tuple[str, int], PageDebug],
        extra_debug: dict | None = None,
        hybrid_top1_page: int | None = None,
    ) -> StageResult:
        top = pages[0]
        top5 = []
        anchor_scores = []
        for page in pages[:5]:
            item = debug.get((page.filename, int(page.page_number)))
            if item is None:
                anchor_score, anchor_hits = evidence_anchor_score(
                    question,
                    page.content,
                    page_number=int(page.page_number),
                    page_index=self.page_index,
                    question_type=question_type,
                )
                item = PageDebug(
                    filename=page.filename,
                    page=int(page.page_number),
                    hybrid_score=float(page.score),
                    adjusted_score=float(page.score),
                    anchor_score=anchor_score,
                    anchor_hits=anchor_hits,
                )
            top5.append(
                {
                    "filename": page.filename,
                    "page": int(page.page_number),
                    "hybrid_score": item.hybrid_score,
                    "adjusted_score": item.adjusted_score,
                    "anchor_score": item.anchor_score,
                    "summary_penalty": item.summary_penalty,
                }
            )
            anchor_scores.append(
                {
                    "page": int(page.page_number),
                    "score": item.anchor_score,
                    "hits": item.anchor_hits,
                }
            )
        stage_debug = {
            "stage": name,
            "question_type": question_type,
            "strong_entity": strong_entity,
            "hybrid_top1": hybrid_top1_page if hybrid_top1_page is not None else int(pages[0].page_number),
            "anchor_calibrated_page": int(pages[0].page_number),
            "neighbor_calibrated_page": int(pages[0].page_number),
            "use_reranker": False,
            "final_page": int(top.page_number),
            "anchor_scores": anchor_scores,
            "summary_penalty_applied": any(item["summary_penalty"] for item in top5),
            "top5_candidate_pages": top5,
        }
        if extra_debug:
            stage_debug.update(extra_debug)
            stage_debug["final_page"] = int(top.page_number)
        return StageResult(
            filename=top.filename,
            page=int(top.page_number),
            pages=pages,
            debug=stage_debug,
        )
