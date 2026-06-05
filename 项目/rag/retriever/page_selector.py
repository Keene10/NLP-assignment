from __future__ import annotations

import json
import re
from pathlib import Path

from config.config import RETRIEVAL_CHUNKS_PATH
from rag.llm.llm import LLMService
from rag.retriever.page_index import PageTextIndex
from rag.vector.vector_db import VectorDB


def ordered_candidate_pages(center_page: int, radius: int) -> list[int]:
    start = max(1, center_page - radius)
    return list(range(start, center_page + radius + 1))


def expanded_query_terms(query: str) -> set[str]:
    normalized_query = PageTextIndex.normalize_text(query)
    expansions: set[str] = set()
    rules = {
        "收入": ("收入", "营收", "营业收入"),
        "营收": ("收入", "营收", "营业收入"),
        "占比": ("占比", "比重", "结构", "占公司总营收", "占公司"),
        "结构": ("结构", "占比", "比重", "构成"),
        "主要产品": ("主要产品", "产品", "业务板块", "第一大业务板块"),
        "客户": ("客户", "主要客户", "定点", "项目定点", "大客户"),
        "市场潜力": ("市场空间", "市场规模", "成长空间", "渗透率"),
        "增长潜力": ("增长潜力", "成长空间", "市场空间", "增速", "复合增速", "CAGR"),
        "成长空间": ("成长空间", "市场空间", "市场规模", "渗透率", "复合增速"),
        "发展前景": ("发展前景", "市场空间", "成长空间", "渗透率"),
        "竞争优势": ("竞争优势", "优势", "壁垒", "领先", "龙头"),
        "竞争格局": ("竞争格局", "市场份额", "份额", "集中度"),
        "供应链": ("供应链", "供应商", "质量", "供应稳定"),
        "产品质量": ("产品质量", "质量", "品控", "供应商"),
        "盈利预测": ("盈利预测", "营业收入", "归母净利润", "净利润", "估值"),
        "估值": ("估值", "盈利预测", "PE", "买入评级"),
        "数字化转型": ("数字化转型", "数字建筑", "云转型", "数字造价", "数字施工", "数字设计"),
        "数字造价": ("数字造价", "云转型", "造价业务", "工程造价", "订阅"),
        "数字施工": ("数字施工", "智慧工地", "施工总承包", "材料核算", "项目管理"),
        "数字孪生": ("数字孪生", "BIM", "CIM", "实景三维"),
        "奶酪": ("奶酪", "奶酪业务", "奶酪棒", "零售"),
        "速冻": ("速冻", "速冻米面", "餐饮", "零售端", "供应链"),
        "电池盒": ("电池盒", "热成型", "新能源", "渗透率"),
        "传感器": ("传感器", "力传感器", "智能装备", "机器人"),
    }
    for trigger, values in rules.items():
        if PageTextIndex.normalize_text(trigger) in normalized_query:
            expansions.update(PageTextIndex.normalize_text(value) for value in values)
    return {term for term in expansions if len(term) >= 2}


def page_profile(page_index: PageTextIndex, query: str, content: str) -> dict:
    normalized_content = PageTextIndex.normalize_text(content)
    terms = set(page_index.important_terms(query))
    terms.update(expanded_query_terms(query))
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
    matched_terms = sorted(terms, key=lambda term: (-len(term), term))[:18]

    role_flags = []
    if "图表目录" in content or ("目 录" in content and content.count("...") >= 3):
        role_flags.append("目录/图表目录页")
    if "风险提示" in content and len(terms) <= 2:
        role_flags.append("风险提示页")
    if any(mark in content for mark in ("【表格抽取】", "【疑似表格", "图表", "表")):
        role_flags.append("含图表/表格")
    if "【OCR补充】" in content:
        role_flags.append("含OCR补充")

    scored_lines = []
    for raw_line in re.split(r"[\n。；;]", content):
        line = " ".join(raw_line.split())
        if len(line) < 8:
            continue
        normalized_line = PageTextIndex.normalize_text(line)
        term_hits = sum(1 for term in terms if term in normalized_line)
        if term_hits <= 0:
            continue
        number_hits = len(re.findall(r"\d+(?:\.\d+)?%?", normalized_line))
        year_hits = len(re.findall(r"\d{4}年", normalized_line))
        conclusion_hits = sum(
            1
            for cue in ("因此", "预计", "我们认为", "空间", "增长", "优势", "格局", "潜力", "壁垒")
            if cue in line
        )
        has_evidence_mark = any(mark in line for mark in ("【表格抽取】", "【疑似表格", "图表", "表"))
        score = 4 * term_hits + number_hits + year_hits + conclusion_hits + (3 if has_evidence_mark else 0)
        scored_lines.append((score, line[:220]))

    if not scored_lines:
        fallback_lines = []
        seen = set()
        for raw_line in re.split(r"[\n。；;]", content):
            line = " ".join(raw_line.split())
            if len(line) < 12 or line in seen:
                continue
            seen.add(line)
            normalized_line = PageTextIndex.normalize_text(line)
            number_hits = len(re.findall(r"\d+(?:\.\d+)?%?", normalized_line))
            has_evidence_mark = any(mark in line for mark in ("【表格抽取】", "【疑似表格", "图表", "表", "【OCR补充】"))
            if number_hits <= 0 and not has_evidence_mark and len(line) < 24:
                continue
            score = number_hits + (3 if has_evidence_mark else 0) + min(len(line) / 120, 2)
            fallback_lines.append((score, line[:220]))
        fallback_lines.sort(key=lambda item: item[0], reverse=True)
        selected = [line for _, line in fallback_lines[:5]]
        evidence_hint = (
            "\n".join(f"- {line}" for line in selected)
            if selected
            else "本页没有抽取到明显证据句，但全文片段仍作为候选页内容提供。"
        )
        evidence_score = sum(score for score, _ in fallback_lines[:5])
    else:
        selected = []
        seen = set()
        for _, line in sorted(scored_lines, key=lambda item: item[0], reverse=True):
            if line in seen:
                continue
            seen.add(line)
            selected.append(line)
            if len(selected) >= 6:
                break
        evidence_hint = "\n".join(f"- {line}" for line in selected)
        evidence_score = sum(score for score, _ in sorted(scored_lines, key=lambda item: item[0], reverse=True)[:6])

    return {
        "role": "；".join(role_flags) if role_flags else "正文页",
        "matched_terms": matched_terms,
        "number_count": len(re.findall(r"\d+(?:\.\d+)?%?", normalized_content)),
        "evidence_score": evidence_score,
        "evidence_hint": evidence_hint,
    }


def build_candidates(
    page_index: PageTextIndex,
    question: str,
    filename: str,
    center_page: int,
    radius: int,
    max_chars_per_page: int,
) -> list[dict]:
    candidates = []
    for page in ordered_candidate_pages(center_page, radius):
        if not page_index.get_documents_by_page(filename, page):
            continue
        content, chunk_ids = page_index.build_page_content(
            filename,
            page,
            query=question,
            max_chars=max_chars_per_page,
        )
        if not content:
            continue
        candidates.append(
            {
                "filename": filename,
                "page": page,
                "distance_from_base_page": abs(page - center_page),
                "chunk_ids": chunk_ids,
                "content": content,
                "profile": page_profile(page_index, question, content),
            }
        )
    return candidates


def normalize_scores(items: dict[tuple[str, int], dict], field: str) -> None:
    max_score = max((float(item.get(field, 0.0)) for item in items.values()), default=0.0)
    output_field = f"{field}_norm"
    for item in items.values():
        item[output_field] = float(item.get(field, 0.0)) / max_score if max_score > 0 else 0.0


def collect_vector_signals(
    vector_db: VectorDB,
    question: str,
    top_k: int,
    restrict_filename: str | None,
) -> dict[tuple[str, int], dict]:
    grouped: dict[tuple[str, int], dict] = {}
    for rank, (document, raw_score) in enumerate(vector_db.search(question, k=top_k, score_threshold=0), start=1):
        metadata = document.metadata or {}
        filename = metadata.get("filename") or metadata.get("source")
        page = metadata.get("page_number") or metadata.get("page")
        if not filename or page is None:
            continue
        if restrict_filename and filename != restrict_filename:
            continue
        key = (filename, int(page))
        relevance = vector_db.relevance_score(float(raw_score))
        item = grouped.setdefault(
            key,
            {
                "filename": filename,
                "page": int(page),
                "vector_score": 0.0,
                "vector_rank_score": 0.0,
                "vector_hits": 0,
                "sources": set(),
            },
        )
        item["vector_score"] = max(item["vector_score"], relevance)
        item["vector_rank_score"] = max(item["vector_rank_score"], 1.0 / rank)
        item["vector_hits"] += 1
        item["sources"].add("embedding")
    return grouped


def collect_keyword_signals(
    page_index: PageTextIndex,
    question: str,
    restrict_filename: str | None,
    top_k: int,
) -> dict[tuple[str, int], dict]:
    scored = []
    for key in page_index.page_texts:
        filename, page = key
        if restrict_filename and filename != restrict_filename:
            continue
        keyword_score = page_index.keyword_score(question, key)
        bonus_score = page_index.bonus_score(question, key)
        exact_score = page_index.exact_phrase_score(question, key)
        early_penalty = page_index.early_page_penalty(question, key)
        combined = keyword_score + 1.8 * exact_score + 1.2 * bonus_score - 0.2 * early_penalty
        if combined <= 0:
            continue
        scored.append(
            {
                "filename": filename,
                "page": int(page),
                "keyword_score": keyword_score,
                "bonus_score": bonus_score,
                "exact_score": exact_score,
                "early_penalty": early_penalty,
                "keyword_combined": combined,
                "sources": {"bm25", "exact_phrase"} if exact_score > 0 else {"bm25"},
            }
        )

    scored.sort(key=lambda item: item["keyword_combined"], reverse=True)
    return {
        (item["filename"], item["page"]): item
        for item in scored[:top_k]
    }


def merge_candidate_signals(*groups: dict[tuple[str, int], dict]) -> dict[tuple[str, int], dict]:
    merged: dict[tuple[str, int], dict] = {}
    for group in groups:
        for key, item in group.items():
            target = merged.setdefault(
                key,
                {
                    "filename": item["filename"],
                    "page": int(item["page"]),
                    "vector_score": 0.0,
                    "vector_rank_score": 0.0,
                    "vector_hits": 0,
                    "keyword_score": 0.0,
                    "keyword_combined": 0.0,
                    "bonus_score": 0.0,
                    "exact_score": 0.0,
                    "early_penalty": 0.0,
                    "sources": set(),
                },
            )
            for field in (
                "vector_score",
                "vector_rank_score",
                "keyword_score",
                "keyword_combined",
                "bonus_score",
                "exact_score",
                "early_penalty",
            ):
                target[field] = max(float(target.get(field, 0.0)), float(item.get(field, 0.0)))
            target["vector_hits"] += int(item.get("vector_hits", 0))
            target["sources"].update(item.get("sources", set()))
    return merged


def rerank_candidate_pages(
    page_index: PageTextIndex,
    vector_db: VectorDB,
    question: str,
    planned: dict,
    fusion_top_k: int,
    vector_top_k: int,
    keyword_top_k: int,
    restrict_to_planned_file: bool = True,
) -> list[dict]:
    restrict_filename = planned["filename"] if restrict_to_planned_file else None
    vector_signals = collect_vector_signals(vector_db, question, vector_top_k, restrict_filename)
    keyword_signals = collect_keyword_signals(page_index, question, restrict_filename, keyword_top_k)
    merged = merge_candidate_signals(vector_signals, keyword_signals)

    base_key = (planned["filename"], int(planned["page"]))
    base_item = merged.setdefault(
        base_key,
        {
            "filename": planned["filename"],
            "page": int(planned["page"]),
            "vector_score": 0.0,
            "vector_rank_score": 0.0,
            "vector_hits": 0,
            "keyword_score": 0.0,
            "keyword_combined": 0.0,
            "bonus_score": 0.0,
            "exact_score": 0.0,
            "early_penalty": 0.0,
            "sources": set(),
        },
    )
    base_item["sources"].add("base_page_plan")
    base_item["base_plan_score"] = 1.0

    normalize_scores(merged, "vector_score")
    normalize_scores(merged, "vector_rank_score")
    normalize_scores(merged, "keyword_combined")
    normalize_scores(merged, "keyword_score")
    for item in merged.values():
        source_count_bonus = min(len(item["sources"]), 4) * 0.04
        item["fusion_score"] = (
            0.40 * item.get("vector_score_norm", 0.0)
            + 0.12 * item.get("vector_rank_score_norm", 0.0)
            + 0.34 * item.get("keyword_combined_norm", 0.0)
            + 0.08 * item.get("keyword_score_norm", 0.0)
            + 0.35 * item.get("exact_score", 0.0)
            + 0.25 * item.get("bonus_score", 0.0)
            + 0.08 * item.get("base_plan_score", 0.0)
            + source_count_bonus
            - 0.10 * item.get("early_penalty", 0.0)
        )

    ranked = sorted(
        merged.values(),
        key=lambda item: (
            item["fusion_score"],
            item.get("exact_score", 0.0),
            item.get("keyword_combined", 0.0),
            item.get("vector_score", 0.0),
        ),
        reverse=True,
    )
    selected = ranked[:fusion_top_k]
    for rank, item in enumerate(selected, start=1):
        item["fusion_rank"] = rank
        item["sources"] = sorted(item["sources"])
    return selected


def build_fusion_candidates(
    page_index: PageTextIndex,
    vector_db: VectorDB,
    question: str,
    planned: dict,
    fusion_top_k: int,
    vector_top_k: int,
    keyword_top_k: int,
    neighbor_radius: int,
    max_chars_per_page: int,
) -> list[dict]:
    top_pages = rerank_candidate_pages(
        page_index=page_index,
        vector_db=vector_db,
        question=question,
        planned=planned,
        fusion_top_k=fusion_top_k,
        vector_top_k=vector_top_k,
        keyword_top_k=keyword_top_k,
    )
    signal_by_key = {
        (item["filename"], int(item["page"])): item
        for item in top_pages
    }

    candidate_keys: list[tuple[str, int]] = []
    for item in top_pages:
        filename = item["filename"]
        page = int(item["page"])
        for neighbor in ordered_candidate_pages(page, neighbor_radius):
            key = (filename, neighbor)
            if key not in candidate_keys and page_index.get_documents_by_page(filename, neighbor):
                candidate_keys.append(key)

    candidates = []
    for filename, page in candidate_keys:
        content, chunk_ids = page_index.build_page_content(
            filename,
            page,
            query=question,
            max_chars=max_chars_per_page,
        )
        if not content:
            continue
        signal = dict(signal_by_key.get((filename, page), {}))
        is_top_page = bool(signal)
        if not signal:
            parent = min(
                top_pages,
                key=lambda item: abs(int(item["page"]) - page) if item["filename"] == filename else 10_000,
            )
            signal = {
                "filename": filename,
                "page": page,
                "fusion_rank": parent.get("fusion_rank"),
                "fusion_score": max(parent.get("fusion_score", 0.0) - 0.08 * abs(int(parent["page"]) - page), 0.0),
                "sources": [f"neighbor_of_page_{parent['page']}"],
                "vector_score": 0.0,
                "keyword_score": 0.0,
                "keyword_combined": 0.0,
                "exact_score": 0.0,
                "bonus_score": 0.0,
            }
        profile = page_profile(page_index, question, content)
        candidates.append(
            {
                "filename": filename,
                "page": page,
                "chunk_ids": chunk_ids,
                "content": content,
                "profile": profile,
                "fusion": {
                    "is_top_page": is_top_page,
                    "fusion_rank": signal.get("fusion_rank"),
                    "fusion_score": round(float(signal.get("fusion_score", 0.0)), 6),
                    "sources": signal.get("sources", []),
                    "vector_score": round(float(signal.get("vector_score", 0.0)), 6),
                    "keyword_score": round(float(signal.get("keyword_score", 0.0)), 6),
                    "keyword_combined": round(float(signal.get("keyword_combined", 0.0)), 6),
                    "exact_score": round(float(signal.get("exact_score", 0.0)), 6),
                    "bonus_score": round(float(signal.get("bonus_score", 0.0)), 6),
                },
            }
        )

    candidates.sort(
        key=lambda item: (
            item["fusion"].get("fusion_rank") or 999,
            -item["fusion"].get("is_top_page", False),
            item["page"],
        )
    )
    return candidates


def build_prompt(question: str, candidates: list[dict]) -> str:
    evidence_rows = []
    blocks = []
    for index, candidate in enumerate(candidates, start=1):
        profile = candidate["profile"]
        fusion = candidate.get("fusion") or {}
        matched_terms = "、".join(profile["matched_terms"]) or "无明显匹配"
        top_hint = (profile["evidence_hint"] or "").splitlines()[0:1]
        top_hint_text = top_hint[0][:180] if top_hint else "无"
        evidence_rows.append(
            f"[{index}] page={candidate['page']}; "
            f"fusion_rank={fusion.get('fusion_rank')}; "
            f"fusion_score={fusion.get('fusion_score')}; "
            f"retrieval_sources={','.join(fusion.get('sources') or [])}; "
            f"role={profile['role']}; "
            f"evidence_score={profile['evidence_score']}; "
            f"matched_terms={matched_terms}; "
            f"number_count={profile['number_count']}; "
            f"top_hint={top_hint_text}"
        )
        blocks.append(
            f"[{index}] filename: {candidate['filename']}\n"
            f"page: {candidate['page']}\n"
            f"fusion_rank: {fusion.get('fusion_rank')}\n"
            f"fusion_score: {fusion.get('fusion_score')}\n"
            f"retrieval_sources: {','.join(fusion.get('sources') or [])}\n"
            f"vector_score: {fusion.get('vector_score')}\n"
            f"bm25_score: {fusion.get('keyword_score')}\n"
            f"exact_score: {fusion.get('exact_score')}\n"
            f"page_role: {profile['role']}\n"
            f"evidence_score: {profile['evidence_score']}\n"
            f"matched_terms: {matched_terms}\n"
            f"evidence_hint:\n{profile['evidence_hint']}\n"
            f"content:\n{candidate['content']}"
        )

    return f"""你是金融研报 RAG 系统中的“证据页选择器”。你只负责选择页码，不要回答问题。

【候选页证据表】
{chr(10).join(evidence_rows)}

【候选页全文】
{chr(10).join(blocks)}

【问题】
{question}

【选择规则】
1. 只从候选页中选择一个 filename 和 page。
2. 候选页来自 embedding、BM25、exact phrase、业务关键词和基础页码计划的融合重排序；fusion_rank 是检索置信度，不等于最终答案页。
3. 不要只看候选页码列表，必须阅读每个候选页的 evidence_hint 和 content。
4. 选择最直接包含答案依据的页，重点比较问题里的年份、指标、业务名称、百分比、金额、图表和结论。
5. 相邻页都相关时，选择包含核心数字、图表、表格或结论最多的主证据页，而不是只起补充作用的邻页。
6. “【OCR补充】”“【表格抽取】”和图表文字都视为有效证据。
7. 输出前自检：reason 必须引用所选页中的 2-4 个关键词、数字或图表线索。
8. 输出严格 JSON，不要 Markdown，不要代码块，字段为 filename、page、confidence、reason；page 必须是数字。

【输出示例】
{{"filename":"xxx.pdf","page":12,"confidence":0.83,"reason":"该页包含问题中的核心指标和图表依据。"}}

【输出】
"""


def parse_json_object(text: str) -> dict | None:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
    match = re.search(r"\{.*\}", candidate, flags=re.S)
    if match:
        candidate = match.group(0)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def select_evidence_pages(
    questions: list[dict],
    base_plan: dict[int, dict],
    chunks_path: str | Path = RETRIEVAL_CHUNKS_PATH,
    vector_db_path: str | Path = "outputs/vector_db",
    vector_backend: str = "simple",
    fusion_top_k: int = 5,
    vector_top_k: int = 80,
    keyword_top_k: int = 80,
    radius: int = 4,
    max_chars_per_page: int = 1500,
    start_index: int = 0,
    limit: int = 0,
    progress_every: int = 5,
    sleep_seconds: float = 0,
) -> tuple[list[dict], list[dict]]:
    import time

    page_index = PageTextIndex(chunks_path)
    vector_db = VectorDB(persist_directory=vector_db_path, backend=vector_backend)
    llm = LLMService()
    end_index = len(questions) if limit <= 0 else min(len(questions), start_index + limit)
    selected_items = list(enumerate(questions[start_index:end_index], start=start_index))
    selected_plan = []
    debug_items = []

    for offset, (index, item) in enumerate(selected_items, start=1):
        planned = base_plan.get(index)
        if not planned:
            raise RuntimeError(f"Missing base page plan for question {index}")

        question = item["question"]
        center_item = planned
        center_page = int(planned["page"])
        candidates = build_candidates(
            page_index=page_index,
            question=question,
            filename=planned["filename"],
            center_page=center_page,
            radius=radius,
            max_chars_per_page=max_chars_per_page,
        )
        for candidate in candidates:
            distance = abs(int(candidate["page"]) - center_page)
            candidate["fusion"] = {
                "is_top_page": distance == 0,
                "fusion_rank": 1 if distance == 0 else None,
                "fusion_score": round(max(0.0, 1.0 - 0.08 * distance), 6),
                "sources": ["hybrid_top1_center"] if distance == 0 else [f"neighbor_of_hybrid_top1_page_{center_page}"],
                "distance_from_center": distance,
            }
        valid = {(candidate["filename"], str(candidate["page"])) for candidate in candidates}
        raw_answer = ""
        error = ""
        parsed = None

        if candidates:
            try:
                raw_answer = llm.generate(build_prompt(question, candidates))
                parsed = parse_json_object(raw_answer)
            except Exception as exc:
                error = f"llm_exception: {exc}"
        else:
            error = "no_candidate_pages"

        if parsed and (parsed.get("filename"), str(parsed.get("page"))) in valid:
            selected = {
                "filename": parsed.get("filename"),
                "page": int(parsed.get("page")),
                "confidence": parsed.get("confidence"),
                "reason": parsed.get("reason", ""),
            }
        else:
            selected = {
                "filename": planned["filename"],
                "page": int(planned["page"]),
                "confidence": None,
                "reason": "fallback_to_hybrid_top1_center",
            }
            if not error:
                error = "invalid_or_out_of_candidate_output"

        answer_pages = []
        seen_answer_pages = set()
        answer_candidates = sorted(
            candidates,
            key=lambda candidate: (
                0 if (candidate.get("fusion") or {}).get("is_top_page") else 1,
                (candidate.get("fusion") or {}).get("fusion_rank") or 999,
                int(candidate["page"]),
            ),
        )
        for candidate in answer_candidates:
            answer_key = (candidate["filename"], int(candidate["page"]))
            if answer_key in seen_answer_pages:
                continue
            seen_answer_pages.add(answer_key)
            fusion = candidate.get("fusion") or {}
            profile = candidate.get("profile") or {}
            answer_pages.append(
                {
                    "filename": candidate["filename"],
                    "page": int(candidate["page"]),
                    "chunk_ids": candidate.get("chunk_ids", []),
                    "fusion_rank": fusion.get("fusion_rank"),
                    "fusion_score": fusion.get("fusion_score"),
                    "retrieval_sources": fusion.get("sources") or [],
                    "evidence_score": profile.get("evidence_score"),
                    "matched_terms": profile.get("matched_terms") or [],
                    "evidence_hint": profile.get("evidence_hint", ""),
                }
            )

        selected_plan.append({"index": index, **selected, "answer_pages": answer_pages})
        debug_items.append(
            {
                "index": index,
                "question": question,
                "base_plan": planned,
                "selected": selected,
                "candidate_pages": [
                    {
                        "filename": candidate["filename"],
                        "page": candidate["page"],
                        "chunk_ids": candidate["chunk_ids"],
                        "fusion": candidate.get("fusion", {}),
                        "profile": {
                            key: value
                            for key, value in candidate["profile"].items()
                        },
                    }
                    for candidate in candidates
                ],
                "raw_answer": raw_answer,
                "error": error,
            }
        )
        if progress_every > 0 and (
            offset == 1 or offset == len(selected_items) or offset % progress_every == 0
        ):
            print(f"page selected {offset}/{len(selected_items)} index={index} status={error or 'ok'}")
        if sleep_seconds > 0 and offset < len(selected_items):
            time.sleep(sleep_seconds)

    return selected_plan, debug_items
