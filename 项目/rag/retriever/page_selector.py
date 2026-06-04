from __future__ import annotations

import json
import re
from pathlib import Path

from config.config import RETRIEVAL_CHUNKS_PATH
from rag.llm.llm import LLMService
from rag.retriever.page_index import PageTextIndex


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
        evidence_hint = "未抽取到明显关键词提示，请直接阅读本页全文判断。"
        evidence_score = 0
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


def build_prompt(question: str, candidates: list[dict]) -> str:
    evidence_rows = []
    blocks = []
    for index, candidate in enumerate(candidates, start=1):
        profile = candidate["profile"]
        matched_terms = "、".join(profile["matched_terms"]) or "无明显匹配"
        top_hint = (profile["evidence_hint"] or "").splitlines()[0:1]
        top_hint_text = top_hint[0][:180] if top_hint else "无"
        evidence_rows.append(
            f"[{index}] page={candidate['page']}; "
            f"distance={candidate['distance_from_base_page']}; "
            f"role={profile['role']}; "
            f"evidence_score={profile['evidence_score']}; "
            f"matched_terms={matched_terms}; "
            f"number_count={profile['number_count']}; "
            f"top_hint={top_hint_text}"
        )
        blocks.append(
            f"[{index}] filename: {candidate['filename']}\n"
            f"page: {candidate['page']}\n"
            f"distance_from_base_page: {candidate['distance_from_base_page']}\n"
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
2. 候选页按页码升序排列，顺序不代表优先级；不要默认选择第一页、中间页或 distance 最小的页。
3. 选择最直接包含答案依据的页，重点比较问题里的年份、指标、业务名称、百分比、金额、图表和结论。
4. 相邻页都相关时，选择包含核心数字、图表、表格或结论最多的主证据页。
5. “【OCR补充】”“【表格抽取】”和图表文字都视为有效证据。
6. 输出前自检：reason 必须引用所选页中的 2-4 个关键词、数字或图表线索。
7. 输出严格 JSON，不要 Markdown，不要代码块，字段为 filename、page、confidence、reason；page 必须是数字。

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
    radius: int = 4,
    max_chars_per_page: int = 1500,
    start_index: int = 0,
    limit: int = 0,
    progress_every: int = 5,
    sleep_seconds: float = 0,
) -> tuple[list[dict], list[dict]]:
    import time

    page_index = PageTextIndex(chunks_path)
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
        candidates = build_candidates(
            page_index=page_index,
            question=question,
            filename=planned["filename"],
            center_page=int(planned["page"]),
            radius=radius,
            max_chars_per_page=max_chars_per_page,
        )
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
                "reason": "fallback_to_base_page_plan",
            }
            if not error:
                error = "invalid_or_out_of_candidate_output"

        selected_plan.append({"index": index, **selected})
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
                        "profile": {
                            key: value
                            for key, value in candidate["profile"].items()
                            if key != "evidence_hint"
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
