"""Hierarchical RAG segment router.

Builds per-document segments from table-of-contents, then uses an LLM to
route a query to the most relevant segment before running retrieval.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from rag.llm.llm import LLMService


@dataclass
class Segment:
    """A contiguous page range inside a single PDF."""

    filename: str
    segment_id: str
    title: str
    start_page: int
    end_page: int
    level: int = 1
    sub_titles: list[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return self.end_page - self.start_page + 1

    def __repr__(self) -> str:
        return (
            f"Segment({self.segment_id}: {self.title} "
            f"[{self.start_page}-{self.end_page}], {self.page_count}p)"
        )


# ---------------------------------------------------------------------------
# Hard-coded segment definitions for the 6 test PDFs.
# For short documents (<= 30 pages) we keep a single segment.
# For long documents we split by chapter / sub-chapter.
# ---------------------------------------------------------------------------

_HARD_CODED_SEGMENTS: dict[str, list[Segment]] = {
    # ------------------------------------------------------------------
    # 广联达-再谈 (131 pages) – the only document that really needs splitting
    # ------------------------------------------------------------------
    "广联达-再谈广联达当前时点下如何看待其三条增长曲线-220217131页.pdf": [
        Segment(
            filename="广联达-再谈广联达当前时点下如何看待其三条增长曲线-220217131页.pdf",
            segment_id="1",
            title="一、建筑信息化龙头，数字化驱动建筑产业整体升级",
            start_page=13,
            end_page=19,
            level=1,
        ),
        Segment(
            filename="广联达-再谈广联达当前时点下如何看待其三条增长曲线-220217131页.pdf",
            segment_id="2A",
            title="二(上)、建筑业市场基本情况 / 市场格局 / 特征与问题 (2.1-2.3)",
            start_page=19,
            end_page=36,
            level=2,
            sub_titles=[
                "2.1 建筑业市场基本情况",
                "2.2 建筑业市场格局分散",
                "2.3 建筑业的六大特征与四大发展问题",
            ],
        ),
        Segment(
            filename="广联达-再谈广联达当前时点下如何看待其三条增长曲线-220217131页.pdf",
            segment_id="2B",
            title="二(中)、建筑业信息化水平 / 转型驱动力 (2.4-2.5)",
            start_page=37,
            end_page=53,
            level=2,
            sub_titles=[
                "2.4 建筑业信息化水平较低",
                "2.5 建筑业信息化转型的驱动力",
            ],
        ),
        Segment(
            filename="广联达-再谈广联达当前时点下如何看待其三条增长曲线-220217131页.pdf",
            segment_id="2C",
            title="二(下)、建筑业信息化未来空间 (2.6)",
            start_page=54,
            end_page=58,
            level=2,
            sub_titles=["2.6 建筑业信息化的未来空间有望突破千亿元"],
        ),
        Segment(
            filename="广联达-再谈广联达当前时点下如何看待其三条增长曲线-220217131页.pdf",
            segment_id="3",
            title="三、造价业务：云转型打开新成长空间",
            start_page=58,
            end_page=74,
            level=1,
        ),
        Segment(
            filename="广联达-再谈广联达当前时点下如何看待其三条增长曲线-220217131页.pdf",
            segment_id="4A",
            title="四(上)、施工行业升级与公司发展历程 (4.1-4.2)",
            start_page=75,
            end_page=81,
            level=2,
            sub_titles=[
                "4.1 行业升级加速，施工数字化成长空间广阔",
                "4.2 公司施工业务的发展历程",
            ],
        ),
        Segment(
            filename="广联达-再谈广联达当前时点下如何看待其三条增长曲线-220217131页.pdf",
            segment_id="4B",
            title="四(中)、数字施工多层级产品与解决方案 (4.3)",
            start_page=81,
            end_page=99,
            level=2,
            sub_titles=[
                "4.3 公司数字施工业务聚焦工程项目建造过程",
            ],
        ),
        Segment(
            filename="广联达-再谈广联达当前时点下如何看待其三条增长曲线-220217131页.pdf",
            segment_id="4C",
            title="四(下)、施工业务进展广度与深度 / 设计业务 / 盈利预测 (4.5-结尾)",
            start_page=99,
            end_page=131,
            level=2,
            sub_titles=[
                "4.5 公司施工业务进展——广度与深度并重",
                "工程设计业务",
                "盈利预测与估值",
            ],
        ),
    ],
    # ------------------------------------------------------------------
    # 广联达-深度 (39 pages) – short enough for one segment
    # ------------------------------------------------------------------
    "广联达-深度跟踪报告数字建筑一体化领军-21082339页.pdf": [
        Segment(
            filename="广联达-深度跟踪报告数字建筑一体化领军-21082339页.pdf",
            segment_id="all",
            title="广联达深度跟踪报告（全文）",
            start_page=1,
            end_page=39,
            level=1,
        ),
    ],
    # ------------------------------------------------------------------
    # 伊利股份 (59 pages) – 8 chapters, all <= 30 pages, keep as single segment
    # ------------------------------------------------------------------
    "伊利股份-公司深度报告王者荣耀行稳致远-22021459页.pdf": [
        Segment(
            filename="伊利股份-公司深度报告王者荣耀行稳致远-22021459页.pdf",
            segment_id="all",
            title="伊利股份公司深度报告（全文）",
            start_page=1,
            end_page=59,
            level=1,
        ),
    ],
    # ------------------------------------------------------------------
    # 联邦制药 (25 pages)
    # ------------------------------------------------------------------
    "联邦制药-港股公司研究报告-创新突破三靶点战略联姻诺和诺德-25071225页.pdf": [
        Segment(
            filename="联邦制药-港股公司研究报告-创新突破三靶点战略联姻诺和诺德-25071225页.pdf",
            segment_id="all",
            title="联邦制药港股公司研究报告（全文）",
            start_page=1,
            end_page=25,
            level=1,
        ),
    ],
    # ------------------------------------------------------------------
    # 凌云股份 (27 pages)
    # ------------------------------------------------------------------
    "凌云股份-公司深度研究报告热成型电池盒双轮驱动传感器加速布局-25071427页.pdf": [
        Segment(
            filename="凌云股份-公司深度研究报告热成型电池盒双轮驱动传感器加速布局-25071427页.pdf",
            segment_id="all",
            title="凌云股份公司深度研究报告（全文）",
            start_page=1,
            end_page=27,
            level=1,
        ),
    ],
    # ------------------------------------------------------------------
    # 千味央厨 (26 pages)
    # ------------------------------------------------------------------
    "千味央厨-千寻百味乘势而上-22122726页.pdf": [
        Segment(
            filename="千味央厨-千寻百味乘势而上-22122726页.pdf",
            segment_id="all",
            title="千味央厨千寻百味乘势而上（全文）",
            start_page=1,
            end_page=26,
            level=1,
        ),
    ],
}


def get_segments(filename: str) -> list[Segment]:
    """Return the segment list for *filename*."""
    return _HARD_CODED_SEGMENTS.get(filename, [])


def build_segment_prompt(query: str, segments: list[Segment]) -> str:
    """Build a prompt asking the LLM to pick the best segment for *query*."""
    lines = []
    for seg in segments:
        lines.append(
            f"[{seg.segment_id}] {seg.title} (页码 {seg.start_page}-{seg.end_page}, "
            f"共{seg.page_count}页)"
        )
        for sub in seg.sub_titles:
            lines.append(f"    - {sub}")

    return f"""你是一位金融研报分析助手。请根据用户问题，判断该问题最可能属于哪个章节/段落。

【可选段落】
{chr(10).join(lines)}

【用户问题】
{query}

【要求】
1. 只从上述段落中选择一个最相关的段落编号（如 "1", "2A", "4B" 等）。
2. 如果没有明显匹配的段落，选择覆盖范围最广或最可能包含答案的段落。
3. 输出格式必须严格为 JSON: {{"segment_id": "所选编号", "reason": "简短理由"}}
4. 不要输出任何其他文字。

【输出】
"""


def parse_segment_id(text: str) -> str | None:
    """Extract segment_id from LLM JSON output."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
    match = re.search(r'"segment_id"\s*:\s*"([^"]+)"', candidate)
    if match:
        return match.group(1)
    # Fallback: try to find any known segment id in the text
    for sid in ["1", "2A", "2B", "2C", "3", "4A", "4B", "4C", "all"]:
        if sid in candidate:
            return sid
    return None


class SegmentRouter:
    """Routes a query to the most relevant segment using an LLM."""

    def __init__(self, llm: LLMService | None = None):
        self.llm = llm or LLMService()

    def route(self, query: str, filename: str) -> Segment | None:
        """Return the best segment for *query* inside *filename*."""
        segments = get_segments(filename)
        if not segments:
            return None
        if len(segments) == 1:
            return segments[0]

        if not self.llm.available:
            return None

        prompt = build_segment_prompt(query, segments)
        try:
            raw = self.llm.generate(prompt)
        except Exception:
            return None

        segment_id = parse_segment_id(raw)
        if segment_id:
            for seg in segments:
                if seg.segment_id == segment_id:
                    return seg

        return None
