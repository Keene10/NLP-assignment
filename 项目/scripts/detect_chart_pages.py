#!/usr/bin/env python3
"""
检测财报 PDF 中需要多模态识别的图表/表格页面。

根据 PDF 的特性使用不同策略：
1. 有图表目录的 PDF（广联达再谈、联邦制药、伊利股份、凌云股份）：
   解析图表目录，提取图表/表格对应的页码。
2. 无图表目录但页面内标注"图表 N:"的 PDF（千味央厨）：
   扫描所有页面文本，匹配内联的图表标记。
3. 无图表目录的 PDF（广联达深度跟踪报告）：
   扫描所有页面，查找图表下方常见的"资料来源"字样来判断。

输出格式：JSON，每个文件对应一个包含 chart_pages 列表的记录。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError as exc:
    raise ImportError("请先安装 PyMuPDF: pip install pymupdf") from exc


CHART_HEADER_RE = re.compile(r"^(图表|图|表)\s*\d+")
CHART_INLINE_RE = re.compile(r"(图表|图|表)\s*\d+[:：]")

# 文件名关键字 -> 检测策略映射
STRATEGY_OVERRIDES = {
    "广联达-深度跟踪报告": "source_citation",
}


def find_chart_pages_by_toc(pdf: fitz.Document) -> list[int] | None:
    """
    通过图表目录查找图表页码。

    在前 8 页中定位图表目录起始页（包含大量以"图/表/图表"开头的行），
    然后合并连续的目录页，逐行提取行尾的最后一个数字作为页码。
    """
    pages: set[int] = set()
    toc_start: int | None = None

    # 1. 定位图表目录起始页
    for i in range(min(8, pdf.page_count)):
        text = pdf.load_page(i).get_text()
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        chart_line_count = sum(1 for ln in lines if CHART_HEADER_RE.match(ln))
        if chart_line_count >= 10:
            toc_start = i
            break

    if toc_start is None:
        return None

    # 2. 合并连续的目录页（通常图表目录会跨 2~4 页）
    toc_text = ""
    for i in range(toc_start, min(toc_start + 6, pdf.page_count)):
        text = pdf.load_page(i).get_text()
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        chart_line_count = sum(1 for ln in lines if CHART_HEADER_RE.match(ln))
        # 当前页仍有较多图表条目，或者是起始页本身
        if chart_line_count >= 3 or i == toc_start:
            toc_text += text + "\n"
        else:
            break

    # 3. 逐行解析：取每行最后一个数字作为页码
    for line in toc_text.split("\n"):
        line = line.strip()
        if not CHART_HEADER_RE.match(line):
            continue
        nums = re.findall(r"\d+", line)
        if len(nums) >= 2:  # 至少有序号和一个页码
            page_num = int(nums[-1])
            if 1 <= page_num <= pdf.page_count:
                pages.add(page_num)

    return sorted(pages) if pages else None


def find_chart_pages_by_source_citation(pdf: fitz.Document) -> list[int]:
    """
    广联达深度跟踪报告专用策略。

    该报告无图表目录，但所有包含图表/表格的页面下方都会标注"资料来源"。
    扫描所有页面，收集包含"资料来源"的页码。
    """
    pages: set[int] = set()
    for i in range(pdf.page_count):
        text = pdf.load_page(i).get_text()
        if "资料来源" in text:
            pages.add(i + 1)
    return sorted(pages)


def find_chart_pages_by_inline_markers(pdf: fitz.Document) -> list[int]:
    """
    千味央厨等无图表目录但页面内会标注"图表 N:"的 PDF 专用策略。

    扫描所有页面文本，匹配内联的图表标记（如"图表1："、"图表 2:"等）。
    """
    pages: set[int] = set()
    for i in range(pdf.page_count):
        text = pdf.load_page(i).get_text()
        if CHART_INLINE_RE.search(text):
            pages.add(i + 1)
    return sorted(pages)


def detect_strategy(filename: str) -> str:
    """根据文件名判断应使用的检测策略。"""
    for keyword, strategy in STRATEGY_OVERRIDES.items():
        if keyword in filename:
            return strategy
    return "auto"


def process_pdf(pdf_path: Path, forced_strategy: str | None = None) -> dict:
    """处理单个 PDF，返回包含文件名、策略和图表页码的结果字典。"""
    pdf = fitz.open(str(pdf_path))
    filename = pdf_path.name

    # 仅在用户显式指定策略时覆盖自动检测
    strategy = forced_strategy if forced_strategy and forced_strategy != "auto" else detect_strategy(filename)
    chart_pages: list[int] | None = None

    if strategy == "source_citation":
        chart_pages = find_chart_pages_by_source_citation(pdf)
        used_strategy = "source_citation"
    elif strategy == "inline_marker":
        chart_pages = find_chart_pages_by_inline_markers(pdf)
        used_strategy = "inline_marker"
    elif strategy == "toc":
        chart_pages = find_chart_pages_by_toc(pdf)
        used_strategy = "toc"
    else:  # auto
        # 优先尝试图表目录解析
        chart_pages = find_chart_pages_by_toc(pdf)
        if chart_pages is not None:
            used_strategy = "toc"
        else:
            # 回退到内联标记扫描
            chart_pages = find_chart_pages_by_inline_markers(pdf)
            used_strategy = "inline_marker"

    pdf.close()

    return {
        "filename": filename,
        "strategy": used_strategy,
        "chart_pages": chart_pages if chart_pages is not None else [],
        "chart_page_count": len(chart_pages) if chart_pages is not None else 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检测财报 PDF 中包含图表/表格、需要多模态识别的页面。"
    )
    parser.add_argument(
        "--input-dir",
        default="财报数据库/test",
        help="输入目录，包含待分析的 PDF 文件（默认: 财报数据库/test）",
    )
    parser.add_argument(
        "--output",
        default="outputs/chart_pages.json",
        help="输出 JSON 文件路径（默认: outputs/chart_pages.json）",
    )
    parser.add_argument(
        "--strategy",
        choices=["auto", "toc", "source_citation", "inline_marker"],
        default="auto",
        help="强制指定检测策略（默认根据文件名自动判断）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"错误：输入目录不存在: {input_dir}", file=sys.stderr)
        return 1

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"警告：在 {input_dir} 中未找到 PDF 文件", file=sys.stderr)
        return 0

    results = []
    for pdf_path in pdf_files:
        result = process_pdf(pdf_path, forced_strategy=args.strategy)
        results.append(result)
        print(
            f"[{result['strategy']}] {result['filename']}: "
            f"{result['chart_page_count']} 个图表页"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存至: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
