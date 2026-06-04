#!/usr/bin/env python3
"""
使用阿里云百炼 Qwen3-VL API 对财报 PDF 中的图表页面进行多模态描述。

流程：
1. 读取 detect_chart_pages.py 生成的 chart_pages.json
2. 用 PyMuPDF 将图表页渲染为高清图片
3. 调用百炼 Qwen3-VL API 生成结构化描述
4. 保存为 {filename}_chart_descriptions.json

前置准备：
- 安装依赖：pip install dashscope openai PyMuPDF
- 获取百炼 API Key：https://help.aliyun.com/zh/model-studio/get-api-key
- 将 API Key 配置到环境变量：export DASHSCOPE_API_KEY=sk-xxxx
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("请先安装 PyMuPDF: pip install PyMuPDF") from None

# 优先使用 OpenAI 兼容接口（更通用，与项目现有 langchain-openai 一致）
try:
    from openai import OpenAI
except ImportError:
    raise ImportError("请先安装 openai: pip install openai") from None

ROOT_DIR = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Prompt 设计：针对财务研报图表优化
# ---------------------------------------------------------------------------
CHART_ANALYSIS_PROMPT = """\
你是一位专业的金融研报分析师。请仔细观察这张图片，它来自一份证券公司的公司深度研究报告。

请按以下要求输出分析结果（JSON格式）：
1. 判断图片类型：table（表格）/ bar_chart（柱状图）/ line_chart（折线图）/ pie_chart（饼图）/ diagram（流程图/架构图）/ image（示意图/照片）/ other
2. 提取标题：图片中的标题文字（如果有）
3. 详细描述：
   - 如果是表格：描述表头含义、行列结构、关键数据（极值、同比变化等）
   - 如果是统计图表：描述坐标轴含义、数据系列、趋势变化、关键节点、极值
   - 如果是示意图/流程图：描述图中各环节关系和文字信息
   - 如果是图片：描述画面内容和图中文字
4. 关键结论：从该图中可以得出的1-3条核心结论或数据洞察

请严格按以下JSON格式输出，不要输出markdown代码块标记，只输出纯JSON字符串：
{
  "type": "...",
  "title": "...",
  "description": "...",
  "key_conclusions": ["...", "..."]
}
"""

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "qwen3-vl-max"  # 备选: qwen3-vl-plus, qwen2.5-vl-72b-instruct
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def get_api_key() -> str:
    key = os.getenv("DASHSCOPE_API_KEY", "")
    if not key:
        raise RuntimeError(
            "环境变量 DASHSCOPE_API_KEY 未设置。"
            "请先获取百炼 API Key 并执行：export DASHSCOPE_API_KEY=sk-xxxx"
        )
    return key


def page_to_image(
    pdf_path: Path,
    page_number: int,
    dpi: int = 200,
) -> bytes:
    """将 PDF 指定页渲染为 PNG 图片字节。"""
    pdf = fitz.open(str(pdf_path))
    page = pdf.load_page(page_number - 1)  # 0-based
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    pdf.close()
    return pix.tobytes("png")


def image_to_base64(image_bytes: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"


def call_qwen_vl(
    client: OpenAI,
    model: str,
    image_base64: str,
    prompt: str = CHART_ANALYSIS_PROMPT,
    max_tokens: int = 2048,
    temperature: float = 0.1,
) -> dict[str, Any]:
    """调用百炼 Qwen-VL API，返回解析后的 JSON 结果。"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_base64},
                    },
                ],
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    raw_text = response.choices[0].message.content or ""
    usage = response.usage

    # 尝试从模型输出中解析 JSON
    parsed = None
    try:
        # 先尝试直接解析
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # 如果失败，尝试提取 ```json ... ``` 块
        import re
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

    return {
        "raw_response": raw_text,
        "parsed": parsed,
        "usage": {
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
        },
    }


def process_file(
    client: OpenAI,
    model: str,
    pdf_path: Path,
    chart_pages: list[int],
    image_output_dir: Path,
    desc_output_dir: Path,
    dpi: int,
    dry_run: bool,
) -> list[dict]:
    """处理单个 PDF 的所有图表页，返回描述结果列表。"""
    results = []
    file_stem = pdf_path.stem

    # 创建该文件对应的图片子目录
    file_image_dir = image_output_dir / file_stem
    file_image_dir.mkdir(parents=True, exist_ok=True)

    total = len(chart_pages)
    for idx, page_number in enumerate(chart_pages, 1):
        print(f"  [{idx}/{total}] 处理第 {page_number} 页...", end="", flush=True)

        # 1. 渲染页面为图片
        image_bytes = page_to_image(pdf_path, page_number, dpi=dpi)
        image_path = file_image_dir / f"page_{page_number:03d}.png"
        image_path.write_bytes(image_bytes)

        if dry_run:
            print(" (dry-run，跳过 API 调用)")
            results.append({
                "page": page_number,
                "image_path": str(image_path.relative_to(ROOT_DIR)),
                "dry_run": True,
            })
            continue

        # 2. 调用 Qwen-VL API
        image_b64 = image_to_base64(image_bytes)
        try:
            api_result = call_qwen_vl(client, model, image_b64)
        except Exception as exc:
            print(f" 失败: {exc}")
            results.append({
                "page": page_number,
                "image_path": str(image_path.relative_to(ROOT_DIR)),
                "error": str(exc),
            })
            continue

        # 3. 组装结果
        record = {
            "page": page_number,
            "image_path": str(image_path.relative_to(ROOT_DIR)),
            "description": api_result["parsed"] if api_result["parsed"] else api_result["raw_response"],
            "raw_response": api_result["raw_response"],
            "usage": api_result["usage"],
        }
        results.append(record)

        tokens_info = api_result["usage"]["total_tokens"]
        print(f" 完成 (tokens: {tokens_info})")

        # 简单限速，避免触发 QPS 限制
        time.sleep(0.5)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用阿里云百炼 Qwen3-VL 对财报图表页面进行多模态描述。"
    )
    parser.add_argument(
        "--chart-pages-json",
        default="outputs/chart_pages.json",
        help="detect_chart_pages.py 生成的 JSON 文件路径",
    )
    parser.add_argument(
        "--pdf-dir",
        default="财报数据库/test",
        help="PDF 文件所在目录",
    )
    parser.add_argument(
        "--image-output-dir",
        default="outputs/chart_images",
        help="图表页面截图保存目录",
    )
    parser.add_argument(
        "--desc-output-dir",
        default="outputs/chart_descriptions",
        help="描述结果 JSON 保存目录",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"百炼模型名称 (默认: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PDF 渲染分辨率 DPI (默认: 200)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅生成截图，不调用 API（用于测试）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    chart_pages_path = ROOT_DIR / args.chart_pages_json
    pdf_dir = ROOT_DIR / args.pdf_dir
    image_output_dir = ROOT_DIR / args.image_output_dir
    desc_output_dir = ROOT_DIR / args.desc_output_dir

    if not chart_pages_path.exists():
        print(f"错误：找不到图表页码文件: {chart_pages_path}", file=sys.stderr)
        print("请先运行: python scripts/detect_chart_pages.py", file=sys.stderr)
        return 1

    # 读取 chart_pages.json
    with open(chart_pages_path, "r", encoding="utf-8") as f:
        chart_pages_data = json.load(f)

    print(f"共检测到 {len(chart_pages_data)} 个 PDF 文件需要处理\n")

    # 初始化百炼客户端（OpenAI 兼容接口）
    if not args.dry_run:
        api_key = get_api_key()
        client = OpenAI(api_key=api_key, base_url=BASE_URL)
        print(f"已连接百炼 API，使用模型: {args.model}")
    else:
        client = None  # type: ignore[assignment]
        print("【Dry-run 模式】仅生成截图，不调用 API")

    image_output_dir.mkdir(parents=True, exist_ok=True)
    desc_output_dir.mkdir(parents=True, exist_ok=True)

    total_pages = sum(item["chart_page_count"] for item in chart_pages_data)
    processed_pages = 0

    for item in chart_pages_data:
        filename = item["filename"]
        chart_pages = item["chart_pages"]
        pdf_path = pdf_dir / filename

        if not pdf_path.exists():
            print(f"\n⚠️ 跳过：找不到 PDF 文件: {pdf_path}")
            continue

        print(f"\n📄 {filename} ({item['chart_page_count']} 页图表)")

        results = process_file(
            client=client,
            model=args.model,
            pdf_path=pdf_path,
            chart_pages=chart_pages,
            image_output_dir=image_output_dir,
            desc_output_dir=desc_output_dir,
            dpi=args.dpi,
            dry_run=args.dry_run,
        )

        # 保存该文件的描述结果
        output_file = desc_output_dir / f"{Path(filename).stem}_chart_descriptions.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "filename": filename,
                    "model": args.model,
                    "dpi": args.dpi,
                    "total_pages": len(chart_pages),
                    "descriptions": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"  ✓ 结果已保存: {output_file.relative_to(ROOT_DIR)}")

        processed_pages += len(chart_pages)

    print(f"\n🎉 全部完成！共处理 {processed_pages} 个图表页面。")
    if not args.dry_run:
        print(f"   截图目录: {image_output_dir.relative_to(ROOT_DIR)}")
        print(f"   描述目录: {desc_output_dir.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
