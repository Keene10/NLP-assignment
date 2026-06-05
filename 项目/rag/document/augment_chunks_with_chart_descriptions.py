from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append multimodal chart/page descriptions to an existing chunks.jsonl file."
    )
    parser.add_argument("--chunks", default="outputs/extracted_text_ocr/chunks.jsonl")
    parser.add_argument("--chart-description-dir", default="归档/chart_descriptions")
    parser.add_argument("--output-dir", default="outputs/extracted_text_ocr_multimodal")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    file_path = Path(path)
    return file_path if file_path.is_absolute() else ROOT_DIR / file_path


def read_chunks(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def stringify_description(description: Any) -> list[str]:
    if not description:
        return []
    if isinstance(description, str):
        return [description.strip()] if description.strip() else []
    if not isinstance(description, dict):
        text = str(description).strip()
        return [text] if text else []

    lines: list[str] = []
    field_labels = [
        ("chart_index", "图表编号"),
        ("chart_type", "图表类型"),
        ("type", "内容类型"),
        ("title", "标题"),
        ("one_liner", "一句话结论"),
        ("description", "页面描述"),
        ("key_entities", "关键实体/数字"),
        ("key_conclusions", "核心结论"),
        ("detailed_facts", "详细事实"),
        ("trend_conclusion", "趋势结论"),
    ]
    for field, label in field_labels:
        value = description.get(field)
        if value in (None, "", []):
            continue
        if isinstance(value, list):
            rendered = "；".join(str(item).strip() for item in value if str(item).strip())
        else:
            rendered = str(value).strip()
        if rendered:
            lines.append(f"{label}: {rendered}")

    used = {field for field, _ in field_labels}
    for field, value in description.items():
        if field in used or value in (None, "", []):
            continue
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False)
        else:
            rendered = str(value).strip()
        if rendered:
            lines.append(f"{field}: {rendered}")
    return lines


def load_chart_description_chunks(description_dir: Path, start_chunk_id: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    next_chunk_id = start_chunk_id
    for path in sorted(description_dir.glob("*_chart_descriptions.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        filename = data.get("filename") or path.name.replace("_chart_descriptions.json", ".pdf")
        descriptions = data.get("descriptions") or []
        for item in as_list(descriptions):
            if not isinstance(item, dict):
                continue
            page = item.get("page")
            if page is None:
                continue
            try:
                page_number = int(page)
            except (TypeError, ValueError):
                continue
            description_lines = stringify_description(item.get("description"))
            if not description_lines:
                continue
            text = "\n".join(
                [
                    "【多模态图表说明】",
                    f"文件: {filename}",
                    f"页码: {page_number}",
                    *description_lines,
                ]
            )
            chunks.append(
                {
                    "chunk_id": next_chunk_id,
                    "filename": filename,
                    "page_number": page_number,
                    "page": page_number,
                    "page_index": page_number - 1,
                    "source_type": "multimodal_chart_description",
                    "has_multimodal_chart_description": True,
                    "image_path": item.get("image_path", ""),
                    "text": text,
                }
            )
            next_chunk_id += 1
    return chunks


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    chunks_path = resolve_path(args.chunks)
    description_dir = resolve_path(args.chart_description_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_rows = read_chunks(chunks_path)
    max_chunk_id = max((int(row.get("chunk_id") or 0) for row in original_rows), default=0)
    chart_rows = load_chart_description_chunks(description_dir, start_chunk_id=max_chunk_id + 1)
    combined_rows = original_rows + chart_rows

    output_chunks = output_dir / "chunks.jsonl"
    write_jsonl(output_chunks, combined_rows)
    manifest = {
        "base_chunks": str(chunks_path),
        "chart_description_dir": str(description_dir),
        "output_chunks": str(output_chunks),
        "original_chunk_count": len(original_rows),
        "multimodal_chunk_count": len(chart_rows),
        "combined_chunk_count": len(combined_rows),
        "files": sorted({row["filename"] for row in chart_rows}),
    }
    (output_dir / "multimodal_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
