from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load financial-report PDFs, run table/OCR enhancement, split text, and export chunks."
    )
    parser.add_argument("--input-dir", default="财报数据库/test")
    parser.add_argument("--output-dir", default="outputs/extracted_text_ocr")
    parser.add_argument("--ocr-cache-dir", default="outputs/ocr_cache")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--ocr-dpi", type=int, default=144)
    parser.add_argument("--ocr-min-confidence", type=float, default=0.5)
    parser.add_argument("--table-mode", default="text", choices=["text", "pdfplumber"])
    parser.add_argument("--disable-ocr", action="store_true")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    file_path = Path(path)
    return file_path if file_path.is_absolute() else ROOT_DIR / file_path


def safe_name(name: str) -> str:
    return "".join(char if char not in '<>:"/\\|?*' else "_" for char in name)


def document_key(document: Any) -> tuple[str, int]:
    metadata = document.metadata or {}
    return (
        metadata.get("filename") or Path(metadata.get("source", "")).name,
        int(metadata.get("page_number") or metadata.get("page") or 1),
    )


def write_page_files(output_dir: Path, documents: list[Any]) -> None:
    pages_dir = output_dir / "pages"
    all_pages = []
    for document in documents:
        metadata = document.metadata or {}
        filename, page_number = document_key(document)
        file_dir = pages_dir / safe_name(Path(filename).stem)
        file_dir.mkdir(parents=True, exist_ok=True)
        text = document.page_content or ""
        header = f"# {filename} page {page_number}\n\n"
        (file_dir / f"page_{page_number:03d}.txt").write_text(text, encoding="utf-8")
        all_pages.append(header + text)
    (output_dir / "all_pages.md").write_text("\n\n".join(all_pages), encoding="utf-8")


def write_chunk_files(output_dir: Path, chunks: list[Any]) -> None:
    chunks_dir = output_dir / "chunks"
    rows = []
    markdown_blocks = []
    for chunk_id, document in enumerate(chunks):
        metadata = dict(document.metadata or {})
        filename, page_number = document_key(document)
        metadata["chunk_id"] = chunk_id
        metadata["filename"] = filename
        metadata["page_number"] = page_number
        metadata["page"] = page_number
        document.metadata = metadata

        text = document.page_content or ""
        rows.append({**metadata, "text": text})
        file_dir = chunks_dir / safe_name(Path(filename).stem)
        file_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = file_dir / f"chunk_{chunk_id:05d}_page_{page_number:03d}.txt"
        chunk_path.write_text(text, encoding="utf-8")
        markdown_blocks.append(f"# chunk {chunk_id} | {filename} | page {page_number}\n\n{text}")

    with (output_dir / "chunks.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "all_chunks.md").write_text("\n\n".join(markdown_blocks), encoding="utf-8")


def write_manifest(output_dir: Path, documents: list[Any], chunks: list[Any], args: argparse.Namespace) -> None:
    by_file: dict[str, dict] = {}
    for document in documents:
        metadata = document.metadata or {}
        filename, _ = document_key(document)
        item = by_file.setdefault(
            filename,
            {
                "filename": filename,
                "page_count": 0,
                "table_pages": 0,
                "ocr_pages": 0,
                "image_pages": 0,
            },
        )
        item["page_count"] += 1
        item["table_pages"] += int(bool(metadata.get("has_tables")))
        item["ocr_pages"] += int(bool(metadata.get("has_ocr")))
        item["image_pages"] += int(bool(metadata.get("has_images")))

    manifest = {
        "input_dir": str(resolve_path(args.input_dir)),
        "output_dir": str(output_dir),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "enable_ocr": not args.disable_ocr,
        "ocr_cache_dir": str(resolve_path(args.ocr_cache_dir)),
        "page_count": len(documents),
        "chunk_count": len(chunks),
        "files": list(by_file.values()),
    }
    (output_dir / "ocr_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    from rag.document.processor import DocumentProcessor

    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    ocr_cache_dir = resolve_path(args.ocr_cache_dir)

    if args.recreate and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = DocumentProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        extract_tables=True,
        table_mode=args.table_mode,
        enable_ocr=not args.disable_ocr,
        ocr_cache_dir=ocr_cache_dir,
        ocr_dpi=args.ocr_dpi,
        ocr_min_confidence=args.ocr_min_confidence,
        ocr_progress=args.progress,
    )
    documents = processor.load_directory(input_dir, recursive=False, extensions={".pdf"})
    chunks = processor.split_documents(documents)
    write_page_files(output_dir, documents)
    write_chunk_files(output_dir, chunks)
    write_manifest(output_dir, documents, chunks, args)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "pages": len(documents),
                "chunks": len(chunks),
                "chunks_jsonl": str(output_dir / "chunks.jsonl"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
