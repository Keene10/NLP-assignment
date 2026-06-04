from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from langchain_core.documents import Document

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.vector.vector_db import VectorDB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the vector database from exported document chunks.")
    parser.add_argument("--chunks", default="outputs/extracted_text_ocr/chunks.jsonl")
    parser.add_argument("--vector-db-path", default="outputs/vector_db")
    parser.add_argument("--backend", default="simple", choices=["auto", "chroma", "simple"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--recreate", action="store_true", default=True)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    file_path = Path(path)
    return file_path if file_path.is_absolute() else ROOT_DIR / file_path


def load_chunks(chunks_path: Path) -> list[Document]:
    documents = []
    with chunks_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.pop("text", "")
            if not text:
                continue
            documents.append(Document(page_content=text, metadata=row))
    return documents


def main() -> None:
    args = parse_args()
    chunks_path = resolve_path(args.chunks)
    vector_db_path = resolve_path(args.vector_db_path)
    documents = load_chunks(chunks_path)
    if not documents:
        raise RuntimeError(f"No chunks found in {chunks_path}")

    vector_db = VectorDB(persist_directory=vector_db_path, backend=args.backend)
    vector_db.create(documents, batch_size=args.batch_size, recreate=args.recreate)
    summary = {
        "chunks": len(documents),
        "backend": vector_db.backend,
        "vector_db_path": str(vector_db_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
