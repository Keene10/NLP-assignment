#!/usr/bin/env python3
"""Run Hierarchical RAG on test.json and produce predictions + evaluation.

Usage:
    python scripts/run_hierarchical_rag.py \
        --questions-file 财报数据库/test.json \
        --ground-truth 财报数据库/test_ground_truth.json \
        --output outputs/hierarchical_rag.json \
        --backend simple \
        --vector-db-path outputs/vector_db \
        --initial-k 80 \
        --final-pages 5 \
        --max-chars-per-page 5000 \
        --limit 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.config import RETRIEVAL_CHUNKS_PATH
from rag.retriever.unified_rag import UnifiedRAGService
from rag.vector.vector_db import VectorDB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hierarchical RAG pipeline")
    parser.add_argument("--questions-file", default="财报数据库/test.json")
    parser.add_argument("--ground-truth", default="财报数据库/test_ground_truth.json")
    parser.add_argument("--output", default="outputs/hierarchical_rag.json")
    parser.add_argument("--debug-output", default="outputs/hierarchical_rag_debug.json")
    parser.add_argument("--backend", default="simple", choices=["auto", "chroma", "simple"])
    parser.add_argument("--vector-db-path", default="outputs/vector_db")
    parser.add_argument("--initial-k", type=int, default=80)
    parser.add_argument("--final-pages", type=int, default=5)
    parser.add_argument("--max-chars-per-page", type=int, default=5000)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all remaining")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--skip-hierarchical", action="store_true", help="Run flat RAG as baseline instead")
    parser.add_argument("--run-llm", action="store_true", default=True, help="Generate answers with LLM")
    parser.add_argument("--no-run-llm", action="store_false", dest="run_llm", help="Skip LLM answer generation (retrieval only)")
    return parser.parse_args()


def load_json(path: str | Path) -> list | dict:
    file_path = ROOT_DIR / path if not Path(path).is_absolute() else Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data, indent: int = 2) -> None:
    file_path = ROOT_DIR / path if not Path(path).is_absolute() else Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(data, ensure_ascii=False, indent=indent), encoding="utf-8")


def normalize_page(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def evaluate(predictions: list[dict], ground_truth: list[dict]) -> dict:
    if len(predictions) != len(ground_truth):
        raise ValueError(f"Length mismatch: {len(predictions)} vs {len(ground_truth)}")

    total = len(ground_truth)
    file_matches = []
    page_matches = []
    answer_scores = []
    cases = []

    for idx, (pred, truth) in enumerate(zip(predictions, ground_truth)):
        file_match = pred.get("filename") == truth.get("filename")
        page_match = pred.get("page") == truth.get("page")
        answer = pred.get("answer") or ""
        truth_answer = truth.get("answer") or ""

        # Simple char F1
        from collections import Counter
        pred_norm = re.sub(r"\s+", "", answer.lower())
        truth_norm = re.sub(r"\s+", "", truth_answer.lower())
        if not pred_norm and not truth_norm:
            f1 = 1.0
        elif not pred_norm or not truth_norm:
            f1 = 0.0
        else:
            pc = Counter(pred_norm)
            tc = Counter(truth_norm)
            common = sum((pc & tc).values())
            if common <= 0:
                f1 = 0.0
            else:
                precision = common / len(pred_norm)
                recall = common / len(truth_norm)
                f1 = 2 * precision * recall / (precision + recall)

        file_matches.append(file_match)
        page_matches.append(page_match)
        answer_scores.append(f1)
        cases.append({
            "index": idx,
            "question": truth.get("question", ""),
            "truth": {
                "filename": truth.get("filename"),
                "page": truth.get("page"),
                "answer": truth_answer,
            },
            "prediction": {
                "filename": pred.get("filename"),
                "page": pred.get("page"),
                "answer": answer,
            },
            "file_match": file_match,
            "page_match": page_match,
            "answer_char_f1": f1,
        })

    cases.sort(key=lambda x: (x["file_match"] and x["page_match"], x["answer_char_f1"]))
    summary = {
        "total": total,
        "filename_accuracy": sum(file_matches) / total,
        "page_accuracy": sum(page_matches) / total,
        "file_and_page_accuracy": sum(1 for f, p in zip(file_matches, page_matches) if f and p) / total,
        "answer_char_f1_avg": sum(answer_scores) / total,
        "rough_score": (
            0.2 * sum(file_matches) / total
            + 0.2 * sum(page_matches) / total
            + 0.6 * sum(answer_scores) / total
        ),
    }
    return {"summary": summary, "cases": cases}


def main() -> None:
    import re  # noqa: F811 – imported above but used inside evaluate; keep here for closure
    args = parse_args()

    questions = load_json(args.questions_file)
    if not isinstance(questions, list):
        raise ValueError("questions file must be a JSON list")

    end_index = len(questions) if args.limit <= 0 else min(len(questions), args.start_index + args.limit)
    selected = list(enumerate(questions[args.start_index:end_index], start=args.start_index))

    print(f"Running {'flat' if args.skip_hierarchical else 'hierarchical'} RAG on {len(selected)} questions...")

    vector_db = VectorDB(
        persist_directory=args.vector_db_path,
        backend=args.backend,
    )

    if args.skip_hierarchical:
        from rag.retriever.rag import RAGService
        rag = RAGService(vector_db=vector_db)
    else:
        rag = UnifiedRAGService(vector_db=vector_db)

    predictions = []
    debug_items = []

    for offset, (idx, item) in enumerate(selected, start=1):
        question = item["question"]
        filename = item.get("filename")

        try:
            if args.skip_hierarchical:
                result = rag.answer(
                    question,
                    run_llm=args.run_llm,
                    max_chars_per_page=args.max_chars_per_page,
                )
            else:
                result = rag.answer(
                    question,
                    filename=filename,
                    run_llm=args.run_llm,
                    max_chars_per_page=args.max_chars_per_page,
                )
        except Exception as exc:
            result = {
                "filename": filename or "",
                "page": -1,
                "answer": "",
                "sources": [],
                "llm_used": False,
                "raw_answer": "",
                "error": f"exception: {exc}",
            }

        pred = {
            "filename": result.get("filename") or "",
            "page": normalize_page(result.get("page", -1)),
            "answer": result.get("answer") or "",
        }
        predictions.append(pred)

        debug_item = {
            "index": idx,
            "question": question,
            "prediction": pred,
            "sources": result.get("sources", []),
            "route_info": result.get("route_info") if not args.skip_hierarchical else None,
            "error": result.get("error", ""),
            "raw_answer": result.get("raw_answer", ""),
        }
        debug_items.append(debug_item)

        # Periodic save
        if args.progress_every > 0 and (
            offset == 1 or offset == len(selected) or offset % args.progress_every == 0
        ):
            print(f"  processed {offset}/{len(selected)} index={idx} error={result.get('error') or 'ok'}")
            write_json(args.output, predictions)
            write_json(args.debug_output, debug_items)

        if args.sleep_seconds > 0 and offset < len(selected):
            time.sleep(args.sleep_seconds)

    # Final save
    write_json(args.output, predictions)
    write_json(args.debug_output, debug_items)

    # Update questions file with predictions (same format as cli.py)
    filled = [dict(q) for q in questions]
    for idx, pred in enumerate(predictions):
        if idx < len(filled):
            filled[idx]["filename"] = pred["filename"]
            filled[idx]["page"] = pred["page"]
            filled[idx]["answer"] = pred["answer"]
    write_json(args.questions_file, filled, indent=4)

    print(f"\nPredictions saved to {args.output}")

    # Evaluation
    if not args.skip_evaluate and (ROOT_DIR / args.ground_truth).exists():
        ground_truth = load_json(args.ground_truth)
        eval_result = evaluate(predictions, ground_truth)
        eval_path = Path(args.output).with_suffix(".eval.json")
        write_json(eval_path, eval_result)
        print(f"Evaluation saved to {eval_path}")
        print(json.dumps(eval_result["summary"], ensure_ascii=False, indent=2))
    else:
        print("Ground truth not found or evaluation skipped.")


if __name__ == "__main__":
    main()
