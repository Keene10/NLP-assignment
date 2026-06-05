from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.config import OPENAI_API_KEY, RETRIEVAL_CHUNKS_PATH
from rag.retriever.page_selector import select_evidence_pages
from rag.retriever.rag import RAGService
from rag.vector.vector_db import VectorDB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final financial-report RAG pipeline. Results are filled into the selected test JSON."
    )
    parser.add_argument("--questions-file", default="财报数据库/test_new.json")
    parser.add_argument("--ground-truth", default="财报数据库/test_new_ground_truth.json")
    parser.add_argument("--backend", default="simple", choices=["auto", "chroma", "simple"])
    parser.add_argument("--vector-db-path", default="outputs/vector_db")
    parser.add_argument("--base-page-plan", default="outputs/optimized_page_plan_new.json")
    parser.add_argument("--selected-page-plan", default="outputs/llm_selected_page_plan_new.json")
    parser.add_argument("--page-selector-debug", default="outputs/llm_selected_page_plan_debug_new.json")
    parser.add_argument("--debug-output", default="outputs/final_debug_new.json")
    parser.add_argument("--evaluation-output", default="outputs/final_evaluation_new.json")
    parser.add_argument("--chunks", default=RETRIEVAL_CHUNKS_PATH)
    parser.add_argument("--initial-k", type=int, default=200)
    parser.add_argument("--base-retrieval-pages", type=int, default=50)
    parser.add_argument("--neighbor-pages", type=int, default=4)
    parser.add_argument("--page-selector-neighbor-pages", type=int, default=4)
    parser.add_argument("--page-selector-max-chars", type=int, default=1500)
    parser.add_argument("--max-chars-per-page", type=int, default=1800)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all remaining questions.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-page-selector", action="store_true")
    parser.add_argument("--use-existing-selected-page-plan", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path):
    file_path = ROOT_DIR / path if not Path(path).is_absolute() else Path(path)
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, data, indent: int = 2) -> None:
    file_path = ROOT_DIR / path if not Path(path).is_absolute() else Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(data, ensure_ascii=False, indent=indent), encoding="utf-8")


def load_questions(path: str | Path) -> list[dict]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("questions file must be a JSON list")
    for index, item in enumerate(data):
        if not isinstance(item, dict) or "question" not in item:
            raise ValueError(f"invalid question item at index {index}")
    return data


def load_page_plan(path: str | Path) -> dict[int, dict]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("page plan must be a JSON list")
    plan = {}
    for position, item in enumerate(data):
        index = int(item.get("index", position))
        plan[index] = {
            "filename": item["filename"],
            "page": int(item["page"]),
        }
    return plan


def normalize_page(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def make_prediction(result: dict) -> dict:
    return {
        "filename": result.get("filename") or "",
        "page": normalize_page(result.get("page", -1)),
        "answer": result.get("answer") or "",
    }


def make_debug_item(index: int, question_item: dict, result: dict) -> dict:
    return {
        "index": index,
        "question": question_item["question"],
        "predicted": make_prediction(result),
        "llm_used": result.get("llm_used", False),
        "error": result.get("error", ""),
        "sources": result.get("sources", []),
        "raw_answer": result.get("raw_answer", ""),
        "page_plan": result.get("page_plan"),
    }


def is_done(debug_item: dict) -> bool:
    predicted = debug_item.get("predicted") or {}
    return bool(debug_item.get("llm_used")) and bool(predicted.get("answer"))


def load_resume_debug(path: str | Path) -> dict[int, dict]:
    file_path = ROOT_DIR / path if not Path(path).is_absolute() else Path(path)
    if not file_path.exists():
        return {}
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return {}
    return {int(item["index"]): item for item in data if isinstance(item, dict) and "index" in item}


def fill_questions_file(
    questions_path: str | Path,
    questions: list[dict],
    debug_by_index: dict[int, dict],
) -> None:
    filled = [dict(item) for item in questions]
    for index, debug_item in debug_by_index.items():
        if index < 0 or index >= len(filled):
            continue
        prediction = debug_item.get("predicted") or {}
        filled[index]["filename"] = prediction.get("filename") or ""
        filled[index]["page"] = normalize_page(prediction.get("page", -1))
        filled[index]["answer"] = prediction.get("answer") or ""
    write_json(questions_path, filled, indent=4)


def build_base_page_plan(args: argparse.Namespace, questions: list[dict]) -> dict[int, dict]:
    plan_path = ROOT_DIR / args.base_page_plan
    if plan_path.exists():
        return load_page_plan(plan_path)

    print("未找到基础页码计划，开始用本地检索生成，不调用 API。")
    rag = RAGService(
        vector_db=VectorDB(
            persist_directory=args.vector_db_path,
            backend=args.backend,
        )
    )
    end_index = len(questions) if args.limit <= 0 else min(len(questions), args.start_index + args.limit)
    selected = list(enumerate(questions[args.start_index:end_index], start=args.start_index))
    plan_items = []
    for offset, (index, item) in enumerate(selected, start=1):
        pages = rag.retrieve_pages(
            item["question"],
            initial_k=args.initial_k,
            final_pages=args.base_retrieval_pages,
            max_chars_per_page=args.max_chars_per_page,
        )
        if not pages:
            raise RuntimeError(f"local retrieval found no page for question {index}")
        plan_items.append(
            {
                "index": index,
                "filename": pages[0].filename,
                "page": normalize_page(pages[0].page_number),
            }
        )
        if args.progress_every > 0 and (
            offset == 1 or offset == len(selected) or offset % args.progress_every == 0
        ):
            print(f"base page planned {offset}/{len(selected)} index={index}")
    write_json(plan_path, plan_items)
    return load_page_plan(plan_path)


def run_page_selection(args: argparse.Namespace, questions: list[dict], base_plan: dict[int, dict]) -> dict[int, dict]:
    selected_plan_path = ROOT_DIR / args.selected_page_plan
    if args.use_existing_selected_page_plan and selected_plan_path.exists():
        return load_page_plan(selected_plan_path)

    selected_plan, selector_debug = select_evidence_pages(
        questions=questions,
        base_plan=base_plan,
        chunks_path=args.chunks,
        radius=args.page_selector_neighbor_pages,
        max_chars_per_page=args.page_selector_max_chars,
        start_index=args.start_index,
        limit=args.limit,
        progress_every=args.progress_every,
        sleep_seconds=args.sleep_seconds,
    )
    write_json(args.selected_page_plan, selected_plan)
    write_json(args.page_selector_debug, selector_debug)
    return load_page_plan(args.selected_page_plan)


def run_generation(args: argparse.Namespace, questions: list[dict], page_plan: dict[int, dict], force_page: bool) -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY 为空，请先在 .env 中填写 API key。")

    rag = RAGService(
        vector_db=VectorDB(
            persist_directory=args.vector_db_path,
            backend=args.backend,
        )
    )
    end_index = len(questions) if args.limit <= 0 else min(len(questions), args.start_index + args.limit)
    selected = list(enumerate(questions[args.start_index:end_index], start=args.start_index))
    selected_indices = [index for index, _ in selected]
    debug_by_index = load_resume_debug(args.debug_output) if args.resume else {}

    for offset, (index, item) in enumerate(selected, start=1):
        if index in debug_by_index and is_done(debug_by_index[index]):
            continue

        planned = page_plan.get(index)
        if not planned:
            raise RuntimeError(f"missing page plan for question {index}")

        try:
            result = rag.answer_forced_page(
                item["question"],
                filename=planned["filename"],
                page=planned["page"],
                neighbor_pages=args.neighbor_pages,
                max_chars_per_page=args.max_chars_per_page,
                run_llm=True,
                include_prompt=False,
                force_page=force_page,
            )
        except Exception as exc:
            result = {
                "filename": planned["filename"],
                "page": planned["page"],
                "answer": "",
                "sources": [],
                "llm_used": False,
                "raw_answer": "",
                "error": f"exception: {exc}",
                "page_plan": planned,
            }

        debug_by_index[index] = make_debug_item(index, item, result)
        ordered_debug = [debug_by_index[i] for i in selected_indices if i in debug_by_index]
        write_json(args.debug_output, ordered_debug)
        fill_questions_file(args.questions_file, questions, debug_by_index)

        if args.progress_every > 0 and (
            offset == 1 or offset == len(selected) or offset % args.progress_every == 0
        ):
            print(f"answered {offset}/{len(selected)} index={index} status={result.get('error') or 'ok'}")
        if args.sleep_seconds > 0 and offset < len(selected):
            time.sleep(args.sleep_seconds)

    return {
        "debug_output": str((ROOT_DIR / args.debug_output).resolve()),
        "questions_file": str((ROOT_DIR / args.questions_file).resolve()),
        "processed_questions": len(selected),
    }


def normalize_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。！？；：、\"'（）()\[\]【】《》,.!?;:\-_*#<>/\\|]", "", text)
    return text.lower()


def char_f1(prediction: str, truth: str) -> float:
    prediction = normalize_text(prediction)
    truth = normalize_text(truth)
    if not prediction and not truth:
        return 1.0
    if not prediction or not truth:
        return 0.0
    pred_counter = Counter(prediction)
    truth_counter = Counter(truth)
    common = sum((pred_counter & truth_counter).values())
    if common <= 0:
        return 0.0
    precision = common / len(prediction)
    recall = common / len(truth)
    return 2 * precision * recall / (precision + recall)


def evaluate_predictions(predictions: list[dict], ground_truth: list[dict]) -> dict:
    if len(predictions) != len(ground_truth):
        raise ValueError("prediction and ground-truth lengths are different")
    total = len(ground_truth)
    cases = []
    file_matches = []
    page_matches = []
    answer_scores = []
    for index, (prediction, truth) in enumerate(zip(predictions, ground_truth)):
        file_match = prediction.get("filename") == truth.get("filename")
        page_match = prediction.get("page") == truth.get("page")
        score = char_f1(prediction.get("answer") or "", truth.get("answer") or "")
        file_matches.append(file_match)
        page_matches.append(page_match)
        answer_scores.append(score)
        cases.append(
            {
                "index": index,
                "question": truth.get("question", ""),
                "truth": {
                    "filename": truth.get("filename"),
                    "page": truth.get("page"),
                    "answer": truth.get("answer"),
                },
                "prediction": {
                    "filename": prediction.get("filename"),
                    "page": prediction.get("page"),
                    "answer": prediction.get("answer"),
                },
                "file_match": file_match,
                "page_match": page_match,
                "answer_char_f1": score,
                "answer_sequence_ratio": SequenceMatcher(
                    None,
                    normalize_text(prediction.get("answer") or ""),
                    normalize_text(truth.get("answer") or ""),
                ).ratio(),
            }
        )

    cases.sort(key=lambda item: (item["file_match"] and item["page_match"], item["answer_char_f1"]))
    summary = {
        "total": total,
        "filename_accuracy": sum(file_matches) / total,
        "page_accuracy": sum(page_matches) / total,
        "file_and_page_accuracy": sum(1 for f, p in zip(file_matches, page_matches) if f and p) / total,
        "answer_char_f1_avg": sum(answer_scores) / total,
        "answer_empty_count": sum(1 for item in predictions if not item.get("answer")),
        "rough_score": 0.2 * sum(file_matches) / total + 0.2 * sum(page_matches) / total + 0.6 * sum(answer_scores) / total,
    }
    return {"summary": summary, "worst_cases": cases[:20]}


def main() -> None:
    args = parse_args()
    questions = load_questions(args.questions_file)
    base_plan = build_base_page_plan(args, questions)

    if args.skip_page_selector:
        active_plan = base_plan
        force_page = False
    else:
        print("开始 API 页码选择阶段：只选证据页，不生成答案。")
        active_plan = run_page_selection(args, questions, base_plan)
        force_page = True

    print("开始 API 答案生成，并把结果直接填入 test.json。")
    result = run_generation(args, questions, active_plan, force_page=force_page)

    if not args.skip_evaluate and (ROOT_DIR / args.ground_truth).exists():
        evaluation = evaluate_predictions(load_questions(args.questions_file), load_json(args.ground_truth))
        write_json(args.evaluation_output, evaluation)
        result["evaluation_output"] = str((ROOT_DIR / args.evaluation_output).resolve())
        result.update(evaluation["summary"])

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
