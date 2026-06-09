from __future__ import annotations

import argparse
import json
import os
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
os.chdir(ROOT_DIR)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from config.config import (
    OPENAI_API_KEY,
    PAGE_RERANKER_BATCH_SIZE,
    PAGE_RERANKER_CANDIDATES,
    PAGE_RERANKER_MAX_CHARS,
    PAGE_RERANKER_NEIGHBOR_PAGES,
    PAGE_RERANKER_MODEL,
    PAGE_RERANKER_WEIGHT,
    RETRIEVAL_CHUNKS_PATH,
    VECTOR_DB_PATH,
)
from rag.retriever.evaluate import evaluate as evaluate_with_ppt_metrics
from rag.retriever.page_calibration import HybridPageExperiment
from rag.retriever.page_selector import select_evidence_pages
from rag.retriever.rag import RAGService
from rag.vector.vector_db import VectorDB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final financial-report RAG pipeline. Results are filled into test.json."
    )
    parser.add_argument("--questions-file", default="财报数据库/test_new.json")
    parser.add_argument("--ground-truth", default="财报数据库/test/test_new_ground_truth.json")
    parser.add_argument("--backend", default="simple", choices=["auto", "chroma", "simple"])
    parser.add_argument("--vector-db-path", default=VECTOR_DB_PATH)
    parser.add_argument("--base-page-plan", default="outputs/optimized_page_plan.json")
    parser.add_argument("--use-page-plan-file", action="store_true")
    parser.add_argument(
        "--page-plan-mode",
        default="calibrated",
        choices=["calibrated", "legacy"],
        help="calibrated uses hybrid + anchors + selective reranker; legacy uses the old retrieve_pages flow.",
    )
    parser.add_argument("--selected-page-plan", default="outputs/llm_selected_page_plan.json")
    parser.add_argument("--page-selector-debug", default="outputs/llm_selected_page_plan_debug.json")
    parser.add_argument("--debug-output", default="outputs/final_debug.json")
    parser.add_argument("--evaluation-output", default="outputs/final_evaluation.json")
    parser.add_argument("--chunks", default=RETRIEVAL_CHUNKS_PATH)
    parser.add_argument("--initial-k", type=int, default=200)
    parser.add_argument("--base-retrieval-pages", type=int, default=50)
    parser.add_argument("--reranker-model", default=PAGE_RERANKER_MODEL)
    parser.add_argument("--reranker-candidates", type=int, default=PAGE_RERANKER_CANDIDATES)
    parser.add_argument("--reranker-batch-size", type=int, default=PAGE_RERANKER_BATCH_SIZE)
    parser.add_argument("--reranker-max-chars", type=int, default=PAGE_RERANKER_MAX_CHARS)
    parser.add_argument("--reranker-weight", type=float, default=PAGE_RERANKER_WEIGHT)
    parser.add_argument("--reranker-neighbor-pages", type=int, default=PAGE_RERANKER_NEIGHBOR_PAGES)
    parser.add_argument(
        "--reranker-query-mode",
        default="original",
        choices=["original", "keywords", "original_keywords"],
    )
    parser.add_argument("--disable-reranker", action="store_true")
    parser.add_argument("--disable-targeted-retrieval", action="store_true")
    parser.add_argument("--allow-reranker-cross-file", action="store_true")
    parser.add_argument("--neighbor-pages", type=int, default=4)
    parser.add_argument("--page-selector-top-k", type=int, default=5)
    parser.add_argument("--page-selector-vector-k", type=int, default=80)
    parser.add_argument("--page-selector-keyword-k", type=int, default=80)
    parser.add_argument("--page-selector-neighbor-pages", type=int, default=4)
    parser.add_argument("--page-selector-max-chars", type=int, default=1500)
    parser.add_argument("--retrieval-max-chars-per-page", type=int, default=1800)
    parser.add_argument("--max-chars-per-page", type=int, default=2400)
    parser.add_argument("--answer-context-mode", default="topk", choices=["topk", "selected"])
    parser.add_argument("--answer-max-pages", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all remaining questions.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-page-selector",
        action="store_true",
        default=True,
        help="Default: use local retrieval/reranker page plan directly; do not call API to select page.",
    )
    parser.add_argument(
        "--use-api-page-selector",
        dest="skip_page_selector",
        action="store_false",
        help="Optional ablation: call API to choose final page from local candidates.",
    )
    parser.add_argument("--use-existing-selected-page-plan", action="store_true")
    parser.add_argument("--evaluate-page-only", action="store_true")
    parser.add_argument("--page-evaluation-output", default="outputs/page_plan_evaluation.json")
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
        plan_item = {
            "filename": item["filename"],
            "page": int(item["page"]),
        }
        for field in ("confidence", "reason", "answer_pages"):
            if field in item:
                plan_item[field] = item[field]
        plan[index] = plan_item
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
    if args.use_page_plan_file:
        if not plan_path.exists():
            raise FileNotFoundError(f"page plan file not found: {plan_path}")
        return load_page_plan(plan_path)

    print("开始用本地检索生成临时页码计划，不调用 API，不使用预存页码顺序计划。")
    rag = RAGService(
        vector_db=VectorDB(
            persist_directory=args.vector_db_path,
            backend=args.backend,
        )
    )
    page_experiment = (
        HybridPageExperiment(rag=rag, candidate_pages=args.base_retrieval_pages)
        if args.page_plan_mode == "calibrated"
        else None
    )
    end_index = len(questions) if args.limit <= 0 else min(len(questions), args.start_index + args.limit)
    selected = list(enumerate(questions[args.start_index:end_index], start=args.start_index))
    plan_items = []
    for offset, (index, item) in enumerate(selected, start=1):
        page_plan_debug = {}
        if page_experiment is not None and not args.disable_reranker and not args.disable_targeted_retrieval:
            hybrid_candidates = page_experiment.retrieve_hybrid_candidates(
                item["question"],
                initial_k=args.initial_k,
                max_chars_per_page=args.retrieval_max_chars_per_page,
            )
            stage_results = page_experiment.run_all_stages(item["question"], hybrid_candidates)
            selected_stage = stage_results["F_advanced_guarded"]
            pages = selected_stage.pages
            page_plan_debug = selected_stage.debug
        else:
            pages = rag.retrieve_pages(
                item["question"],
                initial_k=args.initial_k,
                final_pages=args.base_retrieval_pages,
                max_chars_per_page=args.retrieval_max_chars_per_page,
                reranker_model="" if args.disable_reranker else args.reranker_model,
                reranker_candidates=args.reranker_candidates,
                reranker_batch_size=args.reranker_batch_size,
                reranker_max_chars=args.reranker_max_chars,
                reranker_weight=args.reranker_weight,
                restrict_reranker_to_top_file=not args.allow_reranker_cross_file,
                reranker_query_mode=args.reranker_query_mode,
                reranker_neighbor_pages=args.reranker_neighbor_pages,
                targeted_retrieval=not args.disable_targeted_retrieval,
            )
        if not pages:
            raise RuntimeError(f"local retrieval found no page for question {index}")
        plan_items.append(
            {
                "index": index,
                "filename": pages[0].filename,
                "page": normalize_page(pages[0].page_number),
                "page_plan_mode": args.page_plan_mode,
                "page_plan_debug": page_plan_debug,
                "answer_pages": [
                    {
                        "filename": page.filename,
                        "page": normalize_page(page.page_number),
                        "chunk_ids": page.chunk_ids,
                        "fusion_score": page.score,
                    }
                    for page in pages[: max(1, args.answer_max_pages)]
                ],
            }
        )
        if args.progress_every > 0 and (
            offset == 1 or offset == len(selected) or offset % args.progress_every == 0
        ):
            print(f"base page planned {offset}/{len(selected)} index={index}")
    return {int(item["index"]): item for item in plan_items}


def evaluate_page_plan(args: argparse.Namespace, questions: list[dict], page_plan: dict[int, dict]) -> dict:
    end_index = len(questions) if args.limit <= 0 else min(len(questions), args.start_index + args.limit)
    selected_indices = list(range(args.start_index, end_index))
    predictions = []
    for index in selected_indices:
        item = questions[index]
        planned = page_plan.get(index, {})
        predictions.append(
            {
                "question": item.get("question", ""),
                "filename": planned.get("filename", ""),
                "page": normalize_page(planned.get("page", -1)),
                "answer": "",
            }
        )
    if not (ROOT_DIR / args.ground_truth).exists():
        result = {
            "summary": {
                "total": len(predictions),
                "note": "ground truth file not found; page accuracy was not computed",
            },
            "predictions": predictions,
        }
    else:
        ground_truth = load_json(args.ground_truth)
        selected_truth = [ground_truth[index] for index in selected_indices]
        result = evaluate_with_ppt_metrics(predictions, selected_truth)
    write_json(args.page_evaluation_output, result)
    return result


def run_page_selection(args: argparse.Namespace, questions: list[dict], base_plan: dict[int, dict]) -> dict[int, dict]:
    selected_plan_path = ROOT_DIR / args.selected_page_plan
    if args.use_existing_selected_page_plan and selected_plan_path.exists():
        return load_page_plan(selected_plan_path)

    selected_plan, selector_debug = select_evidence_pages(
        questions=questions,
        base_plan=base_plan,
        chunks_path=args.chunks,
        vector_db_path=args.vector_db_path,
        vector_backend=args.backend,
        fusion_top_k=args.page_selector_top_k,
        vector_top_k=args.page_selector_vector_k,
        keyword_top_k=args.page_selector_keyword_k,
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
            answer_pages = list(planned.get("answer_pages") or [])[: max(1, args.answer_max_pages)]
            if args.answer_context_mode == "topk" and force_page and answer_pages:
                result = rag.answer_forced_candidates(
                    item["question"],
                    filename=planned["filename"],
                    page=planned["page"],
                    candidate_pages=answer_pages,
                    neighbor_pages=args.neighbor_pages,
                    max_chars_per_page=args.max_chars_per_page,
                    run_llm=True,
                    include_prompt=False,
                    force_page=force_page,
                )
            else:
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

    if args.evaluate_page_only:
        result = evaluate_page_plan(args, questions, base_plan)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return

    if not args.skip_page_selector:
        print("开始 API 页码选择阶段：只选证据页，不生成答案。")
        active_plan = run_page_selection(args, questions, base_plan)
        force_page = True
    else:
        print("使用本地检索页码计划作为最终 page，不调用 API 选页。")
        active_plan = base_plan
        force_page = True

    print("开始 API 答案生成，并把结果直接填入 test.json。")
    result = run_generation(args, questions, active_plan, force_page=force_page)

    if not args.skip_evaluate and (ROOT_DIR / args.ground_truth).exists():
        evaluation = evaluate_with_ppt_metrics(load_questions(args.questions_file), load_json(args.ground_truth))
        write_json(args.evaluation_output, evaluation)
        result["evaluation_output"] = str((ROOT_DIR / args.evaluation_output).resolve())
        result.update(evaluation["summary"])

        # 将每个问题的评测分数写回 questions_file
        summary = evaluation["summary"]
        rouge1_avg = summary["answer_rouge1_avg"]
        rouge2_avg = summary["answer_rouge2_avg"]
        bleu_avg = summary["answer_bleu_avg"]
        case_by_index: dict[int, dict] = {c["index"]: c for c in evaluation.get("cases", [])}
        scored_questions = []
        for idx, item in enumerate(questions):
            item = dict(item)
            case = case_by_index.get(idx)
            if case is not None:
                file_match = int(case.get("file_match", False))
                page_match = int(case.get("page_match", False))
                answer_similarity = case.get("answer_similarity_avg", 0.0)
                item["rouge1"] = case.get("answer_rouge1", 0.0)
                item["rouge2"] = case.get("answer_rouge2", 0.0)
                item["bleu"] = case.get("answer_bleu", 0.0)
                item["file_accuracy"] = file_match
                item["page_accuracy"] = page_match
                item["page_score"] = round(0.2 * page_match, 2)
                item["rouge1_avg"] = rouge1_avg
                item["rouge2_avg"] = rouge2_avg
                item["bleu_avg"] = bleu_avg
                item["final_score"] = round(
                    0.2 * file_match + 0.2 * page_match + 0.6 * answer_similarity, 6
                )
            scored_questions.append(item)
        write_json(args.questions_file, scored_questions, indent=4)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
