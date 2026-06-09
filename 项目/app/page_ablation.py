from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from config.config import VECTOR_DB_PATH
from rag.retriever.evaluate import evaluate
from rag.retriever.page_calibration import HybridPageExperiment
from rag.retriever.rag import RAGService
from rag.vector.vector_db import VectorDB


STAGE_ORDER = [
    "A_pure_hybrid",
    "B_summary_penalty",
    "C_anchor_calibration",
    "D_neighbor_calibration",
    "E_selective_reranker",
    "F_advanced_guarded",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline page-selection ablation for test_new.")
    parser.add_argument("--questions-file", default="财报数据库/test_new.json")
    parser.add_argument("--ground-truth", default="财报数据库/test/test_new_ground_truth.json")
    parser.add_argument("--backend", default="simple", choices=["auto", "chroma", "simple"])
    parser.add_argument("--vector-db-path", default=VECTOR_DB_PATH)
    parser.add_argument("--output-dir", default="outputs/page_ablation_test_new")
    parser.add_argument("--initial-k", type=int, default=200)
    parser.add_argument("--candidate-pages", type=int, default=50)
    parser.add_argument("--max-chars-per-page", type=int, default=1800)
    parser.add_argument("--progress-every", type=int, default=5)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def load_json(path: str | Path):
    with resolve_path(path).open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def write_json(path: str | Path, data) -> None:
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_page(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def by_file_accuracy(predictions: list[dict], ground_truth: list[dict]) -> dict[str, dict]:
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for prediction, truth in zip(predictions, ground_truth):
        filename = truth.get("filename") or ""
        stats[filename][0] += 1
        if prediction.get("filename") == truth.get("filename") and prediction.get("page") == truth.get("page"):
            stats[filename][1] += 1
    return {
        filename: {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
        }
        for filename, (total, correct) in sorted(stats.items())
    }


def wrong_cases(predictions: list[dict], ground_truth: list[dict], debug_items: list[dict]) -> list[dict]:
    cases = []
    for index, (prediction, truth, debug) in enumerate(zip(predictions, ground_truth, debug_items)):
        page_match = prediction.get("filename") == truth.get("filename") and prediction.get("page") == truth.get("page")
        if page_match:
            continue
        cases.append(
            {
                "index": index,
                "question": truth.get("question", ""),
                "question_type": debug.get("question_type"),
                "truth": {
                    "filename": truth.get("filename"),
                    "page": truth.get("page"),
                },
                "prediction": {
                    "filename": prediction.get("filename"),
                    "page": prediction.get("page"),
                },
                "top5_candidate_pages": debug.get("top5_candidate_pages", []),
                "anchor_scores": debug.get("anchor_scores", []),
                "use_reranker": debug.get("use_reranker", False),
            }
        )
    return cases


def transition_effect(
    previous_predictions: list[dict] | None,
    current_predictions: list[dict],
    ground_truth: list[dict],
) -> dict:
    if previous_predictions is None:
        return {
            "changed": 0,
            "fixed": 0,
            "regressed": 0,
            "wrong_to_wrong_changed": 0,
        }
    changed = fixed = regressed = wrong_to_wrong_changed = 0
    for previous, current, truth in zip(previous_predictions, current_predictions, ground_truth):
        previous_ok = previous.get("filename") == truth.get("filename") and previous.get("page") == truth.get("page")
        current_ok = current.get("filename") == truth.get("filename") and current.get("page") == truth.get("page")
        if (previous.get("filename"), previous.get("page")) == (current.get("filename"), current.get("page")):
            continue
        changed += 1
        if not previous_ok and current_ok:
            fixed += 1
        elif previous_ok and not current_ok:
            regressed += 1
        elif not previous_ok and not current_ok:
            wrong_to_wrong_changed += 1
    return {
        "changed": changed,
        "fixed": fixed,
        "regressed": regressed,
        "wrong_to_wrong_changed": wrong_to_wrong_changed,
    }


def make_prediction(stage_result) -> dict:
    return {
        "filename": stage_result.filename,
        "page": normalize_page(stage_result.page),
        "answer": "",
    }


def main() -> None:
    args = parse_args()
    questions = load_json(args.questions_file)
    ground_truth = load_json(args.ground_truth)
    if len(questions) != len(ground_truth):
        raise ValueError(f"Length mismatch: questions={len(questions)}, ground_truth={len(ground_truth)}")

    rag = RAGService(
        vector_db=VectorDB(
            persist_directory=args.vector_db_path,
            backend=args.backend,
        )
    )
    experiment = HybridPageExperiment(rag=rag, candidate_pages=args.candidate_pages)

    predictions_by_stage = {stage: [] for stage in STAGE_ORDER}
    debug_by_stage = {stage: [] for stage in STAGE_ORDER}

    for index, item in enumerate(questions):
        candidates = experiment.retrieve_hybrid_candidates(
            item["question"],
            initial_k=args.initial_k,
            max_chars_per_page=args.max_chars_per_page,
        )
        if not candidates:
            raise RuntimeError(f"No page candidates for question index={index}")
        stage_results = experiment.run_all_stages(item["question"], candidates)
        for stage in STAGE_ORDER:
            result = stage_results[stage]
            predictions_by_stage[stage].append(make_prediction(result))
            debug_item = {
                "index": index,
                "question": item["question"],
                **result.debug,
            }
            debug_by_stage[stage].append(debug_item)
        if args.progress_every > 0 and (
            index == 0 or index + 1 == len(questions) or (index + 1) % args.progress_every == 0
        ):
            print(f"ablation planned {index + 1}/{len(questions)}")

    output_dir = resolve_path(args.output_dir)
    summary = {}
    previous_predictions = None
    for stage in STAGE_ORDER:
        predictions = predictions_by_stage[stage]
        debug_items = debug_by_stage[stage]
        evaluation = evaluate(predictions, ground_truth)
        stage_result = {
            "summary": evaluation["summary"],
            "by_file_accuracy": by_file_accuracy(predictions, ground_truth),
            "transition_effect_vs_previous": transition_effect(previous_predictions, predictions, ground_truth),
            "wrong_cases": wrong_cases(predictions, ground_truth, debug_items),
            "debug": debug_items,
            "predictions": predictions,
        }
        write_json(output_dir / f"{stage}.json", stage_result)
        summary[stage] = {
            "page_accuracy": evaluation["summary"]["page_accuracy"],
            "file_and_page_accuracy": evaluation["summary"]["file_and_page_accuracy"],
            "transition_effect_vs_previous": stage_result["transition_effect_vs_previous"],
            "wrong_count": len(stage_result["wrong_cases"]),
        }
        previous_predictions = predictions

    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote ablation files to {output_dir}")


if __name__ == "__main__":
    main()
