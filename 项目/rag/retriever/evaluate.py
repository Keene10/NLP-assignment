from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate filled test.json against a ground-truth JSON file.")
    parser.add_argument("--predictions", default="财报数据库/test.json")
    parser.add_argument("--ground-truth", default="财报数据库/test_ground_truth.json")
    parser.add_argument("--output", default="outputs/final_evaluation.json")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    file_path = Path(path)
    return file_path if file_path.is_absolute() else ROOT_DIR / file_path


def load_json(path: str | Path):
    with resolve_path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


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


def evaluate(predictions: list[dict], ground_truth: list[dict]) -> dict:
    if len(predictions) != len(ground_truth):
        raise ValueError(f"Length mismatch: predictions={len(predictions)}, ground_truth={len(ground_truth)}")

    total = len(ground_truth)
    cases = []
    file_matches = []
    page_matches = []
    answer_scores = []
    answer_ratios = []
    for index, (prediction, truth) in enumerate(zip(predictions, ground_truth)):
        file_match = prediction.get("filename") == truth.get("filename")
        page_match = prediction.get("page") == truth.get("page")
        answer = prediction.get("answer") or ""
        truth_answer = truth.get("answer") or ""
        f1 = char_f1(answer, truth_answer)
        ratio = SequenceMatcher(None, normalize_text(answer), normalize_text(truth_answer)).ratio()
        file_matches.append(file_match)
        page_matches.append(page_match)
        answer_scores.append(f1)
        answer_ratios.append(ratio)
        cases.append(
            {
                "index": index,
                "question": truth.get("question", ""),
                "truth": {
                    "filename": truth.get("filename"),
                    "page": truth.get("page"),
                    "answer": truth_answer,
                },
                "prediction": {
                    "filename": prediction.get("filename"),
                    "page": prediction.get("page"),
                    "answer": answer,
                },
                "file_match": file_match,
                "page_match": page_match,
                "answer_char_f1": f1,
                "answer_sequence_ratio": ratio,
            }
        )

    cases.sort(key=lambda item: (item["file_match"] and item["page_match"], item["answer_char_f1"]))
    summary = {
        "total": total,
        "filename_accuracy": sum(file_matches) / total,
        "page_accuracy": sum(page_matches) / total,
        "file_and_page_accuracy": sum(1 for f, p in zip(file_matches, page_matches) if f and p) / total,
        "answer_char_f1_avg": sum(answer_scores) / total,
        "answer_sequence_ratio_avg": sum(answer_ratios) / total,
        "answer_empty_count": sum(1 for item in predictions if not item.get("answer")),
        "rough_score": (
            0.2 * sum(file_matches) / total
            + 0.2 * sum(page_matches) / total
            + 0.6 * sum(answer_scores) / total
        ),
    }
    return {"summary": summary, "worst_cases": cases[:20]}


def main() -> None:
    args = parse_args()
    result = evaluate(load_json(args.predictions), load_json(args.ground_truth))
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
