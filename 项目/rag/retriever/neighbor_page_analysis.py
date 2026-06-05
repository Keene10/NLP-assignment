from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


def normalize_page(value) -> int | None:
    try:
        page = int(value)
    except (TypeError, ValueError):
        return None
    return page if page >= 0 else None


def _empty_file_stats() -> dict:
    return {
        "total": 0,
        "filename_correct": 0,
        "exact_page_correct": 0,
        "within_1_page": 0,
        "within_2_pages": 0,
        "wrong_file_count": 0,
        "missing_prediction_count": 0,
    }


def _add_case_to_stats(stats: dict, case: dict) -> None:
    stats["total"] += 1
    if case["missing_prediction"]:
        stats["missing_prediction_count"] += 1
    if case["filename_match"]:
        stats["filename_correct"] += 1
    else:
        stats["wrong_file_count"] += 1
    if case["exact_page_match"]:
        stats["exact_page_correct"] += 1
    if case["within_1_page"]:
        stats["within_1_page"] += 1
    if case["within_2_pages"]:
        stats["within_2_pages"] += 1


def _with_rates(stats: dict) -> dict:
    total = stats["total"] or 1
    enriched = dict(stats)
    enriched["filename_accuracy"] = stats["filename_correct"] / total
    enriched["exact_page_accuracy"] = stats["exact_page_correct"] / total
    enriched["within_1_page_accuracy"] = stats["within_1_page"] / total
    enriched["within_2_pages_accuracy"] = stats["within_2_pages"] / total
    enriched["near_miss_within_1"] = stats["within_1_page"] - stats["exact_page_correct"]
    enriched["near_miss_within_2"] = stats["within_2_pages"] - stats["exact_page_correct"]
    return enriched


def analyze_neighbor_pages(predictions: list[dict], ground_truth: list[dict]) -> dict:
    total = len(ground_truth)
    cases: list[dict] = []
    summary_stats = _empty_file_stats()
    by_file_stats: dict[str, dict] = {}

    for index, truth in enumerate(ground_truth):
        prediction = predictions[index] if index < len(predictions) else {}
        truth_filename = truth.get("filename") or ""
        predicted_filename = prediction.get("filename") or ""
        truth_page = normalize_page(truth.get("page"))
        predicted_page = normalize_page(prediction.get("page"))
        filename_match = bool(predicted_filename) and predicted_filename == truth_filename
        missing_prediction = not predicted_filename or predicted_page is None

        page_delta = None
        if filename_match and truth_page is not None and predicted_page is not None:
            page_delta = predicted_page - truth_page

        exact_page_match = page_delta == 0
        within_1_page = page_delta is not None and abs(page_delta) <= 1
        within_2_pages = page_delta is not None and abs(page_delta) <= 2

        case = {
            "index": index,
            "question": truth.get("question", ""),
            "truth_filename": truth_filename,
            "truth_page": truth_page,
            "predicted_filename": predicted_filename,
            "predicted_page": predicted_page,
            "filename_match": filename_match,
            "page_delta": page_delta,
            "exact_page_match": exact_page_match,
            "within_1_page": within_1_page,
            "within_2_pages": within_2_pages,
            "missing_prediction": missing_prediction,
        }
        cases.append(case)
        _add_case_to_stats(summary_stats, case)
        file_stats = by_file_stats.setdefault(truth_filename, _empty_file_stats())
        _add_case_to_stats(file_stats, case)

    near_miss_cases = [
        case
        for case in cases
        if case["filename_match"] and not case["exact_page_match"] and case["within_2_pages"]
    ]
    wrong_cases = [case for case in cases if not case["exact_page_match"]]
    return {
        "summary": _with_rates(summary_stats),
        "by_file": {filename: _with_rates(stats) for filename, stats in sorted(by_file_stats.items())},
        "near_miss_cases": near_miss_cases,
        "wrong_cases": wrong_cases,
        "cases": cases,
        "prediction_count": len(predictions),
        "ground_truth_count": total,
    }


def extract_predictions(data) -> list[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("predictions"), list):
        return data["predictions"]
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        indexed_predictions = []
        for position, case in enumerate(data["cases"]):
            if not isinstance(case, dict):
                continue
            prediction = case.get("prediction")
            if not isinstance(prediction, dict):
                continue
            indexed_predictions.append((int(case.get("index", position)), prediction))
        return [prediction for _, prediction in sorted(indexed_predictions, key=lambda item: item[0])]
    raise ValueError("predictions must be a JSON list or an object containing a predictions list")


def load_json(path: str | Path):
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = ROOT_DIR / file_path
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, data) -> None:
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = ROOT_DIR / file_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze exact and nearby page matches.")
    parser.add_argument("--predictions", default="财报数据库/test.json")
    parser.add_argument("--ground-truth", default="财报数据库/test_ground_truth.json")
    parser.add_argument("--output", default="outputs/neighbor_page_analysis.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_neighbor_pages(extract_predictions(load_json(args.predictions)), load_json(args.ground_truth))
    write_json(args.output, result)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
