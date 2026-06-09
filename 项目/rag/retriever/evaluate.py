from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


TOKEN_PATTERN = re.compile(
    r"[a-zA-Z]+(?:[-+./][a-zA-Z0-9]+)*|\d+(?:\.\d+)?%?|[\u4e00-\u9fff]"
)
PUNCT_PATTERN = re.compile(r"[\s，。！？；：、\"'“”‘’（）()\[\]【】《》,.!?;:\-_*#<>/\\|]+")


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
    text = PUNCT_PATTERN.sub("", text)
    return text.lower()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text or "").lower())


def ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def rouge_n_f1(prediction: str, truth: str, n: int) -> float:
    pred_counts = ngram_counts(tokenize(prediction), n)
    truth_counts = ngram_counts(tokenize(truth), n)
    if not pred_counts and not truth_counts:
        return 1.0
    if not pred_counts or not truth_counts:
        return 0.0

    overlap = sum((pred_counts & truth_counts).values())
    if overlap <= 0:
        return 0.0
    precision = overlap / sum(pred_counts.values())
    recall = overlap / sum(truth_counts.values())
    return 2 * precision * recall / (precision + recall)


def bleu_score(prediction: str, truth: str, max_order: int = 4) -> float:
    pred_tokens = tokenize(prediction)
    truth_tokens = tokenize(truth)
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0

    effective_order = min(max_order, len(pred_tokens))
    precisions = []
    for n in range(1, effective_order + 1):
        pred_counts = ngram_counts(pred_tokens, n)
        truth_counts = ngram_counts(truth_tokens, n)
        possible = sum(pred_counts.values())
        if possible <= 0:
            continue
        overlap = sum((pred_counts & truth_counts).values())
        if n == 1:
            precision = overlap / possible if overlap > 0 else 0.0
        else:
            precision = (overlap + 1) / (possible + 1)
        precisions.append(precision)

    if not precisions or precisions[0] <= 0:
        return 0.0

    brevity_penalty = 1.0
    if len(pred_tokens) < len(truth_tokens):
        brevity_penalty = math.exp(1 - len(truth_tokens) / max(len(pred_tokens), 1))

    log_precision = sum(math.log(max(precision, 1e-12)) for precision in precisions) / len(precisions)
    return max(0.0, min(1.0, brevity_penalty * math.exp(log_precision)))


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
    rouge1_scores = []
    rouge2_scores = []
    bleu_scores = []
    char_f1_scores = []

    for index, (prediction, truth) in enumerate(zip(predictions, ground_truth)):
        file_match = prediction.get("filename") == truth.get("filename")
        page_match = prediction.get("page") == truth.get("page")
        answer = prediction.get("answer") or ""
        truth_answer = truth.get("answer") or ""

        rouge1 = rouge_n_f1(answer, truth_answer, 1)
        rouge2 = rouge_n_f1(answer, truth_answer, 2)
        bleu = bleu_score(answer, truth_answer)
        answer_similarity = (rouge1 + rouge2 + bleu) / 3
        legacy_char_f1 = char_f1(answer, truth_answer)

        file_matches.append(file_match)
        page_matches.append(page_match)
        rouge1_scores.append(rouge1)
        rouge2_scores.append(rouge2)
        bleu_scores.append(bleu)
        char_f1_scores.append(legacy_char_f1)
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
                "answer_rouge1": rouge1,
                "answer_rouge2": rouge2,
                "answer_bleu": bleu,
                "answer_similarity_avg": answer_similarity,
                "answer_content_score_0_6": 0.6 * answer_similarity,
                "answer_char_f1_legacy": legacy_char_f1,
            }
        )

    avg_rouge1 = sum(rouge1_scores) / total if total else 0.0
    avg_rouge2 = sum(rouge2_scores) / total if total else 0.0
    avg_bleu = sum(bleu_scores) / total if total else 0.0
    answer_similarity_avg = (avg_rouge1 + avg_rouge2 + avg_bleu) / 3

    cases.sort(key=lambda item: (item["file_match"] and item["page_match"], item["answer_similarity_avg"]))
    summary = {
        "total": total,
        "filename_accuracy": sum(file_matches) / total if total else 0.0,
        "page_accuracy": sum(page_matches) / total if total else 0.0,
        "file_and_page_accuracy": sum(1 for f, p in zip(file_matches, page_matches) if f and p) / total
        if total
        else 0.0,
        "answer_rouge1_avg": avg_rouge1,
        "answer_rouge2_avg": avg_rouge2,
        "answer_bleu_avg": avg_bleu,
        "answer_similarity_avg": answer_similarity_avg,
        "answer_content_score_0_6": 0.6 * answer_similarity_avg,
        "answer_char_f1_legacy_avg": sum(char_f1_scores) / total if total else 0.0,
        "answer_empty_count": sum(1 for item in predictions if not item.get("answer")),
        "rough_total_score": (
            0.2 * (sum(file_matches) / total if total else 0.0)
            + 0.2 * (sum(page_matches) / total if total else 0.0)
            + 0.6 * answer_similarity_avg
        ),
    }
    return {"summary": summary, "worst_cases": cases[:20], "cases": cases}


def main() -> None:
    args = parse_args()
    result = evaluate(load_json(args.predictions), load_json(args.ground_truth))
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
