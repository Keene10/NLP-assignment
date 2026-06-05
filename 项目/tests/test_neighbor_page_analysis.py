from rag.retriever.neighbor_page_analysis import analyze_neighbor_pages, extract_predictions


def test_neighbor_page_analysis_counts_exact_and_nearby_matches():
    predictions = [
        {"filename": "a.pdf", "page": 10, "answer": ""},
        {"filename": "a.pdf", "page": 19, "answer": ""},
        {"filename": "b.pdf", "page": 31, "answer": ""},
        {"filename": "b.pdf", "page": 40, "answer": ""},
        {"filename": "wrong.pdf", "page": 8, "answer": ""},
    ]
    ground_truth = [
        {"filename": "a.pdf", "page": 10, "question": "q0", "answer": ""},
        {"filename": "a.pdf", "page": 20, "question": "q1", "answer": ""},
        {"filename": "b.pdf", "page": 33, "question": "q2", "answer": ""},
        {"filename": "b.pdf", "page": 50, "question": "q3", "answer": ""},
        {"filename": "b.pdf", "page": 8, "question": "q4", "answer": ""},
    ]

    result = analyze_neighbor_pages(predictions, ground_truth)

    assert result["summary"]["total"] == 5
    assert result["summary"]["filename_correct"] == 4
    assert result["summary"]["exact_page_correct"] == 1
    assert result["summary"]["within_1_page"] == 2
    assert result["summary"]["within_2_pages"] == 3
    assert result["summary"]["near_miss_within_2"] == 2
    assert result["summary"]["wrong_file_count"] == 1
    assert result["by_file"]["a.pdf"]["exact_page_correct"] == 1
    assert result["by_file"]["a.pdf"]["within_1_page"] == 2
    assert result["by_file"]["b.pdf"]["within_2_pages"] == 1
    assert [case["index"] for case in result["near_miss_cases"]] == [1, 2]


def test_neighbor_page_analysis_handles_length_mismatch():
    predictions = [{"filename": "a.pdf", "page": 1}]
    ground_truth = [
        {"filename": "a.pdf", "page": 1},
        {"filename": "a.pdf", "page": 2},
    ]

    result = analyze_neighbor_pages(predictions, ground_truth)

    assert result["summary"]["total"] == 2
    assert result["summary"]["missing_prediction_count"] == 1
    assert result["cases"][1]["page_delta"] is None


def test_extract_predictions_accepts_evaluation_output_shape():
    data = {
        "summary": {"total": 1},
        "predictions": [{"filename": "a.pdf", "page": 1, "answer": ""}],
    }

    assert extract_predictions(data) == [{"filename": "a.pdf", "page": 1, "answer": ""}]


def test_extract_predictions_accepts_case_output_shape():
    data = {
        "summary": {"total": 2},
        "cases": [
            {"index": 1, "prediction": {"filename": "b.pdf", "page": 2, "answer": ""}},
            {"index": 0, "prediction": {"filename": "a.pdf", "page": 1, "answer": ""}},
        ],
    }

    assert extract_predictions(data) == [
        {"filename": "a.pdf", "page": 1, "answer": ""},
        {"filename": "b.pdf", "page": 2, "answer": ""},
    ]
