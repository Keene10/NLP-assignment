import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def test_cli_defaults_to_updated_test_set(monkeypatch):
    from app import cli

    monkeypatch.setattr(sys, "argv", ["cli.py"])
    args = cli.parse_args()

    assert args.questions_file == "财报数据库/test_new.json"
    assert args.ground_truth == "财报数据库/test_new_ground_truth.json"
    assert args.base_page_plan == "outputs/optimized_page_plan_new.json"
    assert args.selected_page_plan == "outputs/llm_selected_page_plan_new.json"
    assert args.page_selector_debug == "outputs/llm_selected_page_plan_debug_new.json"
    assert args.debug_output == "outputs/final_debug_new.json"
    assert args.evaluation_output == "outputs/final_evaluation_new.json"


def test_hierarchical_runner_defaults_to_updated_test_set(monkeypatch):
    from scripts import run_hierarchical_rag

    monkeypatch.setattr(sys, "argv", ["run_hierarchical_rag.py"])
    args = run_hierarchical_rag.parse_args()

    assert args.questions_file == "财报数据库/test_new.json"
    assert args.ground_truth == "财报数据库/test_new_ground_truth.json"
    assert args.output == "outputs/hierarchical_rag_new.json"
    assert args.debug_output == "outputs/hierarchical_rag_debug_new.json"
