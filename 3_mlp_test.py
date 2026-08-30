from pathlib import Path
import importlib.util


BASE_DIR = Path(__file__).resolve().parent
EVAL_SCRIPT = BASE_DIR / "2_mlp_eval.py"
TEST_CSV = BASE_DIR / "firewall_logs_labeled" / "3_test_logs_combined_labeled.csv"
RESULTS_DIR = BASE_DIR / "test_results"


def load_eval_module():
    spec = importlib.util.spec_from_file_location("mlp_eval", EVAL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    eval_module = load_eval_module()
    eval_module.evaluate(
        csv_path=TEST_CSV,
        stage="test",
        results_dir=RESULTS_DIR,
    )
