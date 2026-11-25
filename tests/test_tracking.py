import mlflow
import numpy as np

from recsys import tracking


def test_clean_makes_names_mlflow_safe():
    assert tracking.clean("recall@10") == "recall_at_10"
    assert tracking.clean("recall@10 new items") == "recall_at_10 new items"
    assert tracking.clean("a:b#c") == "a_b_c"


def test_run_logs_params_metrics_and_skips_blanks(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/m.db")
    with tracking.run("test-exp", "r1", {"test_frac": 0.3}) as r:
        tracking.log_row({"recall@10": 0.034, "empty": np.nan, "label": "x"}, prefix="ranker/")
    done = mlflow.get_run(r.info.run_id)
    assert done.data.params["test_frac"] == "0.3"
    assert done.data.metrics == {"ranker/recall_at_10": 0.034}