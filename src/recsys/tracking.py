import os
import re
from contextlib import contextmanager

import mlflow
import pandas as pd


def clean(name):
    """MLflow metric names allow letters, digits, _ - . space and /, so 'recall@10' becomes 'recall_at_10'."""
    return re.sub(r"[^A-Za-z0-9_\-. /]", "_", name.replace("@", "_at_"))


def tracking_uri():
    return os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


@contextmanager
def run(experiment, name, params=None):
    """Start an MLflow run, log params, and yield it. The store is local unless MLFLOW_TRACKING_URI is set."""
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=name) as r:
        mlflow.log_params({clean(k): v for k, v in (params or {}).items()})
        yield r


def log_row(row, prefix=""):
    """Log one table row (a dict or Series) as metrics, skipping blanks and non-numbers."""
    for k, v in dict(row).items():
        if isinstance(v, (int, float)) and pd.notna(v):
            mlflow.log_metric(clean(f"{prefix}{k}"), float(v))