"""MLflow setup shared by train and evaluate. New MLflow rejects the ./mlruns file store, so default to SQLite."""
import os

import mlflow


def init(experiment: str = "dociq"):
    os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")  # override for DagsHub / a server
    mlflow.set_experiment(experiment)
    return mlflow
