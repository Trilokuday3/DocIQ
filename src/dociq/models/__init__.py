from dataclasses import dataclass


@dataclass
class Prediction:
    pred: dict | None  # None = output could not be parsed as JSON at all
    confidence: float
    raw: str


def load_model(kind: str, model_id: str, **kw):
    """kind: 'donut' or 'qwen'. Imports are lazy so torch is only needed when you load a model."""
    if kind == "donut":
        from dociq.models.donut import DonutExtractor

        return DonutExtractor.load(model_id, **kw)
    if kind == "qwen":
        from dociq.models.baseline_vlm import QwenBaseline

        return QwenBaseline.load(model_id, **kw)
    raise ValueError(f"unknown model kind: {kind}")
