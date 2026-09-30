"""Turns a raw image upload into an ExtractResponse: model call + validation + clean-up + review flag."""
import io
import time

from PIL import Image

from dociq.schema import ExtractResponse, Receipt, clean_receipt, is_valid, total_check


class Service:
    def __init__(self, extractor, threshold: float = 0.85):
        self.extractor, self.threshold = extractor, threshold

    @classmethod
    def load(cls, model_id: str, revision: str | None = None, threshold: float = 0.85, quantize: bool = False):
        from dociq.models.donut import DonutExtractor

        return cls(DonutExtractor.load(model_id, revision=revision, quantize=quantize), threshold)

    def run(self, data: bytes) -> ExtractResponse:
        """Raises ValueError if data is not an image."""
        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:
            raise ValueError("not a valid image") from e
        t0 = time.perf_counter()
        p = self.extractor.predict(image)
        latency_ms = (time.perf_counter() - t0) * 1000
        ok = p.pred is not None and is_valid(p.pred)
        receipt = clean_receipt(Receipt.model_validate(p.pred)) if ok else Receipt()
        check = total_check(receipt)
        return ExtractResponse(
            receipt=receipt,
            confidence=round(p.confidence, 4),
            # Any doubt goes to a human: low confidence, unusable output, or items that don't add up.
            needs_review=(p.confidence < self.threshold) or not ok or check is False,
            total_check=check,
            latency_ms=round(latency_ms, 1),
            model_version=self.extractor.version,
        )
