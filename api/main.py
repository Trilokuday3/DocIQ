"""FastAPI service.  MODEL_ID=you/dociq-donut-cord uvicorn api.main:app --port 8000

Env: MODEL_ID (required), MODEL_REVISION (tag/commit for rollback), REVIEW_THRESHOLD (default 0.85),
     API_KEY (if set, requests must send it in the x-api-key header), QUANTIZE=1 (int8, CPU).
"""
import json
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile
from prometheus_client import Counter, Histogram, make_asgi_app

from dociq.schema import ExtractResponse
from dociq.service import Service

log = logging.getLogger("dociq")
logging.basicConfig(level=logging.INFO, format="%(message)s")
REQUESTS = Counter("dociq_requests_total", "Extract requests", ["status"])
LATENCY = Histogram("dociq_extract_seconds", "Extraction latency")


@asynccontextmanager
async def lifespan(app):
    app.state.extractor = Service.load(  # loaded once at startup, never per request
        os.environ["MODEL_ID"], os.environ.get("MODEL_REVISION"),
        float(os.environ.get("REVIEW_THRESHOLD", 0.85)), os.environ.get("QUANTIZE") == "1")
    yield


def check_key(x_api_key: str | None = Header(default=None)):
    key = os.environ.get("API_KEY")
    if key and x_api_key != key:
        raise HTTPException(401, "Bad or missing x-api-key")


app = FastAPI(title="DocIQ", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.post("/extract", response_model=ExtractResponse, dependencies=[Depends(check_key)])
async def extract(file: UploadFile):
    if file.content_type not in {"image/jpeg", "image/png"}:
        REQUESTS.labels("415").inc()
        raise HTTPException(415, "Upload a JPEG or PNG")
    data = await file.read()
    if len(data) > 5_000_000:
        REQUESTS.labels("413").inc()
        raise HTTPException(413, "Max 5 MB")
    t0 = time.perf_counter()
    try:
        resp = app.state.extractor.run(data)
    except ValueError as e:
        REQUESTS.labels("422").inc()
        raise HTTPException(422, str(e)) from e
    LATENCY.observe(time.perf_counter() - t0)
    REQUESTS.labels("200").inc()
    log.info(json.dumps({"event": "extract", "latency_ms": resp.latency_ms, "confidence": resp.confidence,
                         "needs_review": resp.needs_review, "total_check": resp.total_check,
                         "model_version": resp.model_version}))
    return resp


@app.get("/health")
def health():
    return {"status": "ok"}
