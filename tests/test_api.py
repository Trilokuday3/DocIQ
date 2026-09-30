import io

from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from dociq.models import Prediction
from dociq.service import Service


class FakeExtractor:
    version = "fake@v0"

    def __init__(self, pred, conf=0.95):
        self.p = Prediction(pred, conf, "")

    def predict(self, image):
        return self.p


def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buf, "PNG")
    return buf.getvalue()


GOOD = {"menu": [{"nm": "A", "price": "10.000"}], "sub_total": {"subtotal_price": "10.000"},
        "total": {"total_price": "10.000"}}


def client(pred=GOOD, conf=0.95):
    app.state.extractor = Service(FakeExtractor(pred, conf), threshold=0.85)
    return TestClient(app)  # no `with`: skip the lifespan that would load a real model


def post(c, data, ctype="image/png", headers=None):
    return c.post("/extract", files={"file": ("r.png", data, ctype)}, headers=headers or {})


def test_health():
    assert client().get("/health").json() == {"status": "ok"}


def test_extract_ok_and_numbers_normalised():
    r = post(client(), png())
    assert r.status_code == 200
    j = r.json()
    assert j["receipt"]["total"]["total_price"] == "10000"
    assert j["total_check"] is True and j["needs_review"] is False
    assert j["model_version"] == "fake@v0"


def test_low_confidence_goes_to_review():
    assert post(client(conf=0.3), png()).json()["needs_review"] is True


def test_unparseable_output_goes_to_review():
    j = post(client(pred=None), png()).json()
    assert j["needs_review"] is True and j["receipt"]["menu"] == []


def test_items_not_summing_goes_to_review():
    bad = {**GOOD, "sub_total": {"subtotal_price": "99.000"}}
    j = post(client(bad), png()).json()
    assert j["total_check"] is False and j["needs_review"] is True


def test_wrong_type_oversize_and_not_an_image():
    c = client()
    assert post(c, b"x", "text/plain").status_code == 415
    assert post(c, b"x" * 5_000_001).status_code == 413
    assert post(c, b"not an image").status_code == 422


def test_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    c = client()
    assert post(c, png()).status_code == 401
    assert post(c, png(), headers={"x-api-key": "s3cret"}).status_code == 200


def test_metrics_endpoint():
    assert "dociq_requests_total" in client().get("/metrics/").text
