from PIL import Image

from dociq.data.augment import DEGRADATIONS, degrade
from dociq.schema import Receipt, clean_receipt, is_valid, normalize_number, total_check


def test_messy_outputs_validate():
    assert is_valid({"menu": {"nm": "A"}, "total": {"total_price": 5}})  # dict menu, int price
    assert is_valid({})


def test_garbage_fails_cleanly():
    assert not is_valid(None)
    assert not is_valid({"menu": "oops"})
    assert not is_valid({"total": ["x"]})
    assert not is_valid(["a"])


def test_normalize_number():
    assert normalize_number("25.000") == "25000"
    assert normalize_number("25,000") == "25000"
    assert normalize_number("Rp") == "Rp"
    assert normalize_number(None) is None


def test_total_check():
    r = clean_receipt(Receipt.model_validate(
        {"menu": [{"price": "10.000"}, {"price": "5.000"}], "sub_total": {"subtotal_price": "15.000"}}))
    assert total_check(r) is True
    r.sub_total.subtotal_price = "16000"
    assert total_check(r) is False
    assert total_check(Receipt()) is None


def test_every_degradation_keeps_an_image():
    img = Image.new("RGB", (64, 48), "white")
    for name in DEGRADATIONS:
        assert degrade(img, name).size[0] > 0
