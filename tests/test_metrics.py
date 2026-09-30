import pytest

from dociq.metrics import error_buckets, field_f1, per_field_f1, score, ted_acc, total_exact

GT = {"menu": [{"nm": "A", "price": "10.000"}, {"nm": "B", "price": "5.000"}], "total": {"total_price": "15.000"}}


def test_perfect_match():
    assert field_f1(GT, GT) == 1.0
    assert ted_acc(GT, GT) == 1.0


def test_empty_prediction_is_zero():
    assert field_f1({}, GT) == 0.0
    assert ted_acc({}, GT) == 0.0


def test_one_wrong_field_is_partial():
    bad = {**GT, "total": {"total_price": "99"}}
    assert 0.5 < field_f1(bad, GT) < 1.0
    assert 0.5 < ted_acc(bad, GT) < 1.0


def test_item_order_and_number_format_do_not_matter():
    shuffled = {"menu": list(reversed(GT["menu"])), "total": {"total_price": "15,000"}}
    assert field_f1(shuffled, GT) == 1.0
    assert field_f1(shuffled, GT, normalise=False) < 1.0  # raw score is stricter


def test_total_exact():
    assert total_exact(GT, GT)
    assert not total_exact({}, GT)


def test_score_counts_none_as_invalid_and_empty():
    s = score([GT, None], [GT, GT])
    assert s["json_valid"] == 0.5 and s["total_exact"] == 0.5
    assert s["recall"] == 0.5 and s["precision"] == 1.0


def test_per_field_worst_first():
    bad = {"menu": [{"nm": "A", "price": "1"}, {"nm": "B", "price": "1"}], "total": {"total_price": "15.000"}}
    worst = per_field_f1([bad], [GT])
    assert next(iter(worst)) == "/menu/price"


def test_error_buckets():
    pred = {"menu": [{"nm": "A"}], "total": {"total_price": "1"}, "extra": "x"}
    b = error_buckets(pred, GT)
    assert b["hallucinated"] == 1 and b["wrong_value"] == 1 and b["missed"] >= 1


@pytest.mark.parametrize("g", [{}, {"total": {"total_price": "1"}}])
def test_ted_edge_cases(g):
    assert ted_acc(g, g) == 1.0
