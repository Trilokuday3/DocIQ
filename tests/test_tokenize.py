import pytest

from dociq.data.cord import canon, json2token, load_gt, token2json

RECEIPT = {
    "menu": [
        {"nm": "Nasi Goreng", "cnt": "2", "price": "50,000", "sub": {"nm": "Extra Egg", "price": "5,000"}},
        {"nm": "Es Teh", "cnt": "1", "price": "8,000"},
    ],
    "sub_total": {"subtotal_price": "63,000", "tax_price": "6,300"},
    "total": {"total_price": "69,300"},
}


def test_known_sequence():
    seq = json2token({"total": {"total_price": "1"}, "menu": {"nm": "A"}}, set())
    assert seq == "<s_total><s_total_price>1</s_total_price></s_total><s_menu><s_nm>A</s_nm></s_menu>"


def test_collects_tags():
    tags = set()
    json2token(RECEIPT, tags)
    assert {"<s_menu>", "</s_menu>", "<s_sub>", "</s_sub>"} <= tags


def test_round_trip_nested_list():
    assert token2json(json2token(RECEIPT, set())) == RECEIPT


def test_single_item_list_collapses():
    gt = {"menu": [{"nm": "A"}]}
    assert token2json(json2token(gt, set())) == {"menu": {"nm": "A"}}
    assert canon(gt) == {"menu": {"nm": "A"}}


def test_truncated_output_does_not_crash():
    assert token2json("<s_menu><s_nm>Nasi") == {"menu": {"nm": "Nasi"}}


@pytest.mark.parametrize("junk", ["", "just text", "<sep/>", "</s_a>"])
def test_junk_output_is_empty_dict(junk):
    assert token2json(junk) == {}


@pytest.mark.slow
def test_matches_official_processor_on_train_labels():
    from transformers import DonutProcessor

    proc = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    for gt in list(load_gt("train"))[:200]:
        seq = json2token(gt, set())
        assert token2json(seq) == proc.token2json(seq)


@pytest.mark.slow
def test_round_trip_all_train_labels():
    pytest.importorskip("datasets")
    for gt in load_gt("train"):
        assert token2json(json2token(gt, set())) == canon(gt)
