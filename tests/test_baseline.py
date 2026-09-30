import pytest

pytest.importorskip("torch")
from dociq.models.baseline_vlm import PROMPTS, parse_json  # noqa: E402


def test_parse_json_tolerates_fences_and_chatter():
    assert parse_json('Sure!\n```json\n{"total": {"total_price": "5"}}\n```') == {"total": {"total_price": "5"}}


def test_parse_json_failure_is_none():
    assert parse_json("no json here") is None
    assert parse_json('{"a": ') is None
    assert parse_json("[1, 2]") is None


def test_prompts_present():
    assert set(PROMPTS) == {"v1", "v2", "v3"}
