"""Pydantic contract for what DocIQ returns, plus number clean-up used by the API."""
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Base(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)


class MenuItem(_Base):
    nm: str | None = None
    cnt: str | None = None
    price: str | None = None


class SubTotal(_Base):
    subtotal_price: str | None = None
    tax_price: str | None = None


class Total(_Base):
    total_price: str | None = None
    cashprice: str | None = None
    changeprice: str | None = None


class Receipt(_Base):
    menu: list[MenuItem] = Field(default_factory=list)
    sub_total: SubTotal = Field(default_factory=SubTotal)
    total: Total = Field(default_factory=Total)

    @field_validator("menu", mode="before")
    @classmethod
    def _one_item_is_a_dict(cls, v):
        # token2json collapses a 1-item list to a dict (see explain/01), so undo that here.
        return [v] if isinstance(v, dict) else v


class ExtractResponse(BaseModel):
    receipt: Receipt
    confidence: float
    needs_review: bool
    total_check: bool | None = None  # do item prices sum to the subtotal? None = can't tell
    latency_ms: float
    model_version: str


def is_valid(pred) -> bool:
    """True if pred is a dict that fits the Receipt schema."""
    try:
        Receipt.model_validate(pred)
        return isinstance(pred, dict)
    except Exception:
        return False


_NUM = re.compile(r"^\d[\d.,\s]*$")


def normalize_number(s: str | None) -> str | None:
    """'25.000' / '25,000' / '25 000' -> '25000'. Non-numeric text is returned unchanged."""
    if s is None or not _NUM.match(s.strip()):
        return s
    return re.sub(r"[.,\s]", "", s)


def _to_int(s: str | None) -> int | None:
    s = normalize_number(s)
    return int(s) if s and s.isdigit() else None


def clean_receipt(r: Receipt) -> Receipt:
    """Normalise every *price field to digits only (pure function, returns a copy)."""
    d = r.model_dump()
    for it in d["menu"]:
        it["price"] = normalize_number(it["price"])
    for grp in ("sub_total", "total"):
        for k, v in d[grp].items():
            d[grp][k] = normalize_number(v)
    return Receipt.model_validate(d)


def total_check(r: Receipt) -> bool | None:
    """Items sum to the subtotal? None when either side is missing/unreadable."""
    prices = [_to_int(i.price) for i in r.menu]
    sub = _to_int(r.sub_total.subtotal_price)
    if sub is None or not prices or None in prices:
        return None
    return sum(prices) == sub
