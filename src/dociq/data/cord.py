"""CORD v2 labels <-> the flat tag sequence Donut learns to generate."""
import json
import re


def json2token(obj, new_tokens: set, sort_keys=True):
    """Nested JSON -> '<s_k>...</s_k>' string; every tag is collected into new_tokens."""
    if isinstance(obj, dict):
        out = ""
        keys = sorted(obj.keys(), reverse=True) if sort_keys else obj.keys()
        for k in keys:
            new_tokens.update({f"<s_{k}>", f"</s_{k}>"})
            out += f"<s_{k}>" + json2token(obj[k], new_tokens, sort_keys) + f"</s_{k}>"
        return out
    if isinstance(obj, list):
        return "<sep/>".join(json2token(x, new_tokens, sort_keys) for x in obj)
    return str(obj)


_TAG = re.compile(r"<s_([^>]+)>|</s_([^>]+)>|<sep/>")


def _parse(s: str, i: int, close: str | None):
    """Parse s[i:] up to the </s_close> tag (or end of string). Returns (value, next index)."""
    items, cur, text = [], {}, ""
    while True:
        m = _TAG.search(s, i)
        if m is None:  # truncated output: keep what we have
            text += s[i:]
            i = len(s)
            break
        text += s[i:m.start()]
        i = m.end()
        if m.group(1):  # <s_key>: the value is everything up to its closing tag
            cur[m.group(1)], i = _parse(s, i, m.group(1))
        elif m.group(2):  # </s_key>: ours ends this level, a stray one is ignored
            if m.group(2) == close:
                break
        else:  # <sep/>: next list element
            items.append(cur or text.strip())
            cur, text = {}, ""
    items.append(cur or text.strip())
    return (items[0] if len(items) == 1 else items), i


def token2json(seq: str):
    """Inverse of json2token. Like Donut's processor.token2json, a 1-item list collapses to the item.
    Always returns a dict: output with no top-level tags (empty, plain text) is {}."""
    v = _parse(seq, 0, None)[0]
    return v if isinstance(v, dict) else {}


def canon(obj):
    """What a label looks like after a json2token -> token2json round trip (str leaves, 1-item lists collapsed)."""
    if isinstance(obj, dict):
        return {k: canon(v) for k, v in obj.items()}
    if isinstance(obj, list):
        items = [canon(x) for x in obj]
        return items[0] if len(items) == 1 else items
    return str(obj)


def load_gt(split: str = "train"):
    """Yield the parsed gt_parse dict for every receipt in a CORD v2 split (needs the `data` extra)."""
    from datasets import load_dataset

    ds = load_dataset("naver-clova-ix/cord-v2", split=split)
    for ex in ds:
        yield json.loads(ex["ground_truth"])["gt_parse"]
