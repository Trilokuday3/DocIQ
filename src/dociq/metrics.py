"""Scoring: field F1, tree-edit-distance accuracy, JSON validity, total exact match."""
import re
from collections import Counter

from zss import Node, simple_distance

from dociq.schema import is_valid

_NUM = re.compile(r"^\d[\d.,\s]*$")


def norm(v) -> str:
    """Strip spaces; numbers lose their . , separators so '25.000' == '25,000'."""
    s = str(v).strip()
    return re.sub(r"[.,\s]", "", s) if _NUM.match(s) else re.sub(r"\s+", " ", s)


def flatten(obj, prefix="", normalise=True):
    """Nested JSON -> list of (path, value) pairs. List positions are dropped, so item order is free."""
    if isinstance(obj, dict):
        return [p for k, v in obj.items() for p in flatten(v, f"{prefix}/{k}", normalise)]
    if isinstance(obj, list):
        return [p for x in obj for p in flatten(x, prefix, normalise)]
    return [(prefix, norm(obj) if normalise else str(obj))]


def counts(pred, gt, normalise=True):
    """(true positives, predicted count, ground-truth count) over multisets of pairs."""
    p, g = Counter(flatten(pred, normalise=normalise)), Counter(flatten(gt, normalise=normalise))
    return sum((p & g).values()), sum(p.values()), sum(g.values())


def prf(tp, n_pred, n_gt):
    if n_pred == 0 and n_gt == 0:
        return 1.0, 1.0, 1.0
    p, r = (tp / n_pred if n_pred else 0.0), (tp / n_gt if n_gt else 0.0)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def field_f1(pred, gt, normalise=True) -> float:
    return prf(*counts(pred, gt, normalise))[2]


def to_tree(obj, label="root", normalise=True) -> Node:
    n = Node(label)
    if isinstance(obj, dict):
        for k in sorted(obj):
            for x in obj[k] if isinstance(obj[k], list) else [obj[k]]:
                n.addkid(to_tree(x, k, normalise))
    else:
        n.addkid(Node(norm(obj) if normalise else str(obj)))
    return n


def _size(n: Node) -> int:
    return 1 + sum(_size(c) for c in Node.get_children(n))


def ted_acc(pred, gt, normalise=True) -> float:
    """1 - TED(pred, gt) / TED(empty, gt), floored at 0. Both trees share the root, so an empty
    prediction scores exactly 0 and a perfect one scores 1."""
    g = to_tree(gt, normalise=normalise)
    denom = _size(g) - 1
    p = to_tree(pred, normalise=normalise)
    if denom == 0:
        return 1.0 if _size(p) == 1 else 0.0
    return max(0.0, 1 - simple_distance(p, g) / denom)


def total_exact(pred, gt, normalise=True) -> bool:
    f = norm if normalise else str
    try:
        return f(pred["total"]["total_price"]) == f(gt["total"]["total_price"])
    except (KeyError, TypeError):
        return False


def score(preds: list, gts: list, normalise=True) -> dict:
    """Aggregate over a dataset. preds may contain None (unparseable output): scored as {} and invalid.
    Field F1 is micro-averaged (pool all pairs), as in the Donut paper."""
    tp = np_ = ng = 0
    ted, valid, tot = [], [], []
    for p, g in zip(preds, gts, strict=True):
        valid.append(p is not None and is_valid(p))
        p = p or {}
        a, b, c = counts(p, g, normalise)
        tp, np_, ng = tp + a, np_ + b, ng + c
        ted.append(ted_acc(p, g, normalise))
        tot.append(total_exact(p, g, normalise))
    n = len(gts)
    prec, rec, f1 = prf(tp, np_, ng)
    return {
        "n": n, "precision": prec, "recall": rec, "f1": f1,
        "ted_acc": sum(ted) / n, "json_valid": sum(valid) / n, "total_exact": sum(tot) / n,
    }


def per_field_f1(preds: list, gts: list, normalise=True) -> dict:
    """F1 per leaf path, e.g. '/menu/nm' -> 0.82. Sorted worst first: tells you what to fix."""
    tp, npred, ngt = Counter(), Counter(), Counter()
    for p, g in zip(preds, gts, strict=True):
        cp, cg = Counter(flatten(p or {}, normalise=normalise)), Counter(flatten(g, normalise=normalise))
        for (path, _), c in (cp & cg).items():
            tp[path] += c
        for (path, _), c in cp.items():
            npred[path] += c
        for (path, _), c in cg.items():
            ngt[path] += c
    out = {k: prf(tp[k], npred[k], ngt[k])[2] for k in ngt}
    return dict(sorted(out.items(), key=lambda kv: kv[1]))


def error_buckets(pred, gt, normalise=True) -> Counter:
    """missed = in gt only, hallucinated = path absent from gt, wrong_value = path exists but value differs."""
    p, g = Counter(flatten(pred or {}, normalise=normalise)), Counter(flatten(gt, normalise=normalise))
    gpaths = {k for k, _ in g}
    out = Counter()
    for (path, _val), c in (g - p).items():
        out["wrong_value" if any(pp == path for pp, _ in (p - g)) else "missed"] += c
    for (path, _val), c in (p - g).items():
        if path not in gpaths:
            out["hallucinated"] += c
    return out
