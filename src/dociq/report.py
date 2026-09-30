"""Build README tables from reports/*.json(l):  python -m dociq.report > reports/results.md"""
import json
from pathlib import Path

from dociq.metrics import error_buckets


def results_table(d="reports") -> str:
    rows = ["| Run | Field F1 | TED acc | JSON valid | Total exact | p50 ms | Cost / 1k docs |", "|---|---|---|---|---|---|---|"]
    for f in sorted(Path(d).glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("degradation"):
            continue
        n = r["norm"]
        rows.append(f"| {f.stem} | {n['f1']:.3f} | {n['ted_acc']:.3f} | {n['json_valid']:.2f} | {n['total_exact']:.2f} "
                    f"| {r['latency_p50_ms']:.0f} | ${r['cost_per_1k_usd']:.3f} |")
    return "\n".join(rows)


def robustness_table(d="reports") -> str:
    """F1 drop per degradation, relative to the clean run of the same name."""
    rows = ["| Run | Degradation | Field F1 | Drop vs clean |", "|---|---|---|---|"]
    for f in sorted(Path(d).glob("*__*.json")):
        r = json.loads(f.read_text())
        clean = Path(d) / (f.stem.split("__")[0] + ".json")
        if not clean.exists():
            continue
        c = json.loads(clean.read_text())["norm"]["f1"]
        rows.append(f"| {f.stem.split('__')[0]} | {r['degradation']} | {r['norm']['f1']:.3f} | {c - r['norm']['f1']:+.3f} |")
    return "\n".join(rows)


def confidence_table(jsonl: str, thresholds=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)) -> str:
    """For each threshold: share of receipts sent to review, and F1 of the ones that were NOT."""
    recs = [json.loads(line) for line in Path(jsonl).read_text(encoding="utf-8").splitlines()]
    rows = ["| Threshold | Sent to review | F1 of auto-accepted |", "|---|---|---|"]
    for t in thresholds:
        kept = [r["f1"] for r in recs if r["conf"] >= t]
        rows.append(f"| {t} | {1 - len(kept) / len(recs):.0%} | {sum(kept) / len(kept):.3f} |" if kept
                    else f"| {t} | 100% | - |")
    return "\n".join(rows)


def error_summary(jsonl: str) -> dict:
    tot: dict = {}
    for line in Path(jsonl).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        for k, v in error_buckets(r["pred"], r["gt"]).items():
            tot[k] = tot.get(k, 0) + v
    return tot


if __name__ == "__main__":
    print("## Results\n" + results_table())
    print("\n## Robustness\n" + robustness_table())
