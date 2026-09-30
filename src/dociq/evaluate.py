"""Score any model on CORD with one harness.

  python -m dociq.evaluate --kind donut --model you/dociq-donut-cord --split test --name donut
  python -m dociq.evaluate --kind qwen  --model Qwen/Qwen2.5-VL-3B-Instruct --split validation --prompt v2 --name qwen_v2
  python -m dociq.evaluate --model $MODEL_ID --n 10 --min-f1 0.75      # CI gate (exits 1 below threshold)
  python -m dociq.evaluate ... --degrade all                           # robustness runs: <name>__blur, ...
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from datasets import load_dataset

from dociq.data.augment import DEGRADATIONS, degrade
from dociq.metrics import field_f1, score, ted_acc
from dociq.models import load_model


def run(model, split, n, degradation=None, out_jsonl=None):
    rows = load_dataset("naver-clova-ix/cord-v2", split=split)
    rows = rows.select(range(min(n, len(rows)))) if n else rows
    preds, gts, lat, records = [], [], [], []
    for i, ex in enumerate(rows):
        gt = json.loads(ex["ground_truth"])["gt_parse"]
        img = degrade(ex["image"], degradation) if degradation else ex["image"]
        t0 = time.perf_counter()
        p = model.predict(img)
        lat.append((time.perf_counter() - t0) * 1000)
        preds.append(p.pred)
        gts.append(gt)
        records.append({"i": i, "pred": p.pred, "gt": gt, "conf": p.confidence, "latency_ms": lat[-1],
                        "f1": field_f1(p.pred or {}, gt), "ted": ted_acc(p.pred or {}, gt), "raw": p.raw})
    if out_jsonl:
        Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
        Path(out_jsonl).write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    res = {"norm": score(preds, gts), "raw": score(preds, gts, normalise=False)}
    lat.sort()
    res["latency_p50_ms"] = statistics.median(lat)
    res["latency_p95_ms"] = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="donut", choices=["donut", "qwen"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--prompt", default="v1", help="qwen only: v1|v2|v3")
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--n", type=int, default=0, help="0 = whole split")
    ap.add_argument("--name", default=None, help="results go to reports/<name>.json (+ .jsonl)")
    ap.add_argument("--degrade", choices=[*DEGRADATIONS, "all"])
    ap.add_argument("--quantize", action="store_true", help="donut only: int8 decoder (CPU)")
    ap.add_argument("--gpu-hourly", type=float, default=0.35, help="USD/hour used for cost per 1k docs")
    ap.add_argument("--min-f1", type=float, help="exit 1 if field F1 is below this (CI gate)")
    ap.add_argument("--mlflow-run", help="also log to MLflow under this run name")
    a = ap.parse_args()

    kw = {"revision": a.revision, "quantize": a.quantize} if a.kind == "donut" else {"prompt": a.prompt}
    model = load_model(a.kind, a.model, **kw)
    name = a.name or a.kind
    levels = list(DEGRADATIONS) if a.degrade == "all" else [a.degrade]
    results = {}
    for d in ([None] + levels if a.degrade == "all" else levels):
        tag = name if d is None else f"{name}__{d}"
        res = run(model, a.split, a.n, d, f"reports/{tag}.jsonl")
        res["cost_per_1k_usd"] = res["latency_p50_ms"] / 1000 / 3600 * a.gpu_hourly * 1000
        res.update(model=a.model, kind=a.kind, split=a.split, degradation=d)
        Path("reports").mkdir(exist_ok=True)
        Path(f"reports/{tag}.json").write_text(json.dumps(res, indent=1))
        results[tag] = res
        print(tag, json.dumps({k: round(v, 4) for k, v in res["norm"].items()}))
        if a.mlflow_run:
            from dociq import tracking

            mlflow = tracking.init()
            with mlflow.start_run(run_name=f"{a.mlflow_run}{'' if d is None else '__' + d}"):
                mlflow.log_params({"model": a.model, "split": a.split, "n": a.n, "prompt": a.prompt})
                mlflow.log_metrics({**{f"norm_{k}": v for k, v in res["norm"].items() if k != "n"},
                                    "latency_p50_ms": res["latency_p50_ms"], "latency_p95_ms": res["latency_p95_ms"]})
                mlflow.log_artifact(f"reports/{tag}.jsonl")
    if a.min_f1 is not None and name in results and results[name]["norm"]["f1"] < a.min_f1:
        print(f"GATE FAILED: F1 {results[name]['norm']['f1']:.3f} < {a.min_f1}")
        sys.exit(1)


if __name__ == "__main__":
    main()
