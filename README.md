# DocIQ: receipt photo to validated JSON

Fine-tunes **Donut** (OCR-free vision-to-sequence transformer) on CORD v2, benchmarks it against a zero-shot
Qwen2.5-VL-3B, and serves it behind a FastAPI endpoint that flags uncertain receipts for human review.

> Repo: https://github.com/Trilokuday3/DocIQ · Demo: _link after deploy_ · API docs: _link after deploy_ · Model: _HF Hub link after training_

## Results (fill in from `python -m dociq.report`)

| Model | Field F1 | TED acc | JSON valid | Total exact | p50 latency (T4) | Cost / 1k docs |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-3B zero-shot | | | | | | |
| Donut fine-tuned (this repo) | | | | | | |
| Donut official CORD checkpoint (reference) | | | | | | |

Robustness (F1 drop under blur / rotation / JPEG q20 / 50% brightness): _table from `reports/`_

## Reproduce

```bash
pip install -e .[train,dev]
python -m pytest -q                                            # unit tests
python -m dociq.train --config configs/train_donut.yaml        # train (Kaggle/Colab GPU)
python -m dociq.evaluate --kind qwen  --model Qwen/Qwen2.5-VL-3B-Instruct --split validation --prompt v1 --name qwen_v1
python -m dociq.evaluate --kind donut --model outputs/donut/best --split test --name donut --degrade all
python -m dociq.report > reports/results.md
```

Kaggle: import `notebooks/02_kaggle_run.ipynb` (runs baseline, training, evaluation). Model card template: `MODEL_CARD.md`.

Serve: `pip install -e .[infer,serve] && MODEL_ID=<hub id> uvicorn api.main:app` then open `/docs`.
Docker: `docker build -f api/Dockerfile -t dociq .`

## Layout
`src/dociq/` data (json<->tokens, augment), models (Donut, Qwen), metrics, schema, train, evaluate, report, service ·
`api/` FastAPI · `demo/` Gradio · `configs/` YAML per run · `tests/` · `notebooks/` EDA

## Limitations
1,000 Indonesian receipts only; fixed input resolution; CPU latency is seconds per receipt (int8 quantization helps).
