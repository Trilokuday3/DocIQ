"""Gradio demo.  MODEL_ID=you/dociq-donut-cord python demo/app.py
Optional: BASELINE_ID=Qwen/Qwen2.5-VL-3B-Instruct adds a zero-shot comparison (needs a GPU Space).
Drop sample receipts into demo/samples/ to get click-to-try examples. Free CPU Spaces are slow and sleep: say so."""
import os
from pathlib import Path

import gradio as gr

from dociq.service import Service

HERE = Path(__file__).parent
service = Service.load(os.environ["MODEL_ID"], os.environ.get("MODEL_REVISION"), quantize=os.environ.get("QUANTIZE") == "1")
baseline = None
if os.environ.get("BASELINE_ID"):
    from dociq.models import load_model

    baseline = load_model("qwen", os.environ["BASELINE_ID"], prompt=os.environ.get("BASELINE_PROMPT", "v1"))


def extract(image, compare):
    import io

    buf = io.BytesIO()
    image.convert("RGB").save(buf, "JPEG")
    r = service.run(buf.getvalue())
    badge = "NEEDS REVIEW" if r.needs_review else "auto-accepted"
    other = None
    if compare and baseline is not None:
        p = baseline.predict(image)
        other = p.pred if p.pred is not None else {"error": "unparseable output", "raw": p.raw}
    return r.receipt.model_dump(), r.confidence, badge, f"{r.latency_ms:.0f} ms", other


def read(name):
    f = HERE.parent / "reports" / name
    return f.read_text(encoding="utf-8") if f.exists() else "_Run the benchmark to fill this in._"


with gr.Blocks(title="DocIQ") as demo:
    gr.Markdown("# DocIQ: receipt to JSON\nFine-tuned Donut, OCR-free. Low-confidence receipts are flagged for human review.")
    with gr.Tab("Try it"):
        with gr.Row():
            img = gr.Image(type="pil", label="Receipt")
            with gr.Column():
                out = gr.JSON(label="Extracted receipt")
                conf = gr.Number(label="Confidence (0-1)")
                badge = gr.Textbox(label="Status")
                lat = gr.Textbox(label="Latency")
        cmp_ = gr.Checkbox(label="Also run zero-shot baseline", visible=baseline is not None)
        base_out = gr.JSON(label="Baseline output", visible=baseline is not None)
        gr.Button("Extract").click(extract, [img, cmp_], [out, conf, badge, lat, base_out])
        samples = sorted(str(p) for p in (HERE / "samples").glob("*.*"))
        if samples:
            gr.Examples(samples, img)
    with gr.Tab("Benchmark"):
        gr.Markdown(read("results.md"))

if __name__ == "__main__":
    demo.launch()
