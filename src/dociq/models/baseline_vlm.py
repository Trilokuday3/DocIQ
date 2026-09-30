"""Zero-shot Qwen2.5-VL baseline: same interface as DonutExtractor, JSON via prompting."""
import json
import re

import torch

from dociq.models import Prediction

SCHEMA = """{"menu": [{"nm": str, "cnt": str, "price": str}],
 "sub_total": {"subtotal_price": str, "tax_price": str},
 "total": {"total_price": str, "cashprice": str, "changeprice": str}}"""

PROMPTS = {
    "v1": f"""Extract this receipt as JSON with exactly this shape:
{SCHEMA}
Copy numbers exactly as printed. Omit fields that are not present.
Return only JSON.""",
    "v2": f"""Extract this receipt as JSON with exactly this shape:
{SCHEMA}
Example output:
{{"menu": [{{"nm": "NASI GORENG", "cnt": "2", "price": "50.000"}}], "sub_total": {{"subtotal_price": "50.000"}}, "total": {{"total_price": "55.000"}}}}
Copy numbers exactly as printed. Omit fields that are not present.
Return only JSON.""",
    "v3": f"""You are a receipt parser. Output ONE JSON object and nothing else, using ONLY these keys:
{SCHEMA}
- nm is the item name, cnt the quantity, price the line total, all as strings copied exactly as printed.
- total_price is the final amount due. Never invent values; omit any key you cannot read.
No markdown, no explanation.""",
}


def parse_json(text: str):
    """First {...} block in the text as a dict, or None (markdown fences and chatter are tolerated)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        obj = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class QwenBaseline:
    def __init__(self, model, processor, prompt: str, max_new_tokens: int):
        self.model, self.processor, self.prompt, self.max_new_tokens = model.eval(), processor, prompt, max_new_tokens

    @classmethod
    def load(cls, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", prompt: str = "v1", max_new_tokens: int = 768):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto")
        # Cap image tokens so a T4 doesn't run out of memory on big receipts.
        processor = AutoProcessor.from_pretrained(model_id, max_pixels=1024 * 28 * 28)
        return cls(model, processor, PROMPTS[prompt], max_new_tokens)

    @torch.inference_mode()
    def predict(self, image) -> Prediction:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image.convert("RGB")], return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        raw = self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return Prediction(parse_json(raw), 0.0, raw)
