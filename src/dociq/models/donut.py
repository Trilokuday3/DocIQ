"""Donut: model/processor setup for training, and DonutExtractor for inference."""
import math

import torch
from transformers import DonutProcessor, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

from dociq.data.cord import token2json
from dociq.models import Prediction

BASE = "naver-clova-ix/donut-base"
TASK_START = "<s_cord-v2>"


def build(new_tokens: set, img_size=(960, 720), max_len=768, base=BASE):
    """Fresh donut-base with the CORD tags added to the vocabulary. img_size = (height, width)."""
    config = VisionEncoderDecoderConfig.from_pretrained(base)
    config.encoder.image_size = list(img_size)
    config.decoder.max_length = max_len
    processor = DonutProcessor.from_pretrained(base)
    model = VisionEncoderDecoderModel.from_pretrained(base, config=config)
    processor.image_processor.size = {"height": img_size[0], "width": img_size[1]}
    processor.image_processor.do_align_long_axis = False

    # Tags must exist in the tokenizer BEFORE the embedding matrix is resized to match.
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": sorted(new_tokens) + [TASK_START, "<sep/>"]}
    )
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(TASK_START)
    return model, processor


class DonutExtractor:
    def __init__(self, model, processor, device, version: str, max_len: int):
        self.model, self.processor, self.device = model.eval(), processor, device
        self.version, self.max_len = version, max_len

    @classmethod
    def load(cls, model_id: str, revision: str | None = None, device: str | None = None,
             quantize: bool = False, max_len: int = 768):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        processor = DonutProcessor.from_pretrained(model_id, revision=revision)
        model = VisionEncoderDecoderModel.from_pretrained(model_id, revision=revision)
        if quantize:  # int8 decoder linears: faster on CPU, measure the F1 change before shipping
            model.decoder = torch.ao.quantization.quantize_dynamic(model.decoder, {torch.nn.Linear}, dtype=torch.qint8)
        elif device == "cuda":
            model = model.half()
        return cls(model.to(device), processor, device, f"{model_id}@{revision or 'main'}", max_len)

    @torch.inference_mode()
    def predict(self, image, num_beams: int = 1) -> Prediction:
        tok = self.processor.tokenizer
        pixel_values = self.processor(image.convert("RGB"), return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device, dtype=self.model.dtype if self.device == "cuda" else None)
        start = tok(TASK_START, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        out = self.model.generate(
            pixel_values, decoder_input_ids=start, max_length=self.max_len, num_beams=num_beams,
            bad_words_ids=[[tok.unk_token_id]], return_dict_in_generate=True, output_scores=True,
        )
        seq = tok.batch_decode(out.sequences)[0]
        seq = seq.replace(tok.eos_token, "").replace(tok.pad_token, "").replace(TASK_START, "", 1)
        return Prediction(token2json(seq), self._confidence(out, num_beams), seq)

    def _confidence(self, out, num_beams: int) -> float:
        """exp(mean token log-prob): 1.0 = certain of every token, ~0 = guessing."""
        if num_beams > 1:
            return math.exp(float(out.sequences_scores[0]))
        logp = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0]
        gen = out.sequences[0, -len(logp):]
        keep = gen != self.processor.tokenizer.pad_token_id
        return math.exp(float(logp[keep].mean())) if keep.any() else 0.0
