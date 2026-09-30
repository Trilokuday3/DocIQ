"""Torch dataset for Donut fine-tuning (needs the `train` extra)."""
import json

import torch

from dociq.data.cord import json2token


def collect_tokens(ground_truths) -> set:
    """Every <s_key>/</s_key> tag used by the given ground_truth strings."""
    tokens: set = set()
    for s in ground_truths:
        json2token(json.loads(s)["gt_parse"], tokens)
    return tokens


class CordDataset(torch.utils.data.Dataset):
    def __init__(self, split, processor, max_len: int):
        self.ds, self.p, self.max_len = split, processor, max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[i]
        gt = json.loads(ex["ground_truth"])["gt_parse"]
        target = json2token(gt, set()) + self.p.tokenizer.eos_token
        pixel_values = self.p(ex["image"].convert("RGB"), return_tensors="pt").pixel_values[0]
        ids = self.p.tokenizer(
            target, add_special_tokens=False, max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt",
        ).input_ids[0]
        labels = ids.clone()
        labels[labels == self.p.tokenizer.pad_token_id] = -100  # padding must not count in the loss
        return {"pixel_values": pixel_values, "labels": labels}
