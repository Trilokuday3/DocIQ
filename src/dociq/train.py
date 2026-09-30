"""Fine-tune Donut on CORD v2:  python -m dociq.train --config configs/train_donut.yaml [--resume]"""
import argparse
import json
import math
import subprocess

import numpy as np
import yaml
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from dociq.data.cord import token2json
from dociq.data.dataset import CordDataset, collect_tokens
from dociq.metrics import score
from dociq.models.donut import TASK_START, build


def make_compute_metrics(processor):
    tok = processor.tokenizer
    calls = {"n": 0}

    def to_json(ids):
        ids = np.where(ids == -100, tok.pad_token_id, ids)
        s = tok.decode(ids).replace(tok.eos_token, "").replace(tok.pad_token, "")
        return token2json(s.replace(TASK_START, "", 1))

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        preds = preds[0] if isinstance(preds, tuple) else preds
        p = [to_json(x) for x in preds]
        g = [to_json(x) for x in eval_pred.label_ids]
        calls["n"] += 1
        try:  # 3 predicted-vs-true examples per evaluation, as MLflow text artifacts
            import mlflow

            mlflow.log_text(json.dumps([{"pred": a, "true": b} for a, b in zip(p[:3], g[:3], strict=False)], indent=1),
                            f"examples/eval_{calls['n']}.json")
        except Exception:
            pass
        return score(p, g)

    return compute_metrics


def main(cfg_path: str, resume: bool):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    ds = load_dataset("naver-clova-ix/cord-v2")
    # Vocabulary always comes from the FULL train split so a data-fraction ablation keeps the same tags.
    new_tokens = collect_tokens(ds["train"]["ground_truth"])
    model, processor = build(new_tokens, tuple(cfg["img_size"]), cfg["max_len"], cfg["base"])

    train = ds["train"]
    if cfg["train_fraction"] < 1:
        train = train.shuffle(seed=cfg["seed"]).select(range(int(len(train) * cfg["train_fraction"])))
    val = ds["validation"].select(range(cfg["val_n"]))

    steps_per_epoch = math.ceil(len(train) / (cfg["batch_size"] * cfg["grad_accum"]))
    args = Seq2SeqTrainingArguments(
        output_dir=cfg["output_dir"], seed=cfg["seed"], run_name=cfg["run_name"],
        per_device_train_batch_size=cfg["batch_size"], per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"], learning_rate=cfg["lr"],
        num_train_epochs=cfg["epochs"], warmup_steps=cfg["warmup_steps"], lr_scheduler_type="linear",
        max_grad_norm=1.0, fp16=cfg["fp16"], bf16=cfg.get("bf16", False), gradient_checkpointing=True,
        eval_strategy="steps", eval_steps=steps_per_epoch, save_strategy="steps", save_steps=steps_per_epoch,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="ted_acc",
        predict_with_generate=True, generation_max_length=cfg["max_len"], generation_num_beams=cfg["num_beams"],
        logging_steps=10, report_to=["mlflow"], remove_unused_columns=False, dataloader_num_workers=2,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=CordDataset(train, processor, cfg["max_len"]),
        eval_dataset=CordDataset(val, processor, cfg["max_len"]),
        compute_metrics=make_compute_metrics(processor),
        callbacks=[_EarlyStop(cfg["patience"])],
    )

    from dociq import tracking

    mlflow = tracking.init()
    with mlflow.start_run(run_name=cfg["run_name"]):
        mlflow.log_params({k: v for k, v in cfg.items()})
        mlflow.log_artifact(cfg_path)
        try:
            mlflow.set_tag("git_commit", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
        except Exception:
            pass
        trainer.train(resume_from_checkpoint=resume or None)
        best = f"{cfg['output_dir']}/best"
        trainer.save_model(best)
        processor.save_pretrained(best)
        if cfg.get("hub_model_id"):  # needs HF_TOKEN in the environment
            model.push_to_hub(cfg["hub_model_id"])
            processor.push_to_hub(cfg["hub_model_id"])
        print("saved", best)


def _EarlyStop(patience: int):
    from transformers import EarlyStoppingCallback

    return EarlyStoppingCallback(early_stopping_patience=patience)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    main(a.config, a.resume)
