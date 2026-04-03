#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import torch
import random
from pathlib import Path
from tqdm import tqdm

from llm_infra_lab.manifest import load_yaml, utc_now, write_json
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def print_next_steps(cfg: dict, run_dir: Path) -> None:
    print("Training scaffold created.")
    print(f"Run directory: {run_dir}")
    print(f"Base model: {cfg['model']['name_or_path']}")
    print("Implement these pieces next:")
    print("1. Load tokenizer and model.")
    print("2. Load jsonl train/validation splits.")
    print("3. Build a completion-only data collator.")
    print("4. Create TrainingArguments or a custom loop.")
    print("5. Add checkpoint resume and end-of-run summary logic.")

def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE= get_default_device()


def get_tokens_masks_approx(prompts, texts, tokenizer, device, max_length=None):
    def _encode(batch_texts):
        kwargs = {"return_tensors": "pt", "padding": True, "truncation": True}
        if max_length is not None:
            kwargs["max_length"] = max_length
        return tokenizer(batch_texts, **kwargs)

    prompt_enc = _encode(prompts)
    prompt_attn_mask = prompt_enc["attention_mask"].to(device)
    prompt_len = prompt_attn_mask.sum(dim=1)

    text_enc = _encode(texts)
    input_ids = text_enc["input_ids"].to(device)
    attention_mask = text_enc["attention_mask"].to(device)
    text_lens = attention_mask.sum(dim=1)

    seq_len = input_ids.size(1)
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    eos_positions = (
        (input_ids == tokenizer.eos_token_id)
        & (positions >= prompt_len.unsqueeze(1))
        & (positions < text_lens.unsqueeze(1))
    )
    has_eos = eos_positions.any(dim=1)
    first_eos = torch.where(
        has_eos,
        eos_positions.float().argmax(dim=1),
        text_lens - 1,
    )
    shifted_positions = torch.arange(seq_len - 1, device=device).unsqueeze(0)
    start = (prompt_len - 1).clamp(min=0).unsqueeze(1)
    end = first_eos.unsqueeze(1)
    answer_mask = ((shifted_positions >= start) & (shifted_positions < end)).float()
    return input_ids, attention_mask, answer_mask


def get_tokens_masks_labels(prompts, completions, tokenizer, device, max_length=None):
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    prompt_encodings = tokenizer(prompts, add_special_tokens=False)
    completion_encodings = tokenizer(completions, add_special_tokens=False)

    input_id_rows: list[list[int]] = []
    attention_mask_rows: list[list[int]] = []
    label_rows: list[list[int]] = []

    for prompt_ids, completion_ids in zip(
        prompt_encodings["input_ids"],
        completion_encodings["input_ids"],
    ):
        input_ids = prompt_ids + completion_ids
        if max_length is not None:
            input_ids = input_ids[:max_length]

        effective_prompt_len = min(len(prompt_ids), len(input_ids))

        input_id_rows.append(input_ids)
        attention_mask_rows.append([1] * len(input_ids))
        label_rows.append([-100] * effective_prompt_len + completion_ids[:len(input_ids) - effective_prompt_len])

    max_seq_len = max(len(row) for row in input_id_rows)

    padded_input_ids = []
    padded_attention_masks = []
    padded_labels = []
    for input_ids, attention_mask, labels in zip(
        input_id_rows,
        attention_mask_rows,
        label_rows,
    ):
        pad_len = max_seq_len - len(input_ids)
        padded_input_ids.append(input_ids + [pad_token_id] * pad_len)
        padded_attention_masks.append(attention_mask + [0] * pad_len)
        padded_labels.append(labels + [-100] * pad_len)

    return (
        torch.tensor(padded_input_ids, device=device),
        torch.tensor(padded_attention_masks, device=device),
        torch.tensor(padded_labels, device=device),
    )

def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    config_dir = Path(args.config).resolve().parent
    train_cfg = cfg["train"]
    run_dir = Path(train_cfg["output_root"]) / args.run_name
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)


    write_json(
        run_dir / "run_state.json",
        {
            "started_at": utc_now(),
            "run_name": args.run_name,
            "model": cfg["model"]["name_or_path"],
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "status": "scaffold_only",
            "next_steps": [
                "Implement tokenizer and model loading",
                "Implement dataset loading from jsonl",
                "Implement completion-only collation",
                "Implement Trainer or custom loop",
                "Implement checkpoint save and resume behavior",
            ],
        },
    )
    print_next_steps(cfg, run_dir)
    model_cfg = cfg["model"]
    dtype_name = model_cfg.get("torch_dtype", "auto")
    torch_dtype = "auto" if dtype_name == "auto" else getattr(torch, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        torch_dtype=torch_dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    model.to(DEVICE)
    inputs = tokenizer("Write a Python function for fibonacci.", return_tensors="pt")
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
    )
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(text)

    data_cfg = cfg["data"]
    train_path = (config_dir.resolve().parent  / data_cfg["train_path"]).resolve()
    val_path = (config_dir.resolve().parent  / data_cfg["val_path"]).resolve()
    train_dataset = [json.loads(line) for line in train_path.open("r", encoding="utf-8")]
    val_dataset = [json.loads(line) for line in val_path.open("r", encoding="utf-8")]
    data_max_length = data_cfg.get("max_length", 2048)

    # TODO: Create TrainingArguments or a custom loop for full-weight training.
    train_config = cfg["train"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.get("learning_rate", 1e-5))
    for epoch in range(train_config.get("num_train_epochs", 1)):
        rng = random.Random(train_config.get("seed", 42) + epoch)
        rng.shuffle(train_dataset)
        batch_size = train_config.get("per_device_train_batch_size", 1)
        pbar = tqdm(range(0, len(train_dataset), batch_size), desc=f"epoch {epoch}", leave=False)
        for step_idx, idx in enumerate(pbar):
            batch = train_dataset[idx : idx + batch_size]
            prompts = [record['prompt'] for record in batch]
            completions = [record['completion'] for record in batch]
            response_ids, response_attn_mask, labels = get_tokens_masks_labels(
                prompts,
                completions,
                tokenizer,
                DEVICE,
                data_max_length
            )
            model.train()
            with torch.enable_grad():
                outputs = model(
                    input_ids=response_ids,
                    attention_mask=response_attn_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)



    # TODO: Implement checkpoint save cadence and resume-from-checkpoint behavior.
    # TODO: Add eval hooks and a compact end-of-run summary.


if __name__ == "__main__":
    main()
