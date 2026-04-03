#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

from llm_infra_lab.manifest import load_yaml, utc_now, write_json
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE= get_default_device()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def _sorted_checkpoints(checkpoints_dir: Path) -> list[Path]:
    return sorted(
        (path for path in checkpoints_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )


def _resolve_resume_checkpoint_path(resume_from_checkpoint: str | None, checkpoints_dir: Path) -> Path | None:
    if not resume_from_checkpoint:
        return None
    if resume_from_checkpoint == "latest":
        checkpoints = _sorted_checkpoints(checkpoints_dir)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
        return checkpoints[-1] / "training_state.pt"

    checkpoint_path = Path(resume_from_checkpoint).expanduser()
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "training_state.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _save_checkpoint(
    checkpoints_dir: Path,
    *,
    model: AutoModelForCausalLM,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    resume_epoch: int,
    reason: str,
    save_total_limit: int | None,
) -> None:
    checkpoint_dir = checkpoints_dir / f"checkpoint-{global_step:08d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    saved_at = utc_now()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "resume_epoch": resume_epoch,
            "reason": reason,
            "saved_at": saved_at,
        },
        checkpoint_dir / "training_state.pt",
    )
    write_json(
        checkpoint_dir / "metadata.json",
        {
            "global_step": global_step,
            "resume_epoch": resume_epoch,
            "reason": reason,
            "saved_at": saved_at,
        },
    )
    if save_total_limit is not None and save_total_limit > 0:
        checkpoints = _sorted_checkpoints(checkpoints_dir)
        for stale_checkpoint in checkpoints[:-save_total_limit]:
            shutil.rmtree(stale_checkpoint)


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _latest_checkpoint_path(checkpoints_dir: Path) -> str | None:
    checkpoints = _sorted_checkpoints(checkpoints_dir)
    if not checkpoints:
        return None
    return str(checkpoints[-1] / "training_state.pt")


def _num_optimizer_steps_per_epoch(num_examples: int, batch_size: int, accumulation_steps: int) -> int:
    if num_examples == 0:
        return 0
    microbatches = math.ceil(num_examples / batch_size)
    return math.ceil(microbatches / accumulation_steps)


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


def evaluate_loss(
    *,
    model: AutoModelForCausalLM,
    dataset: list[dict],
    tokenizer: AutoTokenizer,
    batch_size: int,
    device: torch.device,
    max_length: int | None,
) -> float | None:
    if not dataset:
        return None

    model.eval()
    total_loss = 0.0
    total_supervised_tokens = 0
    with torch.no_grad():
        for start_idx in range(0, len(dataset), batch_size):
            batch = dataset[start_idx : start_idx + batch_size]
            prompts = [record["prompt"] for record in batch]
            completions = [record["completion"] for record in batch]
            input_ids, attention_mask, labels = get_tokens_masks_labels(
                prompts,
                completions,
                tokenizer,
                device,
                max_length,
            )
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            supervised_tokens = int((labels != -100).sum().item())
            if supervised_tokens == 0:
                continue
            total_loss += float(outputs.loss.item()) * supervised_tokens
            total_supervised_tokens += supervised_tokens
    model.train()
    if total_supervised_tokens == 0:
        return None
    return total_loss / total_supervised_tokens

def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    config_dir = Path(args.config).resolve().parent
    train_cfg = cfg["train"]
    run_dir = Path(train_cfg["output_root"]) / args.run_name
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()


    write_json(
        run_dir / "run_state.json",
        {
            "started_at": started_at,
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

    data_cfg = cfg["data"]
    train_path = (config_dir.resolve().parent  / data_cfg["train_path"]).resolve()
    val_path = (config_dir.resolve().parent  / data_cfg["val_path"]).resolve()
    train_dataset = _load_jsonl(train_path)
    val_dataset = _load_jsonl(val_path)
    data_max_length = data_cfg.get("max_length", 2048)

    train_config = cfg["train"]
    if train_config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.get("learning_rate", 1e-5),
        weight_decay=train_config.get("weight_decay", 0.01),
    )
    resume_checkpoint_path = _resolve_resume_checkpoint_path(args.resume_from_checkpoint, checkpoints_dir)
    resume_epoch = 0
    global_step = 0
    if resume_checkpoint_path is not None:
        checkpoint = torch.load(resume_checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        _move_optimizer_state_to_device(optimizer, DEVICE)
        global_step = checkpoint.get("global_step", 0)
        resume_epoch = checkpoint.get("resume_epoch", checkpoint.get("epoch", 0))
        print(
            f"Resumed from checkpoint {resume_checkpoint_path} "
            f"at optimizer step {global_step}, starting from epoch {resume_epoch}."
        )

    num_train_epochs = train_config.get("num_train_epochs", 1)
    train_batch_size = train_config.get("per_device_train_batch_size", 1)
    eval_batch_size = train_config.get("per_device_eval_batch_size", train_batch_size)
    accumulation_steps = train_config.get("gradient_accumulation_steps", 1)
    num_update_steps_per_epoch = _num_optimizer_steps_per_epoch(
        len(train_dataset),
        train_batch_size,
        accumulation_steps,
    )
    num_training_steps = num_update_steps_per_epoch * num_train_epochs
    num_warmup_steps = int(num_training_steps * train_config.get("warmup_ratio", 0.0))
    save_steps = max(int(train_config.get("save_steps", 0) or 0), 0)
    eval_steps = max(int(train_config.get("eval_steps", 0) or 0), 0)
    logging_steps = max(int(train_config.get("logging_steps", 0) or 0), 0)
    save_total_limit = train_config.get("save_total_limit")
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max(num_training_steps, 1),
    )
    model.train()
    last_saved_global_step = global_step if resume_checkpoint_path is not None else None
    last_saved_resume_epoch = resume_epoch if resume_checkpoint_path is not None else None
    last_train_avg_loss: float | None = None
    eval_history: list[dict] = []
    best_val_loss: float | None = None
    if resume_checkpoint_path is not None:
        scheduler.last_epoch = global_step - 1
    for epoch in range(resume_epoch, num_train_epochs):
        rng = random.Random(train_config.get("seed", 42) + epoch)
        shuffled_indices = list(range(len(train_dataset)))
        rng.shuffle(shuffled_indices)
        pbar = tqdm(range(0, len(shuffled_indices), train_batch_size), desc=f"epoch {epoch}", leave=False)
        accumulated_microbatches = 0
        accumulated_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for start_idx in pbar:
            batch_indices = shuffled_indices[start_idx : start_idx + train_batch_size]
            batch = [train_dataset[index] for index in batch_indices]
            prompts = [record["prompt"] for record in batch]
            completions = [record["completion"] for record in batch]
            response_ids, response_attn_mask, labels = get_tokens_masks_labels(
                prompts,
                completions,
                tokenizer,
                DEVICE,
                data_max_length
            )
            with torch.enable_grad():
                outputs = model(
                    input_ids=response_ids,
                    attention_mask=response_attn_mask,
                    labels=labels
                )
                raw_loss = outputs.loss
                loss = raw_loss / accumulation_steps
                loss.backward()
                accumulated_microbatches += 1
                accumulated_loss += raw_loss.item()
                should_step = (
                    accumulated_microbatches == accumulation_steps
                    or start_idx + train_batch_size >= len(shuffled_indices)
                )
                if should_step:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if logging_steps and global_step % logging_steps == 0:
                        avg_loss = accumulated_loss / accumulated_microbatches
                        last_train_avg_loss = avg_loss
                        tqdm.write(
                            f"epoch={epoch} optimizer_step={global_step} "
                            f"microbatches={accumulated_microbatches} avg_loss={avg_loss:.4f}"
                        )
                    if eval_steps and global_step % eval_steps == 0:
                        val_loss = evaluate_loss(
                            model=model,
                            dataset=val_dataset,
                            tokenizer=tokenizer,
                            batch_size=eval_batch_size,
                            device=DEVICE,
                            max_length=data_max_length,
                        )
                        eval_record = {
                            "epoch": epoch,
                            "global_step": global_step,
                            "val_loss": val_loss,
                            "evaluated_at": utc_now(),
                        }
                        eval_history.append(eval_record)
                        if val_loss is not None and (best_val_loss is None or val_loss < best_val_loss):
                            best_val_loss = val_loss
                        tqdm.write(
                            f"eval epoch={epoch} optimizer_step={global_step} val_loss="
                            f"{val_loss:.4f}" if val_loss is not None else
                            f"eval epoch={epoch} optimizer_step={global_step} val_loss=NA"
                        )
                    if save_steps and global_step % save_steps == 0:
                        _save_checkpoint(
                            checkpoints_dir,
                            model=model,
                            optimizer=optimizer,
                            global_step=global_step,
                            resume_epoch=epoch,
                            reason="mid_epoch",
                            save_total_limit=save_total_limit,
                        )
                        last_saved_global_step = global_step
                        last_saved_resume_epoch = epoch
                    accumulated_microbatches = 0
                    accumulated_loss = 0.0
        _save_checkpoint(
            checkpoints_dir,
            model=model,
            optimizer=optimizer,
            global_step=global_step,
            resume_epoch=epoch + 1,
            reason="epoch_end",
            save_total_limit=save_total_limit,
        )
        last_saved_global_step = global_step
        last_saved_resume_epoch = epoch + 1

    final_resume_epoch = num_train_epochs
    if global_step != last_saved_global_step or final_resume_epoch != last_saved_resume_epoch:
        _save_checkpoint(
            checkpoints_dir,
            model=model,
            optimizer=optimizer,
            global_step=global_step,
            resume_epoch=final_resume_epoch,
            reason="train_end",
            save_total_limit=save_total_limit,
        )
    final_val_loss = evaluate_loss(
        model=model,
        dataset=val_dataset,
        tokenizer=tokenizer,
        batch_size=eval_batch_size,
        device=DEVICE,
        max_length=data_max_length,
    )
    if final_val_loss is not None:
        eval_history.append(
            {
                "epoch": final_resume_epoch,
                "global_step": global_step,
                "val_loss": final_val_loss,
                "evaluated_at": utc_now(),
                "reason": "train_end",
            }
        )
        if best_val_loss is None or final_val_loss < best_val_loss:
            best_val_loss = final_val_loss

    summary = {
        "run_name": args.run_name,
        "status": "completed",
        "started_at": started_at,
        "finished_at": utc_now(),
        "model": cfg["model"]["name_or_path"],
        "train_rows": len(train_dataset),
        "val_rows": len(val_dataset),
        "num_train_epochs": num_train_epochs,
        "optimizer_steps": global_step,
        "last_train_avg_loss": last_train_avg_loss,
        "final_val_loss": final_val_loss,
        "best_val_loss": best_val_loss,
        "latest_checkpoint": _latest_checkpoint_path(checkpoints_dir),
        "eval_history": eval_history,
    }
    write_json(run_dir / "summary.json", summary)
    write_json(
        run_dir / "run_state.json",
        {
            "started_at": started_at,
            "finished_at": summary["finished_at"],
            "run_name": args.run_name,
            "model": cfg["model"]["name_or_path"],
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "status": "completed",
            "optimizer_steps": global_step,
            "final_val_loss": final_val_loss,
            "summary_path": str(run_dir / "summary.json"),
        },
    )
    print(f"Training complete. Summary written to {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
