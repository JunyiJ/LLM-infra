#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import orjson
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
) -> dict:
    checkpoint_dir = checkpoints_dir / f"checkpoint-{global_step:08d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    saved_at = utc_now()
    started_at = time.perf_counter()
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
    checkpoint_size_bytes = _directory_size_bytes(checkpoint_dir)
    metadata = {
        "global_step": global_step,
        "resume_epoch": resume_epoch,
        "reason": reason,
        "saved_at": saved_at,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_size_bytes": checkpoint_size_bytes,
        "checkpoint_size_gb": round(checkpoint_size_bytes / (1024 ** 3), 4),
        "write_time_sec": round(time.perf_counter() - started_at, 4),
    }
    write_json(checkpoint_dir / "metadata.json", metadata)
    if save_total_limit is not None and save_total_limit > 0:
        checkpoints = _sorted_checkpoints(checkpoints_dir)
        for stale_checkpoint in checkpoints[:-save_total_limit]:
            shutil.rmtree(stale_checkpoint)
    return metadata


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


def _directory_size_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.cuda.current_device())
    if _mps_available():
        return "Apple MPS"
    return "cpu"


def _peak_gpu_memory_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def _reset_peak_gpu_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as handle:
        handle.write(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS))
        handle.write(b"\n")


def _write_run_report(path: Path, rows: list[dict]) -> None:
    headers = [
        "run_name",
        "gpu",
        "seq_len",
        "batch",
        "grad_accum",
        "tokens_per_sec",
        "peak_vram_gb",
        "checkpoint_gb",
        "resume_success",
        "final_val_loss",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(row.get(header, "")) for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    run_metrics_path = run_dir / "step_metrics.jsonl"
    checkpoint_metrics_path = run_dir / "checkpoint_metrics.jsonl"
    run_report_path = run_dir / "run_report.md"
    gpu_hourly_cost = cfg.get("infra", {}).get("gpu_hourly_cost_usd")
    if gpu_hourly_cost is None:
        env_cost = os.environ.get("GPU_HOURLY_COST_USD")
        gpu_hourly_cost = float(env_cost) if env_cost else None
    gpu_name = cfg.get("infra", {}).get("gpu_name") or _gpu_name()


    write_json(
        run_dir / "run_state.json",
        {
            "started_at": started_at,
            "run_name": args.run_name,
            "model": cfg["model"]["name_or_path"],
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "status": "running",
            "gpu_name": gpu_name,
            "gpu_hourly_cost_usd": gpu_hourly_cost,
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
    resume_success: bool | None = None
    if resume_checkpoint_path is not None:
        checkpoint = torch.load(resume_checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        _move_optimizer_state_to_device(optimizer, DEVICE)
        global_step = checkpoint.get("global_step", 0)
        resume_epoch = checkpoint.get("resume_epoch", checkpoint.get("epoch", 0))
        resume_success = True
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
    checkpoint_history: list[dict] = []
    best_val_loss: float | None = None
    step_records: list[dict] = []
    total_step_time_sec = 0.0
    total_step_tokens = 0
    global_peak_memory_bytes = _peak_gpu_memory_bytes(DEVICE) or 0
    if resume_checkpoint_path is not None:
        scheduler.last_epoch = global_step - 1
    for epoch in range(resume_epoch, num_train_epochs):
        rng = random.Random(train_config.get("seed", 42) + epoch)
        shuffled_indices = list(range(len(train_dataset)))
        rng.shuffle(shuffled_indices)
        pbar = tqdm(range(0, len(shuffled_indices), train_batch_size), desc=f"epoch {epoch}", leave=False)
        accumulated_microbatches = 0
        accumulated_loss = 0.0
        accumulated_tokens = 0
        accumulated_supervised_tokens = 0
        optimizer.zero_grad(set_to_none=True)
        _reset_peak_gpu_memory(DEVICE)
        _sync_device(DEVICE)
        optimizer_step_started_at = time.perf_counter()
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
            accumulated_tokens += int(response_attn_mask.sum().item())
            accumulated_supervised_tokens += int((labels != -100).sum().item())
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
                    _sync_device(DEVICE)
                    optimizer.step()
                    scheduler.step()
                    _sync_device(DEVICE)
                    step_time_sec = time.perf_counter() - optimizer_step_started_at
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    peak_memory_bytes = _peak_gpu_memory_bytes(DEVICE)
                    if peak_memory_bytes is not None:
                        global_peak_memory_bytes = max(global_peak_memory_bytes, peak_memory_bytes)
                    tokens_per_sec = accumulated_tokens / step_time_sec if step_time_sec > 0 else None
                    supervised_tokens_per_sec = (
                        accumulated_supervised_tokens / step_time_sec if step_time_sec > 0 else None
                    )
                    step_cost_usd = (
                        (gpu_hourly_cost * step_time_sec / 3600.0) if gpu_hourly_cost is not None else None
                    )
                    step_record = {
                        "epoch": epoch,
                        "optimizer_step": global_step,
                        "microbatches": accumulated_microbatches,
                        "avg_loss": accumulated_loss / accumulated_microbatches,
                        "step_time_sec": round(step_time_sec, 4),
                        "tokens": accumulated_tokens,
                        "supervised_tokens": accumulated_supervised_tokens,
                        "tokens_per_sec": round(tokens_per_sec, 2) if tokens_per_sec is not None else None,
                        "supervised_tokens_per_sec": (
                            round(supervised_tokens_per_sec, 2) if supervised_tokens_per_sec is not None else None
                        ),
                        "peak_gpu_memory_bytes": peak_memory_bytes,
                        "peak_gpu_memory_gb": (
                            round(peak_memory_bytes / (1024 ** 3), 4) if peak_memory_bytes is not None else None
                        ),
                        "cost_per_completed_optimizer_step_usd": (
                            round(step_cost_usd, 6) if step_cost_usd is not None else None
                        ),
                        "logged_at": utc_now(),
                    }
                    step_records.append(step_record)
                    _append_jsonl(run_metrics_path, step_record)
                    total_step_time_sec += step_time_sec
                    total_step_tokens += accumulated_tokens
                    if logging_steps and global_step % logging_steps == 0:
                        last_train_avg_loss = step_record["avg_loss"]
                        tqdm.write(
                            f"epoch={epoch} optimizer_step={global_step} "
                            f"microbatches={accumulated_microbatches} avg_loss={step_record['avg_loss']:.4f} "
                            f"step_time={step_record['step_time_sec']:.2f}s "
                            f"tokens_per_sec={step_record['tokens_per_sec']}"
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
                        _append_jsonl(run_dir / "eval_metrics.jsonl", eval_record)
                        if val_loss is not None and (best_val_loss is None or val_loss < best_val_loss):
                            best_val_loss = val_loss
                        tqdm.write(
                            f"eval epoch={epoch} optimizer_step={global_step} val_loss="
                            f"{val_loss:.4f}" if val_loss is not None else
                            f"eval epoch={epoch} optimizer_step={global_step} val_loss=NA"
                        )
                    if save_steps and global_step % save_steps == 0:
                        checkpoint_record = _save_checkpoint(
                            checkpoints_dir,
                            model=model,
                            optimizer=optimizer,
                            global_step=global_step,
                            resume_epoch=epoch,
                            reason="mid_epoch",
                            save_total_limit=save_total_limit,
                        )
                        checkpoint_history.append(checkpoint_record)
                        _append_jsonl(checkpoint_metrics_path, checkpoint_record)
                        last_saved_global_step = global_step
                        last_saved_resume_epoch = epoch
                    optimizer_step_started_at = time.perf_counter()
                    _reset_peak_gpu_memory(DEVICE)
                    accumulated_microbatches = 0
                    accumulated_loss = 0.0
                    accumulated_tokens = 0
                    accumulated_supervised_tokens = 0
        checkpoint_record = _save_checkpoint(
            checkpoints_dir,
            model=model,
            optimizer=optimizer,
            global_step=global_step,
            resume_epoch=epoch + 1,
            reason="epoch_end",
            save_total_limit=save_total_limit,
        )
        checkpoint_history.append(checkpoint_record)
        _append_jsonl(checkpoint_metrics_path, checkpoint_record)
        last_saved_global_step = global_step
        last_saved_resume_epoch = epoch + 1

    final_resume_epoch = num_train_epochs
    if global_step != last_saved_global_step or final_resume_epoch != last_saved_resume_epoch:
        checkpoint_record = _save_checkpoint(
            checkpoints_dir,
            model=model,
            optimizer=optimizer,
            global_step=global_step,
            resume_epoch=final_resume_epoch,
            reason="train_end",
            save_total_limit=save_total_limit,
        )
        checkpoint_history.append(checkpoint_record)
        _append_jsonl(checkpoint_metrics_path, checkpoint_record)
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

    avg_tokens_per_sec = (total_step_tokens / total_step_time_sec) if total_step_time_sec > 0 else None
    avg_step_time_sec = (total_step_time_sec / global_step) if global_step > 0 else None
    latest_checkpoint = _latest_checkpoint_path(checkpoints_dir)
    latest_checkpoint_size_bytes = _directory_size_bytes(Path(latest_checkpoint).parent) if latest_checkpoint else None
    run_table_row = {
        "run_name": args.run_name,
        "gpu": gpu_name,
        "seq_len": data_max_length,
        "batch": train_batch_size,
        "grad_accum": accumulation_steps,
        "tokens_per_sec": round(avg_tokens_per_sec, 2) if avg_tokens_per_sec is not None else "",
        "peak_vram_gb": round(global_peak_memory_bytes / (1024 ** 3), 4) if global_peak_memory_bytes else "",
        "checkpoint_gb": (
            round(latest_checkpoint_size_bytes / (1024 ** 3), 4) if latest_checkpoint_size_bytes is not None else ""
        ),
        "resume_success": resume_success if resume_success is not None else "",
        "final_val_loss": round(final_val_loss, 6) if final_val_loss is not None else "",
    }
    _write_run_report(run_report_path, [run_table_row])
    _append_jsonl(
        Path(train_cfg["output_root"]) / "run_index.jsonl",
        {
            **run_table_row,
            "model": cfg["model"]["name_or_path"],
            "gpu_hourly_cost_usd": gpu_hourly_cost,
            "avg_step_time_sec": round(avg_step_time_sec, 4) if avg_step_time_sec is not None else None,
            "latest_checkpoint": latest_checkpoint,
        },
    )
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
        "latest_checkpoint": latest_checkpoint,
        "gpu_name": gpu_name,
        "gpu_hourly_cost_usd": gpu_hourly_cost,
        "avg_step_time_sec": avg_step_time_sec,
        "avg_tokens_per_sec": avg_tokens_per_sec,
        "peak_gpu_memory_bytes": global_peak_memory_bytes or None,
        "peak_gpu_memory_gb": (
            round(global_peak_memory_bytes / (1024 ** 3), 4) if global_peak_memory_bytes else None
        ),
        "resume_success": resume_success,
        "latest_checkpoint_size_bytes": latest_checkpoint_size_bytes,
        "latest_checkpoint_size_gb": (
            round(latest_checkpoint_size_bytes / (1024 ** 3), 4)
            if latest_checkpoint_size_bytes is not None
            else None
        ),
        "cost_per_completed_optimizer_step_usd": (
            round((gpu_hourly_cost * avg_step_time_sec / 3600.0), 6)
            if gpu_hourly_cost is not None and avg_step_time_sec is not None
            else None
        ),
        "eval_history": eval_history,
        "checkpoint_history": checkpoint_history,
        "run_table": run_table_row,
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
            "avg_tokens_per_sec": avg_tokens_per_sec,
            "peak_gpu_memory_gb": summary["peak_gpu_memory_gb"],
            "summary_path": str(run_dir / "summary.json"),
        },
    )
    print(f"Training complete. Summary written to {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
