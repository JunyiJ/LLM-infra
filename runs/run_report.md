# Consolidated Run Comparison

Completed runs with both reported metrics and enough artifacts to compare side by side are summarized below.

Shared across all runs in this report: GPU is `NVIDIA RTX 6000 Ada Generation`, checkpoint size is about `8.6266 GB`, and `resume_success` is empty in the per-run reports.

## `data.max_length = 1024`

| run_name | train config | major_config_changes | tokens_per_sec | peak_vram_gb | final_val_loss |
| --- | --- | --- | ---: | ---: | ---: |
| gpu-rtx-6000-ada-run-1 | batch `2` x accum `16` = eff `32`, gc `true` | Baseline | 5857.23 | 15.8146 | 0.775000 |
| gpu-rtx-6000-ada-run-7 | batch `4` x accum `8` = eff `32`, gc `true` | `per_device_train_batch_size` changed from 2 to 4 and `gradient_accumulation_steps` changed from 16 to 8; effective batch size stayed 32 | 5205.58 | 20.0433 | 0.774036 |

## `data.max_length = 2048`

| run_name | train config | major_config_changes | tokens_per_sec | peak_vram_gb | final_val_loss |
| --- | --- | --- | ---: | ---: | ---: |
| gpu-rtx-6000-ada-run-2 | batch `2` x accum `16` = eff `32`, gc `true` | `data.max_length` changed from 1024 to 2048 | 5366.93 | 20.0646 | 0.743245 |
| gpu-rtx-6000-ada-run-3 | batch `2` x accum `16` = eff `32`, gc `false` | `train.gradient_checkpointing` changed from true to false | 6701.84 | 31.9037 | 0.743372 |
| gpu-rtx-6000-ada-run-8 | batch `2` x accum `32` = eff `64`, gc `true` | `data.max_length` changed from 1024 to 2048 and `gradient_accumulation_steps` changed from 16 to 32; effective batch size increased from 32 to 64 | 5266.88 | 20.0637 | 0.744627 |

Runs `gpu-rtx-6000-ada-run-4`, `gpu-rtx-6000-ada-run-5`, and `gpu-rtx-6000-ada-run-6` were not included because they do not yet have a completed `run_report.md`.
