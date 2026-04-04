# Consolidated Run Comparison

Completed runs with both reported metrics and enough artifacts to compare side by side are summarized below.

| run_name | gpu | data.max_length | train.per_device_train_batch_size | train.gradient_accumulation_steps | effective_batch_size | train.gradient_checkpointing | major_config_changes | tokens_per_sec | peak_vram_gb | checkpoint_gb | resume_success | final_val_loss |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- | ---: |
| gpu-rtx-6000-ada-run-1 | NVIDIA RTX 6000 Ada Generation | 1024 | 2 | 16 | 32 | true | Baseline | 5857.23 | 15.8146 | 8.6266 |  | 0.775000 |
| gpu-rtx-6000-ada-run-2 | NVIDIA RTX 6000 Ada Generation | 2048 | 2 | 16 | 32 | true | `data.max_length` changed from 1024 to 2048 | 5366.93 | 20.0646 | 8.6266 |  | 0.743245 |
| gpu-rtx-6000-ada-run-3 | NVIDIA RTX 6000 Ada Generation | 2048 | 2 | 16 | 32 | false | `train.gradient_checkpointing` changed from true to false | 6701.84 | 31.9037 | 8.6266 |  | 0.743372 |
| gpu-rtx-6000-ada-run-7 | NVIDIA RTX 6000 Ada Generation | 1024 | 4 | 8 | 32 | true | `per_device_train_batch_size` changed from 2 to 4 and `gradient_accumulation_steps` changed from 16 to 8; effective batch size stayed 32 | 5205.58 | 20.0433 | 8.6266 |  | 0.774036 |
| gpu-rtx-6000-ada-run-8 | NVIDIA RTX 6000 Ada Generation | 2048 | 2 | 32 | 64 | true | `data.max_length` changed from 1024 to 2048 and `gradient_accumulation_steps` changed from 16 to 32; effective batch size increased from 32 to 64 | 5266.88 | 20.0637 | 8.6266 |  | 0.744627 |

Runs `gpu-rtx-6000-ada-run-4`, `gpu-rtx-6000-ada-run-5`, and `gpu-rtx-6000-ada-run-6` were not included because they do not yet have a completed `run_report.md`.
