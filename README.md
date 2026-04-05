# LLM Infra Lab

This repo is scoped for a `full-weight coding post-training` project that is deliberately more infra-heavy than algorithm-heavy.

The default stack is:

- base model: `Qwen/Qwen2.5-1.5B-Instruct`
- train dataset: `codeparrot/apps`
- eval set: `EvalPlus` (`HumanEval+` and `MBPP+`)
- serving: `vLLM` OpenAI-compatible server
- execution target: local control plane on Mac, training and serving on Runpod

## Why this project

The point is not to implement another PPO or GRPO variant. The point is to operate a small training system end to end:

- build a reproducible training dataset
- run full-weight post-training with resumable checkpoints
- record manifest metadata for each run
- serve a checkpoint behind a real inference endpoint
- benchmark promoted checkpoints on a fixed eval stack

## Repo layout

- `configs/experiment.yaml`: single source of truth for model, data, and run metadata
- `scripts/prepare_apps.py`: APPS preparation scaffold with TODOs for filtering and prompt construction
- `scripts/train_full_sft.py`: training scaffold with TODOs for collator, trainer, and resume policy
- `scripts/make_manifest.py`: writes a run manifest that ties data, model, and output paths together
- `scripts/serve_vllm.sh`: serves a checkpoint with vLLM
- `scripts/run_evalplus.sh`: runs EvalPlus against an OpenAI-compatible endpoint
- `scripts/smoke_test_vllm.py`: sends one smoke-test request to a vLLM endpoint
- `scripts/benchmark_vllm.py`: collects serving latency and tokens/sec from a vLLM endpoint
- `scripts/sample_gpu_memory.sh`: samples GPU memory and utilization with `nvidia-smi`
- `scripts/dump_humaneval_samples.py`: dumps HumanEval prompts and served-model completions for side-by-side inspection
- `src/llm_infra_lab/`: prompt formatting and manifest helpers

## Recommended workflow

1. Create a conda environment and install this repo.
2. Prepare a small APPS subset.
3. Generate a run manifest.
4. Train on Runpod with full weights.
5. Serve the resulting checkpoint with vLLM.
6. Run EvalPlus and write a short benchmark note.

## Quickstart

```bash
conda create -n llm-infra python=3.10 -y
conda activate llm-infra
pip install -e .
```

For local development on your Mac, this is enough for data prep, manifests, and training scaffolding.

## Quick RunPod setup

Attach a network volume and keep the repo, virtualenv, caches, models, and checkpoints under `/workspace`. For this project, `50 GB` is a practical minimum if everything lives there.

Use GitHub over HTTPS instead of SSH:

```bash
cd /workspace
git clone https://github.com/JunyiJ/LLM-infra.git
cd /workspace/LLM-infra
git remote set-url origin https://github.com/JunyiJ/LLM-infra.git
git config --global credential.helper "store --file=/workspace/.git-credentials"
```

When you push, Git will prompt for your GitHub username and a Personal Access Token instead of an SSH key.

Set up the Python environment under `/workspace`:

```bash
python3 -m venv /workspace/LLM-infra/.venv
source /workspace/LLM-infra/.venv/bin/activate
python -m pip install --upgrade pip
pip install --no-cache-dir -e .
pip install --no-cache-dir vllm evalplus
```

This repo uses `pyproject.toml`, so add or update Python requirements there and rerun `pip install --no-cache-dir -e .` after changes.

If you want to keep temp files and package caches on the network volume too:

```bash
mkdir -p /workspace/tmp /workspace/pip-cache /workspace/hf-home /workspace/.cache
export TMPDIR=/workspace/tmp
export PIP_CACHE_DIR=/workspace/pip-cache
export HF_HOME=/workspace/hf-home
export XDG_CACHE_HOME=/workspace/.cache
export HF_HUB_CACHE=/workspace/hf-home/hub
```

When you recreate a pod with the same network volume, you do not need to reinstall the environment. Restore the same environment variables and reactivate the existing virtualenv:

```bash
export TMPDIR=/workspace/tmp
export PIP_CACHE_DIR=/workspace/pip-cache
export HF_HOME=/workspace/hf-home
export XDG_CACHE_HOME=/workspace/.cache
export HF_HUB_CACHE=/workspace/hf-home/hub

cd /workspace/LLM-infra
source /workspace/LLM-infra/.venv/bin/activate
```

If you want cost-per-step logging on Runpod, set the GPU hourly price either in `configs/experiment.yaml` under `infra.gpu_hourly_cost_usd` or through:

```bash
export GPU_HOURLY_COST_USD=0.33
```

Prepare data:

```bash
python scripts/prepare_apps.py \
  --dataset-name codeparrot/apps \
  --output-dir data/apps \
  --max-train-samples 3000 \
  --max-val-samples 300 \
  --difficulty interview,competition
```

Generate a manifest:

```bash
python scripts/make_manifest.py \
  --config configs/experiment.yaml \
  --run-name qwen25_1p5b_apps_full_sft
```

Train:

```bash
python scripts/train_full_sft.py \
  --config configs/experiment.yaml \
  --run-name qwen25_1p5b_apps_full_sft
```

Local smoke test on Mac:

```bash
python scripts/train_full_sft.py \
  --config configs/mac_smoke_test.yaml \
  --run-name qwen25_1p5b_apps_mac_smoke
```

This smoke config is only for validating the training loop, checkpointing, and eval hooks with a short sequence length and minimal batch settings. It is not intended for a real full-weight run on local hardware.

## Run Outputs

Each training run now records:

- `runs/<run_name>/config_snapshot.yaml`: exact config snapshot used for the run
- `runs/<run_name>/manifest.json`: run manifest with config hash, data hashes, command, and environment info
- `runs/<run_name>/git_info.json`: git branch, commit, and dirty-worktree state at launch time
- `runs/<run_name>/step_metrics.jsonl`: optimizer-step metrics including step time, tokens/sec, peak GPU memory, and cost per step
- `runs/<run_name>/checkpoint_metrics.jsonl`: checkpoint size and write-time history
- `runs/<run_name>/eval_metrics.jsonl`: validation loss over time
- `runs/<run_name>/summary.json`: final run summary
- `runs/<run_name>/run_report.md`: one-row markdown table for quick comparison
- `runs/<run_name>/final/`: exported Hugging Face model directory for inference and vLLM serving
- `runs/run_index.jsonl`: append-only run index across experiments

The `run_report.md` table includes:

- GPU
- sequence length
- batch size
- gradient accumulation
- average tokens/sec
- peak VRAM
- latest checkpoint size
- resume success
- final validation loss

Serve:

```bash
bash scripts/serve_vllm.sh runs/qwen25_1p5b_apps_full_sft/final
```

Smoke test the served checkpoint:

```bash
python scripts/smoke_test_vllm.py \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key local-dev \
  --model qwen25-1p5b-apps-full-sft
```

Collect serving metrics:

```bash
python scripts/benchmark_vllm.py \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key local-dev \
  --model qwen25-1p5b-apps-full-sft \
  --benchmark-name base_model_serving \
  --output runs/base_model_serving_metrics.json
```

Sample GPU memory while serving or benchmarking:

```bash
bash scripts/sample_gpu_memory.sh runs/base_model_gpu_memory.csv 1
```

Evaluate:

```bash
export OPENAI_API_KEY=local-dev
bash scripts/run_evalplus.sh http://127.0.0.1:8000/v1 Qwen/Qwen2.5-1.5B-Instruct humaneval
```

## Inference Evaluation Workflow

To collect both quality and serving metrics, do the following for the `base model` and then repeat for the `tuned model`.

1. Start vLLM with the target model.
2. In a second shell, start GPU memory sampling with `scripts/sample_gpu_memory.sh`.
3. Run `scripts/benchmark_vllm.py` to collect latency and tokens/sec.
4. Run `scripts/run_evalplus.sh` to collect benchmark quality metrics such as `pass@1`.
5. Stop the memory sampler and keep the CSV with the benchmark output.

This gives you the full comparison surface:

- model name: from the vLLM served model name and benchmark output file
- benchmark: from EvalPlus
- pass@1: from EvalPlus
- latency: from `benchmark_vllm.py`
- tokens/sec: from `benchmark_vllm.py`
- GPU memory: from `sample_gpu_memory.sh`

To inspect raw HumanEval outputs from the base and tuned models:

```bash
python scripts/dump_humaneval_samples.py \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key local-dev \
  --model qwen25-1p5b-base \
  --model qwen25-1p5b-sft-run-2 \
  --max-samples 10 \
  --output runs/humaneval_base_vs_tuned_samples.json
```

## Prompt format

Each training row should be converted into:

- `prompt`: system and user text with problem statement and optional starter code
- `completion`: one canonical Python solution

The prompt should ask for `code only` output. This keeps the training target aligned with functional code benchmarks.

## What you should implement yourself

This repo leaves these surfaces intentionally incomplete:

- APPS row filtering and canonical-solution selection
- final prompt template decisions
- completion-only masking in the collator
- Trainer or custom loop construction
- checkpoint promotion and eval gating policy

The repo already gives you the layout and the command boundaries so the work stays infra-oriented instead of turning into a blank notebook.

## GPU guidance

For this project, start with a `48 GB` Runpod GPU.

- best ROI target: `RTX A6000 48GB` or `A40 48GB`
- safest first phase: single GPU full-weight SFT
- stretch goal: move to distributed training after the single-GPU run is stable

## Time budget

- week 1, `6-8h`: prepare data, get the first run working, validate checkpoints
- later weeks, `3-4h`: run ablations, serve checkpoints, benchmark, and document tradeoffs

## Notes

- `APPS` is the easiest first training dataset to operationalize.
- `EvalPlus` is intentionally held out as the stable evaluation spine.
- Once the pipeline is stable, you can swap in `TACO` or a custom verifier-backed rollout stage.
