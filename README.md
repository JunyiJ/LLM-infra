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

Install serving and benchmark extras on the Runpod machine:

```bash
pip install vllm evalplus
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

Serve:

```bash
bash scripts/serve_vllm.sh runs/qwen25_1p5b_apps_full_sft/checkpoints/final
```

Evaluate:

```bash
bash scripts/run_evalplus.sh http://127.0.0.1:8000/v1 Qwen/Qwen2.5-1.5B-Instruct humaneval
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
