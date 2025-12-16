# Token-Level Rewards for DPO, PPO & GRPO (RLSF)

A complete framework for training LLMs with token-level rewards derived from symbolic feedback (compilers, runtime execution, etc.), inspired by the [RLSF paper](https://arxiv.org/abs/2405.16661).

This project implements a pipeline that generates C++ code, compiles it, runs it with sanitizers (ASan/UBSan), maps errors to specific source lines, and trains a model to prefer correct code using token-weighted DPO, PPO, or GRPO.

## 🚀 Quick Start

### 1. Generate Training Data

**Option A: Generate Synthetic MultiPL-E Dataset (Recommended)**

Generate high-quality synthetic C++ problems using OpenAI API:

```bash
# Install data generation dependencies
uv sync --extra data_generation

# Generate 50 problems using GPT-4o
export OPENAI_API_KEY="sk-..."
make generate-multipl NUM=50

# Validate generated data
make validate-multipl

# Run full pipeline (compile, run, compute rewards)
make pipeline
```

See [MULTIPL_DATASET_GUIDE.md](./MULTIPL_DATASET_GUIDE.md) for detailed instructions.

**Option B: Use Manual Examples**

Run the pipeline with hand-crafted examples:

```bash
python -m cpp_pipeline.run_pipeline
```

*Outputs: `cpp_pipeline/examples/`, `cpp_pipeline/compiled/`, `cpp_pipeline/rewards/`*

### 2. Train with Modal
Submit a training job to Modal (uses L4 GPU). Each run requires a unique experiment name.

**Option A: DPO (Direct Preference Optimization)**
Use offline dataset of chosen/rejected pairs with pre-computed rewards.
```bash
# With token-level rewards
modal run training/modal_train_dpo.py --experiment-name dpo-token-rewards-v1

# Vanilla DPO baseline
modal run training/modal_train_dpo.py --experiment-name dpo-baseline --no-use-token-level-rewards
```

**Option B: PPO (Proximal Policy Optimization)**
Use online generation with real-time feedback (compile/run).
```bash
# With token-level rewards
modal run training/modal_train_ppo.py --experiment-name ppo-token-rewards-v1

# Vanilla PPO baseline
modal run training/modal_train_ppo.py --experiment-name ppo-baseline --no-use-token-level-rewards
```

**Option C: GRPO (Group Relative Policy Optimization)**
Use online generation with group-wise reward normalization.
```bash
# With token-level rewards
modal run training/modal_train_grpo.py --experiment-name grpo-token-rewards-v1

# Vanilla GRPO baseline
modal run training/modal_train_grpo.py --experiment-name grpo-baseline --no-use-token-level-rewards
```

### 3. Download Results
Results are stored in Modal Volume `dpo-training-vol` under `/experiments/{experiment-name}`.
**Use `--force` to overwrite existing local files.**

```bash
# List all experiments
modal volume ls dpo-training-vol /experiments

# Download specific experiment (force overwrite local)
modal volume get --force dpo-training-vol /experiments/dpo-token-rewards-v1 ./results/dpo-token-rewards-v1
modal volume get --force dpo-training-vol /experiments/ppo-token-rewards-v1 ./results/ppo-token-rewards-v1
modal volume get --force dpo-training-vol /experiments/grpo-token-rewards-v1 ./results/grpo-token-rewards-v1

# View experiment metadata
modal volume get dpo-training-vol /experiments/dpo-token-rewards-v1/run_metadata.json -
```

**Experiment structure:**
```
experiments/
├── dpo-token-rewards-v1/
│   ├── final_model/           # LoRA adapter + tokenizer (~35MB)
│   ├── runs/                  # TensorBoard logs
│   └── run_metadata.json      # Experiment config & timestamp
├── ppo-baseline/
│   └── ...
└── grpo-token-rewards-v1/
    └── ...
```

*Note: Intermediate checkpoints and optimizer states are NOT saved to minimize storage and download time.*

### 4. Generate Completions
Generate code completions for evaluation using a trained model.

```bash
# Base model (no fine-tuning)
uv run modal run training/modal_inference.py --run-name base

# Fine-tuned model
uv run modal run training/modal_inference.py \
  --adapter-path dpo-token-rewards-v1/final_model \
  --run-name dpo-eval
```

*Outputs: `/data/inference_results/completions_*.json` on Modal volume*

### 5. Evaluate Completions

Evaluate generated completions by compiling and running them with g++ and sanitizers.

**Option A: Modal (Cloud CPU Evaluation)**
```bash
# Evaluate on Modal
uv run modal run training/modal_eval_completions.py \
  --completions-path /data/inference_results/completions_dpo-eval.json

# Download results
modal volume get dpo-training-vol /data/inference_results/eval_completions_dpo-eval.json ./
modal volume get dpo-training-vol /data/inference_results/eval_completions_dpo-eval_artifacts ./
```

**Option B: Docker (Local Linux Evaluation)**
```bash
# First, download completions from Modal
modal volume get dpo-training-vol /data/inference_results/completions_dpo-eval.json ./completions/

# Build Docker image (one-time)
docker-compose build

# Run evaluation in Docker
docker-compose run --rm eval uv run python training/eval_completions.py \
  /completions/completions_dpo-eval.json \
  /eval_results/eval_dpo.json

# Results appear in ./eval_results/
```

**Option C: End-to-End on Modal (Inference + Evaluation)**
```bash
# Run both inference and evaluation in one command
uv run modal run training/modal_eval_end_to_end.py \
  --adapter-path dpo-token-rewards-v1/final_model \
  --num-problems 10

# Download results
modal volume get dpo-training-vol /data/evaluation_results ./results/
```

## 🏗 Architecture

### 1. Data Pipeline (`cpp_pipeline/`)
Modular stages to transform code into training data.

| Stage | Script | Description |
|-------|--------|-------------|
| **1. Create** | `create_examples.py` | Generates C++ source pairs (chosen/rejected). |
| **2. Compile** | `compile_examples.py` | Compiles with `g++ -fsanitize=address -fsanitize=undefined`. Captures syntax errors. |
| **3. Run** | `run_examples.py` | Executes binaries. Captures runtime crashes (segfaults, overflows) & stack traces. |
| **4. Rewards** | `compute_rewards.py` | Maps errors to tokens. Assigns rewards: **1.0** (Clean), **0.2** (Runtime Error), **0.0** (Compile Error). |
| **5. Dataset** | `prepare_dataset.py` | Loads everything into a HuggingFace Dataset for training. |

### 2. Training (`training/`)
Scripts for fine-tuning the model.

- **`train_dpo.py`**: DPO training loop.
  - Model: `Qwen/Qwen2.5-Coder-0.5B`
  - Method: LoRA + Token-Weighted DPO
  - Trainer: Custom `TokenRewardDPOTrainer`
- **`train_ppo.py`**: PPO training loop.
  - Method: Online PPO + Token-Weighted Rewards
  - Reward Function: Compiles & Runs generated code on-the-fly.
  - Trainer: Custom `TokenRewardPPOTrainer`
- **`train_grpo.py`**: GRPO training loop.
  - Method: Online GRPO + Token-Weighted Rewards
  - Reward Function: Compiles & Runs generated code with group normalization
  - Trainer: Custom `TokenRewardGRPOTrainer`
- **`modal_train_dpo.py`**: Infrastructure wrapper for DPO.
- **`modal_train_ppo.py`**: Infrastructure wrapper for PPO.
- **`modal_train_grpo.py`**: Infrastructure wrapper for GRPO.

### 3. Custom Trainers
- **`dpo_trainer_token_rewards.py`**: Extends `DPOTrainer` to weight log-probs by per-token rewards.
- **`ppo_trainer_token_rewards.py`**: Extends `PPOTrainer` to inject dense token-level rewards during the PPO update step.
- **`grpo_trainer_token_rewards.py`**: Extends `GRPOTrainer` to compute token-level return-to-go and group-normalized advantages.

## 📂 Project Structure

```
.
├── cpp_pipeline/          # Data generation & processing
│   ├── examples/          # Source code & metadata
│   ├── compiled/          # Compiler/Runtime feedback JSONs
│   ├── rewards/           # Computed token rewards JSONs
│   ├── cpp_utils.py       # Compilation/Execution helpers
│   └── ...
├── training/              # Training scripts
│   ├── train_dpo.py       # DPO training script
│   ├── train_ppo.py       # PPO training script
│   ├── train_grpo.py      # GRPO training script
│   ├── modal_train_dpo.py # Modal wrapper (DPO)
│   ├── modal_train_ppo.py # Modal wrapper (PPO)
│   ├── modal_train_grpo.py # Modal wrapper (GRPO)
│   └── config/            # Shared training configs (LoRA, etc.)
├── dpo_trainer_token_rewards.py  # Custom DPO Trainer
├── ppo_trainer_token_rewards.py  # Custom PPO Trainer
├── grpo_trainer_token_rewards.py # Custom GRPO Trainer
└── README.md              # This file
```

## 🧠 Key Concepts

**Reward Curriculum:**
The reward scheme creates a natural curriculum for the model:
1.  **Syntactic Correctness**: First, learn to compile (avoid 0.0 penalty).
2.  **Runtime Safety**: Next, learn to not crash (avoid 0.2 penalty).
3.  **Functional Correctness**: Finally, write clean code (aim for 1.0).

**Symbolic Feedback:**
Instead of a single scalar reward ("Bad Code"), we give precise feedback:
- "This specific line caused a segfault."
- "This specific line has a syntax error."
This allows the model to learn *exactly* what went wrong.

## 🛠 Prerequisites

- Python 3.10+
- `g++` (with ASan/UBSan support)
- `modal` (for cloud training)
- Dependencies: `transformers`, `peft`, `trl`, `datasets`, `torch`


## 📋 Complete Pipeline Example

### Full Workflow: Train → Inference → Evaluation

```bash
# Step 1: Train a model (DPO example)
uv run modal run training/modal_train_dpo.py --experiment-name dpo-token-rewards-v1

# Step 2: Download trained model (optional, for local inspection)
modal volume get --force dpo-training-vol /experiments/dpo-token-rewards-v1 ./results/dpo-token-rewards-v1

# Step 3: Generate completions (inference)
uv run modal run training/modal_inference.py \
  --adapter-path dpo-token-rewards-v1/final_model \
  --run-name dpo-eval

# Step 4a: Evaluate on Modal (recommended for cloud workflow)
uv run modal run training/modal_eval_completions.py \
  --completions-path /data/inference_results/completions_dpo-eval.json

# Download evaluation results
modal volume get dpo-training-vol /data/inference_results/eval_completions_dpo-eval.json ./
modal volume get dpo-training-vol /data/inference_results/eval_completions_dpo-eval_artifacts ./

# OR Step 4b: Evaluate with Docker (for local development)
modal volume get dpo-training-vol /data/inference_results/completions_dpo-eval.json ./completions/
docker-compose build
docker-compose run --rm eval uv run python training/eval_completions.py \
  /completions/completions_dpo-eval.json \
  /eval_results/eval_dpo.json
```

### Quick Comparison: PPO, DPO, GRPO

```bash
# Train all three methods
uv run modal run training/modal_train_dpo.py --experiment-name dpo-v1
uv run modal run training/modal_train_ppo.py --experiment-name ppo-v1
uv run modal run training/modal_train_grpo.py --experiment-name grpo-v1

# Generate completions for each
uv run modal run training/modal_inference.py --adapter-path dpo-v1/final_model --run-name dpo
uv run modal run training/modal_inference.py --adapter-path ppo-v1/final_model --run-name ppo
uv run modal run training/modal_inference.py --adapter-path grpo-v1/final_model --run-name grpo
uv run modal run training/modal_inference.py --run-name base  # Base model comparison

# Evaluate all on Modal
uv run modal run training/modal_eval_completions.py --completions-path completions_dpo.json
uv run modal run training/modal_eval_completions.py --completions-path completions_ppo.json
uv run modal run training/modal_eval_completions.py --completions-path completions_grpo.json
uv run modal run training/modal_eval_completions.py --completions-path completions_base.json

# Download and compare results
modal volume get dpo-training-vol /data/inference_results/eval_*.json ./eval_results/
```