# Token-Level Rewards for DPO & PPO (RLSF)

A complete framework for training LLMs with token-level rewards derived from symbolic feedback (compilers, runtime execution, etc.), inspired by the [RLSF paper](https://arxiv.org/abs/2405.16661).

This project implements a pipeline that generates C++ code, compiles it, runs it with sanitizers (ASan/UBSan), maps errors to specific source lines, and trains a model to prefer correct code using token-weighted DPO or PPO.

## 🚀 Quick Start

### 1. Generate Data & Rewards
Run the full pipeline to create examples, compile/run them, and compute token rewards.

```bash
python -m cpp_pipeline.run_pipeline
```
*Outputs: `cpp_pipeline/examples/`, `cpp_pipeline/compiled/`, `cpp_pipeline/rewards/`*

### 2. Train with Modal
Submit a training job to Modal (uses T4 GPU).

**Option A: DPO (Direct Preference Optimization)**
Use offline dataset of chosen/rejected pairs with pre-computed rewards.
```bash
modal run training/modal_train_dpo.py
```

**Option B: PPO (Proximal Policy Optimization)**
Use online generation with real-time feedback (compile/run).
```bash
modal run training/modal_train_ppo.py
```

*Outputs: Model checkpoints in Modal Volume `dpo-training-vol`*

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
  - Model: `Qwen/Qwen2.5-Coder-0.5B-Instruct`
  - Method: LoRA + Token-Weighted DPO
  - Trainer: Custom `TokenRewardDPOTrainer`
- **`train_ppo.py`**: PPO training loop.
  - Method: Online PPO + Token-Weighted Rewards
  - Reward Function: Compiles & Runs generated code on-the-fly.
  - Trainer: Custom `TokenRewardPPOTrainer`
- **`modal_train_dpo.py`**: Infrastructure wrapper for DPO.
- **`modal_train_ppo.py`**: Infrastructure wrapper for PPO.

### 3. Custom Trainers
- **`dpo_trainer_token_rewards.py`**: Extends `DPOTrainer` to weight log-probs by per-token rewards.
- **`ppo_trainer_token_rewards.py`**: Extends `PPOTrainer` to inject dense token-level rewards during the PPO update step.

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
│   ├── modal_train.py     # Modal wrapper (DPO)
│   └── modal_train_ppo.py # Modal wrapper (PPO)
├── dpo_trainer_token_rewards.py  # Custom DPO Trainer
├── ppo_trainer_token_rewards.py  # Custom PPO Trainer
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
