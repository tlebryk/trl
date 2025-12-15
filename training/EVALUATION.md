# Evaluation Pipeline

This directory contains scripts for evaluating trained models on the MultiPL-E humaneval-cpp benchmark.

## 📋 Overview

The evaluation pipeline is split into **two separate stages**:

1. **Inference** (GPU required) - Generate code completions
2. **Evaluation** (No GPU) - Compile, run, and compute metrics

This separation allows you to:
- Run expensive inference once on Modal with GPU
- Run evaluation multiple times locally without GPU
- Experiment with different metrics without re-running inference

## 🚀 Quick Start

### Option 1: End-to-End on Modal (Easiest)

Run both inference and evaluation on Modal in one command:

```bash
# Quick test (10 problems)
modal run training/modal_eval_end_to_end.py --num-problems 10

# Full evaluation with base model
modal run training/modal_eval_end_to_end.py

# Evaluate fine-tuned model (shorthand path)
modal run training/modal_eval_end_to_end.py --adapter-path token-rewards-v1/final_model

# Or use full path
modal run training/modal_eval_end_to_end.py --adapter-path /data/experiments/token-rewards-v1/final_model
```

### Option 2: Separate Inference + Evaluation (Most Flexible)

**Step 1: Generate completions on Modal (GPU)**

```bash
# Base model
modal run training/modal_inference.py --num-problems 10 --run-name base_test

# Fine-tuned model (using shorthand path)
modal run training/modal_inference.py \
  --adapter-path token-rewards-v1/final_model \
  --run-name token_rewards \
  --num-samples 20
```

**Step 2: Download completions from Modal volume**

```bash
# List available completions
modal volume ls dpo-training-vol inference_results/

# Download specific completions file
modal volume get dpo-training-vol \
  inference_results/completions_token_rewards.json \
  ./completions_token_rewards.json
```

**Step 3: Evaluate locally (no GPU needed)**

```bash
# Evaluate completions
python training/eval_completions.py ./completions_token_rewards.json

# Or specify custom output path
python training/eval_completions.py \
  ./completions_token_rewards.json \
  ./eval_token_rewards.json
```

## 📊 Metrics

The evaluation computes:

1. **Compilation Success Rate**: % of generated code that compiles without errors
2. **Runtime Success Rate**: % that compiles AND runs without crashes/sanitizer errors (ASan/UBSan)
3. **Pass@k**: Standard metric for code generation
   - pass@1: Probability that at least 1 sample passes
   - pass@5: Probability that at least 1 of 5 samples passes
   - pass@10: Probability that at least 1 of 10 samples passes
4. **Problems with ≥1 Pass**: Number of problems with at least one passing solution

## 📁 Files

### `modal_inference.py`
Generates code completions on Modal with GPU.

**Key parameters:**
- `--adapter-path`: Path to LoRA adapter in volume (or omit for base model)
  - Supports shorthand: `token-rewards-v1/final_model`
  - Or full path: `/data/experiments/token-rewards-v1/final_model`
  - Auto-prepends `/data/experiments/` if relative path is given
- `--num-problems`: Number of problems to evaluate (default: all ~164)
- `--num-samples`: Samples per problem for pass@k (default: 10)
- `--temperature`: Sampling temperature (default: 0.8)
- `--run-name`: Name for this run (used in output filename)

**Output:** Saves `completions_*.json` to Modal volume at `/data/inference_results/`

### `eval_completions.py`
Evaluates completions locally (no GPU needed).

**Usage:**
```bash
python training/eval_completions.py <completions.json> [output.json]
```

**What it does:**
1. Loads completions JSON
2. For each completion:
   - Compiles with `g++ -fsanitize=address -fsanitize=undefined`
   - Runs executable
   - Records success/failure
3. Computes metrics (compile rate, runtime rate, pass@k)
4. Saves `eval_*.json` with detailed results

**Requirements:** `g++` with ASan/UBSan support

### `modal_eval_end_to_end.py`
Convenience wrapper that runs both inference (GPU) and evaluation (CPU) on Modal.

**Usage:** Same as `modal_inference.py`

**What it does:**
1. Calls `generate_completions()` on GPU instance
2. Calls `evaluate_completions()` on CPU instance
3. Returns final metrics

## 📈 Comparing Models

To compare base model vs. fine-tuned models:

```bash
# 1. Generate completions for each model
modal run training/modal_inference.py --run-name base
modal run training/modal_inference.py \
  --adapter-path token-rewards-v1/final_model \
  --run-name token_rewards
modal run training/modal_inference.py \
  --adapter-path vanilla-dpo-v1/final_model \
  --run-name vanilla_dpo

# 2. Download all completions
modal volume get dpo-training-vol inference_results/completions_base.json ./
modal volume get dpo-training-vol inference_results/completions_token_rewards.json ./
modal volume get dpo-training-vol inference_results/completions_vanilla_dpo.json ./

# 3. Evaluate all
python training/eval_completions.py ./completions_base.json
python training/eval_completions.py ./completions_token_rewards.json
python training/eval_completions.py ./completions_vanilla_dpo.json

# 4. Compare results
# Results are in eval_*.json files with detailed metrics
```

## 🔍 Output Format

### Completions JSON (`completions_*.json`)
```json
{
  "metadata": {
    "model_id": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "adapter_path": "/data/experiments/my-experiment/final_model",
    "model_type": "experiments_my-experiment_final_model",
    "num_problems": 164,
    "num_samples": 10,
    "temperature": 0.8,
    "timestamp": "2025-12-15T10:30:00"
  },
  "problems": [
    {
      "name": "HumanEval/0",
      "prompt": "...",
      "tests": "...",
      "completions": [
        {
          "sample_idx": 0,
          "generated_code": "..."
        },
        ...
      ]
    },
    ...
  ]
}
```

### Evaluation Results JSON (`eval_*.json`)
```json
{
  "metadata": { ... },
  "problems": [
    {
      "name": "HumanEval/0",
      "samples": [
        {
          "sample_idx": 0,
          "compiled": true,
          "runtime_clean": true
        },
        ...
      ],
      "compile_rate": 0.9,
      "runtime_rate": 0.7,
      "passed_any": true
    },
    ...
  ],
  "summary": {
    "total_samples": 1640,
    "compile_success_rate": 0.85,
    "runtime_success_rate": 0.65,
    "pass@1": 0.65,
    "pass@5": 0.82,
    "pass@10": 0.91,
    "problems_with_any_pass": 145,
    "problems_with_any_pass_rate": 0.88
  }
}
```

## 🛠 Advanced Usage

### Custom Sampling Strategy

Generate more samples for better pass@k estimation:

```bash
modal run training/modal_inference.py \
  --adapter-path token-rewards-v1/final_model \
  --num-samples 50 \
  --temperature 0.6 \
  --run-name token_rewards_t06_n50
```

### Subset Evaluation

Test on a small subset first:

```bash
# Quick sanity check (10 problems)
modal run training/modal_eval_end_to_end.py --num-problems 10

# Medium test (50 problems)
modal run training/modal_eval_end_to_end.py --num-problems 50

# Full benchmark (all ~164 problems)
modal run training/modal_eval_end_to_end.py
```

### Re-evaluate with Different Metrics

Since completions are saved, you can modify `eval_completions.py` to add new metrics and re-run without doing inference again:

```bash
# Generate once
modal run training/modal_inference.py --run-name my_model

# Download
modal volume get dpo-training-vol inference_results/completions_my_model.json ./

# Evaluate multiple times (modify metrics in eval_completions.py)
python training/eval_completions.py ./completions_my_model.json
```

## 📝 Notes

- **GPU Usage**: Only `modal_inference.py` and `modal_eval_end_to_end.py` use GPU (for inference)
- **Compilation**: Requires `g++` with ASan/UBSan support (included in Modal eval image)
- **Timeout**: Compilation timeout: 10s, Runtime timeout: 5s
- **Storage**: Completions and eval results are saved to Modal volume `/data/inference_results/`
- **Cost**: Running inference on T4 GPU ~$0.60/hour, evaluation on CPU ~$0.10/hour
