# Evaluation Pipeline

Evaluate trained models on MultiPL-E humaneval-cpp. The pipeline has two stages: **Inference** (GPU) and **Evaluation** (CPU).

## 🚀 Quick Start (End-to-End on Modal)

Run inference and evaluation in one go:

```bash
# Quick test (10 problems)
modal run training/modal_eval_end_to_end.py --num-problems 10

# Full evaluation (base model)
modal run training/modal_eval_end_to_end.py

# Fine-tuned model (shorthand or full path)
modal run training/modal_eval_end_to_end.py --adapter-path token-rewards-v1/final_model
```

## 🛠 Flexible Workflow (Separate Inference & Eval)

**1. Inference on Modal (GPU)**
```bash
# Base model
modal run training/modal_inference.py --num-problems 10 --run-name base_test

# Fine-tuned model
modal run training/modal_inference.py \
  --adapter-path token-rewards-v1/final_model \
  --run-name token_rewards --num-samples 20
```

**2. Download Results**
```bash
modal volume ls dpo-training-vol inference_results/
modal volume get dpo-training-vol inference_results/completions_token_rewards.json ./
```

**3. Evaluate Locally (No GPU)**
```bash
python training/eval_completions.py ./completions_token_rewards.json [optional_output.json]
```

## 📈 Comparing Models

1.  **Generate** completions for base and fine-tuned models using `modal_inference.py`.
2.  **Download** all `completions_*.json` files.
3.  **Evaluate** locally using `eval_completions.py`.
4.  **Compare** `eval_*.json` outputs.

## 📁 Key Files & Metrics

*   **`modal_inference.py`**: Generates completions on GPU.
*   **`eval_completions.py`**: Compiles/runs code locally. calculates metrics.
*   **`modal_eval_end_to_end.py`**: Wrapper for both stages on Modal.

**Metrics:**
*   **Compilation/Runtime Rate**: % valid code.
*   **Pass@k**: (1, 5, 10) Probability of passing tests.
*   **Problems with ≥1 Pass**: Absolute count of solved problems.

## 🔍 Output JSON Structure

**Completions (`completions_*.json`)**
```json
{
  "metadata": { ... },
  "problems": [
    { "name": "HumanEval/0", "completions": [{ "sample_idx": 0, "generated_code": "..." }] }
  ]
}
```

**Results (`eval_*.json`)**
```json
{
  "summary": {
    "compile_success_rate": 0.85,
    "pass@1": 0.65,
    "pass@10": 0.91,
    "problems_with_any_pass": 145
  },
  "problems": [ ...detailed results... ]
}
```
