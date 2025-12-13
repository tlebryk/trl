# C++ Code Generation Pipeline for RLSF

## Pipeline Overview

```
1. Create Examples → 2. Compile → 3. Compute Rewards → 4. Prepare Dataset → 5. Train
```

## Data Flow

1. **Example Creation** (`create_examples.py`)
   - Define prompts and code pairs (chosen/rejected)
   - Save as `.cpp` files + `metadata.json`
   - Structure: `examples/{example_id}/chosen.cpp`, `rejected.cpp`, `metadata.json`

2. **Compilation** (`compile_examples.py`)
   - Compile each `.cpp` file
   - Extract compiler feedback (errors, warnings, line numbers)
   - Save feedback as JSON: `compiled/{example_id}/chosen_feedback.json`

3. **Reward Computation** (`compute_rewards.py`)
   - Convert compiler feedback → token-level rewards
   - Map errors to specific tokens
   - Save rewards: `rewards/{example_id}/chosen_rewards.json`

4. **Dataset Preparation** (`prepare_dataset.py`)
   - Load examples + rewards
   - Create HuggingFace Dataset
   - Ready for training

## Data Schema

### metadata.json
```json
{
  "example_id": "example_001",
  "prompt": "Write a C++ function that...",
  "created_at": "2025-01-XX",
  "model_version": null,  // For online RL: track which model generated this
  "source": "manual"  // or "model_generated"
}
```

### feedback.json
```json
{
  "compiles": false,
  "errors": [
    {
      "line": 4,
      "column": 17,
      "type": "error",
      "message": "expected ';' after return statement"
    }
  ],
  "warnings": [],
  "stderr": "...",
  "compiled_at": "2025-01-XX"
}
```

### rewards.json
```json
{
  "token_rewards": [1.0, 1.0, 0.0, 0.0, ...],
  "tokenizer_name": "gpt2",
  "computed_at": "2025-01-XX"
}
```

## Online RL Support

For online RL where the model generates code:
- Store `model_version` in metadata
- Can recompute rewards for same code with different tokenizers
- Can join rewards from multiple sources (compiler + tests)

