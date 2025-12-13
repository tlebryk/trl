# Pipeline Summary

See the main [README](../README.md) for the complete project overview.

## Pipeline Details

The pipeline processes data in 5 stages:

1.  **Creation**: Generates `examples/{id}/chosen.cpp` and `rejected.cpp`.
2.  **Compilation**: Runs `g++` and parses stderr for line numbers.
3.  **Execution**: Runs binaries and parses ASan/UBSan stack traces.
4.  **Reward Computation**:
    *   Uses `Qwen/Qwen2.5-Coder-0.5B-Instruct` tokenizer.
    *   Maps error lines to tokens using offset mapping.
    *   Assigns rewards: `0.0` (Compile Error), `0.2` (Runtime Error), `1.0` (Clean).
5.  **Dataset Preparation**: Exports to HuggingFace format.

## Debugging

Intermediate results are saved as JSON for easy inspection:
- `compiled/{id}/*.json`: Raw compiler/runtime output.
- `rewards/{id}/*.json`: Per-token reward arrays.
