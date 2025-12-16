# C++ Synthetic Data Pipeline Status

**Date:** December 16, 2025
**Current Status:** Partially Working / In Progress

## Architecture
The pipeline is split into two distinct stages to allow for easier debugging and validation:

1.  **Stage 1: Generation (`cpp_pipeline/stage1_generate.py`)**
    *   **Goal:** Generate raw C++ competitive programming problems using `gpt-5.2`.
    *   **Process:**
        *   Generates a **Prompt** (docstring + signature).
        *   Generates **Tests** (main function with assertions).
        *   Generates **Chosen** completion (correct solution).
        *   Generates **Rejected** completion (buggy solution).
    *   **Output:** JSON files in `cpp_pipeline/raw_examples/example_XXX/raw.json`.
    *   **Current State:** Working. Generates prompts requesting standard headers to avoid macOS `<bits/stdc++.h>` issues.

2.  **Stage 2: Validation (`cpp_pipeline/stage2_validate.py`)**
    *   **Goal:** Compile and run generated code to ensure correctness.
    *   **Process:**
        *   Stitches `Prompt + Completion + Tests`.
        *   Compiles using `clang++ -std=c++17`.
        *   **Validates Chosen:** Must compile and run with exit code 0.
        *   **Validates Rejected:** Must fail compilation OR fail runtime (exit code != 0).
    *   **Output:** Validated C++ files in `cpp_pipeline/validated_examples/example_XXX/`.
        *   `chosen.cpp`
        *   `rejected.cpp`
        *   `metadata.json`
    *   **Current State:** Working logic, but success rate depends on LLM output quality.

## Current Results (Last Run)
*   **Generated:** 2 examples.
*   **Validated:** 1 example passed, 1 failed.
    *   `example_001`: **PASSED**.
    *   `example_002`: **FAILED** (Chosen solution failed to compile).

## Artifact Locations
*   **Raw Gen Data:** `cpp_pipeline/raw_examples/`
*   **Validated Data:** `cpp_pipeline/validated_examples/`

## Known Issues / Blockers
1.  **Header Files:** macOS `clang` does not support `<bits/stdc++.h>`. The prompt was updated to request standard headers, but older or random generations might still use it, causing compilation failures.
2.  **Success Rate:** Currently ~50% based on a tiny sample size. Invalid C++ syntax or hallucinated libraries in the "Chosen" solution are the main failure modes.
3.  **Integration:** Need to ensure `prepare_dataset.py` reads from `cpp_pipeline/validated_examples` for the final training step.

## Next Recommended Steps
1.  Scale up generation (e.g., generate 20, expect ~10 valid).
2.  Review `prepare_dataset.py` to point to the new `validated_examples` directory.
3.  Run the full training loop.
