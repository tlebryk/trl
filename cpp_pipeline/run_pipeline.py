"""
Run the complete C++ code pipeline.

This script runs all stages:
1. Create examples
2. Compile examples
3. Run examples (runtime checks)
4. Compute rewards
5. Prepare dataset
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cpp_pipeline import (
    create_all_examples,
    compile_all_examples,
    run_all_examples,
    compute_rewards_for_all_examples,
    prepare_dataset,
)


def main():
    """Run the complete pipeline."""
    print("=" * 80)
    print("C++ CODE GENERATION PIPELINE")
    print("=" * 80)
    
    base_dir = "cpp_pipeline"
    
    # Stage 1: Create examples
    print("\n[1/5] Creating examples...")
    print("-" * 80)
    example_ids = create_all_examples(str(Path(base_dir) / "examples"))
    print(f"✓ Created {len(example_ids)} examples")
    
    # Stage 2: Compile examples
    print("\n[2/5] Compiling examples...")
    print("-" * 80)
    compile_all_examples(base_dir)
    print("✓ Compilation complete")

    # Stage 3: Run examples
    print("\n[3/5] Running examples (Runtime Checks)...")
    print("-" * 80)
    run_all_examples(base_dir)
    print("✓ Execution complete")
    
    # Stage 4: Compute rewards
    print("\n[4/5] Computing token-level rewards...")
    print("-" * 80)
    compute_rewards_for_all_examples(
        tokenizer_name="Qwen/Qwen2.5-Coder-0.5B", 
        base_dir=base_dir,
        runtime_error_reward=0.2  # Better than 0.0 (compile error), worse than 1.0 (clean)
    )
    print("✓ Rewards computed")
    
    # Stage 5: Prepare dataset
    print("\n[5/5] Preparing dataset...")
    print("-" * 80)
    dataset = prepare_dataset(base_dir)
    print(f"✓ Dataset ready with {len(dataset)} examples")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nDataset features: {list(dataset.features.keys())}")
    print(f"\nExample usage:")
    print(f"  from cpp_pipeline import prepare_dataset")
    print(f"  dataset = prepare_dataset()")
    print(f"  # Use dataset with DPOTrainer")


if __name__ == "__main__":
    main()
