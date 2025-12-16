"""
Compute token-level rewards from compiler feedback.

This module:
1. Loads examples and compiler feedback
2. Converts compiler errors to token-level rewards
3. Saves rewards to rewards/{example_id}/
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from transformers import AutoTokenizer
from .cpp_utils import create_token_rewards_from_compiler_errors


def compute_rewards_for_example(
    example_id: str,
    tokenizer_name: str = "Qwen/Qwen2.5-Coder-0.5B",
    base_dir: str = "cpp_pipeline",
    error_reward: float = 0.0,
    runtime_error_reward: float = 0.2,
    warning_reward: float = 0.5,
    clean_reward: float = 1.0,
) -> Dict:
    """
    Compute token-level rewards for an example.
    
    Args:
        example_id: Example identifier
        tokenizer_name: Tokenizer to use
        base_dir: Base directory for pipeline
        error_reward: Reward for compile errors
        runtime_error_reward: Reward for runtime errors (default 0.2, better than compile error but worse than clean)
        warning_reward: Reward for warning lines
        clean_reward: Reward for clean lines
    
    Returns:
        Dict with chosen_rewards and rejected_rewards
    """
    examples_dir = Path(base_dir) / "examples" / example_id
    compiled_dir = Path(base_dir) / "compiled" / example_id
    rewards_dir = Path(base_dir) / "rewards" / example_id
    rewards_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load code and feedback
    chosen_code = (examples_dir / "chosen.cpp").read_text()
    rejected_code = (examples_dir / "rejected.cpp").read_text()
    
    chosen_feedback = json.loads((compiled_dir / "chosen_feedback.json").read_text())
    rejected_feedback = json.loads((compiled_dir / "rejected_feedback.json").read_text())
    
    # Compute rewards
    chosen_rewards = create_token_rewards_from_compiler_errors(
        chosen_code,
        chosen_feedback["errors"],
        tokenizer,
        error_reward=error_reward,
        runtime_error_reward=runtime_error_reward,
        warning_reward=warning_reward,
        clean_reward=clean_reward,
    )
    
    rejected_rewards = create_token_rewards_from_compiler_errors(
        rejected_code,
        rejected_feedback["errors"],
        tokenizer,
        error_reward=error_reward,
        runtime_error_reward=runtime_error_reward,
        warning_reward=warning_reward,
        clean_reward=clean_reward,
    )
    
    # Save rewards
    chosen_rewards_data = {
        "token_rewards": chosen_rewards,
        "tokenizer_name": tokenizer_name,
        "computed_at": datetime.now().isoformat(),
        "error_reward": error_reward,
        "runtime_error_reward": runtime_error_reward,
        "warning_reward": warning_reward,
        "clean_reward": clean_reward,
        "num_tokens": len(chosen_rewards),
    }
    
    rejected_rewards_data = {
        "token_rewards": rejected_rewards,
        "tokenizer_name": tokenizer_name,
        "computed_at": datetime.now().isoformat(),
        "error_reward": error_reward,
        "runtime_error_reward": runtime_error_reward,
        "warning_reward": warning_reward,
        "clean_reward": clean_reward,
        "num_tokens": len(rejected_rewards),
    }
    
    (rewards_dir / "chosen_rewards.json").write_text(
        json.dumps(chosen_rewards_data, indent=2)
    )
    
    (rewards_dir / "rejected_rewards.json").write_text(
        json.dumps(rejected_rewards_data, indent=2)
    )
    
    return {
        "chosen_rewards": chosen_rewards_data,
        "rejected_rewards": rejected_rewards_data,
    }


def compute_rewards_for_all_examples(
    tokenizer_name: str = "Qwen/Qwen2.5-Coder-0.5B",
    base_dir: str = "cpp_pipeline",
    error_reward: float = 0.0,
    runtime_error_reward: float = 0.2,
    warning_reward: float = 0.5,
    clean_reward: float = 1.0,
):
    """
    Compute rewards for all examples.
    """
    examples_dir = Path(base_dir) / "examples"
    
    if not examples_dir.exists():
        print(f"Examples directory not found: {examples_dir}")
        return
    
    example_dirs = [d for d in examples_dir.iterdir() if d.is_dir()]
    
    print(f"Computing rewards for {len(example_dirs)} examples using {tokenizer_name}...")
    
    for example_dir in sorted(example_dirs):
        example_id = example_dir.name
        print(f"\nComputing rewards for {example_id}...")
        
        try:
            results = compute_rewards_for_example(
                example_id,
                tokenizer_name,
                base_dir,
                error_reward,
                runtime_error_reward,
                warning_reward,
                clean_reward,
            )
            
            chosen_rewards = results["chosen_rewards"]["token_rewards"]
            rejected_rewards = results["rejected_rewards"]["token_rewards"]
            
            chosen_avg = sum(chosen_rewards) / len(chosen_rewards) if chosen_rewards else 0.0
            rejected_avg = sum(rejected_rewards) / len(rejected_rewards) if rejected_rewards else 0.0
            
            chosen_penalized = sum(1 for r in chosen_rewards if r < 1.0)
            rejected_penalized = sum(1 for r in rejected_rewards if r < 1.0)
            
            print(f"  Chosen: {len(chosen_rewards)} tokens, avg reward {chosen_avg:.2f}, {chosen_penalized} penalized (< 1.0)")
            print(f"  Rejected: {len(rejected_rewards)} tokens, avg reward {rejected_avg:.2f}, {rejected_penalized} penalized (< 1.0)")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")


if __name__ == "__main__":
    compute_rewards_for_all_examples()

