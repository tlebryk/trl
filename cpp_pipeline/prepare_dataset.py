"""
Prepare HuggingFace Dataset from examples + rewards.

This module loads all examples, metadata, and rewards and creates
a dataset ready for training with DPO.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
from datasets import Dataset

# Add project root to path to import training config
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from training.config.reward_config import RewardConfig


def load_example_data(example_id: str, base_dir: str = "cpp_pipeline") -> Dict:
    """
    Load all data for an example: metadata, code, feedback, rewards.
    
    Args:
        example_id: Example identifier
        base_dir: Base directory for pipeline
    
    Returns:
        Dict with all example data
    """
    examples_dir = Path(base_dir) / "examples" / example_id
    compiled_dir = Path(base_dir) / "compiled" / example_id
    rewards_dir = Path(base_dir) / "rewards" / example_id
    
    # Load metadata
    metadata = json.loads((examples_dir / "metadata.json").read_text())
    
    # Load code
    chosen_code = (examples_dir / "chosen.cpp").read_text()
    rejected_code = (examples_dir / "rejected.cpp").read_text()
    
    # Load feedback
    chosen_feedback = json.loads((compiled_dir / "chosen_feedback.json").read_text())
    rejected_feedback = json.loads((compiled_dir / "rejected_feedback.json").read_text())
    
    # Load rewards
    chosen_rewards = json.loads((rewards_dir / "chosen_rewards.json").read_text())
    rejected_rewards = json.loads((rewards_dir / "rejected_rewards.json").read_text())

    # Compute sequence-level rewards from token-level rewards
    reward_config = RewardConfig()
    chosen_reward = reward_config.sequence_reward_from_tokens(chosen_rewards["token_rewards"])
    rejected_reward = reward_config.sequence_reward_from_tokens(rejected_rewards["token_rewards"])

    return {
        "example_id": example_id,
        "prompt": metadata["prompt"],
        "chosen": chosen_code,
        "rejected": rejected_code,
        # Token-level rewards
        "chosen_token_rewards": chosen_rewards["token_rewards"],
        "rejected_token_rewards": rejected_rewards["token_rewards"],
        # Sequence-level rewards (mean of token rewards)
        "chosen_reward": chosen_reward,
        "rejected_reward": rejected_reward,
        # Compilation metadata
        "chosen_compiles": chosen_feedback["compiles"],
        "rejected_compiles": rejected_feedback["compiles"],
        "chosen_errors": chosen_feedback["errors"],
        "rejected_errors": rejected_feedback["errors"],
        "metadata": metadata,
    }


def prepare_dataset(base_dir: str = "cpp_pipeline") -> Dataset:
    """
    Prepare HuggingFace Dataset from all examples.
    
    Args:
        base_dir: Base directory for pipeline
    
    Returns:
        HuggingFace Dataset ready for training
    """
    examples_dir = Path(base_dir) / "examples"
    
    if not examples_dir.exists():
        raise ValueError(f"Examples directory not found: {examples_dir}")
    
    example_dirs = [d for d in examples_dir.iterdir() if d.is_dir()]
    
    print(f"Loading {len(example_dirs)} examples...")
    
    examples = []
    for example_dir in sorted(example_dirs):
        example_id = example_dir.name
        try:
            example_data = load_example_data(example_id, base_dir)
            examples.append(example_data)
        except Exception as e:
            print(f"Warning: Failed to load {example_id}: {e}")
    
    print(f"Loaded {len(examples)} examples")
    
    # Create dataset
    dataset = Dataset.from_list(examples)
    
    return dataset


if __name__ == "__main__":
    dataset = prepare_dataset()
    
    print(f"\nDataset created:")
    print(f"  Examples: {len(dataset)}")
    print(f"  Features: {dataset.features.keys()}")
    
    print(f"\nFirst example:")
    print(f"  Prompt: {dataset[0]['prompt']}")
    print(f"  Chosen compiles: {dataset[0]['chosen_compiles']}")
    print(f"  Rejected compiles: {dataset[0]['rejected_compiles']}")
    print(f"  Chosen rewards: {len(dataset[0]['chosen_token_rewards'])} tokens")
    print(f"  Rejected rewards: {len(dataset[0]['rejected_token_rewards'])} tokens")

