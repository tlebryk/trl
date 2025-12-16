from typing import List, Dict
import json
import os
from pathlib import Path
from datasets import Dataset
from cpp_pipeline.dataset_utils import load_and_split_dataset, get_prompt_and_tests

def get_multipl_e_dataset(split: str = "train") -> List[Dict[str, str]]:
    """
    Loads the MultiPL-E dataset split and returns a list of dictionaries 
    compatible with the training pipeline.
    
    Args:
        split: "train" or "test"
        
    Returns:
        List of dicts with keys "prompt" and "tests".
    """
    train_ds, test_ds = load_and_split_dataset()
    
    selected_ds = train_ds if split == "train" else test_ds
    
    examples = []
    for i in range(len(selected_ds)):
        item = selected_ds[i]
        prompt, tests = get_prompt_and_tests(item)
        examples.append({
            "prompt": prompt,
            "tests": tests
        })
        
    return examples

def get_dpo_dataset(data_path: str = "data/training_data/dpo_training_data.jsonl") -> Dataset:
    """
    Load the synthetic DPO training data from JSONL.
    """
    if not os.path.exists(data_path):
        # Fallback to local project path if running elsewhere
        project_root = Path(__file__).parent.parent
        data_path = str(project_root / data_path)
        
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"DPO training data not found at {data_path}. Run cpp_pipeline/prepare_training_data.py first.")
        
    data = []
    with open(data_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                
    return Dataset.from_list(data)
