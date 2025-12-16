import math
from typing import Tuple, Dict, Any
from datasets import load_dataset, Dataset

def load_and_split_dataset() -> Tuple[Dataset, Dataset]:
    """
    Loads the nuprl/MultiPL-E humaneval-cpp dataset and splits it 50:50.
    Returns (train_dataset, test_dataset).
    """
    # Load the full dataset. 
    # Note: MultiPL-E typically exposes a 'test' split which contains all problems.
    full_dataset = load_dataset("nuprl/MultiPL-E", "humaneval-cpp", split="test")
    
    total_rows = len(full_dataset)
    split_idx = math.floor(total_rows * 0.5)
    
    # Simple slicing
    train_dataset = full_dataset.select(range(0, split_idx))
    test_dataset = full_dataset.select(range(split_idx, total_rows))
    
    print(f"Loaded dataset with {total_rows} total rows.")
    print(f"Split into Train: {len(train_dataset)} rows, Test: {len(test_dataset)} rows.")
    
    return train_dataset, test_dataset

def get_prompt_and_tests(example: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extracts the prompt and tests from a dataset example.
    
    Args:
        example: A dictionary-like object from the dataset row.
        
    Returns:
        Tuple of (prompt, tests)
    """
    prompt = example['prompt']
    tests = example['tests']
    return prompt, tests

def stitch_solution(prompt: str, solution: str, tests: str) -> str:
    """
    Stitches the prompt, solution, and tests into a compilable C++ file.
    """
    # Verify the prompt usually ends with { or similar, and solution shouldn't duplicate it if possible, 
    # but usually the solution follows the prompt. 
    # The prompt in MultiPL-E ends with the function signature opening brace `{`.
    # The tests typically start with `}` to close the function, then `int main() { ... }`.
    
    return f"{prompt}\n{solution}\n{tests}"
