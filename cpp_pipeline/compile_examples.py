"""
Compile C++ examples and gather compiler feedback.

This module:
1. Loads examples from examples/
2. Compiles each chosen.cpp and rejected.cpp
3. Saves compiler feedback to compiled/{example_id}/
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from .cpp_utils import compile_cpp_code, parse_compiler_errors


def compile_example(example_id: str, base_dir: str = "cpp_pipeline", compiler: str = "g++") -> Dict:
    """
    Compile both chosen and rejected code for an example.
    
    Args:
        example_id: Example identifier (e.g., "example_001")
        base_dir: Base directory for pipeline
        compiler: Compiler to use
    
    Returns:
        Dict with chosen_feedback and rejected_feedback
    """
    examples_dir = Path(base_dir) / "examples" / example_id
    compiled_dir = Path(base_dir) / "compiled" / example_id
    compiled_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Compile chosen code
    chosen_code = (examples_dir / "chosen.cpp").read_text()
    chosen_success, chosen_stderr, chosen_errors = compile_cpp_code(chosen_code, compiler)
    
    chosen_feedback = {
        "compiles": chosen_success,
        "errors": chosen_errors,
        "warnings": [e for e in chosen_errors if e["type"] == "warning"],
        "stderr": chosen_stderr,
        "compiled_at": datetime.now().isoformat(),
        "compiler": compiler,
    }
    
    (compiled_dir / "chosen_feedback.json").write_text(
        json.dumps(chosen_feedback, indent=2)
    )
    results["chosen_feedback"] = chosen_feedback
    
    # Compile rejected code
    rejected_code = (examples_dir / "rejected.cpp").read_text()
    rejected_success, rejected_stderr, rejected_errors = compile_cpp_code(rejected_code, compiler)
    
    rejected_feedback = {
        "compiles": rejected_success,
        "errors": rejected_errors,
        "warnings": [e for e in rejected_errors if e["type"] == "warning"],
        "stderr": rejected_stderr,
        "compiled_at": datetime.now().isoformat(),
        "compiler": compiler,
    }
    
    (compiled_dir / "rejected_feedback.json").write_text(
        json.dumps(rejected_feedback, indent=2)
    )
    results["rejected_feedback"] = rejected_feedback
    
    return results


def compile_all_examples(base_dir: str = "cpp_pipeline", compiler: str = "g++"):
    """
    Compile all examples in the examples directory.
    
    Args:
        base_dir: Base directory for pipeline
        compiler: Compiler to use
    """
    examples_dir = Path(base_dir) / "examples"
    
    if not examples_dir.exists():
        print(f"Examples directory not found: {examples_dir}")
        return
    
    example_dirs = [d for d in examples_dir.iterdir() if d.is_dir()]
    
    print(f"Compiling {len(example_dirs)} examples...")
    
    for example_dir in sorted(example_dirs):
        example_id = example_dir.name
        print(f"\nCompiling {example_id}...")
        
        try:
            results = compile_example(example_id, base_dir, compiler)
            
            chosen_compiles = results["chosen_feedback"]["compiles"]
            rejected_compiles = results["rejected_feedback"]["compiles"]
            rejected_errors = len(results["rejected_feedback"]["errors"])
            
            print(f"  Chosen: {'✓ compiles' if chosen_compiles else '✗ errors'}")
            print(f"  Rejected: {'✓ compiles' if rejected_compiles else f'✗ {rejected_errors} errors'}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")


if __name__ == "__main__":
    compile_all_examples()

