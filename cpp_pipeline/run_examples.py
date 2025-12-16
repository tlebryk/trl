"""
Run compiled C++ examples and gather runtime feedback.

This module:
1. Loads compiled examples from compiled/
2. Runs the executables (if compilation succeeded)
3. Updates feedback with runtime information
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from .cpp_utils import run_cpp_executable, link_executable

def run_example(example_id: str, base_dir: str = "cpp_pipeline") -> Dict:
    """
    Run both chosen and rejected executables for an example.
    
    Args:
        example_id: Example identifier
        base_dir: Base directory for pipeline
    
    Returns:
        Dict with updated feedback
    """
    examples_dir = Path(base_dir) / "examples" / example_id
    compiled_dir = Path(base_dir) / "compiled" / example_id
    
    # Load existing feedback
    chosen_feedback_path = compiled_dir / "chosen_feedback.json"
    rejected_feedback_path = compiled_dir / "rejected_feedback.json"
    
    if not chosen_feedback_path.exists() or not rejected_feedback_path.exists():
        print(f"  Skipping {example_id}: compilation feedback missing")
        return {}
        
    chosen_feedback = json.loads(chosen_feedback_path.read_text())
    rejected_feedback = json.loads(rejected_feedback_path.read_text())
    
    # Run chosen code
    if chosen_feedback.get("compiles", False):
        # We need to re-link or just run the .o file? 
        # Actually compile_cpp_code produces a .o file. We need to link it to an executable.
        # But wait, compile_cpp_code in cpp_utils cleans up the .o file!
        # We need to change compile_cpp_code to keep the executable or re-compile here.
        # For simplicity, let's just re-compile to executable here since source exists.
        
        # Or better: modify compile_examples.py to save the executable? 
        # Given the modular design, it's better if "compile" stage produces binaries.
        # But compile_examples just calls compile_cpp_code which cleans up.
        
        # Let's re-compile here for execution. 
        src_path = examples_dir / "chosen.cpp"
        exe_path = compiled_dir / "chosen.exe"
        
        # Compile to executable
        build_success = subprocess.run(
            ["g++", "-std=c++17", "-g", "-fsanitize=address", "-fsanitize=undefined", str(src_path), "-o", str(exe_path)],
            capture_output=True
        ).returncode == 0
        
        if build_success:
            success, stdout, stderr, errors = run_cpp_executable(str(exe_path))
            chosen_feedback["runtime_success"] = success
            chosen_feedback["stdout"] = stdout
            chosen_feedback["stderr"] = stderr # Contains sanitizer output
            chosen_feedback["runtime_errors"] = errors
            
            # Merge runtime errors into errors list
            chosen_feedback["errors"].extend(errors)
            
            # Cleanup
            if exe_path.exists():
                os.unlink(exe_path)
        else:
             chosen_feedback["runtime_success"] = False
             chosen_feedback["runtime_errors"] = [{"line": None, "type": "build_error", "message": "Failed to build executable for runtime"}]

    # Run rejected code
    if rejected_feedback.get("compiles", False):
        src_path = examples_dir / "rejected.cpp"
        exe_path = compiled_dir / "rejected.exe"
        
        build_success = subprocess.run(
            ["g++", "-std=c++17", "-g", "-fsanitize=address", "-fsanitize=undefined", str(src_path), "-o", str(exe_path)],
            capture_output=True
        ).returncode == 0
        
        if build_success:
            success, stdout, stderr, errors = run_cpp_executable(str(exe_path))
            rejected_feedback["runtime_success"] = success
            rejected_feedback["stdout"] = stdout
            rejected_feedback["stderr"] = stderr
            rejected_feedback["runtime_errors"] = errors
            
            # Merge runtime errors into errors list
            rejected_feedback["errors"].extend(errors)
            
            # Cleanup
            if exe_path.exists():
                os.unlink(exe_path)
        else:
             rejected_feedback["runtime_success"] = False
             rejected_feedback["runtime_errors"] = [{"line": None, "type": "build_error", "message": "Failed to build executable for runtime"}]

    # Save updated feedback
    chosen_feedback_path.write_text(json.dumps(chosen_feedback, indent=2))
    rejected_feedback_path.write_text(json.dumps(rejected_feedback, indent=2))
    
    return {
        "chosen": chosen_feedback,
        "rejected": rejected_feedback
    }

def run_all_examples(base_dir: str = "cpp_pipeline"):
    """
    Run all compiled examples.
    """
    examples_dir = Path(base_dir) / "examples"
    
    if not examples_dir.exists():
        print(f"Examples directory not found: {examples_dir}")
        return
    
    example_dirs = [d for d in examples_dir.iterdir() if d.is_dir()]
    
    print(f"Running {len(example_dirs)} examples...")
    
    for example_dir in sorted(example_dirs):
        example_id = example_dir.name
        print(f"\nRunning {example_id}...")
        
        try:
            results = run_example(example_id, base_dir)
            
            if not results:
                continue
                
            chosen = results["chosen"]
            rejected = results["rejected"]
            
            if chosen.get("compiles"):
                status = "✓ passed" if chosen.get("runtime_success") else f"✗ failed ({len(chosen.get('runtime_errors', []))} errors)"
                print(f"  Chosen: {status}")
            else:
                print(f"  Chosen: (skipped - build failed)")
                
            if rejected.get("compiles"):
                status = "✓ passed" if rejected.get("runtime_success") else f"✗ failed ({len(rejected.get('runtime_errors', []))} errors)"
                print(f"  Rejected: {status}")
            else:
                print(f"  Rejected: (skipped - build failed)")

        except Exception as e:
            print(f"  ✗ Error: {e}")

if __name__ == "__main__":
    run_all_examples()

