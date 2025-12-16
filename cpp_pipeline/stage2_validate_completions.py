import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any

from cpp_pipeline.cpp_utils import compile_cpp_code, link_executable, run_cpp_executable, clean_generated_code

class CompletionValidator:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Log file for detailed feedback
        self.log_file = Path(output_file).parent / "validation_feedback.jsonl"

    def _clean_response(self, content: str) -> str:
        """Clean markdown and extra whitespace."""
        if not content:
            return ""
            
        if "```cpp" in content:
            content = content.split("```cpp")[1].split("```")[0].strip()
        elif "```c++" in content:
             content = content.split("```c++")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Remove closing brace if present at the very end
        content = content.strip()
        if content.endswith("}"):
            content = content[:-1].strip()
            
        # Heuristic: If content starts with the function signature, strip it.
        lines = content.split('\n')
        if lines:
            first_line = lines[0].strip()
            if (first_line.endswith('{') or first_line.endswith(';')) and len(lines) > 1:
                if '(' in first_line and ')' in first_line:
                    content = '\n'.join(lines[1:]).strip()
        return content

    def validate_solution(self, prompt: str, completion: str, tests: str, role: str) -> Dict[str, Any]:
        """
        Validate the solution by compiling and running it with the tests.
        Returns a dictionary with detailed feedback.
        role: "chosen" or "rejected"
        """
        result = {
            "role": role,
            "status": "unknown",
            "compile_success": False,
            "compile_stderr": "",
            "run_success": False,
            "run_stderr": "",
            "valid_pair_member": False
        }

        # Clean and Stitch code
        cleaned_completion = clean_generated_code(completion)
        
        full_code = f"{prompt}\n{cleaned_completion}\n{tests}"
        
        # MAC OS FIX: Replace <bits/stdc++.h> with standard headers if on Darwin
        if sys.platform == "darwin" and "#include<bits/stdc++.h>" in full_code:
            standard_headers = """
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <stack>
#include <numeric>
#include <limits>
#include <iomanip>
"""
            full_code = full_code.replace("#include<bits/stdc++.h>", standard_headers)
        
        # Compile
        success, stderr, errors = compile_cpp_code(full_code)
        result["compile_success"] = success
        result["compile_stderr"] = stderr

        if not success:
            if role == "chosen":
                result["status"] = "compile_error"
                result["valid_pair_member"] = False
            else: # rejected
                result["status"] = "compile_error"
                result["valid_pair_member"] = True # A rejected solution failing to compile is valid
            return result

        # Run
        import tempfile
        src_path = ""
        exe_path = ""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write(full_code)
                src_path = f.name
            
            exe_path = src_path + ".exe"
            if link_executable(src_path, exe_path):
                run_success, stdout, stderr, run_errors = run_cpp_executable(exe_path)
                result["run_success"] = run_success
                result["run_stderr"] = stderr
                
                if run_success:
                    result["status"] = "passed"
                    if role == "chosen":
                        result["valid_pair_member"] = True
                    else: # rejected
                        result["valid_pair_member"] = False # Rejected should NOT pass
                else:
                    result["status"] = "runtime_error"
                    if role == "chosen":
                        result["valid_pair_member"] = False
                    else: # rejected
                        result["valid_pair_member"] = True # Runtime error is good for rejected
            else:
                result["status"] = "link_error"
                result["valid_pair_member"] = False if role == "chosen" else True # Arguably link error is like compile error
                result["compile_stderr"] += "\nLink failed"

        finally:
            if src_path and os.path.exists(src_path):
                os.unlink(src_path)
            if exe_path and os.path.exists(exe_path):
                os.unlink(exe_path)
        
        return result

    def run(self):
        print(f"Loading raw completions from {self.input_file}...")
        if not os.path.exists(self.input_file):
            print(f"Error: Input file {self.input_file} not found.")
            return

        valid_pairs = []
        total = 0
        
        with open(self.input_file, "r") as f:
            lines = f.readlines()
            
        print(f"Processing {len(lines)} records...")
        
        # Clear log file
        with open(self.log_file, "w") as f:
            pass

        for line in tqdm(lines):
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            prompt = record["prompt"]
            tests = record["tests"]
            raw_chosen = record["raw_chosen"]
            raw_rejected = record["raw_rejected"]
            example_idx = record.get("example_idx", -1)
            
            # Clean
            chosen = self._clean_response(raw_chosen)
            rejected = self._clean_response(raw_rejected)
            
            # Validate Chosen (Must Pass)
            chosen_res = self.validate_solution(prompt, chosen, tests, role="chosen")
            
            # Validate Rejected (Must Fail)
            rejected_res = self.validate_solution(prompt, rejected, tests, role="rejected")
            
            # Log feedback
            feedback = {
                "example_idx": example_idx,
                "prompt_snippet": prompt[:50] + "...",
                "chosen_validation": chosen_res,
                "rejected_validation": rejected_res
            }
            with open(self.log_file, "a") as f:
                f.write(json.dumps(feedback) + "\n")

            # Decision Logic
            if not chosen_res["valid_pair_member"]:
                print(f"Example {example_idx}: Chosen failed ({chosen_res['status']})")
                continue
                
            if not rejected_res["valid_pair_member"]:
                print(f"Example {example_idx}: Rejected passed unexpected ({rejected_res['status']})")
                continue
                
            # Save valid pair
            valid_pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "tests": tests,
                "metadata": {
                    "chosen_status": chosen_res["status"],
                    "rejected_status": rejected_res["status"]
                }
            })
            
        # Write output
        with open(self.output_file, "w") as f:
            for pair in valid_pairs:
                f.write(json.dumps(pair) + "\n")
                
        print(f"Saved {len(valid_pairs)} valid pairs (from {total} raw records) to {self.output_file}")
        print(f"Detailed validation feedback saved to {self.log_file}")

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Validate C++ completions.")
    parser.add_argument("--input-file", type=str, default="data/raw_completions/raw_completions.jsonl", help="Input raw JSONL")
    parser.add_argument("--output-file", type=str, default="data/training_data/dpo_training_data.jsonl", help="Output validated JSONL")
    args = parser.parse_args()
    
    validator = CompletionValidator(args.input_file, args.output_file)
    validator.run()

if __name__ == "__main__":
    main()
