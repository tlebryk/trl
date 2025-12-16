import os
import json
import argparse
import torch
import shutil
import sys
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List, Dict, Any

# Ensure we can import from local
sys.path.insert(0, str(Path(__file__).parent.parent))

from cpp_pipeline.cpp_utils import (
    compile_cpp_code, 
    link_executable, 
    run_cpp_executable, 
    clean_generated_code,
    create_token_rewards_from_compiler_errors
)

class RewardComputer:
    def __init__(self, input_file: str, output_file: str, model_name: str = "Qwen/Qwen2.5-Coder-0.5B", debug: bool = False):
        self.input_file = input_file
        self.output_file = output_file
        self.model_name = model_name
        self.debug = debug
        
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _compute_rewards_for_completion(self, prompt: str, completion: str, tests: str, role: str = "unknown", idx: int = -1) -> List[float]:
        """
        Compute token-level rewards for a single completion.
        Returns a list of rewards aligned with the completion tokens.
        """
        # 1. Clean completion (remove main, etc.)
        cleaned_response = clean_generated_code(completion)
        
        # Simple extraction heuristic to match what we do in validation
        if "```cpp" in cleaned_response:
            code_body = cleaned_response.split("```cpp")[1].split("```")[0]
        elif "```c++" in cleaned_response:
            code_body = cleaned_response.split("```c++")[1].split("```")[0]
        elif "```" in cleaned_response:
            code_body = cleaned_response.split("```")[1].split("```")[0]
        else:
            code_body = cleaned_response
            
        # Strip potentially hallucinated closing braces if they unbalance the code
        code_body_stripped = code_body.rstrip()
        if code_body_stripped.endswith('}'):
            open_count = code_body_stripped.count('{')
            close_count = code_body_stripped.count('}')
            if close_count > open_count:
                last_brace = code_body_stripped.rfind('}')
                code_body = code_body_stripped[:last_brace]

        # 2. Stitch full code
        full_code = f"{prompt}\n{code_body}\n{tests}"
        
        # MAC OS FIX: Replace <bits/stdc++.h> with standard headers if on Darwin
        # This is required for local validation/reward computation on macOS
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
        
        # 3. Compile
        success, stderr, errors = compile_cpp_code(full_code)
        
        # 4. Run if successful
        runtime_errors = []
        if success:
            import tempfile
            src_path = ""
            exe_path = ""
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                    f.write(full_code)
                    src_path = f.name
                
                exe_path = src_path + ".exe"
                if link_executable(src_path, exe_path):
                    run_success, _, run_stderr, run_errors = run_cpp_executable(exe_path)
                    if not run_success:
                        runtime_errors = run_errors
                        # If run failed but parsing returned no errors (e.g. timeout or segfault without stacktrace), 
                        # ensure we have at least one error
                        if not runtime_errors:
                            runtime_errors = [{"line": None, "type": "runtime_error", "message": f"Runtime failed: {run_stderr[:100]}..."}]
                else:
                    # Link failed
                    runtime_errors = [{"line": None, "type": "link_error", "message": "Linking failed"}]
            finally:
                if src_path and os.path.exists(src_path):
                    os.unlink(src_path)
                if exe_path and os.path.exists(exe_path):
                    os.unlink(exe_path)
        
        all_errors = errors + runtime_errors
        
        if self.debug:
            status = "CLEAN"
            if not success: status = "COMPILE_ERROR"
            elif runtime_errors: status = "RUNTIME_ERROR"
            
            if status != "CLEAN" or idx < 3: # Print first 3 always, others if error
                print(f"\n[{idx}] {role.upper()} Status: {status}")
                if all_errors:
                    for e in all_errors:
                        print(f"  - Line {e.get('line')}: {e.get('message')}")

        # 5. Compute dense rewards for the FULL code
        # Rewards: 1.0 (clean), 0.0 (compile error), 0.2 (runtime error)
        full_code_rewards = create_token_rewards_from_compiler_errors(
            full_code,
            all_errors,
            self.tokenizer,
            error_reward=0.0,
            runtime_error_reward=0.2,
            clean_reward=1.0
        )
        
        # 6. Extract rewards corresponding to the COMPLETION
        completion_tokens = self.tokenizer(completion, add_special_tokens=False)["input_ids"]
        expected_len = len(completion_tokens)
        
        # Default: all 1.0 (optimistic fallback)
        final_rewards = [1.0] * expected_len
        
        # Find where code_body starts in full_code
        code_body_start_idx = full_code.find(code_body)
        
        if code_body_start_idx != -1:
            # Get offset mapping for full_code
            full_encoding = self.tokenizer(full_code, add_special_tokens=False, return_offsets_mapping=True)
            full_offsets = full_encoding['offset_mapping']
            
            body_start = code_body_start_idx
            body_end = body_start + len(code_body)
            
            # Identify which lines in full_code correspond to the body
            # To debug why errors might be missed
            if self.debug:
                body_lines_start = full_code[:body_start].count('\n') + 1
                body_lines_end = full_code[:body_end].count('\n') + 1
                print(f"  Body range in full_code: char {body_start}-{body_end}, lines {body_lines_start}-{body_lines_end}")
                
                if all_errors:
                    for e in all_errors:
                        e_line = e.get('line')
                        if e_line:
                            if body_lines_start <= e_line <= body_lines_end:
                                print(f"  -> Error at line {e_line} IS inside body range.")
                            else:
                                print(f"  -> Error at line {e_line} IS OUTSIDE body range ({body_lines_start}-{body_lines_end}).")
            
            # Extract rewards that overlap with code_body
            relevant_rewards = []
            for idx, (start, end) in enumerate(full_offsets):
                # We check intersection. If token is inside the body range.
                mid = (start + end) / 2
                if body_start <= mid < body_end:
                    if idx < len(full_code_rewards):
                        relevant_rewards.append(full_code_rewards[idx])
            
            if relevant_rewards:
                # Check if any penalty exists in relevant rewards
                min_reward = min(relevant_rewards)
                if self.debug and min_reward < 1.0:
                    print(f"  -> Found penalty in body tokens! Min reward: {min_reward}")
                elif self.debug and all_errors:
                    print(f"  -> WARNING: Errors exist but body tokens are clean (all 1.0). Error likely outside body.")

                # If we have runtime errors but relevant_rewards are all clean (1.0),
                # it means the error was mapped to lines outside the body (e.g. test harness).
                # In this case, we should punish the WHOLE body.
                if any(e['type'] == 'runtime_error' for e in all_errors) and min(relevant_rewards) == 1.0:
                    if self.debug:
                        print(f"  -> Applying global runtime penalty (0.2) to body because runtime error occurred outside body lines.")
                    relevant_rewards = [0.2] * len(relevant_rewards)

                # Similarly for compile errors if strict
                if not success and min(relevant_rewards) == 1.0:
                     if self.debug:
                        print(f"  -> Applying global compile penalty (0.0) to body.")
                     relevant_rewards = [0.0] * len(relevant_rewards)

                # Now map 'relevant_rewards' (code_body) to 'completion' tokens
                comp_body_start = completion.find(code_body)
                if comp_body_start != -1:
                    comp_encoding = self.tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
                    comp_offsets = comp_encoding['offset_mapping']
                    
                    final_rewards = [1.0] * expected_len
                    
                    r_idx = 0
                    comp_body_end = comp_body_start + len(code_body)
                    
                    for t_idx, (start, end) in enumerate(comp_offsets):
                        mid = (start + end) / 2
                        if comp_body_start <= mid < comp_body_end:
                            if r_idx < len(relevant_rewards):
                                final_rewards[t_idx] = relevant_rewards[r_idx]
                                r_idx += 1
                            else:
                                final_rewards[t_idx] = relevant_rewards[-1] if relevant_rewards else 1.0
        
        return final_rewards

    def run(self, max_examples: int = None):
        print(f"Loading dataset from {self.input_file}...")
        if not os.path.exists(self.input_file):
            print(f"Error: File {self.input_file} not found.")
            return

        with open(self.input_file, "r") as f:
            lines = f.readlines()
            
        if max_examples:
            lines = lines[:max_examples]
            print(f"Limiting to {max_examples} examples.")
            
        print(f"Processing {len(lines)} records...")
        
        updated_records = []
        
        for i, line in enumerate(tqdm(lines)):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            prompt = record["prompt"]
            tests = record["tests"]
            chosen = record["chosen"]
            rejected = record["rejected"]
            
            # Compute rewards
            chosen_rewards = self._compute_rewards_for_completion(prompt, chosen, tests, role="chosen", idx=i)
            rejected_rewards = self._compute_rewards_for_completion(prompt, rejected, tests, role="rejected", idx=i)
            
            # Add to record
            record["chosen_token_rewards"] = chosen_rewards
            record["rejected_token_rewards"] = rejected_rewards
            
            updated_records.append(record)
            
        # Write output
        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w") as f:
            for record in updated_records:
                f.write(json.dumps(record) + "\n")
                
        print(f"Saved {len(updated_records)} records with token rewards to {self.output_file}")

def main():
    parser = argparse.ArgumentParser(description="Stage 3: Compute token-level rewards for DPO dataset.")
    parser.add_argument("--input-file", type=str, default="data/training_data/dpo_training_data.jsonl", help="Input validated JSONL")
    parser.add_argument("--output-file", type=str, default="data/training_data/dpo_training_data_with_token_rewards.jsonl", help="Output JSONL with rewards")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-0.5B", help="Model for tokenization")
    parser.add_argument("--debug", action="store_true", help="Enable debug printing")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of examples for debugging")
    args = parser.parse_args()
    
    computer = RewardComputer(args.input_file, args.output_file, args.model, args.debug)
    computer.run(args.max_examples)

if __name__ == "__main__":
    main()
