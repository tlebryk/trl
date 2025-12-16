import os
import sys
import json
import argparse
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
from tqdm import tqdm

from cpp_pipeline.load_data import get_multipl_e_dataset
from cpp_pipeline.cpp_utils import compile_cpp_code, link_executable, run_cpp_executable, clean_generated_code

# Load environment variables if needed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class TrainingDataGenerator:
    def __init__(self, api_key: str, output_dir: str = "data/training_data"):
        self.client = OpenAI(api_key=api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_completion_pair(self, prompt: str) -> Tuple[str, str]:
        """
        Generate chosen (correct) and rejected (incorrect) completions for a given prompt.
        """
        # INSERT_YOUR_CODE
        import random

        # Flip a weighted coin:
        # 70% chance for a runtime error, 30% for a compiler error
        error_type = "runtime"
        if random.random() < 0.75:
            error_type = "compilation"

        chosen_code = ""
        rejected_code = ""

        # 1. Chosen Completion
        system_msg_chosen = "You are an expert C++ competitive programmer. Write a correct and efficient solution for the given problem."
        user_msg_chosen = f"""Complete the following C++ function.
Prompt:
{prompt}

The code should complete the function body.
The output should NOT repeat the signature. It should start with the code inside the function.
Do NOT include the closing brace '}}' at the end, as the test file starts with it.
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {"role": "system", "content": system_msg_chosen},
                    {"role": "user", "content": user_msg_chosen}
                ],
                temperature=0.7,
                reasoning_effort="none"

            )
            chosen_code = self._clean_response(response.choices[0].message.content)
        except Exception as e:
            print(f"Error generating chosen completion: {e}")

        # 2. Rejected Completion
        system_msg_rejected = "You are an expert C++ competitive programmer. Write an INCORRECT solution for the given problem. It should either fail to compile or have a runtime bug."
        user_msg_rejected = f"""Complete the following C++ function with a 1-2 bugs.
Prompt:
{prompt}

The bugs should be subtle {error_type} error(s). A compiler error can be as simple as deleting some semicolons or declaring the wrong type but ideally is more subtle/complex.
Do NOT include the closing brace '}}' at the end.
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {"role": "system", "content": system_msg_rejected},
                    {"role": "user", "content": user_msg_rejected}
                ],
                temperature=0.7,
                reasoning_effort="none"

            )
            rejected_code = self._clean_response(response.choices[0].message.content)
        except Exception as e:
            print(f"Error generating rejected completion: {e}")

        return chosen_code, rejected_code

    def _clean_response(self, content: str, prompt_signature: str = None) -> str:
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
            
        # Heuristic: If content starts with the function signature (or looks like it), strip it.
        # We check if the first line ends with '{' or contains the function name
        lines = content.split('\n')
        if lines:
            first_line = lines[0].strip()
            # If the first line resembles a signature (ends with { or ;) and we have more lines
            if (first_line.endswith('{') or first_line.endswith(';')) and len(lines) > 1:
                # Check if it looks like a function decl (has parens)
                if '(' in first_line and ')' in first_line:
                    # Remove the first line
                    content = '\n'.join(lines[1:]).strip()
        
        return content

    def validate_solution(self, prompt: str, completion: str, tests: str, expect_success: bool = True) -> bool:
        """
        Validate the solution by compiling and running it with the tests.
        """
        # Stitch code
        cleaned_completion = clean_generated_code(completion)
        
        full_code = f"{prompt}\n{cleaned_completion}\n{tests}"
        
        # MAC OS FIX: Replace <bits/stdc++.h> with standard headers if on Darwin
        if sys.platform == "darwin" and "#include<bits/stdc++.h>" in full_code:
            standard_headers = """
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <queue>
#include <stack>
#include <numeric>
#include <limits>
"""
            full_code = full_code.replace("#include<bits/stdc++.h>", standard_headers)
        
        # Compile
        success, stderr, errors = compile_cpp_code(full_code)
        
        if not success:
            if expect_success:
                print(f"Validation failed (compile error):")
                print(f"--- Full Code ---\n{full_code}\n-----------------")
                print(f"--- Stderr ---\n{stderr}\n--------------")
                return False
            else:
                return True # Failed as expected

        # Run
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(full_code)
            src_path = f.name
            
        exe_path = src_path + ".exe"
        try:
            if link_executable(src_path, exe_path):
                run_success, stdout, stderr, run_errors = run_cpp_executable(exe_path)
                
                if run_success:
                    return True if expect_success else False
                else:
                    if expect_success:
                         print(f"Validation failed (runtime error): {stderr}")
                    return False if expect_success else True
            else:
                 if expect_success:
                     print("Link failed")
                 return False if expect_success else True
        finally:
            if os.path.exists(src_path):
                os.unlink(src_path)
            if os.path.exists(exe_path):
                os.unlink(exe_path)
        
        return False

    def run(self, max_examples: Optional[int] = None):
        print("Loading dataset...")
        examples = get_multipl_e_dataset("train")
        
        if max_examples:
            examples = examples[:max_examples]
            print(f"Limiting to {max_examples} examples.")
            
        valid_pairs = []
        
        print(f"Processing {len(examples)} examples...")
        for i, ex in enumerate(tqdm(examples)):
            prompt = ex["prompt"]
            tests = ex["tests"]
            
            # Generate
            chosen, rejected = self.generate_completion_pair(prompt)
            
            if not chosen or not rejected:
                print(f"Skipping example {i}: generation failed")
                continue
                
            # Validate Chosen (Must Pass)
            if not self.validate_solution(prompt, chosen, tests, expect_success=True):
                print(f"Example {i}: Chosen solution failed validation.")
                continue
                
            # Validate Rejected (Must Fail)
            if not self.validate_solution(prompt, rejected, tests, expect_success=False):
                print(f"Example {i}: Rejected solution unexpected passed validation.")
                print(f"--- Rejected Code (Unexpectedly Passed) ---\n{rejected}\n-------------------------------------------")
                continue
                
            # If we get here, we have a good pair
            valid_pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "tests": tests
            })
            
        # Save results
        output_file = self.output_dir / "dpo_training_data.jsonl"
        with open(output_file, "w") as f:
            for pair in valid_pairs:
                f.write(json.dumps(pair) + "\n")
                
        print(f"Saved {len(valid_pairs)} valid pairs to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data for DPO.")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--output-dir", type=str, default="data/training_data", help="Output directory")
    args = parser.parse_args()
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    generator = TrainingDataGenerator(api_key, args.output_dir)
    generator.run(args.max_examples)

if __name__ == "__main__":
    main()
