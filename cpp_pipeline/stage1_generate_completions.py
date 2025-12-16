import os
import json
import argparse
import random
from pathlib import Path
from typing import Tuple
from openai import OpenAI
from tqdm import tqdm

from cpp_pipeline.load_data import get_multipl_e_dataset

# Load environment variables if needed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class CompletionGenerator:
    def __init__(self, api_key: str, output_file: str):
        self.client = OpenAI(api_key=api_key)
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def generate_completion_pair(self, prompt: str) -> Tuple[str, str]:
        """
        Generate chosen (correct) and rejected (incorrect) completions for a given prompt.
        """
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
                # temperature=0.7, # o1/gpt-5.2 forces temp=1
                extra_body={
                    "reasoning_effort": "medium"
                }
            )
            chosen_code = response.choices[0].message.content
        except Exception as e:
            print(f"Error generating chosen completion: {e}")

        # 2. Rejected Completion
        # Flip a weighted coin:
        # 70% chance for a runtime error, 30% for a compiler error
        error_type = "runtime"
        if random.random() < 0.3:
            error_type = "compilation"

        system_msg_rejected = "You are an expert C++ competitive programmer. Write an INCORRECT solution for the given problem. It should either fail to compile or have a runtime bug."
        user_msg_rejected = f"""Complete the following C++ function with a 1-2 bugs.
Prompt:
{prompt}

The bugs should be subtle {error_type} error(s).
Do NOT include the closing brace '}}' at the end.
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {"role": "system", "content": system_msg_rejected},
                    {"role": "user", "content": user_msg_rejected}
                ],
                # temperature=0.7,
                extra_body={
                    "reasoning_effort": "medium"
                }
            )
            rejected_code = response.choices[0].message.content
        except Exception as e:
            print(f"Error generating rejected completion: {e}")

        return chosen_code, rejected_code

    def run(self, max_examples: int = None):
        print("Loading dataset...")
        examples = get_multipl_e_dataset("train")
        
        if max_examples:
            examples = examples[:max_examples]
            print(f"Limiting to {max_examples} examples.")
            
        print(f"Generating completions for {len(examples)} examples...")
        
        # Clear output file if starting fresh
        with open(self.output_file, "w") as f:
            pass

        for i, ex in enumerate(tqdm(examples)):
            prompt = ex["prompt"]
            tests = ex["tests"]
            
            # Generate raw completions
            raw_chosen, raw_rejected = self.generate_completion_pair(prompt)
            
            record = {
                "example_idx": i,
                "prompt": prompt,
                "tests": tests,
                "raw_chosen": raw_chosen,
                "raw_rejected": raw_rejected
            }
            
            # Append to file
            with open(self.output_file, "a") as f:
                f.write(json.dumps(record) + "\n")
                
        print(f"Raw completions saved to {self.output_file}")

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Generate raw C++ completions.")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--output-file", type=str, default="data/raw_completions/raw_completions.jsonl", help="Output file path")
    args = parser.parse_args()
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    generator = CompletionGenerator(api_key, args.output_file)
    generator.run(args.max_examples)

if __name__ == "__main__":
    main()
