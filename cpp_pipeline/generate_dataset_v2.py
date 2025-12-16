import os
import json
import subprocess
import argparse
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
from tqdm import tqdm

# Load environment variables if needed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class CPPSyntheticDataPipeline:
    def __init__(self, api_key: str, output_dir: str = "cpp_pipeline"):
        self.client = OpenAI(api_key=api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Few-shot examples from the user-provided file
        self.example_prompt = """#include<assert.h>
#include<bits/stdc++.h>
// Check if in given vector of numbers, are any two numbers closer to each other than
// given threshold.
// >>> has_close_elements((std::vector<float>({(float)1.0f, (float)2.0f, (float)3.0f})), (0.5f))
// (false)
// >>> has_close_elements((std::vector<float>({(float)1.0f, (float)2.8f, (float)3.0f, (float)4.0f, (float)5.0f, (float)2.0f})), (0.3f))
// (true)
bool has_close_elements(std::vector<float> numbers, float threshold) {
"""
        
        self.example_tests = """}
int main() {
    auto candidate = has_close_elements;
    assert(candidate((std::vector<float>({(float)1.0f, (float)2.0f, (float)3.9f, (float)4.0f, (float)5.0f, (float)2.2f})), (0.3f)) == (true));
    assert(candidate((std::vector<float>({(float)1.0f, (float)2.0f, (float)3.9f, (float)4.0f, (float)5.0f, (float)2.2f})), (0.05f)) == (false));
    assert(candidate((std::vector<float>({(float)1.0f, (float)2.0f, (float)5.9f, (float)4.0f, (float)5.0f})), (0.95f)) == (true));
    assert(candidate((std::vector<float>({(float)1.0f, (float)2.0f, (float)5.9f, (float)4.0f, (float)5.0f})), (0.8f)) == (false));
    assert(candidate((std::vector<float>({(float)1.0f, (float)2.0f, (float)3.0f, (float)4.0f, (float)5.0f, (float)2.0f})), (0.1f)) == (true));
    assert(candidate((std::vector<float>({(float)1.1f, (float)2.2f, (float)3.1f, (float)4.1f, (float)5.1f})), (1.0f)) == (true));
    assert(candidate((std::vector<float>({(float)1.1f, (float)2.2f, (float)3.1f, (float)4.1f, (float)5.1f})), (0.5f)) == (false));
}
"""

    def generate_prompts(self, n: int = 2) -> List[str]:
        """Generate n C++ problem prompts."""
        print(f"Generating {n} prompts...")
        
        prompts = []
        for _ in range(n):
            system_msg = "You are an expert C++ competitive programmer. Generate a C++ coding problem prompt. It should include imports, a comment description with doctests (>>> format), and the function signature ending with '{\\n'."
            user_msg = f"Generate a C++ problem prompt similar to this format:\n\n{self.example_prompt}\n\nMake sure it is a valid C++ problem with imports, comments, and signature."
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-5.2",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    temperature=0.7,
                    reasoning={
                        "effort": "medium"
                    }
                )
                content = response.choices[0].message.content
                # Clean markdown
                if "```cpp" in content:
                    content = content.split("```cpp")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                # Ensure it ends with { or {\n
                if not content.strip().endswith("{"):
                     if not content.strip().endswith(";"):
                        content = content.strip() + " {\n"
                
                prompts.append(content)
            except Exception as e:
                print(f"Error generating prompt: {e}")
        
        return prompts

    def generate_tests(self, prompt: str) -> str:
        """Generate tests for a given prompt."""
        system_msg = "You are an expert C++ competitive programmer. Generate a C++ main function with assertions to test a given problem."
        user_msg = f"""Given this problem prompt:
{prompt}

Generate a set of tests similar to this format:
{self.example_tests}

The output should start with '}}\nint main() {{' and contain assertions testing the function defined in the prompt.
Ensure the logic matches the prompt's requirements.
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.7
            )
            content = response.choices[0].message.content
             # Clean markdown
            if "```cpp" in content:
                content = content.split("```cpp")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            return content
        except Exception as e:
            print(f"Error generating tests: {e}")
            return ""

    def generate_completions(self, prompt: str, tests: str) -> Tuple[str, str]:
        """Generate chosen (correct) and rejected (incorrect) completions."""
        
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
                temperature=0.7
            )
            chosen_code = response.choices[0].message.content
            if "```cpp" in chosen_code:
                chosen_code = chosen_code.split("```cpp")[1].split("```")[0].strip()
            elif "```" in chosen_code:
                chosen_code = chosen_code.split("```")[1].split("```")[0].strip()
            
            # Remove closing brace if present at the very end
            chosen_code = chosen_code.strip()
            if chosen_code.endswith("}"):
                chosen_code = chosen_code[:-1].strip()
                
        except Exception as e:
            print(f"Error generating chosen completion: {e}")
            chosen_code = ""

        # 2. Rejected Completion
        system_msg_rejected = "You are an expert C++ competitive programmer. Write an INCORRECT solution for the given problem. It should either fail to compile or have a runtime bug."
        user_msg_rejected = f"""Complete the following C++ function with a BUGGY implementation.
Prompt:
{prompt}

The bug should be subtle if possible, or a compilation error.
Do NOT include the closing brace '}}' at the end.
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {"role": "system", "content": system_msg_rejected},
                    {"role": "user", "content": user_msg_rejected}
                ],
                temperature=0.7
            )
            rejected_code = response.choices[0].message.content
            if "```cpp" in rejected_code:
                rejected_code = rejected_code.split("```cpp")[1].split("```")[0].strip()
            elif "```" in rejected_code:
                rejected_code = rejected_code.split("```")[1].split("```")[0].strip()
            
             # Remove closing brace if present at the very end
            rejected_code = rejected_code.strip()
            if rejected_code.endswith("}"):
                rejected_code = rejected_code[:-1].strip()

        except Exception as e:
            print(f"Error generating rejected completion: {e}")
            rejected_code = ""

        return chosen_code, rejected_code

    def stitch_code(self, prompt: str, completion: str, tests: str) -> str:
        """Combine parts into a single file content."""
        # Clean up newlines to avoid massive gaps or running together
        return f"{prompt.strip()}\n{completion.strip()}\n{tests.strip()}"

    def validate_code(self, code: str) -> bool:
        """Compile and run the code. Return True if success (compiles and runs 0), False otherwise."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(code)
            source_path = f.name
        
        output_path = source_path.replace('.cpp', '.out')
        
        try:
            # Compile
            compile_result = subprocess.run(
                ["g++-15", "-std=c++17", "-o", output_path, source_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if compile_result.returncode != 0:
                print(f"Compilation failed: {compile_result.stderr}")
                return False
            
            # Run
            run_result = subprocess.run(
                [output_path],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if run_result.returncode != 0:
                print(f"Runtime failed: {run_result.stderr}")
                return False
                
            return True
            
        except Exception as e:
            print(f"Validation exception: {e}")
            return False
        finally:
            if os.path.exists(source_path):
                os.remove(source_path)
            if os.path.exists(output_path):
                os.remove(output_path)

    def run(self, num_examples: int = 2):
        prompts = self.generate_prompts(num_examples)
        
        count = 0
        for i, prompt in enumerate(prompts):
            print(f"Processing prompt {i+1}/{num_examples}...")
            
            tests = self.generate_tests(prompt)
            if not tests:
                continue
                
            chosen_comp, rejected_comp = self.generate_completions(prompt, tests)
            
            full_chosen = self.stitch_code(prompt, chosen_comp, tests)
            full_rejected = self.stitch_code(prompt, rejected_comp, tests)
            
            # Validate Chosen
            if not self.validate_code(full_chosen):
                print(f"Chosen code failed validation. Skipping.")
                continue
                
            # Validate Rejected (should fail)
            if self.validate_code(full_rejected):
                print(f"Rejected code PASSED validation (it should fail). Skipping.")
                continue
            
            # Save
            example_id = f"example_{count+1:03d}"
            example_dir = self.output_dir / example_id
            example_dir.mkdir(parents=True, exist_ok=True)
            
            with open(example_dir / "chosen.cpp", "w") as f:
                f.write(full_chosen)
            
            with open(example_dir / "rejected.cpp", "w") as f:
                f.write(full_rejected)
            
            metadata = {
                "example_id": example_id,
                "prompt": prompt,
                "tests": tests,
                "chosen_completion": chosen_comp,
                "rejected_completion": rejected_comp
            }
            
            with open(example_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
                
            print(f"Saved {example_id}")
            count += 1

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic C++ competitive programming problems.")
    parser.add_argument("--num-examples", type=int, default=2, help="Number of examples to generate")
    args = parser.parse_args()
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable.")
        return

    pipeline = CPPSyntheticDataPipeline(api_key=api_key)
    pipeline.run(num_examples=args.num_examples)

if __name__ == "__main__":
    main()
